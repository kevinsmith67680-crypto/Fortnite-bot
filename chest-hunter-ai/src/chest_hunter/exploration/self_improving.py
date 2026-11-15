"""Self-improving exploration from completed research episodes (Phase 7).

Lightweight statistics, no new neural network. Learned per map *category* (the map family:
open, indoor, mixed, complex), from completed **training** episodes only:

* chest discoveries per newly seen cell, by region type (enclosed, open, elevated) - fed to
  the explorer's chest-discovery value ``C``;
* travel speed (metres per simulated second while navigating);
* interaction success (openings per interaction attempt);
* recovery outcomes per failure class (resolved / attempted);
* information gained per camera scan (newly seen cells per scan);
* the reward of each exploration strategy (chests opened per simulated minute).

:class:`StrategyBandit` is a contextual bandit (context = map category, arms = the five
predefined strategies of ``exploration.active``) using Thompson sampling with a Normal
posterior on each arm's mean reward. During training it samples an arm per episode; in
evaluation it only exploits (the arm with the best posterior mean) and is never updated.

:class:`ExplorationMemory` keeps all of it in one versioned JSON document (atomic writes).
Updates happen **between** episodes, never during one. Held-out maps are refused for
updates (``generator7.assert_training_map``), and nothing map-specific (positions of chests,
spawn outcomes) is stored - only category-level aggregates - so the memory cannot carry a
held-out layout's answers.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from chest_hunter.exploration.active import REGION_TYPES, STRATEGIES
from chest_hunter.runtime.checkpoint import atomic_write_text

MEMORY_FORMAT: Final = "chest-hunter-exploration-memory"
MEMORY_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """What one completed episode contributes (the agent's own measurements, plus the
    ground-truth reward used for scoring a strategy in training)."""

    category: str
    strategy: str
    throughput_per_min: float
    cells_seen_by_region: dict[str, float]
    discoveries_by_region: dict[str, float]
    travel_m: float
    travel_s: float
    interaction_attempts: int
    openings: int
    recovery: dict[str, dict[str, int]]
    scans: int
    cells_from_scans: float


class StrategyBandit:
    """Thompson sampling over strategies, one context per map category."""

    def __init__(self, arms: Sequence[str] = STRATEGIES, *, prior_mean: float = 1.0,
                 prior_count: float = 1.0, noise_sd: float = 0.8) -> None:
        self.arms = tuple(arms)
        self.prior_mean = prior_mean
        self.prior_count = prior_count
        self.noise_sd = noise_sd
        self.stats: dict[str, dict[str, list[float]]] = {}      # context -> arm -> [n, sum]

    def _arm(self, context: str, arm: str) -> list[float]:
        return self.stats.setdefault(context, {}).setdefault(arm, [0.0, 0.0])

    def posterior(self, context: str, arm: str) -> tuple[float, float]:
        n, total = self._arm(context, arm)
        count = self.prior_count + n
        mean = (self.prior_mean * self.prior_count + total) / count
        return mean, self.noise_sd / math.sqrt(count)

    def select(self, context: str, rng: random.Random) -> str:
        draws = [(rng.gauss(*self.posterior(context, a)), -i, a)
                 for i, a in enumerate(self.arms)]
        return max(draws)[2]

    def best(self, context: str) -> str:
        """Exploit only (evaluation): best posterior mean; ties to the first arm."""
        return max(((self.posterior(context, a)[0], -i, a) for i, a in enumerate(self.arms)))[2]

    def update(self, context: str, arm: str, reward: float) -> None:
        if arm not in self.arms:
            raise ValueError(f"unknown arm {arm!r}")
        s = self._arm(context, arm)
        s[0] += 1.0
        s[1] += float(reward)

    def to_dict(self) -> dict[str, Any]:
        return {"arms": list(self.arms), "prior_mean": self.prior_mean,
                "prior_count": self.prior_count, "noise_sd": self.noise_sd,
                "stats": self.stats}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyBandit:
        b = cls(data["arms"], prior_mean=data["prior_mean"], prior_count=data["prior_count"],
                noise_sd=data["noise_sd"])
        b.stats = {c: {a: [float(v[0]), float(v[1])] for a, v in arms.items()}
                   for c, arms in data.get("stats", {}).items()}
        return b


def _empty_category() -> dict[str, Any]:
    return {"episodes": 0,
            "cells_by_region": {r: 0.0 for r in REGION_TYPES},
            "discoveries_by_region": {r: 0.0 for r in REGION_TYPES},
            "travel_m": 0.0, "travel_s": 0.0, "interaction_attempts": 0, "openings": 0,
            "recovery": {}, "scans": 0, "cells_from_scans": 0.0}


class ExplorationMemory:
    """Persistent research memory for exploration (versioned, category-level only)."""

    def __init__(self, *, agent_version: str = "", generator_version: str = "") -> None:
        self.agent_version = agent_version
        self.generator_version = generator_version
        self.categories: dict[str, dict[str, Any]] = {}
        self.bandit = StrategyBandit()
        self.updates = 0

    # -- learned quantities --------------------------------------------------------------------

    def discovery_rates(self, category: str, *, alpha: float = 1.0, beta: float = 80.0
                        ) -> dict[str, float]:
        """Chests per newly seen cell, by region type (Beta-smoothed towards 1/80)."""
        c = self.categories.get(category) or _empty_category()
        return {r: (c["discoveries_by_region"].get(r, 0.0) + alpha)
                / (c["cells_by_region"].get(r, 0.0) + beta) for r in REGION_TYPES}

    def travel_speed(self, category: str, default: float = 3.5) -> float:
        c = self.categories.get(category)
        if not c or c["travel_s"] < 30.0:
            return default
        return float(c["travel_m"]) / float(c["travel_s"])

    def interaction_success(self, category: str) -> float | None:
        c = self.categories.get(category)
        if not c or not c["interaction_attempts"]:
            return None
        return float(c["openings"]) / float(c["interaction_attempts"])

    def recovery_success(self, category: str) -> dict[str, float]:
        c = self.categories.get(category) or {}
        return {k: v["resolved"] / v["attempts"] for k, v in c.get("recovery", {}).items()
                if v["attempts"]}

    def scan_gain(self, category: str) -> float | None:
        c = self.categories.get(category)
        if not c or not c["scans"]:
            return None
        return float(c["cells_from_scans"]) / float(c["scans"])

    # -- updates (between training episodes only) ------------------------------------------------

    def update(self, outcome: EpisodeOutcome, *, map_id: str, training: bool) -> None:
        from chest_hunter.simulation3d.generator7 import assert_training_map

        if not training:
            raise ValueError("exploration memory is frozen in evaluation mode")
        assert_training_map(map_id)
        c = self.categories.setdefault(outcome.category, _empty_category())
        c["episodes"] += 1
        for r in REGION_TYPES:
            c["cells_by_region"][r] += float(outcome.cells_seen_by_region.get(r, 0.0))
            c["discoveries_by_region"][r] += float(outcome.discoveries_by_region.get(r, 0.0))
        c["travel_m"] += outcome.travel_m
        c["travel_s"] += outcome.travel_s
        c["interaction_attempts"] += outcome.interaction_attempts
        c["openings"] += outcome.openings
        for cls, v in outcome.recovery.items():
            slot = c["recovery"].setdefault(cls, {"attempts": 0, "resolved": 0})
            slot["attempts"] += int(v.get("attempts", 0))
            slot["resolved"] += int(v.get("resolved", 0))
        c["scans"] += outcome.scans
        c["cells_from_scans"] += outcome.cells_from_scans
        self.bandit.update(outcome.category, outcome.strategy, outcome.throughput_per_min)
        self.updates += 1

    # -- persistence -------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"format": MEMORY_FORMAT, "version": MEMORY_VERSION,
                "agent_version": self.agent_version,
                "generator_version": self.generator_version, "updates": self.updates,
                "categories": self.categories, "bandit": self.bandit.to_dict()}

    def save(self, path: str | Path) -> None:
        atomic_write_text(Path(path), json.dumps(self.to_dict(), indent=1, sort_keys=True)
                          + "\n")

    @classmethod
    def load(cls, path: str | Path, *, agent_version: str = "",
             generator_version: str = "") -> ExplorationMemory:
        """Load a memory of the same versions; another version starts empty (records of a
        different agent or generator are never mixed in)."""
        p = Path(path)
        mem = cls(agent_version=agent_version, generator_version=generator_version)
        if not p.exists():
            return mem
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("format") != MEMORY_FORMAT or data.get("version") != MEMORY_VERSION:
            raise ValueError(f"{p} is not a version {MEMORY_VERSION} exploration memory")
        if (agent_version and data.get("agent_version") != agent_version) or (
                generator_version and data.get("generator_version") != generator_version):
            return mem
        mem.categories = data.get("categories", {})
        mem.bandit = StrategyBandit.from_dict(data["bandit"])
        mem.updates = int(data.get("updates", 0))
        return mem


def outcome_from_episode(category: str, strategy: str, runner: Any, result: Any
                         ) -> EpisodeOutcome:
    """Collect an episode's contribution (agent measurements + ground-truth reward)."""
    ex = runner.explorer
    minutes = result.sim_seconds / 60.0 if result.sim_seconds else 0.0
    opened = (result.truth or {}).get("opened_truth") or result.chests_opened
    m = result.metrics
    return EpisodeOutcome(
        category=category, strategy=strategy,
        throughput_per_min=opened / minutes if minutes else 0.0,
        cells_seen_by_region=dict(ex.cells_by_region) if ex else {},
        discoveries_by_region=dict(ex.discoveries_by_region) if ex else {},
        travel_m=m.travel_distance, travel_s=float(ex.travel_seconds) if ex else 0.0,
        interaction_attempts=m.interaction_attempts, openings=m.chests_opened,
        recovery={k: {"attempts": v["attempts"], "resolved": v["resolved"]}
                  for k, v in m.recovery_log.to_dict().items()},
        scans=ex.stats.scans_planned if ex else 0,
        cells_from_scans=float(sum(ex.stats.gains_realised)) if ex else 0.0)
