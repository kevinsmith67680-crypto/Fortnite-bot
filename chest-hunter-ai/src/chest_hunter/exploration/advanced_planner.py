"""Phase 6 chest-search planning: greedy weighted and short-horizon (beam search).

Each candidate ``i`` (an unresolved spawn point, or a chest the agent sees) is scored::

    S_i = (p_i * q_i + lambda * I_i) / (T_i + eps) - mu * R_i

* ``p_i`` - estimated probability that a chest is there: the Bayesian estimate from earlier
  episodes (``spawn_estimator``), the configured prior without history, and 1 for a chest
  the agent currently sees;
* ``q_i`` - estimated probability of reaching and opening it: route-memory reliability for
  its region (Beta posterior; 0.9 without history);
* ``I_i`` - expected information gain (episode entropy plus posterior-variance reduction);
* ``T_i`` - estimated seconds: path length / speed x the learned travel-time factor, plus
  approach overhead and the interaction duration (weighted by ``p_i``);
* ``R_i`` - expected recovery cost: ``1 - q_i`` plus this episode's failures at that spawn
  point (failed plans, unsuccessful inspections, failed interactions), so repeatedly
  unreachable destinations sink;
* ``lambda``, ``mu``, ``eps`` - configuration (``planning`` section).

Strategies:

* ``greedy`` - the highest single-target score;
* ``short_horizon`` - sequences of up to ``planning_horizon`` targets by beam search (width
  ``beam_width``) over at most ``maximum_candidates`` candidates. A sequence is worth
  ``(sum p q + lambda sum I) / (sum T + eps) - mu sum R``: a target next to other likely
  chests (a productive cluster) is preferred over an isolated one at the same distance.
  The first leg uses the real path length (NavMesh query, as in Phase 4/5 for the four best
  candidates); later legs use straight distance x ``detour_factor`` - no extra path
  queries. Work is bounded by the candidate and beam limits; if a level of the search
  takes longer than ``planning_budget_ms`` (wall clock) the best sequence so far is used and
  the overrun is counted.

Ties are broken deterministically (lower spawn-point id first). Known-empty, opened,
unreachable and abandoned spawn points are never candidates (the agent's world model
resolves them for the rest of the episode). Planning runs only at decision points (no goal,
a new chest seen, a route finished or failed), never every frame.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chest_hunter.exploration.adaptive_planner import AdaptivePlanner
from chest_hunter.exploration.frontier_scoring import ScoringWeights
from chest_hunter.exploration.route_memory import RouteMemory, region_of
from chest_hunter.exploration.spawn_estimator import (
    SpawnEstimate,
    SpawnEstimator,
    information_gain,
)

STRATEGIES = ("greedy", "short_horizon")


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    strategy: str = "short_horizon"
    planning_horizon: int = 3
    beam_width: int = 5
    maximum_candidates: int = 20
    planning_budget_ms: float = 15.0
    exploration_weight: float = 0.15        # lambda
    recovery_weight: float = 0.005          # mu
    epsilon_s: float = 1.0
    interaction_seconds: float = 2.0
    approach_overhead_s: float = 4.0
    detour_factor: float = 1.25
    learning_weight: float = 4.0            # weight of variance reduction inside I

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"planning strategy must be one of {', '.join(STRATEGIES)}")
        if not 1 <= self.planning_horizon <= 5:
            raise ValueError("planning_horizon must be between 1 and 5")
        if not 1 <= self.beam_width <= 50:
            raise ValueError("beam_width must be between 1 and 50")
        if not 1 <= self.maximum_candidates <= 200:
            raise ValueError("maximum_candidates must be between 1 and 200")
        if self.planning_budget_ms <= 0 or self.epsilon_s <= 0 or self.detour_factor < 1.0:
            raise ValueError("planning_budget_ms and epsilon_s must be positive and "
                             "detour_factor at least 1")
        if self.exploration_weight < 0 or self.recovery_weight < 0:
            raise ValueError("exploration_weight and recovery_weight must be >= 0")


@dataclass(frozen=True, slots=True)
class Target:
    key: int                                  # spawn point id (negative: track without one)
    xyz: tuple[float, float, float]
    p: float
    q: float
    info: float
    risk: float

    @property
    def value(self) -> float:
        return self.p * self.q


@dataclass(slots=True)
class PlanningStats:
    decisions: int = 0
    sequences_evaluated: int = 0
    budget_overruns: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    chosen_by_sequence: int = 0               # first leg differs from the greedy choice

    def to_dict(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)
        return {"decisions": self.decisions, "sequences_evaluated": self.sequences_evaluated,
                "budget_overruns": self.budget_overruns,
                "chosen_differently_from_greedy": self.chosen_by_sequence,
                "latency_mean_ms": round(sum(lat) / len(lat), 4) if lat else None,
                "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 4) if lat else None,
                "latency_max_ms": round(lat[-1], 4) if lat else None}


class AdvancedPlanner(AdaptivePlanner):
    """Drop-in planner for ``VisualAgent3D`` (``planner.attach(agent)``); keeps the Phase 5
    route-memory bookkeeping and replaces the three scoring decisions."""

    def __init__(self, estimates: dict[int, SpawnEstimate] | None = None,
                 route_memory: RouteMemory | None = None,
                 config: PlanningConfig | None = None) -> None:
        self.config = config or PlanningConfig()
        weights = ScoringWeights(exploration=self.config.exploration_weight,
                                 learning=self.config.learning_weight,
                                 interaction_seconds=self.config.interaction_seconds,
                                 approach_overhead_s=self.config.approach_overhead_s,
                                 epsilon_s=self.config.epsilon_s)
        super().__init__(estimates, route_memory, weights)
        self.stats = PlanningStats()
        self._clock = time.perf_counter

    @classmethod
    def from_database(cls, db: Any, map_id: str, layout_version: str | None, *,
                      prior_strength: float = 4.0, weights: ScoringWeights | None = None,
                      use_route_memory: bool = True,
                      config: PlanningConfig | None = None) -> AdvancedPlanner:
        estimates = SpawnEstimator(db, prior_strength=prior_strength).estimate(
            map_id, layout_version)
        memory = RouteMemory(db, map_id, layout_version) if use_route_memory else None
        return cls(estimates, memory, config)

    # -- target model -------------------------------------------------------------------------

    def _reliability(self, xyz: tuple[float, float, float], location_id: int | None) -> float:
        if self.routes is None or self.agent is None:
            return 0.9
        return float(self.routes.reliability(region_of(*xyz), location_id))

    def _failures(self, location_id: int | None) -> int:
        agent = self.agent
        if agent is None or location_id is None:
            return 0
        prior = agent.world.priors.get(location_id)
        inspections = prior.inspections if prior is not None else 0
        plans = agent._plan_failures.get(location_id, 0)
        attempts = sum(n for k, n in agent._attempts.items()
                       if isinstance(k, tuple) and len(k) > 1 and k[1] == location_id)
        return int(inspections + plans + attempts)

    def prior_target(self, prior: Any) -> Target:
        e = self.estimates.get(prior.location_id)
        if e is not None:
            h, dv = information_gain(e.alpha, e.beta)
            p = e.mean
        else:
            p = float(prior.probability)
            h, dv = information_gain(max(1e-3, 4 * p), max(1e-3, 4 * (1 - p)))
        q = self._reliability(prior.xyz, prior.location_id)
        info = h + self.config.learning_weight * dv * 12.0
        risk = (1.0 - q) + self._failures(prior.location_id)
        return Target(prior.location_id, prior.xyz, p, q, info, risk)

    def track_target(self, track: Any) -> Target:
        q = self._reliability(track.xyz, track.location_id)
        loc = track.location_id
        risk = (1.0 - q) + self._failures(loc)
        return Target(loc if loc is not None else -track.track_id, track.xyz, 1.0, q, 0.0, risk)

    def _seconds(self, target: Target, travel_m: float, speed: float,
                 start: tuple[float, float, float]) -> float:
        factor = 1.0
        if self.routes is not None:
            factor = self.routes.travel_factor(region_of(*start), region_of(*target.xyz))
        cfg = self.config
        return travel_m / max(0.1, speed) * factor + cfg.approach_overhead_s \
            + cfg.interaction_seconds * target.p

    def single_score(self, target: Target, seconds: float) -> float:
        cfg = self.config
        return (target.value + cfg.exploration_weight * target.info) / (seconds + cfg.epsilon_s) \
            - cfg.recovery_weight * target.risk

    # -- sequences --------------------------------------------------------------------------------

    def sequence_value(self, legs: Sequence[tuple[Target, float]]) -> float:
        cfg = self.config
        gain = sum(t.value + cfg.exploration_weight * t.info for t, _ in legs)
        seconds = sum(s for _, s in legs)
        risk = sum(t.risk for t, _ in legs)
        return gain / (seconds + cfg.epsilon_s) - cfg.recovery_weight * risk

    def best_sequence(self, first: Target, first_seconds: float, pool: Sequence[Target],
                      speed: float) -> tuple[float, tuple[int, ...]]:
        """Best value of a sequence that starts with ``first`` (beam search, bounded)."""
        cfg = self.config
        horizon = cfg.planning_horizon if cfg.strategy == "short_horizon" else 1
        start_legs = ((first, first_seconds),)
        best: tuple[float, tuple[int, ...]] = (self.sequence_value(start_legs), (first.key,))
        if horizon <= 1:
            return best
        others = [t for t in pool if t.key != first.key][:cfg.maximum_candidates]
        beam: list[tuple[float, tuple[tuple[Target, float], ...]]] = [(best[0], start_legs)]
        started = self._clock()
        for _depth in range(1, horizon):
            expanded = []
            for _, legs in beam:
                used = {t.key for t, _ in legs}
                last = legs[-1][0]
                for t in others:
                    if t.key in used:
                        continue
                    straight = math.dist(last.xyz, t.xyz) * cfg.detour_factor
                    seq = (*legs, (t, self._seconds(t, straight, speed, last.xyz)))
                    expanded.append((self.sequence_value(seq), seq))
            self.stats.sequences_evaluated += len(expanded)
            if not expanded:
                break
            expanded.sort(key=lambda e: (-e[0], tuple(t.key for t, _ in e[1])))
            beam = expanded[:cfg.beam_width]
            if beam[0][0] > best[0]:
                best = (beam[0][0], tuple(t.key for t, _ in beam[0][1]))
            if 1000.0 * (self._clock() - started) > cfg.planning_budget_ms:
                self.stats.budget_overruns += 1
                self.events["planning_budget_overrun"] += 1
                break
        return best

    def _pool(self, unresolved: Sequence[Any], pos: tuple[float, float, float],
              speed: float) -> list[Target]:
        targets = [self.prior_target(p) for p in unresolved]
        scored = [(self.single_score(t, self._seconds(t, math.dist(pos, t.xyz)
                                                       * self.config.detour_factor, speed, pos)),
                   t) for t in targets]
        scored.sort(key=lambda e: (-e[0], e[1].key))
        return [t for _, t in scored[:self.config.maximum_candidates]]

    # -- decisions used by the agent ---------------------------------------------------------------

    def preselect_score(self, prior: Any, pos: tuple[float, float, float], straight: float,
                        speed: float, unresolved: list[Any]) -> float:
        """Ranking before path queries (straight distance x detour factor)."""
        t = self.prior_target(prior)
        return self.single_score(t, self._seconds(t, straight * self.config.detour_factor,
                                                  speed, pos))

    def inspection_score(self, prior: Any, pos: tuple[float, float, float], length: float,
                         speed: float, unresolved: list[Any]) -> float:
        """Score of inspecting ``prior`` next, given the real path length to it."""
        started = self._clock()
        t = self.prior_target(prior)
        seconds = self._seconds(t, length, speed, pos)
        if self.config.strategy == "greedy":
            value = self.single_score(t, seconds)
        else:
            pool = self._pool(unresolved, pos, speed)
            value, _seq = self.best_sequence(t, seconds, pool, speed)
        self._decided(started)
        self.events["inspections_scored"] += 1
        return value

    def choose_track(self, targets: list[Any], pos: tuple[float, float, float],
                     speed: float) -> Any:
        """Which visible chest first: p = 1, then the same sequence value over the
        unresolved spawn points that would follow it."""
        started = self._clock()
        unresolved = [p for p in self.agent.world.unresolved()
                      if p.status.value == "unknown"] if self.agent is not None else []
        pool = self._pool(unresolved, pos, speed) if self.config.strategy == "short_horizon" \
            else []
        best: tuple[float, int, Any] | None = None
        greedy: tuple[float, int, Any] | None = None
        for track in targets:
            t = self.track_target(track)
            seconds = self._seconds(t, math.dist(pos, track.xyz) * 1.15, speed, pos)
            single = self.single_score(t, seconds)
            value = single if not pool else self.best_sequence(t, seconds, pool, speed)[0]
            cand = (value, -int(track.track_id), track)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
            g = (single, -int(track.track_id), track)
            if greedy is None or (g[0], g[1]) > (greedy[0], greedy[1]):
                greedy = g
        assert best is not None and greedy is not None
        if best[2] is not greedy[2]:
            self.stats.chosen_by_sequence += 1
        self._decided(started)
        self.events["tracks_chosen"] += 1
        return best[2]

    def _decided(self, started: float) -> None:
        self.stats.decisions += 1
        if len(self.stats.latencies_ms) < 100_000:
            self.stats.latencies_ms.append(1000.0 * (self._clock() - started))

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out.update({"strategy": self.config.strategy, "planning": self.stats.to_dict()})
        return out
