"""Adaptive target selection for the visual agent (Phase 5).

``AdaptivePlanner`` plugs into ``VisualAgent3D`` (``planner.attach(agent)``) and replaces
two decisions of the Phase 4 controller:

* which unresolved spawn point to inspect next (``inspection_score``), using Bayesian spawn
  estimates from earlier episodes, learned travel times and reliabilities, information
  gain and cluster value (``frontier_scoring``);
* which visible chest to go to first (``choose_track``), preferring reliable targets and
  dense clusters instead of simply the nearest one.

It also records every navigation attempt in route memory. When a statistic is missing it
falls back to the Phase 4 behaviour (configured probability, straight travel time), and
the Phase 4 rule itself remains available as :func:`frontier_scoring.baseline_score`.
Known-empty, opened and unreachable spawn points are never candidates: the agent's world
model removes them for the rest of the episode.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from chest_hunter.database import DatabaseManager
from chest_hunter.exploration.frontier_scoring import Candidate, ScoringWeights, score
from chest_hunter.exploration.route_memory import RouteAttempt, RouteMemory, region_of
from chest_hunter.exploration.spawn_estimator import (
    SpawnEstimate,
    SpawnEstimator,
    information_gain,
)


class AdaptivePlanner:
    def __init__(self, estimates: dict[int, SpawnEstimate] | None = None,
                 route_memory: RouteMemory | None = None,
                 weights: ScoringWeights | None = None) -> None:
        self.estimates = dict(estimates or {})
        self.routes = route_memory
        self.weights = weights or ScoringWeights()
        self.agent: Any = None
        self._attempt: RouteAttempt | None = None
        self.events: Counter[str] = Counter()

    @classmethod
    def from_database(cls, db: DatabaseManager, map_id: str, layout_version: str | None, *,
                      prior_strength: float = 4.0, weights: ScoringWeights | None = None,
                      use_route_memory: bool = True) -> AdaptivePlanner:
        estimator = SpawnEstimator(db, prior_strength=prior_strength)
        estimates = estimator.estimate(map_id, layout_version)
        memory = RouteMemory(db, map_id, layout_version) if use_route_memory else None
        return cls(estimates, memory, weights)

    def attach(self, agent: Any, *, map_id: str = "") -> None:
        self.agent = agent
        agent.planner = self

    # -- estimates ------------------------------------------------------------------------

    def probability(self, prior: Any) -> float:
        e = self.estimates.get(prior.location_id)
        return e.mean if e is not None else float(prior.probability)

    def _candidate(self, prior: Any, travel_seconds: float) -> Candidate:
        e = self.estimates.get(prior.location_id)
        if e is not None:
            h, dv = information_gain(e.alpha, e.beta)
        else:
            p = float(prior.probability)
            h, dv = information_gain(max(1e-3, 4 * p), max(1e-3, 4 * (1 - p)))
        rel = 0.9
        if self.routes is not None and self.agent is not None:
            rel = self.routes.reliability(region_of(*prior.xyz), prior.location_id)
        return Candidate(prior.location_id, prior.xyz, self.probability(prior), rel, h, dv,
                         travel_seconds)

    def _travel(self, start: tuple[float, float, float], dest: tuple[float, float, float],
                length: float, speed: float) -> float:
        factor = 1.0
        if self.routes is not None:
            factor = self.routes.travel_factor(region_of(*start), region_of(*dest))
        return length / max(0.1, speed) * factor

    # -- decisions --------------------------------------------------------------------------

    def preselect_score(self, prior: Any, pos: tuple[float, float, float], straight: float,
                        speed: float, unresolved: list[Any]) -> float:
        cand = self._candidate(prior, self._travel(pos, prior.xyz, straight * 1.25, speed))
        others = [self._candidate(p, 0.0) for p in unresolved]
        return score(cand, others, self.weights)

    def inspection_score(self, prior: Any, pos: tuple[float, float, float], length: float,
                         speed: float, unresolved: list[Any]) -> float:
        cand = self._candidate(prior, self._travel(pos, prior.xyz, length, speed))
        others = [self._candidate(p, 0.0) for p in unresolved]
        self.events["inspections_scored"] += 1
        return score(cand, others, self.weights)

    def choose_track(self, targets: list[Any], pos: tuple[float, float, float],
                     speed: float) -> Any:
        """A seen chest has p = 1; rank by reliability, time and the value of what is near."""
        unresolved = [p for p in self.agent.world.unresolved()] if self.agent else []
        others = [self._candidate(p, 0.0) for p in unresolved]

        def value(track: Any) -> tuple[float, int]:
            dist = math.dist(pos, track.xyz)
            rel = 0.9
            if self.routes is not None:
                rel = self.routes.reliability(region_of(*track.xyz), track.location_id)
            c = Candidate(-track.track_id, track.xyz, 1.0, rel, 0.0, 0.0,
                          self._travel(pos, track.xyz, dist * 1.15, speed))
            return (score(c, others, self.weights), -track.track_id)

        self.events["tracks_chosen"] += 1
        return max(targets, key=value)

    # -- route memory hooks ---------------------------------------------------------------------

    def route_started(self, goal: Any, pos: tuple[float, float, float], length: float,
                      speed: float, now: float) -> None:
        self.route_finished(now, success=False, failure_kind="replaced")
        if self.routes is not None:
            self._attempt = self.routes.begin(pos, goal.position, goal.location_id, length,
                                              speed, now)

    def route_finished(self, now: float, *, success: bool, failure_kind: str | None = None
                       ) -> None:
        if self._attempt is not None and self.routes is not None:
            if failure_kind == "replaced":
                self._attempt.finished = now          # superseded by a replan: not counted
                self._attempt.success = None
            else:
                self.routes.finish(self._attempt, now, success=success, failure_kind=failure_kind)
                self.events["routes_ok" if success else f"routes_{failure_kind}"] += 1
        self._attempt = None

    def route_failed_immediately(self, goal: Any, pos: tuple[float, float, float],
                                 now: float, kind: str) -> None:
        if self.routes is not None:
            attempt = self.routes.begin(pos, goal.position, goal.location_id, 0.0, 1.0, now)
            self.routes.finish(attempt, now, success=False, failure_kind=kind)
            self.events[f"routes_{kind}"] += 1

    def recovery(self) -> None:
        if self._attempt is not None:
            self._attempt.recoveries += 1

    def opened(self, location_id: int) -> None:
        if self.routes is not None:
            for a in reversed(self.routes.pending):
                if a.location_id == location_id:
                    a.chest_opened = True
                    break
        self.events["opened"] += 1

    def flush(self, session_id: int | None) -> int:
        if self.routes is None:
            return 0
        self.routes.pending = [a for a in self.routes.pending if a.success is not None]
        return self.routes.flush(session_id)

    def summary(self) -> dict[str, Any]:
        observed = sum(1 for e in self.estimates.values() if e.observed)
        return {"estimates": len(self.estimates), "estimates_observed": observed,
                "events": dict(self.events),
                "route_memory": self.routes.summary() if self.routes else None}
