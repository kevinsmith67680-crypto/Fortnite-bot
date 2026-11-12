"""Chest selection: which spawn location should the agent visit next?

Every strategy scores the agent's *pending* candidates (enabled, on the
session's map, not yet visited/unreachable/abandoned in this session) using
travel times derived from shortest *path* length (the same cost A* returns),
never straight-line distance:

    T_i = path_length_i / player_speed + leg_overhead + P_i * estimated_open_seconds_i

``leg_overhead = speed / acceleration`` approximates the time lost to
accelerating away and braking to a stop on each leg. The interaction term is
the *expected* interaction time: if no chest spawned there is nothing to
open, so an empty spawn point costs travel time only.

Strategies
----------
``nearest``   Minimum travel time (ties: lower chest ID).
``weighted``  Maximum expected openings per second, ``S_i = P_i * Q_i / T_i``,
              where ``P_i`` is the spawn probability and ``Q_i`` is 1 for
              reachable and 0 for unreachable locations (unreachable ones are
              removed before scoring).
``lookahead`` Short-horizon cluster planning: for each first chest, the best
              rate ``sum(P) / sum(T)`` over sequences of up to ``depth``
              chests (beam-limited), then pick the best first chest. This
              avoids grabbing an isolated chest when a dense cluster is near.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from chest_hunter.models import STRATEGY_NAMES
from chest_hunter.navigation.astar import DistanceOracle
from chest_hunter.simulation.collision import Cell, GridMap, Point

_MIN_TIME: Final = 1e-6
LOOKAHEAD_BEAM: Final = 8


@dataclass(frozen=True, slots=True)
class Candidate:
    chest_id: int
    cell: Cell
    position: Point
    spawn_probability: float
    estimated_open_seconds: float


@dataclass(frozen=True, slots=True)
class CandidateScore:
    chest_id: int
    path_length: float
    travel_time: float
    total_time: float
    probability: float
    reachability: float
    score: float


@dataclass(frozen=True)
class TargetSelection:
    target: Candidate
    score: CandidateScore
    strategy: str
    unreachable: tuple[int, ...]
    evaluated: int


class RoutePlanner:
    """Scores candidates and selects the next target for a strategy."""

    def __init__(self, oracle: DistanceOracle, *, strategy: str, speed: float,
                 acceleration: float, lookahead_depth: int = 2) -> None:
        if strategy not in STRATEGY_NAMES:
            raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGY_NAMES}.")
        if speed <= 0 or acceleration <= 0:
            raise ValueError("speed and acceleration must be positive")
        self.oracle = oracle
        self.strategy = strategy
        self.speed = speed
        self.leg_overhead = speed / acceleration
        self.lookahead_depth = max(1, lookahead_depth)
        self.selections = 0

    # -- scoring ------------------------------------------------------------

    def score(self, grid: GridMap, from_cell: Cell, candidate: Candidate) -> CandidateScore:
        length = self.oracle.distance(grid, from_cell, candidate.cell)
        if math.isinf(length):
            return CandidateScore(candidate.chest_id, length, math.inf, math.inf,
                                  candidate.spawn_probability, 0.0, 0.0)
        travel = length / self.speed + (self.leg_overhead if length > 0 else 0.0)
        expected_interaction = candidate.spawn_probability * candidate.estimated_open_seconds
        total = max(travel + expected_interaction, _MIN_TIME)
        return CandidateScore(
            chest_id=candidate.chest_id,
            path_length=length,
            travel_time=travel,
            total_time=total,
            probability=candidate.spawn_probability,
            reachability=1.0,
            score=candidate.spawn_probability * 1.0 / total,
        )

    def evaluate(
        self, grid: GridMap, from_cell: Cell, candidates: Sequence[Candidate]
    ) -> tuple[list[tuple[Candidate, CandidateScore]], list[int]]:
        """Return ``(reachable (candidate, score) pairs, unreachable chest IDs)``."""
        reachable: list[tuple[Candidate, CandidateScore]] = []
        unreachable: list[int] = []
        for candidate in candidates:
            scored = self.score(grid, from_cell, candidate)
            if scored.reachability == 0.0:
                unreachable.append(candidate.chest_id)
            else:
                reachable.append((candidate, scored))
        return reachable, unreachable

    # -- selection ----------------------------------------------------------

    def select(self, grid: GridMap, from_cell: Cell,
               candidates: Sequence[Candidate]) -> TargetSelection | None:
        """Pick the next target, or ``None`` if no candidate is reachable."""
        reachable, unreachable = self.evaluate(grid, from_cell, candidates)
        self.selections += 1
        if not reachable:
            return None
        if self.strategy == "nearest":
            best = min(reachable, key=lambda pair: (pair[1].travel_time, pair[0].chest_id))
        elif self.strategy == "weighted":
            best = min(reachable, key=lambda pair: (-pair[1].score, pair[1].total_time,
                                                    pair[0].chest_id))
        else:
            best = self._lookahead(grid, reachable)
        return TargetSelection(best[0], best[1], self.strategy, tuple(unreachable),
                               len(reachable))

    def _leg_time(self, grid: GridMap, a: Candidate, b: Candidate) -> float:
        length = self.oracle.distance(grid, a.cell, b.cell)
        if math.isinf(length):
            return math.inf
        travel = length / self.speed + (self.leg_overhead if length > 0 else 0.0)
        return travel + b.spawn_probability * b.estimated_open_seconds

    def _lookahead(
        self, grid: GridMap, reachable: list[tuple[Candidate, CandidateScore]]
    ) -> tuple[Candidate, CandidateScore]:
        """Best first step of the highest-rate sequence of up to ``depth`` chests."""
        by_score = sorted(reachable, key=lambda p: (-p[1].score, p[0].chest_id))
        firsts = by_score[:LOOKAHEAD_BEAM * 2]
        pool = [candidate for candidate, _ in by_score]

        def best_rate(current: Candidate, gained: float, elapsed: float,
                      used: frozenset[int], depth_left: int) -> float:
            rate = gained / max(elapsed, _MIN_TIME)
            if depth_left == 0:
                return rate
            options: list[tuple[float, Candidate, float]] = []
            for nxt in pool:
                if nxt.chest_id in used:
                    continue
                leg = self._leg_time(grid, current, nxt)
                if math.isinf(leg):
                    continue
                options.append((nxt.spawn_probability / max(leg, _MIN_TIME), nxt, leg))
            options.sort(key=lambda o: (-o[0], o[1].chest_id))
            for _, nxt, leg in options[:LOOKAHEAD_BEAM]:
                rate = max(rate, best_rate(nxt, gained + nxt.spawn_probability, elapsed + leg,
                                           used | {nxt.chest_id}, depth_left - 1))
            return rate

        best_pair = by_score[0]
        best_value = -1.0
        for candidate, scored in firsts:
            value = best_rate(candidate, candidate.spawn_probability, scored.total_time,
                              frozenset({candidate.chest_id}), self.lookahead_depth - 1)
            if value > best_value + 1e-12:
                best_value = value
                best_pair = (candidate, scored)
        return best_pair
