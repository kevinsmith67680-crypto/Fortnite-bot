"""Adaptive exploration: scoring, route memory, recovery classes and the planner in an
episode."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from chest_hunter.agent.controller_3d import Agent3DConfig
from chest_hunter.agent.recovery_3d import FAILURE_CLASSES, RecoveryLog, classify
from chest_hunter.agent.runner_3d import Runner3D, _layout_hash
from chest_hunter.database import DatabaseManager
from chest_hunter.exploration.adaptive_planner import AdaptivePlanner
from chest_hunter.exploration.frontier_scoring import (
    Candidate,
    ScoringWeights,
    baseline_score,
    score,
)
from chest_hunter.exploration.route_memory import RouteMemory, region_of
from chest_hunter.exploration.spawn_estimator import SpawnEstimate
from chest_hunter.scenarios import seed_arena_3d
from tests.phase5_helpers import settings


def cand(key: int, p: float, t: float, xyz: tuple[float, float, float] = (0, 0, 0),
         q: float = 0.9, h: float = 0.5, dv: float = 0.0) -> Candidate:
    return Candidate(key, xyz, p, q, h, dv, t)


class Scoring(unittest.TestCase):
    W = ScoringWeights(cluster=0.0)

    def test_probability_time_and_reliability(self) -> None:
        self.assertGreater(score(cand(1, 0.9, 10), [], self.W), score(cand(2, 0.3, 10), [], self.W))
        self.assertGreater(score(cand(1, 0.6, 5), [], self.W), score(cand(2, 0.6, 30), [], self.W))
        self.assertGreater(score(cand(1, 0.6, 10, q=0.95), [], self.W),
                           score(cand(2, 0.6, 10, q=0.3), [], self.W))

    def test_information_gain_rewards_uncertainty(self) -> None:
        w = ScoringWeights(cluster=0.0, exploration=1.0)
        self.assertGreater(score(cand(1, 0.5, 10, h=1.0, dv=0.02), [], w),
                           score(cand(2, 0.5, 10, h=0.1, dv=0.0), [], w))

    def test_clusters_are_preferred(self) -> None:
        w = ScoringWeights(cluster=0.5)
        lonely = cand(1, 0.6, 10, xyz=(50, 50, 0))
        clustered = cand(2, 0.6, 10, xyz=(0, 0, 0))
        others = [cand(10 + k, 0.8, 0, xyz=(3 * k, 1, 0)) for k in range(1, 4)]
        self.assertGreater(score(clustered, others, w), score(lonely, others, w))

    def test_baseline_rule(self) -> None:
        self.assertAlmostEqual(baseline_score(0.5, 9.0), 0.05)


class RouteMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = DatabaseManager(":memory:")
        self.db.initialize()
        self.addCleanup(self.db.close)
        sessions = __import__("chest_hunter.session_repository",
                              fromlist=["SessionRepository"]).SessionRepository(self.db)
        self.session_ids = [sessions.start_session("m").id for _ in range(4)]

    def _record(self, memory: RouteMemory, session: int, *, success: bool, est: float = 10.0,
                actual: float = 15.0, dest: tuple[float, float, float] = (40.0, 40.0, 0.0)) -> None:
        a = memory.begin((1.0, 1.0, 0.0), dest, None, est * 5.0, 5.0, 0.0)
        memory.finish(a, actual, success=success, failure_kind=None if success else "path_blocked")
        memory.flush(session)

    def test_travel_factor_and_reliability_are_learned(self) -> None:
        memory = RouteMemory(self.db, "m", "v1")
        self.assertEqual(memory.travel_factor("r0_0", "r2_2"), 1.0)
        self.assertAlmostEqual(memory.reliability("r2_2"), 0.9)
        for s in self.session_ids[:3]:
            self._record(memory, s, success=True)
        fresh = RouteMemory(self.db, "m", "v1")
        self.assertAlmostEqual(fresh.travel_factor(region_of(1, 1), region_of(40, 40)), 1.5)
        self.assertGreater(fresh.reliability(region_of(40, 40)), 0.9)

    def test_temporary_vs_persistent_failures(self) -> None:
        memory = RouteMemory(self.db, "m", "v1")
        self._record(memory, self.session_ids[0], success=False)
        for s in self.session_ids[1:]:
            self._record(memory, s, success=True)
        recovered = RouteMemory(self.db, "m", "v1")
        self.assertFalse(recovered.persistent_failure(region_of(40, 40)))
        blocked = RouteMemory(self.db, "m2", "v1")
        for s in self.session_ids:
            self._record(blocked, s, success=False)
        again = RouteMemory(self.db, "m2", "v1")
        self.assertTrue(again.persistent_failure(region_of(40, 40)))
        # Four decayed failures against the Beta(4.5, 0.5) prior: well below the 0.9 default.
        self.assertLess(again.reliability(region_of(40, 40)), 0.65)

    def test_layout_versions_and_invalidation(self) -> None:
        memory = RouteMemory(self.db, "m", "v1")
        self._record(memory, self.session_ids[0], success=False)
        self._record(memory, self.session_ids[1], success=False)
        other_version = RouteMemory(self.db, "m", "v2")
        self.assertAlmostEqual(other_version.reliability(region_of(40, 40)), 0.9)
        self.assertEqual(memory.invalidate(layout_version="v1"), 2)
        self.assertAlmostEqual(RouteMemory(self.db, "m", "v1").reliability(region_of(40, 40)),
                               0.9)

    def test_regions(self) -> None:
        self.assertEqual(region_of(1, 1), "r0_0")
        self.assertEqual(region_of(31, 46, 3.5), "r2_3_L1")


class RecoveryClasses(unittest.TestCase):
    def test_every_reason_maps_to_a_class(self) -> None:
        for reason in ("stuck", "approach_blocked", "approach_timeout", "alignment_failed",
                       "interaction_failed", "interaction_interrupted",
                       "verification_inconclusive", "target_lost", "localisation_uncertainty",
                       "unreachable", "model_failure", None, "something-new"):
            self.assertIn(classify(reason), FAILURE_CLASSES)

    def test_log_and_budgets(self) -> None:
        log = RecoveryLog()
        log.record("t1", "path_blocked")
        log.record("t1", "interaction_failure")
        log.resolve("t1")
        self.assertEqual(log.to_dict()["path_blocked"], {"attempts": 1, "resolved": 1,
                                                         "smart_strategy_used": 0})
        for _ in range(20):
            log.use("interaction_failure")
        self.assertFalse(log.can_use("interaction_failure"))


@dataclass
class FakePrior:
    location_id: int
    xyz: tuple[float, float, float]
    probability: float


class PlannerUnit(unittest.TestCase):
    def test_estimates_override_configured_probability(self) -> None:
        est = {1: SpawnEstimate(1, "A", 0.5, 2.0, 2.0, 8, 0, 8)}
        planner = AdaptivePlanner(est)
        self.assertAlmostEqual(planner.probability(FakePrior(1, (0, 0, 0), 0.5)), 10 / 12)
        self.assertAlmostEqual(planner.probability(FakePrior(2, (0, 0, 0), 0.4)), 0.4)
        high = planner.inspection_score(FakePrior(1, (10, 0, 0), 0.5), (0, 0, 0), 10, 5, [])
        low = planner.inspection_score(FakePrior(2, (10, 0, 0), 0.4), (0, 0, 0), 10, 5, [])
        self.assertGreater(high, low)


class PlannerInEpisode(unittest.TestCase):
    def test_adaptive_episode_runs_records_routes_and_stays_correct(self) -> None:
        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        planner = AdaptivePlanner.from_database(db, "research_arena_3d",
                                                _layout_hash("research_arena_3d"))
        runner = Runner3D.from_settings(db, settings(), mode="hybrid", seed=7, backend="reference",
                                        max_ticks=1800, evaluate=True, planner=planner,
                                        config=Agent3DConfig(smart_recovery=True))
        result = runner.run()
        self.assertEqual(result.outcome, "tick_limit")
        self.assertLessEqual(result.chests_opened, result.truth["opened_truth"])
        written = planner.flush(result.session_id)
        self.assertGreater(written, 0)
        rows = db.query_all("SELECT COUNT(*) AS n FROM route_memory WHERE map_id = ?",
                            ("research_arena_3d",))
        self.assertEqual(rows[0]["n"], written)
        self.assertGreater(planner.events["inspections_scored"]
                           + planner.events["tracks_chosen"], 0)
        # Unreachable spawn points are never chosen again once marked.
        unreachable = [p for p in runner.controller.world.priors.values()
                       if p.status.value == "unreachable"]
        for p in unreachable:
            self.assertNotEqual(getattr(runner.controller.goal, "location_id", None),
                                p.location_id)


if __name__ == "__main__":
    unittest.main()
