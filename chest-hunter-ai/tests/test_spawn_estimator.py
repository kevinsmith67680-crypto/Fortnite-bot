"""Bayesian spawn estimation from the observation history."""

from __future__ import annotations

import math
import unittest

from chest_hunter.analytics.stats import beta_cdf, beta_interval, beta_ppf, summarise, t_ppf
from chest_hunter.chest_repository import ChestRepository
from chest_hunter.database import DatabaseManager
from chest_hunter.exploration.spawn_estimator import (
    SpawnEstimator,
    information_gain,
)
from chest_hunter.run_repository import RunRepository
from chest_hunter.session_repository import SessionRepository

MAP = "estimator_map"


class StatisticsHelpers(unittest.TestCase):
    def test_beta_distribution(self) -> None:
        self.assertAlmostEqual(beta_cdf(0.5, 1, 1), 0.5, places=10)
        self.assertAlmostEqual(beta_cdf(0.3, 2, 5), 0.579825, places=5)
        self.assertAlmostEqual(beta_ppf(0.5, 3, 3), 0.5, places=8)
        lo, hi = beta_interval(1, 1)
        self.assertAlmostEqual(lo, 0.025, places=6)
        self.assertAlmostEqual(hi, 0.975, places=6)

    def test_student_t_and_summary(self) -> None:
        self.assertAlmostEqual(t_ppf(0.975, 99), 1.984217, places=4)
        self.assertAlmostEqual(t_ppf(0.975, 9), 2.262157, places=4)
        s = summarise([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((s["n"], s["mean"], s["median"]), (4, 2.5, 2.5))
        self.assertLess(s["ci_low"], 2.5)
        self.assertGreater(s["ci_high"], 2.5)
        self.assertIsNone(summarise([])["mean"])

    def test_information_gain(self) -> None:
        h_half, dv_half = information_gain(1, 1)
        h_sure, dv_sure = information_gain(99, 1)
        self.assertAlmostEqual(h_half, 1.0)
        self.assertLess(h_sure, 0.1)
        self.assertGreater(dv_half, dv_sure)
        _, dv_many = information_gain(50, 50)
        self.assertLess(dv_many, dv_half)                  # well-known spots teach less


class Estimates(unittest.TestCase):
    def setUp(self) -> None:
        self.db = DatabaseManager(":memory:")
        self.db.initialize()
        self.addCleanup(self.db.close)
        chests = ChestRepository(self.db)
        self.a = chests.add_chest(MAP, "Spot A", 1, 1, spawn_probability=0.8).id
        self.b = chests.add_chest(MAP, "Spot B", 5, 5, spawn_probability=0.5).id
        self.c = chests.add_chest(MAP, "Spot C", 9, 9, spawn_probability=0.3).id
        self.sessions = SessionRepository(self.db)
        self.runs = RunRepository(self.db)

    def _session(self, observations: list[tuple[int, bool]], *, status: str = "completed",
                 layout: str | None = "v1") -> int:
        s = self.sessions.start_session(MAP)
        assert s.id is not None
        params = {"layout_hash": layout} if layout is not None else {}
        self.runs.create_run(s.id, strategy="test", seed=1, parameters=params)
        for loc, present in observations:
            self.sessions.record_observation(s.id, loc, chest_present=present)
        if status == "completed":
            self.sessions.complete_session(s.id)
        elif status == "aborted":
            self.sessions.abort_session(s.id)
        return s.id

    def test_unobserved_locations_keep_their_prior(self) -> None:
        est = SpawnEstimator(self.db, prior_strength=4.0).estimate(MAP, "v1")
        self.assertAlmostEqual(est[self.a].mean, 0.8)
        self.assertFalse(est[self.a].observed)
        self.assertAlmostEqual(est[self.c].mean, 0.3)

    def test_presence_and_absence_update_the_posterior(self) -> None:
        for _ in range(3):
            self._session([(self.a, False), (self.b, True)])
        est = SpawnEstimator(self.db, prior_strength=4.0).estimate(MAP, "v1")
        # A: Beta(3.2 + 0, 0.8 + 3) -> 3.2 / 7
        self.assertAlmostEqual(est[self.a].mean, 3.2 / 7.0)
        self.assertEqual((est[self.a].present, est[self.a].absent), (0, 3))
        self.assertAlmostEqual(est[self.b].mean, (2 + 3) / 7.0)
        lo, hi = est[self.b].credible_interval()
        self.assertLess(lo, est[self.b].mean)
        self.assertGreater(hi, est[self.b].mean)

    def test_duplicates_in_one_session_count_once(self) -> None:
        self._session([(self.a, False), (self.a, False), (self.a, True)])
        est = SpawnEstimator(self.db).estimate(MAP, "v1")
        self.assertEqual((est[self.a].present, est[self.a].absent, est[self.a].sessions),
                         (1, 0, 1))

    def test_conflicting_and_sparse_observations(self) -> None:
        self._session([(self.b, True)])
        self._session([(self.b, False)])
        est = SpawnEstimator(self.db, prior="uniform").estimate(MAP, "v1")
        self.assertAlmostEqual(est[self.b].mean, 0.5)          # Beta(2, 2)
        self.assertEqual(est[self.b].sessions, 2)
        sparse = est[self.b].credible_interval()
        for _ in range(20):
            self._session([(self.b, True)])
            self._session([(self.b, False)])
        dense = SpawnEstimator(self.db, prior="uniform").estimate(MAP, "v1")[self.b]
        self.assertLess(dense.credible_interval()[1] - dense.credible_interval()[0],
                        sparse[1] - sparse[0])

    def test_only_valid_sessions_and_matching_layouts(self) -> None:
        self._session([(self.a, False)], status="aborted")
        self._session([(self.a, False)], status="running")
        self._session([(self.a, False)], layout="other-version")
        est = SpawnEstimator(self.db).estimate(MAP, "v1")
        self.assertFalse(est[self.a].observed)
        self._session([(self.a, False)], layout=None)          # unversioned (manual) session
        self.assertTrue(SpawnEstimator(self.db).estimate(MAP, "v1")[self.a].observed)
        strict = SpawnEstimator(self.db, include_unversioned=False).estimate(MAP, "v1")
        self.assertFalse(strict[self.a].observed)

    def test_store_never_changes_configured_probabilities(self) -> None:
        self._session([(self.a, False)])
        estimator = SpawnEstimator(self.db)
        n = estimator.store(estimator.estimate(MAP, "v1"), "v1")
        self.assertEqual(n, 3)
        n = estimator.store(estimator.estimate(MAP, "v1"), "v1")    # upsert, no duplicates
        rows = self.db.query_all("SELECT COUNT(*) AS n FROM spawn_estimates")
        self.assertEqual(rows[0]["n"], 3)
        conf = ChestRepository(self.db).get_chest(self.a).spawn_probability
        self.assertAlmostEqual(conf, 0.8)
        stored = self.db.query_all("SELECT posterior_mean FROM spawn_estimates "
                                   "WHERE chest_location_id = ?", (self.a,))[0]["posterior_mean"]
        self.assertAlmostEqual(stored, 3.2 / 5.0)

    def test_rewards_are_not_spawn_evidence(self) -> None:
        s = self.sessions.start_session(MAP)
        assert s.id is not None
        self.runs.create_run(s.id, strategy="test", seed=1, parameters={"layout_hash": "v1"})
        self.sessions.record_observation(s.id, self.c, chest_present=True, chest_opened=True,
                                         reward_received=True)
        self.sessions.complete_session(s.id)
        est = SpawnEstimator(self.db).estimate(MAP, "v1")[self.c]
        self.assertEqual((est.present, est.absent), (1, 0))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            SpawnEstimator(self.db, prior="optimistic")
        with self.assertRaises(ValueError):
            SpawnEstimator(self.db, prior_strength=0)
        self.assertTrue(math.isclose(sum(1 for _ in SpawnEstimator(self.db).report(MAP)), 3))


if __name__ == "__main__":
    unittest.main()
