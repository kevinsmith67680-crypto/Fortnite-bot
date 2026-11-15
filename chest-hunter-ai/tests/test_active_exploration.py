"""Phase 7 active exploration: coverage map, frontier candidates and scoring, termination,
camera sector search, uncertainty thresholds, discovery-mode world model."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from chest_hunter.exploration.active import (
    STRATEGIES,
    ActiveExplorer,
    CameraSectors,
    Candidate,
    ConfidenceThresholds,
    CoverageMap,
    ExplorationConfig,
)


def _room_fan(explorer: ActiveExplorer, origin: tuple[float, float, float], half: float
              ) -> None:
    """Raycast distances from ``origin`` to the walls of a square room of side 2*half
    centred on the origin (what NAV_RAYCAST would return)."""
    dists = []
    for b in explorer.bearings():
        r = math.radians(b)
        dx, dy = abs(math.cos(r)), abs(math.sin(r))
        d = min(half / dx if dx > 1e-9 else math.inf, half / dy if dy > 1e-9 else math.inf)
        dists.append(min(d, explorer.cfg.ray_reach_m))
    explorer.add_fan(origin, dists)


class CoverageMapTest(unittest.TestCase):
    def test_fan_marks_walkable_cells_open_edges_and_walls(self) -> None:
        m = CoverageMap(2.0)
        new = m.integrate_fan((10.0, 10.0, 0.0), [0.0, 90.0, 180.0, 270.0],
                              [15.0, 3.0, 3.0, 3.0], reach=15.0)
        self.assertGreater(new, 0)
        self.assertTrue(m.cells[m.key(24.0, 10.0, 0.0)].open_edge)     # full-length ray
        self.assertTrue(m.cells[m.key(10.0, 14.5, 0.0)].blocked)       # beyond a short ray
        self.assertTrue(all(c.views == 0 for c in m.cells.values()))
        self.assertEqual(m.coverage(), 0.0)

    def test_view_wedge_respects_direction_range_and_walls(self) -> None:
        m = CoverageMap(2.0)
        m.integrate_fan((10.0, 10.0, 0.0), [k * 22.5 for k in range(16)], [12.0] * 16, 15.0)
        new = m.integrate_view((10.0, 10.0, 1.6), 0.0, 90.0, 11.0, now=1.0,
                               free_range=lambda b: 12.0, level_z=0.0)
        self.assertGreater(new, 0)
        self.assertGreater(m.cells[m.key(16.0, 10.0, 0.0)].views, 0)   # in front
        self.assertEqual(m.cells[m.key(4.0, 10.0, 0.0)].views, 0)      # behind
        # a wall 3 m ahead hides what lies beyond it
        m2 = CoverageMap(2.0)
        m2.integrate_fan((10.0, 10.0, 0.0), [0.0], [12.0], 15.0)
        m2.integrate_view((10.0, 10.0, 1.6), 0.0, 90.0, 11.0, now=1.0,
                          free_range=lambda b: 3.0, level_z=0.0)
        self.assertEqual(m2.cells[m2.key(18.0, 10.0, 0.0)].views, 0)
        # seen once is counted once
        again = m.integrate_view((10.0, 10.0, 1.6), 0.0, 90.0, 11.0, now=2.0,
                                 free_range=lambda b: 12.0, level_z=0.0)
        self.assertEqual(again, 0)

    def test_memory_is_bounded(self) -> None:
        m = CoverageMap(2.0, max_cells=100)
        for k in range(10):
            m.integrate_fan((10.0 * k, 0.0, 0.0), [k2 * 10.0 for k2 in range(36)],
                            [15.0] * 36, 15.0)
        self.assertLessEqual(len(m.cells), 100)
        self.assertGreater(m.dropped, 0)

    def test_levels_are_separate(self) -> None:
        m = CoverageMap(2.0)
        self.assertNotEqual(m.key(1.0, 1.0, 0.0), m.key(1.0, 1.0, 2.5))


class FrontierTest(unittest.TestCase):
    def _explorer(self, strategy: str = "max_information") -> ActiveExplorer:
        ex = ActiveExplorer(ExplorationConfig(strategy=strategy), clock=lambda: 0.0)
        _room_fan(ex, (20.0, 20.0, 0.0), 14.0)
        return ex

    def test_candidates_are_unseen_walkable_areas(self) -> None:
        ex = self._explorer()
        ex.observe((20.0, 20.0, 1.6), 0.0, 0.0, 90.0, now=0.0)       # looked east only
        target = ex.select((20.0, 20.0, 0.0), 3.5, lambda p: math.dist(p, (20.0, 20.0, 0.0)))
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(ex.map.cells[target.key].views, 0)
        self.assertTrue(ex.map.cells[target.key].walkable)
        self.assertLess(target.point[0], 20.0 + 2.0, "the unseen side was not preferred")

    def test_unreachable_candidates_are_blacklisted_not_retried(self) -> None:
        ex = self._explorer()
        calls: list[tuple[float, float, float]] = []

        def no_path(p: tuple[float, float, float]) -> float | None:
            calls.append(p)
            return None

        self.assertIsNone(ex.select((20.0, 20.0, 0.0), 3.5, no_path))
        self.assertLessEqual(len(calls), ex.cfg.path_queries)     # bounded per selection
        first = set(calls)
        calls.clear()
        ex.select((20.0, 20.0, 0.0), 3.5, no_path)
        self.assertFalse(first & set(calls), "a blacklisted candidate was queried again")

    def test_reached_areas_are_not_searched_again(self) -> None:
        ex = self._explorer()
        t = ex.select((20.0, 20.0, 0.0), 3.5, lambda p: 5.0)
        assert t is not None
        ex.reached(t)
        t2 = ex.select((20.0, 20.0, 0.0), 3.5, lambda p: 5.0)
        assert t2 is not None
        b = ex.cfg.block_cells
        self.assertNotEqual((t.key[0] // b, t.key[1] // b), (t2.key[0] // b, t2.key[1] // b))

    def test_exploration_terminates_when_everything_is_seen(self) -> None:
        ex = self._explorer()
        for yaw in range(0, 360, 30):
            ex.observe((20.0, 20.0, 1.6), 0.0, float(yaw), 90.0, now=0.0)
        for cell in ex.map.cells.values():                 # also the far corners
            if cell.walkable:
                cell.views = max(cell.views, 1)
                cell.open_edge = False
        ex._candidates = None
        self.assertIsNone(ex.select((20.0, 20.0, 0.0), 3.5, lambda p: 5.0))
        self.assertTrue(ex.done)

    def test_planning_respects_its_time_budget(self) -> None:
        ticks = iter(range(0, 10_000, 30))                 # every clock call: +30 ms
        ex = ActiveExplorer(ExplorationConfig(planning_budget_ms=20.0),
                            clock=lambda: next(ticks) / 1000.0)
        _room_fan(ex, (20.0, 20.0, 0.0), 14.0)
        ex.select((20.0, 20.0, 0.0), 3.5, lambda p: 5.0)
        self.assertEqual(ex.stats.path_queries, 1)
        self.assertEqual(ex.stats.budget_overruns, 1)


class ScoringTest(unittest.TestCase):
    def _cand(self, gain: float, chest: float = 0.0, risk: float = 0.0,
              x: float = 0.0) -> Candidate:
        return Candidate((int(x), 0, 0), (x, 0.0, 0.0), gain, chest, risk, "open")

    def _score(self, strategy: str, c: Candidate, travel: float) -> float:
        return ActiveExplorer(ExplorationConfig(strategy=strategy)).score(c, travel)

    def test_score_follows_the_formula(self) -> None:
        cfg = ExplorationConfig()
        ex = ActiveExplorer(cfg)
        c = self._cand(12.0, 0.5, 2.0)
        expected = (1.0 * 12.0 + 0.0 * 0.5) / (10.0 + cfg.scan_seconds + cfg.epsilon_s) \
            - 0.01 * 2.0
        self.assertAlmostEqual(ex.score(c, 10.0), expected)

    def test_strategies_rank_differently(self) -> None:
        near_small, far_big = self._cand(4.0), self._cand(40.0)
        self.assertGreater(self._score("nearest_frontier", near_small, 2.0),
                           self._score("nearest_frontier", far_big, 20.0))
        self.assertLess(self._score("max_information", near_small, 2.0),
                        self._score("max_information", far_big, 20.0))
        plain, chesty = self._cand(10.0, 0.0), self._cand(10.0, 0.5)
        self.assertGreater(self._score("chest_prior", chesty, 10.0) -
                           self._score("chest_prior", plain, 10.0),
                           self._score("max_information", chesty, 10.0) -
                           self._score("max_information", plain, 10.0))
        safe, risky = self._cand(10.0, risk=0.0), self._cand(10.0, risk=2.0)
        self.assertGreater(self._score("risk_adjusted", safe, 10.0) -
                           self._score("risk_adjusted", risky, 10.0), 0.5)

    def test_cluster_strategy_prefers_areas_near_discoveries(self) -> None:
        ex = ActiveExplorer(ExplorationConfig(strategy="cluster"))
        ex.note_discovery((30.0, 0.0, 0.0))
        self.assertGreater(ex.score(self._cand(10.0, x=28.0), 10.0),
                           ex.score(self._cand(10.0, x=-28.0), 10.0))

    def test_information_gain_counts_unseen_cells_and_open_edges(self) -> None:
        ex = ActiveExplorer(ExplorationConfig(), clock=lambda: 0.0)
        ex.add_fan((20.0, 20.0, 0.0), [15.0] * ex.cfg.ray_count)
        cands = ex._build_candidates((20.0, 20.0, 0.0), ())
        edge = [c for c in cands if any(ex.map.cells[k].open_edge for k in ex.map.cells
                                        if k == c.key)]
        self.assertTrue(cands)
        self.assertTrue(all(c.gain >= ex.cfg.min_gain_cells for c in cands))
        self.assertTrue(edge or max(c.gain for c in cands) > 4.0)

    def test_every_strategy_is_valid(self) -> None:
        for s in STRATEGIES:
            ExplorationConfig(strategy=s)
        with self.assertRaises(ValueError):
            ExplorationConfig(strategy="random_walk")


class CameraSectorTest(unittest.TestCase):
    def test_sector_coverage_tracks_observed_directions(self) -> None:
        cs = CameraSectors(12, cooldown_s=20.0)
        cs.observe(0.0, 90.0, now=5.0)
        observed = [k for k in range(12) if cs.last_observed[k] == 5.0]
        self.assertTrue(observed)
        self.assertTrue(all(abs((cs.centre(k) + 180.0) % 360.0 - 180.0) <= 45.0
                            for k in observed))

    def test_scan_prioritises_information_and_respects_its_limit(self) -> None:
        cs = CameraSectors(12, min_gain=1.5)
        gains = [0.0] * 12
        gains[3], gains[9], gains[4] = 10.0, 2.0, 6.0
        plan = cs.plan(0.0, gains, limit=2)
        self.assertLessEqual(len(plan), 2)
        self.assertIn(cs.centre(3), plan)
        self.assertEqual(cs.plan(0.0, [0.5] * 12), [], "nothing to learn: no scan")

    def test_scan_sweeps_in_one_direction(self) -> None:
        cs = CameraSectors(12, min_gain=1.0)
        gains = [3.0] * 12
        plan = cs.plan(0.0, gains, limit=5)
        idx = [cs.index(y) for y in plan]
        steps = {(b - a) % 12 for a, b in zip(idx, idx[1:], strict=False)}
        self.assertEqual(len(steps), 1, f"the camera changed direction: {idx}")

    def test_recently_seen_sectors_are_discounted_unless_uncertain(self) -> None:
        cs = CameraSectors(12, cooldown_s=20.0)
        cs.observe(0.0, 30.0, now=10.0)
        k = cs.index(0.0)
        unseen = [4.0] * 12
        self.assertLess(cs.gains(unseen, now=15.0)[k], cs.gains(unseen, now=40.0)[k])
        cs.note_detection(0.0, 0.4, confirmed=False)      # rescan an uncertain detection
        self.assertGreater(cs.gains(unseen, now=15.0)[k], unseen[k])


class ThresholdTest(unittest.TestCase):
    def _track(self, confidence: float, hits: int, sigma: float = 0.5,
               category: str = "chest") -> SimpleNamespace:
        return SimpleNamespace(confidence=confidence, hits=hits, sigma=sigma,
                               category=category)

    def test_classification(self) -> None:
        th = ConfidenceThresholds()
        self.assertEqual(th.classify(self._track(0.8, 3)), "approach")
        self.assertEqual(th.classify(self._track(0.8, 1)), "inspect")       # too few hits
        self.assertEqual(th.classify(self._track(0.8, 3, sigma=3.0)), "inspect")
        self.assertEqual(th.classify(self._track(0.4, 3)), "inspect")
        self.assertEqual(th.classify(self._track(0.2, 9)), "inspect")       # seen often
        self.assertEqual(th.classify(self._track(0.2, 1)), "ignore")
        self.assertEqual(th.classify(self._track(0.9, 9, category="chest_opened")), "ignore")


class DiscoveryWorldModelTest(unittest.TestCase):
    def test_tracks_get_one_spawn_point_and_opened_chests_are_not_duplicated(self) -> None:
        from chest_hunter.localization.world_model_3d import (
            ASSOCIATION_RADIUS,
            PriorStatus3D,
            WorldModel3D,
        )
        from chest_hunter.vision.tracker_3d import TrackState3D

        world = WorldModel3D([], discover=True, thresholds=ConfidenceThresholds())

        def track(tid: int, x: float) -> SimpleNamespace:
            return SimpleNamespace(track_id=tid, location_id=None, category="chest",
                                   xyz=(x, 5.0, 0.0), confidence=0.9, hits=4, sigma=0.3,
                                   state=TrackState3D.CONFIRMED)

        first = track(1, 10.0)
        world.tracker = SimpleNamespace(tracks=[first], get=lambda tid: first if tid == 1
                                        else None, confirmed=lambda: [first])
        world._associate()
        self.assertEqual(len(world.priors), 1)
        prior = next(iter(world.priors.values()))
        self.assertLess(prior.location_id, 0)
        prior.status = PriorStatus3D.OPENED
        # the opened chest is seen again as a new track from another side
        again = track(2, 10.0 + ASSOCIATION_RADIUS * 0.5)
        world.tracker = SimpleNamespace(tracks=[first, again], get=lambda tid: {1: first,
                                                                                 2: again}[tid],
                                        confirmed=lambda: [first, again])
        world._associate()
        self.assertEqual(len(world.priors), 1, "an opened chest got a second spawn point")
        self.assertEqual(again.location_id, prior.location_id)
        self.assertNotIn(prior.location_id, {p.location_id for p in world.chest_targets()})


if __name__ == "__main__":
    unittest.main()
