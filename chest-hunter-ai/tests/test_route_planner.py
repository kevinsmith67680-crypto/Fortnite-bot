"""Tests for chest selection strategies."""

from __future__ import annotations

import unittest

from chest_hunter.navigation.astar import DistanceOracle
from chest_hunter.navigation.route_planner import Candidate, RoutePlanner
from chest_hunter.simulation.collision import GridMap


def candidate(chest_id: int, x: int, y: int, p: float = 1.0, open_s: float = 0.0) -> Candidate:
    return Candidate(chest_id, (x, y), (x + 0.5, y + 0.5), p, open_s)


def planner(strategy: str, depth: int = 2) -> RoutePlanner:
    return RoutePlanner(DistanceOracle(), strategy=strategy, speed=5.0, acceleration=12.0,
                        lookahead_depth=depth)


class RoutePlannerTests(unittest.TestCase):
    def test_nearest_uses_path_length_not_database_order_or_straight_line(self) -> None:
        grid = GridMap(30, 30)
        grid.fill_rect(12, 0, 1, 25)  # wall between agent and the straight-line closest chest
        near_by_line = candidate(1, 14, 10)      # 4 cells away in a straight line
        near_by_path = candidate(2, 10, 17)      # 7 cells away, no wall
        selection = planner("nearest").select(grid, (10, 10), [near_by_line, near_by_path])
        assert selection is not None
        self.assertEqual(selection.target.chest_id, 2)
        self.assertEqual(selection.strategy, "nearest")

    def test_weighted_prefers_expected_openings_per_second(self) -> None:
        grid = GridMap(40, 40)
        unlikely_near = candidate(1, 13, 10, p=0.1)
        likely_far = candidate(2, 20, 10, p=1.0)
        self.assertEqual(planner("nearest").select(grid, (10, 10), [unlikely_near, likely_far])
                         .target.chest_id, 1)  # type: ignore[union-attr]
        selection = planner("weighted").select(grid, (10, 10), [unlikely_near, likely_far])
        assert selection is not None
        self.assertEqual(selection.target.chest_id, 2)
        self.assertGreater(selection.score.score, 0.0)

    def test_score_formula_uses_path_time_and_expected_interaction(self) -> None:
        grid = GridMap(20, 20)
        plan = planner("weighted")
        scored = plan.score(grid, (0, 0), candidate(1, 10, 0, p=0.5, open_s=2.0))
        travel = 10 / 5.0 + 5.0 / 12.0
        self.assertAlmostEqual(scored.travel_time, travel)
        self.assertAlmostEqual(scored.total_time, travel + 0.5 * 2.0)
        self.assertAlmostEqual(scored.score, 0.5 / (travel + 1.0))
        self.assertEqual(scored.reachability, 1.0)

    def test_unreachable_candidates_get_zero_q_and_are_excluded(self) -> None:
        grid = GridMap(20, 20)
        for x, y in [(14, 14), (15, 14), (16, 14), (14, 15), (16, 15), (14, 16), (15, 16),
                     (16, 16)]:
            grid.set_blocked(x, y)
        enclosed = candidate(1, 15, 15, p=1.0)
        open_one = candidate(2, 3, 3, p=0.5)
        plan = planner("weighted")
        self.assertEqual(plan.score(grid, (0, 0), enclosed).reachability, 0.0)
        selection = plan.select(grid, (0, 0), [enclosed, open_one])
        assert selection is not None
        self.assertEqual(selection.target.chest_id, 2)
        self.assertEqual(selection.unreachable, (1,))
        self.assertIsNone(plan.select(grid, (0, 0), [enclosed]))
        self.assertIsNone(plan.select(grid, (0, 0), []))

    def test_ties_are_broken_deterministically(self) -> None:
        grid = GridMap(21, 21)
        left, right = candidate(7, 5, 10), candidate(3, 15, 10)
        for strategy in ("nearest", "weighted", "lookahead"):
            with self.subTest(strategy=strategy):
                chosen = {planner(strategy).select(grid, (10, 10), order).target.chest_id  # type: ignore[union-attr]
                          for order in ([left, right], [right, left])}
                self.assertEqual(chosen, {3})

    def test_lookahead_prefers_a_cluster_over_an_isolated_chest(self) -> None:
        grid = GridMap(40, 21)
        isolated = candidate(1, 6, 10)
        cluster = [candidate(2, 15, 10), candidate(3, 16, 10), candidate(4, 17, 10)]
        options = [isolated, *cluster]
        self.assertEqual(planner("weighted").select(grid, (10, 10), options)
                         .target.chest_id, 1)  # type: ignore[union-attr]
        self.assertEqual(planner("lookahead").select(grid, (10, 10), options)
                         .target.chest_id, 2)  # type: ignore[union-attr]

    def test_selection_changes_as_candidates_are_resolved(self) -> None:
        grid = GridMap(30, 30)
        options = [candidate(1, 2, 0), candidate(2, 6, 0), candidate(3, 12, 0)]
        plan = planner("nearest")
        order = []
        position = (0, 0)
        while options:
            selection = plan.select(grid, position, options)
            assert selection is not None
            order.append(selection.target.chest_id)
            position = selection.target.cell
            options = [c for c in options if c.chest_id != selection.target.chest_id]
        self.assertEqual(order, [1, 2, 3])
        self.assertEqual(plan.selections, 3)

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            RoutePlanner(DistanceOracle(), strategy="random", speed=1, acceleration=1)
        with self.assertRaises(ValueError):
            RoutePlanner(DistanceOracle(), strategy="nearest", speed=0, acceleration=1)


if __name__ == "__main__":
    unittest.main()
