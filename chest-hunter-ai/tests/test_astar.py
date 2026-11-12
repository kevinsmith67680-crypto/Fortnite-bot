"""Tests for A*, the cached PathFinder and Dijkstra distance fields."""

from __future__ import annotations

import math
import random
import unittest

from chest_hunter.navigation.astar import (
    SQRT2,
    DistanceOracle,
    FailureReason,
    PathFinder,
    astar,
    distance_field,
)
from chest_hunter.simulation.collision import GridMap
from chest_hunter.simulation.map_generator import MapGenerator


def assert_valid_path(test: unittest.TestCase, grid: GridMap, path: tuple[tuple[int, int], ...],
                      *, diagonal: bool = True, corner_cutting: bool = False) -> float:
    """Check contiguity, walkability and corner rules; return the path cost."""
    cost = 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:], strict=False):
        dx, dy = x1 - x0, y1 - y0
        test.assertLessEqual(max(abs(dx), abs(dy)), 1)
        test.assertNotEqual((dx, dy), (0, 0))
        test.assertFalse(grid.is_blocked(x1, y1), f"path enters blocked cell {(x1, y1)}")
        if dx and dy:
            test.assertTrue(diagonal, "diagonal step in a 4-directional path")
            if not corner_cutting:
                test.assertFalse(grid.is_blocked(x0 + dx, y0) or grid.is_blocked(x0, y0 + dy),
                                 f"corner cut at {(x0, y0)} -> {(x1, y1)}")
            cost += SQRT2
        else:
            cost += 1.0
    return cost


class AStarTests(unittest.TestCase):
    def test_straight_line_path(self) -> None:
        grid = GridMap(10, 10)
        result = astar(grid, (0, 0), (5, 0))
        self.assertTrue(result.found)
        self.assertEqual(result.path, tuple((x, 0) for x in range(6)))
        self.assertEqual(result.cost, 5.0)

    def test_route_around_a_wall(self) -> None:
        grid = GridMap(10, 10)
        grid.fill_rect(5, 0, 1, 9)  # gap only at y = 9
        result = astar(grid, (0, 0), (9, 0))
        self.assertTrue(result.found)
        self.assertIn((5, 9), result.path)
        cost = assert_valid_path(self, grid, result.path)
        self.assertAlmostEqual(cost, result.cost)
        self.assertGreater(result.cost, 9.0)

    def test_blocked_and_enclosed_destinations(self) -> None:
        grid = GridMap(10, 10)
        grid.set_blocked(7, 7)
        self.assertEqual(astar(grid, (0, 0), (7, 7)).reason, FailureReason.GOAL_BLOCKED)
        grid = GridMap(10, 10)
        for x, y in [(6, 6), (7, 6), (8, 6), (6, 7), (8, 7), (6, 8), (7, 8), (8, 8)]:
            grid.set_blocked(x, y)
        result = astar(grid, (0, 0), (7, 7))
        self.assertFalse(result.found)
        self.assertEqual(result.reason, FailureReason.UNREACHABLE)
        self.assertEqual(result.path, ())
        self.assertTrue(math.isinf(result.cost))
        self.assertGreater(result.expanded, 0)

    def test_start_equals_destination(self) -> None:
        result = astar(GridMap(5, 5), (2, 2), (2, 2))
        self.assertTrue(result.found)
        self.assertEqual(result.path, ((2, 2),))
        self.assertEqual(result.cost, 0.0)

    def test_start_or_destination_outside_map(self) -> None:
        grid = GridMap(5, 5)
        self.assertEqual(astar(grid, (-1, 0), (2, 2)).reason, FailureReason.START_OUT_OF_BOUNDS)
        self.assertEqual(astar(grid, (0, 0), (5, 2)).reason, FailureReason.GOAL_OUT_OF_BOUNDS)
        grid.set_blocked(0, 0)
        self.assertEqual(astar(grid, (0, 0), (2, 2)).reason, FailureReason.START_BLOCKED)

    def test_diagonal_movement_and_four_directional_mode(self) -> None:
        grid = GridMap(10, 10)
        diagonal = astar(grid, (0, 0), (5, 5))
        self.assertAlmostEqual(diagonal.cost, 5 * SQRT2)
        self.assertEqual(len(diagonal.path), 6)
        four = astar(grid, (0, 0), (5, 5), allow_diagonal=False)
        self.assertEqual(four.cost, 10.0)
        assert_valid_path(self, grid, four.path, diagonal=False)

    def test_no_illegal_corner_cutting(self) -> None:
        grid = GridMap(4, 4)
        grid.set_blocked(1, 0)
        grid.set_blocked(0, 1)
        self.assertFalse(astar(grid, (0, 0), (1, 1)).found)
        permissive = astar(grid, (0, 0), (1, 1), prevent_corner_cutting=False)
        self.assertTrue(permissive.found)
        self.assertAlmostEqual(permissive.cost, SQRT2)

        grid = GridMap(6, 6)
        grid.set_blocked(2, 2)
        result = astar(grid, (1, 1), (3, 3))
        assert_valid_path(self, grid, result.path)
        self.assertNotIn((2, 2), result.path)

    def test_paths_on_generated_maps_are_valid_and_optimal(self) -> None:
        grid = MapGenerator(40, 40, density=0.25, seed=3).generate().grid
        rng = random.Random(11)
        walkable = list(grid.walkable_cells())
        for _ in range(25):
            start, goal = rng.choice(walkable), rng.choice(walkable)
            result = astar(grid, start, goal)
            field = distance_field(grid, goal)
            expected = field[start[1] * grid.width + start[0]]
            if math.isinf(expected):
                self.assertFalse(result.found)
                continue
            self.assertTrue(result.found)
            self.assertAlmostEqual(result.cost, expected, places=9)
            self.assertAlmostEqual(assert_valid_path(self, grid, result.path), result.cost)

    def test_deterministic_repeatability(self) -> None:
        grid_a = MapGenerator(30, 30, density=0.2, seed=5).generate().grid
        grid_b = MapGenerator(30, 30, density=0.2, seed=5).generate().grid
        results = {astar(grid_a, (1, 1), (28, 28)).path for _ in range(3)}
        results.add(astar(grid_b, (1, 1), (28, 28)).path)
        self.assertEqual(len(results), 1)


class PathFinderCacheTests(unittest.TestCase):
    def test_cache_hits_and_invalidation_on_obstacle_change(self) -> None:
        grid = GridMap(20, 20)
        finder = PathFinder()
        first = finder.find_path(grid, (0, 0), (10, 0))
        second = finder.find_path(grid, (0, 0), (10, 0))
        self.assertIs(first, second)
        self.assertEqual((finder.searches, finder.cache_hits), (1, 1))

        grid.fill_rect(5, 0, 1, 19)  # wall appears: cached path is stale
        third = finder.find_path(grid, (0, 0), (10, 0))
        self.assertEqual(finder.searches, 2)
        self.assertNotIn((5, 0), third.path)
        self.assertEqual(finder.invalidations, 1)

        replacement = grid.copy()  # a different grid object also invalidates
        finder.find_path(replacement, (0, 0), (10, 0))
        self.assertEqual(finder.searches, 3)

    def test_cache_can_be_disabled(self) -> None:
        finder = PathFinder(cache_paths=False)
        grid = GridMap(5, 5)
        finder.find_path(grid, (0, 0), (4, 4))
        finder.find_path(grid, (0, 0), (4, 4))
        self.assertEqual((finder.searches, finder.cache_hits), (2, 0))


class DistanceOracleTests(unittest.TestCase):
    def test_distances_are_symmetric_cached_and_match_astar(self) -> None:
        grid = MapGenerator(30, 30, density=0.2, seed=8).generate().grid
        oracle = DistanceOracle()
        cells = [c for c in grid.walkable_cells()][::97][:6]
        for a in cells:
            for b in cells:
                d_ab = oracle.distance(grid, a, b)
                d_ba = oracle.distance(grid, b, a)
                self.assertAlmostEqual(d_ab, d_ba, places=9)
                expected = astar(grid, a, b)
                self.assertAlmostEqual(d_ab, expected.cost if expected.found else math.inf)
        self.assertEqual(oracle.fields_computed, len(cells))
        self.assertTrue(math.isinf(oracle.distance(grid, (-1, 0), cells[0])))

    def test_oracle_invalidates_when_grid_changes(self) -> None:
        grid = GridMap(10, 10)
        oracle = DistanceOracle()
        self.assertEqual(oracle.distance(grid, (0, 0), (9, 0)), 9.0)
        grid.fill_rect(5, 0, 1, 10)
        self.assertTrue(math.isinf(oracle.distance(grid, (0, 0), (9, 0))))
        self.assertEqual(oracle.fields_computed, 2)


if __name__ == "__main__":
    unittest.main()
