"""Tests for the occupancy grid, circle collision and line-of-sight checks."""

from __future__ import annotations

import random
import unittest

from chest_hunter.simulation.collision import (
    GridMap,
    circle_collides,
    move_with_collision,
    segment_clear,
    supercover_cells,
)


class GridMapTests(unittest.TestCase):
    def test_bounds_and_out_of_bounds_cells_are_blocked(self) -> None:
        grid = GridMap(5, 4)
        self.assertTrue(grid.in_bounds(4, 3))
        self.assertFalse(grid.in_bounds(5, 0))
        self.assertTrue(grid.is_blocked(-1, 0))
        self.assertTrue(grid.is_blocked(0, 4))
        self.assertTrue(grid.is_walkable(2, 2))

    def test_version_changes_only_when_occupancy_changes(self) -> None:
        grid = GridMap(5, 5)
        grid.set_blocked(1, 1)
        self.assertEqual(grid.version, 1)
        grid.set_blocked(1, 1)  # no change
        self.assertEqual(grid.version, 1)
        self.assertEqual(grid.fill_rect(0, 0, 3, 1), 3)
        self.assertEqual(grid.version, 2)
        self.assertEqual(grid.fill_rect(0, 0, 3, 1), 0)
        self.assertEqual(grid.version, 2)
        with self.assertRaises(ValueError):
            grid.set_blocked(9, 9)

    def test_fill_rect_clips_to_bounds_and_counts(self) -> None:
        grid = GridMap(4, 4)
        grid.fill_rect(2, 2, 10, 10)
        self.assertEqual(grid.blocked_count, 4)
        self.assertAlmostEqual(grid.blocked_fraction, 0.25)
        self.assertEqual(len(list(grid.walkable_cells())), 12)

    def test_copy_is_independent(self) -> None:
        grid = GridMap(4, 4)
        grid.set_blocked(0, 0)
        clone = grid.copy()
        self.assertEqual(clone, grid)
        clone.set_blocked(1, 1)
        self.assertNotEqual(clone, grid)
        self.assertFalse(grid.is_blocked(1, 1))

    def test_cell_helpers(self) -> None:
        self.assertEqual(GridMap.cell_of(2.99, 0.01), (2, 0))
        self.assertEqual(GridMap.cell_of(-0.5, 1.0), (-1, 1))
        self.assertEqual(GridMap.cell_center((3, 4)), (3.5, 4.5))


class CircleCollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = GridMap(10, 10)
        self.grid.fill_rect(5, 0, 1, 10)  # wall column x = 5

    def test_free_space_and_walls(self) -> None:
        self.assertFalse(circle_collides(self.grid, 2.5, 2.5, 0.3))
        self.assertTrue(circle_collides(self.grid, 4.8, 2.5, 0.3))   # overlaps wall
        self.assertTrue(circle_collides(self.grid, 5.5, 2.5, 0.3))   # inside wall
        self.assertFalse(circle_collides(self.grid, 4.7, 2.5, 0.3))  # touching only
        self.assertTrue(circle_collides(self.grid, 0.1, 2.5, 0.3))   # map edge

    def test_corner_distance_is_respected(self) -> None:
        grid = GridMap(4, 4)
        grid.set_blocked(2, 2)
        # Diagonal distance to the corner (2, 2) from (1.75, 1.75) is ~0.354.
        self.assertFalse(circle_collides(grid, 1.75, 1.75, 0.3))
        self.assertTrue(circle_collides(grid, 1.85, 1.85, 0.3))

    def test_move_slides_along_walls(self) -> None:
        x, y, bx, by = move_with_collision(self.grid, 4.5, 2.5, 0.4, 0.1, 0.3)
        self.assertTrue(bx)
        self.assertFalse(by)
        self.assertEqual(x, 4.5)
        self.assertAlmostEqual(y, 2.6)

    def test_random_moves_never_end_inside_obstacles(self) -> None:
        rng = random.Random(7)
        grid = GridMap(20, 20)
        for _ in range(60):
            grid.set_blocked(rng.randrange(20), rng.randrange(20))
        x, y = next((cx + 0.5, cy + 0.5) for cx, cy in grid.walkable_cells()
                    if not circle_collides(grid, cx + 0.5, cy + 0.5, 0.3))
        for _ in range(5000):
            dx, dy = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
            x, y, _, _ = move_with_collision(grid, x, y, dx, dy, 0.3)
            self.assertFalse(circle_collides(grid, x, y, 0.3))


class LineOfSightTests(unittest.TestCase):
    def test_supercover_horizontal_and_diagonal(self) -> None:
        self.assertEqual(list(supercover_cells((0.5, 0.5), (3.5, 0.5))),
                         [(0, 0), (1, 0), (2, 0), (3, 0)])
        cells = list(supercover_cells((0.5, 0.5), (2.5, 2.5)))
        # Passing exactly through corners includes both side cells.
        for expected in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]:
            self.assertIn(expected, cells)
        self.assertEqual(cells[-1], (2, 2))

    def test_supercover_handles_negative_directions_and_single_cell(self) -> None:
        self.assertEqual(list(supercover_cells((3.5, 0.5), (0.5, 0.5))),
                         [(3, 0), (2, 0), (1, 0), (0, 0)])
        self.assertEqual(list(supercover_cells((1.2, 1.2), (1.8, 1.9))), [(1, 1)])

    def test_segment_clear(self) -> None:
        grid = GridMap(10, 10)
        grid.fill_rect(5, 0, 1, 8)
        self.assertTrue(segment_clear(grid, (1.5, 1.5), (4.5, 1.5)))
        self.assertFalse(segment_clear(grid, (1.5, 1.5), (8.5, 1.5)))
        self.assertTrue(segment_clear(grid, (1.5, 8.5), (8.5, 8.5), 0.3))
        # Thin line fits past the wall end, the agent's body does not.
        self.assertTrue(segment_clear(grid, (4.5, 8.4), (6.5, 7.9)))
        self.assertFalse(segment_clear(grid, (4.5, 8.4), (6.5, 7.9), 0.3))

    def test_diagonal_squeeze_between_blocked_cells_is_rejected(self) -> None:
        grid = GridMap(4, 4)
        grid.set_blocked(1, 0)
        grid.set_blocked(0, 1)
        self.assertFalse(segment_clear(grid, (0.5, 0.5), (1.5, 1.5)))


if __name__ == "__main__":
    unittest.main()
