"""Tests for occupancy-grid estimation from rendered frames."""

from __future__ import annotations

import unittest

import numpy as np

from chest_hunter.config import SimulationSettings, load_config
from chest_hunter.simulation.collision import GridMap
from chest_hunter.simulation.entities import Obstacle
from chest_hunter.simulation.environment import Environment, World, WorldTransform
from chest_hunter.simulation.map_generator import MapGenerator
from chest_hunter.vision.observation import SimulatorObservationProvider
from chest_hunter.vision.obstacle_detector import (
    BLOCKED,
    FREE,
    UNKNOWN,
    ObstacleDetector,
    OccupancyGrid,
)
from chest_hunter.vision.preprocessing import Preprocessor
from chest_hunter.vision.templates import TemplateLibrary

LIBRARY = TemplateLibrary.load(load_config().template_dir())
SIM = SimulationSettings(default_map="t", default_spawn_probability=1.0, default_open_seconds=2.0,
                         decoration_density=0.004)


def make_env(seed: int = 4) -> Environment:
    grid = MapGenerator(100, 100, density=0.15, seed=seed).generate([(50, 50)]).grid
    world = World("t", seed, grid, (), (), (50.5, 50.5), WorldTransform(1.0))
    return Environment(world, SIM, seed=seed)


class ObstacleDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_env()
        self.provider = SimulatorObservationProvider(self.env, seed=2)
        self.grid = OccupancyGrid(100, 100)
        self.detector = ObstacleDetector()
        self.pre = Preprocessor(LIBRARY)

    def observe(self) -> tuple[int, int, int, int]:
        obs = self.provider.capture()
        assert obs is not None
        update = self.detector.observe(self.pre.prepare(obs), self.grid)
        return update.window

    def truth(self) -> np.ndarray:
        return np.frombuffer(self.env.grid.cells, np.uint8).reshape(100, 100)

    def test_visible_cells_match_ground_truth_and_unseen_stay_unknown(self) -> None:
        x0, x1, y0, y1 = self.observe()
        window = self.grid.cells[y0:y1, x0:x1]
        truth = self.truth()[y0:y1, x0:x1]
        self.assertTrue(np.all(window != UNKNOWN))
        self.assertTrue(np.array_equal(window == BLOCKED, truth == 1))
        self.assertEqual((x1 - x0, y1 - y0), (40, 22))
        outside = self.grid.cells.copy()
        outside[y0:y1, x0:x1] = 0
        self.assertTrue(np.all(outside[outside != 0] == UNKNOWN))
        self.assertEqual(self.grid.counts()["unknown"], 100 * 100 - 40 * 22)
        self.assertGreater(self.grid.seen[y0, x0], 0)

    def test_map_grows_as_the_agent_moves(self) -> None:
        self.observe()
        known_before = self.grid.known_fraction
        version = self.grid.version
        for x, y in ((20.5, 20.5), (80.5, 80.5), (20.5, 80.5), (80.5, 20.5)):
            self.env.agent.position_x, self.env.agent.position_y = x, y
            self.observe()
        self.assertGreater(self.grid.known_fraction, known_before + 0.2)
        self.assertGreater(self.grid.version, version)
        known = self.grid.cells != UNKNOWN
        self.assertTrue(np.array_equal(self.grid.cells[known] == BLOCKED, self.truth()[known] == 1))

    def test_new_obstacles_are_discovered(self) -> None:
        x0, x1, y0, y1 = self.observe()
        truth = self.truth()
        # A vertical 1x3 strip that is free in ground truth, visible, away from the agent.
        strip = next((x, y) for x in range(x0 + 2, x1 - 2) for y in range(y0 + 2, y1 - 5)
                     if abs(x - 50) > 3 and all(truth[y + k, x] == 0 for k in range(3)))
        x, y = strip
        self.assertEqual(self.grid.state(x, y), FREE)
        self.env.add_obstacle(Obstacle(x, y, 1, 3))
        obs = self.provider.capture()
        assert obs is not None
        update = self.detector.observe(self.pre.prepare(obs), self.grid)
        self.assertTrue(update.changed)
        self.assertEqual(update.newly_blocked, 3)
        self.assertEqual(self.grid.state(x, y + 1), BLOCKED)

    def test_out_of_map_cells_are_blocked_for_queries(self) -> None:
        self.assertEqual(self.grid.state(-1, 5), BLOCKED)
        self.assertEqual(self.grid.state(5, 5), UNKNOWN)

    def test_small_custom_map_with_out_of_bounds_border(self) -> None:
        grid = GridMap(30, 30)
        grid.fill_rect(10, 10, 5, 5)
        env = Environment(World("s", 0, grid, (), (), (5.5, 5.5), WorldTransform(1.0)),
                          SIM, seed=0)
        occupancy = OccupancyGrid(30, 30)
        obs = SimulatorObservationProvider(env).capture()
        assert obs is not None
        ObstacleDetector().observe(Preprocessor(LIBRARY).prepare(obs), occupancy)
        self.assertEqual(occupancy.state(12, 12), BLOCKED)
        self.assertEqual(occupancy.state(5, 5), FREE)


if __name__ == "__main__":
    unittest.main()
