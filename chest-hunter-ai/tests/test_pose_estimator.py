"""Tests for visual self-localisation."""

from __future__ import annotations

import math
import random
import unittest

import numpy as np

from chest_hunter.config import SimulationSettings, load_config
from chest_hunter.localization.pose_estimator import (
    AgentMarkerDetector,
    PoseEstimator,
    PoseMeasurement,
)
from chest_hunter.simulation import sprites
from chest_hunter.simulation.camera import CameraModel
from chest_hunter.simulation.collision import GridMap
from chest_hunter.simulation.environment import Environment, World, WorldTransform
from chest_hunter.vision.observation import SimulatorObservationProvider, VisualObservation
from chest_hunter.vision.preprocessing import Preprocessor
from chest_hunter.vision.templates import TemplateLibrary

LIBRARY = TemplateLibrary.load(load_config().template_dir())
SIM = SimulationSettings(default_map="t", default_spawn_probability=1.0, default_open_seconds=2.0)


class MarkerDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        world = World("t", 0, GridMap(100, 100), (), (), (50.5, 50.5), WorldTransform(1.0))
        self.env = Environment(world, SIM, seed=0)
        self.provider = SimulatorObservationProvider(self.env, seed=1)
        self.detector = AgentMarkerDetector()

    def measure(self, x: float, y: float, heading: float = 0.0) -> PoseMeasurement | None:
        self.env.agent.position_x, self.env.agent.position_y = x, y
        self.env.agent.heading = heading
        obs = self.provider.capture()
        assert obs is not None
        return self.detector.detect(Preprocessor(LIBRARY).prepare(obs))

    def test_world_position_with_a_moving_camera(self) -> None:
        rng = random.Random(5)
        errors = []
        origins = set()
        for _ in range(60):
            x, y = rng.uniform(2, 98), rng.uniform(2, 98)
            m = self.measure(x, y, rng.uniform(-math.pi, math.pi))
            assert m is not None
            errors.append(math.dist((m.x, m.y), (x, y)))
            assert self.provider.last_camera is not None
            origins.add((self.provider.last_camera.origin_x, self.provider.last_camera.origin_y))
        self.assertGreater(len(origins), 10)            # the camera really moved
        self.assertLess(float(np.mean(errors)), 0.05)
        self.assertLess(max(errors), 0.1)

    def test_heading_estimate(self) -> None:
        for heading in (0.0, 1.0, -2.5):
            m = self.measure(40.5, 40.5, heading)
            assert m is not None and m.heading is not None
            diff = math.atan2(math.sin(m.heading - heading), math.cos(m.heading - heading))
            self.assertLess(abs(diff), math.radians(15))

    def test_missing_marker(self) -> None:
        cam = CameraModel(0.0, 0.0, 16.0, 640, 360)
        frame = np.full((360, 640, 3), sprites.GROUND, np.uint8)
        obs = VisualObservation(frame, 0.0, 640, 360, 0, cam)
        self.assertIsNone(self.detector.detect(Preprocessor(LIBRARY).prepare(obs)))
        self.assertEqual(self.detector.failures, 1)

    def test_coordinate_transform_is_not_identity(self) -> None:
        m = self.measure(70.25, 33.75)
        assert m is not None
        self.assertNotAlmostEqual(m.pixel[0], 70.25, delta=5)   # pixels are not cells
        self.assertAlmostEqual(m.x, 70.25, delta=0.05)


class PoseFilterTests(unittest.TestCase):
    def test_filter_reduces_noise_with_motion_prediction(self) -> None:
        rng = random.Random(1)
        estimator = PoseEstimator(smoothing_alpha=0.4)
        dt, vx = 1 / 30, 3.0
        true_x, raw_err, est_err = 10.0, [], []
        for tick in range(600):
            true_x += vx * dt
            estimator.predict(vx, 0.0, dt)
            if tick % 3 == 0:   # 10 Hz noisy fixes
                mx = true_x + rng.gauss(0, 0.2)
                estimator.correct(PoseMeasurement(mx, 5.0 + rng.gauss(0, 0.2), None, 1.0,
                                                  tick * dt, (0, 0)))
                if tick > 30:
                    raw_err.append(abs(mx - true_x))
                    est_err.append(abs((estimator.x or 0) - true_x))
        self.assertLess(float(np.mean(est_err)), 0.75 * float(np.mean(raw_err)))

    def test_confidence_degrades_and_pose_can_be_lost(self) -> None:
        estimator = PoseEstimator(max_missing_observations=2, dead_reckoning_seconds=2.0)
        self.assertEqual(estimator.confidence(), 0.0)
        self.assertTrue(estimator.is_lost)
        estimator.correct(PoseMeasurement(1.0, 1.0, 0.0, 1.0, 0.0, (0, 0)))
        self.assertAlmostEqual(estimator.confidence(0.0), 1.0)
        self.assertAlmostEqual(estimator.confidence(1.0), 0.5)
        self.assertEqual(estimator.confidence(3.0), 0.0)
        for t in (0.1, 0.2, 0.3):
            estimator.miss(t)
        self.assertTrue(estimator.is_lost)
        estimator.correct(PoseMeasurement(1.2, 1.0, None, 0.9, 0.4, (0, 0)))
        self.assertFalse(estimator.is_lost)

    def test_blocked_body_prediction_is_reset_by_the_next_fix(self) -> None:
        # The body is pressed against a wall while velocity is still commanded: dead
        # reckoning runs ahead. One clear fix must bring the estimate back, not 40% of it.
        estimator = PoseEstimator(smoothing_alpha=0.4)
        estimator.correct(PoseMeasurement(10.0, 5.0, 0.0, 1.0, 0.0, (0, 0)))
        for _ in range(15):                       # 0.5 s at 5 cells/s, true body stays put
            estimator.predict(5.0, 0.0, 1 / 30)
        self.assertGreater((estimator.x or 0) - 10.0, 2.0)
        estimator.correct(PoseMeasurement(10.0, 5.0, 0.0, 1.0, 0.5, (0, 0)))
        self.assertAlmostEqual(estimator.x or 0, 10.0)
        self.assertEqual(estimator.resets, 1)
        # Small disagreements are still smoothed.
        estimator.correct(PoseMeasurement(10.2, 5.0, 0.0, 1.0, 0.6, (0, 0)))
        self.assertAlmostEqual(estimator.x or 0, 10.08)
        self.assertEqual(estimator.resets, 1)

    def test_prediction_moves_the_estimate(self) -> None:
        estimator = PoseEstimator()
        estimator.predict(1.0, 0.0, 1.0)            # before initialisation: ignored
        self.assertIsNone(estimator.position)
        estimator.correct(PoseMeasurement(0.0, 0.0, None, 1.0, 0.0, (0, 0)))
        estimator.predict(2.0, -1.0, 0.5)
        self.assertEqual(estimator.position, (1.0, -0.5))
        self.assertAlmostEqual(estimator.heading or 0.0, math.atan2(-1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
