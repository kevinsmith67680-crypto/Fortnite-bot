"""Tests for the observation interface, camera geometry and preprocessing."""

from __future__ import annotations

import unittest

import numpy as np

from chest_hunter.config import SimulationSettings
from chest_hunter.simulation.camera import (
    SUPPORTED_RESOLUTIONS,
    CameraModel,
    FollowCamera,
    FrameRenderer,
    SensorModel,
)
from chest_hunter.simulation.collision import GridMap
from chest_hunter.simulation.environment import ChestSpec, Environment, World, WorldTransform
from chest_hunter.vision import preprocessing as pp
from chest_hunter.vision.observation import (
    ObservationError,
    SimulatorObservationProvider,
    VisualObservation,
)

SIM = SimulationSettings(default_map="t", default_spawn_probability=1.0, default_open_seconds=2.0,
                         decoration_density=0.0)


def make_env(size: int = 60) -> Environment:
    grid = GridMap(size, size)
    grid.fill_rect(20, 20, 4, 1)
    grid.version = 0
    chests = (ChestSpec(1, "a", 30.5, 31.5, 1.0, 2.0),)
    world = World("t", 0, grid, (), chests, (30.5, 30.5), WorldTransform(1.0))
    return Environment(world, SIM, seed=1)


def camera(width: int = 640, height: int = 360) -> CameraModel:
    return CameraModel(10.0, 20.0, width / 40, width, height)


class ObservationModelTests(unittest.TestCase):
    def test_validation(self) -> None:
        frame = np.zeros((360, 640, 3), np.uint8)
        VisualObservation(frame, 1.5, 640, 360, 0, camera())
        bad_cases = [
            (np.zeros((360, 640, 3), np.float32), 0.0, 640, 360, 0),  # dtype
            (np.zeros((360, 640), np.uint8), 0.0, 640, 360, 0),       # channels
            (np.zeros((360, 640, 3), np.uint8), 0.0, 320, 180, 0),    # dimensions
            (np.zeros((360, 640, 3), np.uint8), -1.0, 640, 360, 0),   # timestamp
            (np.zeros((360, 640, 3), np.uint8), float("nan"), 640, 360, 0),
            (np.zeros((360, 640, 3), np.uint8), 0.0, 640, 360, -1),   # frame id
        ]
        for frame_arg, ts, w, h, fid in bad_cases:
            with self.subTest(shape=frame_arg.shape, ts=ts, w=w, fid=fid):
                with self.assertRaises(ObservationError):
                    VisualObservation(frame_arg, ts, w, h, fid, camera())
        with self.assertRaises(ObservationError):
            VisualObservation(frame, 0.0, 640, 360, 0, camera(320, 180))

    def test_camera_geometry_round_trip(self) -> None:
        cam = camera()
        self.assertEqual(cam.pixels_per_cell, 16.0)
        self.assertEqual(cam.view_height, 22.5)
        u, v = cam.world_to_pixel(12.25, 30.0)
        self.assertEqual((u, v), (36.0, 200.0))
        self.assertEqual(cam.pixel_to_world(u, v), (12.25, 30.0))
        self.assertTrue(cam.contains(10.5, 20.5))
        self.assertFalse(cam.contains(9.5, 25.0))
        self.assertEqual(cam.visible_cells(100, 100), (10, 50, 20, 42))


class ProviderTests(unittest.TestCase):
    def test_all_resolutions_produce_valid_frames(self) -> None:
        env = make_env()
        for width, height in SUPPORTED_RESOLUTIONS:
            with self.subTest(size=(width, height)):
                provider = SimulatorObservationProvider(env, width=width, height=height)
                obs = provider.capture()
                assert obs is not None
                self.assertEqual(obs.frame.shape, (height, width, 3))
                self.assertEqual(obs.frame.dtype, np.uint8)
                self.assertEqual((obs.width, obs.height), (width, height))
                self.assertEqual(obs.camera.pixels_per_cell, width / 40)
        with self.assertRaises(ObservationError):
            SimulatorObservationProvider(env, width=1000, height=500)

    def test_timestamps_and_frame_ids(self) -> None:
        env = make_env()
        provider = SimulatorObservationProvider(env)
        first = provider.capture()
        for _ in range(15):
            env.advance()
        second = provider.capture()
        assert first is not None and second is not None
        self.assertEqual(first.timestamp, 0.0)
        self.assertAlmostEqual(second.timestamp, 0.5)
        self.assertEqual((first.frame_id, second.frame_id), (0, 1))

    def test_failed_captures_return_none(self) -> None:
        env = make_env()
        provider = SimulatorObservationProvider(env, sensor=SensorModel(frame_dropout=0.9),
                                                seed=3)
        results = [provider.capture() for _ in range(40)]
        self.assertGreater(sum(r is None for r in results), 25)
        self.assertEqual(provider.frames_failed, sum(r is None for r in results))
        self.assertEqual(provider.frames_captured, sum(r is not None for r in results))

    def test_buffer_reuse_and_copies(self) -> None:
        env = make_env()
        shared = SimulatorObservationProvider(env)
        a, b = shared.capture(), shared.capture()
        assert a is not None and b is not None
        self.assertIs(a.frame, b.frame)
        copying = SimulatorObservationProvider(env, reuse_buffer=False)
        c, d = copying.capture(), copying.capture()
        assert c is not None and d is not None
        self.assertIsNot(c.frame, d.frame)

    def test_rendering_is_deterministic_and_display_independent(self) -> None:
        """GUI and headless runs use the same NumPy renderer, so frames are identical."""
        env = make_env()
        a = SimulatorObservationProvider(env, seed=9, reuse_buffer=False)
        b = SimulatorObservationProvider(env, seed=9, reuse_buffer=False)
        for _ in range(3):
            fa, fb = a.capture(), b.capture()
            assert fa is not None and fb is not None
            np.testing.assert_array_equal(fa.frame, fb.frame)

    def test_follow_camera_keeps_agent_in_dead_zone(self) -> None:
        cam = FollowCamera(100, 100, 640, 360)
        first = cam.update(50.0, 50.0)
        self.assertTrue(first.contains(50.0, 50.0, margin=5.0))
        moved = cam.update(80.0, 50.0)
        self.assertGreater(moved.origin_x, first.origin_x)
        self.assertTrue(moved.contains(80.0, 50.0, margin=5.0))
        edge = cam.update(99.5, 99.5)   # clamped, but the edge row is fully visible
        self.assertEqual(edge.visible_cells(100, 100)[1], 100)
        self.assertEqual(edge.visible_cells(100, 100)[3], 100)
        with self.assertRaises(ValueError):
            FollowCamera(100, 100, 650, 360)

    def test_absent_chests_are_not_rendered(self) -> None:
        env = make_env()
        renderer = FrameRenderer(640, 360, sensor=SensorModel(0.0, 0.0))
        cam = FollowCamera(60, 60, 640, 360).update(30.5, 30.5)
        with_chest = renderer.render(env, cam, 0).copy()
        env.chests[1].present = False
        without = renderer.render(env, cam, 0)
        self.assertGreater(int(np.count_nonzero(with_chest != without)), 50)


class PreprocessingTests(unittest.TestCase):
    def test_helpers(self) -> None:
        image = np.full((40, 60, 3), 100, np.uint8)
        image[10:20, 10:30] = 200
        self.assertEqual(pp.resize(image, 0.5).shape, (20, 30, 3))
        self.assertIs(pp.resize(image, 1.0), image)
        self.assertEqual(pp.to_gray(image).shape, (40, 60))
        self.assertEqual(pp.to_hsv(image).shape, (40, 60, 3))
        self.assertEqual(int(pp.adjust_contrast(image, 2.0, 0.0).max()), 255)
        self.assertEqual(pp.crop(image, (50, 30, 20, 20)).shape, (10, 10, 3))
        self.assertEqual(int(pp.normalise(pp.to_gray(image)).min()), 0)
        self.assertGreater(int(np.count_nonzero(pp.edges(pp.to_gray(image)))), 0)

    def test_identical_frames_are_processed_once(self) -> None:
        from chest_hunter.config import load_config
        from chest_hunter.vision.templates import TemplateLibrary

        library = TemplateLibrary.load(load_config().template_dir())
        pre = pp.Preprocessor(library)
        obs = VisualObservation(np.zeros((360, 640, 3), np.uint8), 0.0, 640, 360, 7, camera())
        first = pre.prepare(obs)
        self.assertIs(pre.prepare(obs), first)
        self.assertEqual((pre.frames_processed, pre.frames_skipped), (1, 1))
        mask = first.mask("ground")
        self.assertIs(first.mask("ground"), mask)   # cached per frame


if __name__ == "__main__":
    unittest.main()
