"""Tests for classical chest detection (colour/shape, template, hybrid)."""

from __future__ import annotations

import unittest

import numpy as np

from chest_hunter.config import load_config
from chest_hunter.simulation import sprites
from chest_hunter.simulation.camera import CameraModel
from chest_hunter.vision.chest_detector import (
    Detection,
    iou,
    make_detector,
    non_max_suppression,
)
from chest_hunter.vision.observation import VisualObservation
from chest_hunter.vision.preprocessing import Preprocessor
from chest_hunter.vision.templates import TemplateLibrary

METHODS = ("color", "template", "hybrid")
LIBRARY = TemplateLibrary.load(load_config().template_dir())


def frame_with(objects: list[tuple[str, float, float]], *, width: int = 640, height: int = 360,
               noise: float = 6.0, seed: int = 0) -> VisualObservation:
    """Ground background with sprites at cell coordinates (camera origin 0, 0)."""
    ppc = width / 40
    img = np.full((height, width, 3), sprites.GROUND, np.uint8)
    cam = CameraModel(0.0, 0.0, ppc, width, height)
    for kind, x, y in objects:
        u, v = cam.world_to_pixel(x, y)
        if kind in ("chest", "chest_opened"):
            sprites.draw_chest(img, u, v, ppc, opened=kind == "chest_opened")
        elif kind == "decoration":
            sprites.draw_decoration(img, u, v, ppc, 0.35)
        elif kind == "agent":
            sprites.draw_agent(img, u, v, ppc, 0.0)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return VisualObservation(img, 0.0, width, height, seed, cam)


def detect(method: str, obs: VisualObservation) -> list[Detection]:
    return make_detector(method, LIBRARY).detect(Preprocessor(LIBRARY).prepare(obs))


class ChestDetectorTests(unittest.TestCase):
    def test_chest_present_and_absent(self) -> None:
        for method in METHODS:
            with self.subTest(method=method):
                found = detect(method, frame_with([("chest", 10.5, 10.5)]))
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0].category, "chest")
                self.assertTrue(0.0 <= found[0].confidence <= 1.0)
                cx, cy = found[0].center
                self.assertAlmostEqual(cx, 10.5 * 16, delta=2)
                self.assertAlmostEqual(cy, (22.5 - 10.5) * 16, delta=2)
                self.assertEqual(detect(method, frame_with([])), [])

    def test_multiple_chests(self) -> None:
        objects = [("chest", 5.5, 5.5), ("chest", 15.5, 12.5), ("chest_opened", 30.5, 18.5)]
        for method in METHODS:
            with self.subTest(method=method):
                found = detect(method, frame_with(objects))
                self.assertEqual(sorted(d.category for d in found),
                                 ["chest", "chest", "chest_opened"])

    def test_multiple_scales(self) -> None:
        for width, height in ((320, 180), (480, 270), (640, 360), (800, 450)):
            with self.subTest(size=width):
                found = detect("hybrid", frame_with([("chest", 10.5, 10.5),
                                                     ("chest_opened", 20.5, 10.5)],
                                                    width=width, height=height))
                self.assertEqual(sorted(d.category for d in found), ["chest", "chest_opened"])

    def test_partially_visible_chest_at_frame_edge(self) -> None:
        obs = frame_with([("chest", 0.1, 10.5)])   # about half of the chest is outside
        found = detect("hybrid", obs)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].bbox[0], 0)
        self.assertLessEqual(found[0].bbox[2], 10)

    def test_partial_occlusion_by_the_agent(self) -> None:
        # The agent marker covers roughly 20% and 30% of the chest sprite.
        for dx in (0.55, 0.45):
            with self.subTest(dx=dx):
                found = detect("hybrid", frame_with([("chest", 10.5, 10.5),
                                                     ("agent", 10.5 + dx, 10.6)]))
                self.assertEqual([d.category for d in found], ["chest"])

    def test_similar_coloured_props_are_rejected(self) -> None:
        props = [("decoration", 5.5 + 3 * i, 6.5) for i in range(8)]
        for method in METHODS:
            with self.subTest(method=method):
                self.assertEqual(detect(method, frame_with(props)), [])
                found = detect(method, frame_with(props + [("chest", 12.5, 15.5)]))
                self.assertEqual(len(found), 1)

    def test_opened_chest_identification_under_noise(self) -> None:
        for seed in range(10):
            with self.subTest(seed=seed):
                found = detect("hybrid", frame_with([("chest_opened", 12.5, 9.5)], noise=8.0,
                                                    seed=seed))
                self.assertEqual([d.category for d in found], ["chest_opened"])

    def test_non_max_suppression(self) -> None:
        a = Detection("chest", (10, 10, 18, 15), 0.95, 0.0)
        b = Detection("chest", (11, 11, 18, 15), 0.90, 0.0)
        c = Detection("chest_opened", (12, 10, 18, 15), 0.88, 0.0)
        far = Detection("chest", (100, 100, 18, 15), 0.80, 0.0)
        kept = non_max_suppression([b, c, a, far])
        self.assertEqual(kept, [a, far])
        self.assertGreater(iou(a.bbox, b.bbox), 0.7)
        self.assertEqual(iou(a.bbox, far.bbox), 0.0)

    def test_detection_validation(self) -> None:
        with self.assertRaises(ValueError):
            Detection("chest", (0, 0, 5, 5), 1.5, 0.0)
        with self.assertRaises(ValueError):
            Detection("chest", (0, 0, 0, 5), 0.5, 0.0)
        with self.assertRaises(ValueError):
            make_detector("neural", LIBRARY)

    def test_template_library_scaling_is_cached(self) -> None:
        a = LIBRARY.templates("chest", 8.0)
        self.assertIs(LIBRARY.templates("chest", 8.0), a)
        self.assertLess(a[0].shape[1], LIBRARY.templates("chest", 16.0)[0].shape[1])


if __name__ == "__main__":
    unittest.main()
