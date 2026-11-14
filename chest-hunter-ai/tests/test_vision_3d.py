"""3D perception on frames rendered by the reference arena: chest detection (closed, opened,
partially occluded, none, decoys), distance estimation, tracking and interaction cues.

Ground truth comes from the reference renderer's label buffer (evaluation labels); the
detector itself only ever sees the BGR frame.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from chest_hunter.simulation3d.arena import chest_boxes
from chest_hunter.simulation3d.environment3d import Environment3D, EnvSettings
from chest_hunter.vision.chest_detector_3d import ChestDetector3D, Detection3D, TemplateSet3D
from chest_hunter.vision.depth_estimation import DistanceEstimator, RangeEstimate
from chest_hunter.vision.interaction_cues_3d import detect_cues
from chest_hunter.vision.perspective import CameraPose, bearing_deg
from chest_hunter.vision.tracker_3d import Tracker3D, TrackState3D

MAP = "research_arena_3d"
TEMPLATES = Path(__file__).resolve().parents[1] / "assets" / "templates3d"
FOUNTAIN = 6                                   # Central Plaza - Fountain Square (open ground)


def _env(seed: int = 0) -> Environment3D:
    return Environment3D(MAP, seed=seed, settings=EnvSettings(width=640, height=360,
                                                              odometry_drift=0.0))


def _force_chest(env: Environment3D, index: int, *, present: bool = True,
                 opened: bool = False) -> None:
    c = env.chests[index]
    c.present, c.opened = present, opened
    c.boxes = tuple(chest_boxes(c.x, c.y, c.z, c.yaw_deg, opened=opened)) if present else ()


def _stand(env: Environment3D, index: int, distance: float, side_deg: float = 0.0,
           pitch: float = -12.0) -> None:
    """Put the agent ``distance`` m in front of chest ``index`` and look at it."""
    c = env.chests[index]
    a = math.radians(c.yaw_deg + side_deg)
    ch = env.character
    ch.x, ch.y, ch.z = c.x + distance * math.cos(a), c.y + distance * math.sin(a), c.z
    yaw = bearing_deg(c.x - ch.x, c.y - ch.y)
    ch.cam_yaw = ch.body_yaw = yaw
    ch.cam_pitch = pitch


def _observe(env: Environment3D) -> tuple[np.ndarray, dict, dict]:
    frame, meta = env.observe(labels=True)
    labels = env.evaluation_labels()
    if labels is None:
        raise AssertionError("the reference renderer returned no evaluation labels")
    return frame, meta, labels


def _pose(env: Environment3D) -> CameraPose:
    pose, _ = env.character.camera(env.world)
    return pose


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter)


def _detect(detector: ChestDetector3D, env: Environment3D, frame: np.ndarray) -> list[Detection3D]:
    return detector.detect(frame, env.time, camera_pitch_deg=env.character.cam_pitch,
                           fy=env.intrinsics.fy)


class TestChestDetector3D(unittest.TestCase):
    """The templates are loaded once for the class; each test gets a fresh hybrid detector."""

    templates: TemplateSet3D

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.templates = TemplateSet3D.load(TEMPLATES)

    def setUp(self) -> None:
        self.detector = ChestDetector3D(self.templates, method="hybrid")

    def test_templates_cover_views_and_states(self) -> None:
        templates = self.templates
        names = [t.name for t in templates.templates]
        self.assertEqual(len(names), 72)
        self.assertEqual(sum(t.category == "chest_opened" for t in templates.templates), 36)
        elevations = {t.elevation_deg for t in templates.templates}
        self.assertEqual(len(elevations), 6)
        self.assertLess(min(elevations), 0)
        self.assertEqual(max(elevations), 70.0)

    def test_unknown_detection_method(self) -> None:
        with self.assertRaises(ValueError):
            ChestDetector3D(self.templates, method="neural")

    def test_closed_chest_is_detected(self) -> None:
        for distance in [3.0, 6.0, 10.0]:
            with self.subTest(distance=distance):
                env = _env()
                _force_chest(env, FOUNTAIN)
                _stand(env, FOUNTAIN, distance)
                frame, _, labels = _observe(env)
                truth = [c for c in labels["chests"] if c["spawn_index"] == FOUNTAIN]
                self.assertTrue(truth, "the chest should be visible in the rendered frame")
                dets = _detect(self.detector, env, frame)
                matches = [d for d in dets if _iou(d.bbox, tuple(truth[0]["bbox"])) > 0.4]
                self.assertTrue(matches)
                self.assertEqual(matches[0].category, "chest")

    def test_opened_chest_is_classified_as_opened(self) -> None:
        env = _env()
        _force_chest(env, FOUNTAIN, opened=True)
        _stand(env, FOUNTAIN, 4.0, pitch=-20.0)
        frame, _, labels = _observe(env)
        truth = next(c for c in labels["chests"] if c["spawn_index"] == FOUNTAIN)
        self.assertEqual(truth["category"], "chest_opened")
        dets = [d for d in _detect(self.detector, env, frame)
                if _iou(d.bbox, tuple(truth["bbox"])) > 0.3]
        self.assertTrue(dets)
        self.assertEqual(dets[0].category, "chest_opened")

    def test_side_views_are_detected(self) -> None:
        env = _env()
        _force_chest(env, FOUNTAIN)
        for side in (60.0, 90.0, -120.0):
            _stand(env, FOUNTAIN, 5.0, side_deg=side)
            frame, _, labels = _observe(env)
            truth = next(c for c in labels["chests"] if c["spawn_index"] == FOUNTAIN)
            dets = _detect(self.detector, env, frame)
            self.assertTrue(any(_iou(d.bbox, tuple(truth["bbox"])) > 0.3 for d in dets), side)

    def test_no_chest_no_detection(self) -> None:
        env = _env()
        _force_chest(env, FOUNTAIN, present=False)
        _stand(env, FOUNTAIN, 5.0)
        frame, _, labels = _observe(env)
        self.assertFalse([c for c in labels["chests"] if c["spawn_index"] == FOUNTAIN])
        near_centre = [d for d in _detect(self.detector, env, frame)
                       if abs(d.centre[0] - 320) < 120 and abs(d.centre[1] - 180) < 120]
        self.assertEqual(near_centre, [])

    def test_gold_decoys_are_not_chests(self) -> None:
        env = _env()
        index = next(i for i, s in enumerate(env.arena.spawns) if "decoys" in s.tags)
        _force_chest(env, index, present=False)
        _stand(env, index, 4.0, pitch=-15.0)
        frame, _, labels = _observe(env)
        self.assertTrue(not labels["chests"]
                        or all(c["spawn_index"] != index for c in labels["chests"]))
        # Gold sacks are in view (same gold as the chests) but have no lid/latch structure.
        gold = np.all(np.abs(frame.astype(int) - np.array([24, 197, 245])) < 40, axis=2).sum()
        self.assertGreater(gold, 30)
        labelled = {tuple(c["bbox"]) for c in labels["chests"]}
        false = [d for d in _detect(self.detector, env, frame)
                 if not any(_iou(d.bbox, b) > 0.2 for b in labelled)]
        self.assertEqual(false, [])

    def test_partially_occluded_chest_is_still_found(self) -> None:
        env = _env()
        index = next(i for i, s in enumerate(env.arena.spawns) if s.name == "Pillar")
        _force_chest(env, index)
        _stand(env, index, 5.0, side_deg=35.0)
        frame, _, labels = _observe(env)
        truth = [c for c in labels["chests"] if c["spawn_index"] == index]
        if not truth:
            self.skipTest("chest fully hidden from this viewpoint")
        self.assertTrue(any(_iou(d.bbox, tuple(truth[0]["bbox"])) > 0.2
                            for d in _detect(self.detector, env, frame)))

    def test_distance_estimate_is_within_15_percent(self) -> None:
        for distance in [3.0, 6.0, 9.0, 12.0]:
            with self.subTest(distance=distance):
                env = _env()
                _force_chest(env, FOUNTAIN)
                _stand(env, FOUNTAIN, distance)
                frame, _, labels = _observe(env)
                truth = next(c for c in labels["chests"] if c["spawn_index"] == FOUNTAIN)
                det = max((d for d in _detect(self.detector, env, frame)
                           if _iou(d.bbox, tuple(truth["bbox"])) > 0.3),
                          key=lambda d: d.confidence)
                est = DistanceEstimator().estimate(det, _pose(env), env.intrinsics, (0.0, 3.5),
                                                   own_floor=0.0)
                if est is None:
                    self.fail("no distance estimate for the detected chest")
                self.assertLess(abs(est.distance - truth["distance"]) / truth["distance"], 0.15)
                c = env.chests[FOUNTAIN]
                self.assertLess(math.dist(est.position[:2], (c.x, c.y)),
                                max(0.6, 0.12 * distance))
                self.assertAlmostEqual(est.position[2], 0.0, delta=0.05)
                self.assertGreater(est.sigma, 0)
                self.assertIn(est.method, {"size", "ground", "fused"})

    def test_interaction_prompt_and_progress_are_seen(self) -> None:
        env = _env()
        _force_chest(env, FOUNTAIN)
        _stand(env, FOUNTAIN, 1.3, pitch=-25.0)
        self.assertIs(env.interactable_chest(), env.chests[FOUNTAIN])
        frame, _, labels = _observe(env)
        truth = next(c for c in labels["chests"] if c["spawn_index"] == FOUNTAIN)
        cues = detect_cues(frame, tuple(truth["bbox"]))
        self.assertTrue(cues.prompt)
        env.step({"move_forward": 0.0, "move_right": 0.0, "look_yaw": 0.0, "look_pitch": 0.0,
                  "jump": False, "interact": True}, 20)
        frame, _, labels = _observe(env)
        truth = next(c for c in labels["chests"] if c["spawn_index"] == FOUNTAIN)
        self.assertTrue(detect_cues(frame, tuple(truth["bbox"])).progress)
        # Far away: no prompt.
        _stand(env, FOUNTAIN, 8.0)
        frame, _, _ = _observe(env)
        self.assertFalse(detect_cues(frame, None).prompt)


class TestDistanceEstimator(unittest.TestCase):
    def test_impossible_estimates_are_rejected(self) -> None:
        env = _env()
        est = DistanceEstimator(max_range=40.0)
        pose = CameraPose(30.0, 50.0, 1.6, 0.0, 10.0)        # looking up: no ground intersection
        tiny = Detection3D("chest", (318, 20, 2, 2), 0.9, 0.0, "hybrid")
        self.assertIsNone(est.estimate(tiny, pose, env.intrinsics))
        self.assertGreaterEqual(est.rejected, 1)


def _measure(x: float, y: float, sigma: float = 0.3) -> tuple[Detection3D, RangeEstimate]:
    det = Detection3D("chest", (300, 150, 40, 30), 0.9, 0.0, "hybrid")
    return det, RangeEstimate((x, y, 0.0), 6.0, sigma, "fused", 0.0, 10.0,
                              ray_origin=(x - 6.0, y, 1.6), ray_direction=(1.0, 0.0, -0.25))


class TestTracker3D(unittest.TestCase):
    def test_tracker_confirms_fuses_and_separates(self) -> None:
        tracker = Tracker3D(confirm_hits=2)
        created = tracker.update([_measure(10.0, 5.0)], 0.0, lambda t: True)
        self.assertEqual(len(created), 1)
        self.assertIs(created[0].state, TrackState3D.TENTATIVE)
        tracker.update([_measure(10.2, 5.1)], 0.1, lambda t: True)
        track = tracker.tracks[0]
        self.assertIs(track.state, TrackState3D.CONFIRMED)
        self.assertEqual(track.hits, 2)
        self.assertLess(math.dist(track.xyz[:2], (10.1, 5.05)), 0.2)
        tracker.update([_measure(10.1, 5.0), _measure(20.0, 5.0)], 0.2, lambda t: True)
        self.assertEqual(len(tracker.tracks), 2)              # far measurement -> separate chest

    def test_tracker_loses_unseen_tracks_and_forgets_tentative_ones(self) -> None:
        tracker = Tracker3D(confirm_hits=2, lost_after=1.0, tentative_timeout=0.5)
        tracker.update([_measure(10.0, 5.0)], 0.0, lambda t: True)
        tracker.update([_measure(10.0, 5.0)], 0.1, lambda t: True)
        tracker.update([_measure(30.0, 5.0)], 0.2, lambda t: True)          # tentative
        for k in range(3, 30):
            tracker.update([], 0.1 * k, lambda t: False)                    # out of view
        states = {t.state for t in tracker.tracks}
        self.assertIn(TrackState3D.LOST, states)
        self.assertEqual(len(tracker.tracks), 1)
