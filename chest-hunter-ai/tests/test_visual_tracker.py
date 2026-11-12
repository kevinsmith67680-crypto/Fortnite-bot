"""Tests for chest tracking across frames."""

from __future__ import annotations

import unittest

from chest_hunter.vision.visual_tracker import TrackState, VisualTracker, WorldDetection


def det(x: float, y: float, category: str = "chest", conf: float = 0.9) -> WorldDetection:
    return WorldDetection(category, (x, y), conf, (0, 0, 10, 8))


def always(_x: float, _y: float) -> bool:
    return True


def never(_x: float, _y: float) -> bool:
    return False


class TrackerTests(unittest.TestCase):
    def test_stable_identity_and_confirmation(self) -> None:
        tracker = VisualTracker()
        created = tracker.update([det(10.0, 10.0)], 0.0, always)
        self.assertEqual(len(created), 1)
        track = created[0]
        self.assertIs(track.state, TrackState.TENTATIVE)
        for i in range(1, 6):
            tracker.update([det(10.0 + 0.05 * i, 10.0)], 0.1 * i, always)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertIs(tracker.tracks[0], track)
        self.assertIs(track.state, TrackState.CONFIRMED)
        self.assertEqual(track.observations, 6)
        self.assertAlmostEqual(track.estimated_position[0], 10.2, delta=0.1)

    def test_temporary_misses_then_recovery(self) -> None:
        tracker = VisualTracker(expiration_seconds=3.0)
        for t in (0.0, 0.1):
            tracker.update([det(5.0, 5.0)], t, always)
        track = tracker.tracks[0]
        tracker.update([], 0.5, always)
        self.assertIs(track.state, TrackState.LOST)
        self.assertEqual(track.misses, 1)
        tracker.update([det(5.1, 5.0)], 1.0, always)
        self.assertIs(track.state, TrackState.CONFIRMED)
        self.assertEqual(tracker.tracks[0].track_id, track.track_id)

    def test_expiration_only_while_in_view(self) -> None:
        tracker = VisualTracker(expiration_seconds=2.0, memory_seconds=100.0)
        for t in (0.0, 0.1):
            tracker.update([det(5.0, 5.0)], t, always)
        tracker.update([], 50.0, never)          # out of view: remembered
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([], 50.5, always)         # back in view, unseen for > 2 s
        self.assertEqual(tracker.tracks, [])
        self.assertEqual(tracker.expired_total, 1)

        tracker.update([det(1.0, 1.0)], 0.0, always)
        tracker.update([det(1.0, 1.0)], 0.1, always)
        tracker.update([], 200.0, never)         # beyond memory_seconds
        self.assertEqual(tracker.tracks, [])

    def test_tentative_tracks_expire_quickly(self) -> None:
        tracker = VisualTracker(expiration_seconds=3.0)
        tracker.update([det(3.0, 3.0)], 0.0, always)
        tracker.update([], 1.5, always)
        self.assertEqual(tracker.tracks, [])

    def test_memory_is_bounded(self) -> None:
        tracker = VisualTracker(max_tracks=5)
        for i in range(20):
            tracker.update([det(float(i * 3), 0.0)], float(i), never)
        self.assertEqual(len(tracker.tracks), 5)
        self.assertEqual(tracker.evicted_total, 15)

    def test_no_duplicate_tracks_for_repeated_detections(self) -> None:
        tracker = VisualTracker()
        for t in range(10):
            tracker.update([det(7.0 + (0.1 if t % 2 else -0.1), 7.0)], 0.1 * t, always)
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([det(7.0, 7.0), det(7.3, 7.0)], 1.5, always)
        self.assertEqual(len(tracker.tracks), 2)   # a genuinely separate detection
        tracker.update([det(30.0, 30.0)], 1.6, always)
        self.assertEqual(len(tracker.tracks), 3)

    def test_opening_keeps_the_same_track(self) -> None:
        tracker = VisualTracker()
        for t in (0.0, 0.1):
            tracker.update([det(4.0, 4.0)], t, always)
        tracker.update([det(4.0, 4.0, "chest_opened")], 0.2, always)
        track = tracker.tracks[0]
        self.assertEqual(track.category, "chest_opened")
        self.assertEqual(track.category_history, ["chest"])
        self.assertEqual(len(tracker.tracks), 1)


if __name__ == "__main__":
    unittest.main()
