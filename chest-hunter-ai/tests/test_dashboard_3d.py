"""Debug dashboard: overlays and map are pure functions of the agent's own state; the Tk
window is exercised when a display is available (CI: under Xvfb)."""

from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path

import numpy as np

from chest_hunter.agent.dashboard_3d import draw_camera_overlay, draw_map, stats_text
from chest_hunter.agent.runner_3d import Runner3D
from chest_hunter.config import load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.scenarios import seed_arena_3d
from tests.tk_support import collect_on_main_thread, collect_tk_garbage_after

ROOT = Path(__file__).resolve().parents[1]


def _runner(on_decision=None) -> Runner3D:  # type: ignore[no-untyped-def]
    db = DatabaseManager(":memory:")
    db.initialize()
    seed_arena_3d(db)
    settings = load_config(ROOT / "config" / "settings.json")
    return Runner3D.from_settings(db, settings, mode="hybrid", seed=2, backend="reference",
                                  max_ticks=450, on_decision=on_decision)


class TestDashboard3D(unittest.TestCase):
    def test_overlays_are_drawn_on_copies(self) -> None:
        snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

        def capture(runner: Runner3D, result: object) -> None:
            c = runner.controller
            if c is None or getattr(c, "obs", None) is None or len(snapshots) > 3:
                return
            before = c.obs.frame.copy()
            overlay = draw_camera_overlay(c)
            self.assertIsNotNone(overlay)
            snapshots.append((before, c.obs.frame, overlay, stats_text(c)))  # type: ignore[arg-type]
            self.assertEqual(draw_map(c).shape, (300, 300, 3))

        runner = _runner(capture)
        self.addCleanup(runner.db.close)
        runner.run()
        self.assertTrue(snapshots)
        for before, after, overlay, text in snapshots:
            self.assertTrue(np.array_equal(before, after))   # the agent's frame is untouched
            self.assertEqual(overlay.shape, before.shape)
            self.assertFalse(np.array_equal(overlay, before))
            self.assertIn("State:", text)
            self.assertIn("Chests opened", text)

    @unittest.skipUnless(os.environ.get("DISPLAY"), "no display (run under xvfb-run)")
    def test_dashboard_window_updates(self) -> None:
        try:
            importlib.import_module("tkinter")
        except ImportError:
            self.skipTest("this Python build has no Tk")
        collect_tk_garbage_after(self)  # the episode runs the mock server's threads
        from chest_hunter.agent.dashboard_3d import Dashboard3D

        dashboard = Dashboard3D(refresh_seconds=0.0)
        try:
            runner = _runner(dashboard.update)
            self.addCleanup(runner.db.close)
            runner.run()
            self.assertGreater(dashboard.frames_shown, 5)
        finally:
            dashboard.close()


def tearDownModule() -> None:
    collect_on_main_thread()  # this module's Tk objects are freed here, on the main thread
