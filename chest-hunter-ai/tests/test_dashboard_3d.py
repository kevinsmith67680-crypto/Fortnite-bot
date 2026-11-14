"""Debug dashboard: overlays and map are pure functions of the agent's own state; the Tk
window is exercised when a display is available (CI: under Xvfb)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from chest_hunter.agent.dashboard_3d import draw_camera_overlay, draw_map, stats_text
from chest_hunter.agent.runner_3d import Runner3D
from chest_hunter.config import load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.scenarios import seed_arena_3d

ROOT = Path(__file__).resolve().parents[1]


def _runner(on_decision=None) -> Runner3D:  # type: ignore[no-untyped-def]
    db = DatabaseManager(":memory:")
    db.initialize()
    seed_arena_3d(db)
    settings = load_config(ROOT / "config" / "settings.json")
    return Runner3D.from_settings(db, settings, mode="hybrid", seed=2, backend="reference",
                                  max_ticks=450, on_decision=on_decision)


def test_overlays_are_drawn_on_copies() -> None:
    snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    def capture(runner: Runner3D, result: object) -> None:
        c = runner.controller
        if c is None or getattr(c, "obs", None) is None or len(snapshots) > 3:
            return
        before = c.obs.frame.copy()
        overlay = draw_camera_overlay(c)
        assert overlay is not None
        snapshots.append((before, c.obs.frame, overlay, stats_text(c)))
        assert draw_map(c).shape == (300, 300, 3)

    _runner(capture).run()
    assert snapshots
    for before, after, overlay, text in snapshots:
        assert np.array_equal(before, after)              # the agent's frame is untouched
        assert overlay.shape == before.shape and not np.array_equal(overlay, before)
        assert "State:" in text and "Chests opened" in text


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="no display (run under xvfb-run)")
def test_dashboard_window_updates() -> None:
    pytest.importorskip("tkinter", reason="this Python build has no Tk")
    from chest_hunter.agent.dashboard_3d import Dashboard3D

    dashboard = Dashboard3D(refresh_seconds=0.0)
    try:
        _runner(dashboard.update).run()
        assert dashboard.frames_shown > 5
    finally:
        dashboard.close()
