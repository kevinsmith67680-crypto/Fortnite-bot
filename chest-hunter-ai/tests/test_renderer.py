"""GUI smoke tests. Skipped automatically when Tkinter or a display is unavailable.

Run them headlessly with a virtual display, e.g.::

    xvfb-run -a python -m unittest tests.test_renderer -v
"""

from __future__ import annotations

import contextlib
import unittest

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - depends on the Python build
    tk = None  # type: ignore[assignment]

from chest_hunter.agent.runner import SimulationRunner, load_world, make_oracle
from chest_hunter.config import load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.demo_data import seed_demo_chests
from chest_hunter.models import STRATEGY_NAMES, SessionStatus
from chest_hunter.session_repository import SessionRepository


def make_root() -> tk.Tk | None:
    if tk is None:
        return None
    try:
        root = tk.Tk()
    except tk.TclError:
        return None
    root.withdraw()
    return root


class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        root = make_root()
        if root is None:
            self.skipTest("Tkinter or a display is not available")
        self.root = root
        from chest_hunter.simulation.renderer import COLOURS, SimulationApp

        self.colours = COLOURS
        self.db = DatabaseManager(":memory:")
        self.db.initialize()
        seed_demo_chests(self.db)
        self.settings = load_config()
        world = load_world(self.db, self.settings, map_id="research_arena_01", map_seed=42)
        oracle = make_oracle(self.settings)
        self.finished: list[SimulationRunner] = []

        def factory(strategy: str) -> SimulationRunner:
            return SimulationRunner(self.db, self.settings, strategy=strategy, seed=42,
                                    world=world, oracle=oracle)

        self.app = SimulationApp(self.root, factory, strategies=STRATEGY_NAMES,
                                 strategy="nearest", cell_size=4, tick_rate=30,
                                 render_rate=15, speed=10, autostart=False,
                                 on_finish=self.finished.append)

    def tearDown(self) -> None:
        if hasattr(self, "app"):
            with contextlib.suppress(tk.TclError):  # already destroyed
                self.app.close()
            self.db.close()

    def test_full_episode_runs_and_renders(self) -> None:
        app = self.app
        self.assertFalse(app.running)
        self.assertEqual(app.info_vars["simulation"].get(), "ready (press Start)")
        app.start()
        self.assertTrue(app.running)
        for _ in range(5000):
            app.advance_frame()
            if self.finished:
                break
        app.render()
        self.assertEqual(len(self.finished), 1)
        runner = self.finished[0]
        self.assertEqual(runner.metrics.outcome, "completed")
        self.assertEqual(app.info_vars["agent_state"].get(), "COMPLETED")
        self.assertEqual(app.info_vars["opened"].get(), str(runner.metrics.chests_opened))
        opened_fill = {app.canvas.itemcget(item, "fill")
                       for cid, item in app._chest_items.items()
                       if runner.env.chests[cid].opened}
        self.assertEqual(opened_fill, {self.colours["chest_opened"]})
        self.assertGreater(len(app.canvas.find_withtag("obstacle")), 50)

    def test_pause_resume_reset_and_speed(self) -> None:
        app = self.app
        app.start()
        app.advance_frame()
        tick = app.runner.env.tick  # type: ignore[union-attr]
        self.assertEqual(tick, 20)  # 10x speed: 10 * 30 / 15 ticks per frame
        app.pause()
        for _ in range(5):
            app.advance_frame()
        self.assertEqual(app.runner.env.tick, tick)  # type: ignore[union-attr]
        app.resume()
        app.speed_var.set("1x")
        app.advance_frame()
        self.assertEqual(app.runner.env.tick, tick + 2)  # type: ignore[union-attr]
        app.show_paths_var.set(False)
        app.render()
        self.assertEqual(app.canvas.itemcget(app._path_item, "state"), "hidden")

        old_session = app.runner.session_id  # type: ignore[union-attr]
        app.strategy_var.set("weighted")
        app.reset()
        assert old_session is not None
        status = SessionRepository(self.db).require_session(old_session).status
        self.assertIs(status, SessionStatus.ABORTED)
        self.assertEqual(app.runner.strategy, "weighted")  # type: ignore[union-attr]
        self.assertFalse(app.running)


if __name__ == "__main__":
    unittest.main()
