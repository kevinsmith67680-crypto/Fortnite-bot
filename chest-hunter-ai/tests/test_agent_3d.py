"""Closed-loop 3D agent on the reference arena through the real loopback bridge.

Covers: full episodes in every mode, database records (no duplicates, openings match ground
truth, rewards only with openings), determinism, the information boundary (static audit of
the visual agent's code and a run against a server that refuses every privileged command)
and the controller's state machine.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from collections import Counter
from pathlib import Path

import chest_hunter.agent.controller_3d as controller_module
import chest_hunter.localization.world_model_3d as world_model_module
import chest_hunter.navigation.navigation_3d as navigation_module
from chest_hunter.agent.controller_3d import TRANSITIONS_3D, Agent3DState, VisualAgent3D
from chest_hunter.agent.runner_3d import MODES_3D, STRATEGY_3D, Result3D, Runner3D
from chest_hunter.config import Settings, load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ValidationError
from chest_hunter.scenarios import ARENA_3D_ID, seed_arena_3d
from chest_hunter.vision import (
    chest_detector_3d,
    depth_estimation,
    interaction_cues_3d,
    tracker_3d,
)

ROOT = Path(__file__).resolve().parents[1]
Run = tuple[DatabaseManager, Result3D]


def _settings() -> Settings:
    return load_config(ROOT / "config" / "settings.json")


def _db() -> DatabaseManager:
    db = DatabaseManager(":memory:")
    db.initialize()
    seed_arena_3d(db)
    return db


def _run(settings: Settings, db: DatabaseManager, mode: str, *, seed: int = 5,
         seconds: float = 60.0, evaluate: bool = False) -> Result3D:
    return Runner3D.from_settings(db, settings, mode=mode, seed=seed, backend="reference",
                                  max_ticks=int(seconds * 30), evaluate=evaluate).run()


def _hybrid_run(settings: Settings) -> Run:
    """The 90-second hybrid episode several tests inspect (computed once per test class)."""
    db = _db()
    return db, _run(settings, db, "hybrid", seconds=90.0, evaluate=True)


class DatabaseTestCase(unittest.TestCase):
    """Closes the in-memory databases a test opens when the test ends."""

    def new_db(self) -> DatabaseManager:
        """A database seeded with the 3D arena (``_db()``), closed after the test."""
        db = _db()
        self.addCleanup(db.close)
        return db


class TestEpisodes(DatabaseTestCase):
    settings: Settings
    hybrid_run: Run

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.settings = _settings()
        cls.hybrid_run = _hybrid_run(cls.settings)
        cls.addClassCleanup(cls.hybrid_run[0].close)

    def test_hybrid_episode_opens_real_chests(self) -> None:
        _, result = self.hybrid_run
        self.assertEqual(result.outcome, "tick_limit")
        # 1e-6 relative tolerance.
        self.assertAlmostEqual(result.sim_seconds, 90.0, delta=1e-6 * 90.0)
        self.assertGreaterEqual(result.chests_opened, 2)
        self.assertGreaterEqual(result.truth["opened_truth"], result.chests_opened)
        self.assertIsNotNone(result.vision)
        self.assertGreater(result.vision["frames"], 0)  # type: ignore[index]

    def test_database_records_match_ground_truth(self) -> None:
        db, result = self.hybrid_run
        rows = db.connection.execute(
            "SELECT chest_location_id, chest_present, chest_opened, reward_received "
            "FROM chest_observations WHERE session_id = ?", (result.session_id,)).fetchall()
        opened = [r[0] for r in rows if r[2]]
        self.assertTrue(opened, "openings must be recorded")
        # Every recorded opening really happened, and no location was opened twice.
        self.assertLessEqual(set(opened), set(result.truth["location_ids_opened"]))
        self.assertEqual(max(Counter(opened).values()), 1)
        # Rewards only on openings; presence recorded for every opening.
        self.assertTrue(all(r[1] for r in rows if r[2]))
        self.assertTrue(all(r[3] is None or r[2] for r in rows))

    def test_session_and_run_rows(self) -> None:
        db, result = self.hybrid_run
        status = db.connection.execute("SELECT status FROM simulation_sessions WHERE id = ?",
                                       (result.session_id,)).fetchone()[0]
        self.assertEqual(status, "completed")
        run = db.connection.execute(
            "SELECT strategy, outcome, simulated_seconds, navigation_failures, parameters_json "
            "FROM simulation_runs WHERE session_id = ?", (result.session_id,)).fetchone()
        self.assertEqual(run[0], STRATEGY_3D["hybrid"])
        self.assertEqual(run[1], "tick_limit")
        self.assertAlmostEqual(run[2], 90.0, delta=1e-6 * 90.0)    # 1e-6 relative tolerance
        self.assertGreaterEqual(run[3], 0)
        self.assertIn('"environment": "3d"', run[4])

    def test_oracle_baseline_is_an_upper_bound(self) -> None:
        result = _run(self.settings, self.new_db(), "oracle", seconds=90.0)
        self.assertEqual(result.outcome, "tick_limit")
        self.assertGreaterEqual(result.chests_opened, self.hybrid_run[1].chests_opened - 1)
        self.assertEqual(result.chests_opened, result.truth["opened_truth"])

    def test_visual_navigation_mode_runs_without_the_navmesh_service(self) -> None:
        # The server refuses NAV_QUERY in this mode (navigation="local"); the episode must
        # still run to its tick limit using local probes only.
        db = self.new_db()
        result = _run(self.settings, db, "visual_nav", seconds=45.0)
        self.assertEqual(result.outcome, "tick_limit")
        self.assertGreaterEqual(result.chests_opened, 1)
        self.assertGreaterEqual(result.metrics.navigation_failures, 0)

    def test_episodes_are_deterministic(self) -> None:
        a = _run(self.settings, self.new_db(), "hybrid", seed=11, seconds=40.0)
        b = _run(self.settings, self.new_db(), "hybrid", seed=11, seconds=40.0)
        self.assertEqual((a.chests_opened, a.ticks, a.metrics.travel_distance),
                         (b.chests_opened, b.ticks, b.metrics.travel_distance))

    def test_unknown_mode_and_backend(self) -> None:
        with self.assertRaises(ValidationError):
            Runner3D(self.new_db(), self.settings, mode="teleport")
        with self.assertRaises(ValidationError):
            Runner3D(self.new_db(), self.settings, backend="fortnite")
        self.assertEqual(set(MODES_3D), {"oracle", "hybrid", "visual_nav"})

    def test_unreal_backend_without_a_server_fails_cleanly(self) -> None:
        import socket
        settings = self.settings
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        db = self.new_db()
        runner = Runner3D(db, settings, mode="hybrid", backend="unreal", port=port)
        runner.settings = settings
        with self.assertRaises(Exception) as cm:
            runner.run()
        self.assertIn("research server", str(cm.exception))
        # No session is left behind when the connection could not be made.
        self.assertEqual(
            db.connection.execute("SELECT COUNT(*) FROM simulation_sessions").fetchone()[0], 0)

    def test_map_without_spawn_points(self) -> None:
        db = DatabaseManager(":memory:")
        self.addCleanup(db.close)
        db.initialize()
        with self.assertRaisesRegex(ValidationError, "seed-demo"):
            Runner3D(db, self.settings, mode="hybrid").run()


class TestInformationBoundary(DatabaseTestCase):
    FORBIDDEN_NAMES = {"oracle_state", "evaluation_labels", "GET_ORACLE_STATE", "GET_EVALUATION",
                       "Environment3D", "MockUnrealServer", "BridgeSession", "cached_arena",
                       "BridgeClient"}
    FORBIDDEN_MODULES = ("chest_hunter.simulation3d", "chest_hunter.unreal_bridge.mock_server")

    settings: Settings

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.settings = _settings()

    @staticmethod
    def _names(tree: ast.AST) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                out.add(node.id)
            elif isinstance(node, ast.Attribute):
                out.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.add(node.value)
        return out

    def test_perception_and_navigation_modules_cannot_reach_ground_truth(self) -> None:
        for module in [world_model_module, navigation_module, chest_detector_3d,
                       depth_estimation, tracker_3d, interaction_cues_3d]:
            with self.subTest(module=module.__name__):
                tree = ast.parse(inspect.getsource(module))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        self.assertFalse(node.module.startswith(self.FORBIDDEN_MODULES),
                                         node.module)
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertFalse(alias.name.startswith(self.FORBIDDEN_MODULES),
                                             alias.name)
                self.assertFalse(self._names(tree) & self.FORBIDDEN_NAMES)

    def test_visual_agent_class_has_no_privileged_access(self) -> None:
        tree = ast.parse(inspect.getsource(controller_module))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                self.assertFalse(mod.startswith(self.FORBIDDEN_MODULES))
        visual = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "VisualAgent3D")
        self.assertFalse(self._names(visual) & self.FORBIDDEN_NAMES)

    def test_visual_agent_only_receives_the_agent_port(self) -> None:
        params = list(inspect.signature(VisualAgent3D.__init__).parameters)
        self.assertEqual(params[1], "port")
        self.assertFalse({"client", "env", "oracle", "server"} & set(params))

    def test_hybrid_run_against_a_server_without_privileged_modes(self) -> None:
        """With evaluate=False the reference server starts with oracle=False and
        evaluation=False: any privileged request would fail the episode with 'forbidden'."""
        db = self.new_db()
        runner = Runner3D.from_settings(db, self.settings, mode="hybrid", seed=3,
                                        backend="reference", max_ticks=900, evaluate=False)
        result = runner.run()
        self.assertEqual(result.outcome, "tick_limit")
        self.assertIsNone(result.truth.get("present"))
        self.assertIsNotNone(runner.server)
        self.assertIs(runner.server.session.oracle_enabled, False)
        self.assertIs(runner.server.session.evaluation_enabled, False)


class TestStateMachine(unittest.TestCase):
    def test_every_state_is_reachable_from_initializing(self) -> None:
        seen = {Agent3DState.INITIALIZING}
        frontier = [Agent3DState.INITIALIZING]
        while frontier:
            s = frontier.pop()
            for nxt in TRANSITIONS_3D.get(s, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        self.assertEqual(seen, set(Agent3DState))

    def test_terminal_states_have_no_exits(self) -> None:
        for terminal in (Agent3DState.COMPLETED, Agent3DState.FAILED):
            self.assertFalse(TRANSITIONS_3D.get(terminal))

    def test_interaction_requires_alignment_and_verification(self) -> None:
        self.assertIn(Agent3DState.INTERACTING, TRANSITIONS_3D[Agent3DState.ALIGNING])
        self.assertNotIn(Agent3DState.INTERACTING, TRANSITIONS_3D[Agent3DState.NAVIGATING])
        self.assertIn(Agent3DState.VERIFYING_OPEN, TRANSITIONS_3D[Agent3DState.INTERACTING])


class TestScenario(DatabaseTestCase):
    def test_seeded_arena_matches_the_layout(self) -> None:
        db = self.new_db()
        count = db.connection.execute("SELECT COUNT(*) FROM chest_locations WHERE map_id = ?",
                                      (ARENA_3D_ID,)).fetchone()[0]
        self.assertEqual(count, 35)
        again = seed_arena_3d(db)
        self.assertEqual(again.inserted, 0)                          # idempotent
