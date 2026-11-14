"""Closed-loop 3D agent on the reference arena through the real loopback bridge.

Covers: full episodes in every mode, database records (no duplicates, openings match ground
truth, rewards only with openings), determinism, the information boundary (static audit of
the visual agent's code and a run against a server that refuses every privileged command)
and the controller's state machine.
"""

from __future__ import annotations

import ast
import inspect
from collections import Counter
from pathlib import Path

import pytest

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


@pytest.fixture(scope="module")
def settings() -> Settings:
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


@pytest.fixture(scope="module")
def hybrid_run(settings: Settings) -> tuple[DatabaseManager, Result3D]:
    db = _db()
    return db, _run(settings, db, "hybrid", seconds=90.0, evaluate=True)


class TestEpisodes:
    def test_hybrid_episode_opens_real_chests(self, hybrid_run: Run) -> None:
        _, result = hybrid_run
        assert result.outcome == "tick_limit" and result.sim_seconds == pytest.approx(90.0)
        assert result.chests_opened >= 2
        assert result.truth["opened_truth"] >= result.chests_opened
        assert result.vision is not None and result.vision["frames"] > 0

    def test_database_records_match_ground_truth(self, hybrid_run: Run) -> None:
        db, result = hybrid_run
        rows = db.connection.execute(
            "SELECT chest_location_id, chest_present, chest_opened, reward_received "
            "FROM chest_observations WHERE session_id = ?", (result.session_id,)).fetchall()
        opened = [r[0] for r in rows if r[2]]
        assert opened, "openings must be recorded"
        # Every recorded opening really happened, and no location was opened twice.
        assert set(opened) <= set(result.truth["location_ids_opened"])
        assert max(Counter(opened).values()) == 1
        # Rewards only on openings; presence recorded for every opening.
        assert all(r[1] for r in rows if r[2])
        assert all(r[3] is None or r[2] for r in rows)

    def test_session_and_run_rows(self, hybrid_run: tuple[DatabaseManager, Result3D]) -> None:
        db, result = hybrid_run
        status = db.connection.execute("SELECT status FROM simulation_sessions WHERE id = ?",
                                       (result.session_id,)).fetchone()[0]
        assert status == "completed"
        run = db.connection.execute(
            "SELECT strategy, outcome, simulated_seconds, navigation_failures, parameters_json "
            "FROM simulation_runs WHERE session_id = ?", (result.session_id,)).fetchone()
        assert run[0] == STRATEGY_3D["hybrid"] and run[1] == "tick_limit"
        assert run[2] == pytest.approx(90.0) and run[3] >= 0
        assert '"environment": "3d"' in run[4]

    def test_oracle_baseline_is_an_upper_bound(self, settings: Settings,
                                               hybrid_run: Run) -> None:
        result = _run(settings, _db(), "oracle", seconds=90.0)
        assert result.outcome == "tick_limit"
        assert result.chests_opened >= hybrid_run[1].chests_opened - 1
        assert result.chests_opened == result.truth["opened_truth"]

    def test_visual_navigation_mode_runs_without_the_navmesh_service(self,
                                                                     settings: Settings) -> None:
        # The server refuses NAV_QUERY in this mode (navigation="local"); the episode must
        # still run to its tick limit using local probes only.
        db = _db()
        result = _run(settings, db, "visual_nav", seconds=45.0)
        assert result.outcome == "tick_limit"
        assert result.chests_opened >= 1 and result.metrics.navigation_failures >= 0

    def test_episodes_are_deterministic(self, settings: Settings) -> None:
        a = _run(settings, _db(), "hybrid", seed=11, seconds=40.0)
        b = _run(settings, _db(), "hybrid", seed=11, seconds=40.0)
        assert (a.chests_opened, a.ticks, a.metrics.travel_distance) == \
            (b.chests_opened, b.ticks, b.metrics.travel_distance)

    def test_unknown_mode_and_backend(self, settings: Settings) -> None:
        with pytest.raises(ValidationError):
            Runner3D(_db(), settings, mode="teleport")
        with pytest.raises(ValidationError):
            Runner3D(_db(), settings, backend="fortnite")
        assert set(MODES_3D) == {"oracle", "hybrid", "visual_nav"}

    def test_unreal_backend_without_a_server_fails_cleanly(self, settings: Settings) -> None:
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        db = _db()
        runner = Runner3D(db, settings, mode="hybrid", backend="unreal", port=port)
        runner.settings = settings
        with pytest.raises(Exception) as exc:
            runner.run()
        assert "research server" in str(exc.value)
        # No session is left behind when the connection could not be made.
        assert db.connection.execute("SELECT COUNT(*) FROM simulation_sessions").fetchone()[0] == 0

    def test_map_without_spawn_points(self, settings: Settings) -> None:
        db = DatabaseManager(":memory:")
        db.initialize()
        with pytest.raises(ValidationError, match="seed-demo"):
            Runner3D(db, settings, mode="hybrid").run()


class TestInformationBoundary:
    FORBIDDEN_NAMES = {"oracle_state", "evaluation_labels", "GET_ORACLE_STATE", "GET_EVALUATION",
                       "Environment3D", "MockUnrealServer", "BridgeSession", "cached_arena",
                       "BridgeClient"}
    FORBIDDEN_MODULES = ("chest_hunter.simulation3d", "chest_hunter.unreal_bridge.mock_server")

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

    @pytest.mark.parametrize("module", [world_model_module, navigation_module, chest_detector_3d,
                                        depth_estimation, tracker_3d, interaction_cues_3d])
    def test_perception_and_navigation_modules_cannot_reach_ground_truth(
            self, module: object) -> None:
        tree = ast.parse(inspect.getsource(module))          # type: ignore[arg-type]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(self.FORBIDDEN_MODULES), node.module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(self.FORBIDDEN_MODULES), alias.name
        assert not self._names(tree) & self.FORBIDDEN_NAMES

    def test_visual_agent_class_has_no_privileged_access(self) -> None:
        tree = ast.parse(inspect.getsource(controller_module))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                assert not mod.startswith(self.FORBIDDEN_MODULES)
        visual = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                      and n.name == "VisualAgent3D")
        assert not self._names(visual) & self.FORBIDDEN_NAMES

    def test_visual_agent_only_receives_the_agent_port(self) -> None:
        params = list(inspect.signature(VisualAgent3D.__init__).parameters)
        assert params[1] == "port"
        assert not {"client", "env", "oracle", "server"} & set(params)

    def test_hybrid_run_against_a_server_without_privileged_modes(self, settings: Settings) -> None:
        """With evaluate=False the reference server starts with oracle=False and
        evaluation=False: any privileged request would fail the episode with 'forbidden'."""
        db = _db()
        runner = Runner3D.from_settings(db, settings, mode="hybrid", seed=3, backend="reference",
                                        max_ticks=900, evaluate=False)
        result = runner.run()
        assert result.outcome == "tick_limit" and result.truth.get("present") is None
        assert runner.server is not None
        assert runner.server.session.oracle_enabled is False
        assert runner.server.session.evaluation_enabled is False


class TestStateMachine:
    def test_every_state_is_reachable_from_initializing(self) -> None:
        seen = {Agent3DState.INITIALIZING}
        frontier = [Agent3DState.INITIALIZING]
        while frontier:
            s = frontier.pop()
            for nxt in TRANSITIONS_3D.get(s, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        assert seen == set(Agent3DState)

    def test_terminal_states_have_no_exits(self) -> None:
        for terminal in (Agent3DState.COMPLETED, Agent3DState.FAILED):
            assert not TRANSITIONS_3D.get(terminal)

    def test_interaction_requires_alignment_and_verification(self) -> None:
        assert Agent3DState.INTERACTING in TRANSITIONS_3D[Agent3DState.ALIGNING]
        assert Agent3DState.INTERACTING not in TRANSITIONS_3D[Agent3DState.NAVIGATING]
        assert Agent3DState.VERIFYING_OPEN in TRANSITIONS_3D[Agent3DState.INTERACTING]


def test_seeded_arena_matches_the_layout() -> None:
    db = _db()
    count = db.connection.execute("SELECT COUNT(*) FROM chest_locations WHERE map_id = ?",
                                  (ARENA_3D_ID,)).fetchone()[0]
    assert count == 35
    again = seed_arena_3d(db)
    assert again.inserted == 0                                   # idempotent
