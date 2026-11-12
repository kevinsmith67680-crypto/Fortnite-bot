"""Tests for the autonomous agent FSM, the runner and its SQLite integration."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chest_hunter.agent.runner import SimulationRunner, load_world
from chest_hunter.agent.states import TRANSITIONS, AgentState, is_valid_transition
from chest_hunter.chest_repository import ChestRepository
from chest_hunter.config import (
    AgentSettings,
    DatabaseSettings,
    LoggingSettings,
    NavigationSettings,
    Settings,
    SimulationSettings,
)
from chest_hunter.database import DatabaseManager
from chest_hunter.demo_data import DEMO_CHESTS, seed_demo_chests
from chest_hunter.exceptions import ChestHunterError, ValidationError
from chest_hunter.models import SessionStatus
from chest_hunter.run_repository import RunRepository
from chest_hunter.session_repository import SessionRepository
from chest_hunter.simulation.collision import GridMap, circle_collides
from chest_hunter.simulation.entities import Obstacle
from chest_hunter.simulation.environment import ChestSpec, World, WorldTransform

MAP = "test_arena"
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_settings(sim: dict[str, Any] | None = None, nav: dict[str, Any] | None = None,
                  agent: dict[str, Any] | None = None) -> Settings:
    simulation = SimulationSettings(default_map=MAP, default_spawn_probability=0.8,
                                    default_open_seconds=2.0, world_units_per_cell=1.0)
    return Settings(
        database=DatabaseSettings(Path(":memory:")),
        simulation=replace(simulation, **(sim or {})),
        logging=LoggingSettings("WARNING"),
        config_path=Path("settings.json"),
        base_dir=Path("."),
        navigation=replace(NavigationSettings(), **(nav or {})),
        agent=replace(AgentSettings(), **(agent or {})),
    )


class Harness:
    """In-memory database + hand-made world (1 world unit = 1 grid cell)."""

    def __init__(self, chests: list[tuple[float, float, float]], *, size: int = 30,
                 walls: tuple[tuple[int, int, int, int], ...] = (),
                 start: tuple[float, float] = (2.5, 2.5),
                 settings: Settings | None = None) -> None:
        self.settings = settings or make_settings()
        self.db = DatabaseManager(":memory:")
        self.db.initialize()
        repo = ChestRepository(self.db)
        for i, (x, y, p) in enumerate(chests):
            repo.add_chest(MAP, f"Chest {i + 1}", x, y, spawn_probability=p,
                           estimated_open_seconds=2.0)
        grid = GridMap(size, size)
        obstacles = []
        for x, y, w, h in walls:
            grid.fill_rect(x, y, w, h)
            obstacles.append(Obstacle(x, y, w, h))
        grid.version = 0
        specs = tuple(ChestSpec(loc.id or 0, loc.location_name, loc.x, loc.y,
                                loc.spawn_probability, loc.estimated_open_seconds)
                      for loc in repo.get_chests_by_map(MAP))
        self.world = World(MAP, 0, grid, tuple(obstacles), specs, start, WorldTransform(1.0))

    def runner(self, **kwargs: object) -> SimulationRunner:
        kwargs.setdefault("seed", 1)
        kwargs.setdefault("clock_base", BASE_TIME)
        return SimulationRunner(self.db, self.settings, map_id=MAP, world=self.world,
                                **kwargs)  # type: ignore[arg-type]

    def observation_rows(self, session_id: int) -> list[tuple[int, int, int, float]]:
        rows = self.db.query_all(
            "SELECT chest_location_id, chest_present, chest_opened, interaction_seconds "
            "FROM chest_observations WHERE session_id = ? ORDER BY id", (session_id,))
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def close(self) -> None:
        self.db.close()


class AgentTestCase(unittest.TestCase):
    def harness(self, *args: object, **kwargs: object) -> Harness:
        h = Harness(*args, **kwargs)  # type: ignore[arg-type]
        self.addCleanup(h.close)
        return h


class StateMachineTests(AgentTestCase):
    def test_transition_table(self) -> None:
        self.assertEqual(set(TRANSITIONS), set(AgentState))
        for terminal in (AgentState.COMPLETED, AgentState.FAILED):
            self.assertEqual(TRANSITIONS[terminal], frozenset())
        for state in AgentState:
            if not state.is_terminal:
                self.assertTrue(is_valid_transition(state, AgentState.FAILED))
        self.assertFalse(is_valid_transition(AgentState.NAVIGATING, AgentState.INTERACTING))

    def test_initial_state_and_initialisation(self) -> None:
        h = self.harness([(10.5, 2.5, 1.0)])
        runner = h.runner()
        self.assertIs(runner.state, AgentState.INITIALIZING)
        self.assertFalse(runner.started)
        session_id = runner.start()
        self.assertIs(runner.state, AgentState.SELECTING_TARGET)
        session = SessionRepository(h.db).require_session(session_id)
        self.assertIs(session.status, SessionStatus.RUNNING)
        self.assertEqual(session.map_id, MAP)
        assert runner.controller is not None
        self.assertEqual(len(runner.controller.tracker), 1)
        with self.assertRaises(RuntimeError):
            runner.start()

    def test_full_episode_visits_every_location(self) -> None:
        h = self.harness([(10.5, 2.5, 1.0), (20.5, 20.5, 0.0), (5.5, 25.5, 1.0)],
                         walls=((15, 0, 1, 25),))
        runner = h.runner()
        result = runner.run()
        self.assertEqual(result.outcome, "completed")
        self.assertIs(result.state, AgentState.COMPLETED)
        m = result.metrics
        self.assertEqual((m.locations_visited, m.chests_found, m.chests_opened), (3, 2, 2))
        assert runner.controller is not None
        visited = Counter(t.target for t in runner.controller.history)
        for state in (AgentState.SELECTING_TARGET, AgentState.PLANNING_ROUTE,
                      AgentState.NAVIGATING, AgentState.VERIFYING_CHEST,
                      AgentState.INTERACTING, AgentState.CONFIRMING_OPEN,
                      AgentState.COMPLETED):
            self.assertGreater(visited[state], 0, state)
        for t in runner.controller.history:
            self.assertTrue(is_valid_transition(t.source, t.target))

    def test_target_selection_is_not_database_order(self) -> None:
        # Chest 1 is far away, chest 3 is next to the start.
        h = self.harness([(25.5, 25.5, 1.0), (15.5, 15.5, 1.0), (4.5, 4.5, 1.0)])
        runner = h.runner(strategy="nearest")
        runner.run()
        assert runner.controller is not None
        order = [o.chest_id for o in runner.controller.observations]
        self.assertEqual(order, [3, 2, 1])


class ChestHandlingTests(AgentTestCase):
    def test_absent_chest_is_recorded_without_interaction(self) -> None:
        h = self.harness([(10.5, 2.5, 0.0)])
        runner = h.runner()
        result = runner.run()
        assert result.session_id is not None
        self.assertEqual(h.observation_rows(result.session_id), [(1, 0, 0, 0.0)])
        assert runner.controller is not None
        self.assertEqual(runner.controller.transition_counts[AgentState.INTERACTING], 0)
        self.assertEqual(result.metrics.chests_opened, 0)

    def test_successful_interaction_respects_duration(self) -> None:
        h = self.harness([(10.5, 2.5, 1.0)], settings=make_settings(
            sim={"chest_interaction_seconds": 1.5}))
        runner = h.runner()
        result = runner.run()
        assert result.session_id is not None
        rows = h.observation_rows(result.session_id)
        self.assertEqual(len(rows), 1)
        chest_id, present, opened, seconds = rows[0]
        self.assertEqual((chest_id, present, opened), (1, 1, 1))
        self.assertAlmostEqual(seconds, 1.5, places=6)
        self.assertTrue(runner.env.chests[1].opened)

    def test_failed_interactions_are_bounded_and_recorded(self) -> None:
        h = self.harness([(10.5, 2.5, 1.0), (2.5, 12.5, 0.0)], settings=make_settings(
            sim={"interaction_failure_probability": 1.0}, agent={"max_recovery_attempts": 2}))
        result = h.runner().run()
        m = result.metrics
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(m.chests_opened, 0)
        self.assertEqual(m.failed_interactions, 3)  # first attempt + 2 retries
        self.assertEqual(m.abandoned_targets, 1)
        self.assertEqual(m.recoveries, 3)
        stats = result.session_statistics
        assert stats is not None
        self.assertEqual(stats.failed_interactions, 3)
        self.assertEqual(stats.locations_visited, 2)

    def test_unreachable_location_is_reported_not_observed(self) -> None:
        enclosure = ((18, 18, 5, 1), (18, 22, 5, 1), (18, 19, 1, 3), (22, 19, 1, 3))
        h = self.harness([(20.5, 20.5, 1.0), (8.5, 2.5, 1.0)], walls=enclosure)
        result = h.runner().run()
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.metrics.unreachable_targets, 1)
        assert result.session_id is not None
        self.assertEqual([r[0] for r in h.observation_rows(result.session_id)], [2])
        run = RunRepository(h.db).get_run(result.session_id)
        assert run is not None
        self.assertEqual(run.unreachable_targets, 1)

    def test_rewards_are_granted_separately(self) -> None:
        chests = [(5.5 + 4 * i, 2.5, 1.0) for i in range(5)]
        with_rewards = self.harness(chests, settings=make_settings(sim={"reward_probability": 1.0}))
        result = with_rewards.runner().run()
        self.assertEqual(result.metrics.rewards, result.metrics.chests_opened)
        without = self.harness(chests).runner().run()
        self.assertEqual(without.metrics.rewards, 0)
        self.assertEqual(without.metrics.chests_opened, result.metrics.chests_opened)


class RecoveryTests(AgentTestCase):
    def test_recovers_when_the_path_becomes_blocked(self) -> None:
        h = self.harness([(25.5, 2.5, 1.0)])
        runner = h.runner()
        runner.start()
        while runner.state is not AgentState.NAVIGATING:
            runner.step()
        for _ in range(10):
            runner.step()
        # Drop a wall across the route, leaving a gap at the top.
        runner.env.add_obstacle(Obstacle(15, 0, 1, 25))
        result = runner.run()
        self.assertEqual(result.outcome, "completed")
        self.assertGreaterEqual(result.metrics.recoveries, 1)
        self.assertEqual(result.metrics.chests_opened, 1)
        assert runner.controller is not None
        reasons = {t.reason for t in runner.controller.history
                   if t.target is AgentState.RECOVERING}
        self.assertIn("path_blocked", reasons)
        self.assertGreaterEqual(runner.pathfinder.searches, 2)

    def test_sealed_target_is_abandoned_without_infinite_loop(self) -> None:
        h = self.harness([(25.5, 2.5, 1.0), (2.5, 10.5, 1.0)],
                         settings=make_settings(agent={"max_recovery_attempts": 1}))
        runner = h.runner(strategy="nearest", max_ticks=20_000)
        runner.start()
        while runner.state is not AgentState.NAVIGATING:
            runner.step()
        runner.step()
        target = runner.env.agent.current_target
        # Seal whichever target is being approached inside a box.
        chest = runner.env.chests[target or 0]
        cx, cy = chest.cell
        for ob in (Obstacle(cx - 2, cy - 2, 5, 1), Obstacle(cx - 2, cy + 2, 5, 1),
                   Obstacle(cx - 2, cy - 1, 1, 3), Obstacle(cx + 2, cy - 1, 1, 3)):
            runner.env.add_obstacle(ob)
        result = runner.run()
        self.assertEqual(result.outcome, "completed")
        m = result.metrics
        self.assertEqual(m.unreachable_targets + m.abandoned_targets, 1)
        self.assertEqual(m.chests_opened, 1)
        self.assertLess(m.ticks, 20_000)

    def test_agent_never_overlaps_obstacles(self) -> None:
        settings = make_settings(sim={"map_width": 100, "map_height": 100,
                                      "world_units_per_cell": 10.0})
        db = DatabaseManager(":memory:")
        self.addCleanup(db.close)
        db.initialize()
        seed_demo_chests(db, MAP)
        runner = SimulationRunner(db, settings, map_id=MAP, strategy="weighted", seed=42)
        runner.start()
        grid = runner.env.grid
        while runner.step():
            agent = runner.env.agent
            self.assertFalse(circle_collides(grid, agent.position_x, agent.position_y,
                                             agent.radius))
        self.assertEqual(runner.metrics.outcome, "completed")


class DatabaseIntegrationTests(AgentTestCase):
    def setUp(self) -> None:
        self.h = self.harness([(6.5, 6.5, 0.5), (12.5, 3.5, 0.5), (20.5, 15.5, 0.5),
                               (3.5, 22.5, 0.5), (25.5, 25.5, 0.5), (15.5, 26.5, 0.5)])

    def test_session_run_and_statistics_match_metrics(self) -> None:
        result = self.h.runner(strategy="weighted", seed=7).run()
        assert result.session_id is not None
        session = SessionRepository(self.h.db).require_session(result.session_id)
        self.assertIs(session.status, SessionStatus.COMPLETED)
        stats = result.session_statistics
        m = result.metrics
        assert stats is not None
        self.assertEqual(stats.locations_visited, m.locations_visited)
        self.assertEqual(stats.chests_found, m.chests_found)
        self.assertEqual(stats.chests_opened, m.chests_opened)
        self.assertEqual(stats.failed_interactions, m.failed_interactions)
        self.assertEqual(stats.total_observations, m.observations_recorded)
        assert stats.duration_seconds is not None and stats.chests_opened_per_minute is not None
        self.assertAlmostEqual(stats.duration_seconds, m.simulated_seconds, places=5)
        assert m.chests_per_minute is not None
        self.assertAlmostEqual(stats.chests_opened_per_minute, m.chests_per_minute, places=3)
        self.assertEqual(m.chests_opened, m.chests_spawned)  # every spawned chest opened

        run = RunRepository(self.h.db).get_run(result.session_id)
        assert run is not None
        self.assertEqual((run.strategy, run.seed, run.episode, run.outcome),
                         ("weighted", 7, 0, "completed"))
        self.assertEqual(run.ticks, m.ticks)
        self.assertEqual(run.parameters["time_basis"], "simulated")
        self.assertEqual(run.parameters["navigation"]["default_strategy"], "weighted")

    def test_observations_belong_to_the_session_map_without_duplicates(self) -> None:
        result = self.h.runner().run()
        assert result.session_id is not None
        rows = self.h.db.query_all(
            "SELECT c.map_id, o.chest_location_id, o.chest_opened FROM chest_observations o "
            "JOIN chest_locations c ON c.id = o.chest_location_id WHERE o.session_id = ?",
            (result.session_id,))
        self.assertEqual({r[0] for r in rows}, {MAP})
        ids = [r[1] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 6)

    def test_repeatable_across_sessions_and_episodes(self) -> None:
        first = self.h.runner(seed=3).run()
        second = self.h.runner(seed=3).run()
        assert first.session_id is not None and second.session_id is not None
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(self.h.observation_rows(first.session_id),
                         self.h.observation_rows(second.session_id))
        self.assertEqual(first.metrics.ticks, second.metrics.ticks)
        patterns = {tuple(r[1] for r in self.h.observation_rows(
            self.h.runner(seed=3, episode=e).run().session_id or 0)) for e in range(1, 6)}
        self.assertGreater(len(patterns), 1)

    def test_chunked_and_single_stepping_give_identical_results(self) -> None:
        a = self.h.runner(seed=5)
        while a.step(1):
            pass
        b = self.h.runner(seed=5)
        while b.step(997):
            pass
        self.assertEqual(a.metrics.ticks, b.metrics.ticks)
        self.assertEqual(a.env.agent.position, b.env.agent.position)

    def test_tick_limit_completes_the_session(self) -> None:
        result = self.h.runner(max_ticks=45).run()
        self.assertEqual(result.outcome, "tick_limit")
        self.assertEqual(result.metrics.ticks, 45)
        self.assertAlmostEqual(result.metrics.simulated_seconds, 1.5)
        stats = result.session_statistics
        assert stats is not None and stats.duration_seconds is not None
        self.assertEqual(stats.status, SessionStatus.COMPLETED)
        self.assertAlmostEqual(stats.duration_seconds, 1.5, places=5)

    def test_cancel_aborts_the_session(self) -> None:
        runner = self.h.runner()
        runner.step(20)
        runner.cancel("test")
        self.assertTrue(runner.finished)
        self.assertEqual(runner.metrics.outcome, "cancelled")
        assert runner.session_id is not None
        session = SessionRepository(self.h.db).require_session(runner.session_id)
        self.assertIs(session.status, SessionStatus.ABORTED)
        run = RunRepository(self.h.db).get_run(runner.session_id)
        assert run is not None
        self.assertEqual((run.outcome, run.failure_reason), ("cancelled", "test"))
        self.assertFalse(runner.step())
        runner.cancel()  # idempotent

    def test_no_database_queries_or_astar_searches_while_navigating(self) -> None:
        statements: list[str] = []
        runner = self.h.runner()
        runner.start()
        while runner.state is not AgentState.NAVIGATING:
            runner.step()
        self.h.db.connection.set_trace_callback(statements.append)
        try:
            searches = runner.pathfinder.searches
            navigating_ticks = 0
            while runner.state is AgentState.NAVIGATING:
                runner.step()
                navigating_ticks += 1
            self.assertGreater(navigating_ticks, 20)
            self.assertEqual(runner.pathfinder.searches, searches)
            # The tick that ends navigation may verify a chest; that alone writes nothing
            # until the observation batch is flushed.
            self.assertEqual(statements, [])
        finally:
            self.h.db.connection.set_trace_callback(None)

    def test_database_statements_scale_with_events_not_ticks(self) -> None:
        statements: list[str] = []
        runner = self.h.runner()
        self.h.db.connection.set_trace_callback(statements.append)
        try:
            result = runner.run()
        finally:
            self.h.db.connection.set_trace_callback(None)
        self.assertGreater(result.metrics.ticks, 500)
        self.assertLess(len(statements), 40 + 10 * result.metrics.observations_recorded)
        self.assertLess(len(statements), result.metrics.ticks / 5)

    def test_invalid_runner_arguments(self) -> None:
        with self.assertRaises(ValidationError):
            self.h.runner(strategy="teleport")
        with self.assertRaises(ValidationError):
            self.h.runner(max_ticks=0)
        with self.assertRaises(ValidationError):
            self.h.runner(seed=-1)


class DemoMapTests(unittest.TestCase):
    def test_load_world_requires_chests(self) -> None:
        with DatabaseManager(":memory:") as db:
            db.initialize()
            with self.assertRaises(ChestHunterError):
                load_world(db, make_settings(), map_id="empty", map_seed=1)

    def test_demo_map_episode_completes_quickly(self) -> None:
        settings = make_settings(sim={"world_units_per_cell": 10.0})
        with DatabaseManager(":memory:") as db:
            db.initialize()
            seed_demo_chests(db, MAP)
            result = SimulationRunner(db, settings, map_id=MAP, seed=42).run()
        m = result.metrics
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(m.candidate_locations, len(DEMO_CHESTS))
        self.assertEqual(m.locations_visited + m.unreachable_targets + m.abandoned_targets,
                         len(DEMO_CHESTS))
        self.assertEqual(m.chests_opened, m.chests_spawned)
        self.assertGreater(m.ticks / max(m.cpu_seconds, 1e-9), 500)

    def test_headless_run_does_not_import_tkinter(self) -> None:
        code = textwrap.dedent("""
            import sys
            from chest_hunter.config import load_config
            from chest_hunter.database import DatabaseManager
            from chest_hunter.demo_data import seed_demo_chests
            from chest_hunter.agent import SimulationRunner
            with DatabaseManager(":memory:") as db:
                db.initialize()
                seed_demo_chests(db)
                result = SimulationRunner(db, load_config(), max_ticks=600).run()
            assert result.outcome == "tick_limit", result.outcome
            assert "tkinter" not in sys.modules
            print("ok")
        """)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
        env.pop("DISPLAY", None)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env=env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
