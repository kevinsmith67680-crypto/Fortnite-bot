"""Phase 3 integration tests: visual planning, the visual FSM, interaction,
SQLite persistence and ground-truth isolation."""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
import textwrap
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from chest_hunter.agent.runner import SimulationRunner
from chest_hunter.chest_repository import ChestRepository
from chest_hunter.config import (
    DatabaseSettings,
    LoggingSettings,
    Settings,
    SimulationSettings,
    VisionSettings,
    load_config,
)
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ValidationError
from chest_hunter.localization.pose_estimator import PoseEstimator
from chest_hunter.localization.world_state import SpawnPrior, SpawnStatus, WorldModel
from chest_hunter.navigation.visual_planner import VisualPlanner
from chest_hunter.run_repository import RunRepository
from chest_hunter.scenarios import VISUAL_ARENA_ID, seed_scenario
from chest_hunter.simulation.collision import GridMap
from chest_hunter.simulation.entities import Obstacle
from chest_hunter.simulation.environment import (
    ChestSpec,
    Environment,
    InteractionResult,
    World,
    WorldTransform,
)
from chest_hunter.vision.obstacle_detector import BLOCKED, FREE, UNKNOWN
from chest_hunter.vision.visual_tracker import VisualTracker
from tests.tk_support import collect_on_main_thread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SMALL_MAP = "visual_test"


def small_settings(**sim: Any) -> Settings:
    base = load_config(PROJECT_ROOT / "config" / "settings.json")
    simulation = replace(base.simulation, map_width=60, map_height=40, world_units_per_cell=1.0,
                         decoration_density=0.0, default_map=SMALL_MAP, **sim)
    return replace(base, simulation=simulation, database=DatabaseSettings(Path(":memory:")),
                   logging=LoggingSettings("WARNING"))


class SmallWorld:
    """A hand-made 60 x 40 world with chests registered in an in-memory database."""

    def __init__(self, chests: list[tuple[float, float, float]],
                 walls: tuple[tuple[int, int, int, int], ...] = (),
                 settings: Settings | None = None, start: tuple[float, float] = (5.5, 5.5)):
        self.settings = settings or small_settings()
        self.db = DatabaseManager(":memory:")
        self.db.initialize()
        repo = ChestRepository(self.db)
        for i, (x, y, p) in enumerate(chests):
            repo.add_chest(SMALL_MAP, f"Spot {i + 1}", x, y, spawn_probability=p)
        grid = GridMap(60, 40)
        obstacles = []
        for x, y, w, h in walls:
            grid.fill_rect(x, y, w, h)
            obstacles.append(Obstacle(x, y, w, h))
        grid.version = 0
        specs = tuple(ChestSpec(c.id or 0, c.location_name, c.x, c.y, c.spawn_probability,
                                c.estimated_open_seconds)
                      for c in repo.get_chests_by_map(SMALL_MAP))
        self.world = World(SMALL_MAP, 0, grid, tuple(obstacles), specs, start,
                           WorldTransform(1.0))

    def runner(self, **kwargs: Any) -> SimulationRunner:
        kwargs.setdefault("seed", 1)
        kwargs.setdefault("max_ticks", 9000)
        kwargs.setdefault("agent", "visual")
        kwargs.setdefault("evaluate", kwargs["agent"] != "oracle")
        return SimulationRunner(self.db, self.settings, map_id=SMALL_MAP, world=self.world,
                                **kwargs)

    def rows(self, session_id: int) -> list[tuple[int, int, int]]:
        return [(r[0], r[1], r[2]) for r in self.db.query_all(
            "SELECT chest_location_id, chest_present, chest_opened FROM chest_observations "
            "WHERE session_id = ? ORDER BY id", (session_id,))]


THREE_CHESTS = [(30.5, 20.5, 1.0), (50.5, 10.5, 1.0), (45.5, 32.5, 0.0)]
DIVIDER = ((20, 0, 1, 30),)   # wall with a gap at the top


# ---------------------------------------------------------------------------
# World model and planner
# ---------------------------------------------------------------------------


class PlannerTests(unittest.TestCase):
    def make(self, priors: list[SpawnPrior] | None = None) -> tuple[WorldModel, VisualPlanner]:
        world = WorldModel(40, 20, priors or [], pose=PoseEstimator(), tracker=VisualTracker())
        planner = VisualPlanner(world, speed=5.0, acceleration=12.0, view_width=10,
                                view_height=6, open_seconds=2.0, unknown_cost=3.0)
        return world, planner

    def test_planning_grid_follows_the_occupancy_estimate(self) -> None:
        world, _ = self.make()
        version = world.planning_grid.version
        world.occupancy.cells[:, :10] = FREE
        world.occupancy.cells[5, 3] = BLOCKED
        world._sync_planning_grid((0, 40, 0, 20))
        self.assertGreater(world.planning_grid.version, version)
        self.assertTrue(world.planning_grid.is_blocked(3, 5))
        self.assertFalse(world.planning_grid.is_blocked(20, 5))   # unknown is not a wall
        self.assertEqual(world.unknown_cells[5 * 40 + 20], 1)
        self.assertEqual(world.unknown_cells[5 * 40 + 2], 0)

    def test_unknown_space_costs_more_than_explored_space(self) -> None:
        world, planner = self.make()
        world.occupancy.cells[0:2, :] = FREE        # known corridor along the bottom
        world._sync_planning_grid((0, 40, 0, 20))
        result = planner.plan((0, 1), (39, 1))
        self.assertTrue(result.found)
        self.assertTrue(all(y <= 1 for _, y in result.path))
        unknown = planner.plan((0, 10), (39, 10))   # through unexplored space
        self.assertTrue(unknown.found)
        self.assertGreater(unknown.cost, result.cost)

    def test_planning_starts_from_a_walkable_cell_when_the_estimate_is_in_a_wall(self) -> None:
        world, planner = self.make()
        world.occupancy.cells[:, :] = FREE
        world.occupancy.cells[:, 10] = BLOCKED       # a wall the estimate drifted into
        world._sync_planning_grid((0, 40, 0, 20))
        self.assertEqual(planner.start_cell((5.5, 5.5)), (5, 5))   # free: unchanged
        start = planner.start_cell((10.2, 5.5))                    # just inside the wall
        self.assertEqual(start, (9, 5))
        self.assertTrue(planner.plan(start, (3, 3)).found)
        self.assertFalse(planner.plan((10, 5), (3, 3)).found)      # what used to happen

    def test_frontier_selection_and_priors(self) -> None:
        prior = SpawnPrior(1, "p", 35.5, 15.5, 1.0)
        world, planner = self.make([prior])
        world.occupancy.cells[:, :20] = FREE
        world.occupancy.cells[:, 20] = BLOCKED
        world.occupancy.cells[0:3, 20] = FREE      # door at the bottom
        world.occupancy.cells[0:3, 21:24] = FREE
        world._sync_planning_grid((0, 40, 0, 20))
        goals = planner.frontier_goals((5.0, 10.0))
        self.assertTrue(goals)
        for goal in goals:
            x, y = goal.cell
            self.assertEqual(world.occupancy.state(x, y), FREE)
            self.assertTrue(any(world.occupancy.state(x + dx, y + dy) == UNKNOWN
                                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        choice = planner.choose_exploration((5.0, 10.0), 0.0)
        assert choice is not None
        self.assertGreaterEqual(choice[0].cell[0], 20)   # through the door, toward the prior
        world.occupancy.cells[:] = FREE
        world._sync_planning_grid((0, 40, 0, 20))
        self.assertEqual(planner.frontier_goals((5.0, 10.0)), [])

    def test_unreachable_frontiers_are_blacklisted(self) -> None:
        world, planner = self.make()
        world.occupancy.cells[:, :20] = FREE
        world.occupancy.cells[:, 20] = BLOCKED
        world.occupancy.cells[:, 21:] = BLOCKED
        world.occupancy.cells[8:12, 30:34] = FREE     # free pocket with unknown inside
        world.occupancy.cells[9:11, 31:33] = UNKNOWN
        world._sync_planning_grid((0, 40, 0, 20))
        self.assertIsNone(planner.choose_exploration((5.0, 10.0), 0.0))
        self.assertTrue(planner._blacklist)


# ---------------------------------------------------------------------------
# Full episode on the acceptance arena
# ---------------------------------------------------------------------------


class AcceptanceEpisodeTests(unittest.TestCase):
    """20 spawn points, random presence, walls, an unreachable vault (spec section 23)."""

    runner: SimulationRunner
    db: DatabaseManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.db = DatabaseManager(":memory:")
        cls.db.initialize()
        seed_scenario(cls.db)
        cls.settings = load_config(PROJECT_ROOT / "config" / "settings.json")
        cls.runner = SimulationRunner(cls.db, cls.settings, map_id=VISUAL_ARENA_ID, seed=42,
                                      agent="visual", evaluate=True, max_ticks=20_000)
        cls.result = cls.runner.run()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.close()

    def test_episode_completes_autonomously(self) -> None:
        self.assertEqual(self.result.outcome, "completed")
        self.assertLess(self.result.metrics.ticks, 20_000)
        controller = self.runner.controller
        states = Counter(t.target.value for t in controller.history)
        for state in ("OBSERVING", "LOCALIZING", "EXPLORING", "TARGET_DETECTED",
                      "PLANNING_ROUTE", "NAVIGATING", "ALIGNING", "INTERACTING",
                      "VERIFYING_OPEN", "COMPLETED"):
            self.assertGreater(states[state], 0, state)

    def test_perception_quality(self) -> None:
        summary = self.runner.vision_evaluator.summary()
        self.assertGreaterEqual(summary["precision"], 0.95)
        self.assertGreaterEqual(summary["recall"], 0.85)
        self.assertEqual(summary["localisation_success_rate"], 1.0)
        self.assertLess(summary["estimate_error_cells"]["mean"], 0.2)
        world = self.runner.controller.world
        self.assertGreater(world.occupancy.known_fraction, 0.5)
        known = world.occupancy.cells != UNKNOWN
        truth = np.frombuffer(self.runner.env.grid.cells, np.uint8).reshape(100, 100)
        wrong = np.count_nonzero((world.occupancy.cells[known] == BLOCKED) != (truth[known] == 1))
        self.assertLess(wrong / known.sum(), 0.01)

    def test_every_reachable_chest_is_opened_and_visually_verified(self) -> None:
        env = self.runner.env
        vault = next(c for c in env.chests.values() if c.location_name == "Sealed Vault - Inside")
        reachable_present = [c for c in env.chests.values() if c.present and c is not vault]
        self.assertTrue(all(c.opened for c in reachable_present))
        self.assertFalse(vault.opened)
        self.assertEqual(self.result.metrics.chests_opened, len(reachable_present))
        priors = self.runner.controller.world.priors
        self.assertIs(priors[vault.id].status, SpawnStatus.UNREACHABLE)
        self.assertTrue(all(p.resolved for p in priors.values()))

    def test_sqlite_records_are_consistent(self) -> None:
        sid = self.result.session_id
        assert sid is not None
        rows = self.db.query_all(
            "SELECT o.chest_location_id, o.chest_present, o.chest_opened, c.map_id "
            "FROM chest_observations o JOIN chest_locations c ON c.id = o.chest_location_id "
            "WHERE o.session_id = ?", (sid,))
        env = self.runner.env
        self.assertEqual({r[3] for r in rows}, {VISUAL_ARENA_ID})
        opened = [r[0] for r in rows if r[2]]
        self.assertEqual(len(opened), len(set(opened)))          # no duplicate openings
        for location_id, present, was_opened, _ in rows:
            self.assertEqual(bool(present), env.chests[location_id].present)
            if was_opened:
                self.assertTrue(env.chests[location_id].opened)
        stats = self.result.session_statistics
        assert stats is not None
        self.assertEqual(stats.chests_opened, self.result.metrics.chests_opened)
        run = RunRepository(self.db).get_run(sid)
        assert run is not None
        self.assertEqual(run.strategy, "visual-frontier")
        self.assertEqual(run.parameters["agent_kind"], "visual")
        vault_id = next(c.id for c in env.chests.values() if c.location_name.startswith("Sealed"))
        self.assertNotIn(vault_id, [r[0] for r in rows])          # never observed up close

    def test_deterministic_repeat(self) -> None:
        again = SimulationRunner(self.db, self.settings, map_id=VISUAL_ARENA_ID, seed=42,
                                 agent="visual", max_ticks=20_000).run()
        assert again.session_id is not None and self.result.session_id is not None
        rows = lambda sid: [tuple(r) for r in self.db.query_all(  # noqa: E731
            "SELECT chest_location_id, chest_present, chest_opened FROM chest_observations "
            "WHERE session_id = ? ORDER BY id", (sid,))]
        self.assertEqual(rows(self.result.session_id), rows(again.session_id))
        self.assertEqual(again.metrics.ticks, self.result.metrics.ticks)


# ---------------------------------------------------------------------------
# Ground-truth isolation
# ---------------------------------------------------------------------------

POLICY_MODULES = [
    "vision/preprocessing.py", "vision/chest_detector.py", "vision/obstacle_detector.py",
    "vision/interaction_detector.py", "vision/visual_tracker.py",
    "vision/perception_pipeline.py", "vision/templates.py",
    "localization/pose_estimator.py", "localization/world_state.py",
    "navigation/visual_planner.py", "navigation/local_controller.py",
    "agent/visual_controller.py",
]
FORBIDDEN = ("chest_hunter.simulation.environment", "chest_hunter.simulation.entities",
             "chest_hunter.simulation.map_generator", "chest_hunter.simulation.sprites",
             "chest_hunter.simulation.camera", "chest_hunter.simulation.interface",
             "chest_hunter.analytics.vision_metrics")


class GroundTruthIsolationTests(unittest.TestCase):
    def test_policy_modules_do_not_import_simulator_internals(self) -> None:
        for module in POLICY_MODULES:
            tree = ast.parse((SRC / "chest_hunter" / module).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
            with self.subTest(module=module):
                self.assertFalse(imported & set(FORBIDDEN), imported & set(FORBIDDEN))

    def test_controller_holds_no_simulator_objects(self) -> None:
        world = SmallWorld(THREE_CHESTS)
        runner = world.runner()
        runner.start()
        controller = runner.controller
        forbidden = (Environment, World)
        for owner in (controller, controller.world, controller.planner, controller.local,
                      controller.local.body, controller.pipeline):
            for name, value in vars(owner).items():
                values = value if isinstance(value, (list, tuple)) else [value]
                for item in values:
                    self.assertNotIsInstance(item, forbidden, f"{type(owner).__name__}.{name}")
        port = controller.port
        self.assertFalse(any(isinstance(v, Environment) for k, v in vars(port).items()
                             if not k.startswith("_VisualSimulatorInterface__")))
        runner.cancel()
        world.db.close()

    def test_hidden_chests_are_treated_as_absent(self) -> None:
        """Present-but-invisible chests: an agent reading ground truth would open them."""
        world = SmallWorld(THREE_CHESTS)
        runner = world.runner()
        hidden = {cid for cid, chest in runner.env.chests.items() if chest.present}
        self.assertEqual(len(hidden), 2)
        runner.env.hidden_chests = set(hidden)
        result = runner.run()
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.metrics.chests_opened, 0)
        self.assertFalse(any(c.opened for c in runner.env.chests.values()))
        assert result.session_id is not None
        recorded = {(cid, present) for cid, present, _ in world.rows(result.session_id)}
        for cid in hidden:
            self.assertIn((cid, 0), recorded)      # judged empty from what it could see
        world.db.close()


# ---------------------------------------------------------------------------
# Behaviour and recovery on a small world
# ---------------------------------------------------------------------------


class VisualBehaviourTests(unittest.TestCase):
    def tearDown(self) -> None:
        for world in getattr(self, "_worlds", []):
            world.db.close()

    def world(self, *args: Any, **kwargs: Any) -> SmallWorld:
        w = SmallWorld(*args, **kwargs)
        self.__dict__.setdefault("_worlds", []).append(w)
        return w

    def test_complete_small_episode(self) -> None:
        world = self.world(THREE_CHESTS, DIVIDER)
        result = world.runner().run()
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.metrics.chests_opened, 2)
        assert result.session_id is not None
        self.assertEqual(sorted(world.rows(result.session_id)), [(1, 1, 1), (2, 1, 1), (3, 0, 0)])

    def test_inconclusive_verification_is_credited_later_on_visual_evidence(self) -> None:
        world = self.world([(30.5, 20.5, 1.0)])
        runner = world.runner()
        runner.start()
        controller = runner.visual_controller
        # Blind the target check used by ALIGN/VERIFY so verification is inconclusive;
        # only the late check (which reads raw detections) can confirm the open.
        controller._target_appearance = lambda result, goal: (False, False)  # type: ignore[method-assign]
        result = runner.run()
        self.assertTrue(runner.env.chests[1].opened)
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(controller.late_verifications, 1)
        self.assertEqual(result.metrics.chests_opened, 1)
        self.assertIs(controller.world.priors[1].status, SpawnStatus.OPENED)
        assert result.session_id is not None
        self.assertEqual(world.rows(result.session_id), [(1, 1, 1)])

    def test_full_frame_marker_search_without_regions_of_interest(self) -> None:
        base = small_settings()
        settings = replace(base, vision=replace(base.vision, use_regions_of_interest=False))
        world = self.world(THREE_CHESTS, DIVIDER, settings=settings)
        result = world.runner().run()
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.metrics.chests_opened, 2)

    def test_visual_agents_require_vision_enabled(self) -> None:
        base = small_settings()
        settings = replace(base, vision=replace(base.vision, enabled=False))
        world = self.world(THREE_CHESTS, settings=settings)
        for agent in ("visual", "chest_vision"):
            with self.assertRaises(ValidationError):
                world.runner(agent=agent)
        self.assertEqual(world.runner(agent="oracle").run().outcome, "completed")

    def test_failed_interactions_are_bounded_and_recorded(self) -> None:
        settings = small_settings(interaction_failure_probability=1.0)
        world = self.world([(30.5, 20.5, 1.0)], settings=settings)
        runner = world.runner()
        result = runner.run()
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.metrics.chests_opened, 0)
        assert result.session_id is not None
        rows = world.rows(result.session_id)
        self.assertTrue(rows)
        self.assertTrue(all(r == (1, 1, 0) for r in rows))
        self.assertLessEqual(len(rows), settings.visual_navigation.maximum_recovery_attempts + 1)
        self.assertIs(runner.controller.world.priors[1].status, SpawnStatus.FAILED)

    def test_out_of_range_interaction_does_nothing(self) -> None:
        world = self.world([(30.5, 20.5, 1.0)])
        env = Environment(world.world, world.settings.simulation, seed=1)
        for _ in range(100):
            self.assertEqual(env.hold_interact(), InteractionResult.NOT_PRESENT)
            env.advance()
        self.assertFalse(env.chests[1].opened)
        self.assertEqual(env.hold_fraction, 0.0)

    def test_new_obstacle_triggers_replanning(self) -> None:
        world = self.world([(55.5, 20.5, 1.0)], start=(5.5, 20.5))
        runner = world.runner()
        runner.start()
        while not (runner.state.value == "NAVIGATING" and runner.controller.goal is not None
                   and runner.controller.goal.kind == "chest"):
            runner.step()
        # Drop a wall (with a gap at the top) between the agent and its target.
        wall_x = int((runner.env.agent.position_x + 55.5) / 2)
        self.assertLess(runner.env.agent.position_x + 1, wall_x)
        runner.env.add_obstacle(Obstacle(wall_x, 0, 1, 35))
        result = runner.run()
        self.assertGreaterEqual(runner.controller.replans, 1)
        self.assertEqual(result.metrics.chests_opened, 1)

    def test_stuck_agent_recovers(self) -> None:
        world = self.world([(40.5, 20.5, 1.0)])
        runner = world.runner()
        runner.start()
        port = runner.interface
        real_move = port.move

        def jammed(vx: float, vy: float) -> None:   # wheels slip for 12 simulated seconds
            if runner.env.time < 12.0:
                runner.env.agent.stop()
                return
            real_move(vx, vy)

        port.move = jammed  # type: ignore[method-assign]
        result = runner.run()
        reasons = Counter(t.reason for t in runner.controller.history
                          if t.target.value == "RECOVERING")
        self.assertGreater(reasons["stuck"], 0)
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.metrics.chests_opened, 1)

    def test_lost_target_returns_to_exploration(self) -> None:
        world = self.world([(40.5, 20.5, 1.0)])
        runner = world.runner(max_ticks=3000)
        runner.start()
        while not (runner.controller.goal is not None and runner.controller.goal.kind == "chest"):
            runner.step()
        runner.env.hidden_chests = {1}           # the chest vanishes from view
        result = runner.run()
        self.assertGreaterEqual(runner.controller.targets_lost, 1)
        self.assertEqual(result.metrics.chests_opened, 0)
        self.assertFalse(runner.env.chests[1].opened)

    def test_frame_dropout_is_survived(self) -> None:
        settings = small_settings()
        settings = replace(settings, vision=replace(settings.vision, frame_dropout=0.6))
        world = self.world(THREE_CHESTS, settings=settings)
        runner = world.runner()
        result = runner.run()
        self.assertGreater(runner.controller.failed_frames, 50)
        self.assertEqual(result.outcome, "completed")
        self.assertEqual(result.metrics.chests_opened, 2)

    def test_missing_marker_fails_safely(self) -> None:
        world = self.world(THREE_CHESTS)
        runner = world.runner(max_ticks=3000)
        with mock.patch("chest_hunter.simulation.sprites.draw_agent"):
            result = runner.run()
        self.assertEqual(result.outcome, "failed")
        self.assertIn("localisation", result.end_reason or "")
        self.assertLess(result.metrics.ticks, 3000)
        assert result.session_id is not None
        session = world.db.query_one("SELECT status FROM simulation_sessions WHERE id = ?",
                                     (result.session_id,))
        assert session is not None
        self.assertEqual(session["status"], "aborted")

    def test_low_resolution_camera(self) -> None:
        settings = small_settings()
        settings = replace(settings, vision=VisionSettings(default_width=320, default_height=180))
        world = self.world(THREE_CHESTS, settings=settings)
        result = world.runner().run()
        self.assertEqual(result.metrics.chests_opened, 2)

    def test_headless_visual_run_needs_no_tkinter(self) -> None:
        code = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            from test_visual_navigation import SmallWorld, THREE_CHESTS
            w = SmallWorld(THREE_CHESTS)
            r = w.runner().run()
            assert r.metrics.chests_opened == 2, r.metrics
            assert "tkinter" not in sys.modules
            print("ok")
        """)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
        env.pop("DISPLAY", None)
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env=env, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn("ok", done.stdout)


class SettingsFixtureTests(unittest.TestCase):
    def test_small_settings_are_valid(self) -> None:
        settings = small_settings()
        self.assertIsInstance(settings.simulation, SimulationSettings)
        self.assertTrue(settings.template_dir().is_dir())
        self.assertTrue(math.isclose(settings.simulation.world_units_per_cell, 1.0))


def tearDownModule() -> None:
    collect_on_main_thread()  # this module's Tk objects are freed here, on the main thread


if __name__ == "__main__":
    unittest.main()
