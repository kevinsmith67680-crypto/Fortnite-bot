"""Phase 6 workflow: configuration, CLI, monitoring, profiling, events, experience
partitions, selector monitors, ground-truth isolation, operations panel and the end-to-end
acceptance run (on the Python reference arena; Unreal Engine is not available here)."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chest_hunter.config import load_config
from chest_hunter.database import MIGRATIONS, SCHEMA_VERSION, DatabaseManager
from chest_hunter.exceptions import ConfigError
from tests.phase5_helpers import ROOT, settings
from tests.tk_support import collect_on_main_thread, collect_tk_garbage_after


def _tmpdir(case: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp(prefix="ch_p6_"))
    case.addCleanup(shutil.rmtree, path, True)
    return path


class Configuration(unittest.TestCase):
    def _load(self, case: unittest.TestCase, **sections: Any) -> Any:
        data = json.loads((ROOT / "config" / "settings.json").read_text())
        data.update(sections)
        path = _tmpdir(case) / "settings.json"
        path.write_text(json.dumps(data))
        return load_config(path)

    def test_defaults_are_loaded(self) -> None:
        s = settings()
        self.assertEqual(s.supervisor.max_recovery_attempts, 3)
        self.assertEqual(s.planning.strategy, "short_horizon")
        self.assertTrue(s.performance.adaptive_vision)
        self.assertEqual(s.endurance.save_checkpoint_every, 10)

    def test_invalid_values_are_rejected(self) -> None:
        for section, bad in (("supervisor", {"mode": "panic"}),
                             ("supervisor", {"max_recovery_attempts": 0}),
                             ("planning", {"strategy": "global"}),
                             ("planning", {"beam_width": 0}),
                             ("performance", {"vision_profile": "turbo"}),
                             ("endurance", {"save_checkpoint_every": 0}),
                             ("supervisor", {"unknown_key": 1})):
            with self.assertRaises(ConfigError, msg=f"{section} {bad}"):
                self._load(self, **{section: bad})

    def test_sections_are_optional(self) -> None:
        data = json.loads((ROOT / "config" / "settings.json").read_text())
        for key in ("supervisor", "planning", "performance", "endurance"):
            data.pop(key)
        path = _tmpdir(self) / "settings.json"
        path.write_text(json.dumps(data))
        self.assertEqual(load_config(path).planning.beam_width, 5)


class Profiling(unittest.TestCase):
    def test_report_ranks_components_with_percentiles(self) -> None:
        from chest_hunter.runtime.profiler import Profiler, format_report

        prof = Profiler()
        for k in range(100):
            prof.add("fast", 0.001)
            prof.add("slow", 0.010 + 0.0001 * k)
        rows = prof.report()
        self.assertEqual([r["component"] for r in rows], ["slow", "fast"])
        slow = rows[0]
        self.assertEqual(slow["calls"], 100)
        self.assertAlmostEqual(slow["median_ms"], 14.95, places=2)
        self.assertGreater(slow["p95_ms"], slow["median_ms"])
        self.assertAlmostEqual(slow["max_ms"], 19.9, places=3)
        self.assertIn("| slow |", format_report(rows))

    def test_samples_are_bounded(self) -> None:
        from chest_hunter.runtime.profiler import Profiler

        prof = Profiler(max_samples=50)
        for k in range(10_000):
            prof.add("x", k * 1e-6)
        self.assertEqual(len(prof._series["x"].samples), 50)
        self.assertEqual(prof.report()[0]["calls"], 10_000)

    def test_sections_measure_wall_and_cpu_time(self) -> None:
        from chest_hunter.runtime.profiler import NULL_PROFILER, Profiler

        prof = Profiler()
        with prof.section("work"):
            sum(i * i for i in range(20000))
        row = prof.report()[0]
        self.assertGreater(row["total_s"], 0.0)
        self.assertGreaterEqual(row["cpu_total_s"], 0.0)
        with NULL_PROFILER.section("anything"):
            pass
        self.assertEqual(NULL_PROFILER.report(), [])


class ResourceMonitoring(unittest.TestCase):
    def test_snapshot_fields(self) -> None:
        from chest_hunter.analytics.resource_monitor import ResourceMonitor
        from chest_hunter.runtime.profiler import Profiler

        prof = Profiler()
        prof.add("chest_detection", 0.004)
        prof.add("learned_inference", 0.001)
        mon = ResourceMonitor(interval_seconds=0.01, profiler=prof)
        snap = mon.sample()
        for key in ("process_cpu_percent", "process_rss_mb", "process_threads",
                    "system_cpu_percent", "system_memory_used_mb", "latency_ms"):
            self.assertIn(key, snap)
        self.assertGreater(snap["process_rss_mb"], 1.0)
        self.assertAlmostEqual(snap["latency_ms"]["perception"], 4.0, places=3)
        self.assertAlmostEqual(snap["latency_ms"]["policy"], 1.0, places=3)
        if snap["gpu"] is None:
            self.assertTrue(snap["gpu_note"])           # never an invented GPU figure
        self.assertIsNone(snap["unreal_frame_time_ms"])
        self.assertEqual(mon.sample()["latency_ms"]["perception"], None)   # nothing new

    def test_rate_limited_sampling(self) -> None:
        from chest_hunter.analytics.resource_monitor import ResourceMonitor

        mon = ResourceMonitor(interval_seconds=3600.0, query_gpu=False)
        self.assertIsNotNone(mon.maybe_sample())
        self.assertIsNone(mon.maybe_sample())


class Events(unittest.TestCase):
    def test_bounded_log_with_exact_counts(self) -> None:
        from chest_hunter.runtime.events import EventLog

        log = EventLog(capacity=10, episode=3)
        for k in range(25):
            log.emit("stationary" if k % 2 else "state_timeout", sim_time=k, detail=str(k))
        self.assertEqual(len(log), 10)
        self.assertEqual(log.total, 25)
        self.assertEqual(log.dropped, 15)
        self.assertEqual(log.counts["stationary"], 12)
        path = _tmpdir(self) / "events.jsonl"
        self.assertEqual(log.write_jsonl(path), 10)
        first = json.loads(path.read_text().splitlines()[0])
        self.assertEqual(first["episode"], 3)
        with self.assertRaises(ValueError):
            log.emit("x", severity="fatal")


class ExperiencePartitions(unittest.TestCase):
    def test_migration_from_version_4_keeps_route_records(self) -> None:
        path = _tmpdir(self) / "v4.db"
        db = DatabaseManager(path)
        with db.transaction() as conn:
            for _v, statements in MIGRATIONS[:4]:
                for statement in statements:
                    conn.execute(statement)
            conn.execute("PRAGMA user_version = 4")
            conn.execute("INSERT INTO route_memory (map_id, layout_version, start_region, "
                         "dest_region, estimated_length, estimated_seconds, actual_seconds, "
                         "success, created_at) VALUES ('m', 'L', 'a', 'b', 10, 2, 3, 1, 't')")
        db.close()
        db = DatabaseManager(path)
        self.addCleanup(db.close)
        self.assertEqual(db.initialize(), SCHEMA_VERSION)
        row = db.query_all("SELECT partition_key, actual_seconds FROM route_memory")[0]
        self.assertEqual((row["partition_key"], row["actual_seconds"]), ("", 3.0))

    def test_route_memory_reads_only_its_partition(self) -> None:
        from chest_hunter.exploration.route_memory import RouteMemory

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        with db.transaction() as conn:
            for key, success in (("A", 1), ("A", 1), ("B", 0), ("B", 0), ("", 0)):
                conn.execute("INSERT INTO route_memory (map_id, layout_version, start_region, "
                             "dest_region, estimated_length, estimated_seconds, actual_seconds,"
                             " success, created_at, partition_key) VALUES ('m', 'L', 's', 'd', "
                             "10, 2, 2, ?, 't', ?)", (success, key))
        a = RouteMemory(db, "m", "L", partition_key="A")
        b = RouteMemory(db, "m", "L", partition_key="B")
        legacy = RouteMemory(db, "m", "L", partition_key="A", include_legacy=True)
        everything = RouteMemory(db, "m", "L")
        self.assertGreater(a.reliability("d"), b.reliability("d"))
        self.assertLess(legacy.reliability("d"), a.reliability("d"))
        self.assertLess(everything.reliability("d"), a.reliability("d"))

    def test_partitions_differ_by_environment_agent_and_configuration(self) -> None:
        from chest_hunter.runtime.configurations import PHASE6_SETUPS, partition

        base = {"map_id": "research_arena_3d", "env_overrides": {"noise_std": 5.0},
                "setup": PHASE6_SETUPS["phase4_deterministic"], "model_version": None,
                "width": 640, "height": 360, "fov_deg": 90.0, "decision_ticks": 2}
        ref = partition(**base)
        for change in ({"env_overrides": {"noise_std": 9.0}},
                       {"setup": PHASE6_SETUPS["phase5_hybrid"]},
                       {"width": 320, "height": 180}):
            other = partition(**{**base, **change})
            self.assertNotEqual(other["key"], ref["key"], change)
        self.assertEqual(partition(**base), ref)            # deterministic

    def test_spawn_estimates_can_be_restricted_to_an_environment(self) -> None:
        from chest_hunter.exploration.spawn_estimator import SpawnEstimator
        from chest_hunter.scenarios import seed_arena_3d

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        loc = db.query_all("SELECT id FROM chest_locations LIMIT 1")[0]["id"]
        with db.transaction() as conn:
            for k, env in enumerate(("E1", "E2", None)):
                conn.execute("INSERT INTO simulation_sessions (map_id, status, started_at) "
                             "VALUES ('research_arena_3d', 'running', '2026-01-01T00:00:00')")
                sid = conn.execute("SELECT max(id) FROM simulation_sessions").fetchone()[0]
                params = {"layout_hash": "L"} | ({"partition": {"environment_version": env}}
                                                 if env else {})
                conn.execute("INSERT INTO simulation_runs (session_id, strategy, seed, episode,"
                             " parameters_json, outcome, created_at, finished_at) VALUES "
                             "(?, 's', 1, ?, ?, 'completed', 't', 't')",
                             (sid, k, json.dumps(params)))
                conn.execute("INSERT INTO chest_observations (session_id, chest_location_id, "
                             "chest_present, chest_opened, observed_at) VALUES (?, ?, 1, 0, "
                             "'2026-01-01T00:00:01')", (sid, loc))
                conn.execute("UPDATE simulation_sessions SET status = 'completed', "
                             "ended_at = '2026-01-01T00:00:02' WHERE id = ?", (sid,))
        counts = {e: SpawnEstimator(db, environment_version=e).observation_counts(
            "research_arena_3d", "L").get(loc, (0, 0, 0))[2] for e in (None, "E1")}
        self.assertEqual(counts[None], 3)
        self.assertEqual(counts["E1"], 2)                    # E1 + the unversioned session
        strict = SpawnEstimator(db, environment_version="E1", include_unversioned=False)
        self.assertEqual(strict.observation_counts("research_arena_3d", "L")[loc][2], 1)


class EpisodeExperience(unittest.TestCase):
    def test_every_visual_episode_leaves_an_experience_summary(self) -> None:
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.exploration.experience import summarise_experience
        from chest_hunter.runtime.configurations import PHASE6_SETUPS, partition
        from chest_hunter.scenarios import seed_arena_3d

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        keys = []
        for k, setup in enumerate(("phase4_deterministic", "phase6_deterministic_supervised")):
            part = partition(map_id="research_arena_3d", env_overrides={},
                             setup=PHASE6_SETUPS[setup], model_version=None, width=320,
                             height=180, fov_deg=90.0, decision_ticks=2)
            keys.append(part["key"])
            Runner3D.from_settings(db, settings(), mode="hybrid", seed=7, episode=k,
                                   backend="reference", max_ticks=600, width=320, height=180,
                                   partition=part).run()
        rows = db.query_all("SELECT * FROM episode_experience ORDER BY session_id")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["partition_key"] for r in rows], keys)
        self.assertTrue(0.0 <= rows[0]["coverage"] <= 1.0)
        self.assertGreater(rows[0]["spawn_points"], 0)
        summary = summarise_experience(db, "research_arena_3d")
        self.assertEqual(set(summary), set(keys))                # never mixed
        one = summarise_experience(db, "research_arena_3d", partition_key=keys[0])
        self.assertEqual(list(one), [keys[0]])
        self.assertEqual(one[keys[0]]["episodes"], 1)

    def test_the_oracle_baseline_leaves_none(self) -> None:
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.scenarios import seed_arena_3d

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        Runner3D.from_settings(db, settings(), mode="oracle", seed=7, backend="reference",
                               max_ticks=300, width=320, height=180).run()
        self.assertEqual(db.query_all("SELECT COUNT(*) AS n FROM episode_experience")[0]["n"],
                         0)


class SelectorMonitors(unittest.TestCase):
    def _selector(self, **limits: Any) -> Any:
        from chest_hunter.learning.policy_selector import PolicySelector, SafetyLimits

        agent = SimpleNamespace(now=0.0, telemetry=None)
        policy = SimpleNamespace(validated=lambda: True)
        sel = PolicySelector.__new__(PolicySelector)
        sel.agent = agent
        sel.policy = policy
        sel.limits = SafetyLimits(**limits)
        from collections import deque

        from chest_hunter.learning.policy_selector import SelectorStats
        sel.stats = SelectorStats()
        sel._signs = deque(maxlen=sel.limits.oscillation_window)
        sel._mode = "learned"
        sel._recover_until = -1.0
        sel.switches = deque(maxlen=10)
        sel._failed_hold_streak = 0
        return sel

    def _out(self, yaw: float) -> Any:
        return SimpleNamespace(look_yaw=yaw, move_right=0.0)

    def test_oscillation_is_off_by_default(self) -> None:
        sel = self._selector()
        self.assertFalse(any(sel._oscillating(self._out((-1) ** k * 0.8)) for k in range(40)))

    def test_left_right_left_is_detected(self) -> None:
        sel = self._selector(oscillation_flips=8)
        hits = [sel._oscillating(self._out((-1) ** k * 0.8)) for k in range(24)]
        self.assertTrue(any(hits))
        self.assertGreaterEqual(sel.stats.fallbacks["oscillation"], 1)

    def test_steady_turning_is_not_oscillation(self) -> None:
        sel = self._selector(oscillation_flips=8)
        self.assertFalse(any(sel._oscillating(self._out(0.6)) for _ in range(40)))

    def test_failed_holds_hand_over_to_the_state_machine(self) -> None:
        sel = self._selector(max_failed_holds=3)
        for _ in range(3):
            sel._failed_hold()
        self.assertEqual(sel.stats.fallbacks["failed_interactions_handover"], 1)
        self.assertEqual(sel.switches[-1]["reason"], "handover: failed_interactions")


class GroundTruthIsolation(unittest.TestCase):
    """The Phase 6 decision code never imports the simulator or the privileged channel."""

    MODULES = ("runtime/supervisor.py", "runtime/state_spec.py", "runtime/events.py",
               "runtime/profiler.py", "exploration/advanced_planner.py",
               "learning/policy_selector.py")

    def test_no_simulator_imports(self) -> None:
        src = ROOT / "src" / "chest_hunter"
        for rel in self.MODULES:
            tree = ast.parse((src / rel).read_text())
            names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                     and n.module] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                                      for a in n.names]
            for name in names:
                self.assertFalse(name.startswith(("chest_hunter.simulation3d",
                                                  "chest_hunter.simulation.",
                                                  "chest_hunter.unreal_bridge.mock_server")),
                                 f"{rel} imports {name}")
            text = (src / rel).read_text()
            for privileged in ("oracle_state", "evaluation_labels"):
                self.assertNotIn(privileged, text, f"{rel} uses {privileged}")


class CommandLine(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = _tmpdir(self)
        self.db = self.dir / "cli.db"

    def _cli(self, *args: str) -> tuple[int, str]:
        from chest_hunter.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--db", str(self.db), "--log-level", "ERROR", *args])
        return code, out.getvalue()

    def test_simulate_3d_with_phase6_options(self) -> None:
        self._cli("init-db")
        self._cli("seed-demo", "--scenario", "research_arena_3d")
        code, out = self._cli("simulate-3d", "--agent", "hybrid", "--planner", "short-horizon",
                              "--supervisor", "enabled", "--adaptive-compute", "--profile",
                              "--seed", "42", "--backend", "reference", "--max-ticks", "600",
                              "--observation-width", "320", "--observation-height", "180")
        self.assertEqual(code, 0, out)
        for text in ("Supervisor:", "Planner:", "short_horizon", "| simulator_step |",
                     "NOT Unreal Engine"):
            self.assertIn(text, out)

    def test_endurance_test_command(self) -> None:
        out_dir = self.dir / "endurance"
        code, out = self._cli("endurance-test", "--episodes", "2", "--max-ticks-per-episode",
                              "450", "--map", "proc_train_001", "--output", str(out_dir))
        self.assertEqual(code, 0, out)
        self.assertIn("Completion rate", out)
        self.assertTrue((out_dir / "checkpoint.json").exists())
        code, out = self._cli("endurance-test", "--episodes", "2", "--max-ticks-per-episode",
                              "450", "--map", "proc_train_001", "--output", str(out_dir))
        self.assertEqual(code, 2)                           # refuses to overwrite a run
        self.assertIn("--resume", out)

    def test_profile_agent_command(self) -> None:
        out_dir = self.dir / "profile"
        code, out = self._cli("profile-agent", "--episodes", "1", "--max-ticks", "450",
                              "--observation-width", "320", "--observation-height", "180",
                              "--vision-profiles", "phase5,efficient", "--output", str(out_dir))
        self.assertEqual(code, 0, out)
        data = json.loads((out_dir / "profile.json").read_text())
        self.assertEqual(set(data["profiles"]), {"phase5", "efficient"})
        names = {r["component"] for r in data["profiles"]["phase5"]["components"]}
        self.assertTrue({"chest_detection", "simulator_step", "preprocessing"} <= names)

    def test_benchmark_phase6_command(self) -> None:
        out_dir = self.dir / "bench"
        code, out = self._cli("benchmark-phase6", "--episodes", "1", "--max-ticks", "450",
                              "--configurations",
                              "phase4_deterministic,phase6_deterministic_optimised",
                              "--output-dir", str(out_dir))
        self.assertEqual(code, 0, out)
        summary = json.loads((out_dir / "summary.json").read_text())
        self.assertEqual(summary["episodes_completed"], 1)
        self.assertIn("phase6_deterministic_optimised",
                      summary["paired_vs_phase4_deterministic"])
        self.assertTrue((out_dir / "episodes.csv").exists())


@unittest.skipUnless(os.environ.get("DISPLAY"), "no display (run under xvfb-run)")
class OperationsPanelWindow(unittest.TestCase):
    def test_panel_shows_a_running_episode_and_sets_flags(self) -> None:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            self.skipTest("this Python build has no Tk")
        collect_tk_garbage_after(self)              # Tk next to the mock server's threads
        from chest_hunter.agent.operations_panel import OperationsPanel
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.runtime.operations import OperationsControl
        from chest_hunter.runtime.supervisor import SupervisorConfig
        from chest_hunter.scenarios import seed_arena_3d

        control = OperationsControl()
        panel = OperationsPanel(control, refresh_seconds=0.0)
        self.addCleanup(panel.close)
        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        runner = Runner3D.from_settings(db, settings(), mode="hybrid", seed=1,
                                        backend="reference", max_ticks=300, width=320,
                                        height=180, supervisor=SupervisorConfig(),
                                        on_decision=panel.on_decision)
        runner.run()
        self.assertGreater(panel.updates, 10)
        self.assertNotEqual(panel.values["state"].get(), "-")
        control.pause()
        self.assertTrue(control.paused)
        control.resume()
        control.request_stop("test")
        self.assertTrue(control.stop_requested)


class OperationsSnapshot(unittest.TestCase):
    def test_snapshot_has_the_operator_fields(self) -> None:
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.runtime.operations import snapshot
        from chest_hunter.runtime.supervisor import SupervisorConfig
        from chest_hunter.scenarios import seed_arena_3d

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        snaps: list[dict[str, Any]] = []
        runner = Runner3D.from_settings(
            db, settings(), mode="hybrid", seed=1, backend="reference", max_ticks=300,
            width=320, height=180, supervisor=SupervisorConfig(),
            on_decision=lambda r, _res: snaps.append(snapshot(r)) if len(snaps) < 3 else None)
        runner.run()
        for key in ("episode", "state", "policy", "planner", "target", "chests_opened",
                    "chests_per_minute", "coverage", "recoveries", "supervisor_warnings",
                    "frames_per_sim_second", "transitions"):
            self.assertIn(key, snaps[-1])


class Acceptance(unittest.TestCase):
    """Section 20 on the Python reference arena: repeated hybrid episodes with the supervisor
    and short-horizon planning, reset between episodes, then the acceptance checks that can
    be evaluated here. Unreal Engine is not available, so this is not the Unreal run."""

    def test_repeated_supervised_hybrid_episodes(self) -> None:
        from chest_hunter.analytics.endurance import EnduranceConfig, EnduranceRunner
        from chest_hunter.runtime.checkpoint import read_results
        from tests.phase5_helpers import tiny_model

        out = _tmpdir(self)
        cfg = EnduranceConfig(episodes=4, max_ticks_per_episode=900, seed=21, policy="hybrid",
                              planner="short_horizon", supervisor=True, adaptive_compute=True,
                              maps=("research_arena_3d", "proc_train_002"), width=320,
                              height=180, checkpoint_every=2, model_dir=str(tiny_model()))
        summary = EnduranceRunner(settings(), cfg, out, install_signal_handlers=False).run()
        records, _ = read_results(out / "episodes.jsonl")
        self.assertEqual(len(records), 4)
        # every episode ended normally or was stopped safely (no crash, no exception)
        self.assertTrue(all(r["exception"] is None for r in records))
        self.assertTrue(all(r["outcome"] in ("completed", "tick_limit", "failed")
                            for r in records))
        # bounded supervision: never more interventions than the per-episode budget
        budget = settings().supervisor.max_interventions
        self.assertTrue(all(r["supervisor_interventions"] <= budget for r in records))
        # no duplicate opening records: one opened observation per (session, spawn point)
        db = DatabaseManager(out / "endurance.db")
        self.addCleanup(db.close)
        dup = db.query_all("SELECT session_id, chest_location_id, COUNT(*) AS n "
                           "FROM chest_observations WHERE chest_opened = 1 "
                           "GROUP BY session_id, chest_location_id HAVING n > 1")
        self.assertEqual(dup, [])
        # the agent never counted more openings than really happened (ground truth)
        self.assertEqual(sum(r.get("over_reported") or 0 for r in records), 0)
        # measured resources and a valid report
        self.assertTrue(all(r["rss_mb"] > 0 and r["cpu_seconds"] > 0 for r in records))
        self.assertIn("Completion rate", (out / "report.md").read_text())
        self.assertEqual(summary["previous_agent_released"]["not_released"], 0)


def tearDownModule() -> None:
    collect_on_main_thread()  # this module's Tk objects are freed here, on the main thread


if __name__ == "__main__":
    unittest.main()
