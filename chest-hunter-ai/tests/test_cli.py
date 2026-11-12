"""Tests for the argparse CLI. Every test uses a temporary config and database."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from chest_hunter.chest_repository import ChestRepository
from chest_hunter.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, format_table, main
from chest_hunter.config import default_config_path, load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.demo_data import DEMO_CHESTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
    return env


class CliTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.db_path = self.tmp / "cli.db"
        self.config_path = self.tmp / "settings.json"
        self.write_config()
        # Safety net: the test configuration must never point at the real DB.
        self.assertEqual(load_config(self.config_path).database.path, self.db_path.resolve())

    def write_config(self, **simulation_overrides: object) -> None:
        simulation: dict[str, object] = {
            "default_map": "research_arena_01",
            "default_spawn_probability": 0.8,
            "default_open_seconds": 2.0,
        }
        simulation.update(simulation_overrides)
        config = {
            "database": {"path": str(self.db_path)},
            "simulation": simulation,
            "logging": {"level": "WARNING"},
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def run_cli(self, *args: str, config: bool = True) -> tuple[int, str, str]:
        argv = (["--config", str(self.config_path)] if config else []) + list(args)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        self.assertNotIn("Traceback", err.getvalue())
        return code, out.getvalue(), err.getvalue()

    def ok(self, *args: str) -> str:
        code, out, err = self.run_cli(*args)
        self.assertEqual(code, EXIT_OK, f"{args} failed: {err}")
        return out

    def fails(self, *args: str, code: int = EXIT_FAILURE) -> str:
        actual, _out, err = self.run_cli(*args)
        self.assertEqual(actual, code, f"{args} should exit {code}; stderr: {err}")
        return err

    def chest_count(self) -> int:
        with DatabaseManager(self.db_path) as db:
            return len(ChestRepository(db).get_all_chests())


class HelpAndUsageTests(CliTestCase):
    def test_help_lists_commands(self) -> None:
        code, out, _ = self.run_cli("--help", config=False)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("usage:", out)
        for command in ("init-db", "seed-demo", "list-chests", "add-chest", "start-session",
                        "record-observation", "complete-session", "abort-session",
                        "session-stats", "map-stats"):
            self.assertIn(command, out)

    def test_subcommand_help(self) -> None:
        code, out, _ = self.run_cli("add-chest", "--help", config=False)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("--probability", out)

    def test_version(self) -> None:
        code, out, _ = self.run_cli("--version", config=False)
        self.assertEqual(code, EXIT_OK)
        self.assertRegex(out, r"\d+\.\d+\.\d+")

    def test_missing_command_is_a_usage_error(self) -> None:
        err = self.fails(code=EXIT_USAGE)
        self.assertIn("required", err)

    def test_invalid_command_is_rejected(self) -> None:
        err = self.fails("launch-rocket", code=EXIT_USAGE)
        self.assertIn("invalid choice", err)

    def test_invalid_arguments_are_rejected(self) -> None:
        cases = [
            ("add-chest", "--name", "x", "--x", "abc", "--y", "1"),
            ("add-chest", "--x", "1", "--y", "1"),  # missing --name
            ("session-stats", "--session", "0"),
            ("session-stats", "--session", "one"),
            ("record-observation", "--session", "1", "--chest", "1"),  # no --present/--absent
            ("record-observation", "--session", "1", "--chest", "1", "--present", "--absent"),
            ("list-chests", "--enabled-only", "--disabled-only"),
            ("list-chests", "--sort", "random"),
            ("list-chests", "--bogus-flag"),
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                self.fails(*argv, code=EXIT_USAGE)


class DatabaseCommandTests(CliTestCase):
    def test_init_db_creates_database(self) -> None:
        out = self.ok("init-db")
        self.assertIn("Initialised new database", out)
        self.assertTrue(self.db_path.is_file())

    def test_init_db_twice_preserves_data(self) -> None:
        self.ok("init-db")
        self.ok("add-chest", "--name", "Keep", "--x", "1", "--y", "2")
        out = self.ok("init-db")
        self.assertIn("already initialised", out)
        self.assertEqual(self.chest_count(), 1)

    def test_commands_require_an_initialised_database(self) -> None:
        err = self.fails("list-chests")
        self.assertIn("init-db", err)
        self.assertFalse(self.db_path.exists(), "non-init commands must not create a database")

    def test_db_option_overrides_config_and_works_after_command(self) -> None:
        other = self.tmp / "other.db"
        self.ok("init-db", "--db", str(other))
        self.assertTrue(other.is_file())
        self.assertFalse(self.db_path.exists())
        code, out, _ = self.run_cli("--db", str(other), "list-chests")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("No chest locations found", out)

    def test_invalid_configuration_is_reported_cleanly(self) -> None:
        self.config_path.write_text("{ not json", encoding="utf-8")
        err = self.fails("init-db")
        self.assertIn("not valid JSON", err)
        self.write_config(default_spawn_probability=3)
        err = self.fails("init-db")
        self.assertIn("default_spawn_probability", err)

    def test_missing_configuration_file_is_reported(self) -> None:
        err = self.fails("--config", str(self.tmp / "missing.json"), "init-db")
        self.assertIn("Configuration file not found", err)


class ChestCommandTests(CliTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ok("init-db")

    def test_seed_demo_inserts_records_once(self) -> None:
        out = self.ok("seed-demo")
        self.assertIn(f"{len(DEMO_CHESTS)} inserted", out)
        out = self.ok("seed-demo")
        self.assertIn("0 inserted", out)
        self.assertEqual(self.chest_count(), len(DEMO_CHESTS))
        self.assertGreaterEqual(self.chest_count(), 30)

    def test_list_chests(self) -> None:
        self.ok("seed-demo")
        out = self.ok("list-chests")
        self.assertIn("Central Warehouse - Loading Bay", out)
        self.assertIn("Southern Docks - Crane Cabin", out)
        self.assertIn(f"{len(DEMO_CHESTS)} chest location(s).", out)
        self.assertRegex(out.splitlines()[0], r"^ID\s+Map\s+Location")

        out = self.ok("list-chests", "--map", "some_other_map")
        self.assertIn("No chest locations found", out)
        out = self.ok("list-chests", "--disabled-only")
        self.assertIn("No chest locations found", out)

    def test_add_show_update_delete_chest(self) -> None:
        out = self.ok("add-chest", "--map", "research_arena_01", "--name", "Warehouse Chest",
                      "--x", "120", "--y", "340", "--z", "0", "--probability", "0.85")
        match = re.search(r"Added chest location (\d+)", out)
        assert match is not None
        chest_id = match.group(1)
        self.assertIn("Spawn probability:", out)
        self.assertIn("0.85", out)
        self.assertIn("2.00 s", out)  # default_open_seconds from config

        self.assertIn("Warehouse Chest", self.ok("show-chest", "--chest", chest_id))

        out = self.ok("update-chest", "--chest", chest_id, "--probability", "0.6", "--disable")
        self.assertIn("0.60", out)
        self.assertIn("Enabled:", out)
        self.assertIn("no", out)
        self.assertIn("Warehouse Chest", self.ok("list-chests", "--disabled-only"))

        self.assertIn("Deleted", self.ok("delete-chest", "--chest", chest_id))
        self.assertIn("does not exist", self.fails("delete-chest", "--chest", chest_id))
        self.assertIn("does not exist", self.fails("show-chest", "--chest", chest_id))

    def test_add_chest_uses_configured_defaults(self) -> None:
        self.write_config(default_map="arena_b", default_spawn_probability=0.65,
                          default_open_seconds=3.5)
        out = self.ok("add-chest", "--name", "Defaults", "--x", "1", "--y", "1")
        self.assertIn("arena_b", out)
        self.assertIn("0.65", out)
        self.assertIn("3.50 s", out)

    def test_add_chest_rejects_invalid_values_without_traceback(self) -> None:
        cases = {
            ("--probability", "1.5"): "between 0 and 1",
            ("--probability", "-0.2"): "between 0 and 1",
            ("--seconds", "-1"): "must not be negative",
            ("--z", "nan"): "finite",
            ("--z", "inf"): "finite",
        }
        for extra, message in cases.items():
            with self.subTest(extra=extra):
                err = self.fails("add-chest", "--name", "Bad", "--x", "1", "--y", "1", *extra)
                self.assertTrue(err.startswith("error:"), err)
                self.assertIn(message, err)
        self.assertEqual(self.chest_count(), 0)

    def test_add_duplicate_position_is_rejected(self) -> None:
        self.ok("add-chest", "--name", "One", "--x", "5", "--y", "5")
        err = self.fails("add-chest", "--name", "Two", "--x", "5", "--y", "5")
        self.assertIn("already exists", err)

    def test_update_without_fields_is_rejected(self) -> None:
        self.ok("add-chest", "--name", "One", "--x", "5", "--y", "5")
        self.assertIn("No fields", self.fails("update-chest", "--chest", "1"))


class SessionCommandTests(CliTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ok("init-db")
        self.ok("seed-demo")

    def start_session(self) -> str:
        out = self.ok("start-session", "--map", "research_arena_01", "--quiet")
        session_id = out.strip()
        self.assertTrue(session_id.isdigit(), out)
        return session_id

    def test_end_to_end_session_workflow(self) -> None:
        out = self.ok("start-session", "--map", "research_arena_01", "--notes", "e2e")
        match = re.search(r"Started session (\d+)", out)
        assert match is not None
        session_id = match.group(1)

        out = self.ok("record-observation", "--session", session_id, "--chest", "3",
                      "--present", "--opened", "--seconds", "1.8")
        self.assertIn("present, opened", out)
        self.ok("record-observation", "--session", session_id, "--chest", "4", "--absent")
        out = self.ok("record-observation", "--session", session_id, "--chest", "5",
                      "--present", "--seconds", "2.5")
        self.assertIn("failed interaction", out)

        out = self.ok("unvisited-chests", "--session", session_id)
        self.assertIn(f"{len(DEMO_CHESTS) - 3} unvisited location(s)", out)

        out = self.ok("session-stats", "--session", session_id)
        self.assertIn("[running]", out)
        self.assertIn("unavailable (session still running)", out)

        out = self.ok("complete-session", "--session", session_id)
        self.assertIn("completed", out)

        out = self.ok("session-stats", "--session", session_id)
        self.assertIn("[completed]", out)
        self.assertRegex(out, r"Locations visited:\s+3")
        self.assertRegex(out, r"Observations:\s+3")
        self.assertRegex(out, r"Chests found:\s+2")
        self.assertRegex(out, r"Chests opened:\s+1")
        self.assertRegex(out, r"Failed interactions:\s+1")
        self.assertRegex(out, r"Interaction time:\s+4\.30 s")

        out = self.ok("map-stats", "--map", "research_arena_01")
        self.assertRegex(out, rf"Total chest locations:\s+{len(DEMO_CHESTS)}")
        self.assertRegex(out, r"Total visits:\s+3")
        self.assertRegex(out, r"Successful openings:\s+1")
        self.assertIn("unavailable (needs >= 5 samples, have 3)", out)
        out = self.ok("map-stats", "--min-samples", "3")
        self.assertRegex(out, r"Observed spawn frequency:\s+0\.667 \(from 3 samples\)")

        out = self.ok("list-sessions")
        self.assertIn("completed", out)
        self.assertIn("e2e", out)

    def test_observation_rule_violations_exit_nonzero(self) -> None:
        session_id = self.start_session()
        err = self.fails("record-observation", "--session", session_id, "--chest", "1",
                         "--absent", "--opened")
        self.assertIn("not present", err)
        err = self.fails("record-observation", "--session", session_id, "--chest", "1",
                         "--present", "--seconds", "-3")
        self.assertIn("must not be negative", err)
        self.ok("record-observation", "--session", session_id, "--chest", "1", "--present",
                "--opened")
        err = self.fails("record-observation", "--session", session_id, "--chest", "1",
                         "--present", "--opened")
        self.assertIn("already opened", err)
        self.assertIn("does not exist", self.fails(
            "record-observation", "--session", "999", "--chest", "1", "--present"))

    def test_finished_session_rules(self) -> None:
        session_id = self.start_session()
        self.ok("complete-session", "--session", session_id)
        self.assertIn("already finished", self.fails("complete-session", "--session", session_id))
        self.assertIn("already finished", self.fails("abort-session", "--session", session_id))
        err = self.fails("record-observation", "--session", session_id, "--chest", "1", "--absent")
        self.assertIn("running sessions", err)

        aborted = self.start_session()
        self.assertIn("aborted", self.ok("abort-session", "--session", aborted))
        self.assertIn("aborted", self.ok("list-sessions", "--status", "aborted"))

    def test_referenced_chest_cannot_be_deleted(self) -> None:
        session_id = self.start_session()
        self.ok("record-observation", "--session", session_id, "--chest", "2", "--absent")
        err = self.fails("delete-chest", "--chest", "2")
        self.assertIn("cannot be deleted", err)

    def test_unknown_session_stats(self) -> None:
        self.assertIn("does not exist", self.fails("session-stats", "--session", "77"))


class EntryPointTests(CliTestCase):
    def test_python_dash_m_entry_point(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "chest_hunter", "--config", str(self.config_path), "init-db"],
            cwd=self.tmp,  # proves nothing depends on the current working directory
            env=subprocess_env(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Initialised new database", result.stdout)

        result = subprocess.run(
            [sys.executable, "-m", "chest_hunter", "--config", str(self.config_path), "nope"],
            cwd=self.tmp, env=subprocess_env(), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, EXIT_USAGE)
        self.assertNotIn("Traceback", result.stderr)

    def test_seed_script(self) -> None:
        self.ok("init-db")
        script = PROJECT_ROOT / "scripts" / "seed_demo_chests.py"
        result = subprocess.run(
            [sys.executable, str(script), "--config", str(self.config_path)],
            cwd=self.tmp, env=subprocess_env(), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"{len(DEMO_CHESTS)} inserted", result.stdout)
        self.assertEqual(self.chest_count(), len(DEMO_CHESTS))


class IsolationTests(CliTestCase):
    def test_production_database_is_not_touched(self) -> None:
        production = load_config(default_config_path()).database.path
        before = production.stat() if production.exists() else None

        self.ok("init-db")
        self.ok("seed-demo")
        self.ok("start-session", "--quiet")

        after = production.stat() if production.exists() else None
        if before is None:
            self.assertIsNone(after, "tests must not create the production database")
        else:
            assert after is not None
            self.assertEqual((before.st_mtime_ns, before.st_size),
                             (after.st_mtime_ns, after.st_size))


class FormattingTests(unittest.TestCase):
    def test_format_table_aligns_columns(self) -> None:
        table = format_table(("ID", "Name", "P"), [(1, "a", "0.50"), (22, "long name", "1.00")],
                             right_align=(0, 2))
        lines = table.splitlines()
        self.assertEqual(lines[0], "ID  Name          P")
        self.assertEqual(lines[1], "--  ---------  ----")
        self.assertEqual(lines[2], " 1  a          0.50")
        self.assertEqual(lines[3], "22  long name  1.00")



class SimulationCommandTests(CliTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ok("init-db")
        self.ok("seed-demo")

    def test_help_lists_phase2_commands(self) -> None:
        _, out, _ = self.run_cli("--help", config=False)
        self.assertIn("simulate", out)
        self.assertIn("benchmark", out)
        code, out, _ = self.run_cli("simulate", "--help", config=False)
        self.assertEqual(code, EXIT_OK)
        for option in ("--strategy", "--seed", "--max-ticks", "--speed", "--headless", "--render"):
            self.assertIn(option, out)

    def test_simulate_headless_persists_and_reports(self) -> None:
        out = self.ok("simulate", "--map", "research_arena_01", "--strategy", "weighted",
                      "--seed", "42", "--headless")
        self.assertIn("Autonomous run finished: session 1, strategy weighted, seed 42", out)
        self.assertRegex(out, r"Outcome:\s+completed")
        self.assertRegex(out, r"Chests opened/minute:\s+\d+\.\d+ \(simulated time\)")
        self.assertIn("Session 1 on map research_arena_01 [completed]", out)
        self.assertRegex(out, rf"Locations visited:\s+{len(DEMO_CHESTS)}")
        stats = self.ok("session-stats", "--session", "1")
        self.assertRegex(stats, r"Strategy:\s+weighted")
        self.assertRegex(stats, r"Seed / episode:\s+42 / 0")
        self.assertIn("completed", self.ok("list-sessions"))

    def test_simulate_tick_limit_uses_defaults(self) -> None:
        out = self.ok("simulate", "--max-ticks", "60")
        self.assertIn("tick_limit", out)
        self.assertIn("strategy weighted, seed 42", out)  # config defaults
        self.assertRegex(out, r"Simulated time:\s+2\.000 s \(60 ticks")

    def test_simulate_argument_validation(self) -> None:
        self.fails("simulate", "--strategy", "random", code=EXIT_USAGE)
        self.fails("simulate", "--seed", "-1", code=EXIT_USAGE)
        self.fails("simulate", "--max-ticks", "0", code=EXIT_USAGE)
        self.fails("simulate", "--speed", "3", code=EXIT_USAGE)
        self.fails("simulate", "--headless", "--render", code=EXIT_USAGE)
        err = self.fails("simulate", "--map", "empty_map")
        self.assertIn("no enabled chest locations", err)

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs X11 semantics for DISPLAY")
    def test_render_without_a_display_fails_cleanly(self) -> None:
        env = subprocess_env()
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        result = subprocess.run(
            [sys.executable, "-m", "chest_hunter", "--config", str(self.config_path),
             "simulate", "--render", "--max-ticks", "10"],
            cwd=self.tmp, env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, EXIT_FAILURE, result.stdout + result.stderr)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_benchmark_command_writes_csv_without_touching_sessions(self) -> None:
        csv_path = self.tmp / "bench.csv"
        out = self.ok("benchmark", "--episodes", "2", "--max-ticks", "300", "--csv",
                      str(csv_path), "--quiet")
        self.assertIn("Nearest", out)
        self.assertIn("ProbabilityWeighted", out)
        self.assertIn("SIMULATED time", out)
        self.assertIn("paired episodes", out)
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1 + 2 * 2)
        self.assertIn("No sessions found", self.ok("list-sessions"))

    def test_benchmark_argument_validation(self) -> None:
        self.fails("benchmark", "--strategies", "nearest,teleport", code=EXIT_USAGE)
        self.fails("benchmark", "--episodes", "0", code=EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
