"""Phase 7 deployment: startup checks, configuration versioning, structured logging,
graceful shutdown and reproducible execution."""

from __future__ import annotations

import io
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from chest_hunter.config import CONFIG_VERSION, load_config
from chest_hunter.database import SCHEMA_VERSION, DatabaseManager
from chest_hunter.exceptions import ConfigError
from chest_hunter.logging_config import configure_logging
from chest_hunter.runtime import startup

ROOT = Path(__file__).resolve().parents[1]


def _by_name(checks: list[startup.Check]) -> dict[str, startup.Check]:
    return {c.name: c for c in checks}


class StartupCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = load_config(ROOT / "config" / "settings.json")
        self.settings = replace(base.with_database_path(Path(self.tmp.name) / "db.sqlite"),
                                exploration=replace(base.exploration, memory_path=str(
                                    Path(self.tmp.name) / "memory.json")))

    def test_clean_startup_on_the_reference_backend(self) -> None:
        DatabaseManager(self.settings.database.path).initialize()
        checks = startup.run_startup_checks(self.settings, backend="reference",
                                            model_dir=ROOT / "training/checkpoints/"
                                            "bc_synthetic_v1")
        failed = [c for c in checks if not c.ok]
        self.assertEqual(failed, [], startup.format_checks(checks))
        names = _by_name(checks)
        self.assertIn("bridge:protocol", names)
        self.assertIn("reference-arena-7", names["bridge:environment"].detail)
        self.assertTrue(names["model:load"].ok)
        self.assertIn(f"version {SCHEMA_VERSION}", names["database"].detail)

    def test_newer_database_schema_fails_with_a_remedy(self) -> None:
        path = Path(self.settings.database.path)
        con = sqlite3.connect(path)
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        con.close()
        check = startup.check_database(self.settings)
        self.assertFalse(check.ok)
        self.assertIn("update the program", check.remedy)

    def test_missing_database_is_a_warning(self) -> None:
        check = startup.check_database(self.settings)
        self.assertTrue(check.ok and check.warning)
        self.assertIn("init-db", check.remedy)

    def test_incompatible_model_is_reported(self) -> None:
        bad = Path(self.tmp.name) / "model"
        bad.mkdir()
        self.assertFalse(startup.check_model(self.settings, bad)[0].ok)
        (bad / "policy.json").write_text(json.dumps({"format": "other/2"}))
        res = startup.check_model(self.settings, bad)
        self.assertFalse(res[0].ok)
        self.assertIn("re-export", res[0].remedy)

    def test_unsupported_observation_size_fails(self) -> None:
        s = replace(self.settings, simulation_3d=replace(self.settings.simulation_3d,
                                                         observation_width=333))
        checks = _by_name(startup.check_observation(s))
        self.assertFalse(checks["observation"].ok)

    def test_unreachable_unreal_server_fails_with_instructions(self) -> None:
        s = replace(self.settings, unreal_bridge=replace(self.settings.unreal_bridge, port=1))
        checks = startup.check_bridge(s, "unreal")
        self.assertFalse(checks[0].ok)
        self.assertIn("--backend reference", checks[0].remedy)

    def test_corrupt_exploration_memory_fails(self) -> None:
        Path(self.settings.exploration_memory_path()).write_text("{\"format\": \"x\"}")
        self.assertFalse(startup.check_memory(self.settings).ok)

    def test_locked_versions_follow_python_markers(self) -> None:
        locked = startup._locked_versions(ROOT / "requirements-lock.txt")
        self.assertIn("numpy", locked)
        self.assertIn("opencv-python-headless", locked)
        expected = "2.4.6" if sys.version_info < (3, 12) else "2.5.4"
        self.assertEqual(locked["numpy"], expected)


class ConfigurationVersionTest(unittest.TestCase):
    def _write(self, data: dict[str, object]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "settings.json"
        path.write_text(json.dumps(data))
        return path

    def test_versioned_configuration(self) -> None:
        data = json.loads((ROOT / "config" / "settings.json").read_text())
        self.assertEqual(data["config_version"], CONFIG_VERSION)
        self.assertEqual(load_config(ROOT / "config" / "settings.json").config_version,
                         CONFIG_VERSION)
        newer = dict(data, config_version=CONFIG_VERSION + 1)
        with self.assertRaises(ConfigError):
            load_config(self._write(newer))
        older = {k: v for k, v in data.items() if k not in ("config_version", "exploration")}
        s = load_config(self._write(older))
        self.assertEqual(s.config_version, 0)
        self.assertEqual(s.exploration.mode, "spawn_points")

    def test_exploration_section_is_validated(self) -> None:
        data = json.loads((ROOT / "config" / "settings.json").read_text())
        for bad in ({"strategy": "random"}, {"inspect_confidence": 0.9},
                    {"benchmark_observation_width": 333}, {"unknown_key": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(ConfigError):
                    load_config(self._write(dict(data, exploration=dict(
                        data["exploration"], **bad))))


class LoggingTest(unittest.TestCase):
    def test_json_log_lines(self) -> None:
        stream = io.StringIO()
        logger = configure_logging("INFO", stream=stream, log_format="json")
        try:
            logging.getLogger("chest_hunter.test").info("hello %s", "world")
        finally:
            configure_logging("WARNING")
        entry = json.loads(stream.getvalue().strip().splitlines()[-1])
        self.assertEqual(entry["message"], "hello world")
        self.assertEqual(entry["level"], "INFO")
        self.assertIn("time", entry)
        self.assertIs(logger, logging.getLogger("chest_hunter"))
        with self.assertRaises(ValueError):
            configure_logging("INFO", log_format="xml")


def _cli(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.Popen[str]:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), **(env_extra or {}))
    return subprocess.Popen([sys.executable, "-m", "chest_hunter", *args], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class ProcessTest(unittest.TestCase):
    def test_check_runtime_command(self) -> None:
        proc = _cli("check-runtime", "--json")
        out, err = proc.communicate(timeout=120)
        self.assertEqual(proc.returncode, 0, err)
        checks = json.loads(out)
        self.assertTrue(all(c["ok"] for c in checks))

    def test_sigterm_stops_an_episode_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "db.sqlite")
            DatabaseManager(db).initialize()
            proc = _cli("--db", db, "simulate-3d", "--exploration", "active", "--map",
                        "procedural", "--seed", "3", "--backend", "reference",
                        "--observation-width", "320", "--observation-height", "180",
                        "--max-ticks", "200000")
            time.sleep(8.0)
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 130, err)
        self.assertIn("interrupted", err)
        self.assertIn("p7_", out)

    def test_episodes_are_reproducible(self) -> None:
        outs = []
        with tempfile.TemporaryDirectory() as tmp:
            for k in range(2):
                db = str(Path(tmp) / f"db{k}.sqlite")
                DatabaseManager(db).initialize()
                proc = _cli("--db", db, "simulate-3d", "--exploration", "active", "--map",
                            "procedural", "--seed", "5", "--backend", "reference",
                            "--observation-width", "320", "--observation-height", "180",
                            "--max-ticks", "600")
                out, err = proc.communicate(timeout=300)
                self.assertEqual(proc.returncode, 0, err)
                outs.append([line for line in out.splitlines()
                             if line.split(":")[0].strip() in (
                                 "Outcome", "Chests opened", "Travel distance", "Recoveries",
                                 "Exploration")])
        self.assertEqual(outs[0], outs[1])
        self.assertTrue(outs[0])


if __name__ == "__main__":
    unittest.main()
