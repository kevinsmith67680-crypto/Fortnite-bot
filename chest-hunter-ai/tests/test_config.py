"""Tests for configuration loading and validation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from chest_hunter.config import (
    CONFIG_ENV_VAR,
    default_config_path,
    load_config,
    project_root,
)
from chest_hunter.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_config() -> dict[str, Any]:
    return {
        "database": {"path": "data/test.db"},
        "simulation": {
            "default_map": "research_arena_01",
            "default_spawn_probability": 0.8,
            "default_open_seconds": 2.0,
        },
        "logging": {"level": "INFO"},
    }


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.path = self.tmp / "settings.json"

    def write(self, data: Any) -> Path:
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return self.path

    def test_shipped_settings_file_is_valid_and_cwd_independent(self) -> None:
        original_cwd = Path.cwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, original_cwd)
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop(CONFIG_ENV_VAR, None)
            env.pop("CHEST_HUNTER_HOME", None)
            self.assertEqual(project_root(), PROJECT_ROOT)
            settings = load_config()
        self.assertEqual(settings.config_path, PROJECT_ROOT / "config" / "settings.json")
        self.assertEqual(settings.database.path, PROJECT_ROOT / "data" / "chest_locations.db")
        self.assertEqual(settings.simulation.default_map, "research_arena_01")
        self.assertEqual(settings.simulation.default_spawn_probability, 0.8)
        self.assertEqual(settings.simulation.default_open_seconds, 2.0)
        self.assertEqual(settings.logging.level, "INFO")

    def test_relative_paths_resolve_against_base_dir(self) -> None:
        settings = load_config(self.write(valid_config()), base_dir=self.tmp)
        self.assertEqual(settings.database.path, (self.tmp / "data" / "test.db").resolve())
        self.assertTrue(settings.database.path.is_absolute())

    def test_absolute_database_path_is_kept(self) -> None:
        data = valid_config()
        data["database"]["path"] = str(self.tmp / "abs.db")
        settings = load_config(self.write(data))
        self.assertEqual(settings.database.path, (self.tmp / "abs.db").resolve())

    def test_with_database_path_override(self) -> None:
        settings = load_config(self.write(valid_config()), base_dir=self.tmp)
        override = settings.with_database_path("other/x.db")
        self.assertEqual(override.database.path, (self.tmp / "other" / "x.db").resolve())
        self.assertEqual(override.simulation, settings.simulation)

    def test_environment_variable_selects_config(self) -> None:
        self.write(valid_config())
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: str(self.path)}):
            self.assertEqual(default_config_path(), self.path.resolve())
            self.assertEqual(load_config().config_path, self.path.resolve())

    def test_unknown_top_level_sections_are_tolerated(self) -> None:
        data = valid_config()
        data["navigation"] = {"planner": "astar"}  # reserved for later phases
        load_config(self.write(data))

    def test_missing_file(self) -> None:
        with self.assertRaisesRegex(ConfigError, "not found"):
            load_config(self.tmp / "nope.json")

    def test_malformed_json_reports_position(self) -> None:
        self.path.write_text('{"database": {"path": }', encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, r"not valid JSON \(line 1"):
            load_config(self.path)

    def test_invalid_values_are_rejected_with_helpful_messages(self) -> None:
        def mutate(section: str, key: str, value: Any) -> dict[str, Any]:
            data = valid_config()
            data[section][key] = value
            return data

        cases: list[tuple[Any, str]] = [
            ([], "JSON object"),
            ({"simulation": {}, "logging": {}}, "'database' section"),
            (mutate("database", "path", ""), "database.path"),
            (mutate("database", "path", 5), "database.path"),
            (mutate("simulation", "default_spawn_probability", 1.2), "default_spawn_probability"),
            (mutate("simulation", "default_spawn_probability", "0.5"), "default_spawn_probability"),
            (mutate("simulation", "default_open_seconds", -1), "default_open_seconds"),
            (mutate("simulation", "default_map", "bad map"), "map_id"),
            (mutate("logging", "level", "LOUD"), "logging.level"),
            (mutate("logging", "extra", 1), "unknown keys: extra"),
        ]
        missing_key = valid_config()
        del missing_key["simulation"]["default_open_seconds"]
        cases.append((missing_key, "missing: default_open_seconds"))

        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ConfigError) as ctx:
                    load_config(self.write(data))
                self.assertIn(message, str(ctx.exception))
                self.assertIn(str(self.path), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
