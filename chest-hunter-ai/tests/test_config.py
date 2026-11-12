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
        data["vision"] = {"detector": "none"}  # reserved for later phases
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



class Phase2ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "settings.json"

    def write(self, data: Any) -> Path:
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return self.path

    def test_phase1_files_get_phase2_defaults(self) -> None:
        settings = load_config(self.write(valid_config()))
        sim, nav, agent = settings.simulation, settings.navigation, settings.agent
        self.assertEqual((sim.map_width, sim.map_height, sim.cell_size), (100, 100, 8))
        self.assertEqual((sim.tick_rate, sim.render_rate, sim.default_seed), (30, 15, 42))
        self.assertEqual((sim.player_speed, sim.player_acceleration), (5.0, 12.0))
        self.assertEqual((sim.interaction_radius, sim.chest_interaction_seconds), (1.5, 2.0))
        self.assertIsNone(sim.agent_start)
        self.assertAlmostEqual(sim.tick_seconds, 1 / 30)
        self.assertEqual(nav.default_strategy, "weighted")
        self.assertTrue(nav.allow_diagonal and nav.prevent_corner_cutting and nav.cache_paths)
        self.assertEqual((agent.max_recovery_attempts, agent.arrival_tolerance), (3, 0.5))

    def test_shipped_settings_include_phase2_sections(self) -> None:
        settings = load_config(PROJECT_ROOT / "config" / "settings.json")
        self.assertEqual(settings.simulation.max_ticks, 100_000)
        self.assertEqual(settings.navigation.stuck_timeout_seconds, 5.0)
        self.assertEqual(settings.agent.observation_batch_size, 10)

    def test_phase2_values_are_validated(self) -> None:
        cases = [
            ("simulation", "map_width", 5, "map_width"),
            ("simulation", "tick_rate", 0, "tick_rate"),
            ("simulation", "player_speed", -1, "player_speed"),
            ("simulation", "player_radius", 0.5, "player_radius"),
            ("simulation", "agent_start", [1, "x"], "agent_start"),
            ("simulation", "reward_probability", 1.1, "reward_probability"),
            ("simulation", "obstacle_density", 0.9, "obstacle_density"),
            ("navigation", "default_strategy", "random", "default_strategy"),
            ("navigation", "allow_diagonal", "yes", "allow_diagonal"),
            ("navigation", "lookahead_depth", 4, "lookahead_depth"),
            ("navigation", "typo_key", 1, "unknown keys: typo_key"),
            ("agent", "max_recovery_attempts", -1, "max_recovery_attempts"),
            ("agent", "observation_batch_size", 0, "observation_batch_size"),
        ]
        for section, key, value, message in cases:
            with self.subTest(key=key):
                data = valid_config()
                data.setdefault(section, {})[key] = value
                with self.assertRaises(ConfigError) as ctx:
                    load_config(self.write(data))
                self.assertIn(message, str(ctx.exception))

    def test_cross_field_and_point_parsing(self) -> None:
        data = valid_config()
        data["simulation"]["agent_start"] = [120, 340.5]
        data["navigation"] = {"default_strategy": "Nearest"}
        settings = load_config(self.write(data))
        self.assertEqual(settings.simulation.agent_start, (120.0, 340.5))
        self.assertEqual(settings.navigation.default_strategy, "nearest")
        data["agent"] = {"arrival_tolerance": 2.0}
        with self.assertRaisesRegex(ConfigError, "arrival_tolerance"):
            load_config(self.write(data))


if __name__ == "__main__":
    unittest.main()
