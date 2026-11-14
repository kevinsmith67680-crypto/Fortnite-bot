"""CLI commands and configuration of Phase 4 (simulate-3d, benchmark-3d, export-arena-3d)."""

from __future__ import annotations

import csv
import io
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from chest_hunter.cli import main
from chest_hunter.config import load_config
from chest_hunter.exceptions import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class TempDirTestCase(unittest.TestCase):
    """Each test gets a fresh temporary directory, ``self.tmp_path``, removed afterwards."""

    def setUp(self) -> None:
        self.tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))


class TestCommandsOnSeededDatabase(TempDirTestCase):
    """Each test gets a database in ``self.tmp_path`` seeded with the 3D arena (``self.db``)."""

    def setUp(self) -> None:
        super().setUp()
        path = str(self.tmp_path / "p4.db")
        self.assertEqual(_cli("--db", path, "init-db")[0], 0)
        code, out, _ = _cli("--db", path, "seed-demo", "--scenario", "research_arena_3d")
        self.assertEqual(code, 0)
        self.assertIn("35", out)
        self.db = path

    def test_simulate_3d_reference_backend(self) -> None:
        code, out, err = _cli("--db", self.db, "--log-level", "ERROR", "simulate-3d",
                              "--backend", "reference", "--seed", "4", "--max-ticks", "900",
                              "--observation-width", "640", "--observation-height", "360",
                              "--detector", "hybrid", "--graphics-quality", "low",
                              "--headless-controller")
        self.assertEqual(code, 0, err)
        self.assertTrue("Chests opened" in out or "chests opened" in out.lower())
        code, out, _ = _cli("--db", self.db, "list-sessions")
        self.assertEqual(code, 0)
        self.assertIn("research_arena_3d", out)

    def test_simulate_3d_oracle_and_visual_nav(self) -> None:
        for agent in ("oracle", "visual-nav"):
            code, _, err = _cli("--db", self.db, "--log-level", "ERROR", "simulate-3d",
                                "--backend", "reference", "--agent", agent, "--max-ticks", "300")
            self.assertEqual(code, 0, err)

    def test_simulate_3d_without_an_unreal_server_explains_what_to_do(self) -> None:
        code, _, err = _cli("--db", self.db, "simulate-3d", "--backend", "unreal", "--port",
                            str(_free_port()), "--max-ticks", "30")
        self.assertNotEqual(code, 0)
        self.assertIn("--backend reference", err)
        self.assertNotIn("Traceback", err)

    def test_simulate_3d_rejects_unsupported_sizes_and_remote_hosts(self) -> None:
        code, _, err = _cli("--db", self.db, "simulate-3d", "--backend", "reference",
                            "--observation-width", "123", "--observation-height", "45",
                            "--max-ticks", "30")
        self.assertNotEqual(code, 0)
        self.assertIn("not supported", err)
        code, _, err = _cli("--db", self.db, "simulate-3d", "--backend", "unreal", "--host",
                            "192.168.1.20", "--max-ticks", "30")
        self.assertNotEqual(code, 0)
        self.assertIn("loopback", err)

    def test_benchmark_3d_writes_reports(self) -> None:
        out_dir = self.tmp_path / "bench"
        code, out, err = _cli("--db", self.db, "--log-level", "ERROR", "benchmark-3d",
                              "--episodes", "1", "--seed", "9", "--max-ticks", "600",
                              "--modes", "oracle,hybrid", "--output-dir", str(out_dir), "--quiet")
        self.assertEqual(code, 0, err)
        summary = json.loads((out_dir / "summary_3d.json").read_text())
        self.assertEqual(set(summary["modes"]), {"oracle", "hybrid"})
        with (out_dir / "navigation_3d.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({r["mode"] for r in rows}, {"oracle", "hybrid"})
        self.assertEqual(len(rows), 2)
        self.assertTrue((out_dir / "vision_3d.csv").exists())
        self.assertIn("hybrid", out)

    def test_benchmark_3d_rejects_unknown_modes(self) -> None:
        code, _, err = _cli("--db", self.db, "benchmark-3d", "--episodes", "1", "--modes",
                            "teleport", "--output-dir", str(self.tmp_path))
        self.assertNotEqual(code, 0)
        self.assertIn("mode", err)


class TestCommandsWithoutSeededDatabase(TempDirTestCase):
    def test_simulate_3d_needs_the_scenario(self) -> None:
        path = str(self.tmp_path / "empty.db")
        _cli("--db", path, "init-db")
        code, _, err = _cli("--db", path, "simulate-3d", "--backend", "reference",
                            "--max-ticks", "30")
        self.assertNotEqual(code, 0)
        self.assertIn("seed-demo --scenario research_arena_3d", err)

    def test_export_arena_matches_the_committed_layout(self) -> None:
        target = self.tmp_path / "arena.json"
        code, out, _ = _cli("--db", str(self.tmp_path / "unused.db"), "export-arena-3d",
                            "--output", str(target))
        self.assertEqual(code, 0)
        self.assertIn("35 spawn points", out)
        self.assertFalse((self.tmp_path / "unused.db").exists())      # needs no database
        committed = ROOT / "unreal" / "ChestHunterResearch" / "Content" / "ResearchData" \
            / "research_arena_3d.json"
        self.assertEqual(
            json.loads(target.read_text()), json.loads(committed.read_text()),
            "run 'python -m chest_hunter export-arena-3d' to refresh the Unreal layout")

    def test_config_sections_and_validation(self) -> None:
        settings = load_config(ROOT / "config" / "settings.json")
        self.assertEqual(settings.unreal_bridge.host, "127.0.0.1")
        self.assertEqual(settings.unreal_bridge.port, 47820)
        self.assertEqual((settings.simulation_3d.observation_width,
                          settings.simulation_3d.observation_height), (640, 360))
        self.assertTrue(settings.template_dir_3d().is_dir())
        base = json.loads((ROOT / "config" / "settings.json").read_text())
        for patch, message in [({"unreal_bridge": {"host": "0.0.0.0"}}, "loopback"),
                               ({"simulation_3d": {"observation_width": 1000}},
                                "between 320 and 800"),
                               ({"simulation_3d": {"observation_width": 480}}, "not supported")]:
            cfg = self.tmp_path / "settings.json"
            cfg.write_text(json.dumps(_merge(base, patch)))
            with self.assertRaisesRegex(ConfigError, message):
                load_config(cfg)


def _merge(base: dict, patch: dict) -> dict:
    out = json.loads(json.dumps(base))
    for section, values in patch.items():
        out.setdefault(section, {}).update(values)
    return out
