"""CLI commands and configuration of Phase 4 (simulate-3d, benchmark-3d, export-arena-3d)."""

from __future__ import annotations

import csv
import io
import json
import socket
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from chest_hunter.cli import main
from chest_hunter.config import load_config
from chest_hunter.exceptions import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "p4.db")
    assert _cli("--db", path, "init-db")[0] == 0
    code, out, _ = _cli("--db", path, "seed-demo", "--scenario", "research_arena_3d")
    assert code == 0 and "35" in out
    return path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_simulate_3d_reference_backend(db: str) -> None:
    code, out, err = _cli("--db", db, "--log-level", "ERROR", "simulate-3d", "--backend",
                          "reference", "--seed", "4", "--max-ticks", "900",
                          "--observation-width", "640", "--observation-height", "360",
                          "--detector", "hybrid", "--graphics-quality", "low",
                          "--headless-controller")
    assert code == 0, err
    assert "Chests opened" in out or "chests opened" in out.lower()
    code, out, _ = _cli("--db", db, "list-sessions")
    assert code == 0 and "research_arena_3d" in out


def test_simulate_3d_oracle_and_visual_nav(db: str) -> None:
    for agent in ("oracle", "visual-nav"):
        code, _, err = _cli("--db", db, "--log-level", "ERROR", "simulate-3d", "--backend",
                            "reference", "--agent", agent, "--max-ticks", "300")
        assert code == 0, err


def test_simulate_3d_without_an_unreal_server_explains_what_to_do(db: str) -> None:
    code, _, err = _cli("--db", db, "simulate-3d", "--backend", "unreal", "--port",
                        str(_free_port()), "--max-ticks", "30")
    assert code != 0
    assert "--backend reference" in err and "Traceback" not in err


def test_simulate_3d_rejects_unsupported_sizes_and_remote_hosts(db: str) -> None:
    code, _, err = _cli("--db", db, "simulate-3d", "--backend", "reference",
                        "--observation-width", "123", "--observation-height", "45",
                        "--max-ticks", "30")
    assert code != 0 and "not supported" in err
    code, _, err = _cli("--db", db, "simulate-3d", "--backend", "unreal", "--host",
                        "192.168.1.20", "--max-ticks", "30")
    assert code != 0 and "loopback" in err


def test_simulate_3d_needs_the_scenario(tmp_path: Path) -> None:
    path = str(tmp_path / "empty.db")
    _cli("--db", path, "init-db")
    code, _, err = _cli("--db", path, "simulate-3d", "--backend", "reference", "--max-ticks", "30")
    assert code != 0 and "seed-demo --scenario research_arena_3d" in err


def test_benchmark_3d_writes_reports(db: str, tmp_path: Path) -> None:
    out_dir = tmp_path / "bench"
    code, out, err = _cli("--db", db, "--log-level", "ERROR", "benchmark-3d", "--episodes", "1",
                          "--seed", "9", "--max-ticks", "600", "--modes", "oracle,hybrid",
                          "--output-dir", str(out_dir), "--quiet")
    assert code == 0, err
    summary = json.loads((out_dir / "summary_3d.json").read_text())
    assert set(summary["modes"]) == {"oracle", "hybrid"}
    with (out_dir / "navigation_3d.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["mode"] for r in rows} == {"oracle", "hybrid"} and len(rows) == 2
    assert (out_dir / "vision_3d.csv").exists()
    assert "hybrid" in out


def test_benchmark_3d_rejects_unknown_modes(db: str, tmp_path: Path) -> None:
    code, _, err = _cli("--db", db, "benchmark-3d", "--episodes", "1", "--modes", "teleport",
                        "--output-dir", str(tmp_path))
    assert code != 0 and "mode" in err


def test_export_arena_matches_the_committed_layout(tmp_path: Path) -> None:
    target = tmp_path / "arena.json"
    code, out, _ = _cli("--db", str(tmp_path / "unused.db"), "export-arena-3d", "--output",
                        str(target))
    assert code == 0 and "35 spawn points" in out
    assert not (tmp_path / "unused.db").exists()            # needs no database
    committed = ROOT / "unreal" / "ChestHunterResearch" / "Content" / "ResearchData" \
        / "research_arena_3d.json"
    assert json.loads(target.read_text()) == json.loads(committed.read_text()), \
        "run 'python -m chest_hunter export-arena-3d' to refresh the Unreal layout"


def test_config_sections_and_validation(tmp_path: Path) -> None:
    settings = load_config(ROOT / "config" / "settings.json")
    assert settings.unreal_bridge.host == "127.0.0.1"
    assert settings.unreal_bridge.port == 47820
    assert (settings.simulation_3d.observation_width,
            settings.simulation_3d.observation_height) == (640, 360)
    assert settings.template_dir_3d().is_dir()
    base = json.loads((ROOT / "config" / "settings.json").read_text())
    for patch, message in [({"unreal_bridge": {"host": "0.0.0.0"}}, "loopback"),
                           ({"simulation_3d": {"observation_width": 1000}}, "between 320 and 800"),
                           ({"simulation_3d": {"observation_width": 480}}, "not supported")]:
        cfg = tmp_path / "settings.json"
        cfg.write_text(json.dumps(_merge(base, patch)))
        with pytest.raises(ConfigError, match=message):
            load_config(cfg)


def _merge(base: dict, patch: dict) -> dict:
    out = json.loads(json.dumps(base))
    for section, values in patch.items():
        out.setdefault(section, {}).update(values)
    return out
