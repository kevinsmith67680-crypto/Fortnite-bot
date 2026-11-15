"""Phase 5 CLI, configuration, schema migration, teleoperation and the end-to-end
acceptance workflow (record -> dataset -> train -> export -> CPU inference -> episodes with
every strategy -> unseen map -> exported results). All demonstrations here are SYNTHETIC
(scripted controller); no human data is fabricated."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from chest_hunter.analytics.learning_metrics import progress_table, training_report
from chest_hunter.analytics.policy_comparison import (
    PRIMARY_METRIC,
    STRATEGIES,
    format_report,
    run_policy_benchmark,
    summarise_rows,
)
from chest_hunter.cli import main
from chest_hunter.config import load_config
from chest_hunter.database import MIGRATIONS, SCHEMA_VERSION, DatabaseManager
from chest_hunter.exceptions import ConfigError
from chest_hunter.learning.learned_policy import LearnedPolicy
from tests.phase5_helpers import ROOT, onnx_available, settings, tiny_dataset, tiny_model


def _cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class Configuration(unittest.TestCase):
    def test_learning_section_defaults_and_validation(self) -> None:
        s = settings()
        self.assertEqual(s.learning.policy, "deterministic")
        self.assertFalse(s.learning.adaptive_planning)
        base = json.loads((ROOT / "config" / "settings.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for patch in ({"policy": "neural-magic"}, {"inference_timeout_ms": -1},
                          {"unknown_key": 1}):
                cfg = Path(tmp) / "settings.json"
                data = json.loads(json.dumps(base))
                data["learning"].update(patch)
                cfg.write_text(json.dumps(data))
                with self.subTest(patch=patch), self.assertRaises(ConfigError):
                    load_config(cfg)


class SchemaMigration(unittest.TestCase):
    def test_version_3_database_is_migrated_to_4_without_data_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v3.db"
            db = DatabaseManager(path)
            with db.transaction() as conn:
                for _v, statements in MIGRATIONS[:3]:
                    for statement in statements:
                        conn.execute(statement)
                conn.execute("PRAGMA user_version = 3")
                conn.execute("INSERT INTO chest_locations (map_id, location_name, x, y, z, "
                             "spawn_probability, estimated_open_seconds, enabled, created_at, "
                             "updated_at) VALUES ('m', 'A', 1, 2, 0, 0.4, 2, 1, 't', 't')")
            db.close()
            db = DatabaseManager(path)
            self.assertEqual(db.initialize(), SCHEMA_VERSION)
            self.assertEqual(SCHEMA_VERSION, 4)
            tables = {r["name"] for r in db.query_all(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            self.assertTrue({"spawn_estimates", "route_memory"} <= tables)
            row = db.query_all("SELECT spawn_probability FROM chest_locations")[0]
            self.assertAlmostEqual(row["spawn_probability"], 0.4)
            db.close()


class CommandLine(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.db = str(self.dir / "main.db")
        _cli("--db", self.db, "init-db")
        _cli("--db", self.db, "seed-demo", "--scenario", "research_arena_3d")

    def test_generate_maps_and_estimate_spawns(self) -> None:
        code, out, err = _cli("--db", self.db, "generate-maps", "--split", "test", "--count",
                              "2", "--export-dir", str(self.dir / "layouts"))
        self.assertEqual(code, 0, err)
        self.assertIn("proc_test_001", out)
        self.assertTrue((self.dir / "layouts" / "proc_test_000.json").exists())
        code, out, err = _cli("--db", self.db, "estimate-spawns", "--map", "proc_test_000",
                              "--store")
        self.assertEqual(code, 0, err)
        self.assertIn("Stored", out)
        code, _, err = _cli("--db", self.db, "estimate-spawns", "--map", "nowhere")
        self.assertNotEqual(code, 0)

    def test_record_demo_scripted_and_seed_warning(self) -> None:
        out_dir = self.dir / "demos"
        code, out, err = _cli("--db", self.db, "--log-level", "ERROR", "record-demo",
                              "--map", "proc_train_002", "--episodes", "1", "--seed", "42",
                              "--max-ticks", "120", "--output", str(out_dir),
                              "--experience-db", str(self.dir / "exp.db"))
        self.assertEqual(code, 0, err)
        self.assertIn("SYNTHETIC", out)
        self.assertIn("warning", out)
        meta = json.loads(next(out_dir.glob("episode_*/episode.json")).read_text())
        self.assertTrue(meta["synthetic"])

    def test_train_export_simulate_and_benchmark(self) -> None:
        data = self.dir / "data"
        shutil.copytree(tiny_dataset(), data)
        model = self.dir / "model"
        args = ["--db", self.db, "--log-level", "ERROR", "train-policy", "--dataset", str(data),
                "--output", str(model), "--epochs", "1", "--batch-size", "16"]
        if not onnx_available():
            args.append("--no-export")
        code, out, err = _cli(*args)
        self.assertEqual(code, 0, err)
        self.assertIn("SYNTHETIC", out)
        self.assertTrue((model / "policy.npz").exists())
        if onnx_available():
            self.assertTrue((model / "policy_int8.onnx").exists())
        for extra in (["--policy", "learned"],
                      ["--policy", "hybrid", "--adaptive-planning", "--smart-recovery"]):
            code, out, err = _cli("--db", self.db, "--log-level", "ERROR", "simulate-3d",
                                  "--backend", "reference", "--max-ticks", "240",
                                  "--model", str(model), *extra)
            self.assertEqual(code, 0, err)
            self.assertIn("Learned decisions", out)
        code, out, err = _cli("--db", self.db, "--log-level", "ERROR", "benchmark-policies",
                              "--episodes", "1", "--max-ticks", "240", "--model", str(model),
                              "--strategies", "deterministic,learned",
                              "--output-dir", str(self.dir / "bench"),
                              "--experience-db", str(self.dir / "none.db"))
        self.assertEqual(code, 0, err)
        for name in ("episodes.csv", "summary.json", "report.md"):
            self.assertTrue((self.dir / "bench" / name).exists(), name)

    def test_helpful_errors(self) -> None:
        code, _, err = _cli("--db", self.db, "simulate-3d", "--backend", "reference",
                            "--policy", "learned", "--model", str(self.dir / "missing"),
                            "--max-ticks", "30")
        self.assertNotEqual(code, 0)
        self.assertIn("train-policy", err)
        code, _, err = _cli("--db", self.db, "benchmark-policies", "--strategies", "magic",
                            "--episodes", "1")
        self.assertNotEqual(code, 0)


@unittest.skipUnless(os.environ.get("DISPLAY"), "no display (run under xvfb-run)")
class Teleoperation(unittest.TestCase):
    def test_keys_drive_the_character_and_the_recording_replays(self) -> None:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            self.skipTest("this Python build has no Tk")
        from chest_hunter.learning.demonstration_recorder import (
            DemonstrationRecorder,
            load_episode,
            replay_episode,
        )
        from chest_hunter.learning.teleop import TeleopSession, action_from_keys
        from chest_hunter.unreal_bridge.client import BridgeClient
        from chest_hunter.unreal_bridge.mock_server import MockUnrealServer

        self.assertEqual(action_from_keys({"forward", "turn_right"})["move_forward"], 1.0)

        class Key:
            def __init__(self, keysym: str) -> None:
                self.keysym = keysym

        with tempfile.TemporaryDirectory() as tmp, MockUnrealServer() as server, \
                BridgeClient(port=server.port, connect_attempts=1) as client:
            rec = DemonstrationRecorder(tmp)
            session = TeleopSession(client, rec, episode_id=0, map_id="research_arena_3d",
                                    seed=5, width=320, height=180, realtime=False)
            session._down(Key("w"))
            for _ in range(10):
                session.step()
            session._down(Key("p"))                    # pause: the environment is not stepped
            session.step()
            session._down(Key("p"))
            session._up(Key("w"))
            session._down(Key("Right"))
            for _ in range(5):
                session.step()
            session._down(Key("Escape"))
            directory = next(Path(tmp).glob("episode_*"))
            meta, arrays = load_episode(directory)
            self.assertEqual(meta["source"], "human")
            self.assertEqual(meta["steps"], 15)
            self.assertTrue(meta["replayable"])
            self.assertEqual(float(arrays["action"][0, 0]), 1.0)
            self.assertEqual(float(arrays["action"][-1, 2]), 1.0)
            self.assertTrue(replay_episode(directory, client).exact)
            session.close()
            self.assertIsNone(session.root)
            session.close()                            # idempotent


class LearningDashboard(unittest.TestCase):
    def _runner(self, policy: str, hook: object) -> object:
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.scenarios import seed_arena_3d

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        return Runner3D.from_settings(db, settings(), mode="hybrid", seed=3, backend="reference",
                                      max_ticks=150, width=320, height=180, policy=policy,
                                      learned_policy=LearnedPolicy(tiny_model(), backend="numpy"),
                                      on_decision=hook)

    def test_learning_panel_shows_policy_model_and_run_statistics(self) -> None:
        from chest_hunter.agent.dashboard_3d import learning_text

        texts: list[str] = []
        runner = self._runner("hybrid", lambda r, _res: texts.append(learning_text(r)))
        runner.run()                                              # type: ignore[attr-defined]
        text = texts[-1]
        for expected in ("Policy: hybrid", "Model:", "(synthetic)", "Throughput:",
                         "Navigation success", "Learned decisions"):
            self.assertIn(expected, text)

    @unittest.skipUnless(os.environ.get("DISPLAY"), "no display (run under xvfb-run)")
    def test_policy_can_be_switched_from_the_window(self) -> None:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            self.skipTest("this Python build has no Tk")
        from chest_hunter.agent.dashboard_3d import Dashboard3D

        dashboard = Dashboard3D(refresh_seconds=0.0)
        self.addCleanup(dashboard.close)
        modes: list[str] = []

        def hook(runner: object, result: object) -> None:
            dashboard.update(runner, result)
            if not modes:
                dashboard.policy_choice.set("learned")
                dashboard._switch()
            modes.append(runner.selector.mode)                    # type: ignore[attr-defined]

        self._runner("deterministic", hook).run()                 # type: ignore[attr-defined]
        self.assertEqual(modes[0], "learned")
        self.assertIn("Policy: learned", dashboard.learning.get())


def _row(strategy: str, episode: int, reported: int, truth: int, minutes: float = 2.0) -> dict:
    return {"map_id": "m", "episode": episode, "strategy": strategy, "outcome": "completed",
            "end_reason": "done", "chests_opened": reported, "chests_opened_truth": truth,
            "over_reported": max(0, reported - truth), "under_reported": max(0, truth - reported),
            "throughput_truth_per_min": truth / minutes, "throughput_per_min": reported / minutes,
            "percent_available_opened": 50.0, "discovery_rate": 1.0,
            "navigation_success_rate": 1.0, "interaction_success_rate": 1.0,
            "exploration_coverage": 1.0, "travel_distance_m": 10.0, "travel_per_chest_m": 5.0,
            "recoveries": 0, "sim_seconds": 60 * minutes, "inference_mean_ms": None,
            "learned_share": 0.0, "cpu_seconds": 1.0, "peak_rss_mb": 100.0,
            "credited_by_notification": 0, "recovery_classes": "{}", "fallbacks": "{}"}


class PolicySummary(unittest.TestCase):
    """Ground truth is the primary metric: counting differences must not look like gains."""

    def test_truth_and_reported_throughput_are_compared_separately(self) -> None:
        rows = []
        for e in range(4):
            rows.append(_row("deterministic", e, reported=4, truth=6))   # misses 2 openings
            rows.append(_row("hybrid", e, reported=6, truth=6))          # counts them all
        summary = summarise_rows(rows, ["deterministic", "hybrid"])
        self.assertEqual(summary["primary_metric"], PRIMARY_METRIC)
        paired = summary["paired_vs_deterministic"]["hybrid"]
        self.assertEqual(paired[PRIMARY_METRIC]["difference"]["mean"], 0.0)
        self.assertEqual(paired[PRIMARY_METRIC]["tied"], 4)
        self.assertFalse(paired[PRIMARY_METRIC]["significant_at_95"])
        self.assertEqual(paired["throughput_per_min"]["difference"]["mean"], 1.0)
        self.assertEqual(paired["throughput_per_min"]["better"], 4)
        det = summary["per_strategy"]["deterministic"]
        self.assertEqual(det["under_reported_total"], 8)
        self.assertEqual(det["over_reported_total"], 0)
        report = format_report(summary)
        self.assertIn("chests opened (ground truth)", report)
        self.assertIn("chests counted by the agent", report)


class AcceptanceWorkflow(unittest.TestCase):
    """Spec section 20 on the reference arena with tiny settings (SYNTHETIC demonstrations)."""

    def test_end_to_end(self) -> None:
        dataset = tiny_dataset()                                  # 1-3: record and build
        model = tiny_model()                                      # 4-5: train and export
        report = training_report(model)
        self.assertTrue(report["dataset"]["synthetic_only"])
        self.assertGreater(report["dataset"]["samples"], 0)
        backend = "onnxruntime" if onnx_available() else "numpy"
        policy = LearnedPolicy(model, backend=backend)            # 6: CPU inference
        import numpy as np
        out = policy.predict(np.zeros((360, 640, 3), np.uint8), None)
        self.assertTrue(out.finite())
        self.assertLess(policy.mean_ms, 200.0)
        with tempfile.TemporaryDirectory() as tmp:               # 7-11: strategies compared
            summary = run_policy_benchmark(
                settings(), strategies=tuple(STRATEGIES), map_ids=("research_arena_3d",),
                seed=42, episodes=1, model_dir=model, experience_db=None, max_ticks=300,
                width=320, height=180, output_dir=Path(tmp) / "familiar")
            self.assertEqual(set(summary["per_strategy"]), set(STRATEGIES))
            self.assertEqual(summary["errors"], [])
            self.assertEqual(summary["primary_metric"], PRIMARY_METRIC)
            for cmp in summary["paired_vs_deterministic"].values():
                self.assertEqual(set(cmp), {PRIMARY_METRIC, "throughput_per_min"})
            unseen = run_policy_benchmark(                        # 12: unseen map
                settings(), strategies=("deterministic", "hybrid_adaptive"),
                map_ids=("proc_test_004",), seed=5000, episodes=2, model_dir=model,
                max_ticks=300, width=320, height=180, online=True,
                output_dir=Path(tmp) / "unseen")
            rows = list((Path(tmp) / "unseen" / "episodes.csv").read_text().splitlines())
            self.assertEqual(len(rows), 1 + 4)                    # 13: results exported
            self.assertEqual(unseen["episodes_completed"], 2)
            for entry in summary["per_strategy"].values():
                self.assertEqual(entry["over_reported_total"], 0)
        self.assertTrue(dataset.exists())
        self.assertEqual(progress_table([]), [])


if __name__ == "__main__":
    unittest.main()
