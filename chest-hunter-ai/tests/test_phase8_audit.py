"""Phase 8 audit: regression tests for the confirmed defects and tests of the audit tools.

Each defect test replays the exact episode that reproduced the defect during the audit.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from chest_hunter.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _settings() -> Any:
    return load_config(ROOT / "config" / "settings.json")


class DefectRegressions(unittest.TestCase):
    def test_f11_notification_confirmed_openings_are_credited(self) -> None:
        # research_arena_3d seed 21 episode 0: before the fix, 2 of 6 real openings were
        # judged "closed" in every look and never credited (the agent re-aligned on opened
        # chests and marked them failed).
        from chest_hunter.audit.interaction_eval import InteractionJob, run_interaction_job

        row = run_interaction_job(InteractionJob("clean", 0, 21, str(ROOT / "config" /
                                                                     "settings.json"), 2400))
        self.assertEqual(row["false_success"], 0)
        self.assertEqual(row["false_failure"], 0, row)
        # after the bounded look-back the opening is seen (or credited on the notification)
        self.assertEqual(row["credited"], row["truth_opened"] - row["truncated_by_episode_end"])

    def test_f14_an_opening_is_credited_to_the_spawn_point_reached(self) -> None:
        # seed 21 episode 8 with three failed interactions: a misplaced track had been
        # associated with a spawn point 4.8 m away; the agent opened the chest it stood at
        # and credited the other one (one false success, one false failure).
        from chest_hunter.audit.interaction_eval import InteractionJob, run_interaction_job

        row = run_interaction_job(InteractionJob("failed_interactions", 8, 21,
                                                 str(ROOT / "config" / "settings.json"), 3600))
        self.assertEqual(row["false_success"], 0, row)

    def test_f12_discovered_openings_are_persisted(self) -> None:
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.database import DatabaseManager
        from chest_hunter.exploration.active import ExplorationConfig

        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(Path(tmp) / "x.db")
            db.initialize()
            try:
                result = Runner3D.from_settings(
                    db, _settings(), mode="hybrid", seed=31, episode=7, backend="reference",
                    map_id="p7_open_test_004", collect_truth=True, max_ticks=2400, width=320,
                    height=180, exploration=ExplorationConfig()).run()
                rows = db.query_all("SELECT * FROM discovered_openings WHERE session_id = ?",
                                    (result.session_id,))
            finally:
                db.close()
        self.assertGreater(result.metrics.chests_opened, 0)
        self.assertEqual(len(rows), result.metrics.chests_opened)

    def test_f12_openings_only_for_running_sessions(self) -> None:
        import sqlite3

        from chest_hunter.database import DatabaseManager
        from chest_hunter.session_repository import SessionRepository

        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(Path(tmp) / "x.db")
            db.initialize()
            try:
                repo = SessionRepository(db)
                s = repo.start_session("p7_open_test_004")
                repo.complete_session(s.id)
                with self.assertRaises(Exception) as ctx, db.transaction() as conn:
                    conn.execute("INSERT INTO discovered_openings (session_id, x, y, z, "
                                 "observed_at) VALUES (?, 1.0, 2.0, 0.0, '2030-01-01')",
                                 (s.id,))
                self.assertIsInstance(ctx.exception.__cause__ or ctx.exception,
                                      (sqlite3.Error, Exception))
            finally:
                db.close()


    def test_f17_exploration_progress_clears_the_loop_window(self) -> None:
        # p7_indoor_test_007, evaluation seed, episode 29: 11 false "transition loop"
        # detections and 6 interventions in 150 fault-free seconds before the fix.
        import json

        from chest_hunter.analytics.phase7_benchmark import P7Job, evaluation_seed, run_p7_job

        row = run_p7_job(P7Job("phase7_active_deterministic", "baseline", "indoor",
                               "p7_indoor_test_007", evaluation_seed(42), 29,
                               str(ROOT / "config" / "settings.json"), None, None, 4500, 320,
                               180, evaluate=False))
        loops = json.loads(row["supervisor_detections"] or "{}").get("transition_loop", 0)
        self.assertLessEqual(loops, 5, row["supervisor_detections"])
        self.assertNotEqual(row["outcome"], "failed")

    def test_f17_loop_window_cleared_by_new_coverage(self) -> None:
        from types import SimpleNamespace

        from chest_hunter.runtime.supervisor import (
            EXPLORATION_PROGRESS_CELLS,
            Supervisor,
            SupervisorConfig,
        )

        sup = Supervisor(SupervisorConfig())
        explorer = SimpleNamespace(map=SimpleNamespace(seen_count=0))
        sup.agent = SimpleNamespace(  # type: ignore[assignment]
            metrics=SimpleNamespace(chests_opened=0, targets_reached=0), world=SimpleNamespace(
                priors={}), explorer=explorer, _known_targets=set(), now=0.0)
        for state in ("NAVIGATING", "SCANNING") * 3:
            sup.loops.observe(state)
        self.assertIsNotNone(sup.loops.detect())
        explorer.map.seen_count = EXPLORATION_PROGRESS_CELLS
        sup._check_progress(1.0)
        self.assertIsNone(sup.loops.detect())


class AuditTools(unittest.TestCase):
    def test_static_scan_reports_with_evidence(self) -> None:
        from chest_hunter.audit.static_scan import scan

        res = scan(ROOT)
        self.assertGreater(res.files, 100)
        for f in res.findings:
            self.assertRegex(f.where, r"\S+")
        self.assertFalse(res.of("broad_except:silent"))

    def test_ground_truth_isolation_static_checks(self) -> None:
        from chest_hunter.audit.leakage import static_checks

        checks = static_checks(ROOT / "src")
        self.assertTrue(all(c["pass"] for c in checks), [c for c in checks if not c["pass"]])

    def test_bridge_security_checks_pass(self) -> None:
        from chest_hunter.audit.bridge_security import evaluate

        res = evaluate()
        self.assertEqual(res["passed"], res["total"],
                         [c for c in res["checks"] if not c["pass"]])

    def test_environment_interaction_rules(self) -> None:
        from chest_hunter.audit.interaction_eval import environment_rules

        cases = environment_rules()
        self.assertGreaterEqual(len(cases), 6)
        self.assertTrue(all(c["pass"] for c in cases), [c for c in cases if not c["pass"]])

    def test_readiness_cannot_be_ready_without_unreal(self) -> None:
        from chest_hunter.audit.components import COMPONENTS, WEIGHTS, readiness

        self.assertEqual(len(COMPONENTS), 25)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)
        perfect = {
            "trace": {"stages": {"a": {"exercised": True}}, "outcome": "completed"},
            "inter": {"environment_rules": [{"pass": True}], "agent_crosscheck": {
                "per_scenario": {"clean": {"false_success_rate": 0.0,
                                           "false_failure_rate": 0.0}}}},
            "database": {"checks": [{"check": c, "pass": True} for c in (
                "no duplicated openings", "no recorded opening that did not happen",
                "integrity_check ok", "no foreign-key violations")]},
            "endurance": {"runs": [{"memory_bounded": True}]},
            "unreal": {"unreal_engine": {"installed": False}, "maps_present": []}}
        r = readiness(perfect)
        self.assertEqual(r["verdict"], "UNVERIFIED")
        self.assertFalse(r["gates"]["actual Unreal integration verified"])
        perfect["unreal"] = {"unreal_engine": {"installed": True}, "maps_present": ["m"]}
        self.assertEqual(readiness(perfect)["verdict"], "READY FOR CONTROLLED RESEARCH USE")
        perfect["inter"]["agent_crosscheck"]["per_scenario"]["clean"]["false_failure_rate"] \
            = 0.2
        self.assertEqual(readiness(perfect)["verdict"], "NOT READY")

    def test_vision_harness_on_a_small_dataset(self) -> None:
        from chest_hunter.audit.vision_eval import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            res = evaluate(_settings(), maps=("research_arena_3d",), sizes=((320, 180),),
                           per_chest=2, negatives=3, failures_dir=Path(tmp))
        tally = res["per_resolution"]["320x180"]
        self.assertGreater(tally["frames"], 5)
        self.assertIsNotNone(tally["latency_mean_ms"])


if __name__ == "__main__":
    unittest.main()
