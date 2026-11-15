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

    def test_f19_a_reattributed_opening_is_not_credited_twice(self) -> None:
        # p7_mixed_test_003, evaluation seed, episode 14, deterministic controller with spawn
        # points: the opening was reattributed from spawn point 16 to 21 (F14), but 16 stayed
        # "unverified" and was credited again 0.1 s later while aligning on it, without an
        # interaction (8 true openings, 9 reported, 8 attempts).
        from chest_hunter.analytics.phase7_benchmark import P7Job, evaluation_seed, run_p7_job

        row = run_p7_job(P7Job("audit_policy_deterministic", "unseen_layout", "mixed",
                               "p7_mixed_test_003", evaluation_seed(42), 14,
                               str(ROOT / "config" / "settings.json"), None, None, 6000, 320,
                               180))
        self.assertEqual(row["over_reported"], 0, row)
        self.assertEqual(row["chests_opened_reported"], row["chests_opened_truth"])

    def test_f20_mixed_visual_evidence_alone_does_not_credit_an_opening(self) -> None:
        # p7_open_test_019, evaluation seed, episode 76, two injected interaction failures:
        # a failed interaction was credited on two "opened" looks against two "closed" ones,
        # with no opening notification (8 true openings, 9 reported).
        from chest_hunter.analytics.phase7_benchmark import P7Job, evaluation_seed, run_p7_job

        row = run_p7_job(P7Job("phase7_active", "fault", "open", "p7_open_test_019",
                               evaluation_seed(42), 76, str(ROOT / "config" / "settings.json"),
                               str(ROOT / "training" / "checkpoints" / "bc_synthetic_v1"),
                               str(ROOT / "training" / "exploration_memory_v1.json"), 4500, 320,
                               180, fault_kind="interaction_failure", fault_seed=76,
                               evaluate=False))
        self.assertEqual(row["over_reported"], 0, row)

    def test_f21_a_closed_looking_start_room_does_not_end_the_episode(self) -> None:
        # p7_indoor_test_001, evaluation seed, episode 5: none of the 16 regular rays left the
        # start room through its doorway, the first frontier search found nothing and the
        # episode ended as "exploration complete" after 3.9 s with no chest (18 of 800 Phase 7
        # generalisation episodes, all indoor).
        from chest_hunter.analytics.phase7_benchmark import P7Job, evaluation_seed, run_p7_job

        row = run_p7_job(P7Job("phase7_active_deterministic", "unseen_layout", "indoor",
                               "p7_indoor_test_001", evaluation_seed(42), 5,
                               str(ROOT / "config" / "settings.json"), None, None, 1500, 320,
                               180, evaluate=False))
        self.assertEqual(row["outcome"], "tick_limit", row["end_reason"])
        self.assertGreater(row["exploration_coverage"] or 0.0, 0.05)

    def test_f21_the_dense_fan_is_tried_once_per_place(self) -> None:
        from chest_hunter.exploration.active import ActiveExplorer, ExplorationConfig

        ex = ActiveExplorer(ExplorationConfig())
        bearings = ex.dense_bearings()
        self.assertEqual(len(bearings), 4 * ExplorationConfig().ray_count)
        self.assertFalse(ex.dense_fan_done((1.0, 1.0, 0.0)))
        ex.add_dense_fan((1.0, 1.0, 0.0), bearings, None)    # service failure still counts
        self.assertTrue(ex.dense_fan_done((1.0, 1.0, 0.0)))
        self.assertFalse(ex.dense_fan_done((9.0, 9.0, 0.0)))

    def test_f22_off_mesh_manoeuvres_without_a_trail_change_direction(self) -> None:
        # Phase 6 benchmark, phase6_hybrid_optimised, episode 57: the agent stood off the
        # NavMesh on a raised spot with no walked trail, so every unstick manoeuvre went
        # straight behind the camera, into the same obstacle, until the supervisor aborted
        # the episode for no progress (19 manoeuvres, all at 233 degrees).
        from types import SimpleNamespace

        from chest_hunter.agent import controller_3d as c3

        agent: Any = SimpleNamespace(_off_mesh_waits=0, _off_mesh_maneuvers=0, now=0.0,
                                     _cam_yaw=53.0, _maneuver_bearing=0.0,
                                     _maneuver_until=-1.0, _after_observe=None)
        agent._trail_bearing = lambda: None
        agent._go = lambda state: state
        bearings = []
        for _ in range(4):
            for _ in range(13):                               # 12 waits, then a manoeuvre
                c3.VisualAgent3D._wait_to_land(agent, c3.S.EXPLORING)
            bearings.append(agent._maneuver_bearing)
        self.assertEqual(bearings[0], 233.0)                  # the first attempt is unchanged
        self.assertEqual(len(set(bearings)), 4, bearings)
        agent._trail_bearing = lambda: 10.0                   # a walked trail still wins
        for _ in range(13):
            c3.VisualAgent3D._wait_to_land(agent, c3.S.EXPLORING)
        self.assertEqual(agent._maneuver_bearing, 10.0)

    @unittest.skipUnless((ROOT / "training" / "experience.db").exists(),
                         "needs the local experience database the benchmark used")
    def test_f22_the_stuck_phase6_episode_completes(self) -> None:
        from chest_hunter.analytics.phase6_benchmark import Phase6Job, run_phase6_job

        row = run_phase6_job(Phase6Job(
            "phase6_hybrid_optimised", "research_arena_3d", 42, 57,
            str(ROOT / "config" / "settings.json"), str(ROOT / "training" / "experience.db"),
            str(ROOT / "training" / "checkpoints" / "bc_synthetic_v1"), 12000, 640, 360))
        # Before the fix: tick_limit at 400 s here, and a supervisor abort at 558.6 s in the
        # benchmark's 600 s episode.
        self.assertEqual(row["outcome"], "completed", row["end_reason"])

    def test_f23_comparison_rows_from_other_code_are_not_reused(self) -> None:
        # The audit re-run after F22 resumed the policy and planner row journals of the
        # previous run and reported 320 rows played by the code before the fix.
        from unittest import mock

        from chest_hunter.audit import system

        class Stop(Exception):
            pass

        calls: list[bool] = []

        def fake(settings: Any, **kw: Any) -> Any:
            calls.append(kw["resume"])
            raise Stop

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            audit = system.Audit(_settings(), out, episodes=1, workers=1, held_out=True,
                                 profile=False, verify_db=False, model_dir=None)
            (out / "policy_comparison").mkdir()
            (out / "policy_comparison" / "source_version.txt").write_text("v1\n")
            patch_run = mock.patch(
                "chest_hunter.analytics.phase7_benchmark.run_strategy_comparison", fake)
            for version in ("v2", "v2", None, None):
                with mock.patch.object(system, "_source_version", return_value=version), \
                        patch_run, self.assertRaises(Stop):
                    audit._comparison(("phase7_active",), "policy_comparison")
        # other code: fresh; same code: resumed; unidentifiable code: never resumed
        self.assertEqual(calls, [False, True, False, False])

    def test_f2_a_phase7_map_can_be_exported_for_unreal(self) -> None:
        # Before the fix export-arena-3d exported only research_arena_3d; the Phase 7 maps
        # (including the acceptance map) could not reach the Unreal project at all.
        import io
        import json
        from contextlib import redirect_stderr, redirect_stdout

        from chest_hunter.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "map.json"
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main(["--db", str(Path(tmp) / "unused.db"), "export-arena-3d",
                             "--map", "p7_complex_test_000_unfamiliar", "--output",
                             str(target)])
            self.assertEqual(code, 0, err.getvalue())
            committed = ROOT / "unreal" / "ChestHunterResearch" / "Content" / \
                "ResearchData" / "p7_complex_test_000_unfamiliar.json"
            self.assertEqual(json.loads(target.read_text()), json.loads(committed.read_text()))

    def test_f1_only_service_errors_are_absorbed(self) -> None:
        # Two handlers caught every Exception from navigation-service calls and recorded
        # nothing; now only BridgeError is absorbed (and counted), a programming error
        # propagates.
        from types import SimpleNamespace

        from chest_hunter.agent.controller_3d import VisualAgent3D
        from chest_hunter.unreal_bridge.client import BridgeError

        def agent(exc: BaseException) -> Any:
            def fail(*_: Any, **__: Any) -> bool:
                raise exc
            return SimpleNamespace(navigator=SimpleNamespace(clear_line=fail),
                                   position=(0.0, 0.0, 0.0),
                                   metrics=SimpleNamespace(service_errors=0))

        prior = SimpleNamespace(xyz=(1.0, 1.0, 0.0))
        a = agent(BridgeError("unavailable", "navigation service down"))
        VisualAgent3D._clear_line(a, prior)  # type: ignore[arg-type]
        self.assertEqual(a.metrics.service_errors, 1)
        with self.assertRaises(ZeroDivisionError):
            VisualAgent3D._clear_line(agent(ZeroDivisionError()), prior)  # type: ignore[arg-type]

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

    def test_verification_status_follows_the_measurement(self) -> None:
        from chest_hunter.audit.components import assess_components

        def status(inter: dict[str, Any]) -> str:
            return next(c["status"] for c in assess_components({"inter": inter})
                        if c["id"] == 18)

        def clean(fsr: float, ffr: float) -> dict[str, Any]:
            return {"agent_crosscheck": {"per_scenario": {"clean": {
                "false_success_rate": fsr, "false_failure_rate": ffr}}}}

        self.assertEqual(status({}), "UNVERIFIED")
        self.assertEqual(status(clean(0.0, 0.0)), "PARTIALLY VERIFIED")
        self.assertEqual(status(clean(0.0, 0.06)), "BROKEN")
        self.assertEqual(status(clean(0.01, 0.0)), "BROKEN")

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
