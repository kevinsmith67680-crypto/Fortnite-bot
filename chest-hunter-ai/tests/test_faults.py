"""Phase 7 robustness: controlled failure injection, detection, bounded recovery and clean
termination (short episodes on the Python reference arena)."""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from chest_hunter.agent.controller_3d import Agent3DConfig
from chest_hunter.agent.runner_3d import Runner3D
from chest_hunter.config import load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.exploration.active import ExplorationConfig
from chest_hunter.runtime.faults import (
    DETECTION_SIGNALS,
    FAULT_KINDS,
    FaultInjector,
    FaultPlan,
    Injection,
)
from chest_hunter.runtime.supervisor import SupervisorConfig

MAP = "p7_open_train_000"


def _episode(kind: str, *, start: float = 3.0, duration: float = 2.0, intensity: float = 1.0,
             ticks: int = 1500, policy: object = None) -> tuple[Runner3D, object, FaultInjector]:
    settings = load_config("config/settings.json")
    db = DatabaseManager(":memory:")
    db.initialize()
    inj = FaultInjector(FaultPlan(kind, start, duration, intensity), seed=1)
    runner = Runner3D.from_settings(
        db, settings, mode="hybrid", seed=7, map_id=MAP, backend="reference",
        collect_truth=True, max_ticks=ticks, width=320, height=180,
        config=Agent3DConfig(smart_recovery=True), exploration=ExplorationConfig(),
        supervisor=SupervisorConfig(), faults=inj,
        **({"policy": "hybrid", "learned_policy": policy} if policy is not None else {}))
    try:
        result = runner.run()
    finally:
        db.close()
    return runner, result, inj


class PlanTest(unittest.TestCase):
    def test_plans_are_deterministic_and_inside_their_window(self) -> None:
        for kind in FAULT_KINDS:
            a, b = FaultPlan.random(kind, 3), FaultPlan.random(kind, 3)
            self.assertEqual(a, b)
            self.assertGreaterEqual(a.start_s, 30.0)
            self.assertLessEqual(a.start_s, 120.0)
        with self.assertRaises(ValueError):
            FaultPlan.random("power_cut", 0)

    def test_every_kind_has_detection_signals_or_a_reason_not_to(self) -> None:
        self.assertEqual(set(DETECTION_SIGNALS), set(FAULT_KINDS))
        self.assertIsNone(DETECTION_SIGNALS["missed_detection"])


class InjectionTest(unittest.TestCase):
    def test_frame_loss_is_detected_and_recovered(self) -> None:
        runner, result, inj = _episode("frame_loss")
        a = inj.assess(result)
        self.assertTrue(a["injected"] and a["applied"])
        self.assertTrue(a["detected"])
        self.assertIn("sup:invalid_observation", a["signals"])
        self.assertTrue(a["terminated_cleanly"])
        self.assertNotEqual(result.outcome, "failed")

    def test_corrupted_frames_are_rejected_by_the_supervisor(self) -> None:
        runner, result, inj = _episode("corrupted_frame", intensity=0.3)
        a = inj.assess(result)
        self.assertTrue(a["applied"])
        self.assertTrue(a["detected"])
        self.assertNotEqual(result.outcome, "failed")

    def test_false_positives_do_not_break_the_episode(self) -> None:
        runner, result, inj = _episode("false_detection", duration=2.0)
        a = inj.assess(result)
        self.assertTrue(a["applied"])
        self.assertIsNone(a["detected"])               # indistinguishable by design
        self.assertTrue(a["terminated_cleanly"])
        self.assertEqual(result.metrics.chests_opened, result.truth.get("opened_truth"))

    def test_localisation_jump_is_detected_and_relocalised(self) -> None:
        runner, result, inj = _episode("localisation_drift", duration=0.0, intensity=2.5)
        a = inj.assess(result)
        self.assertTrue(a["applied"])
        self.assertTrue(a["detected"])
        self.assertGreaterEqual(runner.controller.agent.localisation_jumps
                                if hasattr(runner.controller, "agent")
                                else runner.controller.localisation_jumps, 1)
        self.assertTrue(a["terminated_cleanly"])

    def test_bridge_disconnection_is_survived(self) -> None:
        runner, result, inj = _episode("bridge_disconnect", duration=0.0)
        a = inj.assess(result)
        self.assertTrue(a["applied"])
        self.assertTrue(a["detected"])
        self.assertGreaterEqual(runner.client.resends, 1)
        self.assertIsNone(result.communication_failure)

    def test_unreachable_goal_is_given_up_with_bounded_queries(self) -> None:
        runner, result, inj = _episode("unreachable_target", duration=0.0, ticks=2400)
        a = inj.assess(result)
        self.assertTrue(a["applied"])
        blocked = a["details"]["effects"]["blocked a path query"]
        self.assertLessEqual(blocked, 25, "unbounded retries of an unreachable goal")
        self.assertTrue(a["terminated_cleanly"])

    def test_patches_are_removed_after_the_episode(self) -> None:
        runner, _, _ = _episode("missed_detection", duration=2.0, ticks=600)
        agent = getattr(runner.controller, "agent", runner.controller)
        self.assertNotIn("detect", vars(agent.detector))
        self.assertNotIn("step", vars(runner.client))
        self.assertNotIn("nav_query", vars(runner.client))

    def test_a_fault_after_the_episode_is_not_counted_as_tested(self) -> None:
        _, result, inj = _episode("frame_loss", start=500.0, ticks=300)
        a = inj.assess(result)
        self.assertFalse(a["injected"])


@dataclasses.dataclass
class _Out:
    latency_ms: float


class SlowInferenceTest(unittest.TestCase):
    def test_slow_inferences_are_bounded_and_flagged(self) -> None:
        class FakePolicy:
            def predict(self, frame: np.ndarray, telemetry: object) -> _Out:
                return _Out(1.0)

        policy = FakePolicy()
        inj = FaultInjector(FaultPlan("slow_inference", 0.0, 60.0, 5.0))
        agent = SimpleNamespace(now=1.0, detector=SimpleNamespace(detect=lambda *a, **k: []))
        client = SimpleNamespace(step=lambda *a, **k: None, nav_query=lambda *a, **k: None)
        runner = SimpleNamespace(client=client, controller=SimpleNamespace(agent=agent),
                                 learned_policy=policy, server=None)
        inj.attach(runner)
        inj.injection = Injection("slow_inference", 0.0, 60.0)
        outs = [policy.predict(np.zeros((2, 2, 3), np.uint8), None) for _ in range(15)]
        slowed = [o for o in outs if o.latency_ms > 1.0]
        self.assertEqual(len(slowed), 10)                 # bounded: the next 10 inferences
        self.assertEqual(slowed[0].latency_ms, 6.0)       # reported to the selector's timeout
        inj.detach()
        self.assertNotIn("predict", vars(policy))


if __name__ == "__main__":
    unittest.main()
