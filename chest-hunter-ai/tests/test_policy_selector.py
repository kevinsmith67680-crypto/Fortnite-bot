"""Policy selection and the safety layer around the learned policy."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chest_hunter.agent.controller_3d import Agent3DState
from chest_hunter.learning.learned_policy import PolicyOutput
from chest_hunter.learning.policy_selector import PolicySelector, SafetyLimits
from chest_hunter.localization.world_model_3d import PriorStatus3D
from chest_hunter.unreal_bridge.action_adapter import ActionCommand
from chest_hunter.vision.interaction_cues_3d import Cues

PHASES_UNIFORM = (1 / 7,) * 7


def output(**kw: Any) -> PolicyOutput:
    base = dict(move_forward=0.5, move_right=0.0, look_yaw=0.1, look_pitch=0.0, jump_prob=0.0,
                interact_prob=0.0, phase_probs=PHASES_UNIFORM, latency_ms=1.0)
    return PolicyOutput(**{**base, **kw})


def approach_phases(p: float = 0.9) -> tuple[float, ...]:
    rest = (1 - p) / 6
    probs = [rest] * 7
    probs[2] = p                      # "approach"
    return tuple(probs)


class FakePolicy:
    def __init__(self, out: PolicyOutput | Exception, validated: bool = True) -> None:
        self.out = out
        self._validated = validated
        self.calls = 0

    def predict(self, frame: Any, telemetry: Any) -> PolicyOutput:
        self.calls += 1
        if isinstance(self.out, Exception):
            raise self.out
        return self.out

    def validated(self) -> bool:
        return self._validated


@dataclass
class Prior:
    location_id: int
    xyz: tuple[float, float, float]
    status: PriorStatus3D = PriorStatus3D.UNKNOWN

    @property
    def z(self) -> float:
        return self.xyz[2]


@dataclass
class World:
    priors: dict[int, Prior] = field(default_factory=dict)
    all_resolved: bool = False


@dataclass
class Obs:
    frame_id: int
    frame: np.ndarray = field(default_factory=lambda: np.zeros((36, 64, 3), np.uint8))


@dataclass
class Camera:
    pitch_deg: float = -12.0


@dataclass
class Odometry:
    grounded: bool = True


@dataclass
class Telemetry:
    camera: Camera = field(default_factory=Camera)
    odometry: Odometry = field(default_factory=Odometry)
    events: list[dict[str, Any]] = field(default_factory=list)
    sim_time: float = 0.0


class FakeAgent:
    """The parts of VisualAgent3D the selector uses."""

    def __init__(self, state: Agent3DState = Agent3DState.NAVIGATING) -> None:
        self.state = state
        self.action = ActionCommand(move_forward=0.2, move_right=0.1, look_yaw=-0.3)
        self.obs: Obs | None = Obs(1)
        self.now = 0.0
        self.step_seconds = 2 / 30
        self.telemetry: Telemetry | None = Telemetry()
        self.cues = Cues(False, False)
        self.rules = {"pitch_min_deg": -55.0, "pitch_max_deg": 25.0}
        self.world = World({1: Prior(1, (10.0, 10.0, 0.0))})
        self.position = (0.0, 0.0, 0.0)
        self.ticks = 0
        self.credited: list[tuple[int, float, bool | None]] = []
        self.metrics = object()
        self.outcome: str | None = None
        self.end_reason: str | None = None
        self._terminal = False

    def tick(self) -> Agent3DState:
        self.ticks += 1
        self.action = ActionCommand(move_forward=0.2, move_right=0.1, look_yaw=-0.3)
        return self.state

    def initialize(self, telemetry: Any) -> None:
        self.telemetry = telemetry

    def is_terminal(self) -> bool:
        return self._terminal

    def stop(self, outcome: str, reason: str) -> None:
        self._terminal, self.outcome, self.end_reason = True, outcome, reason

    def wants_observation(self) -> bool:
        return True

    def on_step(self, result: Any) -> None:
        self.now = result.telemetry.sim_time
        self.telemetry = result.telemetry

    def credit_opening(self, location_id: int, seconds: float, reward: bool | None) -> None:
        self.credited.append((location_id, seconds, reward))
        self.world.priors[location_id].status = PriorStatus3D.OPENED


@dataclass
class Step:
    telemetry: Telemetry
    observation: Any = None


class SelectorModes(unittest.TestCase):
    def test_deterministic_mode_passes_through(self) -> None:
        agent = FakeAgent()
        sel = PolicySelector(agent, FakePolicy(output()), mode="deterministic")
        sel.tick()
        self.assertEqual(sel.action, agent.action)
        self.assertEqual(sel.stats.learned_decisions, 0)

    def test_mode_needs_a_policy_and_can_switch(self) -> None:
        with self.assertRaises(ValueError):
            PolicySelector(FakeAgent(), None, mode="learned")
        sel = PolicySelector(FakeAgent(), FakePolicy(output()))
        for mode in ("learned", "hybrid", "deterministic"):
            sel.set_mode(mode)
            self.assertEqual(sel.mode, mode)
        with self.assertRaises(ValueError):
            sel.set_mode("telepathic")

    def test_learned_mode_uses_the_model(self) -> None:
        agent = FakeAgent()
        sel = PolicySelector(agent, FakePolicy(output(move_forward=0.7, look_yaw=0.2)),
                             mode="learned")
        sel.tick()
        self.assertAlmostEqual(sel.action.move_forward, 0.7)
        self.assertAlmostEqual(sel.action.look_yaw, 0.2)
        self.assertEqual(agent.ticks, 0)                      # FSM not driving
        self.assertEqual(sel.stats.learned_decisions, 1)

    def test_hybrid_only_in_supported_confident_states(self) -> None:
        agent = FakeAgent(Agent3DState.NAVIGATING)
        sel = PolicySelector(agent, FakePolicy(output(move_forward=0.9,
                                                      phase_probs=approach_phases())),
                             mode="hybrid")
        sel.tick()
        self.assertAlmostEqual(sel.action.move_forward, 0.2)   # deterministic while navigating
        agent.state = Agent3DState.APPROACHING
        agent.obs = Obs(2)
        sel.tick()
        self.assertAlmostEqual(sel.action.move_forward, 0.9)   # learned while approaching
        self.assertEqual(sel.stats.learned_decisions, 1)

    def test_hybrid_low_confidence_and_unvalidated_model_fall_back(self) -> None:
        agent = FakeAgent(Agent3DState.APPROACHING)
        sel = PolicySelector(agent, FakePolicy(output(move_forward=0.9)), mode="hybrid")
        sel.tick()                                              # uniform phases: not confident
        self.assertAlmostEqual(sel.action.move_forward, 0.2)
        self.assertEqual(sel.stats.fallbacks["low_confidence"], 1)
        unvalidated = PolicySelector(FakeAgent(Agent3DState.APPROACHING),
                                     FakePolicy(output(phase_probs=approach_phases()),
                                                validated=False), mode="hybrid")
        unvalidated.tick()
        self.assertEqual(unvalidated.stats.fallbacks["policy_unavailable"], 1)


class SafetyLayer(unittest.TestCase):
    def test_invalid_output_falls_back(self) -> None:
        agent = FakeAgent()
        sel = PolicySelector(agent, FakePolicy(output(move_forward=math.nan)), mode="learned")
        sel.tick()
        self.assertEqual(sel.stats.fallbacks["invalid_output"], 1)
        self.assertEqual(agent.ticks, 1)                      # deterministic decision taken
        self.assertAlmostEqual(sel.action.move_forward, 0.2)

    def test_timeout_and_disable_after_repeated_failures(self) -> None:
        agent = FakeAgent()
        limits = SafetyLimits(timeout_ms=5.0, max_consecutive_failures=3)
        sel = PolicySelector(agent, FakePolicy(output(latency_ms=50.0)), mode="learned",
                             limits=limits)
        for k in range(5):
            agent.obs = Obs(10 + k)
            sel.tick()
        self.assertEqual(sel.stats.fallbacks["timeout"], 3)
        self.assertIn("consecutive failures", sel.stats.disabled_reason or "")
        self.assertGreaterEqual(sel.stats.fallbacks["policy_unavailable"], 2)
        self.assertEqual(sel.stats.learned_decisions, 0)

    def test_model_exceptions_never_crash(self) -> None:
        agent = FakeAgent()
        sel = PolicySelector(agent, FakePolicy(RuntimeError("boom")), mode="learned")
        sel.tick()
        self.assertEqual(sel.stats.fallbacks["model_error"], 1)
        self.assertEqual(agent.ticks, 1)

    def test_ranges_speed_and_pitch_limits(self) -> None:
        agent = FakeAgent()
        agent.telemetry = Telemetry(Camera(pitch_deg=-54.0))
        limits = SafetyLimits(max_move=0.8, max_look=0.5)
        sel = PolicySelector(agent, FakePolicy(output(move_forward=1.0, move_right=1.0,
                                                      look_yaw=1.0, look_pitch=-1.0)),
                             mode="learned", limits=limits)
        sel.tick()
        a = sel.action
        self.assertAlmostEqual(math.hypot(a.move_forward, a.move_right), 0.8, places=6)
        self.assertAlmostEqual(a.look_yaw, 0.5)
        self.assertEqual(a.look_pitch, 0.0)                   # at the pitch limit already
        self.assertTrue(a.to_dict())                          # passes protocol validation

    def test_interaction_needs_the_prompt(self) -> None:
        agent = FakeAgent()
        sel = PolicySelector(agent, FakePolicy(output(interact_prob=0.9)), mode="learned")
        sel.tick()
        self.assertFalse(sel.action.interact)
        self.assertEqual(sel.stats.interact_blocked, 1)
        agent.cues = Cues(True, False)
        agent.obs = Obs(2)
        sel.tick()
        self.assertTrue(sel.action.interact)
        self.assertEqual((sel.action.move_forward, sel.action.move_right), (0.0, 0.0))

    def test_stale_outputs_are_not_reused_for_long(self) -> None:
        agent = FakeAgent()
        policy = FakePolicy(output())
        sel = PolicySelector(agent, policy, mode="learned",
                             limits=SafetyLimits(max_output_age_s=0.5))
        sel.tick()
        agent.now = 0.3
        sel.tick()                                            # same frame, recent: reused
        self.assertEqual(policy.calls, 1)
        agent.now = 2.0
        sel.tick()                                            # same frame, too old
        self.assertEqual(sel.stats.fallbacks["stale_output"], 1)

    def test_stuck_learned_policy_gets_bounded_help(self) -> None:
        agent = FakeAgent()
        limits = SafetyLimits(stuck_seconds=1.0, recovery_seconds=0.5, max_recoveries=2)
        sel = PolicySelector(agent, FakePolicy(output()), mode="learned", limits=limits)
        for k in range(200):                                  # never moves
            agent.now = k * 0.1
            agent.obs = Obs(k + 1)
            sel.tick()
        self.assertEqual(sel.stats.learned_recoveries, 2)
        self.assertGreater(agent.ticks, 0)


class OpeningAttribution(unittest.TestCase):
    def test_notification_next_to_a_spawn_point_is_credited_once(self) -> None:
        agent = FakeAgent()
        agent.world.priors = {1: Prior(1, (1.0, 0.5, 0.0)), 2: Prior(2, (30.0, 0.0, 0.0))}
        agent.cues = Cues(True, False)
        sel = PolicySelector(agent, FakePolicy(output(interact_prob=0.9)), mode="learned")
        sel.tick()
        sel.on_step(Step(Telemetry(sim_time=1.0)))
        event = {"type": "chest_opened_notification", "reward": True}
        sel.on_step(Step(Telemetry(events=[event], sim_time=3.0)))
        self.assertEqual(agent.credited, [(1, 2.0, True)])
        sel.on_step(Step(Telemetry(events=[event], sim_time=4.0)))   # nothing else nearby
        self.assertEqual(len(agent.credited), 1)
        self.assertEqual(sel.stats.fallbacks["unattributed_opening"], 1)

    def test_all_resolved_ends_the_episode(self) -> None:
        agent = FakeAgent()
        agent.world.all_resolved = True
        sel = PolicySelector(agent, FakePolicy(output()), mode="learned")
        sel.tick()
        self.assertTrue(sel.is_terminal())
        self.assertEqual(sel.outcome, "completed")


if __name__ == "__main__":
    unittest.main()
