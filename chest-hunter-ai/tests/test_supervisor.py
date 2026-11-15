"""Phase 6 reliability supervisor: detection, bounded escalation, safe termination."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from chest_hunter.agent.controller_3d import Agent3DState
from chest_hunter.agent.fsm import Transition
from chest_hunter.localization.world_model_3d import PriorStatus3D
from chest_hunter.runtime.state_spec import (
    STATE_SPECS_3D,
    LoopDetector,
    action_violations,
    check_entry,
    describe_specs,
)
from chest_hunter.runtime.supervisor import LADDERS, Supervisor, SupervisorConfig
from chest_hunter.unreal_bridge.action_adapter import ActionCommand

S = Agent3DState


@dataclass
class FakePrior:
    location_id: int
    status: PriorStatus3D = PriorStatus3D.UNKNOWN

    @property
    def resolved(self) -> bool:
        return self.status not in (PriorStatus3D.UNKNOWN, PriorStatus3D.CHEST_SEEN)


@dataclass
class FakeMetrics:
    chests_opened: int = 0
    targets_reached: int = 0
    recoveries: int = 0


@dataclass
class FakeAgent:
    """Just enough of VisualAgent3D for the supervisor; interventions are recorded."""

    state: Agent3DState = S.NAVIGATING
    now: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal: Any = None
    action: ActionCommand = field(default_factory=ActionCommand)
    metrics: FakeMetrics = field(default_factory=FakeMetrics)
    calls: list[str] = field(default_factory=list)
    stopped: tuple[str, str] | None = None
    applicable: bool = True
    reject_invalid: bool = False
    on_transition: Any = None
    on_invalid_transition: Any = None

    def __post_init__(self) -> None:
        self.world = SimpleNamespace(priors={i: FakePrior(i) for i in range(1, 6)})
        self._known_targets: set[int] = set()
        self.follower = object()

    def is_terminal(self) -> bool:
        return self.stopped is not None

    def stop(self, outcome: str, reason: str) -> None:
        self.stopped = (outcome, reason)

    def _intervene(self, name: str) -> bool:
        self.calls.append(name)
        return self.applicable

    def supervisor_replan(self, reason: str) -> bool:
        return self._intervene("replan")

    def supervisor_abandon_target(self, reason: str, *, failed: bool = False) -> bool:
        return self._intervene("fail_target" if failed else "abandon_target")

    def supervisor_relocalise(self, reason: str) -> bool:
        return self._intervene("relocalise")

    def supervisor_redirect(self, reason: str) -> bool:
        return self._intervene("redirect")

    def supervisor_unstick(self, reason: str, *, attempt: int = 1) -> bool:
        return self._intervene("unstick")

    # helpers for the tests
    def enter(self, state: Agent3DState, reason: str | None = None) -> None:
        source, self.state = self.state, state
        if self.on_transition is not None:
            self.on_transition(Transition(0, source, state, reason))


def make(config: SupervisorConfig | None = None, **agent_kw: Any) -> tuple[Supervisor, FakeAgent]:
    agent = FakeAgent(**agent_kw)
    sup = Supervisor(config or SupervisorConfig())
    sup.attach(agent)
    return sup, agent


def step(sup: Supervisor, agent: FakeAgent, dt: float = 1 / 15, *, move: bool = False,
         to: tuple[float, float, float] | None = None) -> None:
    agent.action = ActionCommand(move_forward=1.0 if move else 0.0)
    sup.before_tick()
    sup.validate_action()
    agent.now += dt
    if to is not None:
        agent.position = to
    sup.after_step(None)


class StateSpecifications(unittest.TestCase):
    def test_every_state_is_specified(self) -> None:
        self.assertEqual(set(STATE_SPECS_3D), set(Agent3DState))
        rows = describe_specs()
        self.assertEqual(len(rows), len(Agent3DState))
        for row in rows:
            self.assertIn(row["category"], ("INITIALIZING", "OBSERVING", "EXPLORING",
                                             "SELECTING_TARGET", "PLANNING_ROUTE", "NAVIGATING",
                                             "ALIGNING", "INTERACTING", "VERIFYING", "RECOVERING",
                                             "COMPLETED", "FAILED"))

    def test_time_limits_exceed_the_measured_baseline(self) -> None:
        # Longest visits measured in 18 baseline arena episodes (docs/phase6_baseline_audit.md).
        measured = {S.NAVIGATING: 18.73, S.APPROACHING: 9.27, S.ALIGNING: 1.67,
                    S.INTERACTING: 2.07, S.VERIFYING_OPEN: 2.07, S.OBSERVING: 2.40,
                    S.SCANNING: 1.53}
        for state, longest in measured.items():
            limit = STATE_SPECS_3D[state].max_seconds
            assert limit is not None
            self.assertGreaterEqual(limit, 2 * longest, state)

    def test_permitted_commands(self) -> None:
        self.assertEqual(action_violations(S.ALIGNING, ActionCommand(interact=True)),
                         ["interact"])
        self.assertEqual(action_violations(S.INTERACTING, ActionCommand(move_forward=0.5)),
                         ["move"])
        self.assertEqual(action_violations(S.INTERACTING, ActionCommand(interact=True)), [])

    def test_entry_conditions(self) -> None:
        agent = SimpleNamespace(goal=None, follower=None)
        self.assertEqual(check_entry(S.NAVIGATING, agent), "no goal")
        agent.goal = SimpleNamespace(kind="inspect")
        self.assertEqual(check_entry(S.NAVIGATING, agent), "no route to follow")
        self.assertIn("not a chest", check_entry(S.ALIGNING, agent) or "")
        self.assertIsNone(check_entry(S.EXPLORING, agent))


class LoopDetection(unittest.TestCase):
    def test_a_repeated_cycle_is_reported(self) -> None:
        det = LoopDetector(window=12, repetitions=3)
        seen = None
        for _ in range(3):
            for st in ("NAVIGATING", "RECOVERING", "OBSERVING", "PLANNING_ROUTE"):
                seen = det.observe(st) or seen
        self.assertIsNotNone(seen)
        assert seen is not None
        self.assertEqual(len(seen.period), 4)
        self.assertGreaterEqual(seen.repetitions, 3)

    def test_progress_clears_the_window(self) -> None:
        det = LoopDetector(window=12, repetitions=3)
        for _ in range(2):
            det.observe("NAVIGATING")
            det.observe("RECOVERING")
        det.progress()
        self.assertIsNone(det.observe("NAVIGATING"))
        self.assertIsNone(det.observe("RECOVERING"))

    def test_no_loop_without_repetition(self) -> None:
        det = LoopDetector(window=12, repetitions=3)
        out = [det.observe(s) for s in ("OBSERVING", "SCANNING", "EXPLORING", "PLANNING_ROUTE",
                                         "NAVIGATING", "APPROACHING", "ALIGNING", "INTERACTING")]
        self.assertTrue(all(o is None for o in out))

    def test_bounded_window(self) -> None:
        det = LoopDetector(window=6, repetitions=2)
        for k in range(1000):
            det.observe(f"S{k}")
        self.assertEqual(len(det.recent), 6)


class Detection(unittest.TestCase):
    def test_stuck_agent_is_detected_and_helped_in_order(self) -> None:
        sup, agent = make(SupervisorConfig(stationary_seconds=2.0, cooldown_seconds=0.0),
                          goal=SimpleNamespace(location_id=3, kind="inspect"))
        for _ in range(90):                       # movement requested, the agent never moves
            step(sup, agent, move=True)
        self.assertGreaterEqual(sup.stats.detections["stationary"], 1)
        self.assertEqual(agent.calls[:2], ["replan", "unstick"])
        events = sup.events.of_kind("stationary")
        self.assertTrue(events and events[0].action == "replan")
        self.assertEqual(events[0].goal_location, 3)

    def test_a_moving_agent_is_not_reported(self) -> None:
        sup, agent = make(SupervisorConfig(stationary_seconds=2.0))
        for k in range(60):
            step(sup, agent, move=True, to=(0.1 * k, 0.0, 0.0))
        self.assertEqual(sup.stats.detections["stationary"], 0)

    def test_state_timeout_uses_simulated_time_in_the_state(self) -> None:
        sup, agent = make(SupervisorConfig(cooldown_seconds=0.0), state=S.ALIGNING,
                          goal=SimpleNamespace(location_id=2, kind="chest"))
        limit = STATE_SPECS_3D[S.ALIGNING].max_seconds
        assert limit is not None
        for _ in range(int(limit * 15) + 5):
            step(sup, agent, to=(0.0, 0.0, 0.0))
        self.assertEqual(sup.stats.detections["state_timeout"], 1)
        self.assertEqual(agent.calls[0], "replan")

    def test_transition_loop_escalates_without_repeating_an_action(self) -> None:
        sup, agent = make(SupervisorConfig(cooldown_seconds=0.0), state=S.EXPLORING,
                          goal=SimpleNamespace(location_id=4, kind="inspect"))
        for _ in range(6):
            for st in (S.OBSERVING, S.EXPLORING):
                agent.enter(st)
                step(sup, agent)
        self.assertGreaterEqual(sup.stats.detections["transition_loop"], 2)
        used = [c for c in agent.calls]
        self.assertEqual(len(used), len(set(used)), "an intervention was repeated")
        self.assertEqual(used[0], "abandon_target")

    def test_interaction_failures_mark_the_target_failed(self) -> None:
        sup, agent = make(SupervisorConfig(max_interaction_failures=3), state=S.VERIFYING_OPEN,
                          goal=SimpleNamespace(location_id=5, kind="chest"))
        for _ in range(3):
            agent.enter(S.RECOVERING, "interaction_failed")
            agent.state = S.ALIGNING
            step(sup, agent)
        self.assertEqual(sup.stats.detections["interaction_failures"], 1)
        self.assertIn("fail_target", agent.calls)

    def test_alignment_failures_are_not_interaction_failures(self) -> None:
        sup, agent = make(state=S.ALIGNING, goal=SimpleNamespace(location_id=5, kind="chest"))
        for _ in range(6):
            agent.enter(S.RECOVERING, "alignment_failed")
            agent.state = S.APPROACHING
            step(sup, agent)
        self.assertEqual(sup.stats.detections["interaction_failures"], 0)

    def test_no_progress_ends_in_a_safe_abort(self) -> None:
        sup, agent = make(SupervisorConfig(max_no_progress_seconds=5.0, cooldown_seconds=0.0,
                                           max_recovery_attempts=10), state=S.SCANNING)
        agent.applicable = False                   # nothing the agent could do helps
        for _ in range(int(5.0 * 15) * 2 + 10):
            step(sup, agent, to=(0.0, 0.0, 0.0))
            if agent.is_terminal():
                break
        assert agent.stopped is not None
        self.assertEqual(agent.stopped[0], "failed")
        self.assertIn("no_progress", agent.stopped[1])
        self.assertEqual(sup.stats.interventions["abort"], 1)

    def test_progress_resets_the_no_progress_clock(self) -> None:
        sup, agent = make(SupervisorConfig(max_no_progress_seconds=5.0), state=S.NAVIGATING)
        for k in range(15 * 20):
            if k % 30 == 0:
                agent.metrics.targets_reached += 1
            step(sup, agent, move=True, to=(0.05 * k, 0.0, 0.0))
        self.assertEqual(sup.stats.detections["no_progress"], 0)


class Limits(unittest.TestCase):
    def test_recovery_attempts_per_incident_are_bounded(self) -> None:
        sup, agent = make(SupervisorConfig(max_recovery_attempts=2, cooldown_seconds=0.0,
                                           stationary_seconds=1.0),
                          goal=SimpleNamespace(location_id=1, kind="inspect"))
        for _ in range(400):
            step(sup, agent, move=True)
        self.assertEqual(len(agent.calls), 2)             # then the incident is exhausted
        self.assertIsNone(agent.stopped)                  # stationary never aborts by itself

    def test_episode_intervention_budget_aborts(self) -> None:
        sup, agent = make(SupervisorConfig(max_interventions=3, cooldown_seconds=0.0,
                                           stationary_seconds=1.0), state=S.NAVIGATING)
        for loc in range(1, 6):
            agent.goal = SimpleNamespace(location_id=loc, kind="inspect")
            for _ in range(40):
                step(sup, agent, move=True)
                if agent.is_terminal():
                    break
            if agent.is_terminal():
                break
        assert agent.stopped is not None
        self.assertIn("intervention budget", agent.stopped[1])

    def test_inapplicable_rungs_are_skipped(self) -> None:
        sup, agent = make(SupervisorConfig(cooldown_seconds=0.0, stationary_seconds=1.0))
        agent.applicable = False
        for _ in range(40):
            step(sup, agent, move=True)
        self.assertGreater(sum(sup.stats.skipped.values()), 0)
        self.assertEqual(sum(sup.stats.interventions.values()), 0)

    def test_every_ladder_is_finite_and_known(self) -> None:
        known = {"replan", "abandon_target", "fail_target", "unstick", "relocalise", "redirect",
                 "fallback_deterministic", "abort"}
        for ladder in LADDERS.values():
            self.assertTrue(0 < len(ladder) <= 6)
            self.assertTrue(set(ladder) <= known)

    def test_monitor_mode_applies_nothing(self) -> None:
        sup, agent = make(SupervisorConfig(mode="monitor", stationary_seconds=1.0,
                                           cooldown_seconds=0.0),
                          goal=SimpleNamespace(location_id=2, kind="inspect"))
        for _ in range(100):
            step(sup, agent, move=True)
        self.assertEqual(agent.calls, [])
        self.assertTrue(any(e.action and e.action.startswith("would ")
                            for e in sup.events.of_kind("stationary")))

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            SupervisorConfig(mode="panic")
        with self.assertRaises(ValueError):
            SupervisorConfig(max_recovery_attempts=0)
        with self.assertRaises(ValueError):
            SupervisorConfig(loop_detection_window=2)


class CommandsAndObservations(unittest.TestCase):
    def test_non_finite_commands_are_fixed(self) -> None:
        sup, agent = make()
        agent.action = ActionCommand(move_forward=math.nan, look_yaw=3.0)
        sup.before_tick()
        sup.validate_action()
        self.assertEqual(agent.action.move_forward, 0.0)
        self.assertEqual(agent.action.look_yaw, 1.0)
        self.assertEqual(sup.stats.invalid_actions, 1)

    def test_a_command_the_state_may_not_issue_is_removed(self) -> None:
        sup, agent = make(state=S.ALIGNING)
        sup.before_tick()
        agent.action = ActionCommand(interact=True)
        sup.validate_action()
        self.assertFalse(agent.action.interact)
        self.assertEqual(sup.stats.actions_not_permitted, 1)

    def test_the_last_hold_step_of_an_interaction_is_permitted(self) -> None:
        sup, agent = make(state=S.INTERACTING)
        sup.before_tick()                           # produced in INTERACTING ...
        agent.state = S.VERIFYING_OPEN              # ... the handler switched state
        agent.action = ActionCommand(interact=True)
        sup.validate_action()
        self.assertTrue(agent.action.interact)
        self.assertEqual(sup.stats.actions_not_permitted, 0)

    def _result(self, frame: Any, *, x: float = 0.0, t: float = 1.0) -> Any:
        tel = SimpleNamespace(sim_time=t, odometry=SimpleNamespace(position=(x, 0.0, 0.0)),
                              camera=SimpleNamespace(yaw_deg=0.0, pitch_deg=0.0))
        obs = SimpleNamespace(frame=frame) if frame is not None else None
        from chest_hunter.unreal_bridge.client import StepResult

        return StepResult(1, tel, obs, False, False)  # type: ignore[arg-type]

    def test_blank_or_malformed_frames_are_dropped(self) -> None:
        sup, agent = make()
        good = np.random.default_rng(0).integers(0, 255, (180, 320, 3), dtype=np.uint8)
        self.assertIsNotNone(sup.filter_step(self._result(good), observe=True, width=320,
                                             height=180).observation)
        blank = np.zeros((180, 320, 3), np.uint8)
        self.assertIsNone(sup.filter_step(self._result(blank), observe=True, width=320,
                                          height=180).observation)
        wrong = np.ones((90, 160, 3), np.uint8)
        self.assertIsNone(sup.filter_step(self._result(wrong), observe=True, width=320,
                                          height=180).observation)
        self.assertEqual(sup.stats.invalid_observations, 2)

    def test_a_streak_of_invalid_frames_escalates(self) -> None:
        sup, agent = make(SupervisorConfig(max_invalid_observations=3, cooldown_seconds=0.0))
        blank = np.zeros((180, 320, 3), np.uint8)
        for k in range(8):
            sup.filter_step(self._result(blank, t=k), observe=True, width=320, height=180)
            step(sup, agent)
        self.assertEqual(sup.stats.detections["invalid_observations"], 2)
        self.assertEqual(agent.calls, ["relocalise"])        # then a second full streak ...
        assert agent.stopped is not None                       # ... ends the episode
        self.assertIn("invalid", agent.stopped[1])

    def test_non_finite_telemetry_is_reported(self) -> None:
        sup, _agent = make()
        sup.filter_step(self._result(None, x=math.inf), observe=False, width=320, height=180)
        self.assertEqual(sup.stats.invalid_observations, 1)

    def test_communication_failure_ends_the_episode_cleanly(self) -> None:
        from chest_hunter.unreal_bridge.client import BridgeTimeout

        sup, agent = make()
        sup.communication_failure(BridgeTimeout("timeout", "STEP got no answer"))
        assert agent.stopped is not None
        self.assertEqual(agent.stopped[0], "failed")
        self.assertIn("communication failure", agent.stopped[1])
        self.assertEqual(sup.stats.communication_failures, 1)


class FakeSelector:
    """Stands in for PolicySelector: a mode, stats and set_mode."""

    def __init__(self, agent: FakeAgent, mode: str = "hybrid") -> None:
        self.agent = agent
        self.mode = mode
        self.action = agent.action
        self.stats = SimpleNamespace(fallbacks={}, latencies_ms=[])
        self.switched: list[str] = []
        self._fsm_current = True

    def set_mode(self, mode: str) -> None:
        self.switched.append(mode)
        self.mode = mode

    def is_terminal(self) -> bool:
        return self.agent.is_terminal()

    def stop(self, outcome: str, reason: str) -> None:
        self.agent.stop(outcome, reason)


class ModelMonitoring(unittest.TestCase):
    def _attach(self, config: SupervisorConfig | None = None) -> tuple[Supervisor, FakeAgent,
                                                                        FakeSelector]:
        agent = FakeAgent()
        sel = FakeSelector(agent)
        sup = Supervisor(config or SupervisorConfig())
        sup.attach(sel)
        return sup, agent, sel

    def test_slow_inference_falls_back_to_deterministic_control(self) -> None:
        sup, agent, sel = self._attach(SupervisorConfig(inference_budget_ms=20.0,
                                                        latency_window=10))
        sel.stats.latencies_ms.extend([35.0] * 10)
        step(sup, agent, to=(0.0, 0.0, 0.0))
        self.assertEqual(sel.switched, ["deterministic"])
        self.assertEqual(sup.stats.policy_switches, 1)
        self.assertTrue(sup.events.of_kind("policy_switch"))

    def test_repeated_invalid_outputs_fall_back(self) -> None:
        sup, agent, sel = self._attach(SupervisorConfig(max_model_failures=3))
        sel.stats.fallbacks = {"invalid_output": 2, "timeout": 1}
        step(sup, agent, to=(0.0, 0.0, 0.0))
        self.assertEqual(sel.mode, "deterministic")
        self.assertEqual(len(sup.events.of_kind("model_failure")), 3)   # 2 kinds + incident

    def test_fast_inference_is_left_alone(self) -> None:
        sup, agent, sel = self._attach(SupervisorConfig(latency_window=10))
        sel.stats.latencies_ms.extend([1.0] * 50)
        step(sup, agent, to=(0.0, 0.0, 0.0))
        self.assertEqual(sel.switched, [])


class RealAgent(unittest.TestCase):
    """The supervisor on real episodes of the reference arena."""

    @classmethod
    def setUpClass(cls) -> None:
        from tests.phase5_helpers import settings

        cls.settings = settings()

    def _run(self, *, supervisor: Any, seed: int = 3, ticks: int = 900, hook: Any = None,
             client: Any = None) -> Any:
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.database import DatabaseManager
        from chest_hunter.scenarios import seed_arena_3d

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        runner = Runner3D.from_settings(db, self.settings, mode="hybrid", seed=seed,
                                        backend="reference", max_ticks=ticks, width=320,
                                        height=180, collect_truth=True, supervisor=supervisor,
                                        on_decision=hook, client=client)
        return runner, runner.run()

    def test_monitor_mode_does_not_change_the_episode(self) -> None:
        _r0, plain = self._run(supervisor=None)
        r1, watched = self._run(supervisor=SupervisorConfig(mode="monitor"))
        self.assertEqual((plain.chests_opened, plain.sim_seconds, plain.metrics.analyses,
                          plain.metrics.travel_distance),
                         (watched.chests_opened, watched.sim_seconds, watched.metrics.analyses,
                          watched.metrics.travel_distance))
        self.assertIsNotNone(watched.supervisor)
        self.assertEqual(r1.supervisor.stats.steps, watched.supervisor["steps"])

    def test_an_enforcing_supervisor_episode_completes_and_reports(self) -> None:
        runner, res = self._run(supervisor=SupervisorConfig())
        self.assertIn(res.outcome, ("completed", "tick_limit"))
        self.assertGreater(runner.supervisor.stats.steps, 100)
        self.assertEqual(res.supervisor["communication_failures"], 0)

    def test_invalid_transitions_are_refused_and_counted(self) -> None:
        captured: dict[str, Any] = {}

        def hook(runner: Any, _result: Any) -> None:
            agent = runner.controller
            if "done" not in captured and agent.state is S.NAVIGATING:
                captured["done"] = True
                agent._transition(S.INTERACTING, "test: not in the table")
                captured["state"] = agent.state

        runner, res = self._run(supervisor=SupervisorConfig(), hook=hook)
        self.assertEqual(captured.get("state"), S.NAVIGATING)    # the state did not change
        self.assertGreaterEqual(runner.supervisor.stats.rejected_transitions, 1)
        self.assertGreaterEqual(res.supervisor["detections"].get("invalid_transition", 0), 1)

    def test_lost_connection_ends_the_episode_without_an_exception(self) -> None:
        from chest_hunter.unreal_bridge.client import BridgeConnectionError

        def hook(runner: Any, _result: Any) -> None:
            if runner.controller.decisions == 120:
                def broken(*_a: Any, **_k: Any) -> Any:
                    raise BridgeConnectionError("disconnected", "server went away (test)")
                runner.client.step = broken

        runner, res = self._run(supervisor=SupervisorConfig(), hook=hook)
        self.assertEqual(res.outcome, "failed")
        self.assertIn("communication failure", res.end_reason or "")
        self.assertIsNotNone(res.communication_failure)


class AgentInterventions(unittest.TestCase):
    """The intervention hooks on the real agent keep its transition table."""

    def _episode_until(self, state: Agent3DState, action: str) -> tuple[Any, list[str]]:
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.database import DatabaseManager
        from chest_hunter.scenarios import seed_arena_3d
        from tests.phase5_helpers import settings

        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        seed_arena_3d(db)
        seen: list[str] = []

        def hook(runner: Any, _result: Any) -> None:
            agent = runner.controller
            if not seen and agent.state is state:
                ok = getattr(agent, f"supervisor_{action}")("test")
                seen.append(f"{ok}:{agent.state}")

        runner = Runner3D.from_settings(db, settings(), mode="hybrid", seed=5,
                                        backend="reference", max_ticks=1200, width=320,
                                        height=180, on_decision=hook)
        res = runner.run()
        self.assertIn(res.outcome, ("completed", "tick_limit"))
        return runner, seen

    def test_replan_from_navigation(self) -> None:
        _runner, seen = self._episode_until(S.NAVIGATING, "replan")
        self.assertEqual(seen, ["True:PLANNING_ROUTE"])

    def test_relocalise_from_navigation(self) -> None:
        runner, seen = self._episode_until(S.NAVIGATING, "relocalise")
        self.assertEqual(seen, ["True:OBSERVING"])
        self.assertGreaterEqual(runner.controller.metrics.scans, 1)

    def test_abandon_target_marks_the_spawn_point(self) -> None:
        runner, seen = self._episode_until(S.APPROACHING, "abandon_target")
        self.assertEqual(seen, ["True:EXPLORING"])
        self.assertGreaterEqual(runner.controller.metrics.abandoned_targets, 1)

    def test_unstick_moves_and_reobserves(self) -> None:
        _runner, seen = self._episode_until(S.NAVIGATING, "unstick")
        self.assertEqual(seen, ["True:OBSERVING"])


if __name__ == "__main__":
    unittest.main()
