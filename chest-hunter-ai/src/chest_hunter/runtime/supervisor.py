"""Reliability supervisor for autonomous 3D episodes (Phase 6).

The supervisor watches one episode from the runner's side of the loop. It never reads
simulator ground truth: it sees the controller's state, its commands, the telemetry and
observations the agent receives, and the agent's own bookkeeping.

Per decision step the runner calls:

1. :meth:`Supervisor.validate_action` after the controller chose its command (finite
   values, valid ranges, commands the current state may issue);
2. :meth:`Supervisor.filter_step` on the simulator's reply (telemetry must be finite and
   advancing, frames must have the requested size and must not be blank);
3. :meth:`Supervisor.after_step` once the controller has processed the reply.

Conditions detected (each produces a structured event):

* ``state_timeout`` - longer in a state than its specification allows (simulated time,
  counted only while the state machine drives);
* ``stationary`` - movement requested but the agent did not move;
* ``transition_loop`` - the same cycle of states repeated without progress;
* ``repeated_target`` - the same spawn point planned for again and again without being
  reached;
* ``interaction_failures`` - repeated failed interactions at one spawn point;
* ``no_progress`` - nothing opened, resolved, reached or newly detected for too long;
* ``invalid_transition`` / ``entry_violation`` - a refused transition, or a state entered
  without its entry conditions;
* ``invalid_action`` / ``action_not_permitted`` - a non-finite or out-of-range command, or a
  command the state may not issue (fixed in place);
* ``invalid_observation`` - a malformed frame or telemetry (dropped);
* ``model_failure`` / ``inference_latency`` - invalid learned outputs, errors or time-outs;
* ``communication_failure`` - the bridge failed after the client's own bounded re-send.

Interventions follow a bounded escalation ladder per incident (condition + spawn point):
recalculate the route, select another target, drop stale tracks and re-observe, fall back
to deterministic control, and finally abort the episode with an explicit reason. A rung
that does not apply in the current situation is skipped; an incident never gets the same
intervention twice; ``max_recovery_attempts`` interventions per incident and
``max_interventions`` per episode are hard limits. In ``monitor`` mode everything is
detected and logged but nothing is applied (used to measure false alarms).
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from chest_hunter.agent.controller_3d import Agent3DState, VisualAgent3D
from chest_hunter.agent.fsm import Transition
from chest_hunter.localization.world_model_3d import PriorStatus3D
from chest_hunter.runtime.events import EventLog
from chest_hunter.runtime.state_spec import (
    LoopDetector,
    action_violations,
    check_entry,
    spec_for,
)

S = Agent3DState

#: Escalation ladders: tried in order, each rung at most once per incident. Only the
#: no-progress ladder (the backstop for everything else), an invalid-observation streak and
#: transitions the state table refuses end the episode; the other incidents get bounded help
#: and are left to the agent's own bounded mechanisms (and to the no-progress backstop).
LADDERS: Final[dict[str, tuple[str, ...]]] = {
    "state_timeout": ("replan", "abandon_target", "unstick", "relocalise", "redirect"),
    "stationary": ("replan", "unstick", "abandon_target", "relocalise",
                   "fallback_deterministic"),
    "transition_loop": ("abandon_target", "unstick", "relocalise", "fallback_deterministic"),
    "repeated_target": ("abandon_target",),
    "interaction_failures": ("fail_target",),
    "no_progress": ("replan", "abandon_target", "unstick", "relocalise",
                    "fallback_deterministic", "abort"),
    "invalid_transition": ("redirect", "relocalise", "abort"),
    "entry_violation": ("redirect", "relocalise", "abort"),
    "invalid_observations": ("relocalise", "abort"),
    "model_failure": ("fallback_deterministic",),
    "inference_latency": ("fallback_deterministic",),
}
#: Incidents keyed by the goal's spawn point (others are episode-wide).
PER_TARGET: Final = frozenset({"state_timeout", "stationary", "repeated_target",
                               "interaction_failures"})
#: Failed interactions (an alignment failure is not one: walking round to another side of
#: the chest is the agent's normal strategy for it).
INTERACTION_REASONS: Final = frozenset({"interaction_failed", "interaction_interrupted",
                                        "verification_inconclusive"})
FSM_ACTIONS: Final = frozenset({"replan", "abandon_target", "fail_target", "relocalise",
                                "redirect", "unstick"})


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    enabled: bool = True
    mode: str = "enforce"                    # enforce | monitor
    max_state_seconds: float = 30.0          # default limit for states without their own
    max_recovery_attempts: int = 3           # interventions per incident, then abort
    max_interventions: int = 30              # per episode, then abort
    loop_detection_window: int = 12
    loop_repetitions: int = 3
    max_no_progress_seconds: float = 90.0
    stationary_seconds: float = 6.0
    stationary_distance_m: float = 0.3
    max_goal_plans: int = 8
    max_interaction_failures: int = 3
    max_invalid_observations: int = 5
    max_model_failures: int = 5
    inference_budget_ms: float = 20.0
    latency_window: int = 50
    cooldown_seconds: float = 3.0
    event_capacity: int = 500
    use_state_specs: bool = True             # per-state limits; else max_state_seconds only

    def __post_init__(self) -> None:
        if self.mode not in ("enforce", "monitor"):
            raise ValueError("supervisor mode must be 'enforce' or 'monitor'")
        for name in ("max_state_seconds", "max_no_progress_seconds", "stationary_seconds",
                     "inference_budget_ms"):
            if not getattr(self, name) > 0:
                raise ValueError(f"supervisor.{name} must be positive")
        for name in ("max_recovery_attempts", "max_interventions", "max_goal_plans",
                     "max_interaction_failures", "max_invalid_observations",
                     "max_model_failures", "latency_window", "event_capacity"):
            if getattr(self, name) < 1:
                raise ValueError(f"supervisor.{name} must be at least 1")
        if self.loop_detection_window < 4 or self.loop_repetitions < 2:
            raise ValueError("loop_detection_window >= 4 and loop_repetitions >= 2 required")
        if self.cooldown_seconds < 0 or self.stationary_distance_m < 0:
            raise ValueError("cooldown_seconds and stationary_distance_m must be >= 0")


@dataclass(slots=True)
class SupervisorStats:
    steps: int = 0
    detections: Counter[str] = field(default_factory=Counter)
    suppressed: Counter[str] = field(default_factory=Counter)  # during a cooldown
    interventions: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)   # rung not applicable
    invalid_observations: int = 0
    invalid_actions: int = 0
    actions_not_permitted: int = 0
    rejected_transitions: int = 0
    entry_violations: int = 0
    communication_failures: int = 0
    aborted: str | None = None
    policy_switches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "detections": dict(self.detections),
                "suppressed": dict(self.suppressed),
                "interventions": dict(self.interventions), "skipped": dict(self.skipped),
                "invalid_observations": self.invalid_observations,
                "invalid_actions": self.invalid_actions,
                "actions_not_permitted": self.actions_not_permitted,
                "rejected_transitions": self.rejected_transitions,
                "entry_violations": self.entry_violations,
                "communication_failures": self.communication_failures,
                "policy_switches": self.policy_switches, "aborted": self.aborted,
                "interventions_total": sum(self.interventions.values())}


class Supervisor:
    def __init__(self, config: SupervisorConfig | None = None, *, episode: int | None = None,
                 events: EventLog | None = None) -> None:
        self.config = config or SupervisorConfig()
        self.events = events or EventLog(self.config.event_capacity, episode=episode)
        self.stats = SupervisorStats()
        self.controller: Any = None
        self.agent: VisualAgent3D | None = None
        self.selector: Any = None
        self.loops = LoopDetector(self.config.loop_detection_window,
                                  self.config.loop_repetitions)
        self._rungs: dict[tuple[Any, ...], tuple[int, int]] = {}   # rung, applied
        self._cooldown: dict[str, float] = {}
        self._state_time = 0.0
        self._last_now = 0.0
        self._last_progress = 0.0
        self._progress_marks: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._trail: deque[tuple[float, tuple[float, float, float], bool]] = deque(maxlen=400)
        self._goal_plans: Counter[int] = Counter()
        self._interaction_failures: Counter[int] = Counter()
        self._invalid_streak = 0
        self._model_failures_seen = 0
        self._last_fallbacks: Counter[str] = Counter()
        self._latencies_seen = 0
        self._pending: list[tuple[str, str, dict[str, Any]]] = []
        self._requested_move = False
        self._last_mode: str | None = None
        self._tick_state: Any = None
        self._unsticks = 0

    # -- wiring -------------------------------------------------------------------------------

    @property
    def enforcing(self) -> bool:
        return self.config.mode == "enforce"

    def attach(self, controller: Any) -> None:
        self.controller = controller
        agent = getattr(controller, "agent", controller)
        if not isinstance(agent, VisualAgent3D):
            raise TypeError("the supervisor watches the visual agent (hybrid / visual-nav "
                            "modes), not the oracle baseline")
        self.agent = agent
        self.selector = controller if controller is not agent else None
        agent.reject_invalid = True
        agent.on_transition = self._on_transition
        agent.on_invalid_transition = self._on_invalid_transition
        self._last_now = agent.now
        self._last_progress = agent.now
        self._last_mode = self.selector.mode if self.selector is not None else None

    @property
    def fsm_driving(self) -> bool:
        sel = self.selector
        if sel is None or sel.mode != "learned":
            return True
        return bool(getattr(sel, "_fsm_current", True))

    # -- transition hooks -------------------------------------------------------------------------

    def _on_transition(self, t: Transition) -> None:
        agent = self.agent
        assert agent is not None
        self._state_time = 0.0
        target = str(t.target)
        if target == S.PLANNING_ROUTE.value and agent.goal is not None \
                and agent.goal.location_id is not None:
            self._goal_plans[agent.goal.location_id] += 1
        if target == S.RECOVERING.value and (t.reason or "") in INTERACTION_REASONS \
                and agent.goal is not None and agent.goal.location_id is not None:
            self._interaction_failures[agent.goal.location_id] += 1
        if target in (S.ALIGNING.value,):
            self._mark_progress("target reached")
        loop = self.loops.observe(target)
        if loop is not None:
            self._queue("transition_loop", f"{loop.label} x{loop.repetitions}",
                        {"period": list(loop.period), "repetitions": loop.repetitions})
        if self.config.use_state_specs:
            problem = check_entry(t.target, agent)
            if problem is not None:
                self.stats.entry_violations += 1
                self._queue("entry_violation", f"{target}: {problem}", {})

    def _on_invalid_transition(self, source: Any, target: Any, reason: str | None) -> None:
        self.stats.rejected_transitions += 1
        self._queue("invalid_transition", f"{source} -> {target} refused ({reason})", {})

    def _queue(self, kind: str, detail: str, data: dict[str, Any]) -> None:
        """Conditions found inside a transition are handled after the step (the state
        machine must not be re-entered from its own transition hook)."""
        self._pending.append((kind, detail, data))

    # -- per-step checks ------------------------------------------------------------------------

    def before_tick(self) -> None:
        """Remember the state the coming command is produced in (a handler may switch
        state in the same tick, e.g. the last step of an interaction hold)."""
        assert self.agent is not None
        self._tick_state = self.agent.state

    def validate_action(self) -> None:
        """Fix the controller's command in place: non-finite or out-of-range values, and
        commands neither the state it was produced in nor the current state may issue
        (state-machine commands only)."""
        action = self.controller.action
        bad = []
        for name in ("move_forward", "move_right", "look_yaw", "look_pitch"):
            value = getattr(action, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                setattr(action, name, 0.0)
                bad.append(name)
            elif abs(value) > 1.0:
                setattr(action, name, max(-1.0, min(1.0, float(value))))
                bad.append(name)
        for name in ("jump", "interact", "sprint"):
            if hasattr(action, name) and not isinstance(getattr(action, name), bool):
                setattr(action, name, bool(getattr(action, name)))
        if bad:
            self.stats.invalid_actions += 1
            self._emit("invalid_action", "warning", f"fixed {', '.join(bad)}")
        agent = self.agent
        assert agent is not None
        if self.fsm_driving and self.config.use_state_specs:
            violations = action_violations(agent.state, action)
            if violations and self._tick_state is not None:
                earlier = set(action_violations(self._tick_state, action))
                violations = [v for v in violations if v in earlier]
            if violations:
                self.stats.actions_not_permitted += 1
                if self.enforcing:
                    if "interact" in violations:
                        action.interact = False
                    if "move" in violations:
                        action.move_forward = action.move_right = 0.0
                self._emit("action_not_permitted", "warning",
                           f"{agent.state}: {', '.join(violations)}",
                           action="removed" if self.enforcing else None)
        self._requested_move = math.hypot(action.move_forward, action.move_right) >= 0.3

    def filter_step(self, result: Any, *, observe: bool, width: int, height: int) -> Any:
        """Return ``result`` with an invalid observation removed (the agent then acts on
        telemetry only for this step). Invalid telemetry is reported, not repaired."""
        tel = result.telemetry
        problems = []
        values = [tel.sim_time, *tel.odometry.position, tel.camera.yaw_deg, tel.camera.pitch_deg]
        if not all(math.isfinite(float(v)) for v in values):
            problems.append("non-finite telemetry")
        agent = self.agent
        assert agent is not None
        if tel.sim_time + 1e-9 < agent.now:
            problems.append("simulated time went backwards")
        obs = result.observation
        if observe and obs is None:
            problems.append("requested frame missing")
        if obs is not None:
            frame = obs.frame
            if not isinstance(frame, np.ndarray) or frame.shape != (height, width, 3) \
                    or frame.dtype != np.uint8:
                problems.append(f"frame shape {getattr(frame, 'shape', None)}")
            elif int(frame.max()) == int(frame.min()):
                problems.append("blank frame")
        if not problems:
            self._invalid_streak = 0
            return result
        self.stats.invalid_observations += 1
        self._invalid_streak += 1
        drop = obs is not None and any(p.startswith(("frame", "blank")) for p in problems)
        self._emit("invalid_observation", "warning", "; ".join(problems),
                   action="frame dropped" if drop else None)
        if self._invalid_streak >= self.config.max_invalid_observations:
            self._queue("invalid_observations", f"{self._invalid_streak} invalid steps in a row",
                        {})
        if drop:
            return dataclasses.replace(result, observation=None)
        return result

    def after_step(self, result: Any) -> None:
        agent = self.agent
        assert agent is not None
        self.stats.steps += 1
        now = agent.now
        dt = max(0.0, now - self._last_now)
        self._last_now = now
        if agent.is_terminal():
            self._pending.clear()
            return
        if self.fsm_driving:
            self._state_time += dt
        self._check_progress(now)
        self._trail.append((now, agent.position, self._requested_move))
        pending, self._pending = self._pending, []
        for kind, detail, data in pending:
            self._incident(kind, detail, data)
            if agent.is_terminal():
                return
        self._check_state_time()
        self._check_stationary(now)
        self._check_targets()
        if now - self._last_progress > self.config.max_no_progress_seconds:
            self._incident("no_progress", f"no progress for {now - self._last_progress:.0f} s",
                           {})
            self._last_progress = now           # re-arm: the next rung after another period
        self._check_model()
        self._note_policy_switch()

    def communication_failure(self, exc: BaseException) -> None:
        """The bridge failed after the client's own re-send: end the episode cleanly."""
        self.stats.communication_failures += 1
        reason = f"communication failure: {type(exc).__name__}: {exc}"
        self._emit("communication_failure", "error", reason, action="abort")
        self._abort(reason)

    # -- detectors --------------------------------------------------------------------------------

    def _check_progress(self, now: float) -> None:
        agent = self.agent
        assert agent is not None
        m = agent.metrics
        resolved = sum(1 for p in agent.world.priors.values() if p.resolved
                       or p.status is PriorStatus3D.CHEST_SEEN)
        marks = (m.chests_opened, resolved, m.targets_reached, len(agent._known_targets))
        if marks != self._progress_marks:
            if marks[0] > self._progress_marks[0] or marks[1] > self._progress_marks[1]:
                self._mark_progress("opened or resolved")
            else:
                self._last_progress = now
            self._progress_marks = marks

    def _mark_progress(self, why: str) -> None:
        agent = self.agent
        assert agent is not None
        self._last_progress = agent.now
        self.loops.progress()
        # Resolved situations start their ladders afresh.
        self._rungs = {k: v for k, v in self._rungs.items() if k[0] not in
                       ("transition_loop", "no_progress", "stationary", "state_timeout")}

    def _check_state_time(self) -> None:
        agent = self.agent
        assert agent is not None
        spec = spec_for(agent.state)
        limit = None
        if self.config.use_state_specs and spec is not None:
            limit = spec.max_seconds
        elif agent.state not in (S.EXPLORING, S.PLANNING_ROUTE, S.RECOVERING,
                                 S.TARGET_DETECTED):
            limit = self.config.max_state_seconds
        if limit is not None and self._state_time > limit:
            self._incident("state_timeout",
                           f"{agent.state} for {self._state_time:.1f} s (limit {limit:.0f} s)",
                           {"limit": limit})
            self._state_time = 0.0

    def _check_stationary(self, now: float) -> None:
        window = self.config.stationary_seconds
        if not self._trail or now - self._trail[0][0] < window:
            return
        recent = [(t, p, mv) for t, p, mv in self._trail if now - t <= window]
        if len(recent) < 3 or recent[-1][0] - recent[0][0] < 0.9 * window:
            return
        moving = sum(1 for _, _, mv in recent if mv) / len(recent)
        if moving < 0.8:
            return
        spread = max(math.dist(recent[0][1][:2], p[:2]) for _, p, _ in recent)
        if spread < self.config.stationary_distance_m:
            self._incident("stationary", f"movement requested for {window:.0f} s, moved "
                           f"{spread:.2f} m", {"moved_m": round(spread, 3)})
            self._trail.clear()

    def _check_targets(self) -> None:
        for loc, n in list(self._goal_plans.items()):
            if n > self.config.max_goal_plans:
                self._incident("repeated_target", f"spawn point {loc} planned {n} times "
                               "without being reached", {"plans": n}, location=loc)
                self._goal_plans[loc] = 0
        for loc, n in list(self._interaction_failures.items()):
            if n >= self.config.max_interaction_failures:
                self._incident("interaction_failures",
                               f"{n} failed interactions at spawn point {loc}", {"failures": n},
                               location=loc)
                self._interaction_failures[loc] = 0

    def _check_model(self) -> None:
        sel = self.selector
        if sel is None:
            return
        fb = sel.stats.fallbacks
        failures = sum(fb.get(k, 0) for k in ("invalid_output", "model_error", "timeout"))
        new = failures - self._model_failures_seen
        if new > 0:
            self._model_failures_seen = failures
            for kind in ("invalid_output", "model_error", "timeout"):
                d = fb.get(kind, 0) - self._last_fallbacks.get(kind, 0)
                if d > 0:
                    self._emit("model_failure", "warning", f"{d} x {kind}")
            self._last_fallbacks = Counter(fb)
            if failures >= self.config.max_model_failures and sel.mode != "deterministic":
                self._incident("model_failure", f"{failures} model failures this episode", {})
        lat = sel.stats.latencies_ms
        if len(lat) >= self.config.latency_window and len(lat) > self._latencies_seen:
            self._latencies_seen = len(lat)
            recent = sorted(lat[-self.config.latency_window:])
            p95 = recent[int(0.95 * (len(recent) - 1))]
            if p95 > self.config.inference_budget_ms and sel.mode != "deterministic":
                self._incident("inference_latency", f"p95 {p95:.1f} ms over the last "
                               f"{len(recent)} inferences (budget "
                               f"{self.config.inference_budget_ms:.0f} ms)", {"p95_ms": p95})

    def _note_policy_switch(self) -> None:
        sel = self.selector
        if sel is None:
            return
        if sel.mode != self._last_mode:
            self.stats.policy_switches += 1
            self._emit("policy_switch", "info", f"{self._last_mode} -> {sel.mode}")
            self._last_mode = sel.mode

    # -- escalation -------------------------------------------------------------------------------

    def _incident(self, kind: str, detail: str, data: dict[str, Any], *,
                  location: int | None = None) -> None:
        agent = self.agent
        assert agent is not None
        self.stats.detections[kind] += 1
        now = agent.now
        if location is None and kind in PER_TARGET and agent.goal is not None:
            location = agent.goal.location_id
        key = (kind, location if kind in PER_TARGET else None)
        if now < self._cooldown.get(str(key), -math.inf):
            self.stats.suppressed[kind] += 1          # the last intervention is taking effect
            return
        if kind == "transition_loop":
            self.loops.progress()                     # a new loop must build up afresh
        ladder = LADDERS[kind]
        rung, used = self._rungs.get(key, (0, 0))
        applied: str | None = None
        while applied is None and rung < len(ladder):
            action = ladder[rung]
            rung += 1
            if used >= self.config.max_recovery_attempts:
                if "abort" not in ladder:
                    rung = len(ladder)              # exhausted: no more help for this incident
                    break
                action = "abort"
            if not self.enforcing:
                applied = f"would {action}"
            elif self._apply(action, f"{kind}: {detail}"):
                applied = action
                used += 1
                self.stats.interventions[action] += 1
            else:
                self.stats.skipped[action] += 1
        self._rungs[key] = (rung, used)
        self._cooldown[str(key)] = now + self.config.cooldown_seconds
        self._emit(kind, "error" if applied == "abort" else "warning", detail,
                   action=applied or "none (ladder exhausted)", location=location, rung=rung,
                   **data)
        if self.enforcing and not agent.is_terminal() \
                and sum(self.stats.interventions.values()) >= self.config.max_interventions:
            self._abort(f"supervisor intervention budget ({self.config.max_interventions}) "
                        "exhausted")

    def _apply(self, action: str, reason: str) -> bool:
        agent = self.agent
        assert agent is not None
        if agent.is_terminal():
            return False
        if action in FSM_ACTIONS and not self.fsm_driving:
            return False
        if action == "replan":
            return agent.supervisor_replan(reason)
        if action == "abandon_target":
            return agent.supervisor_abandon_target(reason)
        if action == "fail_target":
            return agent.supervisor_abandon_target(reason, failed=True)
        if action == "relocalise":
            return agent.supervisor_relocalise(reason)
        if action == "redirect":
            return agent.supervisor_redirect(reason)
        if action == "unstick":
            self._unsticks += 1
            return agent.supervisor_unstick(reason, attempt=self._unsticks)
        if action == "fallback_deterministic":
            sel = self.selector
            if sel is None or sel.mode == "deterministic":
                return False
            sel.set_mode("deterministic")
            self._note_policy_switch()
            return True
        if action == "abort":
            self._abort(f"supervisor: {reason}")
            return True
        raise ValueError(f"unknown intervention {action!r}")

    def _abort(self, reason: str) -> None:
        if self.stats.aborted is None:
            self.stats.aborted = reason
        if self.controller is not None and not self.controller.is_terminal():
            self.controller.stop("failed", reason)

    def _emit(self, kind: str, severity: str, detail: str, *, action: str | None = None,
              location: int | None = None, **data: Any) -> None:
        agent = self.agent
        self.events.emit(kind, severity=severity, sim_time=agent.now if agent else 0.0,
                         state=str(agent.state) if agent else None,
                         goal_location=location if location is not None else (
                             agent.goal.location_id if agent and agent.goal else None),
                         detail=detail, action=action, **data)

    # -- reporting --------------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {"mode": self.config.mode, **self.stats.to_dict(),
                "events": self.events.summary()}
