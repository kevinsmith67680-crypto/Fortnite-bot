"""Selection between the deterministic Phase 4 controller and the learned policy.

Modes:

* ``deterministic`` - the Phase 4 ``VisualAgent3D`` drives (default).
* ``learned`` - the imitation policy drives every decision. The deterministic agent keeps
  running perception and bookkeeping (world model, empty-spot votes, database records);
  openings are credited when the agent's own "chest opened" notification arrives next to
  a spawn point. On any model failure the decision falls back to the deterministic agent.
* ``hybrid`` - the deterministic agent drives; in the local visual-servoing states
  (approaching and aligning with a visible chest) its movement and camera commands are
  replaced by the learned ones when the model is validated and confident.

Every learned command passes the safety layer: finite values, clamped ranges, bounded
movement speed and camera rates, camera pitch limits, a bounded action duration (a stale
model output is never reused for long), interaction only when the interaction prompt is
visible, an inference time limit and a fallback for invalid output or errors. Failures
never propagate into the simulation loop.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from chest_hunter.agent.controller_3d import Agent3DState, VisualAgent3D
from chest_hunter.learning.learned_policy import LearnedPolicy, PolicyOutput
from chest_hunter.localization.world_model_3d import PriorStatus3D
from chest_hunter.unreal_bridge.action_adapter import ActionCommand
from chest_hunter.unreal_bridge.client import StepResult
from chest_hunter.unreal_bridge.observation_adapter import Telemetry

MODES = ("deterministic", "learned", "hybrid")
HYBRID_STATES = (Agent3DState.APPROACHING, Agent3DState.ALIGNING)


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    max_move: float = 1.0                 # norm of (move_forward, move_right)
    max_look: float = 0.8                 # |look_yaw|, |look_pitch| as rate fractions
    pitch_margin_deg: float = 2.0         # stop pitching this close to the pitch limits
    timeout_ms: float = 20.0              # an inference slower than this is discarded
    max_consecutive_failures: int = 10    # then the learned policy is disabled (episode)
    max_output_age_s: float = 0.5         # reuse a model output at most this long
    confidence: float = 0.6               # hybrid: P(approach or align) needed
    interact_threshold: float = 0.5
    learned_observation_rate: float = 8.0  # frames per simulated second in learned mode
    stuck_seconds: float = 4.0            # learned mode: no progress -> deterministic help
    recovery_seconds: float = 3.0
    max_recoveries: int = 25
    attribution_radius_m: float = 2.5


@dataclass(slots=True)
class SelectorStats:
    decisions: int = 0
    learned_decisions: int = 0
    deterministic_decisions: int = 0
    inferences: int = 0
    fallbacks: Counter[str] = field(default_factory=Counter)
    interact_blocked: int = 0
    interaction_holds: int = 0
    disabled_reason: str | None = None
    learned_recoveries: int = 0
    openings_attributed: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms)
        return {
            "decisions": self.decisions, "learned_decisions": self.learned_decisions,
            "deterministic_decisions": self.deterministic_decisions,
            "learned_share": round(self.learned_decisions / self.decisions, 4)
            if self.decisions else 0.0,
            "inferences": self.inferences, "fallbacks": dict(self.fallbacks),
            "interact_blocked": self.interact_blocked,
            "interaction_holds": self.interaction_holds, "disabled_reason": self.disabled_reason,
            "learned_recoveries": self.learned_recoveries,
            "openings_attributed": self.openings_attributed,
            "latency_mean_ms": round(sum(lat) / len(lat), 4) if lat else None,
            "latency_p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 4) if lat else None,
            "latency_max_ms": round(lat[-1], 4) if lat else None,
        }


class PolicySelector:
    """Drop-in controller for ``Runner3D`` wrapping a ``VisualAgent3D``."""

    def __init__(self, agent: VisualAgent3D, policy: LearnedPolicy | None, *,
                 mode: str = "deterministic", limits: SafetyLimits | None = None,
                 require_validation: bool = True) -> None:
        self.agent = agent
        self.policy = policy
        self.limits = limits or SafetyLimits()
        self.require_validation = require_validation
        self.stats = SelectorStats()
        self.action = ActionCommand()
        self._mode = "deterministic"
        self._last: PolicyOutput | None = None
        self._last_time = -math.inf
        self._last_frame_id = -1
        self._failures = 0
        self._disabled = False
        self._recover_until = -math.inf
        self._anchor: tuple[tuple[float, float] | None, float] = (None, 0.0)
        self._hold_started: float | None = None
        self._next_observation = 0.0
        self.set_mode(mode)

    # -- mode -------------------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch policies (also at run time, e.g. from the dashboard)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if mode != "deterministic" and self.policy is None:
            raise ValueError(f"mode '{mode}' needs a learned policy")
        self._mode = mode

    @property
    def learned_available(self) -> bool:
        if self.policy is None or self._disabled:
            return False
        return not (self.require_validation and self._mode == "hybrid"
                    and not self.policy.validated())

    # -- runner interface ---------------------------------------------------------------------

    def initialize(self, telemetry: Telemetry) -> None:
        self.agent.initialize(telemetry)
        self._next_observation = telemetry.sim_time

    def is_terminal(self) -> bool:
        return self.agent.is_terminal()

    def stop(self, outcome: str, reason: str) -> None:
        self.agent.stop(outcome, reason)

    @property
    def outcome(self) -> str | None:
        return self.agent.outcome

    @property
    def end_reason(self) -> str | None:
        return self.agent.end_reason

    @property
    def metrics(self) -> Any:
        return self.agent.metrics

    def __getattr__(self, name: str) -> Any:            # world, obs, telemetry, goal, ...
        return getattr(self.agent, name)

    def wants_observation(self) -> bool:
        if self._mode != "learned" or self._in_recovery():
            return self.agent.wants_observation()
        step = self.agent.step_seconds
        return self.agent.now + step + 1e-9 >= self._next_observation

    def tick(self) -> Any:
        self.stats.decisions += 1
        if self._mode == "deterministic":
            return self._deterministic()
        if self._mode == "hybrid":
            state = self.agent.tick()
            self.action = self.agent.action
            if state in HYBRID_STATES and not self.agent.is_terminal():
                out = self._infer("hybrid")
                if out is not None:
                    if out.phase_probability("approach", "align") >= self.limits.confidence:
                        learned = self._safe(out, allow_interact=False)
                        self.action.move_forward = learned.move_forward
                        self.action.move_right = learned.move_right
                        self.action.look_yaw = learned.look_yaw
                        self.action.look_pitch = learned.look_pitch
                        self.stats.learned_decisions += 1
                        return state
                    self.stats.fallbacks["low_confidence"] += 1
            self.stats.deterministic_decisions += 1
            return state
        # learned
        if self.agent.world.all_resolved:
            self.agent.stop("completed", "all spawn points resolved")
            return self.agent.state
        if self._in_recovery():
            return self._safe_fallback()
        out = self._infer("learned")
        if out is None:
            return self._safe_fallback()
        self.action = self._safe(out, allow_interact=True)
        self.stats.learned_decisions += 1
        self._check_progress()
        return self.agent.state

    def on_step(self, result: StepResult) -> None:
        self.agent.on_step(result)
        if result.observation is not None and self._mode == "learned":
            self._next_observation = self.agent.now + 1.0 / self.limits.learned_observation_rate
        if self._mode != "learned":
            return
        if self.action.interact and self._hold_started is None:
            self._hold_started = self.agent.now
            self.stats.interaction_holds += 1
        elif not self.action.interact:
            self._hold_started = None
        for event in result.telemetry.events:
            if event.get("type") == "chest_opened_notification":
                self._attribute_opening(bool(event.get("reward")))

    # -- internals ----------------------------------------------------------------------------

    def _deterministic(self) -> Any:
        state = self.agent.tick()
        self.action = self.agent.action
        self.stats.deterministic_decisions += 1
        return state

    def _safe_fallback(self) -> Any:
        """Learned mode: one deterministic decision. The state machine was idle while the
        learned policy drove, so an error in it must not end the episode: stand still."""
        try:
            return self._deterministic()
        except RuntimeError:
            self.stats.fallbacks["fallback_error"] += 1
            self.action = ActionCommand()
            return self.agent.state

    def _in_recovery(self) -> bool:
        return self.agent.now < self._recover_until

    def _infer(self, context: str) -> PolicyOutput | None:
        if not self.learned_available:
            self.stats.fallbacks["policy_unavailable"] += 1
            return None
        obs = self.agent.obs
        if obs is None:
            self.stats.fallbacks["no_observation"] += 1
            return None
        now = self.agent.now
        if obs.frame_id == self._last_frame_id and self._last is not None:
            if now - self._last_time <= self.limits.max_output_age_s:
                return self._last
            self.stats.fallbacks["stale_output"] += 1
            return None
        assert self.policy is not None
        try:
            out = self.policy.predict(obs.frame, self.agent.telemetry)
        except Exception:                                 # never crash the episode
            return self._failure("model_error")
        self.stats.inferences += 1
        self.stats.latencies_ms.append(out.latency_ms)
        if not out.finite():
            return self._failure("invalid_output")
        if out.latency_ms > self.limits.timeout_ms:
            return self._failure("timeout")
        self._failures = 0
        self._last, self._last_time, self._last_frame_id = out, now, obs.frame_id
        return out

    def _failure(self, reason: str) -> PolicyOutput | None:
        """Count a failed inference; always returns ``None`` (= fall back)."""
        self.stats.fallbacks[reason] += 1
        self._failures += 1
        if self._failures >= self.limits.max_consecutive_failures:
            self._disabled = True
            self.stats.disabled_reason = f"{self._failures} consecutive failures ({reason})"
        return None

    def _safe(self, out: PolicyOutput, *, allow_interact: bool) -> ActionCommand:
        lim = self.limits
        fwd = max(-1.0, min(1.0, out.move_forward))
        right = max(-1.0, min(1.0, out.move_right))
        norm = math.hypot(fwd, right)
        if norm > lim.max_move:
            fwd, right = fwd * lim.max_move / norm, right * lim.max_move / norm
        yaw = max(-lim.max_look, min(lim.max_look, out.look_yaw))
        pitch = max(-lim.max_look, min(lim.max_look, out.look_pitch))
        tel = self.agent.telemetry
        if tel is not None:
            p_min = float(self.agent.rules.get("pitch_min_deg", -55.0)) + lim.pitch_margin_deg
            p_max = float(self.agent.rules.get("pitch_max_deg", 25.0)) - lim.pitch_margin_deg
            if (tel.camera.pitch_deg <= p_min and pitch < 0) or \
                    (tel.camera.pitch_deg >= p_max and pitch > 0):
                pitch = 0.0
        interact = False
        if allow_interact and out.interact_prob >= lim.interact_threshold:
            cues = self.agent.cues
            if cues.prompt or cues.progress:              # interaction precondition
                interact = True
                fwd, right = 0.0, 0.0                     # hold still while holding the button
            else:
                self.stats.interact_blocked += 1
        grounded = tel is None or tel.odometry.grounded
        jump = allow_interact and out.jump_prob >= 0.5 and grounded and not interact
        return ActionCommand(move_forward=fwd, move_right=right, look_yaw=yaw, look_pitch=pitch,
                             jump=jump, interact=interact)

    def _check_progress(self) -> None:
        """Learned mode: hand control to the deterministic agent for a few seconds when the
        learned commands keep the agent in place (bounded number of times per episode)."""
        now = self.agent.now
        pos = self.agent.position
        anchor, since = self._anchor
        if anchor is None or math.dist(pos[:2], anchor) > 1.0 or self.action.interact:
            self._anchor = ((pos[0], pos[1]), now)
            return
        if now - since > self.limits.stuck_seconds \
                and self.stats.learned_recoveries < self.limits.max_recoveries:
            self.stats.learned_recoveries += 1
            self.stats.fallbacks["stuck_recovery"] += 1
            self._recover_until = now + self.limits.recovery_seconds
            self._anchor = ((pos[0], pos[1]), now)

    def _attribute_opening(self, reward: bool) -> None:
        """The agent's own notification says it opened a chest: credit the unresolved spawn
        point next to the agent (estimated position), if there is exactly one candidate."""
        pos = self.agent.position
        candidates = [p for p in self.agent.world.priors.values()
                      if p.status not in (PriorStatus3D.OPENED, PriorStatus3D.EMPTY)
                      and math.dist(pos[:2], p.xyz[:2]) <= self.limits.attribution_radius_m
                      and abs(p.z - pos[2]) <= 1.0]
        if not candidates:
            self.stats.fallbacks["unattributed_opening"] += 1
            return
        prior = min(candidates, key=lambda p: math.dist(pos[:2], p.xyz[:2]))
        seconds = self.agent.now - self._hold_started if self._hold_started is not None else 0.0
        self.agent.credit_opening(prior.location_id, seconds, reward)
        self.stats.openings_attributed += 1
        self._hold_started = None
