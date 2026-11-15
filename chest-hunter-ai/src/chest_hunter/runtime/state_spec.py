"""Formal state specifications for the 3D visual agent (Phase 6).

The transition table itself lives with the agent (``controller_3d.TRANSITIONS_3D``) and is
enforced by :class:`chest_hunter.agent.fsm.StateMachine`. This module adds, for every state:

* the Phase 6 category it belongs to (the prompt's state names; the agent keeps its Phase 4
  names, which are stored in session records and dashboards);
* valid entry conditions, checked right after the agent enters the state;
* permitted actions (movement, interaction), checked on every command the state machine
  produces;
* exit conditions (documentation) and permitted exits (the transition table);
* a maximum duration in *simulated* seconds, measured from baseline episodes (see
  ``docs/phase6_baseline_audit.md``) with a wide margin;
* the recovery transition the supervisor uses, and terminal failure conditions.

:class:`LoopDetector` keeps a bounded window of recent states and reports a cycle that
repeats without progress (e.g. NAVIGATING, RECOVERING, OBSERVING, PLANNING_ROUTE, ...).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from chest_hunter.agent.controller_3d import TRANSITIONS_3D, Agent3DState

S = Agent3DState


@dataclass(frozen=True, slots=True)
class StateSpec:
    state: Agent3DState
    category: str                         # Phase 6 state category
    entry: str                            # valid entry conditions (documentation)
    exits: str                            # exit conditions (documentation)
    may_move: bool
    may_interact: bool
    max_seconds: float | None             # simulated seconds; None = instant or terminal
    recovery: Agent3DState | None         # where the supervisor redirects from here
    terminal_failure: str = ""
    check: Callable[[Any], str | None] | None = None   # entry check; message on violation

    @property
    def allowed_targets(self) -> frozenset[Agent3DState]:
        return TRANSITIONS_3D[self.state]


def _needs_goal(agent: Any) -> str | None:
    return None if getattr(agent, "goal", None) is not None else "no goal"


def _needs_route(agent: Any) -> str | None:
    if getattr(agent, "goal", None) is None:
        return "no goal"
    if getattr(agent, "follower", None) is None:
        return "no route to follow"
    return None


def _needs_chest_goal(agent: Any) -> str | None:
    goal = getattr(agent, "goal", None)
    if goal is None:
        return "no goal"
    if goal.kind != "chest":
        return f"goal is a {goal.kind}, not a chest"
    return None


_SPECS: tuple[StateSpec, ...] = (
    StateSpec(S.INITIALIZING, "INITIALIZING", "episode reset, before the first telemetry",
              "first telemetry received -> OBSERVING", False, False, None, None),
    StateSpec(S.OBSERVING, "OBSERVING", "after start, recovery, a failed plan or off-mesh wait",
              "one analysed frame or 1.5 s -> the queued next state", True, False, 10.0,
              S.EXPLORING),
    StateSpec(S.SCANNING, "OBSERVING", "a look-around was queued (start, inspection, rescan)",
              "scan list done, or a new chest target -> TARGET_DETECTED", True, False, 40.0,
              S.EXPLORING),
    StateSpec(S.EXPLORING, "EXPLORING", "no current goal (instant state)",
              "chest target -> TARGET_DETECTED; inspection goal -> PLANNING_ROUTE; "
              "all resolved -> COMPLETED", True, False, None, None,
              "no reachable unresolved spawn point remains (COMPLETED, not a failure)"),
    StateSpec(S.TARGET_DETECTED, "SELECTING_TARGET", "at least one confirmed chest track",
              "goal chosen -> PLANNING_ROUTE or APPROACHING", True, False, None, S.EXPLORING),
    StateSpec(S.PLANNING_ROUTE, "PLANNING_ROUTE", "a goal is set (instant state)",
              "path found -> NAVIGATING; no path -> OBSERVING (goal dropped)", True, False, None,
              S.EXPLORING, check=_needs_goal),
    StateSpec(S.NAVIGATING, "NAVIGATING", "goal and route set",
              "near the goal -> APPROACHING or SCANNING; stuck -> RECOVERING", True, False,
              45.0, S.PLANNING_ROUTE, check=_needs_route),
    StateSpec(S.APPROACHING, "NAVIGATING", "chest goal within the approach distance",
              "at stand distance -> ALIGNING; blocked or 15 s -> RECOVERING", True, False, 20.0,
              S.PLANNING_ROUTE, check=_needs_chest_goal),
    StateSpec(S.ALIGNING, "ALIGNING", "standing next to the chest goal",
              "prompt visible and aimed -> INTERACTING; 1.6 s -> APPROACHING or RECOVERING",
              True, False, 8.0, S.EXPLORING, check=_needs_chest_goal),
    StateSpec(S.INTERACTING, "INTERACTING", "prompt visible, interaction hold started",
              "hold time reached -> VERIFYING_OPEN; interrupted -> RECOVERING", False, True,
              6.0, S.EXPLORING, check=_needs_chest_goal),
    StateSpec(S.VERIFYING_OPEN, "VERIFYING", "hold completed",
              "opened seen (or notified + seen) -> EXPLORING; closed or 2 s -> RECOVERING",
              True, False, 10.0, S.EXPLORING, check=_needs_goal),
    StateSpec(S.RECOVERING, "RECOVERING", "a handler reported a failure reason (instant)",
              "strategy chosen -> APPROACHING, ALIGNING, PLANNING_ROUTE, OBSERVING or "
              "EXPLORING", True, False, None, S.EXPLORING,
              "episode recovery budget exhausted -> FAILED"),
    StateSpec(S.COMPLETED, "COMPLETED", "all spawn points resolved, or a normal stop",
              "terminal", False, False, None, None),
    StateSpec(S.FAILED, "FAILED", "unrecoverable failure (recovery budget, supervisor abort, "
              "communication loss)", "terminal", False, False, None, None),
)

STATE_SPECS_3D: Final[dict[Agent3DState, StateSpec]] = {s.state: s for s in _SPECS}

if set(STATE_SPECS_3D) != set(Agent3DState):          # pragma: no cover - import-time guard
    raise RuntimeError("every agent state needs a specification")


def spec_for(state: Any) -> StateSpec | None:
    try:
        return STATE_SPECS_3D[Agent3DState(str(state))]
    except ValueError:
        return None


def check_entry(state: Any, agent: Any) -> str | None:
    """Message if the agent's situation violates the entry conditions of ``state``."""
    spec = spec_for(state)
    if spec is None or spec.check is None:
        return None
    return spec.check(agent)


def action_violations(state: Any, action: Any) -> list[str]:
    """Commands the state's specification does not permit (state-machine commands only)."""
    spec = spec_for(state)
    if spec is None:
        return []
    out = []
    if not spec.may_interact and getattr(action, "interact", False):
        out.append("interact")
    if not spec.may_move and (abs(getattr(action, "move_forward", 0.0)) > 1e-9
                              or abs(getattr(action, "move_right", 0.0)) > 1e-9):
        out.append("move")
    return out


@dataclass(frozen=True, slots=True)
class Loop:
    period: tuple[str, ...]
    repetitions: int

    @property
    def label(self) -> str:
        return " -> ".join(self.period)


class LoopDetector:
    """Detects a cycle of states repeated without progress in a bounded window.

    ``observe(state)`` is called on every state entry. A loop is reported when the newest
    ``period * repetitions`` entries consist of one cycle of 2 to ``max_period`` distinct-ish
    states repeated at least ``repetitions`` times. ``progress()`` (an opening, a resolved
    spawn point, a reached target) clears the window, so normal work that passes through
    the same states never accumulates into a loop.
    """

    def __init__(self, window: int = 12, repetitions: int = 3, max_period: int = 4) -> None:
        if window < 4:
            raise ValueError("loop window must be at least 4")
        if repetitions < 2:
            raise ValueError("loop repetitions must be at least 2")
        self.window = window
        self.repetitions = repetitions
        self.max_period = max(2, min(max_period, window // repetitions))
        self.recent: deque[str] = deque(maxlen=window)

    def observe(self, state: Any) -> Loop | None:
        self.recent.append(str(state))
        return self.detect()

    def progress(self) -> None:
        self.recent.clear()

    def detect(self) -> Loop | None:
        seq = list(self.recent)
        for period in range(2, self.max_period + 1):
            need = period * self.repetitions
            if len(seq) < need:
                continue
            tail = seq[-need:]
            cycle = tail[:period]
            if len(set(cycle)) < 2:
                continue
            if all(tail[i] == cycle[i % period] for i in range(need)):
                reps = need // period
                # count further repetitions inside the window
                k = len(seq) - need - 1
                while k >= 0 and seq[k] == cycle[(k - (len(seq) - need)) % period]:
                    k -= 1
                reps = (len(seq) - 1 - k) // period
                return Loop(tuple(cycle), reps)
        return None


def describe_specs(states: Sequence[Agent3DState] | None = None) -> list[dict[str, Any]]:
    """Table of the specifications (for documentation and the dashboard)."""
    rows = []
    for st in states or list(Agent3DState):
        spec = STATE_SPECS_3D[st]
        rows.append({"state": st.value, "category": spec.category, "entry": spec.entry,
                     "exits": spec.exits,
                     "allowed_targets": sorted(t.value for t in spec.allowed_targets),
                     "may_move": spec.may_move, "may_interact": spec.may_interact,
                     "max_seconds": spec.max_seconds,
                     "recovery": spec.recovery.value if spec.recovery else None,
                     "terminal_failure": spec.terminal_failure})
    return rows
