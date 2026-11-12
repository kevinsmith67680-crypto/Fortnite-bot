"""Finite-state machine states and the allowed transitions between them."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class AgentState(StrEnum):
    INITIALIZING = "INITIALIZING"
    SELECTING_TARGET = "SELECTING_TARGET"
    PLANNING_ROUTE = "PLANNING_ROUTE"
    NAVIGATING = "NAVIGATING"
    VERIFYING_CHEST = "VERIFYING_CHEST"
    INTERACTING = "INTERACTING"
    CONFIRMING_OPEN = "CONFIRMING_OPEN"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def consumes_time(self) -> bool:
        """States that advance the agent's physics for a whole tick."""
        return self in TIMED_STATES


TERMINAL_STATES: Final = frozenset({AgentState.COMPLETED, AgentState.FAILED})
TIMED_STATES: Final = frozenset({AgentState.NAVIGATING, AgentState.INTERACTING})

_ANY_END: Final = frozenset({AgentState.COMPLETED, AgentState.FAILED})

TRANSITIONS: Final[dict[AgentState, frozenset[AgentState]]] = {
    AgentState.INITIALIZING: frozenset({AgentState.SELECTING_TARGET}) | _ANY_END,
    AgentState.SELECTING_TARGET: frozenset({AgentState.PLANNING_ROUTE}) | _ANY_END,
    AgentState.PLANNING_ROUTE: frozenset({
        AgentState.NAVIGATING, AgentState.VERIFYING_CHEST,
        AgentState.SELECTING_TARGET, AgentState.RECOVERING,
    }) | _ANY_END,
    AgentState.NAVIGATING: frozenset({
        AgentState.VERIFYING_CHEST, AgentState.RECOVERING,
    }) | _ANY_END,
    AgentState.VERIFYING_CHEST: frozenset({
        AgentState.INTERACTING, AgentState.SELECTING_TARGET, AgentState.RECOVERING,
    }) | _ANY_END,
    AgentState.INTERACTING: frozenset({
        AgentState.CONFIRMING_OPEN, AgentState.RECOVERING,
    }) | _ANY_END,
    AgentState.CONFIRMING_OPEN: frozenset({
        AgentState.SELECTING_TARGET, AgentState.RECOVERING,
    }) | _ANY_END,
    AgentState.RECOVERING: frozenset({
        AgentState.PLANNING_ROUTE, AgentState.INTERACTING, AgentState.SELECTING_TARGET,
    }) | _ANY_END,
    AgentState.COMPLETED: frozenset(),
    AgentState.FAILED: frozenset(),
}


def is_valid_transition(source: AgentState, target: AgentState) -> bool:
    return target in TRANSITIONS[source]


# ---------------------------------------------------------------------------
# Phase 3: visual agent (separate enum so the oracle table is unchanged)
# ---------------------------------------------------------------------------


class VisualAgentState(StrEnum):
    """States of the vision-based policy.

    ==================  =========================================  ==========================
    State               Entry / work                               Exits (failure handling)
    ==================  =========================================  ==========================
    INITIALIZING        session opened, priors loaded              OBSERVING
    OBSERVING (timed)   hold still, request a frame                LOCALIZING
    LOCALIZING          need a confident pose fix                  EXPLORING / TARGET_DETECTED;
                                                                   OBSERVING (retry);
                                                                   RECOVERING (pose lost)
    EXPLORING           pick a frontier / inspection goal          PLANNING_ROUTE;
                                                                   TARGET_DETECTED;
                                                                   COMPLETED (nothing left)
    TARGET_DETECTED     pick the best visible unopened chest       PLANNING_ROUTE; EXPLORING
    PLANNING_ROUTE      A* on the estimated occupancy grid         NAVIGATING; ALIGNING (already
                                                                   there); EXPLORING /
                                                                   TARGET_DETECTED (no path)
    NAVIGATING (timed)  follow waypoints on the estimated pose     ALIGNING / EXPLORING
                                                                   (arrived); PLANNING_ROUTE
                                                                   (wall found); OBSERVING (low
                                                                   confidence); RECOVERING
    ALIGNING (timed)    stop, confirm target + interaction prompt  INTERACTING; PLANNING_ROUTE
                                                                   (move closer); RECOVERING
    INTERACTING (timed) hold interact for the interaction time     VERIFYING_OPEN; RECOVERING
    VERIFYING_OPEN (t)  look for the opened-chest appearance       EXPLORING (success);
                                                                   RECOVERING (still closed)
    RECOVERING          bounded retry / replan / abandon           OBSERVING, PLANNING_ROUTE,
                                                                   ALIGNING, EXPLORING; FAILED
    COMPLETED/FAILED    terminal                                   -
    ==================  =========================================  ==========================
    """

    INITIALIZING = "INITIALIZING"
    OBSERVING = "OBSERVING"
    LOCALIZING = "LOCALIZING"
    EXPLORING = "EXPLORING"
    TARGET_DETECTED = "TARGET_DETECTED"
    PLANNING_ROUTE = "PLANNING_ROUTE"
    NAVIGATING = "NAVIGATING"
    ALIGNING = "ALIGNING"
    INTERACTING = "INTERACTING"
    VERIFYING_OPEN = "VERIFYING_OPEN"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


V = VisualAgentState
_V_END: Final = frozenset({V.COMPLETED, V.FAILED})

VISUAL_TIMED_STATES: Final = frozenset({V.OBSERVING, V.NAVIGATING, V.ALIGNING, V.INTERACTING,
                                        V.VERIFYING_OPEN})

VISUAL_TRANSITIONS: Final[dict[VisualAgentState, frozenset[VisualAgentState]]] = {
    V.INITIALIZING: frozenset({V.OBSERVING}) | _V_END,
    V.OBSERVING: frozenset({V.LOCALIZING, V.RECOVERING}) | _V_END,
    V.LOCALIZING: frozenset({V.EXPLORING, V.TARGET_DETECTED, V.OBSERVING,
                             V.RECOVERING}) | _V_END,
    V.EXPLORING: frozenset({V.TARGET_DETECTED, V.PLANNING_ROUTE, V.OBSERVING}) | _V_END,
    V.TARGET_DETECTED: frozenset({V.PLANNING_ROUTE, V.EXPLORING}) | _V_END,
    V.PLANNING_ROUTE: frozenset({V.NAVIGATING, V.ALIGNING, V.EXPLORING, V.TARGET_DETECTED,
                                 V.RECOVERING}) | _V_END,
    V.NAVIGATING: frozenset({V.ALIGNING, V.EXPLORING, V.TARGET_DETECTED, V.PLANNING_ROUTE,
                             V.OBSERVING, V.RECOVERING}) | _V_END,
    V.ALIGNING: frozenset({V.INTERACTING, V.PLANNING_ROUTE, V.EXPLORING, V.TARGET_DETECTED,
                           V.RECOVERING}) | _V_END,
    V.INTERACTING: frozenset({V.VERIFYING_OPEN, V.RECOVERING}) | _V_END,
    V.VERIFYING_OPEN: frozenset({V.EXPLORING, V.RECOVERING}) | _V_END,
    V.RECOVERING: frozenset({V.OBSERVING, V.PLANNING_ROUTE, V.ALIGNING,
                             V.EXPLORING}) | _V_END,
    V.COMPLETED: frozenset(),
    V.FAILED: frozenset(),
}
