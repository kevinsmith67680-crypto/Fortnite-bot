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
