"""Shared finite-state-machine infrastructure for autonomous controllers.

Both the Phase 2 oracle controller (:mod:`chest_hunter.agent.controller`) and
the Phase 3 visual controller (:mod:`chest_hunter.agent.visual_controller`)
are policies built on this one engine, so tick handling, transition
validation, history, external stops and terminal handling behave the same.

Tick model: instant states may chain within one tick; *timed* states
advance the simulation for the whole tick, so at most one timed handler runs
per tick. A cap on transitions per tick turns any accidental cycle into a
clean FAILED outcome instead of an infinite loop.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=StrEnum)

MAX_TRANSITIONS_PER_TICK = 16


@dataclass(frozen=True, slots=True)
class Transition:
    tick: int
    source: StrEnum
    target: StrEnum
    reason: str | None


class StateMachine(Generic[S]):
    """Generic tick-driven FSM with a validated transition table."""

    def __init__(
        self,
        *,
        initial: S,
        completed: S,
        failed: S,
        transitions: Mapping[S, frozenset[S]],
        timed: frozenset[S],
        tick_source: Callable[[], int],
        recovering: S | None = None,
    ) -> None:
        missing = set(type(initial)) - set(transitions)
        if missing:
            raise ValueError(f"Transition table lacks states: {sorted(missing)}")
        self.state: S = initial
        self._initial = initial
        self._completed = completed
        self._failed = failed
        self._recovering = recovering
        self._table = transitions
        self._timed = timed
        self._tick_source = tick_source
        self._handlers: dict[S, Callable[[], S | None]] = {}
        self.history: deque[Transition] = deque(maxlen=1000)
        self.transition_counts: Counter[S] = Counter()
        self.recovery_reason: str | None = None
        self.outcome: str | None = None
        self.end_reason: str | None = None

    # -- queries --------------------------------------------------------------------

    def is_terminal(self, state: S | None = None) -> bool:
        state = self.state if state is None else state
        return state is self._completed or state is self._failed

    def can_transition(self, source: S, target: S) -> bool:
        return target in self._table[source]

    # -- driving --------------------------------------------------------------------

    def tick(self) -> S:
        """Process one simulation step and return the resulting state."""
        consumed = False
        for _ in range(MAX_TRANSITIONS_PER_TICK):
            state = self.state
            if self.is_terminal(state) or state is self._initial:
                return state
            if state in self._timed and consumed:
                return state
            next_state = self._handlers[state]()
            if state in self._timed:
                consumed = True
            if next_state is None:
                return self.state
            self._transition(next_state, self._reason_for(next_state))
            if self.is_terminal():
                return self.state
        self.outcome = "failed"
        self.end_reason = "state machine did not settle within one tick"
        self._transition(self._failed, self.end_reason)
        return self.state

    def stop(self, outcome: str, reason: str) -> None:
        """End the episode from outside (tick limit, user cancel)."""
        if self.is_terminal():
            return
        self.outcome = outcome
        self.end_reason = reason
        ends_normally = outcome in ("completed", "tick_limit")
        self._transition(self._completed if ends_normally else self._failed, reason)

    # -- helpers for subclasses ---------------------------------------------------------

    def _complete(self, reason: str) -> S:
        self.outcome = "completed"
        self.end_reason = reason
        return self._completed

    def _fail(self, reason: str) -> S:
        self.outcome = "failed"
        self.end_reason = reason
        return self._failed

    def _reason_for(self, state: S) -> str | None:
        if self._recovering is not None and state is self._recovering:
            return self.recovery_reason
        if self.is_terminal(state):
            return self.end_reason
        return None

    def _transition(self, target: S, reason: str | None = None) -> None:
        if not self.can_transition(self.state, target):
            raise RuntimeError(f"Invalid agent transition {self.state} -> {target}")
        tick = self._tick_source()
        self.history.append(Transition(tick, self.state, target, reason))
        self.transition_counts[target] += 1
        logger.debug("tick %d: %s -> %s%s", tick, self.state, target,
                     f" ({reason})" if reason else "")
        self.state = target
        if self.is_terminal(target):
            self._on_terminal()

    def _on_terminal(self) -> None:
        """Hook called once when COMPLETED or FAILED is entered."""
