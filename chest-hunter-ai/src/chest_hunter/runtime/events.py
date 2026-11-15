"""Structured runtime events (Phase 6).

Every condition the supervisor detects, every intervention it applies, every rejected state
transition and every policy switch becomes one :class:`RuntimeEvent`. Events are kept in a
bounded log (the newest ``capacity`` events, plus exact counts of all events by kind), so a
long run cannot accumulate them without limit. They can be exported as JSON lines.

Times: ``sim_time`` is simulated seconds (episode clock); ``wall_time`` is real seconds since
the log was created. The two are never mixed.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger("chest_hunter.runtime")

SEVERITIES: Final = ("info", "warning", "error")


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    seq: int
    kind: str                       # e.g. "state_timeout", "intervention", "policy_switch"
    severity: str                   # info | warning | error
    sim_time: float
    wall_time: float
    episode: int | None = None
    state: str | None = None
    goal_location: int | None = None
    detail: str = ""
    action: str | None = None       # intervention applied, if any
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLog:
    """Bounded log of runtime events with exact per-kind counts."""

    def __init__(self, capacity: int = 500, *, episode: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.episode = episode
        self._events: deque[RuntimeEvent] = deque(maxlen=capacity)
        self.counts: Counter[str] = Counter()
        self.by_severity: Counter[str] = Counter()
        self._seq = 0
        self._wall0 = time.perf_counter()

    def emit(self, kind: str, *, severity: str = "warning", sim_time: float = 0.0,
             state: str | None = None, goal_location: int | None = None, detail: str = "",
             action: str | None = None, **data: Any) -> RuntimeEvent:
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        self._seq += 1
        event = RuntimeEvent(self._seq, kind, severity, round(float(sim_time), 4),
                             round(time.perf_counter() - self._wall0, 4), self.episode, state,
                             goal_location, detail, action, dict(data))
        self._events.append(event)
        self.counts[kind] += 1
        self.by_severity[severity] += 1
        level = {"info": logging.DEBUG, "warning": logging.INFO, "error": logging.WARNING}
        logger.log(level[severity], "[%s] t=%.2f %s %s%s", kind, sim_time, state or "-",
                   detail, f" -> {action}" if action else "")
        return event

    def __iter__(self) -> Iterator[RuntimeEvent]:
        return iter(list(self._events))

    def __len__(self) -> int:
        return len(self._events)

    @property
    def total(self) -> int:
        return self._seq

    @property
    def dropped(self) -> int:
        """Events no longer held (older than the newest ``capacity``); still counted."""
        return self._seq - len(self._events)

    def recent(self, n: int = 10) -> list[RuntimeEvent]:
        return list(self._events)[-n:]

    def of_kind(self, kind: str) -> list[RuntimeEvent]:
        return [e for e in self._events if e.kind == kind]

    def summary(self) -> dict[str, Any]:
        return {"total": self._seq, "held": len(self._events), "dropped": self.dropped,
                "by_kind": dict(self.counts), "by_severity": dict(self.by_severity)}

    def write_jsonl(self, path: str | Path, *, append: bool = True) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a" if append else "w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return len(self._events)
