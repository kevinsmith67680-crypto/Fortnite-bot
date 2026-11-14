"""Simulated clock and batched persistence of chest observations.

Observations are buffered in memory and written to SQLite in one
transaction per batch (``agent.observation_batch_size``), each keeping the
simulated time at which it happened. The buffer is always flushed before a
session is completed or aborted, so the database ends up identical to
unbatched recording; at most one batch can be lost if the process is killed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from chest_hunter.database import DatabaseManager
from chest_hunter.models import utc_now
from chest_hunter.session_repository import SessionRepository

logger = logging.getLogger(__name__)


class _HasTime(Protocol):
    @property
    def time(self) -> float: ...


class SimulationClock:
    """``base`` wall-clock instant plus the environment's simulated elapsed time.

    Session ``started_at`` is the real start time; every later timestamp
    (observations, ``ended_at``) lies on the simulated timeline, so session
    durations and chests-per-minute reflect simulated time.
    """

    def __init__(self, source: _HasTime, base: datetime | None = None) -> None:
        self.source = source
        self.base = base if base is not None else utc_now()

    def __call__(self) -> datetime:
        return self.base + timedelta(seconds=self.source.time)

    def at(self, seconds: float) -> datetime:
        return self.base + timedelta(seconds=seconds)


@dataclass(frozen=True, slots=True)
class PendingObservation:
    chest_id: int
    present: bool
    opened: bool
    interaction_seconds: float
    observed_at: datetime
    reward_received: bool | None = None


class ObservationRecorder:
    """Buffers observations for one session and flushes them in batches."""

    def __init__(self, db: DatabaseManager, sessions: SessionRepository, session_id: int,
                 *, batch_size: int = 10) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._db = db
        self._sessions = sessions
        self.session_id = session_id
        self.batch_size = batch_size
        self._pending: list[PendingObservation] = []
        self.recorded = 0
        self.flushes = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def record(self, chest_id: int, *, present: bool, opened: bool,
               interaction_seconds: float, observed_at: datetime,
               reward_received: bool | None = None) -> None:
        self._pending.append(PendingObservation(chest_id, present, opened,
                                                interaction_seconds, observed_at,
                                                reward_received))
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """Write all buffered observations atomically; return how many were written."""
        if not self._pending:
            return 0
        batch, self._pending = self._pending, []
        with self._db.transaction():
            for item in batch:
                self._sessions.record_observation(
                    self.session_id,
                    item.chest_id,
                    chest_present=item.present,
                    chest_opened=item.opened,
                    interaction_seconds=item.interaction_seconds,
                    observed_at=item.observed_at,
                    reward_received=item.reward_received,
                )
        self.recorded += len(batch)
        self.flushes += 1
        logger.debug("Flushed %d observation(s) for session %d", len(batch), self.session_id)
        return len(batch)
