"""Simulation sessions, chest observations and per-session statistics.

Session lifecycle::

    start_session() -> running --complete_session()--> completed
                              \\--abort_session()-----> aborted

``completed`` and ``aborted`` are final: a finished session cannot be
finished again and does not accept new observations. These rules are checked
here and also enforced by SQLite triggers.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any, Final

from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import (
    DatabaseIntegrityError,
    DuplicateOpeningError,
    RecordNotFoundError,
    SessionStateError,
    ValidationError,
)
from chest_hunter.models import (
    MAX_NOTES_LENGTH,
    ChestObservation,
    Clock,
    SessionStatistics,
    SessionStatus,
    SimulationSession,
    format_timestamp,
    parse_timestamp,
    require_bool,
    require_map_id,
    require_non_negative,
    require_optional_text,
    require_positive_int,
    timestamp_now,
    utc_now,
)

logger = logging.getLogger(__name__)

_SESSION_COLUMNS: Final = "id, map_id, started_at, ended_at, status, notes"
_OBSERVATION_COLUMNS: Final = (
    "id, session_id, chest_location_id, observed_at, chest_present, "
    "chest_opened, interaction_seconds"
)


def _row_to_session(row: sqlite3.Row) -> SimulationSession:
    return SimulationSession(
        id=row["id"],
        map_id=row["map_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        status=SessionStatus(row["status"]),
        notes=row["notes"],
    )


def _row_to_observation(row: sqlite3.Row) -> ChestObservation:
    return ChestObservation(
        id=row["id"],
        session_id=row["session_id"],
        chest_location_id=row["chest_location_id"],
        observed_at=row["observed_at"],
        chest_present=bool(row["chest_present"]),
        chest_opened=bool(row["chest_opened"]),
        interaction_seconds=row["interaction_seconds"],
    )


class SessionRepository:
    """Data access for simulation sessions and their chest observations.

    ``clock`` is injectable so a simulator can run on simulated time and tests
    can produce exact durations.
    """

    def __init__(self, db: DatabaseManager, *, clock: Clock = utc_now) -> None:
        self._db = db
        self._clock = clock

    # -- lifecycle ----------------------------------------------------------

    def start_session(self, map_id: str, *, notes: str | None = None) -> SimulationSession:
        """Create a new ``running`` session on ``map_id`` and return it."""
        session = SimulationSession(
            id=None,
            map_id=map_id,
            started_at=timestamp_now(self._clock),
            ended_at=None,
            status=SessionStatus.RUNNING,
            notes=require_optional_text("notes", notes, MAX_NOTES_LENGTH),
        )
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO simulation_sessions (map_id, started_at, ended_at, status, notes) "
                "VALUES (?, ?, NULL, ?, ?)",
                (session.map_id, session.started_at, session.status.value, session.notes),
            )
            session.id = cursor.lastrowid
        logger.debug("Started session %s on map %s", session.id, session.map_id)
        return session

    def complete_session(self, session_id: int) -> SimulationSession:
        """Mark a running session as ``completed``.

        :raises RecordNotFoundError: if the session does not exist.
        :raises SessionStateError: if the session has already finished.
        """
        return self._finish(session_id, SessionStatus.COMPLETED)

    def abort_session(self, session_id: int) -> SimulationSession:
        """Mark a running session as ``aborted``.

        :raises RecordNotFoundError: if the session does not exist.
        :raises SessionStateError: if the session has already finished.
        """
        return self._finish(session_id, SessionStatus.ABORTED)

    def _finish(self, session_id: int, status: SessionStatus) -> SimulationSession:
        require_positive_int("session_id", session_id)
        verb = "completed" if status is SessionStatus.COMPLETED else "aborted"
        try:
            with self._db.transaction() as conn:
                session = self._fetch_session(conn, session_id)
                if session.status is not SessionStatus.RUNNING:
                    raise SessionStateError(
                        f"Session {session_id} has already finished with status "
                        f"'{session.status.value}'; it cannot be {verb}."
                    )
                ended = self._clock()
                started = parse_timestamp(session.started_at)
                if ended < started:
                    logger.warning(
                        "Clock is earlier than the start of session %d; "
                        "recording a zero duration.",
                        session_id,
                    )
                    ended = started
                ended_at = format_timestamp(ended)
                updated = conn.execute(
                    "UPDATE simulation_sessions SET status = ?, ended_at = ? "
                    "WHERE id = ? AND status = ?",
                    (status.value, ended_at, session_id, SessionStatus.RUNNING.value),
                ).rowcount
                if updated != 1:  # pragma: no cover - guarded by BEGIN IMMEDIATE
                    raise SessionStateError(f"Session {session_id} is no longer running.")
        except DatabaseIntegrityError as exc:
            raise SessionStateError(f"Session {session_id} could not be {verb}: {exc}") from exc
        session.status = status
        session.ended_at = ended_at
        session.validate()
        logger.debug("Session %d %s", session_id, verb)
        return session

    # -- reads --------------------------------------------------------------

    def get_session(self, session_id: int) -> SimulationSession | None:
        """Return the session, or ``None`` if it does not exist."""
        require_positive_int("session_id", session_id)
        row = self._db.query_one(
            f"SELECT {_SESSION_COLUMNS} FROM simulation_sessions WHERE id = ?", (session_id,)
        )
        return None if row is None else _row_to_session(row)

    def require_session(self, session_id: int) -> SimulationSession:
        """Like :meth:`get_session` but raise :class:`RecordNotFoundError` if missing."""
        session = self.get_session(session_id)
        if session is None:
            raise RecordNotFoundError(f"Session {session_id} does not exist.")
        return session

    def list_sessions(
        self,
        *,
        map_id: str | None = None,
        status: SessionStatus | str | None = None,
        newest_first: bool = False,
        limit: int | None = None,
    ) -> list[SimulationSession]:
        """Return sessions ordered by ID (creation order), optionally filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        if map_id is not None:
            clauses.append("map_id = ?")
            params.append(require_map_id(map_id))
        if status is not None:
            try:
                params.append(SessionStatus(status).value)
            except ValueError as exc:
                allowed = ", ".join(s.value for s in SessionStatus)
                raise ValidationError(f"status must be one of {allowed}; got {status!r}.") from exc
            clauses.append("status = ?")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "DESC" if require_bool("newest_first", newest_first) else "ASC"
        sql = f"SELECT {_SESSION_COLUMNS} FROM simulation_sessions {where} ORDER BY id {order}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(require_positive_int("limit", limit))
        return [_row_to_session(row) for row in self._db.query_all(sql, params)]

    def get_observations(self, session_id: int) -> list[ChestObservation]:
        """Return the observations of a session in the order they were recorded."""
        self.require_session(session_id)
        rows = self._db.query_all(
            f"SELECT {_OBSERVATION_COLUMNS} FROM chest_observations "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [_row_to_observation(row) for row in rows]

    # -- observations -------------------------------------------------------

    def record_observation(
        self,
        session_id: int,
        chest_location_id: int,
        *,
        chest_present: bool,
        chest_opened: bool = False,
        interaction_seconds: float = 0.0,
        observed_at: datetime | None = None,
    ) -> ChestObservation:
        """Record the outcome of visiting a chest location during a running session.

        All checks and the insert run in a single ``BEGIN IMMEDIATE``
        transaction, so they cannot race with another writer.

        ``observed_at`` defaults to the repository clock. Callers that buffer
        observations (the Phase 2 agent) pass the time of the event itself; it
        must be timezone-aware and not earlier than the session start.

        :raises ValidationError: invalid values, opened-but-absent, a location
            on another map, or a disabled location.
        :raises RecordNotFoundError: unknown session or chest location.
        :raises SessionStateError: the session has already finished.
        :raises DuplicateOpeningError: the location was already opened
            successfully in this session.
        """
        observation = ChestObservation(
            id=None,
            session_id=session_id,
            chest_location_id=chest_location_id,
            observed_at=(
                timestamp_now(self._clock) if observed_at is None
                else format_timestamp(observed_at)
            ),
            chest_present=chest_present,
            chest_opened=chest_opened,
            interaction_seconds=require_non_negative("interaction_seconds", interaction_seconds),
        )
        try:
            with self._db.transaction() as conn:
                session = self._fetch_session(conn, observation.session_id)
                if session.status is not SessionStatus.RUNNING:
                    raise SessionStateError(
                        f"Session {session.id} is {session.status.value}; observations can "
                        "only be recorded for running sessions."
                    )
                if parse_timestamp(observation.observed_at) < parse_timestamp(
                    session.started_at
                ):
                    raise ValidationError(
                        f"observed_at {observation.observed_at} is earlier than the start "
                        f"of session {session.id} ({session.started_at})."
                    )
                chest = conn.execute(
                    "SELECT id, map_id, enabled FROM chest_locations WHERE id = ?",
                    (observation.chest_location_id,),
                ).fetchone()
                if chest is None:
                    raise RecordNotFoundError(
                        f"Chest location {observation.chest_location_id} does not exist."
                    )
                if chest["map_id"] != session.map_id:
                    raise ValidationError(
                        f"Chest location {chest['id']} is on map '{chest['map_id']}', but "
                        f"session {session.id} is on map '{session.map_id}'."
                    )
                if not chest["enabled"]:
                    raise ValidationError(
                        f"Chest location {chest['id']} is disabled; enable it before "
                        "recording observations."
                    )
                if observation.chest_opened:
                    already = conn.execute(
                        "SELECT id FROM chest_observations WHERE session_id = ? "
                        "AND chest_location_id = ? AND chest_opened = 1",
                        (observation.session_id, observation.chest_location_id),
                    ).fetchone()
                    if already is not None:
                        raise DuplicateOpeningError(
                            f"Chest location {observation.chest_location_id} was already "
                            f"opened in session {observation.session_id} "
                            f"(observation {already['id']})."
                        )
                cursor = conn.execute(
                    """
                    INSERT INTO chest_observations (
                        session_id, chest_location_id, observed_at, chest_present,
                        chest_opened, interaction_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.session_id,
                        observation.chest_location_id,
                        observation.observed_at,
                        int(observation.chest_present),
                        int(observation.chest_opened),
                        observation.interaction_seconds,
                    ),
                )
                observation.id = cursor.lastrowid
        except DatabaseIntegrityError as exc:  # schema-level backstops
            message = str(exc)
            if "UNIQUE" in message.upper():
                raise DuplicateOpeningError(
                    f"Chest location {observation.chest_location_id} was already opened "
                    f"in session {observation.session_id}."
                ) from exc
            if "running session" in message:
                raise SessionStateError(message) from exc
            raise ValidationError(f"The database rejected the observation: {message}") from exc
        logger.debug(
            "Recorded observation %s (session %d, chest %d, present=%s, opened=%s)",
            observation.id,
            observation.session_id,
            observation.chest_location_id,
            observation.chest_present,
            observation.chest_opened,
        )
        return observation

    # -- statistics ---------------------------------------------------------

    def get_session_statistics(self, session_id: int) -> SessionStatistics:
        """Return aggregate statistics for one session.

        See :class:`~chest_hunter.models.SessionStatistics` for definitions.
        Rates are ``None`` (not zero, not an error) when the duration is
        unknown (running session) or zero.
        """
        session = self.require_session(session_id)
        row = self._db.query_one(
            """
            SELECT COUNT(*)                                                       AS observations,
                   COUNT(DISTINCT chest_location_id)                              AS visited,
                   COUNT(DISTINCT CASE WHEN chest_present = 1
                                       THEN chest_location_id END)                AS found,
                   COALESCE(SUM(chest_opened), 0)                                 AS opened,
                   COALESCE(SUM(CASE WHEN chest_present = 1 AND chest_opened = 0
                                     THEN 1 ELSE 0 END), 0)                       AS failed,
                   COALESCE(SUM(interaction_seconds), 0.0)                        AS seconds
              FROM chest_observations
             WHERE session_id = ?
            """,
            (session_id,),
        )
        assert row is not None
        duration = session.duration_seconds
        opened = int(row["opened"])
        per_minute = opened / (duration / 60.0) if duration is not None and duration > 0 else None
        return SessionStatistics(
            session_id=session_id,
            map_id=session.map_id,
            status=session.status,
            locations_visited=int(row["visited"]),
            total_observations=int(row["observations"]),
            chests_found=int(row["found"]),
            chests_opened=opened,
            failed_interactions=int(row["failed"]),
            total_interaction_seconds=float(row["seconds"]),
            duration_seconds=duration,
            chests_opened_per_minute=per_minute,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _fetch_session(conn: sqlite3.Connection, session_id: int) -> SimulationSession:
        row = conn.execute(
            f"SELECT {_SESSION_COLUMNS} FROM simulation_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"Session {session_id} does not exist.")
        return _row_to_session(row)
