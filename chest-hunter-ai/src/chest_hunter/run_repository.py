"""Persistence for autonomous agent runs (``simulation_runs``, schema v2).

Each run is attached 1:1 to a :class:`~chest_hunter.models.SimulationSession`
and records which route-planning strategy, seed and parameters drove it,
plus its final metrics. Chest-level outcomes stay in ``chest_observations``.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Final

from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import (
    ConflictError,
    DatabaseIntegrityError,
    RecordNotFoundError,
    ValidationError,
)
from chest_hunter.models import (
    RUN_OUTCOMES,
    Clock,
    SimulationRun,
    require_positive_int,
    timestamp_now,
    utc_now,
)

_COLUMNS: Final = (
    "session_id, strategy, seed, episode, parameters_json, outcome, failure_reason, "
    "ticks, simulated_seconds, travel_distance, recoveries, unreachable_targets, "
    "abandoned_targets, rewards, created_at, finished_at, navigation_failures"
)


def _row_to_run(row: sqlite3.Row) -> SimulationRun:
    return SimulationRun(
        session_id=row["session_id"],
        strategy=row["strategy"],
        seed=row["seed"],
        episode=row["episode"],
        created_at=row["created_at"],
        parameters=json.loads(row["parameters_json"]),
        outcome=row["outcome"],
        failure_reason=row["failure_reason"],
        ticks=row["ticks"],
        simulated_seconds=row["simulated_seconds"],
        travel_distance=row["travel_distance"],
        recoveries=row["recoveries"],
        unreachable_targets=row["unreachable_targets"],
        abandoned_targets=row["abandoned_targets"],
        rewards=row["rewards"],
        finished_at=row["finished_at"],
        navigation_failures=row["navigation_failures"],
    )


class RunRepository:
    """Create, finish and query :class:`SimulationRun` records."""

    def __init__(self, db: DatabaseManager, *, clock: Clock = utc_now) -> None:
        self._db = db
        self._clock = clock

    def create_run(
        self,
        session_id: int,
        *,
        strategy: str,
        seed: int,
        episode: int = 0,
        parameters: dict[str, Any] | None = None,
    ) -> SimulationRun:
        """Attach run metadata to an existing session.

        :raises RecordNotFoundError: if the session does not exist.
        :raises ConflictError: if the session already has a run.
        """
        run = SimulationRun(
            session_id=session_id,
            strategy=strategy,
            seed=seed,
            episode=episode,
            created_at=timestamp_now(self._clock),
            parameters=dict(parameters or {}),
        )
        try:
            parameters_json = json.dumps(run.parameters, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Run parameters must be JSON-serialisable: {exc}") from exc
        try:
            with self._db.transaction() as conn:
                if conn.execute(
                    "SELECT 1 FROM simulation_sessions WHERE id = ?", (session_id,)
                ).fetchone() is None:
                    raise RecordNotFoundError(f"Session {session_id} does not exist.")
                conn.execute(
                    "INSERT INTO simulation_runs (session_id, strategy, seed, episode, "
                    "parameters_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (run.session_id, run.strategy, run.seed, run.episode, parameters_json,
                     run.created_at),
                )
        except DatabaseIntegrityError as exc:
            raise ConflictError(f"Session {session_id} already has a simulation run.") from exc
        return run

    def finish_run(
        self,
        session_id: int,
        *,
        outcome: str,
        failure_reason: str | None = None,
        ticks: int = 0,
        simulated_seconds: float = 0.0,
        travel_distance: float = 0.0,
        recoveries: int = 0,
        unreachable_targets: int = 0,
        abandoned_targets: int = 0,
        rewards: int = 0,
        navigation_failures: int | None = None,
    ) -> SimulationRun:
        """Store the final outcome and metrics of a run (only once)."""
        require_positive_int("session_id", session_id)
        if outcome not in RUN_OUTCOMES:
            raise ValidationError(
                f"outcome must be one of {', '.join(RUN_OUTCOMES)}; got {outcome!r}."
            )
        finished_at = timestamp_now(self._clock)
        try:
            with self._db.transaction() as conn:
                updated = conn.execute(
                    """
                    UPDATE simulation_runs
                       SET outcome = ?, failure_reason = ?, ticks = ?, simulated_seconds = ?,
                           travel_distance = ?, recoveries = ?, unreachable_targets = ?,
                           abandoned_targets = ?, rewards = ?, finished_at = ?,
                           navigation_failures = ?
                     WHERE session_id = ?
                    """,
                    (outcome, failure_reason, ticks, simulated_seconds, travel_distance,
                     recoveries, unreachable_targets, abandoned_targets, rewards,
                     finished_at, navigation_failures, session_id),
                ).rowcount
        except DatabaseIntegrityError as exc:
            raise ConflictError(
                f"Simulation run for session {session_id} could not be finished: {exc}"
            ) from exc
        if not updated:
            raise RecordNotFoundError(f"Session {session_id} has no simulation run.")
        run = self.get_run(session_id)
        assert run is not None
        return run

    def get_run(self, session_id: int) -> SimulationRun | None:
        require_positive_int("session_id", session_id)
        row = self._db.query_one(
            f"SELECT {_COLUMNS} FROM simulation_runs WHERE session_id = ?", (session_id,)
        )
        return None if row is None else _row_to_run(row)

    def list_runs(self, *, strategy: str | None = None) -> list[SimulationRun]:
        """Return runs ordered by session ID, optionally for one strategy."""
        if strategy is None:
            rows = self._db.query_all(f"SELECT {_COLUMNS} FROM simulation_runs ORDER BY session_id")
        else:
            rows = self._db.query_all(
                f"SELECT {_COLUMNS} FROM simulation_runs WHERE strategy = ? ORDER BY session_id",
                (strategy,),
            )
        return [_row_to_run(row) for row in rows]
