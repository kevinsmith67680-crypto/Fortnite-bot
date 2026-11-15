"""SQLite connection management, transactions and schema migrations.

Design notes
------------
* The connection runs in autocommit mode (``isolation_level=None``) and every
  multi-statement write goes through :meth:`DatabaseManager.transaction`, which
  issues ``BEGIN IMMEDIATE``/``COMMIT``/``ROLLBACK`` explicitly. Nested
  ``transaction()`` blocks become SAVEPOINTs, so higher-level code (e.g. a
  future simulator) can group several repository calls atomically.
* Foreign keys are enabled on every connection and verified.
* The schema version lives in ``PRAGMA user_version``. Migrations are applied
  in order inside a transaction; :meth:`initialize` is idempotent and never
  drops or rewrites existing data.
* All SQL uses ``?`` parameters for values. The only formatted statements use
  internal integer constants or internally generated savepoint names.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from chest_hunter.exceptions import DatabaseError, DatabaseIntegrityError

logger = logging.getLogger(__name__)

MEMORY_DATABASE = ":memory:"

# Largest finite IEEE-754 double; used by CHECK constraints to reject +/-inf.
_MAX_REAL = "1.7976931348623157e308"

_SCHEMA_V1: tuple[str, ...] = (
    f"""
    CREATE TABLE chest_locations (
        id                      INTEGER PRIMARY KEY,
        map_id                  TEXT    NOT NULL CHECK (length(trim(map_id)) > 0),
        location_name           TEXT    NOT NULL CHECK (length(trim(location_name)) > 0),
        x                       REAL    NOT NULL
            CHECK (typeof(x) = 'real' AND x BETWEEN -{_MAX_REAL} AND {_MAX_REAL}),
        y                       REAL    NOT NULL
            CHECK (typeof(y) = 'real' AND y BETWEEN -{_MAX_REAL} AND {_MAX_REAL}),
        z                       REAL    NOT NULL DEFAULT 0.0
            CHECK (typeof(z) = 'real' AND z BETWEEN -{_MAX_REAL} AND {_MAX_REAL}),
        spawn_probability       REAL    NOT NULL
            CHECK (typeof(spawn_probability) = 'real'
                   AND spawn_probability BETWEEN 0.0 AND 1.0),
        estimated_open_seconds  REAL    NOT NULL
            CHECK (typeof(estimated_open_seconds) = 'real'
                   AND estimated_open_seconds BETWEEN 0.0 AND {_MAX_REAL}),
        enabled                 INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
        created_at              TEXT    NOT NULL,
        updated_at              TEXT    NOT NULL
    )
    """,
    # One spawn point per exact coordinate on a map; also serves map_id lookups.
    """
    CREATE UNIQUE INDEX uq_chest_locations_map_position
        ON chest_locations (map_id, x, y, z)
    """,
    """
    CREATE INDEX idx_chest_locations_map_enabled
        ON chest_locations (map_id, enabled)
    """,
    """
    CREATE TABLE simulation_sessions (
        id          INTEGER PRIMARY KEY,
        map_id      TEXT NOT NULL CHECK (length(trim(map_id)) > 0),
        started_at  TEXT NOT NULL,
        ended_at    TEXT,
        status      TEXT NOT NULL DEFAULT 'running'
            CHECK (status IN ('running', 'completed', 'aborted')),
        notes       TEXT,
        CHECK ((status = 'running') = (ended_at IS NULL)),
        CHECK (ended_at IS NULL OR ended_at >= started_at)
    )
    """,
    """
    CREATE INDEX idx_simulation_sessions_map_started
        ON simulation_sessions (map_id, started_at)
    """,
    """
    CREATE INDEX idx_simulation_sessions_status
        ON simulation_sessions (status)
    """,
    f"""
    CREATE TABLE chest_observations (
        id                   INTEGER PRIMARY KEY,
        session_id           INTEGER NOT NULL
            REFERENCES simulation_sessions (id) ON DELETE RESTRICT,
        chest_location_id    INTEGER NOT NULL
            REFERENCES chest_locations (id) ON DELETE RESTRICT,
        observed_at          TEXT    NOT NULL,
        chest_present        INTEGER NOT NULL CHECK (chest_present IN (0, 1)),
        chest_opened         INTEGER NOT NULL CHECK (chest_opened IN (0, 1)),
        interaction_seconds  REAL    NOT NULL DEFAULT 0.0
            CHECK (typeof(interaction_seconds) = 'real'
                   AND interaction_seconds BETWEEN 0.0 AND {_MAX_REAL}),
        CHECK (chest_opened = 0 OR chest_present = 1)
    )
    """,
    """
    CREATE INDEX idx_chest_observations_session_chest
        ON chest_observations (session_id, chest_location_id)
    """,
    """
    CREATE INDEX idx_chest_observations_chest
        ON chest_observations (chest_location_id)
    """,
    # At most one successful opening per chest location per session.
    """
    CREATE UNIQUE INDEX uq_chest_observations_one_opening
        ON chest_observations (session_id, chest_location_id)
        WHERE chest_opened = 1
    """,
    # Observations may only be added to running sessions on the same map.
    """
    CREATE TRIGGER trg_chest_observations_validate_insert
    BEFORE INSERT ON chest_observations
    BEGIN
        SELECT RAISE(ABORT, 'observations can only be recorded for a running session')
        WHERE (SELECT status FROM simulation_sessions WHERE id = NEW.session_id)
              IS NOT 'running';
        SELECT RAISE(ABORT, 'chest location belongs to a different map than the session')
        WHERE (SELECT map_id FROM chest_locations WHERE id = NEW.chest_location_id)
              IS NOT (SELECT map_id FROM simulation_sessions WHERE id = NEW.session_id);
    END
    """,
    # Observations must stay attached to the session/location they were made in.
    """
    CREATE TRIGGER trg_chest_observations_lock_links
    BEFORE UPDATE OF session_id, chest_location_id ON chest_observations
    WHEN NEW.session_id IS NOT OLD.session_id
      OR NEW.chest_location_id IS NOT OLD.chest_location_id
    BEGIN
        SELECT RAISE(ABORT, 'observations cannot be moved to another session or location');
    END
    """,
    # Finished sessions are immutable apart from their notes.
    """
    CREATE TRIGGER trg_simulation_sessions_final_state
    BEFORE UPDATE OF status, ended_at, started_at, map_id ON simulation_sessions
    WHEN OLD.status <> 'running'
    BEGIN
        SELECT RAISE(ABORT, 'session has already finished');
    END
    """,
    """
    CREATE TRIGGER trg_simulation_sessions_lock_identity
    BEFORE UPDATE OF started_at, map_id ON simulation_sessions
    WHEN NEW.started_at IS NOT OLD.started_at OR NEW.map_id IS NOT OLD.map_id
    BEGIN
        SELECT RAISE(ABORT, 'session map_id and started_at cannot be changed');
    END
    """,
    # A referenced location may not silently move to another map.
    """
    CREATE TRIGGER trg_chest_locations_lock_map
    BEFORE UPDATE OF map_id ON chest_locations
    WHEN NEW.map_id IS NOT OLD.map_id
     AND EXISTS (SELECT 1 FROM chest_observations WHERE chest_location_id = OLD.id)
    BEGIN
        SELECT RAISE(ABORT, 'cannot change map_id of a chest location that has observations');
    END
    """,
)

# Schema version 2 (Phase 2): one row per autonomous agent run, keyed by the
# simulation session it drove. Phase 1 tables are not modified.
_SCHEMA_V2: tuple[str, ...] = (
    f"""
    CREATE TABLE simulation_runs (
        session_id           INTEGER PRIMARY KEY
            REFERENCES simulation_sessions (id) ON DELETE RESTRICT,
        strategy             TEXT    NOT NULL CHECK (length(trim(strategy)) > 0),
        seed                 INTEGER NOT NULL CHECK (seed >= 0),
        episode              INTEGER NOT NULL DEFAULT 0 CHECK (episode >= 0),
        parameters_json      TEXT    NOT NULL,
        outcome              TEXT
            CHECK (outcome IS NULL
                   OR outcome IN ('completed', 'tick_limit', 'failed', 'cancelled')),
        failure_reason       TEXT,
        ticks                INTEGER CHECK (ticks IS NULL OR ticks >= 0),
        simulated_seconds    REAL
            CHECK (simulated_seconds IS NULL
                   OR simulated_seconds BETWEEN 0.0 AND {_MAX_REAL}),
        travel_distance      REAL
            CHECK (travel_distance IS NULL OR travel_distance BETWEEN 0.0 AND {_MAX_REAL}),
        recoveries           INTEGER CHECK (recoveries IS NULL OR recoveries >= 0),
        unreachable_targets  INTEGER
            CHECK (unreachable_targets IS NULL OR unreachable_targets >= 0),
        abandoned_targets    INTEGER CHECK (abandoned_targets IS NULL OR abandoned_targets >= 0),
        rewards              INTEGER CHECK (rewards IS NULL OR rewards >= 0),
        created_at           TEXT    NOT NULL,
        finished_at          TEXT,
        CHECK ((outcome IS NULL) = (finished_at IS NULL))
    )
    """,
    """
    CREATE INDEX idx_simulation_runs_strategy
        ON simulation_runs (strategy, seed, episode)
    """,
    """
    CREATE TRIGGER trg_simulation_runs_final
    BEFORE UPDATE ON simulation_runs
    WHEN OLD.outcome IS NOT NULL
    BEGIN
        SELECT RAISE(ABORT, 'simulation run has already finished');
    END
    """,
)

# Schema version 3 (Phase 4): the 3D agent records whether an opening produced a
# (simulated) reward, and every run records its navigation failures. Both columns
# are nullable so existing rows stay valid; nothing else changes.
_SCHEMA_V3: tuple[str, ...] = (
    """
    ALTER TABLE chest_observations ADD COLUMN reward_received INTEGER
        CHECK (reward_received IS NULL OR reward_received IN (0, 1))
    """,
    """
    CREATE TRIGGER trg_chest_observations_reward_requires_opening
    BEFORE INSERT ON chest_observations
    WHEN NEW.reward_received = 1 AND NEW.chest_opened = 0
    BEGIN
        SELECT RAISE(ABORT, 'a reward requires a successful opening');
    END
    """,
    """
    ALTER TABLE simulation_runs ADD COLUMN navigation_failures INTEGER
        CHECK (navigation_failures IS NULL OR navigation_failures >= 0)
    """,
)

# Phase 5: learned statistics are stored *next to* the configured data, never over it:
# chest_locations.spawn_probability stays the configured prior.
_SCHEMA_V4: tuple[str, ...] = (
    """
    CREATE TABLE spawn_estimates (
        id                 INTEGER PRIMARY KEY,
        chest_location_id  INTEGER NOT NULL
                           REFERENCES chest_locations(id) ON DELETE CASCADE,
        layout_version     TEXT    NOT NULL DEFAULT '',
        prior_alpha        REAL    NOT NULL CHECK (prior_alpha > 0),
        prior_beta         REAL    NOT NULL CHECK (prior_beta > 0),
        present_count      INTEGER NOT NULL CHECK (present_count >= 0),
        absent_count       INTEGER NOT NULL CHECK (absent_count >= 0),
        posterior_mean     REAL    NOT NULL CHECK (posterior_mean >= 0 AND posterior_mean <= 1),
        credible_low       REAL    NOT NULL CHECK (credible_low >= 0 AND credible_low <= 1),
        credible_high      REAL    NOT NULL CHECK (credible_high >= 0 AND credible_high <= 1),
        sessions_used      INTEGER NOT NULL CHECK (sessions_used >= 0),
        updated_at         TEXT    NOT NULL,
        UNIQUE (chest_location_id, layout_version)
    )
    """,
    """
    CREATE TABLE route_memory (
        id                 INTEGER PRIMARY KEY,
        map_id             TEXT    NOT NULL CHECK (length(trim(map_id)) > 0),
        layout_version     TEXT    NOT NULL DEFAULT '',
        session_id         INTEGER REFERENCES simulation_sessions(id) ON DELETE SET NULL,
        start_region       TEXT    NOT NULL,
        dest_region        TEXT    NOT NULL,
        chest_location_id  INTEGER REFERENCES chest_locations(id) ON DELETE SET NULL,
        estimated_length   REAL    NOT NULL CHECK (estimated_length >= 0),
        estimated_seconds  REAL    NOT NULL CHECK (estimated_seconds >= 0),
        actual_seconds     REAL    NOT NULL CHECK (actual_seconds >= 0),
        success            INTEGER NOT NULL CHECK (success IN (0, 1)),
        failure_kind       TEXT,
        recoveries         INTEGER NOT NULL DEFAULT 0 CHECK (recoveries >= 0),
        chest_opened       INTEGER CHECK (chest_opened IS NULL OR chest_opened IN (0, 1)),
        invalidated        INTEGER NOT NULL DEFAULT 0 CHECK (invalidated IN (0, 1)),
        created_at         TEXT    NOT NULL
    )
    """,
    """
    CREATE INDEX idx_route_memory_lookup
        ON route_memory (map_id, layout_version, start_region, dest_region)
    """,
)

# Version 5 (Phase 6): experience partitions. Route statistics depend on the environment
# version, the agent version and the simulation configuration as well as the layout, so each
# record carries a partition key ("env=...;agent=...;config=..."; '' for older records,
# which are used only on request). Additive: existing rows and queries keep working.
_SCHEMA_V5: tuple[str, ...] = (
    "ALTER TABLE route_memory ADD COLUMN partition_key TEXT NOT NULL DEFAULT ''",
    """
    CREATE INDEX idx_route_memory_partition
        ON route_memory (map_id, layout_version, partition_key)
    """,
)

MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1, _SCHEMA_V1), (2, _SCHEMA_V2), (3, _SCHEMA_V3), (4, _SCHEMA_V4), (5, _SCHEMA_V5))
"""Ordered ``(version, statements)`` pairs. Append new versions; never edit old ones."""

SCHEMA_VERSION: int = MIGRATIONS[-1][0]

EXPECTED_TABLES: tuple[str, ...] = (
    "chest_locations",
    "simulation_sessions",
    "chest_observations",
    "simulation_runs",
    "spawn_estimates",
    "route_memory",
)


class DatabaseManager:
    """Owns one SQLite connection and the schema of the chest database.

    Usage::

        with DatabaseManager("data/chest_locations.db") as db:
            db.initialize()
            with db.transaction() as conn:
                conn.execute("...", (...))
    """

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 5.0) -> None:
        raw = os.fspath(path)
        if not raw:
            raise DatabaseError("A database path is required.")
        self._is_memory = raw == MEMORY_DATABASE
        self._path: Path | None = None if self._is_memory else Path(raw).expanduser().resolve()
        self._timeout = timeout
        self._connection: sqlite3.Connection | None = None
        self._savepoint_counter = 0

    # -- properties -------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """Absolute database path, or ``None`` for an in-memory database."""
        return self._path

    @property
    def location(self) -> str:
        """Human-readable database location for messages."""
        return MEMORY_DATABASE if self._path is None else str(self._path)

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    def exists(self) -> bool:
        """Whether the database file already exists (always False for memory)."""
        return self._path is not None and self._path.is_file()

    # -- connection lifecycle ---------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Open (or return the already-open) connection with foreign keys enabled.

        Like ``sqlite3.connect`` this creates the database file (and its parent
        directory) if needed; callers that must not create a database should
        check :meth:`exists` first, as the CLI does.
        """
        if self._connection is not None:
            return self._connection
        if self._path is not None:
            if self._path.is_dir():
                raise DatabaseError(f"Database path is a directory: {self._path}")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise DatabaseError(
                    f"Could not create database directory {self._path.parent}: {exc}"
                ) from exc
        try:
            connection = sqlite3.connect(
                MEMORY_DATABASE if self._path is None else str(self._path),
                timeout=self._timeout,
                isolation_level=None,  # explicit transaction control
            )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open database at {self.location}: {exc}") from exc
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]
            if enabled != 1:
                raise DatabaseError("This SQLite build does not support foreign-key enforcement.")
            # Touch the schema so non-database files are rejected immediately.
            connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.Error as exc:
            connection.close()
            raise DatabaseError(
                f"Could not use database at {self.location}: {exc}"
            ) from exc
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        logger.debug("Opened database connection to %s", self.location)
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection, opening it on first use."""
        return self.connect()

    def close(self) -> None:
        """Close the connection. Uncommitted work is rolled back. Idempotent."""
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            if connection.in_transaction:
                connection.rollback()
        finally:
            connection.close()
            logger.debug("Closed database connection to %s", self.location)

    def __enter__(self) -> DatabaseManager:
        # The connection is opened lazily, on first use.
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- schema -----------------------------------------------------------

    def schema_version(self) -> int:
        """Return ``PRAGMA user_version`` (0 means not initialised)."""
        row = self.query_one("PRAGMA user_version")
        return int(row[0]) if row is not None else 0

    def is_initialized(self) -> bool:
        return self.schema_version() >= 1

    def initialize(self) -> int:
        """Create or migrate the schema to :data:`SCHEMA_VERSION`. Idempotent.

        Existing data is never dropped. A database written by a *newer*
        version of this software is refused rather than modified.
        """
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise DatabaseError(
                f"Database {self.location} has schema version {current}, but this "
                f"software only supports up to version {SCHEMA_VERSION}. "
                "Upgrade chest-hunter-ai instead of downgrading the database."
            )

        for version, statements in MIGRATIONS:
            if version <= current:
                continue
            try:
                with self.transaction() as conn:
                    for statement in statements:
                        conn.execute(statement)
                    # PRAGMA values cannot be bound; ``version`` is an internal int.
                    conn.execute(f"PRAGMA user_version = {int(version)}")
            except DatabaseError as exc:
                raise DatabaseError(
                    f"Failed to apply schema migration {version} to {self.location}: {exc}"
                ) from exc
            logger.info("Applied schema migration %d to %s", version, self.location)

        self._verify_schema()
        return SCHEMA_VERSION

    def _verify_schema(self) -> None:
        placeholders = ", ".join("?" for _ in EXPECTED_TABLES)
        rows = self.query_all(
            f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
            EXPECTED_TABLES,
        )
        missing = set(EXPECTED_TABLES) - {row["name"] for row in rows}
        if missing:
            raise DatabaseError(
                f"Database {self.location} is missing required tables: "
                f"{', '.join(sorted(missing))}. It may not be a chest-hunter database."
            )

    # -- transactions -----------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block atomically.

        The outermost block uses ``BEGIN IMMEDIATE`` (taking the write lock
        up front so check-then-write sequences are race-free). Nested blocks
        use SAVEPOINTs. On any exception the block is rolled back; SQLite
        errors are re-raised as :class:`DatabaseIntegrityError` or
        :class:`DatabaseError`.
        """
        conn = self.connect()
        savepoint: str | None = None
        try:
            if conn.in_transaction:
                self._savepoint_counter += 1
                savepoint = f"chest_hunter_sp_{self._savepoint_counter}"
                conn.execute(f"SAVEPOINT {savepoint}")
            else:
                conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not start a transaction on {self.location}: {exc}") from exc

        try:
            yield conn
        except BaseException as exc:
            self._rollback(conn, savepoint)
            if isinstance(exc, sqlite3.IntegrityError):
                raise DatabaseIntegrityError(str(exc)) from exc
            if isinstance(exc, sqlite3.Error):
                raise DatabaseError(f"Database operation failed: {exc}") from exc
            raise

        try:
            if savepoint is not None:
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            else:
                conn.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback(conn, savepoint)
            raise DatabaseError(f"Could not commit transaction: {exc}") from exc

    @staticmethod
    def _rollback(conn: sqlite3.Connection, savepoint: str | None) -> None:
        try:
            if savepoint is not None:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            elif conn.in_transaction:
                conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - best effort during error handling
            logger.exception("Rollback failed")

    # -- convenience query helpers ----------------------------------------

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute one parameterised statement, translating SQLite errors."""
        conn = self.connect()
        try:
            return conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            raise DatabaseIntegrityError(str(exc)) from exc
        except sqlite3.Error as exc:
            raise DatabaseError(f"Database operation failed: {exc}") from exc

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.execute(sql, params).fetchone()
        return row

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.execute(sql, params).fetchall())
