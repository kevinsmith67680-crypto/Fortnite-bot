"""CRUD and statistics for chest spawn locations.

Visit/opening status is never stored on ``chest_locations``. It is derived
from ``chest_observations`` for a specific session, so a spawn point can be
visited again in any later session.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace
from typing import Any, Final

from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import (
    ChestInUseError,
    DatabaseIntegrityError,
    DuplicateChestError,
    RecordNotFoundError,
    ValidationError,
)
from chest_hunter.models import (
    DEFAULT_OPEN_SECONDS,
    DEFAULT_SPAWN_PROBABILITY,
    ChestLocation,
    ChestStatistics,
    Clock,
    require_bool,
    require_finite_number,
    require_map_id,
    require_positive_int,
    timestamp_now,
    utc_now,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_SPAWN_SAMPLES: Final = 5
"""Minimum (session, location) samples before an observed spawn frequency is reported."""

# Whitelisted ORDER BY clauses; every one ends with ``id`` so ordering is total.
SORT_ORDERS: Final[dict[str, str]] = {
    "id": "id",
    "map": "map_id, id",
    "name": "location_name COLLATE NOCASE, id",
    "probability": "spawn_probability DESC, id",
    "position": "map_id, x, y, z, id",
}

_COLUMNS: Final = (
    "id, map_id, location_name, x, y, z, spawn_probability, "
    "estimated_open_seconds, enabled, created_at, updated_at"
)


def _row_to_chest(row: sqlite3.Row) -> ChestLocation:
    return ChestLocation(
        id=row["id"],
        map_id=row["map_id"],
        location_name=row["location_name"],
        x=row["x"],
        y=row["y"],
        z=row["z"],
        spawn_probability=row["spawn_probability"],
        estimated_open_seconds=row["estimated_open_seconds"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class ChestRepository:
    """Data access for :class:`~chest_hunter.models.ChestLocation` records.

    The database must already be initialised (``DatabaseManager.initialize``).
    ``clock`` is injectable so simulators and tests can control timestamps.
    """

    def __init__(self, db: DatabaseManager, *, clock: Clock = utc_now) -> None:
        self._db = db
        self._clock = clock

    # -- create -------------------------------------------------------------

    def add_chest(
        self,
        map_id: str,
        location_name: str,
        x: float,
        y: float,
        z: float = 0.0,
        spawn_probability: float = DEFAULT_SPAWN_PROBABILITY,
        estimated_open_seconds: float = DEFAULT_OPEN_SECONDS,
        enabled: bool = True,
    ) -> ChestLocation:
        """Insert a new chest location and return it with its ID and timestamps.

        :raises ValidationError: if any value is invalid.
        :raises DuplicateChestError: if a location already exists at the same
            ``(map_id, x, y, z)``.
        """
        now = timestamp_now(self._clock)
        chest = ChestLocation(
            id=None,
            map_id=map_id,
            location_name=location_name,
            x=x,
            y=y,
            z=z,
            spawn_probability=spawn_probability,
            estimated_open_seconds=estimated_open_seconds,
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )
        return self.insert_chest(chest)

    def insert_chest(self, chest: ChestLocation) -> ChestLocation:
        """Insert a prepared :class:`ChestLocation` (its ``id`` must be ``None``)."""
        chest.validate()
        if chest.id is not None:
            raise ValidationError("A new chest location must not already have an id.")
        now = timestamp_now(self._clock)
        created_at = chest.created_at or now
        updated_at = chest.updated_at or created_at
        try:
            with self._db.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO chest_locations (
                        map_id, location_name, x, y, z, spawn_probability,
                        estimated_open_seconds, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chest.map_id,
                        chest.location_name,
                        chest.x,
                        chest.y,
                        chest.z,
                        chest.spawn_probability,
                        chest.estimated_open_seconds,
                        int(chest.enabled),
                        created_at,
                        updated_at,
                    ),
                )
                new_id = cursor.lastrowid
        except DatabaseIntegrityError as exc:
            raise self._translate_integrity_error(exc, chest) from exc
        assert new_id is not None
        logger.debug("Added chest location %d (%s)", new_id, chest.location_name)
        return replace(chest, id=new_id, created_at=created_at, updated_at=updated_at)

    # -- read ---------------------------------------------------------------

    def get_chest(self, chest_id: int) -> ChestLocation | None:
        """Return the chest location with ``chest_id``, or ``None`` if it does not exist."""
        require_positive_int("chest_id", chest_id)
        row = self._db.query_one(
            f"SELECT {_COLUMNS} FROM chest_locations WHERE id = ?", (chest_id,)
        )
        return None if row is None else _row_to_chest(row)

    def require_chest(self, chest_id: int) -> ChestLocation:
        """Like :meth:`get_chest` but raise :class:`RecordNotFoundError` if missing."""
        chest = self.get_chest(chest_id)
        if chest is None:
            raise RecordNotFoundError(f"Chest location {chest_id} does not exist.")
        return chest

    def get_all_chests(
        self,
        *,
        map_id: str | None = None,
        enabled: bool | None = None,
        order_by: str = "id",
    ) -> list[ChestLocation]:
        """Return chest locations, optionally filtered, in deterministic order.

        :param map_id: only locations on this map.
        :param enabled: ``True`` for enabled only, ``False`` for disabled only,
            ``None`` for both.
        :param order_by: one of :data:`SORT_ORDERS` (``id`` by default).
        """
        if order_by not in SORT_ORDERS:
            raise ValidationError(
                f"order_by must be one of {', '.join(SORT_ORDERS)}; got {order_by!r}."
            )
        clauses: list[str] = []
        params: list[Any] = []
        if map_id is not None:
            clauses.append("map_id = ?")
            params.append(require_map_id(map_id))
        if enabled is not None:
            clauses.append("enabled = ?")
            params.append(int(require_bool("enabled", enabled)))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.query_all(
            f"SELECT {_COLUMNS} FROM chest_locations {where} ORDER BY {SORT_ORDERS[order_by]}",
            params,
        )
        return [_row_to_chest(row) for row in rows]

    def get_chests_by_map(
        self, map_id: str, *, enabled: bool | None = None
    ) -> list[ChestLocation]:
        """Return the locations on ``map_id`` ordered by ID."""
        return self.get_all_chests(map_id=map_id, enabled=enabled)

    def find_chest_at(
        self, map_id: str, x: float, y: float, z: float = 0.0
    ) -> ChestLocation | None:
        """Return the location at exactly ``(x, y, z)`` on ``map_id``, if any."""
        row = self._db.query_one(
            f"SELECT {_COLUMNS} FROM chest_locations "
            "WHERE map_id = ? AND x = ? AND y = ? AND z = ?",
            (
                require_map_id(map_id),
                require_finite_number("x", x),
                require_finite_number("y", y),
                require_finite_number("z", z),
            ),
        )
        return None if row is None else _row_to_chest(row)

    def list_map_ids(self) -> list[str]:
        """Return every map ID that has at least one chest location, sorted."""
        rows = self._db.query_all(
            "SELECT DISTINCT map_id FROM chest_locations ORDER BY map_id"
        )
        return [row["map_id"] for row in rows]

    def count_observations(self, chest_id: int) -> int:
        """Return how many observations reference ``chest_id`` across all sessions."""
        require_positive_int("chest_id", chest_id)
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM chest_observations WHERE chest_location_id = ?",
            (chest_id,),
        )
        return int(row["n"]) if row is not None else 0

    # -- update -------------------------------------------------------------

    def update_chest(
        self,
        chest_id: int,
        *,
        map_id: str | None = None,
        location_name: str | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        spawn_probability: float | None = None,
        estimated_open_seconds: float | None = None,
        enabled: bool | None = None,
    ) -> ChestLocation:
        """Update the given fields (``None`` = leave unchanged) and return the new record.

        :raises RecordNotFoundError: if the location does not exist.
        :raises ValidationError: if no field is given or a value is invalid.
        :raises ChestInUseError: when moving a location that has observations
            to a different map.
        :raises DuplicateChestError: if the new coordinates are already taken.
        """
        require_positive_int("chest_id", chest_id)
        changes: dict[str, Any] = {
            key: value
            for key, value in {
                "map_id": map_id,
                "location_name": location_name,
                "x": x,
                "y": y,
                "z": z,
                "spawn_probability": spawn_probability,
                "estimated_open_seconds": estimated_open_seconds,
                "enabled": enabled,
            }.items()
            if value is not None
        }
        if not changes:
            raise ValidationError("No fields to update were provided.")

        updated: ChestLocation | None = None
        try:
            with self._db.transaction() as conn:
                row = conn.execute(
                    f"SELECT {_COLUMNS} FROM chest_locations WHERE id = ?", (chest_id,)
                ).fetchone()
                if row is None:
                    raise RecordNotFoundError(f"Chest location {chest_id} does not exist.")
                current = _row_to_chest(row)
                updated = replace(
                    current, **changes, updated_at=timestamp_now(self._clock)
                )  # replace() re-runs validation via __post_init__

                if updated.map_id != current.map_id:
                    in_use = conn.execute(
                        "SELECT COUNT(*) FROM chest_observations WHERE chest_location_id = ?",
                        (chest_id,),
                    ).fetchone()[0]
                    if in_use:
                        raise ChestInUseError(
                            f"Chest location {chest_id} has {in_use} observation(s) on map "
                            f"'{current.map_id}' and cannot be moved to another map. "
                            "Create a new location instead."
                        )

                conn.execute(
                    """
                    UPDATE chest_locations
                       SET map_id = ?, location_name = ?, x = ?, y = ?, z = ?,
                           spawn_probability = ?, estimated_open_seconds = ?,
                           enabled = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (
                        updated.map_id,
                        updated.location_name,
                        updated.x,
                        updated.y,
                        updated.z,
                        updated.spawn_probability,
                        updated.estimated_open_seconds,
                        int(updated.enabled),
                        updated.updated_at,
                        chest_id,
                    ),
                )
        except DatabaseIntegrityError as exc:
            if updated is None:  # pragma: no cover - integrity errors only come from UPDATE
                raise
            raise self._translate_integrity_error(exc, updated) from exc
        assert updated is not None
        logger.debug("Updated chest location %d: %s", chest_id, sorted(changes))
        return updated

    # -- delete -------------------------------------------------------------

    def delete_chest(self, chest_id: int) -> bool:
        """Delete a chest location that has never been observed.

        Returns ``True`` if a record was deleted and ``False`` if it did not
        exist. Locations referenced by observations are protected (also by an
        ``ON DELETE RESTRICT`` foreign key): disable them with
        ``update_chest(chest_id, enabled=False)`` instead.

        :raises ChestInUseError: if observations reference the location.
        """
        require_positive_int("chest_id", chest_id)
        try:
            with self._db.transaction() as conn:
                in_use = conn.execute(
                    "SELECT COUNT(*) FROM chest_observations WHERE chest_location_id = ?",
                    (chest_id,),
                ).fetchone()[0]
                if in_use:
                    raise ChestInUseError(
                        f"Chest location {chest_id} is referenced by {in_use} observation(s) "
                        "and cannot be deleted. Disable it instead "
                        "(update-chest --chest ID --disable)."
                    )
                deleted = conn.execute(
                    "DELETE FROM chest_locations WHERE id = ?", (chest_id,)
                ).rowcount
        except DatabaseIntegrityError as exc:  # FK backstop
            raise ChestInUseError(
                f"Chest location {chest_id} is referenced by observations and cannot be deleted."
            ) from exc
        if deleted:
            logger.debug("Deleted chest location %d", chest_id)
        return bool(deleted)

    # -- session-relative queries -------------------------------------------

    def get_unvisited_chests(self, session_id: int) -> list[ChestLocation]:
        """Return enabled locations on the session's map with no observation in that session.

        :raises RecordNotFoundError: if the session does not exist.
        """
        require_positive_int("session_id", session_id)
        session = self._db.query_one(
            "SELECT map_id FROM simulation_sessions WHERE id = ?", (session_id,)
        )
        if session is None:
            raise RecordNotFoundError(f"Session {session_id} does not exist.")
        rows = self._db.query_all(
            f"""
            SELECT {_COLUMNS} FROM chest_locations AS c
             WHERE c.map_id = ?
               AND c.enabled = 1
               AND NOT EXISTS (
                     SELECT 1 FROM chest_observations AS o
                      WHERE o.session_id = ? AND o.chest_location_id = c.id)
             ORDER BY c.id
            """,
            (session["map_id"], session_id),
        )
        return [_row_to_chest(row) for row in rows]

    # -- statistics ---------------------------------------------------------

    def get_chest_statistics(
        self, map_id: str, *, min_spawn_samples: int = DEFAULT_MIN_SPAWN_SAMPLES
    ) -> ChestStatistics:
        """Return aggregate statistics for ``map_id``.

        * ``average_spawn_probability`` - mean configured probability of
          *enabled* locations (``None`` if there are none).
        * ``total_visits`` - all observations of locations on the map.
        * ``chests_found`` / ``successful_openings`` - observations with a chest
          present / opened.
        * ``observed_spawn_frequency`` - share of distinct (session, location)
          samples in which a chest was ever present. Repeat visits within a
          session count once, so revisiting an already-opened chest does not
          bias the estimate. ``None`` if fewer than ``min_spawn_samples``.
        """
        map_id = require_map_id(map_id)
        min_spawn_samples = require_positive_int("min_spawn_samples", min_spawn_samples)

        locations = self._db.query_one(
            """
            SELECT COUNT(*)                                          AS total,
                   COALESCE(SUM(enabled), 0)                         AS enabled,
                   AVG(CASE WHEN enabled = 1 THEN spawn_probability END) AS avg_probability
              FROM chest_locations
             WHERE map_id = ?
            """,
            (map_id,),
        )
        visits = self._db.query_one(
            """
            SELECT COUNT(*)                          AS visits,
                   COALESCE(SUM(o.chest_present), 0) AS found,
                   COALESCE(SUM(o.chest_opened), 0)  AS opened
              FROM chest_observations AS o
              JOIN chest_locations   AS c ON c.id = o.chest_location_id
             WHERE c.map_id = ?
            """,
            (map_id,),
        )
        samples = self._db.query_one(
            """
            SELECT COUNT(*) AS samples, COALESCE(SUM(found), 0) AS present
              FROM (SELECT MAX(o.chest_present) AS found
                      FROM chest_observations AS o
                      JOIN chest_locations   AS c ON c.id = o.chest_location_id
                     WHERE c.map_id = ?
                     GROUP BY o.session_id, o.chest_location_id)
            """,
            (map_id,),
        )
        assert locations is not None and visits is not None and samples is not None

        sample_count = int(samples["samples"])
        frequency = (
            int(samples["present"]) / sample_count
            if sample_count >= min_spawn_samples
            else None
        )
        avg = locations["avg_probability"]
        return ChestStatistics(
            map_id=map_id,
            total_locations=int(locations["total"]),
            enabled_locations=int(locations["enabled"]),
            average_spawn_probability=None if avg is None else float(avg),
            total_visits=int(visits["visits"]),
            chests_found=int(visits["found"]),
            successful_openings=int(visits["opened"]),
            spawn_samples=sample_count,
            min_spawn_samples=min_spawn_samples,
            observed_spawn_frequency=frequency,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _translate_integrity_error(
        exc: DatabaseIntegrityError, chest: ChestLocation
    ) -> Exception:
        message = str(exc)
        if "UNIQUE" in message.upper():
            return DuplicateChestError(
                f"A chest location already exists on map '{chest.map_id}' at "
                f"({chest.x:g}, {chest.y:g}, {chest.z:g})."
            )
        if "observations" in message:
            return ChestInUseError(message)
        return ValidationError(f"The database rejected the chest location: {message}")
