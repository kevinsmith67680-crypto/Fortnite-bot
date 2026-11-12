"""Typed data models, validation helpers and timestamp utilities.

Models validate themselves on construction (``__post_init__``) so an invalid
object cannot be created accidentally. Repositories call :meth:`validate`
again immediately before writing, because the dataclasses are mutable.
The same rules are reinforced by CHECK constraints and triggers in SQLite.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from chest_hunter.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SPAWN_PROBABILITY = 0.8
DEFAULT_OPEN_SECONDS = 2.0

MAP_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
MAX_LOCATION_NAME_LENGTH = 200
MAX_NOTES_LENGTH = 2000

Clock = Callable[[], datetime]
"""A zero-argument callable returning the current time as an aware datetime."""


# ---------------------------------------------------------------------------
# Timestamps (UTC, ISO 8601)
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def format_timestamp(moment: datetime) -> str:
    """Format an aware datetime as an ISO 8601 UTC string with microseconds.

    A fixed format keeps lexicographic order identical to chronological order.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValidationError("Timestamps must be timezone-aware (UTC expected).")
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp produced by :func:`format_timestamp`."""
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid ISO 8601 timestamp: {value!r}") from exc
    if moment.tzinfo is None:
        raise ValidationError(f"Timestamp lacks a UTC offset: {value!r}")
    return moment.astimezone(timezone.utc)


def timestamp_now(clock: Clock = utc_now) -> str:
    """Return ``clock()`` formatted as an ISO 8601 UTC string."""
    return format_timestamp(clock())


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def require_finite_number(name: str, value: Any) -> float:
    """Return ``value`` as a float, rejecting booleans, non-numbers, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be a finite number, got {value!r}.")
    return number


def require_probability(name: str, value: Any) -> float:
    """Return ``value`` as a float in the closed interval [0, 1]."""
    number = require_finite_number(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1, got {value!r}.")
    return number


def require_non_negative(name: str, value: Any) -> float:
    """Return ``value`` as a finite float that is >= 0."""
    number = require_finite_number(name, value)
    if number < 0.0:
        raise ValidationError(f"{name} must not be negative, got {value!r}.")
    return number


def require_bool(name: str, value: Any) -> bool:
    """Return ``value`` if it is a real ``bool``."""
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be true or false, got {value!r}.")
    return value


def require_positive_int(name: str, value: Any) -> int:
    """Return ``value`` if it is an integer >= 1 (booleans rejected)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{name} must be a positive integer, got {value!r}.")
    return value


def require_optional_id(name: str, value: Any) -> int | None:
    """Return ``value`` if it is ``None`` or a positive integer."""
    return None if value is None else require_positive_int(name, value)


def require_map_id(value: Any) -> str:
    """Validate a map identifier (letters, digits, ``_ . -``; max 64 chars)."""
    if not isinstance(value, str):
        raise ValidationError(f"map_id must be a string, got {value!r}.")
    map_id = value.strip()
    if not MAP_ID_PATTERN.fullmatch(map_id):
        raise ValidationError(
            "map_id must be 1-64 characters, start with a letter or digit and "
            f"contain only letters, digits, '_', '.' or '-'; got {value!r}."
        )
    return map_id


def require_text(name: str, value: Any, max_length: int) -> str:
    """Validate a non-empty, stripped string no longer than ``max_length``."""
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {value!r}.")
    text = value.strip()
    if not text:
        raise ValidationError(f"{name} must not be empty.")
    if len(text) > max_length:
        raise ValidationError(f"{name} must be at most {max_length} characters long.")
    return text


def require_optional_text(name: str, value: Any, max_length: int) -> str | None:
    """Like :func:`require_text`, but ``None`` and blank strings become ``None``."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return require_text(name, value, max_length)


def require_timestamp(name: str, value: Any) -> str:
    """Validate an ISO 8601 timestamp string with a UTC offset."""
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be an ISO 8601 string, got {value!r}.")
    parse_timestamp(value)
    return value


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass
class ChestLocation:
    """A *possible* chest spawn point on a simulated map.

    A location is not a guarantee that a chest exists: ``spawn_probability``
    is an estimate, and actual presence is recorded per session as a
    :class:`ChestObservation`.
    """

    id: int | None
    map_id: str
    location_name: str
    x: float
    y: float
    z: float
    spawn_probability: float
    estimated_open_seconds: float
    enabled: bool
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate and normalise every field in place."""
        self.id = require_optional_id("id", self.id)
        self.map_id = require_map_id(self.map_id)
        self.location_name = require_text(
            "location_name", self.location_name, MAX_LOCATION_NAME_LENGTH
        )
        self.x = require_finite_number("x", self.x)
        self.y = require_finite_number("y", self.y)
        self.z = require_finite_number("z", self.z)
        self.spawn_probability = require_probability(
            "spawn_probability", self.spawn_probability
        )
        self.estimated_open_seconds = require_non_negative(
            "estimated_open_seconds", self.estimated_open_seconds
        )
        self.enabled = require_bool("enabled", self.enabled)
        if self.created_at is not None:
            require_timestamp("created_at", self.created_at)
        if self.updated_at is not None:
            require_timestamp("updated_at", self.updated_at)

    @property
    def position(self) -> tuple[float, float, float]:
        """The location as an ``(x, y, z)`` tuple for simulators and planners."""
        return (self.x, self.y, self.z)


class SessionStatus(StrEnum):
    """Lifecycle states of a simulation session.

    Valid transitions: ``running -> completed`` and ``running -> aborted``.
    ``completed`` and ``aborted`` are final.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"

    @property
    def is_final(self) -> bool:
        return self is not SessionStatus.RUNNING


@dataclass
class SimulationSession:
    """A single research simulation run on one map."""

    id: int | None
    map_id: str
    started_at: str
    ended_at: str | None
    status: SessionStatus
    notes: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.id = require_optional_id("id", self.id)
        self.map_id = require_map_id(self.map_id)
        require_timestamp("started_at", self.started_at)
        try:
            self.status = SessionStatus(self.status)
        except ValueError as exc:
            allowed = ", ".join(status.value for status in SessionStatus)
            raise ValidationError(
                f"status must be one of {allowed}; got {self.status!r}."
            ) from exc
        if self.status is SessionStatus.RUNNING:
            if self.ended_at is not None:
                raise ValidationError("A running session cannot have ended_at set.")
        else:
            if self.ended_at is None:
                raise ValidationError(f"A {self.status.value} session requires ended_at.")
            require_timestamp("ended_at", self.ended_at)
            if parse_timestamp(self.ended_at) < parse_timestamp(self.started_at):
                raise ValidationError("ended_at must not be earlier than started_at.")
        self.notes = require_optional_text("notes", self.notes, MAX_NOTES_LENGTH)

    @property
    def is_active(self) -> bool:
        return self.status is SessionStatus.RUNNING

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed seconds between start and end, or ``None`` while running."""
        if self.ended_at is None:
            return None
        delta = parse_timestamp(self.ended_at) - parse_timestamp(self.started_at)
        return delta.total_seconds()


@dataclass
class ChestObservation:
    """The outcome of visiting/evaluating one chest location in one session."""

    id: int | None
    session_id: int
    chest_location_id: int
    observed_at: str
    chest_present: bool
    chest_opened: bool
    interaction_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.id = require_optional_id("id", self.id)
        self.session_id = require_positive_int("session_id", self.session_id)
        self.chest_location_id = require_positive_int(
            "chest_location_id", self.chest_location_id
        )
        require_timestamp("observed_at", self.observed_at)
        self.chest_present = require_bool("chest_present", self.chest_present)
        self.chest_opened = require_bool("chest_opened", self.chest_opened)
        self.interaction_seconds = require_non_negative(
            "interaction_seconds", self.interaction_seconds
        )
        if self.chest_opened and not self.chest_present:
            raise ValidationError(
                "A chest cannot be recorded as opened when it was not present."
            )


# ---------------------------------------------------------------------------
# Statistics (read-only result objects)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChestStatistics:
    """Aggregate statistics for one map.

    ``observed_spawn_frequency`` is ``None`` (explicitly *unavailable*) when
    fewer than ``min_spawn_samples`` session/location samples exist.
    """

    map_id: str
    total_locations: int
    enabled_locations: int
    average_spawn_probability: float | None
    total_visits: int
    chests_found: int
    successful_openings: int
    spawn_samples: int
    min_spawn_samples: int
    observed_spawn_frequency: float | None

    @property
    def spawn_frequency_available(self) -> bool:
        return self.observed_spawn_frequency is not None


@dataclass(frozen=True)
class SessionStatistics:
    """Aggregate statistics for one simulation session.

    * ``locations_visited`` - distinct chest locations with >= 1 observation.
    * ``total_observations`` - every recorded observation (including revisits).
    * ``chests_found`` - distinct locations where a chest was observed present.
    * ``chests_opened`` - successful openings (at most one per location).
    * ``failed_interactions`` - observations where a chest *was present* but
      was *not* opened. Visiting an empty spawn point is not a failure.
    * ``duration_seconds`` - ``None`` while the session is still running.
    * ``chests_opened_per_minute`` - ``None`` unless the duration is > 0.
    """

    session_id: int
    map_id: str
    status: SessionStatus
    locations_visited: int
    total_observations: int
    chests_found: int
    chests_opened: int
    failed_interactions: int
    total_interaction_seconds: float
    duration_seconds: float | None
    chests_opened_per_minute: float | None
