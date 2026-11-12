"""Exception hierarchy shared by every chest_hunter module.

Every error that a caller is expected to handle derives from
:class:`ChestHunterError`, so the CLI (and future simulator code) can catch a
single base class and report a clean message instead of a traceback.
"""

from __future__ import annotations


class ChestHunterError(Exception):
    """Base class for all expected, user-reportable errors."""


class ConfigError(ChestHunterError):
    """The configuration file is missing, unreadable or invalid."""


class ValidationError(ChestHunterError, ValueError):
    """A value supplied by the caller failed validation."""


class RecordNotFoundError(ChestHunterError, LookupError):
    """A referenced database record does not exist."""


class ConflictError(ChestHunterError):
    """The operation conflicts with existing data or with a business rule."""


class DuplicateChestError(ConflictError):
    """A chest location already exists at the same map coordinates."""


class ChestInUseError(ConflictError):
    """A chest location is referenced by observations and cannot be changed that way."""


class SessionStateError(ConflictError):
    """A session is not in the state required by the requested operation."""


class DuplicateOpeningError(ConflictError):
    """A chest location was already opened successfully in the same session."""


class DatabaseError(ChestHunterError):
    """SQLite reported an error that is not a simple validation problem."""


class DatabaseIntegrityError(DatabaseError):
    """SQLite rejected a write because it violated a schema constraint."""
