"""chest_hunter - Phase 1 data foundation for a chest-hunting research agent.

Phase 1 provides persistent storage only (chest spawn locations, simulation
sessions and chest observations) plus a CLI. It contains no movement,
computer vision, game input or autonomous navigation code, and it never
interacts with a real game client.

Typical programmatic use::

    from chest_hunter import ChestRepository, DatabaseManager, SessionRepository

    with DatabaseManager("data/chest_locations.db") as db:
        db.initialize()
        chests = ChestRepository(db)
        sessions = SessionRepository(db)
"""

from chest_hunter.chest_repository import ChestRepository
from chest_hunter.database import SCHEMA_VERSION, DatabaseManager
from chest_hunter.exceptions import (
    ChestHunterError,
    ChestInUseError,
    ConfigError,
    ConflictError,
    DatabaseError,
    DatabaseIntegrityError,
    DuplicateChestError,
    DuplicateOpeningError,
    RecordNotFoundError,
    SessionStateError,
    ValidationError,
)
from chest_hunter.models import (
    ChestLocation,
    ChestObservation,
    ChestStatistics,
    SessionStatistics,
    SessionStatus,
    SimulationSession,
)
from chest_hunter.session_repository import SessionRepository

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "ChestHunterError",
    "ChestInUseError",
    "ChestLocation",
    "ChestObservation",
    "ChestRepository",
    "ChestStatistics",
    "ConfigError",
    "ConflictError",
    "DatabaseError",
    "DatabaseIntegrityError",
    "DatabaseManager",
    "DuplicateChestError",
    "DuplicateOpeningError",
    "RecordNotFoundError",
    "SessionRepository",
    "SessionStateError",
    "SessionStatistics",
    "SessionStatus",
    "SimulationSession",
    "ValidationError",
    "__version__",
]
