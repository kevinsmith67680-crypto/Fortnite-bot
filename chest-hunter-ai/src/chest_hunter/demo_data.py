"""Deterministic demo dataset for the fictional ``research_arena_01`` map.

These are invented research-arena spawn points, **not** real Fortnite chest
coordinates. The arena is 1000 x 1000 units: ``x`` grows to the east, ``y``
grows to the north and ``z`` is height above ground (negative = below ground).
Coordinates are stored exactly like any other chest location, so a future 2D
simulator can plot them directly.

Seeding is idempotent: a demo spawn point is identified by its map and exact
``(x, y, z)`` position, and positions that already exist are skipped (never
overwritten), so repeated seeding never creates duplicates or clobbers edits.
"""

from __future__ import annotations

from dataclasses import dataclass

from chest_hunter.chest_repository import ChestRepository
from chest_hunter.database import DatabaseManager
from chest_hunter.models import Clock, require_map_id, utc_now

DEMO_MAP_ID = "research_arena_01"
ARENA_WIDTH = 1000.0
ARENA_HEIGHT = 1000.0


@dataclass(frozen=True)
class DemoChest:
    region: str
    spot: str
    x: float
    y: float
    z: float
    spawn_probability: float
    estimated_open_seconds: float

    @property
    def location_name(self) -> str:
        return f"{self.region} - {self.spot}"


DEMO_CHESTS: tuple[DemoChest, ...] = (
    # Central Warehouse (arena centre)
    DemoChest("Central Warehouse", "Loading Bay", 470.0, 455.0, 0.0, 0.90, 2.0),
    DemoChest("Central Warehouse", "Office", 525.0, 540.0, 4.0, 0.75, 2.0),
    DemoChest("Central Warehouse", "Mezzanine", 505.0, 500.0, 8.0, 0.85, 2.0),
    DemoChest("Central Warehouse", "Rear Storage", 548.0, 468.0, 0.0, 0.65, 2.5),
    DemoChest("Central Warehouse", "Forklift Bay", 455.0, 520.0, 0.0, 0.70, 2.0),
    DemoChest("Central Warehouse", "Roof Hatch", 500.0, 475.0, 12.0, 0.55, 3.0),
    DemoChest("Central Warehouse", "Container Stack", 530.0, 430.0, 3.0, 0.80, 2.0),
    # Eastern Village
    DemoChest("Eastern Village", "Blue House Attic", 790.0, 560.0, 6.0, 0.85, 2.0),
    DemoChest("Eastern Village", "Blue House Basement", 792.0, 562.0, -3.0, 0.60, 2.0),
    DemoChest("Eastern Village", "General Store", 835.0, 505.0, 0.0, 0.80, 1.5),
    DemoChest("Eastern Village", "Chapel", 870.0, 540.0, 0.0, 0.70, 2.0),
    DemoChest("Eastern Village", "Well Courtyard", 815.0, 470.0, 0.0, 0.50, 1.5),
    DemoChest("Eastern Village", "Barn Loft", 880.0, 450.0, 5.0, 0.90, 2.5),
    DemoChest("Eastern Village", "Garden Shed", 760.0, 495.0, 0.0, 0.65, 1.5),
    # Northern Forest
    DemoChest("Northern Forest", "Ranger Tower", 450.0, 830.0, 10.0, 0.95, 2.0),
    DemoChest("Northern Forest", "Hollow Log", 395.0, 880.0, 0.0, 0.55, 2.5),
    DemoChest("Northern Forest", "Campsite", 520.0, 870.0, 0.0, 0.75, 2.0),
    DemoChest("Northern Forest", "Hunting Blind", 580.0, 905.0, 4.0, 0.60, 2.0),
    DemoChest("Northern Forest", "Creek Bridge", 470.0, 790.0, 0.0, 0.70, 1.5),
    DemoChest("Northern Forest", "Cabin", 610.0, 840.0, 0.0, 0.85, 2.0),
    DemoChest("Northern Forest", "Treehouse", 360.0, 820.0, 7.0, 0.80, 3.0),
    # Western Outpost
    DemoChest("Western Outpost", "Watchtower", 140.0, 520.0, 9.0, 0.90, 2.0),
    DemoChest("Western Outpost", "Barracks", 165.0, 480.0, 0.0, 0.80, 2.0),
    DemoChest("Western Outpost", "Supply Tent", 120.0, 455.0, 0.0, 0.70, 1.5),
    DemoChest("Western Outpost", "Radio Shack", 185.0, 545.0, 0.0, 0.75, 2.0),
    DemoChest("Western Outpost", "Gate House", 210.0, 500.0, 0.0, 0.60, 2.0),
    DemoChest("Western Outpost", "Bunker", 100.0, 500.0, -4.0, 0.95, 3.0),
    DemoChest("Western Outpost", "Vehicle Depot", 150.0, 410.0, 0.0, 0.65, 2.5),
    # Southern Docks
    DemoChest("Southern Docks", "Harbour Master", 480.0, 140.0, 3.0, 0.85, 2.0),
    DemoChest("Southern Docks", "Pier 1", 430.0, 90.0, 0.0, 0.70, 1.5),
    DemoChest("Southern Docks", "Pier 2", 510.0, 85.0, 0.0, 0.65, 1.5),
    DemoChest("Southern Docks", "Fish Market", 560.0, 150.0, 0.0, 0.75, 2.0),
    DemoChest("Southern Docks", "Cargo Ship Deck", 600.0, 60.0, 6.0, 0.90, 2.5),
    DemoChest("Southern Docks", "Boathouse", 380.0, 130.0, 0.0, 0.60, 2.0),
    DemoChest("Southern Docks", "Crane Cabin", 540.0, 110.0, 14.0, 0.55, 3.0),
)

DEMO_REGIONS: tuple[str, ...] = tuple(dict.fromkeys(chest.region for chest in DEMO_CHESTS))


@dataclass(frozen=True)
class SeedResult:
    map_id: str
    inserted: int
    skipped: int

    @property
    def total(self) -> int:
        return self.inserted + self.skipped


def seed_demo_chests(
    db: DatabaseManager, map_id: str = DEMO_MAP_ID, *, clock: Clock = utc_now
) -> SeedResult:
    """Insert the demo spawn points on ``map_id`` atomically, skipping existing positions."""
    map_id = require_map_id(map_id)
    repository = ChestRepository(db, clock=clock)
    inserted = skipped = 0
    with db.transaction():
        for chest in DEMO_CHESTS:
            if repository.find_chest_at(map_id, chest.x, chest.y, chest.z) is not None:
                skipped += 1
                continue
            repository.add_chest(
                map_id=map_id,
                location_name=chest.location_name,
                x=chest.x,
                y=chest.y,
                z=chest.z,
                spawn_probability=chest.spawn_probability,
                estimated_open_seconds=chest.estimated_open_seconds,
            )
            inserted += 1
    return SeedResult(map_id=map_id, inserted=inserted, skipped=skipped)
