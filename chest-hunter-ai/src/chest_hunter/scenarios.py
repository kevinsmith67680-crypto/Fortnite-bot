"""Named research scenarios: chest layouts plus hand-placed obstacles.

``visual_arena_01`` is the Phase 3 acceptance arena: 20 fictional spawn
locations spread over the 1000 x 1000-unit (100 x 100-cell) map, procedural
walls and obstacles, and a sealed vault whose chest always spawns but cannot
be reached (its enclosure has no door). The agent can see into the vault from
above but never reach it.

Like the Phase 1 demo arena, these coordinates are invented and are not
real game data.
"""

from __future__ import annotations

from dataclasses import dataclass

from chest_hunter.database import DatabaseManager
from chest_hunter.demo_data import DemoChest, SeedResult, seed_demo_chests
from chest_hunter.models import Clock, utc_now
from chest_hunter.simulation.entities import Obstacle

VISUAL_ARENA_ID = "visual_arena_01"

VISUAL_ARENA_CHESTS: tuple[DemoChest, ...] = (
    DemoChest("North Hangar", "Crate Row", 150.0, 870.0, 0.0, 0.90, 2.0),
    DemoChest("North Hangar", "Office", 260.0, 820.0, 0.0, 0.60, 2.0),
    DemoChest("Radar Hill", "Dish", 420.0, 905.0, 0.0, 0.70, 2.0),
    DemoChest("Radar Hill", "Bunker", 520.0, 780.0, 0.0, 0.80, 2.0),
    DemoChest("Sealed Vault", "Inside", 855.0, 855.0, 0.0, 1.00, 2.0),
    DemoChest("East Farm", "Silo", 900.0, 640.0, 0.0, 0.75, 2.0),
    DemoChest("East Farm", "Barn", 820.0, 560.0, 0.0, 0.85, 2.0),
    DemoChest("East Farm", "Well", 760.0, 450.0, 0.0, 0.50, 2.0),
    DemoChest("River Bend", "Dock", 640.0, 300.0, 0.0, 0.70, 2.0),
    DemoChest("River Bend", "Shack", 560.0, 220.0, 0.0, 0.65, 2.0),
    DemoChest("South Quarry", "Pit", 420.0, 90.0, 0.0, 0.80, 2.0),
    DemoChest("South Quarry", "Crane", 300.0, 160.0, 0.0, 0.55, 2.0),
    DemoChest("West Woods", "Camp", 120.0, 380.0, 0.0, 0.75, 2.0),
    DemoChest("West Woods", "Lodge", 90.0, 520.0, 0.0, 0.60, 2.0),
    DemoChest("West Woods", "Tower", 200.0, 600.0, 0.0, 0.90, 2.0),
    DemoChest("Central Plaza", "Fountain", 480.0, 520.0, 0.0, 0.70, 2.0),
    DemoChest("Central Plaza", "Kiosk", 560.0, 470.0, 0.0, 0.50, 2.0),
    DemoChest("Old Mill", "Gear Room", 330.0, 420.0, 0.0, 0.85, 2.0),
    DemoChest("Old Mill", "Loft", 360.0, 300.0, 0.0, 0.60, 2.0),
    DemoChest("Lighthouse", "Base", 930.0, 120.0, 0.0, 0.95, 2.0),
)

# A closed 5 x 5 ring of walls (cells 83-87) around the vault chest at cell (85, 85).
_VAULT_WALLS: tuple[Obstacle, ...] = (
    Obstacle(83, 83, 5, 1, "vault"),
    Obstacle(83, 87, 5, 1, "vault"),
    Obstacle(83, 84, 1, 3, "vault"),
    Obstacle(87, 84, 1, 3, "vault"),
)


@dataclass(frozen=True)
class Scenario:
    map_id: str
    chests: tuple[DemoChest, ...]
    obstacles: tuple[Obstacle, ...]
    unreachable: tuple[str, ...]


SCENARIOS: dict[str, Scenario] = {
    VISUAL_ARENA_ID: Scenario(VISUAL_ARENA_ID, VISUAL_ARENA_CHESTS, _VAULT_WALLS,
                              unreachable=("Sealed Vault - Inside",)),
}


def scenario_obstacles(map_id: str) -> tuple[Obstacle, ...]:
    """Hand-placed obstacles registered for ``map_id`` (empty for other maps)."""
    scenario = SCENARIOS.get(map_id)
    return scenario.obstacles if scenario else ()


def seed_scenario(db: DatabaseManager, map_id: str = VISUAL_ARENA_ID, *,
                  clock: Clock = utc_now) -> SeedResult:
    """Insert a scenario's spawn locations (idempotent, like ``seed-demo``)."""
    scenario = SCENARIOS.get(map_id)
    if scenario is None:
        raise KeyError(f"Unknown scenario map '{map_id}'.")
    return seed_demo_chests(db, map_id, clock=clock, chests=scenario.chests)


ARENA_3D_ID = "research_arena_3d"


def arena_3d_chests() -> tuple[DemoChest, ...]:
    """The 35 spawn points of the Phase 4 3D arena (metres; z = floor height)."""
    from chest_hunter.simulation3d.arena import build_research_arena_3d  # lazy: needs NumPy

    return tuple(DemoChest(s.region, s.name, s.x, s.y, s.z, s.probability, 2.0)
                 for s in build_research_arena_3d().spawns)


def seed_arena_3d(db: DatabaseManager, *, clock: Clock = utc_now) -> SeedResult:
    """Insert the 3D arena's spawn points (idempotent)."""
    return seed_demo_chests(db, ARENA_3D_ID, clock=clock, chests=arena_3d_chests())


def procedural_chests(map_id: str) -> tuple[DemoChest, ...]:
    """Spawn points of a procedural 3D map (``proc_train_NNN`` / ``proc_test_NNN``).

    The database receives the generator's *uninformative prior* (``DATABASE_PRIOR``), not the
    hidden true spawn probabilities: learning those from observations is the point of the
    Phase 5 spawn estimator."""
    from chest_hunter.simulation3d.procedural import build_procedural_arena, database_prior

    return tuple(DemoChest(s.region, s.name, s.x, s.y, s.z, database_prior(s), 2.0)
                 for s in build_procedural_arena(map_id).spawns)


def seed_procedural_map(db: DatabaseManager, map_id: str, *,
                        clock: Clock = utc_now) -> SeedResult:
    """Insert a procedural map's spawn points (idempotent)."""
    return seed_demo_chests(db, map_id, clock=clock, chests=procedural_chests(map_id))
