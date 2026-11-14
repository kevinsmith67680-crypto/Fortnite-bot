"""The Phase 4 research arena: a fictional 100 m x 100 m third-person map.

Single source of truth for the arena layout. The same definition is used by
the Python reference environment and exported as JSON
(:func:`arena_to_json`) for the Unreal project's arena builder, so both
environments contain identical geometry and spawn points.

Axes: X north, Y east, Z up, metres. Regions (all fictional):

* **Central Warehouse**: a large two-level building with a mezzanine reached
  by stairs, crate stacks and pillars.
* **Eastern Settlement**: small houses with doorways (one deliberately
  narrow), a two-storey house with an internal staircase, a fence.
* **Northern Forest**: trees, rocks, fallen logs, a ranger hut and a water
  tower whose platform has no stairs (its chest is visible but unreachable).
* **Western Outpost**: a walled compound with gates, a watchtower reached by
  stairs, sandbags and a tent.
* **Southern Docks**: water, a raised pier, shipping containers (one sealed,
  with a chest inside that is neither visible nor reachable), a boathouse and
  gold-coloured decoy barrels.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Final

from chest_hunter.simulation3d.geometry import Box, box

ARENA_3D_ID: Final = "research_arena_3d"

CHEST_WIDTH: Final = 0.9        # metres, across the chest front
CHEST_DEPTH: Final = 0.6        # metres, front to back
CHEST_BODY_HEIGHT: Final = 0.5
CHEST_LID_HEIGHT: Final = 0.15
CHEST_HEIGHT: Final = CHEST_BODY_HEIGHT + CHEST_LID_HEIGHT


@dataclass(frozen=True, slots=True)
class SpawnPoint3D:
    """A possible chest location. ``z`` is the floor height it stands on."""

    region: str
    name: str
    x: float
    y: float
    z: float
    yaw_deg: float            # direction the chest front faces (0, 90, 180 or 270)
    probability: float
    tags: tuple[str, ...] = ()
    reachable: bool = True    # design intent; verified by the navigation tests

    @property
    def location_name(self) -> str:
        return f"{self.region} - {self.name}"


@dataclass(slots=True)
class Arena3D:
    map_id: str
    boxes: list[Box]
    spawns: list[SpawnPoint3D]
    start: tuple[float, float, float]
    start_yaw_deg: float
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0)
    regions: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    #: Rendering variation (procedural maps): ``lighting`` (global brightness factor),
    #: ``chest_value`` (chest brightness factor), ``chest_hue_shift`` (degrees). Empty for
    #: the hand-made arena, whose rendering is therefore unchanged.
    appearance: dict[str, float] = field(default_factory=dict)


# -- building blocks ---------------------------------------------------------------------


def _wall_x(x0: float, x1: float, y: float, height: float, colour: str, *, thick: float = 0.3,
            gaps: tuple[tuple[float, float], ...] = (), lintel: float = 2.4,
            z0: float = 0.0, kind: str = "wall") -> list[Box]:
    """Wall running along X at ``y`` (thickness towards +y), with door gaps in X."""
    return _wall(x0, x1, height, colour, thick, gaps, lintel, z0, kind,
                 lambda a, b, lo, hi: box(a, y, lo, b, y + thick, hi, kind, colour))


def _wall_y(y0: float, y1: float, x: float, height: float, colour: str, *, thick: float = 0.3,
            gaps: tuple[tuple[float, float], ...] = (), lintel: float = 2.4,
            z0: float = 0.0, kind: str = "wall") -> list[Box]:
    """Wall running along Y at ``x`` (thickness towards +x), with door gaps in Y."""
    return _wall(y0, y1, height, colour, thick, gaps, lintel, z0, kind,
                 lambda a, b, lo, hi: box(x, a, lo, x + thick, b, hi, kind, colour))


def _wall(a0: float, a1: float, height: float, colour: str, thick: float,
          gaps: tuple[tuple[float, float], ...], lintel: float, z0: float, kind: str,
          make: object) -> list[Box]:
    assert callable(make)
    out: list[Box] = []
    cursor = a0
    for g0, g1 in sorted(gaps):
        if g0 > cursor:
            out.append(make(cursor, g0, z0, z0 + height))
        if lintel + z0 < z0 + height:
            out.append(make(g0, g1, z0 + lintel, z0 + height))
        cursor = g1
    if cursor < a1:
        out.append(make(cursor, a1, z0, z0 + height))
    return out


def _building(x0: float, y0: float, x1: float, y1: float, height: float, colour: str,
              doors: dict[str, tuple[float, float]], *, thick: float = 0.3,
              floor: bool = True, name: str = "") -> list[Box]:
    """Rectangular building. ``doors`` maps a side (S, N, W, E) to (centre, width).

    South is the wall at ``x0``, north at ``x1``, west at ``y0``, east at ``y1``.
    """
    def gaps(side: str) -> tuple[tuple[float, float], ...]:
        if side not in doors:
            return ()
        c, w = doors[side]
        return ((c - w / 2, c + w / 2),)

    out: list[Box] = []
    out += _wall_y(y0, y1, x0, height, colour, thick=thick, gaps=gaps("S"))
    out += _wall_y(y0, y1, x1 - thick, height, colour, thick=thick, gaps=gaps("N"))
    out += _wall_x(x0 + thick, x1 - thick, y0, height, colour, thick=thick, gaps=gaps("W"))
    out += _wall_x(x0 + thick, x1 - thick, y1 - thick, height, colour, thick=thick,
                   gaps=gaps("E"))
    if floor:
        out.append(box(x0 + thick, y0 + thick, 0.0, x1 - thick, y1 - thick, 0.02, "floor",
                       "interior_floor", blocks=False, name=f"{name} floor"))
    return out


def _stairs(x: float, y: float, direction: str, steps: int, *, rise: float = 0.25,
            run: float = 0.35, width: float = 1.4, z0: float = 0.0) -> list[Box]:
    """Solid steps starting at (x, y) and climbing towards ``direction`` (+X, -X, +Y, -Y).

    (x, y) is the middle of the bottom edge; each step is a block from ``z0``.
    """
    out = []
    for i in range(steps):
        a, b = i * run, (i + 1) * run
        top = z0 + rise * (i + 1)
        if direction == "+X":
            out.append(box(x + a, y - width / 2, z0, x + b, y + width / 2, top, "stairs", "stairs"))
        elif direction == "-X":
            out.append(box(x - b, y - width / 2, z0, x - a, y + width / 2, top, "stairs", "stairs"))
        elif direction == "+Y":
            out.append(box(x - width / 2, y + a, z0, x + width / 2, y + b, top, "stairs", "stairs"))
        else:
            out.append(box(x - width / 2, y - b, z0, x + width / 2, y - a, top, "stairs", "stairs"))
    return out


def _railing_x(x0: float, x1: float, y: float, z: float, height: float = 1.0) -> Box:
    return box(x0, y, z, x1, y + 0.1, z + height, "railing", "railing")


def _railing_y(y0: float, y1: float, x: float, z: float, height: float = 1.0) -> Box:
    return box(x, y0, z, x + 0.1, y1, z + height, "railing", "railing")


def _container(x0: float, y0: float, colour: str, *, along_y: bool = True) -> Box:
    w, length = 2.4, 6.0
    x1, y1 = (x0 + w, y0 + length) if along_y else (x0 + length, y0 + w)
    return box(x0, y0, 0.0, x1, y1, 2.6, "container", colour)


def _tree(x: float, y: float) -> list[Box]:
    return [box(x - 0.25, y - 0.25, 0.0, x + 0.25, y + 0.25, 3.2, "tree", "trunk"),
            box(x - 1.3, y - 1.3, 2.6, x + 1.3, y + 1.3, 4.4, "canopy", "canopy", blocks=False)]


# -- the arena ---------------------------------------------------------------------------

REGIONS: Final[dict[str, tuple[float, float, float, float]]] = {
    # name: (x0, y0, x1, y1)
    "Central Warehouse": (36.0, 34.0, 66.0, 66.0),
    "Eastern Settlement": (26.0, 70.0, 76.0, 98.0),
    "Northern Forest": (75.0, 2.0, 98.0, 98.0),
    "Western Outpost": (28.0, 2.0, 62.0, 30.0),
    "Southern Docks": (0.0, 2.0, 25.0, 98.0),
}


def _spawns() -> list[SpawnPoint3D]:
    cw, es, nf, wo, sd = ("Central Warehouse", "Eastern Settlement", "Northern Forest",
                          "Western Outpost", "Southern Docks")
    s = SpawnPoint3D
    return [
        # Central Warehouse: interior, behind crates, pillar, mezzanine (upstairs), outside.
        s(cw, "Loading Bay", 43.0, 44.0, 0.02, 0.0, 0.85, ("inside",)),
        s(cw, "Behind Crates", 48.0, 58.8, 0.02, 0.0, 0.7, ("inside", "behind_obstacle")),
        s(cw, "Pillar", 54.0, 51.2, 0.02, 180.0, 0.6, ("inside", "partially_obscured")),
        s(cw, "Mezzanine West", 56.0, 41.0, 3.5, 180.0, 0.75, ("upstairs",)),
        s(cw, "Mezzanine East", 57.5, 59.5, 3.5, 180.0, 0.65, ("upstairs",)),
        s(cw, "South Wall", 37.0, 44.0, 0.0, 180.0, 0.8, ("open",)),
        s("Central Plaza", "Fountain Square", 64.0, 50.0, 0.02, 0.0, 0.7, ("open",)),
        s("South Field", "Haystack", 28.0, 66.0, 0.0, 90.0, 0.6, ("open",)),
        # Eastern Settlement.
        s(es, "House 1", 60.5, 80.5, 0.02, 270.0, 0.8, ("inside", "doorway")),
        s(es, "House 2", 71.5, 83.5, 0.02, 180.0, 0.7, ("inside", "doorway")),
        s(es, "Narrow Door House", 49.5, 91.5, 0.02, 270.0, 0.75, ("inside", "narrow_doorway")),
        s(es, "Two-Storey House Ground", 37.5, 87.5, 0.02, 180.0, 0.6, ("inside",)),
        s(es, "Two-Storey House Upstairs", 38.0, 85.0, 3.5, 180.0, 0.8, ("inside", "upstairs")),
        s(es, "Village Square", 52.0, 79.0, 0.0, 0.0, 0.85, ("open",)),
        s(es, "Behind Fence", 69.0, 93.0, 0.0, 270.0, 0.55, ("behind_obstacle",)),
        # Northern Forest.
        s(nf, "Clearing", 84.0, 40.0, 0.0, 0.0, 0.9, ("open",)),
        s(nf, "Ranger Hut", 91.0, 77.8, 0.02, 270.0, 0.7, ("inside", "doorway")),
        s(nf, "Behind Rocks", 80.2, 22.2, 0.0, 0.0, 0.65, ("behind_obstacle",)),
        s(nf, "Fallen Log", 93.0, 53.2, 0.0, 0.0, 0.6, ("behind_obstacle",)),
        s(nf, "Deep Woods", 86.0, 12.0, 0.0, 90.0, 0.5, ("partially_obscured",)),
        s(nf, "Water Tower Platform", 89.5, 61.5, 3.0, 180.0, 0.9, ("upstairs", "unreachable"),
          reachable=False),
        s(nf, "Treeline", 78.0, 66.0, 0.0, 180.0, 0.7, ("partially_obscured",)),
        # Western Outpost.
        s(wo, "Gatehouse", 44.0, 23.2, 0.0, 90.0, 0.8, ("inside_walls", "doorway")),
        s(wo, "Watchtower Platform", 49.8, 9.5, 4.0, 180.0, 0.75, ("upstairs",)),
        s(wo, "Behind Sandbags", 55.6, 15.0, 0.0, 0.0, 0.7, ("behind_obstacle",)),
        s(wo, "Tent", 42.0, 18.8, 0.0, 0.0, 0.65, ("partially_obscured",)),
        s(wo, "Courtyard", 52.0, 21.0, 0.0, 90.0, 0.85, ("open",)),
        s(wo, "Outside Wall", 32.0, 14.0, 0.0, 0.0, 0.6, ("open",)),
        # Southern Docks.
        s(sd, "Pier End", 1.7, 47.0, 0.5, 0.0, 0.8, ("upstairs", "open")),
        s(sd, "Container Row", 14.45, 31.0, 0.0, 90.0, 0.7, ("behind_obstacle",)),
        s(sd, "Sealed Container", 12.2, 61.0, 0.12, 0.0, 1.0, ("inside", "unreachable"),
          reachable=False),
        s(sd, "Crane Base", 18.0, 82.0, 0.0, 0.0, 0.75, ("open",)),
        s(sd, "Boathouse", 10.5, 20.0, 0.0, 0.0, 0.7, ("inside",)),
        s(sd, "Quay", 8.5, 70.0, 0.0, 0.0, 0.8, ("open",)),
        s(sd, "Behind Barrels", 22.2, 53.6, 0.0, 0.0, 0.65, ("behind_obstacle", "decoys")),
    ]


def _trees(spawns: list[SpawnPoint3D], blocked: list[tuple[float, float, float, float]],
           seed: int = 7) -> list[Box]:
    rng = random.Random(f"forest:{seed}")
    placed: list[tuple[float, float]] = []
    out: list[Box] = []
    attempts = 0
    while len(placed) < 46 and attempts < 6000:
        attempts += 1
        x, y = rng.uniform(77.0, 97.0), rng.uniform(4.0, 96.0)
        if any((x - px) ** 2 + (y - py) ** 2 < 4.2 ** 2 for px, py in placed):
            continue
        if any((x - s.x) ** 2 + (y - s.y) ** 2 < 4.0 ** 2 for s in spawns):
            continue
        if any(x0 - 2.0 <= x <= x1 + 2.0 and y0 - 2.0 <= y <= y1 + 2.0
               for x0, y0, x1, y1 in blocked):
            continue
        if (x - 84.0) ** 2 + (y - 40.0) ** 2 < 7.0 ** 2:      # keep the clearing open
            continue
        placed.append((x, y))
        out += _tree(x, y)
    return out


def build_research_arena_3d() -> Arena3D:
    spawns = _spawns()
    b: list[Box] = []
    # Ground, water and the perimeter fence.
    b.append(box(0.0, 0.0, -0.2, 100.0, 100.0, 0.0, "ground", "grass"))
    b.append(box(0.0, 0.0, -0.2, 6.0, 100.0, 0.03, "water", "water", blocks=False,
                 walkable=False, name="sea"))
    b.append(box(0.0, -0.4, 0.0, 100.0, 0.0, 1.6, "fence", "fence", name="perimeter S"))
    b.append(box(0.0, 100.0, 0.0, 100.0, 100.4, 1.6, "fence", "fence", name="perimeter N"))
    b.append(box(-0.4, 0.0, 0.0, 0.0, 100.0, 1.6, "fence", "fence"))
    b.append(box(100.0, 0.0, 0.0, 100.4, 100.0, 1.6, "fence", "fence"))
    # Paved plaza and roads (visual only, walkable like the ground).
    b.append(box(26.0, 46.0, 0.0, 36.0, 54.0, 0.02, "floor", "concrete", blocks=False))
    b.append(box(60.0, 46.0, 0.0, 70.0, 54.0, 0.02, "floor", "concrete", blocks=False))

    # ---- Central Warehouse (X 40-60, Y 38-62), two levels ----
    b += _building(40.0, 38.0, 60.0, 62.0, 7.0, "warehouse_wall",
                   {"S": (50.0, 3.0), "N": (45.0, 1.5), "E": (48.0, 1.5)}, name="warehouse")
    b += _stairs(47.1, 39.25, "+X", 14, width=1.5)            # up to the mezzanine (z 3.5)
    b.append(box(52.0, 38.3, 3.3, 59.7, 61.7, 3.5, "floor", "interior_floor", name="mezzanine"))
    b.append(_railing_y(40.0, 61.7, 52.0, 3.5))
    for py in (44.0, 50.0, 56.0):
        b.append(box(54.75, py - 0.25, 0.02, 55.25, py + 0.25, 3.3, "pillar", "concrete"))
    b.append(box(46.0, 57.0, 0.02, 47.2, 60.6, 1.2, "crate", "crate", name="crate stack"))
    b.append(box(44.6, 57.4, 0.02, 45.8, 58.6, 1.2, "crate", "crate"))
    b.append(box(57.0, 44.8, 3.5, 58.2, 46.0, 4.7, "crate", "crate", name="mezzanine crate"))
    b.append(box(43.2, 52.2, 0.02, 44.0, 52.7, 0.4, "decoy", "decoy_gold", name="gold sacks"))

    # ---- Eastern Settlement ----
    b += _building(55.0, 75.0, 63.0, 83.0, 3.2, "house_wall", {"W": (59.0, 1.5)}, name="house 1")
    b += _building(66.0, 78.0, 74.0, 86.0, 3.2, "house_wall", {"S": (82.0, 1.5)}, name="house 2")
    b += _building(44.0, 86.0, 52.0, 94.0, 3.2, "house_wall", {"W": (48.0, 1.3)},
                   name="narrow door house")
    b += _building(30.0, 80.0, 40.0, 90.0, 6.8, "house_wall", {"S": (85.5, 1.5)},
                   name="two-storey house")
    b += _stairs(31.4, 81.15, "+X", 14, width=1.5)            # 31.4 -> 36.3, top z 3.5
    # The stairwell opening is wider than the stairs (Y 80.4-81.9) so heads clear the slab.
    b.append(box(30.3, 82.6, 3.3, 39.7, 89.7, 3.5, "floor", "interior_floor", name="upper floor"))
    b.append(box(36.3, 80.3, 3.3, 39.7, 82.6, 3.5, "floor", "interior_floor"))
    b.append(_railing_x(30.3, 36.3, 82.6, 3.5))
    b.append(box(64.0, 90.8, 0.0, 74.0, 91.0, 1.0, "fence", "fence", name="settlement fence"))
    b.append(box(56.5, 70.6, 0.0, 57.9, 71.6, 1.0, "decoy", "decoy_hay", name="hay bale"))
    b.append(box(58.4, 70.6, 0.0, 59.8, 71.6, 1.0, "decoy", "decoy_hay"))
    b.append(box(27.1, 64.1, 0.0, 28.5, 65.1, 1.0, "decoy", "decoy_hay"))

    # ---- Northern Forest ----
    hut = (88.0, 74.0, 94.0, 80.0)
    b += _building(*hut, 3.0, "warehouse_wall", {"W": (91.0, 1.5)}, name="ranger hut")
    rocks = (78.0, 18.5, 81.5, 21.0)
    b.append(box(78.0, 18.5, 0.0, 81.0, 20.0, 2.0, "rock", "rock"))
    b.append(box(79.5, 20.0, 0.0, 81.5, 21.0, 1.4, "rock", "rock"))
    b.append(box(91.0, 51.6, 0.0, 94.0, 52.1, 0.5, "log", "trunk", name="fallen log"))
    b.append(box(76.3, 64.8, 0.0, 77.1, 66.4, 1.0, "bush", "canopy", name="bushes"))
    tower = (88.0, 60.0, 91.0, 63.0)
    b.append(box(88.0, 60.0, 2.8, 91.0, 63.0, 3.0, "platform", "pier", name="water tower deck"))
    for lx, ly in ((88.0, 60.0), (90.7, 60.0), (88.0, 62.7), (90.7, 62.7)):
        b.append(box(lx, ly, 0.0, lx + 0.3, ly + 0.3, 2.8, "post", "post"))
    b.append(box(88.3, 60.3, 4.4, 90.7, 62.7, 6.4, "tank", "container_teal", name="water tank"))
    for lx, ly in ((88.3, 60.3), (90.5, 60.3), (88.3, 62.5), (90.5, 62.5)):
        b.append(box(lx, ly, 3.0, lx + 0.2, ly + 0.2, 4.4, "post", "post"))

    # ---- Western Outpost (compound X 35-60, Y 5-25) ----
    b += _building(35.0, 5.0, 60.0, 25.0, 2.5, "outpost_wall",
                   {"E": (46.25, 2.5), "N": (11.0, 1.5)}, floor=False, name="outpost")
    b += _stairs(40.6, 9.0, "+X", 16, run=0.4, width=1.4)      # 40.6 -> 47.0, top z 4.0
    b.append(box(47.0, 7.0, 3.8, 51.0, 11.0, 4.0, "platform", "pier", name="watchtower"))
    for lx, ly in ((47.0, 7.0), (50.7, 7.0), (47.0, 10.7), (50.7, 10.7)):
        b.append(box(lx, ly, 0.02, lx + 0.3, ly + 0.3, 3.8, "post", "post"))
    b.append(_railing_y(7.0, 11.0, 50.9, 4.0))
    b.append(_railing_x(47.0, 51.0, 7.0, 4.0))
    b.append(_railing_x(47.0, 51.0, 10.9, 4.0))
    b.append(_railing_y(7.0, 8.3, 47.0, 4.0))
    b.append(_railing_y(9.7, 11.0, 47.0, 4.0))
    b.append(box(54.0, 12.0, 0.02, 54.6, 18.0, 1.0, "sandbag", "sandbag", name="sandbags"))
    b.append(box(38.0, 17.0, 0.02, 41.0, 21.0, 2.2, "tent", "tent", name="tent"))
    b.append(box(49.6, 13.0, 0.02, 50.4, 13.5, 0.4, "decoy", "decoy_gold", name="gold sacks"))

    # ---- Southern Docks ----
    b += [box(6.0, 0.0, 0.0, 6.15, 45.5, 1.0, "railing", "railing", name="quay rail"),
          box(6.0, 48.5, 0.0, 6.15, 100.0, 1.0, "railing", "railing")]
    b.append(box(6.0, 45.5, 0.0, 6.6, 48.5, 0.25, "stairs", "stairs", name="pier step"))
    b.append(box(0.6, 45.0, 0.0, 6.0, 49.0, 0.5, "pier", "pier", name="pier"))
    b.append(box(0.6, 45.0, 0.5, 6.0, 45.1, 1.5, "railing", "railing"))
    b.append(box(0.6, 48.9, 0.5, 6.0, 49.0, 1.5, "railing", "railing"))
    b.append(box(0.6, 45.1, 0.5, 0.7, 48.9, 1.5, "railing", "railing"))
    b += [_container(11.0, 24.0, "container_red"), _container(11.0, 32.0, "container_blue"),
          _container(15.5, 24.0, "container_teal"), _container(15.5, 32.0, "container_red"),
          _container(15.5, 62.0, "container_blue"), _container(19.0, 36.0, "container_teal",
                                                                 along_y=False)]
    # The sealed container: a hollow shell with a chest inside, no way in.
    sx0, sy0, sx1, sy1 = 11.0, 58.0, 13.4, 64.0
    b += [box(sx0, sy0, 0.0, sx1, sy1, 0.12, "container", "container_sealed", walkable=True),
          box(sx0, sy0, 2.5, sx1, sy1, 2.6, "container", "container_sealed", name="sealed roof"),
          box(sx0, sy0, 0.12, sx0 + 0.1, sy1, 2.5, "container", "container_sealed"),
          box(sx1 - 0.1, sy0, 0.12, sx1, sy1, 2.5, "container", "container_sealed"),
          box(sx0 + 0.1, sy0, 0.12, sx1 - 0.1, sy0 + 0.1, 2.5, "container", "container_sealed"),
          box(sx0 + 0.1, sy1 - 0.1, 0.12, sx1 - 0.1, sy1, 2.5, "container", "container_sealed")]
    # Boathouse: a roof on posts.
    for lx, ly in ((8.0, 16.0), (13.7, 16.0), (8.0, 23.7), (13.7, 23.7)):
        b.append(box(lx, ly, 0.0, lx + 0.3, ly + 0.3, 3.0, "post", "post"))
    b.append(box(7.8, 15.8, 3.0, 14.2, 24.2, 3.2, "roof", "warehouse_wall", name="boathouse roof"))
    # Crane: four tall legs and a beam.
    for lx, ly in ((16.0, 76.0), (20.5, 76.0), (16.0, 79.5), (20.5, 79.5)):
        b.append(box(lx, ly, 0.0, lx + 0.5, ly + 0.5, 8.0, "post", "container_red"))
    b.append(box(16.0, 76.0, 8.0, 21.0, 80.0, 8.6, "beam", "container_red", name="crane beam"))
    for bx_, by_ in ((20.3, 52.6), (20.3, 53.5), (21.1, 53.05)):          # gold decoy barrels
        b.append(box(bx_, by_, 0.0, bx_ + 0.6, by_ + 0.6, 0.95, "decoy", "decoy_gold",
                     name="gold barrel"))

    blocked_for_trees = [hut, rocks, tower, (90.5, 51.0, 94.5, 52.5), (75.0, 63.5, 78.5, 67.5)]
    b += _trees(spawns, blocked_for_trees)
    return Arena3D(map_id=ARENA_3D_ID, boxes=b, spawns=spawns, start=(30.0, 50.0, 0.02),
                   start_yaw_deg=0.0, regions=dict(REGIONS))


def chest_boxes(spawn_x: float, spawn_y: float, z: float, yaw_deg: float, *,
                opened: bool) -> list[Box]:
    """Geometry of a chest (closed or opened), axis-aligned for yaw in {0, 90, 180, 270}."""
    quarter = round(yaw_deg / 90.0) % 4
    along_x = quarter in (0, 2)          # facing +-X: depth runs along X, width along Y
    hw = CHEST_WIDTH / 2
    hd = CHEST_DEPTH / 2
    ex, ey = (hd, hw) if along_x else (hw, hd)
    fx, fy = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}[quarter]   # front direction
    body_top = z + CHEST_BODY_HEIGHT
    out: list[Box] = []
    if not opened:
        out.append(box(spawn_x - ex, spawn_y - ey, z, spawn_x + ex, spawn_y + ey, body_top,
                       "chest", "chest_body"))
        out.append(box(spawn_x - ex, spawn_y - ey, body_top, spawn_x + ex, spawn_y + ey,
                       body_top + CHEST_LID_HEIGHT, "chest_lid", "chest_lid"))
        lx, ly = spawn_x + fx * (hd + 0.02), spawn_y + fy * (hd + 0.02)
        half = 0.08
        out.append(box(lx - (0.03 if along_x else half), ly - (half if along_x else 0.03),
                       z + 0.32, lx + (0.03 if along_x else half), ly + (half if along_x else 0.03),
                       z + 0.48, "chest_latch", "chest_latch"))
        return out
    # Opened: body with a bright inside, lid standing up at the back.
    out.append(box(spawn_x - ex, spawn_y - ey, z, spawn_x + ex, spawn_y + ey, body_top - 0.02,
                   "chest_open", "chest_open_body"))
    out.append(box(spawn_x - ex + 0.05, spawn_y - ey + 0.05, body_top - 0.02,
                   spawn_x + ex - 0.05, spawn_y + ey - 0.05, body_top, "chest_inside",
                   "chest_inside"))
    bx_, by_ = spawn_x - fx * (hd - 0.06), spawn_y - fy * (hd - 0.06)       # back edge
    if along_x:
        out.append(box(bx_ - 0.06, spawn_y - hw, body_top, bx_ + 0.06, spawn_y + hw,
                       body_top + 0.6, "chest_lid", "chest_lid"))
    else:
        out.append(box(spawn_x - hw, by_ - 0.06, body_top, spawn_x + hw, by_ + 0.06,
                       body_top + 0.6, "chest_lid", "chest_lid"))
    return out


def arena_to_json(arena: Arena3D) -> str:
    """Layout for the Unreal arena builder (metres; the builder converts to cm)."""
    from chest_hunter.simulation3d.geometry import COLOURS

    data = {
        "format": "chest-hunter-arena/1",
        "map_id": arena.map_id,
        "units": "metres",
        "axes": "X north, Y east, Z up (Unreal axes); yaw degrees from +X towards +Y",
        "bounds": list(arena.bounds),
        "start": {"position": list(arena.start), "yaw_deg": arena.start_yaw_deg},
        "chest": {"width": CHEST_WIDTH, "depth": CHEST_DEPTH, "body_height": CHEST_BODY_HEIGHT,
                  "lid_height": CHEST_LID_HEIGHT},
        "colours_bgr": {k: list(v) for k, v in COLOURS.items()},
        "regions": {k: list(v) for k, v in arena.regions.items()},
        "boxes": [b.to_dict() for b in arena.boxes],
        "spawn_points": [dict(asdict(s), location_name=s.location_name, tags=list(s.tags))
                         for s in arena.spawns],
    }
    if arena.appearance:
        data["appearance"] = dict(arena.appearance)
    return json.dumps(data, indent=1)
