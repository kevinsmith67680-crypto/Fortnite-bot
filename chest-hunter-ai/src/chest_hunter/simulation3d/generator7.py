"""Phase 7 environment generator: four families of unseen 3D research environments.

Map ids: ``p7_<category>_<split>_<NNN>`` with an optional shift variant suffix
``p7_<category>_<split>_<NNN>_<variant>``:

* categories - ``open`` (few obstacles, broad visibility), ``indoor`` (a building complex of
  rooms, corridors and doors, limited visibility), ``mixed`` (buildings with interior rooms
  and outdoor areas), ``complex`` (several elevations connected by stairs, an upper floor,
  narrow passages, heavy occlusion);
* splits - ``train`` and ``test`` use disjoint seed ranges (:data:`SEED_BASE`). Held-out
  layouts are never generated from a training seed and vice versa;
  :func:`assert_training_map` refuses test maps and every variant map (variants are
  evaluation-only distribution shifts) wherever data is collected for training;
* variants (distribution shifts used by the generalisation benchmark, applied on top of a
  base layout): ``obstacles`` (new obstacle arrangement, same buildings), ``light`` (lighting
  outside the training range), ``appearance`` (chest colour/brightness and surface textures
  outside the training range), ``start`` (another start position and camera direction),
  ``elevation`` (extra raised levels with chests), ``combined`` (all of them).

Everything is deterministic for a map id: separate seeded streams for the layout, the
obstacles, the appearance and the start pose. Every map is validated with its navigation
graph: the start must be on the graph, the walkable region reachable from it must be large
enough, and the reachable share of the spawn points must meet the target (the spawn points
built to be unreachable - inside sealed rooms - are flagged ``reachable=False``).

:data:`GENERATOR_VERSION` and the generation settings are stored in ``Arena3D.meta``;
learned statistics are partitioned by layout hash, so a change of generator never mixes data.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Final

from chest_hunter.simulation3d.arena import (
    Arena3D,
    SpawnPoint3D,
    _building,
    _stairs,
    _tree,
    _wall_x,
    _wall_y,
)
from chest_hunter.simulation3d.geometry import Box, World3D, box
from chest_hunter.simulation3d.navmesh import NavGrid3D

GENERATOR_VERSION: Final = "7.0"
CATEGORIES: Final = ("open", "indoor", "mixed", "complex")
SPLITS: Final = ("train", "test")
VARIANTS: Final = ("obstacles", "light", "appearance", "start", "elevation", "combined")
SEED_BASE: Final = {"train": 20_000, "test": 70_000}
_ID = re.compile(r"^p7_(open|indoor|mixed|complex)_(train|test)_(\d{3})"
                 r"(?:_(obstacles|light|appearance|start|elevation|combined))?$")

#: Appearance ranges of the training distribution; the shift variants leave them.
TRAIN_LIGHTING: Final = (0.85, 1.12)
TRAIN_CHEST_VALUE: Final = (0.9, 1.08)
TRAIN_CHEST_HUE: Final = (-3.0, 3.0)
TRAIN_SURFACE_VALUE: Final = (0.92, 1.08)


@dataclass(frozen=True, slots=True)
class GeneratorSettings:
    size: float
    complexity: float = 0.5          # 0..1: obstacle, room and level counts
    chest_density: float = 1.0       # multiplier on the number of spawn points
    unreachable_fraction: float = 0.1
    min_reachable: float = 0.8       # share of spawn points (excluding the sealed ones)


DEFAULTS: Final[dict[str, GeneratorSettings]] = {
    "open": GeneratorSettings(size=90.0, complexity=0.3),
    "indoor": GeneratorSettings(size=64.0, complexity=0.6),
    "mixed": GeneratorSettings(size=90.0, complexity=0.5),
    "complex": GeneratorSettings(size=90.0, complexity=0.8),
}


@dataclass(frozen=True, slots=True)
class MapSpec:
    category: str
    split: str
    index: int
    variant: str | None

    @property
    def base_id(self) -> str:
        return f"p7_{self.category}_{self.split}_{self.index:03d}"

    @property
    def map_id(self) -> str:
        return self.base_id + (f"_{self.variant}" if self.variant else "")

    @property
    def seed(self) -> int:
        return SEED_BASE[self.split] + 1000 * CATEGORIES.index(self.category) + self.index


def map_id(category: str, split: str, index: int, variant: str | None = None) -> str:
    if category not in CATEGORIES or split not in SPLITS or not 0 <= index <= 999 \
            or (variant is not None and variant not in VARIANTS):
        raise ValueError(f"category in {CATEGORIES}, split in {SPLITS}, index 0-999, "
                         f"variant in {VARIANTS} or None")
    return MapSpec(category, split, index, variant).map_id


def is_generated(map_id_: str) -> bool:
    return bool(_ID.match(map_id_))


def parse(map_id_: str) -> MapSpec:
    m = _ID.match(map_id_)
    if not m:
        raise ValueError(f"not a Phase 7 map id: {map_id_!r} (expected "
                         "p7_<open|indoor|mixed|complex>_<train|test>_NNN[_<variant>])")
    return MapSpec(m.group(1), m.group(2), int(m.group(3)), m.group(4))


def is_held_out(map_id_: str) -> bool:
    """Test layouts and every shift variant (also of a training layout) are evaluation-only."""
    if not is_generated(map_id_):
        return False
    spec = parse(map_id_)
    return spec.split == "test" or spec.variant is not None


def assert_training_map(map_id_: str) -> None:
    """Training data (demonstrations, persistent statistics) never comes from held-out maps."""
    if is_held_out(map_id_) or map_id_.startswith("proc_test_"):
        raise ValueError(f"{map_id_} is a held-out evaluation map; it must not be used for "
                         "training or for updating persistent statistics")


def settings_hash(settings: GeneratorSettings) -> str:
    return hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()
                          ).hexdigest()[:12]


# -- geometry helpers ---------------------------------------------------------------------------


class _Space:
    """Footprints already used, with margins, inside the map boundary."""

    def __init__(self, size: float) -> None:
        self.size = size
        self.rects: list[tuple[float, float, float, float]] = []

    def free(self, x0: float, y0: float, x1: float, y1: float, margin: float) -> bool:
        if x0 - margin < 1.5 or y0 - margin < 1.5 or x1 + margin > self.size - 1.5 \
                or y1 + margin > self.size - 1.5:
            return False
        return not any(x0 - margin < b1 and b0 < x1 + margin and y0 - margin < c1
                       and c0 < y1 + margin for b0, c0, b1, c1 in self.rects)

    def add(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.rects.append((x0, y0, x1, y1))

    def spot(self, rng: random.Random, w: float, d: float, margin: float,
             tries: int = 300) -> tuple[float, float] | None:
        for _ in range(tries):
            x = rng.uniform(2.0, self.size - 2.0 - w)
            y = rng.uniform(2.0, self.size - 2.0 - d)
            if self.free(x, y, x + w, y + d, margin):
                return x, y
        return None


class _Builder:
    def __init__(self, spec: MapSpec, settings: GeneratorSettings) -> None:
        self.spec = spec
        self.s = settings
        size = settings.size
        self.size = size
        self.boxes: list[Box] = [box(0.0, 0.0, -0.2, size, size, 0.0, "ground", "grass")]
        self.boxes += [box(0.0, -0.4, 0.0, size, 0.0, 1.6, "fence", "fence"),
                       box(0.0, size, 0.0, size, size + 0.4, 1.6, "fence", "fence"),
                       box(-0.4, 0.0, 0.0, 0.0, size, 1.6, "fence", "fence"),
                       box(size, 0.0, 0.0, size + 0.4, size, 1.6, "fence", "fence")]
        self.space = _Space(size)
        self.spawns: list[SpawnPoint3D] = []
        self.regions: dict[str, tuple[float, float, float, float]] = {}
        self.rooms: list[tuple[float, float, float, float]] = []   # interior floor areas
        self.chest_rng = random.Random(f"p7:{spec.base_id}:chests")

    def count(self, low: int, high: int) -> int:
        return int(round(low + (high - low) * self.s.complexity))

    def spawn(self, region: str, label: str, x: float, y: float, z: float, yaw: float,
              tags: tuple[str, ...], *, reachable: bool = True) -> None:
        self.spawns.append(SpawnPoint3D(region, label, round(x, 2), round(y, 2), round(z, 2),
                                        yaw, round(self.chest_rng.uniform(0.2, 0.95), 2), tags,
                                        reachable))

    # -- structures ----------------------------------------------------------------------------

    def building(self, rng: random.Random, name: str, w: float, d: float, *,
                 rooms: tuple[int, int] = (1, 1), doors: int = 1, margin: float = 4.0,
                 colour: str | None = None, at: tuple[float, float] | None = None
                 ) -> tuple[float, float, float, float] | None:
        """A building of ``rooms`` (nx x ny) rooms separated by interior walls with doors."""
        if at is None:
            spot = self.space.spot(rng, w, d, margin)
            if spot is None:
                return None
            x0, y0 = spot
        else:
            x0, y0 = at
        x1, y1 = x0 + w, y0 + d
        sides = rng.sample(["S", "N", "W", "E"], max(1, min(4, doors)))
        outer: dict[str, tuple[float, float]] = {}
        for side in sides:
            width = rng.uniform(1.4, 2.0)
            if side in ("S", "N"):
                outer[side] = (rng.uniform(y0 + 1.8, y1 - 1.8), width)
            else:
                outer[side] = (rng.uniform(x0 + 1.8, x1 - 1.8), width)
        colour = colour or rng.choice(("house_wall", "warehouse_wall", "outpost_wall"))
        self.boxes += _building(x0, y0, x1, y1, 3.2, colour, outer, name=name.lower())
        nx, ny = rooms
        # interior partitions with one door each (every room stays connected)
        for i in range(1, nx):
            px = x0 + w * i / nx
            gaps = []
            for j in range(ny):
                ya, yb = y0 + d * j / ny, y0 + d * (j + 1) / ny
                c = rng.uniform(ya + 1.4, yb - 1.4)
                gaps.append((c - 0.75, c + 0.75))
            self.boxes += _wall_y(y0 + 0.3, y1 - 0.3, px - 0.1, 3.2, colour, thick=0.2,
                                  gaps=tuple(gaps), kind="wall")
        for j in range(1, ny):
            py = y0 + d * j / ny
            gaps = []
            for i in range(nx):
                xa, xb = x0 + w * i / nx, x0 + w * (i + 1) / nx
                c = rng.uniform(xa + 1.4, xb - 1.4)
                gaps.append((c - 0.75, c + 0.75))
            self.boxes += _wall_x(x0 + 0.3, x1 - 0.3, py - 0.1, 3.2, colour, thick=0.2,
                                  gaps=tuple(gaps), kind="wall")
        self.space.add(x0, y0, x1, y1)
        self.regions[name] = (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))
        for i in range(nx):
            for j in range(ny):
                rx0, rx1 = x0 + w * i / nx, x0 + w * (i + 1) / nx
                ry0, ry1 = y0 + d * j / ny, y0 + d * (j + 1) / ny
                self.rooms.append((rx0, ry0, rx1, ry1))
        return x0, y0, x1, y1

    def room_chests(self, rng: random.Random, name: str, room: tuple[float, float, float, float],
                    k: int, z: float = 0.02) -> None:
        x0, y0, x1, y1 = room
        corners = [(x0 + 1.0, y0 + 1.3, 0.0), (x1 - 1.0, y1 - 1.3, 180.0),
                   (x0 + 1.3, y1 - 1.0, 270.0), (x1 - 1.3, y0 + 1.0, 90.0)]
        rng.shuffle(corners)
        for n, (cx, cy, yaw) in enumerate(corners[:k]):
            self.spawn(name, f"Corner {n + 1}", cx, cy, z, yaw, ("inside", "room"))

    def sealed_room(self, rng: random.Random, name: str) -> None:
        """A small room without doors holding a chest that cannot be reached."""
        spot = self.space.spot(rng, 4.0, 4.0, 3.0)
        if spot is None:
            return
        x0, y0 = spot
        self.boxes += _building(x0, y0, x0 + 4.0, y0 + 4.0, 3.2, "outpost_wall", {},
                                name=name.lower())
        self.space.add(x0, y0, x0 + 4.0, y0 + 4.0)
        self.regions[name] = (round(x0, 2), round(y0, 2), round(x0 + 4.0, 2), round(y0 + 4.0, 2))
        self.spawn(name, "Sealed", x0 + 2.0, y0 + 2.0, 0.02, 0.0, ("inside", "unreachable"),
                   reachable=False)

    def plateau(self, rng: random.Random, name: str, height: float, *, chest: bool = True
                ) -> tuple[float, float, float, float] | None:
        steps = int(round(height / 0.25))
        run = 0.35
        for _ in range(200):
            w, d = rng.uniform(6.0, 9.0), rng.uniform(6.0, 9.0)
            x0 = rng.uniform(3.0 + steps * run, self.size - 3.0 - w)
            y0 = rng.uniform(3.0, self.size - 3.0 - d)
            if self.space.free(x0 - steps * run, y0, x0 + w, y0 + d, 3.0):
                break
        else:
            return None
        colour = rng.choice(("concrete", "rock"))
        self.boxes.append(box(x0, y0, 0.0, x0 + w, y0 + d, height, "platform", colour,
                              walkable=True, name=name.lower()))
        sy = y0 + rng.uniform(1.5, d - 1.5)
        self.boxes += _stairs(x0 - steps * run, sy, "+X", steps, rise=0.25, run=run, width=1.6)
        self.space.add(x0 - steps * run, y0, x0 + w, y0 + d)
        self.regions[name] = (round(x0, 2), round(y0, 2), round(x0 + w, 2), round(y0 + d, 2))
        if chest:
            self.spawn(name, "Top", x0 + w - 1.2, y0 + d / 2 + rng.uniform(-1.5, 1.5), height,
                       180.0, ("upstairs", "open"))
        return x0, y0, x0 + w, y0 + d

    def upper_floor(self, rng: random.Random, name: str, footprint: tuple[float, float, float,
                                                                         float]) -> None:
        """A mezzanine at 2.5 m along one side of a building, reached by stairs."""
        x0, y0, x1, y1 = footprint
        depth = min(4.0, (y1 - y0) / 3)
        self.boxes.append(box(x0 + 0.3, y1 - 0.3 - depth, 2.3, x1 - 0.3, y1 - 0.3, 2.5, "floor",
                              "interior_floor", name=f"{name.lower()} upper floor"))
        sx = x0 + rng.uniform(1.5, max(1.6, (x1 - x0) / 2 - 3.0))
        self.boxes += _stairs(sx, y1 - 0.3 - depth - 0.9, "+X", 10, rise=0.25, run=0.35,
                              width=1.4)
        self.spawn(name, "Upper Floor", x1 - 1.4, y1 - 0.3 - depth / 2, 2.5, 180.0,
                   ("inside", "upstairs"))

    def outdoor(self, rng: random.Random, *, containers: int, crates: int, rocks: int,
                groves: int, field: int) -> None:
        for k in range(containers):
            spot = self.space.spot(rng, 6.0, 2.4, 3.0)
            if spot is None:
                continue
            x, y = spot
            self.boxes.append(box(x, y, 0.0, x + 6.0, y + 2.4, 2.6, "container",
                                  rng.choice(("container_red", "container_blue",
                                              "container_teal"))))
            self.space.add(x, y, x + 6.0, y + 2.4)
            self.spawn("Yard", f"Behind Container {k + 1}", x + 3.0, y + 3.3, 0.0, 90.0,
                       ("behind_obstacle",))
            self.space.add(x + 2.4, y + 2.4, x + 3.6, y + 3.6)
        for k in range(crates):
            spot = self.space.spot(rng, 1.2, 1.2, 2.5)
            if spot is None:
                continue
            x, y = spot
            self.boxes.append(box(x, y, 0.0, x + 1.2, y + 1.2, rng.choice((1.0, 1.2, 1.8)),
                                  "crate", "crate"))
            self.space.add(x, y, x + 1.2, y + 1.2)
            if rng.random() < 0.5:
                self.spawn("Yard", f"Crates {k + 1}", x - 0.9, y + 0.6, 0.0, 180.0,
                           ("behind_obstacle",))
                self.space.add(x - 1.5, y, x, y + 1.2)
        for _k in range(rocks):
            spot = self.space.spot(rng, 1.6, 1.2, 2.0)
            if spot is None:
                continue
            x, y = spot
            hay = rng.random() < 0.4
            self.boxes.append(box(x, y, 0.0, x + 1.6, y + 1.2, 1.0 if hay else 1.4,
                                  "decoy" if hay else "rock", "decoy_hay" if hay else "rock"))
            self.space.add(x, y, x + 1.6, y + 1.2)
        made = 0
        for _ in range(groves):
            spot = self.space.spot(rng, 10.0, 10.0, 1.0)
            if spot is None:
                continue
            gx, gy = spot
            made += 1
            placed: list[tuple[float, float]] = []
            for _t in range(rng.randint(4, 8)):
                tx, ty = gx + rng.uniform(1.0, 9.0), gy + rng.uniform(1.0, 9.0)
                if all((tx - px) ** 2 + (ty - py) ** 2 > 3.5 ** 2 for px, py in placed):
                    placed.append((tx, ty))
                    self.boxes += _tree(tx, ty)
            self.space.add(gx, gy, gx + 10.0, gy + 10.0)
            self.spawn(f"Grove {made}", "Clearing", gx + 5.0 + rng.uniform(-1.0, 1.0),
                       gy + 5.0 + rng.uniform(-1.0, 1.0), 0.0,
                       rng.choice((0.0, 90.0, 180.0, 270.0)), ("partially_obscured",))
        for k in range(field):
            spot = self.space.spot(rng, 1.0, 1.0, 2.0)
            if spot is None:
                continue
            x, y = spot
            self.space.add(x, y, x + 1.0, y + 1.0)
            self.spawn("Field", f"Open {k + 1}", x + 0.5, y + 0.5, 0.0,
                       rng.choice((0.0, 90.0, 180.0, 270.0)), ("open",))
            if rng.random() < 0.3:
                dx = rng.choice((-2.5, 2.5))
                self.boxes.append(box(x + dx, y, 0.0, x + dx + 0.8, y + 0.5, 0.4, "decoy",
                                      "decoy_gold"))

    def alleys(self, rng: random.Random, n: int) -> None:
        """Constrained routes: free-standing wall segments forming narrow passages."""
        for k in range(n):
            spot = self.space.spot(rng, 12.0, 4.0, 2.0)
            if spot is None:
                continue
            x, y = spot
            self.boxes += _wall_x(x, x + 12.0, y, 2.8, "outpost_wall", thick=0.3)
            self.boxes += _wall_x(x, x + 12.0, y + 2.4, 2.8, "outpost_wall", thick=0.3)
            self.space.add(x, y, x + 12.0, y + 2.7)
            if rng.random() < 0.6:
                self.spawn(f"Alley {k + 1}", "Passage", x + rng.uniform(3.0, 9.0), y + 1.35,
                           0.0, rng.choice((90.0, 270.0)), ("narrow", "partially_obscured"))


# -- categories ---------------------------------------------------------------------------------


def _open(b: _Builder, layout: random.Random, obstacles: random.Random) -> None:
    if layout.random() < 0.6:                                   # one small shelter
        fp = b.building(layout, "Shelter", layout.uniform(6.0, 8.0), layout.uniform(6.0, 8.0),
                        doors=2)
        if fp is not None:
            b.room_chests(layout, "Shelter", b.rooms[-1], 1)
    b.outdoor(obstacles, containers=b.count(0, 2), crates=b.count(2, 4), rocks=b.count(3, 7),
              groves=b.count(1, 3), field=int(round(b.count(7, 10) * b.s.chest_density)))


def _indoor(b: _Builder, layout: random.Random, obstacles: random.Random) -> None:
    size = b.size
    margin = 4.0
    w, d = size - 2 * margin, size - 2 * margin
    nx, ny = (3, 3) if b.s.complexity < 0.7 else (4, 3)
    b.building(layout, "Complex", w, d, rooms=(nx, ny), doors=3, margin=0.0,
               at=(margin, margin), colour="warehouse_wall")
    for k, room in enumerate(b.rooms):
        if obstacles.random() < 0.55:                             # furniture: crates
            cx = obstacles.uniform(room[0] + 2.5, room[2] - 2.5)
            cy = obstacles.uniform(room[1] + 2.5, room[3] - 2.5)
            b.boxes.append(box(cx - 0.6, cy - 0.6, 0.02, cx + 0.6, cy + 0.6,
                               obstacles.choice((1.0, 1.4)), "crate", "crate"))
        n = 1 if layout.random() < 0.6 * b.s.chest_density else 2
        b.room_chests(layout, f"Room {k + 1}", room, n)
    b.space.add(margin, margin, margin + w, margin + d)


def _mixed(b: _Builder, layout: random.Random, obstacles: random.Random) -> None:
    for k in range(b.count(3, 5)):
        rooms = layout.choice(((1, 1), (2, 1), (1, 2), (2, 2)))
        w = layout.uniform(9.0, 14.0) * (1.4 if rooms[0] == 2 else 1.0)
        d = layout.uniform(9.0, 14.0) * (1.4 if rooms[1] == 2 else 1.0)
        start = len(b.rooms)
        fp = b.building(layout, f"Building {k + 1}", w, d, rooms=rooms,
                        doors=layout.choice((1, 2)))
        if fp is None:
            continue
        for r, room in enumerate(b.rooms[start:]):
            b.room_chests(layout, f"Building {k + 1} Room {r + 1}", room, 1)
    if layout.random() < 0.7:
        b.plateau(layout, "Plateau 1", layout.choice((1.0, 1.5)))
    b.outdoor(obstacles, containers=b.count(1, 3), crates=b.count(2, 5), rocks=b.count(2, 4),
              groves=b.count(1, 3), field=int(round(b.count(3, 6) * b.s.chest_density)))


def _complex(b: _Builder, layout: random.Random, obstacles: random.Random) -> None:
    for k in range(b.count(2, 4)):
        rooms = layout.choice(((2, 1), (1, 2), (2, 2)))
        w = layout.uniform(13.0, 18.0)
        d = layout.uniform(13.0, 18.0)
        start = len(b.rooms)
        fp = b.building(layout, f"Building {k + 1}", w, d, rooms=rooms,
                        doors=layout.choice((1, 2)))
        if fp is None:
            continue
        for r, room in enumerate(b.rooms[start:]):
            b.room_chests(layout, f"Building {k + 1} Room {r + 1}", room, 1)
        if k == 0:
            b.upper_floor(layout, f"Building {k + 1}", fp)
    for k in range(b.count(2, 3)):
        b.plateau(layout, f"Level {k + 1}", (1.0, 1.5, 2.25)[k % 3])
    b.alleys(obstacles, b.count(1, 3))
    b.outdoor(obstacles, containers=b.count(2, 4), crates=b.count(3, 6), rocks=b.count(2, 4),
              groves=b.count(1, 2), field=int(round(b.count(2, 4) * b.s.chest_density)))


BUILDERS: Final = {"open": _open, "indoor": _indoor, "mixed": _mixed, "complex": _complex}


# -- generation ------------------------------------------------------------------------------------


def _appearance(rng: random.Random, variant: str | None) -> dict[str, float]:
    look = {"lighting": rng.uniform(*TRAIN_LIGHTING),
            "chest_value": rng.uniform(*TRAIN_CHEST_VALUE),
            "chest_hue_shift": rng.uniform(*TRAIN_CHEST_HUE),
            "surface_value": rng.uniform(*TRAIN_SURFACE_VALUE)}
    if variant in ("light", "combined"):
        look["lighting"] = rng.choice((rng.uniform(0.62, 0.72), rng.uniform(1.22, 1.32)))
    if variant in ("appearance", "combined"):
        look["chest_value"] = rng.choice((rng.uniform(0.78, 0.84), rng.uniform(1.14, 1.2)))
        look["chest_hue_shift"] = rng.choice((-1.0, 1.0)) * rng.uniform(6.0, 9.0)
        look["surface_value"] = rng.choice((rng.uniform(0.75, 0.82), rng.uniform(1.18, 1.25)))
    return {k: round(v, 3) for k, v in look.items()}


def _generate(spec: MapSpec, settings: GeneratorSettings, attempt: int) -> Arena3D:
    variant = spec.variant
    layout = random.Random(f"p7:{spec.base_id}:layout:{attempt}")
    obstacle_key = "obstacles2" if variant in ("obstacles", "combined") else "obstacles"
    obstacles = random.Random(f"p7:{spec.base_id}:{obstacle_key}:{attempt}")
    b = _Builder(spec, settings)
    BUILDERS[spec.category](b, layout, obstacles)
    if variant in ("elevation", "combined"):
        extra = random.Random(f"p7:{spec.base_id}:elevation:{attempt}")
        for k in range(2):
            b.plateau(extra, f"Extra Level {k + 1}", (1.5, 2.25)[k])
    if settings.unreachable_fraction > 0 and layout.random() < min(1.0, 4 * settings
                                                                   .unreachable_fraction):
        b.sealed_room(layout, "Sealed Room")
    start_rng = random.Random(f"p7:{spec.base_id}:start" + (":shift" if variant in
                                                          ("start", "combined") else ""))
    spot = b.space.spot(start_rng, 1.0, 1.0, 1.5)
    if spot is None and b.rooms:
        r = b.rooms[0]
        spot = ((r[0] + r[2]) / 2 - 0.5, (r[1] + r[3]) / 2 - 0.5)
    if spot is None:
        spot = (b.size / 2, b.size / 2)
    if spec.category == "indoor" and b.rooms:                     # start inside the complex
        r = b.rooms[start_rng.randrange(len(b.rooms))]
        spot = ((r[0] + r[2]) / 2 - 0.5 + start_rng.uniform(-1.0, 1.0),
                (r[1] + r[3]) / 2 - 0.5 + start_rng.uniform(-1.0, 1.0))
    start = (round(spot[0] + 0.5, 2), round(spot[1] + 0.5, 2), 0.02)
    yaw = float(start_rng.choice(range(0, 360, 15)))
    look = _appearance(random.Random(f"p7:{spec.base_id}:appearance"
                                     + (f":{variant}" if variant else "")), variant)
    meta = {"generator_version": GENERATOR_VERSION, "category": spec.category,
            "split": spec.split, "index": spec.index, "variant": variant or "",
            "settings": asdict(settings), "settings_hash": settings_hash(settings),
            "rooms": len(b.rooms), "attempt": attempt}
    return Arena3D(map_id=spec.map_id, boxes=b.boxes, spawns=b.spawns, start=start,
                   start_yaw_deg=yaw, bounds=(0.0, 0.0, b.size, b.size), regions=b.regions,
                   appearance=look, meta=meta)


def validate(arena: Arena3D, settings: GeneratorSettings, *, min_spawns: int = 8,
             min_region_fraction: float = 0.15) -> tuple[Arena3D, World3D, NavGrid3D] | None:
    """Navigation checks; returns the arena with ``reachable`` flags or ``None``."""
    world = World3D(arena.boxes, bounds=arena.bounds)
    nav = NavGrid3D(world)
    if nav.project(arena.start) is None:
        return None
    region = nav.reachable_area(arena.start)
    total = (arena.bounds[2] - arena.bounds[0]) * (arena.bounds[3] - arena.bounds[1])
    if region < min_region_fraction * total:
        return None
    flags = []
    for s in arena.spawns:
        ok = nav.find_path(arena.start, (s.x, s.y, s.z), goal_tolerance=1.2).found
        flags.append(ok)
    intended = [f for f, s in zip(flags, arena.spawns, strict=True) if s.reachable]
    if len(arena.spawns) < min_spawns or not intended \
            or sum(intended) < settings.min_reachable * len(intended):
        return None
    spawns = [replace(s, reachable=ok, tags=s.tags if ok or "unreachable" in s.tags
                      else (*s.tags, "unreachable"))
              for s, ok in zip(arena.spawns, flags, strict=True)]
    arena = replace(arena, spawns=spawns)
    arena.meta = dict(arena.meta, reachable_area_m2=round(region, 1),
                      reachable_spawns=sum(flags))
    return arena, world, nav


@lru_cache(maxsize=32)
def build(map_id_: str, settings: GeneratorSettings | None = None
          ) -> tuple[Arena3D, World3D, NavGrid3D]:
    """Deterministic environment (and navigation graph) for a Phase 7 map id."""
    spec = parse(map_id_)
    settings = settings or DEFAULTS[spec.category]
    for attempt in range(30):
        result = validate(_generate(spec, settings, attempt), settings)
        if result is not None:
            return result
    raise RuntimeError(f"could not generate a valid layout for {map_id_}")


def build_arena(map_id_: str) -> Arena3D:
    return build(map_id_)[0]


def map_ids(category: str, split: str, count: int, start: int = 0,
            variant: str | None = None) -> list[str]:
    return [map_id(category, split, i, variant) for i in range(start, start + count)]
