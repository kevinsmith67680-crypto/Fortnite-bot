"""Procedural variations of the 3D research arena (Phase 5 generalisation tests).

Map ids look like ``proc_train_007`` or ``proc_test_002``. Training and test maps come from
disjoint seed ranges, so a policy trained on ``proc_train_*`` never sees a ``proc_test_*``
layout. Each map varies:

* building arrangement (count, size, wall colour, door sides and widths),
* raised plateaus reached by stairs (terrain elevation, chests on top),
* obstacles (containers, crate stacks, rocks, tree groups, hay bales, gold decoys),
* chest locations and their *true* spawn probabilities (hidden from the agent: the
  database receives an uninformative prior, see :func:`database_prior`),
* lighting (global brightness) and chest appearance (brightness and a small hue shift),
* the agent's start position and camera orientation.

Every generated map is checked with the navigation graph: the start must be on the graph
and at least 85 % of the spawn points reachable (the rest are kept and flagged
``reachable=False``, like the hand-made arena's two unreachable spawns).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import replace
from functools import lru_cache
from typing import Final

from chest_hunter.simulation3d.arena import (
    CHEST_DEPTH,
    CHEST_WIDTH,
    Arena3D,
    SpawnPoint3D,
    _building,
    _stairs,
    _tree,
    arena_to_json,
)
from chest_hunter.simulation3d.geometry import Box, World3D, box
from chest_hunter.simulation3d.navmesh import NavGrid3D

PROC_PREFIX: Final = "proc_"
SPLITS: Final = ("train", "test")
_ID = re.compile(r"^proc_(train|test)_(\d{3})$")
#: Seed ranges per split: disjoint, so test layouts can never leak into training.
SEED_BASE: Final = {"train": 1000, "test": 9000}
SIZE: Final = 90.0
MIN_REACHABLE: Final = 0.85
#: Spawn probability written to the database for procedural maps (the agent's prior). The
#: true per-spawn probabilities are drawn by the generator and are never stored there.
DATABASE_PRIOR: Final = 0.5


def map_id(split: str, index: int) -> str:
    if split not in SPLITS or not 0 <= index <= 999:
        raise ValueError(f"split must be one of {SPLITS} and index 0-999")
    return f"proc_{split}_{index:03d}"


def is_procedural(map_id_: str) -> bool:
    return bool(_ID.match(map_id_))


def parse_map_id(map_id_: str) -> tuple[str, int]:
    m = _ID.match(map_id_)
    if not m:
        raise ValueError(f"not a procedural map id: {map_id_!r} (expected proc_train_NNN or "
                         "proc_test_NNN)")
    return m.group(1), int(m.group(2))


def database_prior(_spawn: SpawnPoint3D) -> float:
    return DATABASE_PRIOR


def layout_hash(arena: Arena3D) -> str:
    """Stable fingerprint of a layout (geometry + spawn points), used to version
    learned statistics: estimates from another layout version are never mixed in."""
    data = json.loads(arena_to_json(arena))
    data.pop("colours_bgr", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


# -- generation -------------------------------------------------------------------------------


class _Footprints:
    """Axis-aligned footprints already used (with margins) for placement checks."""

    def __init__(self) -> None:
        self.rects: list[tuple[float, float, float, float]] = []

    def free(self, x0: float, y0: float, x1: float, y1: float, margin: float) -> bool:
        if x0 - margin < 2.0 or y0 - margin < 2.0 or x1 + margin > SIZE - 2.0 \
                or y1 + margin > SIZE - 2.0:
            return False
        return not any(x0 - margin < b1 and b0 < x1 + margin and y0 - margin < c1
                       and c0 < y1 + margin for b0, c0, b1, c1 in self.rects)

    def add(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.rects.append((x0, y0, x1, y1))


def _chest_rect(x: float, y: float, yaw: float) -> tuple[float, float, float, float]:
    along_x = round(yaw / 90.0) % 2 == 0
    hx, hy = (CHEST_DEPTH / 2, CHEST_WIDTH / 2) if along_x else (CHEST_WIDTH / 2, CHEST_DEPTH / 2)
    return x - hx, y - hy, x + hx, y + hy


def _generate(name: str, rng: random.Random) -> Arena3D:
    boxes: list[Box] = [box(0.0, 0.0, -0.2, SIZE, SIZE, 0.0, "ground", "grass")]
    boxes += [box(0.0, -0.4, 0.0, SIZE, 0.0, 1.6, "fence", "fence"),
              box(0.0, SIZE, 0.0, SIZE, SIZE + 0.4, 1.6, "fence", "fence"),
              box(-0.4, 0.0, 0.0, 0.0, SIZE, 1.6, "fence", "fence"),
              box(SIZE, 0.0, 0.0, SIZE + 0.4, SIZE, 1.6, "fence", "fence")]
    used = _Footprints()
    spawns: list[SpawnPoint3D] = []
    regions: dict[str, tuple[float, float, float, float]] = {}

    def spawn(region: str, label: str, x: float, y: float, z: float, yaw: float,
              tags: tuple[str, ...]) -> None:
        spawns.append(SpawnPoint3D(region, label, round(x, 2), round(y, 2), round(z, 2),
                                   yaw, round(rng.uniform(0.15, 0.95), 2), tags))

    # Buildings with one or two doors, chests in inner corners facing the room.
    for k in range(rng.randint(3, 5)):
        for _ in range(200):
            w, d = rng.uniform(8.0, 13.0), rng.uniform(8.0, 13.0)
            x0, y0 = rng.uniform(4.0, SIZE - 4.0 - w), rng.uniform(4.0, SIZE - 4.0 - d)
            if used.free(x0, y0, x0 + w, y0 + d, 5.0):
                break
        else:
            continue
        x1, y1 = x0 + w, y0 + d
        sides = rng.sample(["S", "N", "W", "E"], rng.choice((1, 2)))
        doors: dict[str, tuple[float, float]] = {}
        for side in sides:
            width = rng.uniform(1.3, 2.0)
            if side in ("S", "N"):
                doors[side] = (rng.uniform(y0 + 2.0, y1 - 2.0), width)
            else:
                doors[side] = (rng.uniform(x0 + 2.0, x1 - 2.0), width)
        colour = rng.choice(("house_wall", "warehouse_wall", "outpost_wall"))
        region = f"Building {k + 1}"
        boxes += _building(x0, y0, x1, y1, 3.2, colour, doors, name=region.lower())
        used.add(x0, y0, x1, y1)
        regions[region] = (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))
        corners = [(x0 + 1.0, y0 + 1.3, 0.0), (x1 - 1.0, y1 - 1.3, 180.0),
                   (x0 + 1.3, y1 - 1.0, 270.0), (x1 - 1.3, y0 + 1.0, 90.0)]
        rng.shuffle(corners)
        for j, (cx, cy, yaw) in enumerate(corners[:rng.randint(1, 2)]):
            spawn(region, f"Corner {j + 1}", cx, cy, 0.02, yaw,
                  ("inside", "doorway"))
        if rng.random() < 0.5:                                    # an interior crate stack
            cx, cy = (x0 + x1) / 2 + rng.uniform(-1.5, 1.5), (y0 + y1) / 2 + rng.uniform(-1.5, 1.5)
            boxes.append(box(cx - 0.6, cy - 0.6, 0.02, cx + 0.6, cy + 0.6, 1.2, "crate", "crate"))

    # Raised plateaus (terrain elevation) with stairs; a chest on top.
    for k in range(rng.randint(1, 2)):
        height = rng.choice((1.0, 1.5))
        steps = int(round(height / 0.25))
        run = 0.35
        for _ in range(200):
            w, d = rng.uniform(6.0, 9.0), rng.uniform(6.0, 9.0)
            x0 = rng.uniform(4.0 + steps * run, SIZE - 4.0 - w)
            y0 = rng.uniform(4.0, SIZE - 4.0 - d)
            if used.free(x0 - steps * run, y0, x0 + w, y0 + d, 4.0):
                break
        else:
            continue
        region = f"Plateau {k + 1}"
        colour = rng.choice(("concrete", "rock"))
        boxes.append(box(x0, y0, 0.0, x0 + w, y0 + d, height, "platform", colour,
                         walkable=True, name=region.lower()))
        sy = y0 + rng.uniform(1.5, d - 1.5)
        boxes += _stairs(x0 - steps * run, sy, "+X", steps, rise=0.25, run=run, width=1.6)
        used.add(x0 - steps * run, y0, x0 + w, y0 + d)
        regions[region] = (round(x0, 2), round(y0, 2), round(x0 + w, 2), round(y0 + d, 2))
        spawn(region, "Top", x0 + w - 1.2, y0 + d / 2 + rng.uniform(-1.5, 1.5), height, 180.0,
              ("upstairs", "open"))

    # Obstacles and outdoor chests.
    def free_spot(margin: float, w: float, d: float) -> tuple[float, float] | None:
        for _ in range(300):
            x, y = rng.uniform(4.0, SIZE - 4.0 - w), rng.uniform(4.0, SIZE - 4.0 - d)
            if used.free(x, y, x + w, y + d, margin):
                return x, y
        return None

    outdoor = 0
    for k in range(rng.randint(2, 3)):                            # containers, chest behind
        spot = free_spot(3.0, 6.0, 2.4)
        if spot is None:
            continue
        x, y = spot
        boxes.append(box(x, y, 0.0, x + 6.0, y + 2.4, 2.6, "container",
                         rng.choice(("container_red", "container_blue", "container_teal"))))
        used.add(x, y, x + 6.0, y + 2.4)
        outdoor += 1
        spawn("Yard", f"Behind Container {k + 1}", x + 3.0, y + 2.4 + 0.9, 0.0, 90.0,
              ("behind_obstacle",))
        used.add(x + 2.4, y + 2.4, x + 3.6, y + 3.6)
    for k in range(rng.randint(3, 5)):                            # crate stacks
        spot = free_spot(2.5, 1.2, 1.2)
        if spot is None:
            continue
        x, y = spot
        boxes.append(box(x, y, 0.0, x + 1.2, y + 1.2, rng.choice((1.0, 1.2, 1.8)), "crate",
                         "crate"))
        used.add(x, y, x + 1.2, y + 1.2)
        if rng.random() < 0.5:
            spawn("Yard", f"Crates {k + 1}", x - 0.9, y + 0.6, 0.0, 180.0, ("behind_obstacle",))
            used.add(x - 1.5, y, x, y + 1.2)
    for _k in range(rng.randint(2, 4)):                           # rocks and hay bales
        spot = free_spot(2.0, 1.6, 1.2)
        if spot is None:
            continue
        x, y = spot
        hay = rng.random() < 0.5
        boxes.append(box(x, y, 0.0, x + 1.6, y + 1.2, 1.0 if hay else 1.4, "decoy" if hay else
                         "rock", "decoy_hay" if hay else "rock"))
        used.add(x, y, x + 1.6, y + 1.2)
    groves = 0
    for _ in range(rng.randint(2, 3)):                            # tree groups
        spot = free_spot(1.0, 10.0, 10.0)
        if spot is None:
            continue
        gx, gy = spot
        groves += 1
        placed: list[tuple[float, float]] = []
        for _t in range(rng.randint(4, 8)):
            tx, ty = gx + rng.uniform(1.0, 9.0), gy + rng.uniform(1.0, 9.0)
            if all((tx - px) ** 2 + (ty - py) ** 2 > 3.5 ** 2 for px, py in placed):
                placed.append((tx, ty))
                boxes += _tree(tx, ty)
        used.add(gx, gy, gx + 10.0, gy + 10.0)
        spawn(f"Grove {groves}", "Clearing", gx + 5.0 + rng.uniform(-1.0, 1.0),
              gy + 5.0 + rng.uniform(-1.0, 1.0), 0.0, rng.choice((0.0, 90.0, 180.0, 270.0)),
              ("partially_obscured",))
    for k in range(rng.randint(4, 7)):                            # open ground
        spot = free_spot(2.0, 1.0, 1.0)
        if spot is None:
            continue
        x, y = spot
        used.add(x, y, x + 1.0, y + 1.0)
        outdoor += 1
        spawn("Field", f"Open {k + 1}", x + 0.5, y + 0.5, 0.0,
              rng.choice((0.0, 90.0, 180.0, 270.0)), ("open",))
        if rng.random() < 0.3:                                    # gold decoy nearby
            dx = rng.choice((-2.5, 2.5))
            boxes.append(box(x + dx, y, 0.0, x + dx + 0.8, y + 0.5, 0.4, "decoy", "decoy_gold"))
            spawns[-1] = replace(spawns[-1], tags=("open", "decoys"))

    # Start: a free spot with a random view direction.
    start_spot = free_spot(1.5, 1.0, 1.0) or (SIZE / 2, SIZE / 2)
    start = (round(start_spot[0] + 0.5, 2), round(start_spot[1] + 0.5, 2), 0.02)
    yaw = float(rng.choice(range(0, 360, 15)))
    appearance = {"lighting": round(rng.uniform(0.85, 1.12), 3),
                  "chest_value": round(rng.uniform(0.9, 1.08), 3),
                  "chest_hue_shift": round(rng.uniform(-3.0, 3.0), 2)}
    return Arena3D(map_id=name, boxes=boxes, spawns=spawns, start=start, start_yaw_deg=yaw,
                   bounds=(0.0, 0.0, SIZE, SIZE), regions=regions, appearance=appearance)


def _validate(arena: Arena3D) -> tuple[Arena3D, World3D, NavGrid3D] | None:
    world = World3D(arena.boxes, bounds=arena.bounds)
    nav = NavGrid3D(world)
    if nav.project(arena.start) is None:
        return None
    reachable = []
    for s in arena.spawns:
        path = nav.find_path(arena.start, (s.x, s.y, s.z), goal_tolerance=1.2)
        reachable.append(path.found)
    if len(arena.spawns) < 15 or sum(reachable) < MIN_REACHABLE * len(arena.spawns):
        return None
    spawns = [replace(s, reachable=ok, tags=s.tags if ok else (*s.tags, "unreachable"))
              for s, ok in zip(arena.spawns, reachable, strict=True)]
    arena = replace(arena, spawns=spawns)
    return arena, world, nav


@lru_cache(maxsize=24)
def build_procedural(map_id_: str) -> tuple[Arena3D, World3D, NavGrid3D]:
    """Deterministic layout (and its navigation graph) for a procedural map id."""
    split, index = parse_map_id(map_id_)
    rng = random.Random(f"proc3d:{split}:{SEED_BASE[split] + index}")
    for _attempt in range(40):
        result = _validate(_generate(map_id_, rng))
        if result is not None:
            return result
    raise RuntimeError(f"could not generate a valid layout for {map_id_}")


def build_procedural_arena(map_id_: str) -> Arena3D:
    return build_procedural(map_id_)[0]
