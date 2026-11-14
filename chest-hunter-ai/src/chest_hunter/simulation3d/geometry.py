"""Axis-aligned box world: collision, ground height and ray casts (metres, Z up).

The research arena is built from axis-aligned boxes, as the Unreal version
is built from scaled cube static meshes. Every box can block movement, can
offer a walkable top surface, and is drawn by the reference renderer.

Queries use a 2D spatial hash, so collision cost does not grow with the size
of the map.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

# BGR colours of the arena palette (shared with the Unreal layout export).
COLOURS: Final[dict[str, tuple[int, int, int]]] = {
    "sky": (235, 206, 150),
    "grass": (70, 150, 80),
    "concrete": (150, 150, 152),
    "asphalt": (95, 95, 100),
    "warehouse_wall": (140, 120, 104),
    "house_wall": (196, 196, 204),
    "outpost_wall": (112, 116, 122),
    "interior_floor": (150, 138, 128),
    "ceiling": (205, 205, 210),
    "stairs": (160, 160, 166),
    "railing": (62, 62, 66),
    "crate": (72, 104, 82),
    "container_red": (48, 48, 176),
    "container_blue": (164, 92, 46),
    "container_teal": (128, 134, 44),
    "container_sealed": (90, 90, 150),
    "water": (170, 112, 40),
    "pier": (110, 120, 128),
    "trunk": (62, 70, 78),
    "canopy": (44, 112, 52),
    "rock": (126, 126, 132),
    "sandbag": (132, 158, 170),
    "tent": (70, 112, 100),
    "fence": (120, 124, 128),
    "post": (80, 80, 84),
    "decoy_gold": (24, 197, 245),       # same gold as chests: false-positive bait
    "decoy_hay": (70, 175, 215),
    "chest_body": (24, 197, 245),
    "chest_lid": (20, 72, 120),
    "chest_latch": (40, 40, 44),
    "chest_inside": (160, 210, 230),
    "chest_open_body": (30, 150, 200),
    "agent_body": (246, 130, 59),
    "agent_head": (150, 180, 220),
    "agent_visor": (255, 255, 255),
    "prompt": (255, 255, 255),
    "prompt_text": (20, 20, 20),
    "progress_fill": (94, 197, 34),
    "progress_frame": (255, 255, 255),
    "reward": (238, 211, 34),
}

#: Box kinds whose top surface can be stood on.
WALKABLE_KINDS: Final = frozenset({"ground", "floor", "stairs", "pier", "platform", "rock_top"})


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned box ``[x0, x1] x [y0, y1] x [z0, z1]`` in metres."""

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    kind: str
    colour: str
    blocks: bool = True
    walkable: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        if not (self.x1 > self.x0 and self.y1 > self.y0 and self.z1 > self.z0):
            raise ValueError(f"degenerate box {self}")
        if self.colour not in COLOURS:
            raise ValueError(f"unknown colour {self.colour!r}")

    @property
    def top(self) -> float:
        return self.z1

    @property
    def centre(self) -> tuple[float, float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, (self.z0 + self.z1) / 2)

    def contains_xy(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x0 - margin <= x <= self.x1 + margin
                and self.y0 - margin <= y <= self.y1 + margin)

    def to_dict(self) -> dict[str, object]:
        return {"min": [self.x0, self.y0, self.z0], "max": [self.x1, self.y1, self.z1],
                "kind": self.kind, "colour": self.colour, "blocks": self.blocks,
                "walkable": self.walkable, "name": self.name}


def box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, kind: str,
        colour: str, *, blocks: bool = True, walkable: bool | None = None,
        name: str = "") -> Box:
    """Build a box from any two opposite corners."""
    return Box(min(x0, x1), min(y0, y1), min(z0, z1), max(x0, x1), max(y0, y1), max(z0, z1),
               kind, colour, blocks, kind in WALKABLE_KINDS if walkable is None else walkable,
               name)


def circle_overlaps_box_xy(x: float, y: float, radius: float, b: Box) -> bool:
    nx = min(max(x, b.x0), b.x1)
    ny = min(max(y, b.y0), b.y1)
    return (x - nx) ** 2 + (y - ny) ** 2 < radius * radius


class World3D:
    """Static boxes plus queries used by physics, navigation and rendering."""

    HASH_CELL = 2.0

    def __init__(self, boxes: Sequence[Box], *, bounds: tuple[float, float, float, float]
                 = (0.0, 0.0, 100.0, 100.0)) -> None:
        self.boxes: tuple[Box, ...] = tuple(boxes)
        self.bounds = bounds
        self._hash: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, b in enumerate(self.boxes):
            if b.kind == "ground":
                continue          # handled analytically (z = 0 plane)
            for key in self._keys(b.x0, b.y0, b.x1, b.y1):
                self._hash[key].append(i)
        arr = np.array([[b.x0, b.y0, b.z0, b.x1, b.y1, b.z1] for b in self.boxes],
                       dtype=np.float64).reshape(-1, 6)
        self._lo = arr[:, :3]
        self._hi = arr[:, 3:]
        self._blocks = np.array([b.blocks for b in self.boxes], dtype=bool)
        self.ground_z = max((b.z1 for b in self.boxes if b.kind == "ground"), default=0.0)

    def _keys(self, x0: float, y0: float, x1: float, y1: float) -> Iterable[tuple[int, int]]:
        c = self.HASH_CELL
        for i in range(math.floor(x0 / c), math.floor(x1 / c) + 1):
            for j in range(math.floor(y0 / c), math.floor(y1 / c) + 1):
                yield i, j

    def nearby(self, x: float, y: float, radius: float) -> list[Box]:
        """Boxes whose footprint may intersect the disc (without the ground)."""
        seen: set[int] = set()
        out: list[Box] = []
        for key in self._keys(x - radius, y - radius, x + radius, y + radius):
            for i in self._hash.get(key, ()):
                if i not in seen:
                    seen.add(i)
                    out.append(self.boxes[i])
        return out

    # -- character support and collision ------------------------------------------------

    def support_height(self, x: float, y: float, z: float, *, radius: float = 0.2,
                       step_height: float = 0.45) -> float:
        """Highest walkable top under the footprint that is at most ``step_height``
        above ``z`` (the ground plane counts). Returns ``-inf`` over a hole."""
        best = self.ground_z if self._inside_bounds(x, y) else -math.inf
        limit = z + step_height + 1e-6
        for b in self.nearby(x, y, radius):
            if b.walkable and b.z1 <= limit and b.z1 > best \
                    and circle_overlaps_box_xy(x, y, radius, b):
                best = b.z1
        return best

    def _inside_bounds(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.bounds
        return x0 <= x <= x1 and y0 <= y <= y1

    def blocked(self, x: float, y: float, z: float, *, radius: float, height: float,
                step_height: float) -> bool:
        """Would a capsule (feet at ``z``) overlap a blocking box it cannot step onto?"""
        if not self._inside_bounds(x, y):
            return True
        lo = z + step_height
        hi = z + height
        for b in self.nearby(x, y, radius):
            if b.blocks and b.z1 > lo and b.z0 < hi and circle_overlaps_box_xy(x, y, radius, b):
                return True
        return False

    # -- ray casts ---------------------------------------------------------------------------

    def raycast(self, origin: Sequence[float], direction: Sequence[float], max_distance: float,
                *, blocking_only: bool = True, ignore: Sequence[Box] = ()) -> float | None:
        """Distance to the first box hit along a ray (vectorised slab test), or ``None``."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return None
        d = d / n
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1 = (self._lo - o) * inv
            t2 = (self._hi - o) * inv
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin <= max_distance)
        if blocking_only:
            hit &= self._blocks
        if ignore:
            ids = {id(b) for b in ignore}
            for i in np.nonzero(hit)[0]:
                if id(self.boxes[i]) in ids:
                    hit[i] = False
        if not hit.any():
            return None
        dist = np.maximum(tmin[hit], 0.0)
        best = float(dist.min())
        return best if best <= max_distance else None

    def line_of_sight(self, a: Sequence[float], b: Sequence[float],
                      ignore: Sequence[Box] = ()) -> bool:
        a_ = np.asarray(a, dtype=np.float64)
        d = np.asarray(b, dtype=np.float64) - a_
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            return True
        hit = self.raycast(a_, d, length - 1e-3, ignore=ignore)
        return hit is None
