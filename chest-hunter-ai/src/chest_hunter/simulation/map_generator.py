"""Deterministic procedural obstacle generation.

The same ``(seed, width, height, density, protected cells)`` always produces
the same map. Obstacles are building outlines with doorways, straight wall
segments and small rocks. A walkable one-cell border is kept around the map,
and every *protected* cell (chest locations, agent start) keeps a clearance
radius free of obstacles, so chests are never embedded in walls.

Obstacles are generated as rectangles and then rasterised into a
:class:`~chest_hunter.simulation.collision.GridMap`; the rectangles are kept
for rendering.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from chest_hunter.simulation.collision import Cell, GridMap
from chest_hunter.simulation.entities import Obstacle

MAX_PLACEMENT_ATTEMPTS = 600


@dataclass(frozen=True)
class GeneratedMap:
    grid: GridMap
    obstacles: tuple[Obstacle, ...]
    seed: int
    density_target: float

    @property
    def density(self) -> float:
        return self.grid.blocked_fraction


class MapGenerator:
    """Builds a :class:`GeneratedMap` for a seed."""

    def __init__(self, width: int, height: int, *, density: float, seed: int,
                 clearance: int = 2) -> None:
        if width < 10 or height < 10:
            raise ValueError("Maps must be at least 10 x 10 cells.")
        if not 0.0 <= density <= 0.4:
            raise ValueError("density must be between 0 and 0.4.")
        self.width = width
        self.height = height
        self.density = density
        self.seed = seed
        self.clearance = clearance

    def generate(self, protected_cells: Iterable[Cell] = ()) -> GeneratedMap:
        rng = random.Random(
            f"chest-hunter:map:{self.seed}:{self.width}x{self.height}:{self.density}")
        protected = self._protected_zone(protected_cells)
        grid = GridMap(self.width, self.height)
        obstacles: list[Obstacle] = []
        target = int(self.density * self.width * self.height)

        attempts = 0
        while grid.blocked_count < target and attempts < MAX_PLACEMENT_ATTEMPTS:
            attempts += 1
            roll = rng.random()
            if roll < 0.4:
                pieces = self._building(rng)
            elif roll < 0.7:
                pieces = self._wall(rng)
            else:
                pieces = self._rock(rng)
            for obstacle in pieces:
                for part in self._exclude_protected(obstacle, protected):
                    obstacles.append(part)
                    grid.fill_rect(int(part.position_x), int(part.position_y),
                                   int(part.width), int(part.height))
        grid.version = 0  # a freshly generated map starts at version 0
        return GeneratedMap(grid=grid, obstacles=tuple(obstacles), seed=self.seed,
                            density_target=self.density)

    # -- helpers ------------------------------------------------------------

    def _protected_zone(self, cells: Iterable[Cell]) -> set[Cell]:
        zone: set[Cell] = set()
        r = self.clearance
        for cx, cy in cells:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    zone.add((cx + dx, cy + dy))
        return zone

    def _random_origin(self, rng: random.Random, w: int, h: int) -> tuple[int, int]:
        # Keep a one-cell walkable border around the whole map.
        x = rng.randint(1, max(1, self.width - 1 - w))
        y = rng.randint(1, max(1, self.height - 1 - h))
        return x, y

    def _building(self, rng: random.Random) -> list[Obstacle]:
        bw = rng.randint(6, min(14, self.width - 3))
        bh = rng.randint(6, min(12, self.height - 3))
        x, y = self._random_origin(rng, bw, bh)
        sides = rng.sample(("south", "north", "west", "east"), 2)
        segments: list[Obstacle] = []
        for side in ("south", "north", "west", "east"):
            if side in ("south", "north"):
                row = y if side == "south" else y + bh - 1
                cells = [(cx, row) for cx in range(x, x + bw)]
            else:
                col = x if side == "west" else x + bw - 1
                cells = [(col, cy) for cy in range(y + 1, y + bh - 1)]
            if side in sides and len(cells) > 4:
                door = rng.randint(1, len(cells) - 3)
                cells = cells[:door] + cells[door + 2:]
            segments.extend(_runs_to_obstacles(cells, "building"))
        return segments

    def _wall(self, rng: random.Random) -> list[Obstacle]:
        length = rng.randint(4, min(16, self.width - 3, self.height - 3))
        if rng.random() < 0.5:
            x, y = self._random_origin(rng, length, 1)
            return [Obstacle(x, y, length, 1, "wall")]
        x, y = self._random_origin(rng, 1, length)
        return [Obstacle(x, y, 1, length, "wall")]

    def _rock(self, rng: random.Random) -> list[Obstacle]:
        w, h = rng.randint(1, 3), rng.randint(1, 3)
        x, y = self._random_origin(rng, w, h)
        return [Obstacle(x, y, w, h, "rock")]

    def _exclude_protected(self, obstacle: Obstacle, protected: set[Cell]) -> list[Obstacle]:
        """Clip to the walkable border and drop protected cells.

        Thin obstacles are split into runs around protected cells; anything
        thicker that touches a protected cell is skipped entirely.
        """
        cells = [
            (cx, cy) for cx, cy in obstacle.cells()
            if 1 <= cx < self.width - 1 and 1 <= cy < self.height - 1
        ]
        if not cells:
            return []
        if obstacle.width > 1 and obstacle.height > 1:
            clipped = len(cells) != len(list(obstacle.cells()))
            if clipped or any(cell in protected for cell in cells):
                return []
            return [obstacle]
        free = [cell for cell in cells if cell not in protected]
        return _runs_to_obstacles(free, obstacle.kind)


def _runs_to_obstacles(cells: list[Cell], kind: str) -> list[Obstacle]:
    """Group cells of a horizontal or vertical line into contiguous rectangles."""
    if not cells:
        return []
    cells = sorted(cells, key=lambda c: (c[1], c[0]))
    horizontal = len({cy for _, cy in cells}) == 1
    if not horizontal:
        cells = sorted(cells, key=lambda c: (c[0], c[1]))
    result: list[Obstacle] = []
    start = previous = cells[0]
    for cell in cells[1:]:
        contiguous = (
            cell == (previous[0] + 1, previous[1]) if horizontal
            else cell == (previous[0], previous[1] + 1)
        )
        if not contiguous:
            result.append(_run(start, previous, kind))
            start = cell
        previous = cell
    result.append(_run(start, previous, kind))
    return result


def _run(start: Cell, end: Cell, kind: str) -> Obstacle:
    return Obstacle(start[0], start[1], end[0] - start[0] + 1, end[1] - start[1] + 1, kind)
