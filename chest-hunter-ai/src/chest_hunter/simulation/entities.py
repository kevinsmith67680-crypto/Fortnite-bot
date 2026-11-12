"""Lightweight simulation entities: the player agent, chests and obstacles.

All positions and distances are in grid units (see ``collision.py``); speeds
are grid units per simulated second.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from chest_hunter.simulation.collision import Cell, GridMap, move_with_collision


@dataclass(slots=True)
class Obstacle:
    """An axis-aligned rectangle of blocked cells (integer cell coordinates)."""

    position_x: float
    position_y: float
    width: float
    height: float
    kind: str = "wall"

    def cells(self) -> Iterator[Cell]:
        x0, y0 = int(self.position_x), int(self.position_y)
        for cy in range(y0, y0 + int(self.height)):
            for cx in range(x0, x0 + int(self.width)):
                yield (cx, cy)

    def contains(self, x: float, y: float) -> bool:
        return (self.position_x <= x < self.position_x + self.width
                and self.position_y <= y < self.position_y + self.height)


@dataclass(frozen=True, slots=True)
class Decoration:
    """A walkable, render-only prop (e.g. a hay bale).

    Decorations are deliberately gold-coloured so that colour-only chest
    detection produces false positives; a good detector must reject them.
    """

    position_x: float
    position_y: float
    radius: float = 0.35
    kind: str = "hay_bale"


@dataclass(slots=True)
class Chest:
    """One spawn location in the current episode.

    ``present`` is decided once per episode from ``spawn_probability``;
    ``opened`` becomes true at most once.
    """

    id: int
    position_x: float
    position_y: float
    spawn_probability: float
    present: bool = False
    opened: bool = False
    location_name: str = ""
    estimated_open_seconds: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return (self.position_x, self.position_y)

    @property
    def cell(self) -> Cell:
        return (math.floor(self.position_x), math.floor(self.position_y))

    @property
    def available(self) -> bool:
        return self.present and not self.opened

    def reset(self, present: bool) -> None:
        self.present = present
        self.opened = False

    def open(self) -> bool:
        """Open the chest. Returns False if it is absent or already open."""
        if not self.present or self.opened:
            return False
        self.opened = True
        return True


@dataclass(slots=True)
class PlayerAgent:
    """The autonomous player: a circle with speed and acceleration limits.

    ``speed`` is the current speed; ``max_speed`` the limit. ``heading`` is in
    radians (0 = east, counter-clockwise positive).
    """

    position_x: float
    position_y: float
    speed: float = 0.0
    heading: float = 0.0
    interaction_radius: float = 1.5
    current_target: int | None = None
    max_speed: float = 5.0
    acceleration: float = 12.0
    radius: float = 0.3
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    distance_travelled: float = 0.0
    collisions: int = 0

    @property
    def position(self) -> tuple[float, float]:
        return (self.position_x, self.position_y)

    @property
    def cell(self) -> Cell:
        return (math.floor(self.position_x), math.floor(self.position_y))

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.position_x, y - self.position_y)

    def in_interaction_range(self, chest: Chest) -> bool:
        return self.distance_to(chest.position_x, chest.position_y) <= self.interaction_radius

    def steer(self, desired_vx: float, desired_vy: float, dt: float) -> None:
        """Change velocity toward the desired velocity within the acceleration limit."""
        desired_speed = math.hypot(desired_vx, desired_vy)
        if desired_speed > self.max_speed:
            scale = self.max_speed / desired_speed
            desired_vx *= scale
            desired_vy *= scale
        dvx = desired_vx - self.velocity_x
        dvy = desired_vy - self.velocity_y
        change = math.hypot(dvx, dvy)
        limit = self.acceleration * dt
        if change > limit:
            dvx *= limit / change
            dvy *= limit / change
        self.velocity_x += dvx
        self.velocity_y += dvy
        self.speed = math.hypot(self.velocity_x, self.velocity_y)
        if self.speed > self.max_speed:  # numerical safety
            scale = self.max_speed / self.speed
            self.velocity_x *= scale
            self.velocity_y *= scale
            self.speed = self.max_speed

    def brake(self, dt: float) -> None:
        self.steer(0.0, 0.0, dt)

    def step(self, grid: GridMap, dt: float) -> bool:
        """Integrate one fixed timestep with collision; return True if movement was blocked."""
        if self.speed <= 1e-9:
            self.velocity_x = self.velocity_y = self.speed = 0.0
            return False
        old_x, old_y = self.position_x, self.position_y
        new_x, new_y, blocked_x, blocked_y = move_with_collision(
            grid, old_x, old_y, self.velocity_x * dt, self.velocity_y * dt, self.radius
        )
        if blocked_x:
            self.velocity_x = 0.0
        if blocked_y:
            self.velocity_y = 0.0
        self.position_x, self.position_y = new_x, new_y
        self.distance_travelled += math.hypot(new_x - old_x, new_y - old_y)
        self.speed = math.hypot(self.velocity_x, self.velocity_y)
        if self.speed > 1e-9:
            self.heading = math.atan2(self.velocity_y, self.velocity_x)
        if blocked_x or blocked_y:
            self.collisions += 1
            return True
        return False

    def stop(self) -> None:
        self.velocity_x = self.velocity_y = self.speed = 0.0
