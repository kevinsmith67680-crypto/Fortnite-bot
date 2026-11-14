"""3D navigation for the visual agent (Phase 4).

The *visual* agent chooses where to go; this module gets it there.

* :class:`NavMeshNavigator` (hybrid mode, the primary Phase 4 target) asks the
  environment's navigation service for a path to an *estimated* destination
  (``NAV_QUERY``, Unreal's ``FindPathToLocationSynchronously``). It only
  ever sends positions the agent estimated itself; it never asks where
  chests are.
* :class:`LocalNavigator` (visual-navigation mode, reduced assistance) gets no
  global paths. It probes its surroundings with short navigation-mesh rays
  (``NAV_RAYCAST``, like a 2D lidar), builds its own occupancy grid per floor
  and plans with the Phase 2 A* (unknown cells allowed at extra cost).
  Targets on another floor are out of reach for it.
* :class:`PathFollower` turns waypoints into a heading and speed (look-ahead
  pursuit, slowing at corners and near the end) and detects lack of progress.
* :class:`ScanPlanner` produces camera yaw targets for an active search.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from chest_hunter.navigation.astar import PathFinder
from chest_hunter.simulation.collision import GridMap
from chest_hunter.unreal_bridge.client import NavResult
from chest_hunter.vision.perspective import angle_difference_deg, bearing_deg

Point3 = tuple[float, float, float]


class NavPort(Protocol):
    def nav_query(self, start: Point3, goal: Point3, *,
                  goal_tolerance: float = 0.5) -> NavResult: ...

    def nav_raycast(self, origin: Point3, bearings_deg: list[float], *,
                    max_distance: float = 8.0) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class Plan:
    waypoints: tuple[Point3, ...]
    length: float


class NavMeshNavigator:
    """Hybrid mode: global paths from the environment's NavMesh service."""

    name = "navmesh"

    def __init__(self, port: NavPort) -> None:
        self.port = port
        self.queries = 0
        self.failures = 0
        self.last_status = "ok"
        self._cache: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[Plan | None, str]] = {}

    def plan(self, start: Point3, goal: Point3, *, tolerance: float = 0.5) -> Plan | None:
        """A path, or ``None`` with :attr:`last_status` saying why. ``start_off_navmesh``
        is transient (the agent is mid-air or squeezed against geometry): it is never
        cached and must not be taken as proof that the goal is unreachable."""
        key = (tuple(round(c * 2) for c in start), tuple(round(c * 4) for c in goal))
        if key in self._cache:
            plan, self.last_status = self._cache[key]
            return plan
        self.queries += 1
        result = self.port.nav_query(start, goal, goal_tolerance=tolerance)
        plan = Plan(result.waypoints, result.length) if result.found and result.waypoints else None
        self.last_status = "ok" if plan is not None else result.status
        if plan is None:
            self.failures += 1
        if self.last_status != "start_off_navmesh":
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[key] = (plan, self.last_status)
        return plan

    def path_length(self, start: Point3, goal: Point3, *, tolerance: float = 1.5) -> float | None:
        plan = self.plan(start, goal, tolerance=tolerance)
        return None if plan is None else plan.length

    def clear_line(self, start: Point3, target: Point3) -> bool:
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
        if dist < 0.5:
            return True
        bearing = bearing_deg(target[0] - start[0], target[1] - start[1])
        reach = self.port.nav_raycast(start, [bearing], max_distance=min(15.0, dist + 0.5))[0]
        return reach >= dist - 0.9

    def note_obstacle(self, position: Point3, bearing: float) -> None:
        self._cache.clear()                  # the NavMesh is authoritative; just replan

    def observe(self, position: Point3) -> None:
        return None


class LocalNavigator:
    """Reduced assistance: own occupancy map from local probes, own A*."""

    name = "local"
    RES = 0.5
    PROBE_BEARINGS = 16
    PROBE_RANGE = 8.0

    def __init__(self, port: NavPort, *, bounds: tuple[float, float, float, float]
                 = (0.0, 0.0, 100.0, 100.0), floors: tuple[float, ...] = (0.0,)) -> None:
        self.port = port
        self.floors = tuple(sorted(set(floors) | {0.0}))
        self.bounds = bounds
        self.nx = int(math.ceil((bounds[2] - bounds[0]) / self.RES))
        self.ny = int(math.ceil((bounds[3] - bounds[1]) / self.RES))
        self.grids: dict[int, np.ndarray] = {}         # floor bucket -> int8 (-1/0/1)
        self.queries = 0
        self.failures = 0
        self.last_status = "ok"
        self.probes = 0
        self._last_probe: Point3 | None = None
        self._finder = PathFinder(cache_paths=False)

    def _floor(self, z: float) -> int:
        return int(round(z * 2))                       # 0.5 m buckets

    def grid(self, z: float) -> np.ndarray:
        key = self._floor(z)
        if key not in self.grids:
            self.grids[key] = np.full((self.nx, self.ny), -1, np.int8)
        return self.grids[key]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int((x - self.bounds[0]) / self.RES), int((y - self.bounds[1]) / self.RES))

    def observe(self, position: Point3) -> None:
        """Probe the surroundings (rate-limited by distance moved)."""
        if self._last_probe is not None and math.dist(self._last_probe, position) < 1.5:
            return
        self._last_probe = position
        bearings = [i * 360.0 / self.PROBE_BEARINGS for i in range(self.PROBE_BEARINGS)]
        reaches = self.port.nav_raycast(position, bearings, max_distance=self.PROBE_RANGE)
        self.probes += 1
        g = self.grid(position[2])
        for b, reach in zip(bearings, reaches, strict=True):
            rad = math.radians(b)
            dx, dy = math.cos(rad), math.sin(rad)
            steps = int(reach / (self.RES * 0.5))
            for k in range(steps + 1):
                i, j = self._cell(position[0] + dx * k * self.RES * 0.5,
                                  position[1] + dy * k * self.RES * 0.5)
                if 0 <= i < self.nx and 0 <= j < self.ny and g[i, j] != 1:
                    g[i, j] = 0
            if reach < self.PROBE_RANGE - 0.01:
                ahead = reach + 0.3
                i, j = self._cell(position[0] + dx * ahead, position[1] + dy * ahead)
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    g[i, j] = 1

    def note_obstacle(self, position: Point3, bearing: float) -> None:
        g = self.grid(position[2])
        rad = math.radians(bearing)
        i, j = self._cell(position[0] + math.cos(rad) * 0.7, position[1] + math.sin(rad) * 0.7)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            g[i, j] = 1

    def plan(self, start: Point3, goal: Point3, *, tolerance: float = 0.5) -> Plan | None:
        self.queries += 1
        self.last_status = "ok"
        if not self.on_floor(start):
            self.failures += 1
            self.last_status = "start_off_navmesh"      # mid-stairs or airborne: transient
            return None
        if abs(goal[2] - start[2]) > 0.8:
            self.failures += 1
            self.last_status = "other_floor"
            return None                                 # no stairs knowledge in this mode
        g = self.grid(start[2])
        grid = GridMap(self.nx, self.ny)
        blocked = np.argwhere(g == 1)
        for i, j in blocked:
            grid.set_blocked(int(i), int(j))
        penalty = bytearray((g.T.flatten() == -1).astype(np.uint8).tobytes())
        self._finder = PathFinder(cache_paths=False, penalty_cells=penalty, penalty=0.6)
        s = self._cell(start[0], start[1])
        gc = self._cell(goal[0], goal[1])
        if grid.is_blocked(*gc):
            free = self._free_near(grid, gc, tolerance)
            if free is None:
                self.failures += 1
                self.last_status = "goal_off_navmesh"
                return None
            gc = free
        if grid.is_blocked(*s):
            grid.set_blocked(s[0], s[1], False)
        result = self._finder.find_path(grid, s, gc)
        if not result.found:
            self.failures += 1
            self.last_status = "unreachable"
            return None
        x0, y0 = self.bounds[0], self.bounds[1]
        pts = [(x0 + (i + 0.5) * self.RES, y0 + (j + 0.5) * self.RES, start[2])
               for i, j in result.path]
        pts = _decimate(pts)
        return Plan(tuple(pts), result.cost * self.RES)

    def _free_near(self, grid: GridMap, cell: tuple[int, int],
                   tolerance: float) -> tuple[int, int] | None:
        reach = max(1, int(tolerance / self.RES))
        best = None
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                c = (cell[0] + di, cell[1] + dj)
                if grid.in_bounds(*c) and not grid.is_blocked(*c):
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, c)
        return None if best is None else best[1]

    def on_floor(self, position: Point3) -> bool:
        """Standing at a known floor height (not on a staircase or in the air)?"""
        return any(abs(position[2] - f) <= 0.3 for f in self.floors)

    def path_length(self, start: Point3, goal: Point3, *, tolerance: float = 1.5) -> float | None:
        if not self.on_floor(start):
            self.last_status = "start_off_navmesh"
            return None
        if abs(goal[2] - start[2]) > 0.8:
            self.last_status = "other_floor"
            return None
        return math.dist(start[:2], goal[:2]) * 1.3      # cheap estimate; planned on demand

    def clear_line(self, start: Point3, target: Point3) -> bool:
        dist = math.hypot(target[0] - start[0], target[1] - start[1])
        if dist < 0.5:
            return True
        bearing = bearing_deg(target[0] - start[0], target[1] - start[1])
        reach = self.port.nav_raycast(start, [bearing], max_distance=min(15.0, dist + 0.5))[0]
        return reach >= dist - 0.9


def _decimate(points: list[Point3]) -> list[Point3]:
    """Keep only the points where the direction changes."""
    if len(points) <= 2:
        return points
    out = [points[0]]
    for a, b, c in zip(points, points[1:], points[2:], strict=False):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) > 1e-9:
            out.append(b)
    out.append(points[-1])
    return out


@dataclass(slots=True)
class PathFollower:
    """Look-ahead pursuit along waypoints with progress-based stuck detection."""

    waypoints: tuple[Point3, ...]
    arrive_radius: float = 0.6
    stuck_seconds: float = 2.5
    index: int = 0
    _best_remaining: float = math.inf
    _last_progress_time: float = 0.0
    history: list[Point3] = field(default_factory=list)

    def remaining(self, position: Point3) -> float:
        if self.index >= len(self.waypoints):
            return 0.0
        total = math.dist(position[:2], self.waypoints[self.index][:2])
        for a, b in zip(self.waypoints[self.index:], self.waypoints[self.index + 1:], strict=False):
            total += math.dist(a[:2], b[:2])
        return total

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def steer(self, position: Point3, now: float) -> tuple[float, float]:
        """(bearing in degrees, speed fraction) towards the current waypoint.

        Waypoints come from a string-pulled path whose segments keep clear of
        obstacles, so steering straight at each waypoint in turn (rather than cutting
        towards a look-ahead point) keeps the agent on those safe segments; it slows
        down before sharp turns.
        """
        while self.index < len(self.waypoints):
            wp = self.waypoints[self.index]
            last = self.index == len(self.waypoints) - 1
            reach = self.arrive_radius if last else 0.35
            if math.hypot(wp[0] - position[0], wp[1] - position[1]) <= reach \
                    and abs(wp[2] - position[2]) <= 0.8:
                self.index += 1
                continue
            break
        if self.done:
            return 0.0, 0.0
        remaining = self.remaining(position)
        if remaining < self._best_remaining - 0.25:
            self._best_remaining = remaining
            self._last_progress_time = now
        target = self.waypoints[self.index]
        bearing = bearing_deg(target[0] - position[0], target[1] - position[1])
        to_wp = math.hypot(target[0] - position[0], target[1] - position[1])
        speed = 1.0 if remaining > 2.0 else max(0.35, remaining / 2.0)
        if self.index + 1 < len(self.waypoints) and to_wp < 1.2:
            nxt = self.waypoints[self.index + 1]
            turn = abs(angle_difference_deg(bearing_deg(nxt[0] - target[0], nxt[1] - target[1]),
                                            bearing))
            if turn > 45.0:
                speed = min(speed, 0.45)
        return bearing, speed

    def stuck(self, now: float) -> bool:
        return now - self._last_progress_time > self.stuck_seconds

    def start(self, now: float) -> None:
        self._last_progress_time = now
        self._best_remaining = math.inf


class ScanPlanner:
    """Camera yaw targets for an active search (never more than one full turn)."""

    @staticmethod
    def full_turn(current_yaw: float, step: float = 60.0) -> list[float]:
        return [(current_yaw + step * k) % 360.0 for k in range(1, int(360 / step) + 1)]

    @staticmethod
    def arc(centre_yaw: float, half_width: float = 75.0, step: float = 50.0,
            start_from: float | None = None) -> list[float]:
        """Sweep ``centre +- half_width``, starting on the side nearer to ``start_from``."""
        n = max(1, int(round(2 * half_width / step)))
        yaws = [(centre_yaw - half_width + k * 2 * half_width / n) % 360.0 for k in range(n + 1)]
        if start_from is not None and abs(angle_difference_deg(yaws[-1], start_from)) < \
                abs(angle_difference_deg(yaws[0], start_from)):
            yaws.reverse()
        return yaws

    @staticmethod
    def towards(points: Sequence[Point3], position: Point3) -> list[float]:
        return [bearing_deg(p[0] - position[0], p[1] - position[1]) for p in points]
