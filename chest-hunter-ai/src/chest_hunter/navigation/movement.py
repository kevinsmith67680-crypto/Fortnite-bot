"""Continuous waypoint following with acceleration limits and stuck detection.

The A* cell path is converted to waypoints (cell centres), optionally
shortened by line-of-sight smoothing that respects the agent's radius, and
followed with simple steering:

* desired speed is limited by the turn angle at the next waypoint (slow
  down for sharp corners) and by the distance left to the stopping point
  (``v = sqrt(2 a d)``), so the agent decelerates smoothly and stops inside
  interaction range instead of overshooting;
* velocity changes are limited by the agent's acceleration;
* a waypoint is consumed when the agent is close to it *or has passed it*,
  which prevents oscillating around a waypoint;
* if the remaining path length has not improved for ``stuck_timeout``
  simulated seconds, :attr:`MoveStatus.STUCK` is reported.

Everything runs on the fixed simulation timestep; no wall-clock time and no
randomness is involved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

from chest_hunter.simulation.collision import Cell, GridMap, Point, segment_clear
from chest_hunter.simulation.entities import PlayerAgent

SMOOTHING_MARGIN = 0.05


class MoveStatus(StrEnum):
    IDLE = "idle"
    MOVING = "moving"
    ARRIVED = "arrived"
    STUCK = "stuck"


def smooth_path(grid: GridMap, points: Sequence[Point], radius: float) -> list[Point]:
    """Greedy line-of-sight shortcutting: drop waypoints the agent can skip safely."""
    if len(points) <= 2:
        return list(points)
    result = [points[0]]
    anchor = points[0]
    index = 1
    while index < len(points) - 1:
        if segment_clear(grid, anchor, points[index + 1], radius):
            index += 1
            continue
        result.append(points[index])
        anchor = points[index]
        index += 1
    result.append(points[-1])
    return result


def build_waypoints(
    grid: GridMap,
    cell_path: Sequence[Cell],
    start: Point,
    goal: Point,
    *,
    radius: float,
    smooth: bool = True,
) -> list[Point]:
    """Turn an A* cell path into continuous waypoints from ``start`` to ``goal``."""
    centres = [GridMap.cell_center(cell) for cell in cell_path]
    points: list[Point] = [start]
    points.extend(centres)  # includes the start cell centre (helps leave corners)
    points.append(goal)
    if smooth:
        return smooth_path(grid, points, radius + SMOOTHING_MARGIN)
    # Without smoothing still drop the start cell centre if it is not needed.
    return points


class MovementController:
    """Steers a :class:`PlayerAgent` along waypoints on a fixed timestep."""

    def __init__(
        self,
        *,
        dt: float,
        stuck_timeout: float,
        waypoint_tolerance: float = 0.35,
        arrival_speed: float | None = None,
        progress_epsilon: float = 0.1,
        min_corner_factor: float = 0.2,
    ) -> None:
        self.dt = dt
        self.stuck_timeout = stuck_timeout
        self.waypoint_tolerance = waypoint_tolerance
        # Default: slow enough to come to rest within one tick without
        # exceeding the acceleration limit (speed <= acceleration * dt).
        self.arrival_speed = arrival_speed
        self.progress_epsilon = progress_epsilon
        self.min_corner_factor = min_corner_factor
        self.clear()

    # -- path management ------------------------------------------------------

    def clear(self) -> None:
        self._waypoints: list[Point] = []
        self._suffix: list[float] = []
        self._index = 0
        self.goal: Point | None = None
        self.stop_radius = 0.0
        self.arrive_radius = 0.0
        self._best_remaining = math.inf
        self._last_progress_time = 0.0

    def set_path(self, waypoints: Sequence[Point], *, goal: Point, stop_radius: float,
                 arrive_radius: float, time: float) -> None:
        """Follow ``waypoints`` and stop within ``arrive_radius`` of ``goal``."""
        if not waypoints:
            raise ValueError("A path needs at least one waypoint.")
        self._waypoints = list(waypoints)
        suffix = [0.0] * len(self._waypoints)
        for i in range(len(self._waypoints) - 2, -1, -1):
            a, b = self._waypoints[i], self._waypoints[i + 1]
            suffix[i] = suffix[i + 1] + math.hypot(b[0] - a[0], b[1] - a[1])
        self._suffix = suffix
        self._index = 1 if len(self._waypoints) > 1 else 0
        self.goal = goal
        self.stop_radius = stop_radius
        self.arrive_radius = arrive_radius
        self._best_remaining = math.inf
        self._last_progress_time = time

    @property
    def has_path(self) -> bool:
        return bool(self._waypoints)

    @property
    def waypoints(self) -> list[Point]:
        return list(self._waypoints)

    def remaining_waypoints(self) -> list[Point]:
        return self._waypoints[self._index:]

    def remaining_distance(self, agent: PlayerAgent) -> float:
        if not self._waypoints:
            return 0.0
        wx, wy = self._waypoints[self._index]
        return agent.distance_to(wx, wy) + self._suffix[self._index]

    def path_is_clear(self, grid: GridMap, agent: PlayerAgent) -> bool:
        """Whether the rest of the path is still free of obstacles."""
        points = [agent.position, *self.remaining_waypoints()]
        return all(
            segment_clear(grid, points[i], points[i + 1], 0.0) for i in range(len(points) - 1)
        )

    # -- per-tick update --------------------------------------------------------

    def update(self, agent: PlayerAgent, grid: GridMap, time: float) -> MoveStatus:
        if not self._waypoints or self.goal is None:
            agent.brake(self.dt)
            agent.step(grid, self.dt)
            return MoveStatus.IDLE

        last = len(self._waypoints) - 1
        while self._index < last and self._reached(agent, self._index):
            self._index += 1

        goal_x, goal_y = self.goal
        goal_distance = agent.distance_to(goal_x, goal_y)
        arrival_speed = (self.arrival_speed if self.arrival_speed is not None
                         else agent.acceleration * self.dt)
        if goal_distance <= self.arrive_radius and agent.speed <= arrival_speed:
            agent.stop()
            return MoveStatus.ARRIVED

        tx, ty = self._waypoints[self._index]
        dx, dy = tx - agent.position_x, ty - agent.position_y
        distance = math.hypot(dx, dy)
        remaining = distance + self._suffix[self._index]
        accel = agent.acceleration
        # Braking profile toward the stopping point, measured along the path
        # (not straight-line, so passing a chest behind a wall does not brake).
        stopping = math.sqrt(2.0 * accel * max(0.0, remaining - self.stop_radius))
        desired_speed = min(agent.max_speed, stopping)
        if self._index < last:
            desired_speed = min(desired_speed, self._corner_speed(agent, distance))

        if distance > 1e-9:
            agent.steer(dx / distance * desired_speed, dy / distance * desired_speed, self.dt)
        else:
            agent.brake(self.dt)
        agent.step(grid, self.dt)

        remaining = self.remaining_distance(agent)
        if remaining < self._best_remaining - self.progress_epsilon:
            self._best_remaining = remaining
            self._last_progress_time = time
        elif time - self._last_progress_time > self.stuck_timeout:
            return MoveStatus.STUCK

        if (agent.distance_to(goal_x, goal_y) <= self.arrive_radius
                and agent.speed <= arrival_speed):
            # Already decelerated this tick; the remaining (<= a*dt) speed is
            # braked away on the next tick without exceeding the limit.
            return MoveStatus.ARRIVED
        return MoveStatus.MOVING

    def tighten(self) -> None:
        """Approach the goal itself (used when the target is in range but not in sight)."""
        self.stop_radius = 0.0
        self.arrive_radius = max(self.arrive_radius * 0.5, 0.25)

    # -- helpers ------------------------------------------------------------------

    def _reached(self, agent: PlayerAgent, index: int) -> bool:
        wx, wy = self._waypoints[index]
        if agent.distance_to(wx, wy) <= self.waypoint_tolerance:
            return True
        # Passed it: agent is beyond the line through the waypoint
        # perpendicular to the incoming segment.
        px, py = self._waypoints[index - 1] if index > 0 else agent.position
        seg_x, seg_y = wx - px, wy - py
        return (agent.position_x - wx) * seg_x + (agent.position_y - wy) * seg_y > 0.0

    def _corner_speed(self, agent: PlayerAgent, distance: float) -> float:
        """Speed allowed now so the agent can take the turn at the next waypoint."""
        i = self._index
        wx, wy = self._waypoints[i]
        px, py = self._waypoints[i - 1] if i > 0 else agent.position
        nx, ny = self._waypoints[i + 1]
        in_x, in_y = wx - px, wy - py
        out_x, out_y = nx - wx, ny - wy
        norm = math.hypot(in_x, in_y) * math.hypot(out_x, out_y)
        cos_turn = (in_x * out_x + in_y * out_y) / norm if norm > 1e-12 else 1.0
        factor = max(self.min_corner_factor, (1.0 + cos_turn) / 2.0)
        corner = agent.max_speed * factor
        return math.sqrt(corner * corner + 2.0 * agent.acceleration * distance)
