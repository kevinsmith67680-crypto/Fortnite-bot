"""Closed-loop local control on the *estimated* pose.

The Phase 2 :class:`~chest_hunter.navigation.movement.MovementController`
(steering, braking profile, corner slow-down, arrival and stuck detection) is
reused unchanged. Instead of the true :class:`PlayerAgent` it drives an
:class:`EstimatedBody`, a stand-in that

* reports the *estimated* position from the pose estimator,
* applies the same speed and acceleration limits to the commanded velocity,
* sends that command to the simulator through the action interface, and
* dead-reckons the pose estimate with the command (predict). Visual fixes
  then correct it (observe), closing the loop.

When pose confidence drops below the threshold, :class:`LocalController`
stops moving instead of moving blindly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from chest_hunter.localization.pose_estimator import PoseEstimator
from chest_hunter.navigation.movement import MovementController, MoveStatus
from chest_hunter.simulation.collision import GridMap, Point


class MoveCommandSink(Protocol):
    def move(self, vx: float, vy: float) -> None: ...


class EstimatedBody:
    """A PlayerAgent-compatible view of the agent built from estimates only."""

    def __init__(self, pose: PoseEstimator, actions: MoveCommandSink, *, max_speed: float,
                 acceleration: float, radius: float) -> None:
        self.pose = pose
        self.actions = actions
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.radius = radius
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.speed = 0.0

    @property
    def position_x(self) -> float:
        return self.pose.x if self.pose.x is not None else 0.0

    @property
    def position_y(self) -> float:
        return self.pose.y if self.pose.y is not None else 0.0

    @property
    def position(self) -> tuple[float, float]:
        return (self.position_x, self.position_y)

    @property
    def heading(self) -> float:
        return self.pose.heading if self.pose.heading is not None else 0.0

    @property
    def cell(self) -> tuple[int, int]:
        return (math.floor(self.position_x), math.floor(self.position_y))

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.position_x, y - self.position_y)

    def steer(self, desired_vx: float, desired_vy: float, dt: float) -> None:
        desired = math.hypot(desired_vx, desired_vy)
        if desired > self.max_speed:
            desired_vx *= self.max_speed / desired
            desired_vy *= self.max_speed / desired
        dvx, dvy = desired_vx - self.velocity_x, desired_vy - self.velocity_y
        change = math.hypot(dvx, dvy)
        limit = self.acceleration * dt
        if change > limit:
            dvx *= limit / change
            dvy *= limit / change
        self.velocity_x += dvx
        self.velocity_y += dvy
        self.speed = math.hypot(self.velocity_x, self.velocity_y)

    def brake(self, dt: float) -> None:
        self.steer(0.0, 0.0, dt)

    def step(self, grid: GridMap | None, dt: float) -> bool:
        """Send the command to the simulator and dead-reckon the estimate."""
        self.actions.move(self.velocity_x, self.velocity_y)
        self.pose.predict(self.velocity_x, self.velocity_y, dt)
        return False

    def stop(self) -> None:
        self.velocity_x = self.velocity_y = self.speed = 0.0


class LocalController:
    """Waypoint following on estimates, with a localisation-confidence gate."""

    LOW_CONFIDENCE = "low_confidence"

    def __init__(self, body: EstimatedBody, movement: MovementController, *,
                 min_confidence: float) -> None:
        self.body = body
        self.movement = movement
        self.min_confidence = min_confidence

    def set_path(self, waypoints: Sequence[Point], *, goal: Point, stop_radius: float,
                 arrive_radius: float, time: float) -> None:
        self.movement.set_path(waypoints, goal=goal, stop_radius=stop_radius,
                               arrive_radius=arrive_radius, time=time)

    def update(self, grid: GridMap, time: float, confidence: float) -> MoveStatus | str:
        if confidence < self.min_confidence:
            self.hold(self.movement.dt)
            return self.LOW_CONFIDENCE
        return self.movement.update(self.body, grid, time)  # type: ignore[arg-type]

    def hold(self, dt: float) -> None:
        """Brake within the acceleration limit and keep the estimate in sync."""
        self.body.brake(dt)
        self.body.step(None, dt)

    def clear(self) -> None:
        self.movement.clear()
