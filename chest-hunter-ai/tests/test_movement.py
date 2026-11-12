"""Tests for path smoothing and continuous waypoint following."""

from __future__ import annotations

import math
import unittest

from chest_hunter.navigation.astar import astar
from chest_hunter.navigation.movement import (
    MovementController,
    MoveStatus,
    build_waypoints,
    smooth_path,
)
from chest_hunter.simulation.collision import GridMap, circle_collides, segment_clear
from chest_hunter.simulation.entities import PlayerAgent

DT = 1.0 / 30.0


def agent_at(x: float, y: float) -> PlayerAgent:
    return PlayerAgent(position_x=x, position_y=y, max_speed=5.0, acceleration=12.0, radius=0.3)


def follow(grid: GridMap, agent: PlayerAgent, goal: tuple[float, float], *,
           max_ticks: int = 3000, stop_radius: float = 1.0,
           arrive_radius: float = 1.5) -> tuple[MoveStatus, int, list[int], MovementController]:
    result = astar(grid, agent.cell, GridMap.cell_of(*goal))
    assert result.found
    waypoints = build_waypoints(grid, result.path, agent.position, goal, radius=agent.radius)
    controller = MovementController(dt=DT, stuck_timeout=5.0)
    controller.set_path(waypoints, goal=goal, stop_radius=stop_radius,
                        arrive_radius=arrive_radius, time=0.0)
    indices = []
    status = MoveStatus.MOVING
    tick = 0
    for tick in range(1, max_ticks + 1):
        previous_speed = agent.speed
        status = controller.update(agent, grid, tick * DT)
        indices.append(controller._index)
        assert agent.speed <= agent.max_speed + 1e-9
        assert abs(agent.speed - previous_speed) <= agent.acceleration * DT + 1e-6
        assert not circle_collides(grid, agent.position_x, agent.position_y, agent.radius)
        if status is not MoveStatus.MOVING:
            break
    return status, tick, indices, controller


class SmoothingTests(unittest.TestCase):
    def test_open_space_collapses_to_a_straight_line(self) -> None:
        grid = GridMap(20, 20)
        result = astar(grid, (1, 1), (15, 9))
        points = build_waypoints(grid, result.path, (1.5, 1.5), (15.5, 9.5), radius=0.3)
        self.assertEqual(points, [(1.5, 1.5), (15.5, 9.5)])

    def test_smoothed_segments_stay_clear_of_walls(self) -> None:
        grid = GridMap(20, 20)
        grid.fill_rect(8, 0, 1, 15)
        result = astar(grid, (2, 2), (15, 2))
        points = build_waypoints(grid, result.path, (2.5, 2.5), (15.5, 2.5), radius=0.3)
        self.assertGreater(len(points), 2)
        self.assertLess(len(points), len(result.path))
        for a, b in zip(points, points[1:], strict=False):
            self.assertTrue(segment_clear(grid, a, b, 0.3))

    def test_short_paths_are_unchanged(self) -> None:
        self.assertEqual(smooth_path(GridMap(5, 5), [(0.5, 0.5), (1.5, 0.5)], 0.3),
                         [(0.5, 0.5), (1.5, 0.5)])


class MovementTests(unittest.TestCase):
    def test_arrives_and_stops_within_interaction_range(self) -> None:
        grid = GridMap(30, 30)
        agent = agent_at(2.5, 2.5)
        goal = (20.5, 12.5)
        status, ticks, _, _ = follow(grid, agent, goal)
        self.assertIs(status, MoveStatus.ARRIVED)
        distance = agent.distance_to(*goal)
        self.assertLessEqual(distance, 1.5)
        self.assertGreaterEqual(distance, 0.5)  # stops near the radius, not on top of it
        self.assertLessEqual(agent.speed, agent.acceleration * DT + 1e-9)  # stops next tick
        straight = math.dist((2.5, 2.5), goal) / 5.0
        self.assertLess(ticks * DT, straight + 2.0)  # efficient: close to straight-line time

    def test_waypoints_progress_monotonically_around_obstacles(self) -> None:
        grid = GridMap(30, 30)
        grid.fill_rect(10, 0, 1, 25)
        grid.fill_rect(20, 5, 1, 25)
        agent = agent_at(2.5, 2.5)
        status, _, indices, controller = follow(grid, agent, (27.5, 2.5))
        self.assertIs(status, MoveStatus.ARRIVED)
        self.assertEqual(indices, sorted(indices))
        self.assertGreater(len(controller.waypoints), 3)

    def test_no_oscillation_near_the_goal(self) -> None:
        grid = GridMap(20, 20)
        agent = agent_at(2.5, 10.5)
        status, ticks, _, _ = follow(grid, agent, (12.5, 10.5), stop_radius=0.0,
                                     arrive_radius=0.5)
        self.assertIs(status, MoveStatus.ARRIVED)
        self.assertLess(ticks, 200)

    def test_stuck_detection_when_the_path_is_impossible(self) -> None:
        grid = GridMap(20, 20)
        grid.fill_rect(10, 0, 1, 20)  # complete wall
        agent = agent_at(5.5, 5.5)
        controller = MovementController(dt=DT, stuck_timeout=2.0)
        controller.set_path([(5.5, 5.5), (15.5, 5.5)], goal=(15.5, 5.5), stop_radius=1.0,
                            arrive_radius=1.5, time=0.0)
        status = MoveStatus.MOVING
        tick = 0
        for tick in range(1, 400):
            status = controller.update(agent, grid, tick * DT)
            if status is not MoveStatus.MOVING:
                break
        self.assertIs(status, MoveStatus.STUCK)
        self.assertLess(tick * DT, 4.0)
        self.assertFalse(controller.path_is_clear(grid, agent))

    def test_idle_without_path_brakes(self) -> None:
        grid = GridMap(10, 10)
        agent = agent_at(5.5, 5.5)
        agent.velocity_x, agent.speed = 3.0, 3.0
        controller = MovementController(dt=DT, stuck_timeout=5.0)
        self.assertIs(controller.update(agent, grid, 0.0), MoveStatus.IDLE)
        self.assertLess(agent.speed, 3.0)
        with self.assertRaises(ValueError):
            controller.set_path([], goal=(1.0, 1.0), stop_radius=0, arrive_radius=1, time=0)


if __name__ == "__main__":
    unittest.main()
