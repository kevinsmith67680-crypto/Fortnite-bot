"""3D navigation in the reference arena: NavMesh-style graph, path following with the real
capsule physics (open terrain, buildings, doorways, obstacles, stairs), unreachable goals,
NavMesh-free local navigation and stuck detection."""

from __future__ import annotations

import math
import unittest

from chest_hunter.navigation.navigation_3d import (
    LocalNavigator,
    NavMeshNavigator,
    PathFollower,
    ScanPlanner,
)
from chest_hunter.simulation3d.environment3d import Environment3D, EnvSettings
from chest_hunter.unreal_bridge.action_adapter import move_towards
from chest_hunter.unreal_bridge.client import NavResult
from chest_hunter.unreal_bridge.protocol import IDLE_ACTION

MAP = "research_arena_3d"


class EnvPort:
    """Navigation service of the reference environment, as the bridge would expose it."""

    def __init__(self, env: Environment3D) -> None:
        self.env = env

    def nav_query(self, start, goal, *, goal_tolerance: float = 0.5) -> NavResult:  # type: ignore[no-untyped-def]
        p = self.env.find_path(start, goal, goal_tolerance)
        return NavResult(p.status, tuple(p.waypoints), p.length)

    def nav_raycast(self, origin, bearings_deg, *, max_distance: float = 8.0) -> list[float]:  # type: ignore[no-untyped-def]
        return [self.env.nav_raycast(origin, b, max_distance) for b in bearings_deg]


def _spawn(env: Environment3D, name: str):  # type: ignore[no-untyped-def]
    return next(s for s in env.arena.spawns if s.location_name.endswith(name))


def _stand_point(sp) -> tuple[float, float, float]:  # type: ignore[no-untyped-def]
    rad = math.radians(sp.yaw_deg)
    return (sp.x + 1.4 * math.cos(rad), sp.y + 1.4 * math.sin(rad), sp.z)


def _true(env: Environment3D) -> tuple[float, float, float]:
    ch = env.character
    return (ch.x, ch.y, ch.z)


def _walk(env: Environment3D, goal: tuple[float, float, float], *, seconds: float = 120.0,
          tolerance: float = 0.8) -> tuple[bool, float]:
    """Plan with the NavMesh service and drive the capsule along the path (true pose)."""
    env.reset()
    nav = NavMeshNavigator(EnvPort(env))
    plan = nav.plan(_true(env), goal, tolerance=tolerance)
    if plan is None:
        raise AssertionError(f"no path to {goal}: {nav.last_status}")
    follower = PathFollower(plan.waypoints, arrive_radius=0.6)
    follower.start(env.time)
    while env.time < seconds and not follower.done:
        pos = _true(env)
        bearing, speed = follower.steer(pos, env.time)
        fwd, right = move_towards(bearing, env.character.cam_yaw, speed)
        yaw_err = (bearing - env.character.cam_yaw + 540) % 360 - 180
        env.step({**IDLE_ACTION, "move_forward": fwd, "move_right": right,
                  "look_yaw": max(-1.0, min(1.0, yaw_err / 30.0))}, 3)
        if follower.stuck(env.time):
            raise AssertionError(f"stuck at {pos} on the way to {goal}")
    end = _true(env)
    return follower.done, math.dist(end[:2], goal[:2])


class TestNavigation3D(unittest.TestCase):
    """All tests share one environment, built once for the class: the tests that move the
    character reset it first."""

    env: Environment3D

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.env = Environment3D(MAP, seed=1, settings=EnvSettings(width=320, height=180))

    def test_the_graph_covers_the_arena(self) -> None:
        env = self.env
        self.assertGreater(env.nav.walkable_count(), 20_000)
        start = env.arena.start
        self.assertIsNotNone(env.nav.project(start))

    def test_paths_exist_to_ground_floor_spawns(self) -> None:
        env = self.env
        for name in [
            "Fountain Square",           # open terrain
            "Loading Bay",               # inside a building
            "House 1",                   # through a doorway
            "Narrow Door House",         # narrow (1.3 m) doorway
            "Behind Crates",             # around obstacles
        ]:
            with self.subTest(name=name):
                p = env.find_path(env.arena.start, _stand_point(_spawn(env, name)), 0.8)
                self.assertEqual(p.status, "ok")
                self.assertGreater(p.length, 0)
                self.assertTrue(all(abs(w[2]) < 0.3 for w in p.waypoints))

    def test_paths_climb_stairs_to_upper_floors(self) -> None:
        env = self.env
        for name in ["Mezzanine West", "Two-Storey House Upstairs", "Watchtower Platform"]:
            with self.subTest(name=name):
                sp = _spawn(env, name)
                p = env.find_path(env.arena.start, _stand_point(sp), 0.8)
                self.assertEqual(p.status, "ok")
                self.assertAlmostEqual(p.waypoints[-1][2], sp.z, delta=0.3)
                # Path length is longer than the straight-line distance (it has to go round to
                # the stairs).
                self.assertGreater(p.length, math.dist(env.arena.start[:2], (sp.x, sp.y)))

    def test_unreachable_spawns_have_no_path(self) -> None:
        env = self.env
        for name in ["Water Tower Platform", "Sealed Container"]:
            with self.subTest(name=name):
                sp = _spawn(env, name)
                self.assertIs(sp.reachable, False)
                p = env.find_path(env.arena.start, (sp.x, sp.y, sp.z), 0.5)
                self.assertNotEqual(p.status, "ok")

    def test_reachable_flags_match_the_graph(self) -> None:
        env = self.env
        # "Reachable" means a walkable cell within interaction range of the chest itself.
        for sp in env.arena.spawns:
            p = env.find_path(env.arena.start, (sp.x, sp.y, sp.z), 1.2)
            self.assertEqual(p.status == "ok", sp.reachable, sp.location_name)

    def test_off_navmesh_statuses(self) -> None:
        env = self.env
        self.assertEqual(env.find_path((-50.0, -50.0, 0.0), (40.0, 50.0, 0.0), 0.5).status,
                         "start_off_navmesh")
        self.assertEqual(env.find_path(env.arena.start, (50.0, 50.0, 40.0), 0.5).status,
                         "goal_off_navmesh")

    def test_capsule_follows_paths_through_doorways_and_up_stairs(self) -> None:
        env = self.env
        for name in ["Fountain Square", "House 1", "Narrow Door House", "Mezzanine West"]:
            with self.subTest(name=name):
                goal = _stand_point(_spawn(env, name))
                done, miss = _walk(env, goal)
                self.assertTrue(done)
                self.assertLess(miss, 1.0)
                self.assertAlmostEqual(env.character.z, goal[2], delta=0.3)

    def test_nav_raycast_stops_at_walls(self) -> None:
        env = self.env
        env.reset()
        origin = (40.0, 44.0, 0.02)                           # inside the warehouse, near a wall
        reaches = [env.nav_raycast(origin, b, 15.0) for b in (0.0, 90.0, 180.0, 270.0)]
        self.assertLess(min(reaches), 15.0)
        self.assertLessEqual(max(reaches), 15.0)

    def test_navmesh_navigator_caches_and_reports_status(self) -> None:
        env = self.env
        env.reset()
        port = EnvPort(env)
        nav = NavMeshNavigator(port)
        goal = _stand_point(_spawn(env, "Fountain Square"))
        self.assertIsNotNone(nav.plan(env.arena.start, goal))
        self.assertIsNotNone(nav.plan(env.arena.start, goal))
        self.assertEqual(nav.queries, 1)                       # second answer from the cache
        sealed = _spawn(env, "Sealed Container")
        self.assertIsNone(nav.plan(env.arena.start, (sealed.x, sealed.y, sealed.z)))
        self.assertIn(nav.last_status, {"unreachable", "goal_off_navmesh"})
        self.assertEqual(nav.failures, 1)
        # start_off_navmesh is transient: never cached.
        self.assertIsNone(nav.plan((-50.0, -50.0, 0.0), goal))
        self.assertIsNone(nav.plan((-50.0, -50.0, 0.0), goal))
        self.assertEqual(nav.queries, 4)

    def test_local_navigator_plans_without_the_navmesh_service(self) -> None:
        env = self.env
        env.reset()
        local = LocalNavigator(EnvPort(env), floors=(0.0, 3.5))
        start = _true(env)
        local.observe(start)
        self.assertEqual(local.probes, 1)
        self.assertGreater((local.grid(0.0) == 0).sum(), 100)
        plan = local.plan(start, (start[0] + 6.0, start[1] + 2.0, 0.0))
        self.assertIsNotNone(plan)
        self.assertGreater(plan.length, 0)  # type: ignore[union-attr]
        # No stairs knowledge: another floor is refused, and mid-stair starts are transient.
        self.assertIsNone(local.plan(start, (56.0, 41.0, 3.5)))
        self.assertEqual(local.last_status, "other_floor")
        self.assertIsNone(local.plan((start[0], start[1], 1.7), start))
        self.assertEqual(local.last_status, "start_off_navmesh")

    def test_local_navigator_routes_around_noted_obstacles(self) -> None:
        env = self.env
        env.reset()
        local = LocalNavigator(EnvPort(env))
        start = (30.0, 50.0, 0.0)
        local.observe(start)
        for k in range(-3, 4):                                  # a wall of noted obstacles ahead
            local.note_obstacle((start[0] + 1.0, start[1] + 0.5 * k, 0.0), 0.0)
        plan = local.plan(start, (start[0] + 5.0, start[1], 0.0))
        self.assertIsNotNone(plan)
        # Had to go round.
        self.assertGreater(plan.length, 5.0)  # type: ignore[union-attr]

    def test_stuck_detection_and_recovery(self) -> None:
        env = self.env
        env.reset()
        # Walk straight at a point behind the warehouse's south wall (x = 40, door at
        # y 48.5-51.5): the capsule slides along the wall, stops making progress -> stuck.
        target = (45.0, 42.0, 0.0)
        follower = PathFollower((target,), arrive_radius=0.6, stuck_seconds=2.0)
        follower.start(env.time)
        stuck_at = None
        while env.time < 20.0:
            bearing, speed = follower.steer(_true(env), env.time)
            fwd, right = move_towards(bearing, env.character.cam_yaw, speed)
            env.step({**IDLE_ACTION, "move_forward": fwd, "move_right": right}, 3)
            if follower.stuck(env.time):
                stuck_at = env.time
                break
        if stuck_at is None:
            self.fail("the capsule never got stuck")
        self.assertLess(env.character.x, 40.0)
        # Recovery: replanning with the NavMesh finds the way round (through the door) and
        # following the new path reaches the target.
        nav = NavMeshNavigator(EnvPort(env))
        nav.note_obstacle(_true(env), 0.0)
        plan = nav.plan(_true(env), target, tolerance=1.0)
        if plan is None:
            self.fail(f"no path round the wall: {nav.last_status}")
        self.assertGreaterEqual(len(plan.waypoints), 2)
        self.assertTrue(
            any(48.5 < w[1] < 51.5 and 39.0 < w[0] < 41.5 for w in plan.waypoints)
            or plan.length > math.dist(_true(env)[:2], target[:2]) + 2.0)
        follower = PathFollower(plan.waypoints, arrive_radius=0.6)
        follower.start(env.time)
        while env.time < stuck_at + 40.0 and not follower.done:
            bearing, speed = follower.steer(_true(env), env.time)
            fwd, right = move_towards(bearing, env.character.cam_yaw, speed)
            env.step({**IDLE_ACTION, "move_forward": fwd, "move_right": right}, 3)
        self.assertTrue(follower.done)
        self.assertLess(math.dist(_true(env)[:2], target[:2]), 1.2)


class TestScanPlanner(unittest.TestCase):
    def test_scan_planner_patterns(self) -> None:
        self.assertEqual(ScanPlanner.full_turn(30.0), [90.0, 150.0, 210.0, 270.0, 330.0, 30.0])
        arc = ScanPlanner.arc(0.0)
        self.assertTrue(all(abs((a + 180) % 360 - 180) <= 75.0 for a in arc))
        # A one-element list holding 45.0 (1e-6 relative tolerance).
        towards = ScanPlanner.towards([(1.0, 1.0, 0.0)], (0.0, 0.0, 0.0))
        self.assertIsInstance(towards, list)
        self.assertEqual(len(towards), 1)
        self.assertAlmostEqual(towards[0], 45.0, delta=1e-6 * 45.0)
