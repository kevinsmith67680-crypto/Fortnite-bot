"""Tests for map generation, world building, chest spawning and interaction rules."""

from __future__ import annotations

import unittest
from dataclasses import replace

from chest_hunter.config import SimulationSettings
from chest_hunter.models import ChestLocation
from chest_hunter.navigation.astar import distance_field
from chest_hunter.simulation.collision import GridMap, circle_collides
from chest_hunter.simulation.entities import Chest, Obstacle
from chest_hunter.simulation.environment import (
    ChestSpec,
    Environment,
    InteractionResult,
    TokenRewardModel,
    World,
    WorldTransform,
    build_world,
)
from chest_hunter.simulation.map_generator import MapGenerator

NOW = "2026-01-01T00:00:00.000000+00:00"
SIM = SimulationSettings(default_map="arena", default_spawn_probability=0.8,
                         default_open_seconds=2.0)


def location(chest_id: int, x: float, y: float, p: float = 0.8,
             enabled: bool = True) -> ChestLocation:
    return ChestLocation(chest_id, "arena", f"Spot {chest_id}", x, y, 0.0, p, 2.0, enabled,
                         NOW, NOW)


def open_world(chests: list[ChestSpec], size: int = 20, start: tuple[float, float] = (1.5, 1.5),
               grid: GridMap | None = None) -> World:
    return World("arena", 0, grid or GridMap(size, size), (), tuple(chests), start,
                 WorldTransform(1.0))


class MapGeneratorTests(unittest.TestCase):
    def test_generation_is_deterministic_per_seed(self) -> None:
        a = MapGenerator(100, 100, density=0.12, seed=42).generate([(50, 50)])
        b = MapGenerator(100, 100, density=0.12, seed=42).generate([(50, 50)])
        c = MapGenerator(100, 100, density=0.12, seed=43).generate([(50, 50)])
        self.assertEqual(a.grid, b.grid)
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertNotEqual(a.grid, c.grid)

    def test_density_border_and_protected_cells(self) -> None:
        protected = [(10, 10), (50, 50), (90, 90)]
        generated = MapGenerator(100, 100, density=0.12, seed=1).generate(protected)
        grid = generated.grid
        self.assertAlmostEqual(generated.density, 0.12, delta=0.02)
        for x in range(100):
            self.assertFalse(grid.is_blocked(x, 0))
            self.assertFalse(grid.is_blocked(x, 99))
        for cx, cy in protected:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    self.assertFalse(grid.is_blocked(cx + dx, cy + dy))
        kinds = {o.kind for o in generated.obstacles}
        self.assertEqual(kinds, {"building", "wall", "rock"})

    def test_obstacle_rectangles_match_the_grid(self) -> None:
        generated = MapGenerator(60, 60, density=0.15, seed=9).generate()
        covered = {cell for o in generated.obstacles for cell in o.cells()}
        blocked = {(x, y) for x in range(60) for y in range(60) if generated.grid.is_blocked(x, y)}
        self.assertEqual(covered, blocked)

    def test_zero_density_and_invalid_arguments(self) -> None:
        self.assertEqual(MapGenerator(20, 20, density=0.0, seed=1).generate().grid.blocked_count, 0)
        with self.assertRaises(ValueError):
            MapGenerator(5, 20, density=0.1, seed=1)
        with self.assertRaises(ValueError):
            MapGenerator(20, 20, density=0.9, seed=1)


class WorldTests(unittest.TestCase):
    def test_transform_round_trip(self) -> None:
        transform = WorldTransform(10.0, origin_x=5.0)
        self.assertEqual(transform.to_grid(105.0, 50.0), (10.0, 5.0))
        self.assertEqual(transform.to_world(10.0, 5.0), (105.0, 50.0))

    def test_build_world_places_chests_and_excludes_invalid_ones(self) -> None:
        locations = [location(1, 120, 340), location(2, 990, 990), location(3, 1500, 10),
                     location(4, 500, 500, enabled=False)]
        world = build_world(locations, replace(SIM, obstacle_density=0.15),
                            map_id="arena", map_seed=4)
        self.assertEqual([c.location_id for c in world.chests], [1, 2])
        self.assertEqual(world.excluded_location_ids, (3, 4))
        self.assertEqual(world.chests[0].cell, (12, 34))
        self.assertEqual(world.start, (50.5, 50.5))
        self.assertFalse(circle_collides(world.grid, *world.start, SIM.player_radius))
        for chest in world.chests:
            self.assertFalse(world.grid.is_blocked(*chest.cell))

    def test_configured_start_position_is_used(self) -> None:
        world = build_world([location(1, 120, 340)], replace(SIM, agent_start=(205.0, 315.0)),
                            map_id="arena", map_seed=1)
        self.assertEqual(world.start_cell, (20, 31))

    def test_demo_like_world_chests_are_reachable(self) -> None:
        locations = [location(i + 1, 100 + 80 * (i % 10), 100 + 200 * (i // 10)) for i in range(40)]
        world = build_world(locations, SIM, map_id="arena", map_seed=42)
        field = distance_field(world.grid, world.start_cell)
        reachable = sum(1 for c in world.chests
                        if field[c.cell[1] * world.width + c.cell[0]] < float("inf"))
        self.assertGreaterEqual(reachable, len(world.chests) - 2)


class SpawnTests(unittest.TestCase):
    def specs(self, count: int, p: float) -> list[ChestSpec]:
        return [ChestSpec(i + 1, f"c{i}", 2.5 + (i % 15), 2.5 + (i // 15), p, 2.0)
                for i in range(count)]

    def test_spawning_is_deterministic_and_episode_dependent(self) -> None:
        world = open_world(self.specs(30, 0.5))
        first = Environment(world, SIM, seed=42, episode=0)
        again = Environment(world, SIM, seed=42, episode=0)
        pattern = [c.present for c in first.chests.values()]
        self.assertEqual(pattern, [c.present for c in again.chests.values()])
        others = {tuple(c.present for c in Environment(world, SIM, seed=42, episode=e)
                        .chests.values()) for e in range(1, 6)}
        self.assertTrue(any(p != tuple(pattern) for p in others))

    def test_spawn_does_not_depend_on_chest_order(self) -> None:
        specs = self.specs(10, 0.5)
        forward = Environment(open_world(specs), SIM, seed=7)
        backward = Environment(open_world(list(reversed(specs))), SIM, seed=7)
        self.assertEqual({c.id: c.present for c in forward.chests.values()},
                         {c.id: c.present for c in backward.chests.values()})

    def test_spawn_probabilities_are_respected(self) -> None:
        world = open_world(self.specs(200, 0.3))
        samples = [c.present for e in range(25)
                   for c in Environment(world, SIM, seed=1, episode=e).chests.values()]
        self.assertAlmostEqual(sum(samples) / len(samples), 0.3, delta=0.03)
        never = Environment(open_world(self.specs(50, 0.0)), SIM, seed=1)
        always = Environment(open_world(self.specs(50, 1.0)), SIM, seed=1)
        self.assertEqual(never.present_count, 0)
        self.assertEqual(always.present_count, 50)

    def test_episode_reset_restores_initial_state(self) -> None:
        env = Environment(open_world(self.specs(5, 1.0)), SIM, seed=3)
        env.chests[1].open()
        env.agent.position_x = 9.0
        env.advance()
        env.reset()
        self.assertEqual(env.tick, 0)
        self.assertEqual(env.agent.position, env.world.start)
        self.assertFalse(any(c.opened for c in env.chests.values()))
        self.assertEqual(env.present_count, 5)

    def test_unwalkable_start_is_rejected(self) -> None:
        grid = GridMap(10, 10)
        grid.set_blocked(1, 1)
        with self.assertRaises(ValueError):
            Environment(open_world([], grid=grid), SIM, seed=1)


class InteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        # A 3-cell interaction radius leaves room for a wall between agent and chest.
        self.sim = replace(SIM, interaction_radius=3.0)
        specs = [ChestSpec(1, "present", 3.5, 1.5, 1.0, 2.0),
                 ChestSpec(2, "absent", 1.5, 3.5, 0.0, 2.0),
                 ChestSpec(3, "far", 15.5, 15.5, 1.0, 2.0)]
        self.env = Environment(open_world(specs), self.sim, seed=1)

    def test_open_rules(self) -> None:
        env = self.env
        self.assertEqual(env.try_open(1, 0.5), InteractionResult.INCOMPLETE)
        self.assertEqual(env.try_open(3, 5.0), InteractionResult.OUT_OF_RANGE)
        self.assertEqual(env.try_open(2, 5.0), InteractionResult.NOT_PRESENT)
        self.assertEqual(env.try_open(99, 5.0), InteractionResult.UNKNOWN_CHEST)
        self.assertEqual(env.try_open(1, 2.0), InteractionResult.OPENED)
        self.assertTrue(env.chests[1].opened)
        self.assertEqual(env.try_open(1, 2.0), InteractionResult.ALREADY_OPENED)
        self.assertEqual(env.opened_count, 1)

    def test_observation_requires_range_and_line_of_sight(self) -> None:
        env = self.env
        self.assertTrue(env.observe_chest(1))
        self.assertFalse(env.observe_chest(2))
        self.assertIsNone(env.observe_chest(3))
        env.add_obstacle(Obstacle(2, 1, 1, 1))  # wall between agent and chest 1
        self.assertIsNone(env.observe_chest(1))
        self.assertEqual(env.try_open(1, 5.0), InteractionResult.OUT_OF_RANGE)

    def test_add_obstacle_is_copy_on_write(self) -> None:
        world_grid = self.env.world.grid
        self.env.add_obstacle(Obstacle(10, 10, 2, 2))
        self.assertIsNot(self.env.grid, world_grid)
        self.assertFalse(world_grid.is_blocked(10, 10))
        self.assertTrue(self.env.grid.is_blocked(11, 11))
        self.env.reset()
        self.assertIs(self.env.grid, world_grid)

    def test_interaction_failures_are_deterministic(self) -> None:
        sim = replace(self.sim, interaction_failure_probability=1.0)
        env = Environment(self.env.world, sim, seed=1)
        self.assertEqual(env.try_open(1, 2.0), InteractionResult.FAILED)
        self.assertFalse(env.chests[1].opened)
        sim = replace(self.sim, interaction_failure_probability=0.5)
        outcomes = [[Environment(self.env.world, sim, seed=9).try_open(1, 2.0)]
                    for _ in range(3)]
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[1], outcomes[2])


class RewardTests(unittest.TestCase):
    def test_reward_model_is_separate_and_deterministic(self) -> None:
        self.assertFalse(TokenRewardModel(0.0, seed=1, episode=0).roll(5))
        self.assertTrue(TokenRewardModel(1.0, seed=1, episode=0).roll(5))
        model_a = TokenRewardModel(0.1, seed=3, episode=2)
        model_b = TokenRewardModel(0.1, seed=3, episode=2)
        rolls_a = [model_a.roll(i) for i in range(200)]
        self.assertEqual(rolls_a, [model_b.roll(i) for i in range(200)])
        self.assertTrue(5 <= sum(rolls_a) <= 40)
        with self.assertRaises(ValueError):
            TokenRewardModel(1.5, seed=1, episode=0)

    def test_rewards_only_for_opened_chests(self) -> None:
        world = open_world([ChestSpec(1, "c", 2.5, 1.5, 1.0, 2.0)])
        env = Environment(world, replace(SIM, reward_probability=1.0), seed=1)
        self.assertFalse(env.grant_reward(1))  # not opened yet
        self.assertEqual(env.try_open(1, 2.0), InteractionResult.OPENED)
        self.assertTrue(env.grant_reward(1))
        self.assertEqual(env.rewards.awarded, [1])


class EntityTests(unittest.TestCase):
    def test_chest_open_rules(self) -> None:
        chest = Chest(1, 0.0, 0.0, 0.5)
        self.assertFalse(chest.open())
        chest.reset(True)
        self.assertTrue(chest.available)
        self.assertTrue(chest.open())
        self.assertFalse(chest.open())

    def test_agent_respects_speed_and_acceleration_limits(self) -> None:
        env = Environment(open_world([]), SIM, seed=1)
        agent = env.agent
        previous = 0.0
        for _ in range(60):
            agent.steer(100.0, 0.0, env.dt)
            agent.step(env.grid, env.dt)
            self.assertLessEqual(agent.speed, agent.max_speed + 1e-9)
            self.assertLessEqual(agent.speed - previous, agent.acceleration * env.dt + 1e-9)
            previous = agent.speed
        self.assertAlmostEqual(agent.speed, agent.max_speed)
        self.assertGreater(agent.distance_travelled, 0)


if __name__ == "__main__":
    unittest.main()
