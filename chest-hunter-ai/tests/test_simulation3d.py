"""The Python reference arena (stand-in for Unreal on the Python side): deterministic
spawns, capsule physics, interaction rules, rendering and the camera contract."""

from __future__ import annotations

import math
import unittest

import numpy as np

from chest_hunter.simulation3d.arena import ARENA_3D_ID, build_research_arena_3d
from chest_hunter.simulation3d.environment3d import (
    Environment3D,
    EnvSettings,
    splitmix64,
    unit_roll,
)
from chest_hunter.unreal_bridge.protocol import IDLE_ACTION
from chest_hunter.vision.perspective import bearing_deg

SMALL = EnvSettings(width=320, height=180)


def _env(seed: int = 1, **kw: object) -> Environment3D:
    settings = EnvSettings(**{**{f: getattr(SMALL, f) for f in SMALL.__dataclass_fields__}, **kw})
    return Environment3D(ARENA_3D_ID, seed=seed, settings=settings)


def _act(**kw: object) -> dict[str, object]:
    return {**IDLE_ACTION, **kw}


class TestSpawnsAndLayout(unittest.TestCase):
    def test_shared_randomness_reference_vectors(self) -> None:
        # The same values are asserted by the Unreal automation test (ResearchProtocolTests.cpp).
        self.assertEqual(splitmix64(0), 0xE220A8397B1DCDAF)
        self.assertEqual(splitmix64(42), 0xBDD732262FEB6E95)
        self.assertEqual(unit_roll(42, 0, 0, 0), 0.6595024198910419)
        self.assertEqual(unit_roll(42, 0, 5, 0), 0.8423371553602427)
        self.assertEqual(unit_roll(7, 3, 12, 1), 0.8942040471837918)
        self.assertEqual(unit_roll(42, 0, 0, 999), 0.4840224294029768)

    def test_spawns_are_deterministic_per_seed_and_episode(self) -> None:
        a, b = _env(42), _env(42)
        c = Environment3D(ARENA_3D_ID, seed=42, episode=1, settings=SMALL)
        pa = [ch.present for ch in a.chests]
        self.assertEqual(pa, [ch.present for ch in b.chests])
        self.assertNotEqual(pa, [ch.present for ch in c.chests])
        self.assertEqual(pa, [unit_roll(42, 0, i) < s.probability
                              for i, s in enumerate(a.arena.spawns)])

    def test_arena_layout(self) -> None:
        arena = build_research_arena_3d()
        self.assertEqual(len(arena.spawns), 35)
        self.assertGreaterEqual({s.region for s in arena.spawns},
                                {"Central Warehouse", "Eastern Settlement", "Northern Forest",
                                 "Western Outpost", "Southern Docks"})
        self.assertEqual(sum(not s.reachable for s in arena.spawns), 2)
        self.assertLessEqual({s.yaw_deg for s in arena.spawns}, {0.0, 90.0, 180.0, 270.0})


class TestCharacterAndCamera(unittest.TestCase):
    def test_walking_moves_the_capsule_and_time_advances(self) -> None:
        env = _env()
        x0 = env.character.x
        env.step(_act(move_forward=1.0), 30)
        self.assertEqual(env.tick, 30)
        self.assertAlmostEqual(env.time, 1.0, delta=1e-6)          # 1e-6 relative tolerance
        # ~5 m/s with acceleration
        self.assertGreater(env.character.x - x0, 3.0)
        self.assertLessEqual(env.character.x - x0, 5.2)

    def test_walls_block_movement(self) -> None:
        env = _env()
        ch = env.character
        ch.x, ch.y, ch.z = 39.0, 44.0, 0.0                         # just south of the wall at x=40
        ch.cam_yaw = 0.0
        env.step(_act(move_forward=1.0), 60)
        self.assertLess(ch.x, 40.0 - 0.3)                          # capsule radius keeps it out

    def test_camera_turn_rate_and_pitch_limits(self) -> None:
        env = _env()
        yaw0 = env.character.cam_yaw
        env.step(_act(look_yaw=1.0), 15)                           # 0.5 s at 120 deg/s
        self.assertAlmostEqual((env.character.cam_yaw - yaw0) % 360, 60.0, delta=0.5)
        env.step(_act(look_pitch=-1.0), 90)
        pitch_min = env.character.p.pitch_min
        # 1e-6 relative tolerance.
        self.assertAlmostEqual(env.character.cam_pitch, pitch_min, delta=1e-6 * abs(pitch_min))

    def test_observation_is_a_camera_image_with_metadata(self) -> None:
        env = _env()
        frame, meta = env.observe()
        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(meta["encoding"], "bgr8")
        self.assertEqual(meta["frame_id"], 1)
        self.assertEqual(set(meta), {"frame_id", "sim_time", "width", "height", "encoding",
                                     "intrinsics", "camera", "odometry"})
        self.assertGreater(frame.std(), 5)                          # not a blank image
        self.assertIsNone(env.evaluation_labels())                  # labels only on request


def _face_chest(env: Environment3D, index: int, distance: float) -> None:
    c = env.chests[index]
    a = math.radians(c.yaw_deg)
    ch = env.character
    ch.x, ch.y, ch.z = c.x + distance * math.cos(a), c.y + distance * math.sin(a), c.z
    ch.cam_yaw = ch.body_yaw = bearing_deg(c.x - ch.x, c.y - ch.y)


def _present_ground_chest(env: Environment3D) -> int:
    return next(i for i, c in enumerate(env.chests)
                if c.present and c.z < 0.1 and "open" in env.arena.spawns[i].tags)


class TestInteraction(unittest.TestCase):
    def test_hold_to_open_takes_two_seconds_and_emits_a_notification(self) -> None:
        env = _env(seed=3)
        i = _present_ground_chest(env)
        _face_chest(env, i, 1.3)
        self.assertIs(env.interactable_chest(), env.chests[i])
        env.step(_act(interact=True), 50)
        self.assertFalse(env.chests[i].opened)
        env.step(_act(interact=True), 10)
        self.assertTrue(env.chests[i].opened)
        events = env.telemetry()["events"]
        self.assertEqual([e["type"] for e in events], ["chest_opened_notification"])
        self.assertNotIn("spawn_index", events[0])                  # says nothing about where

    def test_releasing_the_button_resets_the_hold(self) -> None:
        env = _env(seed=3)
        i = _present_ground_chest(env)
        _face_chest(env, i, 1.3)
        env.step(_act(interact=True), 45)
        env.step(_act(), 1)
        env.step(_act(interact=True), 45)
        self.assertFalse(env.chests[i].opened)

    def test_interaction_needs_range_and_facing(self) -> None:
        for distance, turn in [(2.6, 0.0), (1.3, 120.0)]:
            with self.subTest(distance=distance, turn=turn):
                env = _env(seed=3)
                i = _present_ground_chest(env)
                _face_chest(env, i, distance)
                env.character.cam_yaw = (env.character.cam_yaw + turn) % 360
                self.assertIsNone(env.interactable_chest())
                env.step(_act(interact=True), 70)
                self.assertFalse(env.chests[i].opened)

    def test_rewards_are_seeded_and_only_follow_openings(self) -> None:
        env = _env(seed=3, reward_probability=1.0)
        i = _present_ground_chest(env)
        _face_chest(env, i, 1.3)
        env.step(_act(interact=True), 60)
        self.assertEqual(env.rewards, 1)
        self.assertIs(env.opened_log[0]["reward"], True)
        env2 = _env(seed=3, reward_probability=0.0)
        _face_chest(env2, i, 1.3)
        env2.step(_act(interact=True), 60)
        self.assertEqual(env2.rewards, 0)
        self.assertTrue(env2.chests[i].opened)


class TestEpisodeAndOutputs(unittest.TestCase):
    def test_odometry_drifts_but_stays_close(self) -> None:
        env = _env(odometry_drift=0.02)
        for _ in range(10):
            env.step(_act(move_forward=1.0, look_yaw=0.1), 30)
        odo = env.odometry()["position"]
        true = (env.character.x, env.character.y)
        err = math.dist(odo[:2], true)
        self.assertGreater(err, 0.0)
        self.assertLess(err, 2.0)

    def test_episode_ends_at_the_tick_limit(self) -> None:
        env = _env(max_ticks=10)
        env.step(_act(move_forward=1.0), 30)
        self.assertTrue(env.ended)
        self.assertEqual(env.tick, 10)

    def test_privileged_outputs_are_separate_from_agent_outputs(self) -> None:
        env = _env()
        agent_side = str(env.telemetry()) + str(env.observe()[1]) + str(env.rules())
        for word in ("present", "spawn_index", "opened"):
            self.assertNotIn(word, agent_side)
        oracle = env.oracle_state()
        self.assertEqual(len(oracle["chests"]), 35)
        self.assertIn("present", oracle["chests"][0])

    def test_rendering_is_deterministic_for_a_seed(self) -> None:
        a, b = _env(seed=9), _env(seed=9)
        fa, _ = a.observe()
        fb, _ = b.observe()
        self.assertTrue(np.array_equal(fa, fb))
