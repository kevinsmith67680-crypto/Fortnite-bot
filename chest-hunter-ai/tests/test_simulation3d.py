"""The Python reference arena (stand-in for Unreal on the Python side): deterministic
spawns, capsule physics, interaction rules, rendering and the camera contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

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


def test_shared_randomness_reference_vectors() -> None:
    # The same values are asserted by the Unreal automation test (ResearchProtocolTests.cpp).
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert splitmix64(42) == 0xBDD732262FEB6E95
    assert unit_roll(42, 0, 0, 0) == 0.6595024198910419
    assert unit_roll(42, 0, 5, 0) == 0.8423371553602427
    assert unit_roll(7, 3, 12, 1) == 0.8942040471837918
    assert unit_roll(42, 0, 0, 999) == 0.4840224294029768


def test_spawns_are_deterministic_per_seed_and_episode() -> None:
    a, b, c = _env(42), _env(42), Environment3D(ARENA_3D_ID, seed=42, episode=1, settings=SMALL)
    pa = [ch.present for ch in a.chests]
    assert pa == [ch.present for ch in b.chests]
    assert pa != [ch.present for ch in c.chests]
    assert pa == [unit_roll(42, 0, i) < s.probability for i, s in enumerate(a.arena.spawns)]


def test_arena_layout() -> None:
    arena = build_research_arena_3d()
    assert len(arena.spawns) == 35
    assert {s.region for s in arena.spawns} >= {"Central Warehouse", "Eastern Settlement",
                                                 "Northern Forest", "Western Outpost",
                                                 "Southern Docks"}
    assert sum(not s.reachable for s in arena.spawns) == 2
    assert {s.yaw_deg for s in arena.spawns} <= {0.0, 90.0, 180.0, 270.0}


def test_walking_moves_the_capsule_and_time_advances() -> None:
    env = _env()
    x0 = env.character.x
    env.step(_act(move_forward=1.0), 30)
    assert env.tick == 30 and env.time == pytest.approx(1.0)
    assert 3.0 < env.character.x - x0 <= 5.2                   # ~5 m/s with acceleration


def test_walls_block_movement() -> None:
    env = _env()
    ch = env.character
    ch.x, ch.y, ch.z = 39.0, 44.0, 0.0                         # just south of the wall at x=40
    ch.cam_yaw = 0.0
    env.step(_act(move_forward=1.0), 60)
    assert ch.x < 40.0 - 0.3                                   # capsule radius keeps it out


def test_camera_turn_rate_and_pitch_limits() -> None:
    env = _env()
    yaw0 = env.character.cam_yaw
    env.step(_act(look_yaw=1.0), 15)                           # 0.5 s at 120 deg/s
    assert (env.character.cam_yaw - yaw0) % 360 == pytest.approx(60.0, abs=0.5)
    env.step(_act(look_pitch=-1.0), 90)
    assert env.character.cam_pitch == pytest.approx(env.character.p.pitch_min)


def test_observation_is_a_camera_image_with_metadata() -> None:
    env = _env()
    frame, meta = env.observe()
    assert frame.shape == (180, 320, 3) and frame.dtype == np.uint8
    assert meta["encoding"] == "bgr8" and meta["frame_id"] == 1
    assert set(meta) == {"frame_id", "sim_time", "width", "height", "encoding", "intrinsics",
                         "camera", "odometry"}
    assert frame.std() > 5                                      # not a blank image
    assert env.evaluation_labels() is None                      # labels only on request


def _face_chest(env: Environment3D, index: int, distance: float) -> None:
    c = env.chests[index]
    a = math.radians(c.yaw_deg)
    ch = env.character
    ch.x, ch.y, ch.z = c.x + distance * math.cos(a), c.y + distance * math.sin(a), c.z
    ch.cam_yaw = ch.body_yaw = bearing_deg(c.x - ch.x, c.y - ch.y)


def _present_ground_chest(env: Environment3D) -> int:
    return next(i for i, c in enumerate(env.chests)
                if c.present and c.z < 0.1 and "open" in env.arena.spawns[i].tags)


def test_hold_to_open_takes_two_seconds_and_emits_a_notification() -> None:
    env = _env(seed=3)
    i = _present_ground_chest(env)
    _face_chest(env, i, 1.3)
    assert env.interactable_chest() is env.chests[i]
    env.step(_act(interact=True), 50)
    assert not env.chests[i].opened
    env.step(_act(interact=True), 10)
    assert env.chests[i].opened
    events = env.telemetry()["events"]
    assert [e["type"] for e in events] == ["chest_opened_notification"]
    assert "spawn_index" not in events[0]                       # says nothing about where


def test_releasing_the_button_resets_the_hold() -> None:
    env = _env(seed=3)
    i = _present_ground_chest(env)
    _face_chest(env, i, 1.3)
    env.step(_act(interact=True), 45)
    env.step(_act(), 1)
    env.step(_act(interact=True), 45)
    assert not env.chests[i].opened


@pytest.mark.parametrize(("distance", "turn"), [(2.6, 0.0), (1.3, 120.0)])
def test_interaction_needs_range_and_facing(distance: float, turn: float) -> None:
    env = _env(seed=3)
    i = _present_ground_chest(env)
    _face_chest(env, i, distance)
    env.character.cam_yaw = (env.character.cam_yaw + turn) % 360
    assert env.interactable_chest() is None
    env.step(_act(interact=True), 70)
    assert not env.chests[i].opened


def test_rewards_are_seeded_and_only_follow_openings() -> None:
    env = _env(seed=3, reward_probability=1.0)
    i = _present_ground_chest(env)
    _face_chest(env, i, 1.3)
    env.step(_act(interact=True), 60)
    assert env.rewards == 1 and env.opened_log[0]["reward"] is True
    env2 = _env(seed=3, reward_probability=0.0)
    _face_chest(env2, i, 1.3)
    env2.step(_act(interact=True), 60)
    assert env2.rewards == 0 and env2.chests[i].opened


def test_odometry_drifts_but_stays_close() -> None:
    env = _env(odometry_drift=0.02)
    for _ in range(10):
        env.step(_act(move_forward=1.0, look_yaw=0.1), 30)
    odo = env.odometry()["position"]
    true = (env.character.x, env.character.y)
    err = math.dist(odo[:2], true)
    assert 0.0 < err < 2.0


def test_episode_ends_at_the_tick_limit() -> None:
    env = _env(max_ticks=10)
    env.step(_act(move_forward=1.0), 30)
    assert env.ended and env.tick == 10


def test_privileged_outputs_are_separate_from_agent_outputs() -> None:
    env = _env()
    agent_side = str(env.telemetry()) + str(env.observe()[1]) + str(env.rules())
    for word in ("present", "spawn_index", "opened"):
        assert word not in agent_side
    oracle = env.oracle_state()
    assert len(oracle["chests"]) == 35 and "present" in oracle["chests"][0]


def test_rendering_is_deterministic_for_a_seed() -> None:
    a, b = _env(seed=9), _env(seed=9)
    fa, _ = a.observe()
    fb, _ = b.observe()
    assert np.array_equal(fa, fb)
