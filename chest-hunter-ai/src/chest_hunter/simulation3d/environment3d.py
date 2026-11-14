"""One episode of the 3D reference arena (ground truth lives here).

The environment advances in fixed ticks (default 1/30 s). Each ``step``
applies one validated action for a number of ticks, as the Unreal research
game mode does between bridge ``STEP`` requests.

What the agent may receive (see :meth:`Environment3D.telemetry` and
:meth:`Environment3D.observe`):

* the rendered third-person camera frame and its intrinsics,
* its own camera yaw/pitch and spring-arm length (proprioception),
* odometry: a dead-reckoned position with random-walk drift, the floor
  height, the velocity and a grounded flag,
* reward notifications for its own successful interactions.

What it never receives: chest presence, chest positions, the true pose.
Those are available only through :meth:`oracle_state` and
:meth:`evaluation_labels`, which the bridge exposes solely when the server
is started in oracle or evaluation mode.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from chest_hunter.simulation3d.arena import (
    CHEST_HEIGHT,
    Arena3D,
    build_research_arena_3d,
    chest_boxes,
)
from chest_hunter.simulation3d.character import Character3D, CharacterParams
from chest_hunter.simulation3d.geometry import Box, World3D
from chest_hunter.simulation3d.navmesh import NavGrid3D
from chest_hunter.simulation3d.renderer3d import (
    AGENT_ID,
    Billboard,
    Face,
    Renderer3D,
    box_faces,
    oriented_box_faces,
)
from chest_hunter.vision.perspective import (
    CameraIntrinsics,
    CameraPose,
    angle_difference_deg,
    bearing_deg,
)

MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    """SplitMix64 finaliser (identical in the Unreal C++ code)."""
    z = (value + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def unit_roll(seed: int, episode: int, index: int, salt: int = 0) -> float:
    """Deterministic number in [0, 1) for (seed, episode, spawn index, purpose)."""
    h = splitmix64((seed & 0xFFFFFFFF) << 32 | (episode & 0xFFFFFFFF))
    h = splitmix64(h ^ ((index & 0xFFFF) << 16 | (salt & 0xFFFF)))
    return (h >> 11) / float(1 << 53)


@lru_cache(maxsize=4)
def cached_arena(map_id: str) -> tuple[Arena3D, World3D, NavGrid3D]:
    """Static arena data is built once per process (navigation graph: ~0.4 s)."""
    from chest_hunter.simulation3d.arena import ARENA_3D_ID

    if map_id != ARENA_3D_ID:
        raise ValueError(f"unknown 3D map {map_id!r}; available: {ARENA_3D_ID}")
    arena = build_research_arena_3d()
    world = World3D(arena.boxes, bounds=arena.bounds)
    return arena, world, NavGrid3D(world)


@dataclass(frozen=True, slots=True)
class EnvSettings:
    tick_rate: int = 30
    max_ticks: int = 30 * 60 * 10
    width: int = 640
    height: int = 360
    hfov_deg: float = 90.0
    interaction_radius: float = 1.8
    interaction_cone_deg: float = 60.0
    interaction_seconds: float = 2.0
    interaction_failure_probability: float = 0.0
    reward_probability: float = 0.3
    odometry_drift: float = 0.02        # per-tick sigma = drift * distance moved that tick
    noise_std: float = 5.0
    lighting_variation: float = 0.08
    graphics_quality: str = "low"
    character: CharacterParams = field(default_factory=CharacterParams)


@dataclass(slots=True)
class Chest3D:
    spawn_index: int
    name: str
    x: float
    y: float
    z: float
    yaw_deg: float
    present: bool
    opened: bool = False
    boxes: tuple[Box, ...] = ()

    @property
    def centre(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z + CHEST_HEIGHT / 2)


class Environment3D:
    def __init__(self, map_id: str, *, seed: int, episode: int = 0,
                 settings: EnvSettings | None = None) -> None:
        self.settings = settings or EnvSettings()
        self.map_id = map_id
        self.arena, self.world, self.nav = cached_arena(map_id)
        self.seed = seed
        self.episode = episode
        self.dt = 1.0 / self.settings.tick_rate
        s = self.settings
        self.intrinsics = CameraIntrinsics(s.width, s.height, s.hfov_deg)
        self.renderer = Renderer3D(self.world, self.intrinsics, noise_std=s.noise_std,
                                   lighting_variation=s.lighting_variation,
                                   seed=seed * 7919 + episode)
        self._rng = random.Random(f"env3d:{seed}:{episode}")
        self.reset()

    # -- episode lifecycle --------------------------------------------------------------------

    def reset(self) -> None:
        s = self.settings
        a = self.arena
        self.tick = 0
        self.chests: list[Chest3D] = []
        for i, sp in enumerate(a.spawns):
            present = unit_roll(self.seed, self.episode, i) < sp.probability
            chest = Chest3D(i, sp.location_name, sp.x, sp.y, sp.z, sp.yaw_deg, present)
            chest.boxes = tuple(chest_boxes(sp.x, sp.y, sp.z, sp.yaw_deg, opened=False)) \
                if present else ()
            self.chests.append(chest)
        self.character = Character3D(*a.start, a.start_yaw_deg, s.character)
        self._odo_drift = np.zeros(2)
        self._odo_rng = np.random.default_rng(splitmix64(self.seed * 1_000_003 + self.episode))
        self.hold_target: int | None = None
        self.hold_ticks = 0
        self.reward_flash: dict[int, int] = {}
        self.events: list[dict[str, Any]] = []
        self.opened_log: list[dict[str, Any]] = []
        self.rewards = 0
        self.failed_attempts = 0
        self.frame_id = 0
        self.last_labels: dict[str, Any] | None = None
        self.stopped = False
        self.nav_blocked = self.nav.dynamic_cells(self._dynamic_boxes())

    def find_path(self, start: tuple[float, float, float], goal: tuple[float, float, float],
                  tolerance: float) -> Any:
        return self.nav.find_path(start, goal, goal_tolerance=tolerance, blocked=self.nav_blocked)

    def nav_raycast(self, origin: tuple[float, float, float], bearing: float, reach: float) -> float:
        return self.nav.raycast(origin, bearing, reach, blocked=self.nav_blocked)

    @property
    def time(self) -> float:
        return self.tick * self.dt

    @property
    def ended(self) -> bool:
        return self.stopped or self.tick >= self.settings.max_ticks

    def _dynamic_boxes(self) -> list[Box]:
        return [b for c in self.chests for b in c.boxes]

    # -- stepping ----------------------------------------------------------------------------

    def step(self, action: dict[str, Any], ticks: int) -> None:
        """Apply an already-validated action for ``ticks`` fixed ticks."""
        ch = self.character
        dynamic = self._dynamic_boxes()
        jump = bool(action.get("jump", False))
        for k in range(ticks):
            if self.ended:
                break
            ch.look(float(action.get("look_yaw", 0.0)), float(action.get("look_pitch", 0.0)),
                    self.dt)
            before = (ch.x, ch.y)
            ch.step(self.world, dynamic, float(action.get("move_forward", 0.0)),
                    float(action.get("move_right", 0.0)), jump=jump and k == 0,
                    sprint=bool(action.get("sprint", False)),
                    crouch=bool(action.get("crouch", False)), dt=self.dt)
            moved = math.hypot(ch.x - before[0], ch.y - before[1])
            if moved > 0 and self.settings.odometry_drift > 0:
                self._odo_drift += self._odo_rng.normal(0.0, self.settings.odometry_drift * moved, 2)
            self._interaction(bool(action.get("interact", False)))
            self.tick += 1
            dynamic = self._dynamic_boxes()

    def interactable_chest(self) -> Chest3D | None:
        """The chest the interaction prompt is shown for (range, facing, line of sight)."""
        ch = self.character
        s = self.settings
        best: tuple[float, Chest3D] | None = None
        for c in self.chests:
            if not c.present or c.opened:
                continue
            dist = math.hypot(c.x - ch.x, c.y - ch.y)
            if dist > s.interaction_radius or abs(c.z - ch.z) > 1.0:
                continue
            facing = abs(angle_difference_deg(bearing_deg(c.x - ch.x, c.y - ch.y), ch.cam_yaw))
            if dist > 0.5 and facing > s.interaction_cone_deg:
                continue
            eye = (ch.x, ch.y, ch.z + 1.0)
            if not self.world.line_of_sight(eye, c.centre):
                continue
            if best is None or dist < best[0]:
                best = (dist, c)
        return None if best is None else best[1]

    def _interaction(self, held: bool) -> None:
        target = self.interactable_chest() if held else None
        if target is None:
            self.hold_target, self.hold_ticks = None, 0
            return
        if target.spawn_index != self.hold_target:
            self.hold_target, self.hold_ticks = target.spawn_index, 0
        self.hold_ticks += 1
        if self.hold_ticks * self.dt + 1e-9 < self.settings.interaction_seconds:
            return
        seconds = self.hold_ticks * self.dt
        self.hold_target, self.hold_ticks = None, 0
        roll = unit_roll(self.seed, self.episode, target.spawn_index, salt=1 + self.failed_attempts)
        if roll < self.settings.interaction_failure_probability:
            self.failed_attempts += 1
            self.events.append({"type": "interaction_failed", "sim_time": round(self.time, 4)})
            return
        target.opened = True
        target.boxes = tuple(chest_boxes(target.x, target.y, target.z, target.yaw_deg, opened=True))
        rewarded = unit_roll(self.seed, self.episode, target.spawn_index, salt=999) \
            < self.settings.reward_probability
        self.rewards += int(rewarded)
        self.reward_flash[target.spawn_index] = self.tick + self.settings.tick_rate
        self.opened_log.append({"spawn_index": target.spawn_index, "sim_time": self.time,
                                "seconds": seconds, "reward": rewarded})
        # The notification is the in-game "you opened a chest" toast: it says that the
        # player's own interaction succeeded, not where any other chest is.
        self.events.append({"type": "chest_opened_notification", "reward": rewarded,
                            "sim_time": round(self.time, 4)})

    # -- agent-facing outputs ----------------------------------------------------------------

    def odometry(self) -> dict[str, Any]:
        ch = self.character
        return {"position": [round(ch.x + float(self._odo_drift[0]), 4),
                             round(ch.y + float(self._odo_drift[1]), 4), round(ch.z, 4)],
                "velocity": [round(ch.vx, 4), round(ch.vy, 4), round(ch.vz, 4)],
                "grounded": ch.grounded}

    def camera_state(self) -> dict[str, Any]:
        """The agent's own camera rig: orientation and position relative to its feet."""
        ch = self.character
        pose, arm = ch.camera(self.world)
        return {"yaw_deg": round(pose.yaw_deg, 4), "pitch_deg": round(pose.pitch_deg, 4),
                "arm_length": round(arm, 4),
                "relative_position": [round(pose.x - ch.x, 4), round(pose.y - ch.y, 4),
                                      round(pose.z - ch.z, 4)]}

    def rules(self) -> dict[str, Any]:
        """Game rules the player knows (shown in the research HUD / documentation)."""
        s, p = self.settings, self.character.p
        return {"tick_rate": s.tick_rate, "max_ticks": s.max_ticks,
                "interaction_radius": s.interaction_radius,
                "interaction_cone_deg": s.interaction_cone_deg,
                "interaction_seconds": s.interaction_seconds, "max_speed": p.max_speed,
                "sprint_speed": p.sprint_speed, "yaw_rate_deg": p.yaw_rate,
                "pitch_rate_deg": p.pitch_rate, "pitch_min_deg": p.pitch_min,
                "pitch_max_deg": p.pitch_max, "capsule_radius": p.radius,
                "capsule_height": p.height, "step_height": p.step_height}

    def telemetry(self) -> dict[str, Any]:
        return {
            "sim_time": round(self.time, 6),
            "tick": self.tick,
            "camera": self.camera_state(),
            "odometry": self.odometry(),
            "events": self._drain_events(),
        }

    def _drain_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events

    def observe(self, *, labels: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Render the camera frame. Returns (BGR frame, metadata header fields)."""
        ch = self.character
        pose, arm = ch.camera(self.world)
        faces: list[Face] = []
        for c in self.chests:
            for b in c.boxes:
                faces += box_faces(b, owner=c.spawn_index + 1)
        faces += self._character_faces()
        bills = self._billboards()
        lighting = 1.0 + self._rng.uniform(-1.0, 1.0) * self.settings.lighting_variation
        frame, ids = self.renderer.render(pose, faces, bills, labels=labels, lighting=lighting)
        self.frame_id += 1
        if labels and ids is not None:
            self.last_labels = self._labels(ids, pose)
        header = {
            "frame_id": self.frame_id,
            "sim_time": round(self.time, 6),
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
            "encoding": "bgr8",
            "intrinsics": self.intrinsics.to_dict(),
            "camera": {"yaw_deg": round(pose.yaw_deg, 4), "pitch_deg": round(pose.pitch_deg, 4),
                       "arm_length": round(arm, 4),
                       "relative_position": [round(pose.x - ch.x, 4), round(pose.y - ch.y, 4),
                                             round(pose.z - ch.z, 4)]},
            "odometry": self.odometry(),
        }
        return frame, header

    def _character_faces(self) -> list[Face]:
        ch = self.character
        z = ch.z
        faces = oriented_box_faces(ch.x, ch.y, z + 0.05, 0.56, 0.4, 1.2, ch.body_yaw,
                                   "agent_body", AGENT_ID)
        faces += oriented_box_faces(ch.x, ch.y, z + 1.27, 0.32, 0.32, 0.32, ch.body_yaw,
                                    "agent_head", AGENT_ID)
        fx, fy = math.cos(math.radians(ch.body_yaw)), math.sin(math.radians(ch.body_yaw))
        faces += oriented_box_faces(ch.x + fx * 0.17, ch.y + fy * 0.17, z + 1.4, 0.24, 0.06, 0.1,
                                    ch.body_yaw, "agent_visor", AGENT_ID)
        return faces

    def _billboards(self) -> list[Billboard]:
        out = []
        target = self.interactable_chest()
        if target is not None:
            out.append(Billboard("prompt", (target.x, target.y, target.z + CHEST_HEIGHT + 0.55), 0.32))
            if self.hold_target == target.spawn_index and self.hold_ticks:
                frac = self.hold_ticks * self.dt / self.settings.interaction_seconds
                out.append(Billboard("progress", (target.x, target.y, target.z + CHEST_HEIGHT + 0.2),
                                     0.8, frac))
        for idx, until in list(self.reward_flash.items()):
            if self.tick > until:
                del self.reward_flash[idx]
                continue
            c = self.chests[idx]
            out.append(Billboard("reward", (c.x, c.y, c.z + 1.6), 0.35))
        return out

    # -- privileged outputs (oracle / evaluation only) ------------------------------------------

    def _labels(self, ids: np.ndarray, pose: CameraPose) -> dict[str, Any]:
        chests = []
        present_ids = np.unique(ids)
        for c in self.chests:
            if not c.present:
                continue
            label = c.spawn_index + 1
            if label not in present_ids:
                continue
            ys, xs = np.nonzero(ids == label)
            dx, dy, dz = c.x - pose.x, c.y - pose.y, c.centre[2] - pose.z
            chests.append({
                "spawn_index": c.spawn_index,
                "category": "chest_opened" if c.opened else "chest",
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                         int(ys.max() - ys.min() + 1)],
                "pixels": int(xs.size),
                "position": [c.x, c.y, c.z],
                "distance": math.sqrt(dx * dx + dy * dy + dz * dz),
            })
        ch = self.character
        return {"frame_id": self.frame_id, "chests": chests,
                "camera_pose": [pose.x, pose.y, pose.z, pose.yaw_deg, pose.pitch_deg],
                "agent_position": [ch.x, ch.y, ch.z],
                "agent_pixels": int(np.count_nonzero(ids == AGENT_ID))}

    def evaluation_labels(self) -> dict[str, Any] | None:
        return self.last_labels

    def oracle_state(self) -> dict[str, Any]:
        ch = self.character
        return {
            "agent": {"position": [ch.x, ch.y, ch.z], "camera_yaw_deg": ch.cam_yaw},
            "chests": [{"spawn_index": c.spawn_index, "name": c.name, "position": [c.x, c.y, c.z],
                        "present": c.present, "opened": c.opened} for c in self.chests],
        }

    def summary(self) -> dict[str, Any]:
        return {"ticks": self.tick, "sim_time": round(self.time, 6),
                "chests_opened": len(self.opened_log), "rewards": self.rewards,
                "failed_attempts": self.failed_attempts,
                "travel_distance": round(self.character.distance_travelled, 3)}
