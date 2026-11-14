"""Decode bridge observations and telemetry into typed Python objects.

The camera frame arrives as raw ``bgr8`` bytes in the message payload and is
wrapped as a read-only NumPy view (no copy, no conversion). The header gives
the frame id, simulated timestamp, camera intrinsics, the camera rig state
and odometry. Nothing else is ever exposed to the agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from chest_hunter.unreal_bridge.protocol import SUPPORTED_SIZES, ProtocolError
from chest_hunter.vision.perspective import CameraIntrinsics, CameraPose


@dataclass(frozen=True, slots=True)
class CameraState:
    """The agent's own camera rig (proprioception): orientation and offset from its feet."""

    yaw_deg: float
    pitch_deg: float
    arm_length: float
    relative_position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Odometry:
    """Dead-reckoned position (drifts slowly), floor height, velocity and ground contact."""

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    grounded: bool

    @property
    def speed(self) -> float:
        return math.hypot(self.velocity[0], self.velocity[1])


@dataclass(frozen=True, slots=True)
class Telemetry:
    sim_time: float
    tick: int
    camera: CameraState
    odometry: Odometry
    events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class Observation3D:
    frame: np.ndarray                 # (height, width, 3) uint8 BGR, read-only
    frame_id: int
    episode_id: str
    sim_time: float
    intrinsics: CameraIntrinsics
    camera: CameraState
    odometry: Odometry

    @property
    def width(self) -> int:
        return self.intrinsics.width

    @property
    def height(self) -> int:
        return self.intrinsics.height

    def camera_pose(self) -> CameraPose:
        """Estimated world pose of the camera: odometry position plus the rig offset."""
        px, py, pz = self.odometry.position
        rx, ry, rz = self.camera.relative_position
        return CameraPose(px + rx, py + ry, pz + rz, self.camera.yaw_deg, self.camera.pitch_deg)


def _vec(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProtocolError("bad_header", f"{name} must be a 3-vector")
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ProtocolError("bad_header", f"{name} must be finite")
    return out[0], out[1], out[2]


def decode_camera(data: dict[str, Any]) -> CameraState:
    return CameraState(float(data["yaw_deg"]), float(data["pitch_deg"]),
                       float(data.get("arm_length", 0.0)),
                       _vec(data["relative_position"], "camera.relative_position"))


def decode_odometry(data: dict[str, Any]) -> Odometry:
    return Odometry(_vec(data["position"], "odometry.position"),
                    _vec(data["velocity"], "odometry.velocity"), bool(data["grounded"]))


def decode_telemetry(data: dict[str, Any]) -> Telemetry:
    try:
        return Telemetry(float(data["sim_time"]), int(data["tick"]), decode_camera(data["camera"]),
                         decode_odometry(data["odometry"]),
                         tuple(e for e in data.get("events", ()) if isinstance(e, dict)))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("bad_header", f"malformed telemetry: {exc}") from exc


def decode_observation(meta: dict[str, Any], payload: bytes, episode_id: str) -> Observation3D:
    """Validate an observation header against its payload and wrap the pixels."""
    try:
        width, height = int(meta["width"]), int(meta["height"])
        if (width, height) not in SUPPORTED_SIZES:
            raise ProtocolError("bad_header", f"unsupported observation size {width}x{height}")
        if meta.get("encoding") != "bgr8":
            raise ProtocolError("bad_header", f"unsupported encoding {meta.get('encoding')!r}")
        if len(payload) != width * height * 3:
            raise ProtocolError("bad_header", f"payload has {len(payload)} bytes, expected "
                                              f"{width * height * 3} for {width}x{height} bgr8")
        intr = CameraIntrinsics.from_dict(meta["intrinsics"])
        if (intr.width, intr.height) != (width, height):
            raise ProtocolError("bad_header", "intrinsics do not match the frame size")
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
        return Observation3D(frame=frame, frame_id=int(meta["frame_id"]), episode_id=episode_id,
                             sim_time=float(meta["sim_time"]), intrinsics=intr,
                             camera=decode_camera(meta["camera"]),
                             odometry=decode_odometry(meta["odometry"]))
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("bad_header", f"malformed observation header: {exc}") from exc
