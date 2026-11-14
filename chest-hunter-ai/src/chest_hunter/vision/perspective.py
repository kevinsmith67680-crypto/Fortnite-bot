"""Pinhole-camera geometry for third-person perspective observations (Phase 4).

Coordinate conventions (identical to the Unreal research environment, but in
metres instead of centimetres):

* World axes: X north, Y east, Z up. The frame is left-handed like Unreal's,
  so yaw measured from +X towards +Y equals a compass bearing (0 = north,
  90 = east).
* Camera pose: position plus ``yaw_deg`` and ``pitch_deg`` (positive pitch
  looks up). There is no camera roll.
* Image: ``u`` to the right, ``v`` down, pixel ``i`` covers ``[i, i + 1)``;
  the principal point is the image centre.
* Field of view is horizontal (Unreal's ``FOVAngle``); pixels are square.

Everything here is pure maths: it is shared by the reference renderer, the
perception code and the evaluation code, and never touches simulator state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Projection parameters of an observation (sent with every frame)."""

    width: int
    height: int
    hfov_deg: float
    near: float = 0.1

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 9:
            raise ValueError(f"image too small: {self.width}x{self.height}")
        if not 10.0 <= self.hfov_deg <= 150.0:
            raise ValueError(f"hfov_deg must be within [10, 150], got {self.hfov_deg}")
        if self.near <= 0:
            raise ValueError("near must be positive")

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.cy / self.fy))

    def to_dict(self) -> dict[str, float | int]:
        return {"width": self.width, "height": self.height, "hfov_deg": self.hfov_deg,
                "near": self.near, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraIntrinsics:
        intr = cls(int(data["width"]), int(data["height"]), float(data["hfov_deg"]),
                   float(data.get("near", 0.1)))
        for key in ("fx", "cx", "cy"):   # consistency check of redundant fields
            if key in data and abs(float(data[key]) - getattr(intr, key)) > 1e-3:
                raise ValueError(f"inconsistent intrinsics: {key}={data[key]}")
        return intr

    def scaled(self, width: int, height: int) -> CameraIntrinsics:
        return CameraIntrinsics(width, height, self.hfov_deg, self.near)


@dataclass(frozen=True, slots=True)
class CameraPose:
    """World pose of a camera (no roll)."""

    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(forward, right, up) unit vectors in world coordinates."""
        return camera_basis(self.yaw_deg, self.pitch_deg)

    def rotation(self) -> np.ndarray:
        """3x3 matrix whose rows map world offsets to (right, up, forward)."""
        f, r, u = self.basis()
        return np.stack([r, u, f])


def camera_basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    forward = np.array([cp * cy, cp * sy, sp])
    right = np.array([-sy, cy, 0.0])
    up = np.array([-sp * cy, -sp * sy, cp])
    return forward, right, up


def forward_vector(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    return camera_basis(yaw_deg, pitch_deg)[0]


def world_to_camera(points: np.ndarray, pose: CameraPose) -> np.ndarray:
    """(..., 3) world points -> (..., 3) camera coordinates (right, up, forward)."""
    return (np.asarray(points, dtype=np.float64) - pose.position) @ pose.rotation().T


def project_camera(cam: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Camera coordinates -> pixel (u, v). Depth (``cam[..., 2]``) must be positive."""
    depth = cam[..., 2]
    u = intr.cx + intr.fx * cam[..., 0] / depth
    v = intr.cy - intr.fy * cam[..., 1] / depth
    return u, v


def project(points: np.ndarray, pose: CameraPose,
            intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (u, v, depth). Points behind the camera get depth <= 0."""
    cam = world_to_camera(points, pose)
    depth = cam[..., 2]
    safe = np.where(depth > 1e-9, depth, np.nan)
    u = intr.cx + intr.fx * cam[..., 0] / safe
    v = intr.cy - intr.fy * cam[..., 1] / safe
    return u, v, depth


def project_point(point: Vec3, pose: CameraPose,
                  intr: CameraIntrinsics) -> tuple[float, float, float] | None:
    """Single world point -> (u, v, depth), or ``None`` if behind the near plane."""
    u, v, d = project(np.asarray(point, dtype=np.float64), pose, intr)
    if not float(d) > intr.near:
        return None
    return float(u), float(v), float(d)


def pixel_ray(u: float, v: float, pose: CameraPose, intr: CameraIntrinsics) -> np.ndarray:
    """Unit world direction of the ray through pixel coordinates (u, v)."""
    f, r, up = pose.basis()
    x = (u - intr.cx) / intr.fx
    y = -(v - intr.cy) / intr.fy
    d = f + x * r + y * up
    return d / np.linalg.norm(d)


def pixel_rays(u: np.ndarray, v: np.ndarray, pose: CameraPose,
               intr: CameraIntrinsics) -> np.ndarray:
    """Vectorised :func:`pixel_ray` for arrays of pixel coordinates -> (..., 3)."""
    f, r, up = pose.basis()
    x = (np.asarray(u, dtype=np.float64) - intr.cx) / intr.fx
    y = -(np.asarray(v, dtype=np.float64) - intr.cy) / intr.fy
    d = f + x[..., None] * r + y[..., None] * up
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def intersect_horizontal_plane(origin: np.ndarray, direction: np.ndarray,
                               z: float, max_distance: float = 1e6) -> np.ndarray | None:
    """Point where a ray meets the plane ``Z = z`` in front of the origin, or ``None``."""
    dz = float(direction[2])
    if abs(dz) < 1e-9:
        return None
    t = (z - float(origin[2])) / dz
    if not 0.0 < t <= max_distance:
        return None
    return np.asarray(origin, dtype=np.float64) + t * np.asarray(direction, dtype=np.float64)


def bearing_deg(dx: float, dy: float) -> float:
    """Compass bearing (degrees, 0 = +X north, 90 = +Y east) of a horizontal offset."""
    return math.degrees(math.atan2(dy, dx)) % 360.0


def angle_difference_deg(a: float, b: float) -> float:
    """Signed smallest difference ``a - b`` in degrees, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def triangulate_rays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares closest point to several rays.

    Returns ``(point, conditioning)`` where ``conditioning`` is the smallest
    eigenvalue of the normal matrix divided by the number of rays: near 0
    when the rays are almost parallel (no usable baseline).
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions, strict=True):
        d = d / np.linalg.norm(d)
        m = np.eye(3) - np.outer(d, d)
        a += m
        b += m @ o
    eigen = float(np.linalg.eigvalsh(a)[0]) / max(1, len(origins))
    point = np.linalg.lstsq(a, b, rcond=None)[0]
    return point, eigen
