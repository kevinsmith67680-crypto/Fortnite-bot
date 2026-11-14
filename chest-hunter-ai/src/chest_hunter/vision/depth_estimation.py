"""Estimate where a detected chest is in 3D, with explicit uncertainty (Phase 4).

Two single-frame methods, fused by inverse variance:

* **A, apparent size.** The matched reference view tells how tall (and wide)
  the chest appears in metres for that viewing angle; distance along the ray
  through the box centre is ``f * size_m / size_px``. Partially visible boxes
  use whichever dimension is complete. Measured error: 11-23 % of the
  distance, so its main job is to pick the right floor for method B.
* **B, ground-plane projection.** The ray through the bottom-centre of the box
  is intersected with candidate floor planes. Floors are not assumed to be at
  one height: every plausible floor (the agent's own, and the heights of known
  spawn points) is tried, and the hypothesis most consistent with method A is
  kept. Measured error with the right floor: 2-4 % of the distance. Its
  uncertainty grows with the square of the distance and shrinks with the
  camera's height above that floor.

A third method, **C, multi-view triangulation**, lives in the tracker: rays
from several camera positions are intersected once the baseline is large
enough (:func:`chest_hunter.vision.perspective.triangulate_rays`).

Geometrically impossible estimates (too close, beyond the useful range,
under the ground, far above the camera) are rejected, and disagreement
between A and B beyond 3 sigma marks the estimate as inconsistent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chest_hunter.vision.chest_detector_3d import Detection3D
from chest_hunter.vision.perspective import (
    CameraIntrinsics,
    CameraPose,
    bearing_deg,
    intersect_horizontal_plane,
    pixel_ray,
)

CHEST_HEIGHT = 0.65
CHEST_HALF_DEPTH = 0.3


@dataclass(frozen=True, slots=True)
class RangeEstimate:
    position: tuple[float, float, float]     # chest base centre (z = its floor height)
    distance: float                          # camera to chest centre, metres
    sigma: float                             # 1-sigma position uncertainty, metres
    method: str                              # "size", "ground" or "fused"
    bearing_deg: float
    elevation_deg: float                     # angle below the horizon (positive = looking down)
    consistent: bool = True
    ray_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ray_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)


class DistanceEstimator:
    def __init__(self, *, max_range: float = 40.0, min_range: float = 0.6,
                 pixel_sigma: float = 1.0, size_rel_sigma: float = 0.20) -> None:
        self.max_range = max_range
        self.min_range = min_range
        self.pixel_sigma = pixel_sigma
        self.size_rel_sigma = size_rel_sigma
        self.estimates = 0
        self.rejected = 0
        self.inconsistent = 0

    def estimate(self, det: Detection3D, pose: CameraPose, intr: CameraIntrinsics,
                 floors: Sequence[float] = (0.0,), *,
                 anchors: Sequence[tuple[float, float, float]] = (),
                 own_floor: float | None = None) -> RangeEstimate | None:
        """``floors``: candidate floor heights. ``anchors``: known spawn points (x, y, z);
        a floor hypothesis is plausible when it is the agent's own floor or matches the
        height of a spawn point near the projected position."""
        self.estimates += 1
        cam = pose.position
        uc, vc = det.centre
        ray = pixel_ray(uc, vc, pose, intr)
        elevation = -math.degrees(math.asin(max(-1.0, min(1.0, float(ray[2])))))
        x, y, w, h = det.bbox

        # Method A: apparent size along the centre ray.
        height_ok = not det.partial or (y > 1 and y + h < intr.height - 1)
        if height_ok and h >= 3:
            d_a = intr.fy * det.height_m / h
            rel = self.size_rel_sigma + self.pixel_sigma / h
        else:
            d_a = intr.fx * det.width_m / max(1, w)
            rel = self.size_rel_sigma * 1.6 + self.pixel_sigma / max(1, w)
        if det.partial:
            rel *= 1.8
        sigma_a = max(0.08, d_a * rel)
        centre_a = cam + d_a * ray
        base_a = centre_a - np.array([0.0, 0.0, CHEST_HEIGHT / 2])

        # Method B: ground-plane intersection of the bottom-centre ray, per floor hypothesis.
        best_b: tuple[float, np.ndarray, float] | None = None       # (score, base, sigma)
        if not det.partial or y + h < intr.height - 1:
            ub, vb = det.bottom_centre
            ray_b = pixel_ray(ub, vb - 0.5, pose, intr)
            horizontal = np.array([ray_b[0], ray_b[1], 0.0])
            norm = float(np.linalg.norm(horizontal))
            for z in sorted(set(round(f, 2) for f in floors)):
                height_above = float(cam[2]) - z
                if height_above < 0.25:
                    continue
                hit = intersect_horizontal_plane(cam, ray_b, z, max_distance=self.max_range * 1.5)
                if hit is None or norm < 1e-6:
                    continue
                base_b = hit + horizontal / norm * CHEST_HALF_DEPTH
                d_b = float(np.linalg.norm(base_b + np.array([0, 0, CHEST_HEIGHT / 2]) - cam))
                sigma_b = (0.12 + 0.02 * d_b
                           + d_b * d_b * (1.5 * self.pixel_sigma / intr.fy) / max(0.3, height_above))
                score = abs(d_b - d_a) / math.hypot(sigma_a, sigma_b)
                if not _plausible(base_b, z, anchors, own_floor):
                    score += 2.5
                if best_b is None or score < best_b[0]:
                    best_b = (score, base_b, sigma_b)

        consistent = True
        if best_b is not None and best_b[0] <= 5.5:
            _, base_b, sigma_b = best_b
            wa, wb = 1.0 / sigma_a ** 2, 1.0 / sigma_b ** 2
            base = (base_a * wa + base_b * wb) / (wa + wb)
            sigma = math.sqrt(1.0 / (wa + wb))
            method = "fused"
        else:
            if best_b is not None:
                consistent = False
                self.inconsistent += 1
            base, sigma, method = base_a, sigma_a * (1.5 if not consistent else 1.0), "size"
        centre = base + np.array([0.0, 0.0, CHEST_HEIGHT / 2])
        distance = float(np.linalg.norm(centre - cam))
        if not self.min_range <= distance <= self.max_range or base[2] < -0.5 \
                or base[2] > float(cam[2]) + 6.0:
            self.rejected += 1
            return None
        return RangeEstimate(
            position=(float(base[0]), float(base[1]), float(base[2])), distance=distance,
            sigma=float(sigma), method=method,
            bearing_deg=bearing_deg(float(base[0] - cam[0]), float(base[1] - cam[1])),
            elevation_deg=elevation, consistent=consistent,
            ray_origin=(float(cam[0]), float(cam[1]), float(cam[2])),
            ray_direction=(float(ray[0]), float(ray[1]), float(ray[2])))


def _plausible(base: np.ndarray, z: float, anchors: Sequence[tuple[float, float, float]],
               own_floor: float | None) -> bool:
    if own_floor is not None and abs(z - own_floor) <= 0.15:
        return True
    for ax, ay, az in anchors:
        if abs(az - z) <= 0.15 and (ax - base[0]) ** 2 + (ay - base[1]) ** 2 <= 9.0:
            return True
    return not anchors and own_floor is None
