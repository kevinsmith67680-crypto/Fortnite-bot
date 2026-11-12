"""Visual self-localisation.

1. :class:`AgentMarkerDetector` finds the agent's marker (blue disc with a
   white heading "nose") in a frame. Its sub-pixel centroid (image moments)
   is converted to world coordinates with the observation's camera geometry,
   which accounts for the moving camera's translation and scale.
2. :class:`PoseEstimator` fuses measurements over time:

   * **predict**: dead reckoning from the agent's own *commanded* velocity
     (its intended movement; no access to the true position);
   * **correct**: ``estimate += alpha * (measurement - estimate)`` (exponential
     smoothing weighted by ``smoothing_alpha``). If the fix disagrees with the
     prediction by more than ``RESET_DISTANCE`` cells, the prediction is what
     failed (typically the body was blocked by a wall while the commanded
     velocity kept integrating; marker fixes are accurate to ~0.01 cell), so the
     estimate jumps to the measurement instead of being smoothed towards it;
   * **confidence** = 1 right after a good fix, decaying linearly to 0 over
     ``dead_reckoning_seconds`` without fixes. After
     ``max_missing_observations`` consecutive failed fixes the pose is
     considered lost.

A Kalman filter was not needed: the marker is measured to about a tenth of a
cell, and prediction from commanded motion keeps the estimate from lagging
while moving (tested).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from chest_hunter.vision.preprocessing import PreparedFrame


@dataclass(frozen=True, slots=True)
class PoseMeasurement:
    x: float
    y: float
    heading: float | None
    confidence: float
    timestamp: float
    pixel: tuple[float, float]


class AgentMarkerDetector:
    """Finds the agent marker; optionally searches a region around a hint first."""

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0
        self.total_seconds = 0.0

    def detect(self, frame: PreparedFrame,
               hint_pixel: tuple[float, float] | None = None) -> PoseMeasurement | None:
        started = time.perf_counter()
        try:
            measurement = None
            if hint_pixel is not None:
                measurement = self._search(frame, hint_pixel, radius_cells=3.0)
            if measurement is None:
                measurement = self._search(frame, None, radius_cells=0.0)
            if measurement is None:
                self.failures += 1
            return measurement
        finally:
            self.calls += 1
            self.total_seconds += time.perf_counter() - started

    def _search(self, frame: PreparedFrame, hint: tuple[float, float] | None,
                radius_cells: float) -> PoseMeasurement | None:
        ppc = frame.pixels_per_cell
        img_h, img_w = frame.shape
        x0 = y0 = 0
        x1, y1 = img_w, img_h
        if hint is not None:
            hu = (hint[0] - frame.offset[0]) * frame.scale
            hv = (hint[1] - frame.offset[1]) * frame.scale
            r = radius_cells * ppc
            x0, y0 = max(0, int(hu - r)), max(0, int(hv - r))
            x1, y1 = min(img_w, int(hu + r) + 1), min(img_h, int(hv + r) + 1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                return None
        blue = frame.mask("agent")[y0:y1, x0:x1]
        contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected = math.pi * frame.library.size("agent_radius", ppc) ** 2
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.35 * expected <= area <= 2.0 * expected:
                score = 1.0 - min(1.0, abs(1.0 - area / expected))
                if best is None or score > best[0]:
                    best = (score, contour)
        if best is None:
            return None
        bx, by, bw, bh = cv2.boundingRect(best[1])
        disc = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(disc, [best[1]], -1, 255, -1, offset=(-bx, -by))
        nose = cv2.bitwise_and(frame.mask("agent_nose")[y0 + by:y0 + by + bh, x0 + bx:x0 + bx + bw],
                               cv2.dilate(disc, np.ones((3, 3), np.uint8)))
        combined = cv2.bitwise_or(blue[by:by + bh, bx:bx + bw], nose)
        moments = cv2.moments(combined, binaryImage=True)
        if moments["m00"] <= 0:
            return None
        cu = x0 + bx + moments["m10"] / moments["m00"] + 0.5
        cv_ = y0 + by + moments["m01"] / moments["m00"] + 0.5
        heading = None
        nose_moments = cv2.moments(nose, binaryImage=True)
        if nose_moments["m00"] > 0:
            nu = x0 + bx + nose_moments["m10"] / nose_moments["m00"] + 0.5
            nv = y0 + by + nose_moments["m01"] / nose_moments["m00"] + 0.5
            if math.hypot(nu - cu, nv - cv_) > 0.05 * ppc:
                heading = math.atan2(-(nv - cv_), nu - cu)
        u, v = frame.to_original(cu, cv_)
        x, y = frame.observation.camera.pixel_to_world(u, v)
        return PoseMeasurement(x, y, heading, float(np.clip(best[0] + 0.2, 0.0, 1.0)),
                               frame.observation.timestamp, (u, v))


class PoseEstimator:
    """Fuses commanded-motion prediction with visual fixes."""

    RESET_DISTANCE = 0.5   # cells of disagreement treated as a prediction failure

    def __init__(self, *, smoothing_alpha: float = 0.4, max_missing_observations: int = 5,
                 dead_reckoning_seconds: float = 2.0) -> None:
        self.alpha = smoothing_alpha
        self.max_missing = max_missing_observations
        self.dead_reckoning_seconds = dead_reckoning_seconds
        self.x: float | None = None
        self.y: float | None = None
        self.heading: float | None = None
        self.last_fix_time = -math.inf
        self.fix_confidence = 0.0
        self.missing = 0
        self.fixes = 0
        self.resets = 0
        self.time = 0.0

    @property
    def initialized(self) -> bool:
        return self.x is not None

    @property
    def position(self) -> tuple[float, float] | None:
        if self.x is None or self.y is None:
            return None
        return (self.x, self.y)

    @property
    def is_lost(self) -> bool:
        return not self.initialized or self.missing > self.max_missing

    def confidence(self, now: float | None = None) -> float:
        if not self.initialized:
            return 0.0
        now = self.time if now is None else now
        decay = max(0.0, 1.0 - (now - self.last_fix_time) / self.dead_reckoning_seconds)
        return self.fix_confidence * decay

    def predict(self, vx: float, vy: float, dt: float) -> None:
        """Dead-reckon one tick from the commanded velocity."""
        self.time += dt
        if self.x is None or self.y is None:
            return
        self.x += vx * dt
        self.y += vy * dt
        if vx * vx + vy * vy > 1e-6:
            self.heading = math.atan2(vy, vx)

    def correct(self, measurement: PoseMeasurement) -> None:
        self.time = max(self.time, measurement.timestamp)
        if self.x is None or self.y is None:
            self.x, self.y = measurement.x, measurement.y
        else:
            dx, dy = measurement.x - self.x, measurement.y - self.y
            gain = self.alpha
            if math.hypot(dx, dy) > self.RESET_DISTANCE:
                gain = 1.0
                self.resets += 1
            self.x += gain * dx
            self.y += gain * dy
        if measurement.heading is not None:
            if self.heading is None:
                self.heading = measurement.heading
            else:
                delta = math.atan2(math.sin(measurement.heading - self.heading),
                                   math.cos(measurement.heading - self.heading))
                self.heading += self.alpha * delta
        self.last_fix_time = measurement.timestamp
        self.fix_confidence = measurement.confidence
        self.missing = 0
        self.fixes += 1

    def miss(self, timestamp: float) -> None:
        self.time = max(self.time, timestamp)
        self.missing += 1
