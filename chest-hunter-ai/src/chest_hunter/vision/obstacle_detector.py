"""Obstacle and walkable-space perception: frames to an estimated occupancy grid.

Every cell fully inside the camera view is classified from the pixels
covering it:

* ``BLOCKED`` if at least 50% of its pixels match the wall colour,
* ``FREE`` if at most 20% do (ground, chests, props and UI overlays are all
  walkable),
* otherwise unchanged (ambiguous, e.g. partly covered by a sprite).

Cells never seen stay ``UNKNOWN``; unseen space is never assumed walkable.
Classification uses one ``cv2.resize(INTER_AREA)`` of the wall mask, which
averages each cell's pixels in C, so a full 640x360 frame costs well under a
millisecond.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from chest_hunter.vision.preprocessing import PreparedFrame

UNKNOWN = -1
FREE = 0
BLOCKED = 1

BLOCKED_THRESHOLD = 0.5
FREE_THRESHOLD = 0.2


class OccupancyGrid:
    """Estimated occupancy, indexed ``[y, x]`` with ``y = 0`` the southern row.

    ``version`` increases whenever any cell changes state; ``seen`` counts
    how many times each cell has been observed; ``last_seen`` stores the
    simulated time of the latest observation (``-inf`` if never).
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cells = np.full((height, width), UNKNOWN, dtype=np.int8)
        self.seen = np.zeros((height, width), dtype=np.uint16)
        self.last_seen = np.full((height, width), -np.inf, dtype=np.float32)
        self.version = 0

    def state(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return BLOCKED
        return int(self.cells[y, x])

    @property
    def known_fraction(self) -> float:
        return float(np.count_nonzero(self.cells != UNKNOWN)) / self.cells.size

    def counts(self) -> dict[str, int]:
        return {
            "unknown": int(np.count_nonzero(self.cells == UNKNOWN)),
            "free": int(np.count_nonzero(self.cells == FREE)),
            "blocked": int(np.count_nonzero(self.cells == BLOCKED)),
        }


@dataclass(frozen=True)
class OccupancyUpdate:
    cells_observed: int
    newly_blocked: int
    newly_free: int
    window: tuple[int, int, int, int]   # (x0, x1, y0, y1), half-open

    @property
    def changed(self) -> bool:
        return self.newly_blocked > 0 or self.newly_free > 0


class ObstacleDetector:
    """Updates an :class:`OccupancyGrid` from prepared frames."""

    def __init__(self) -> None:
        self.calls = 0
        self.total_seconds = 0.0

    def observe(self, frame: PreparedFrame, grid: OccupancyGrid) -> OccupancyUpdate:
        started = time.perf_counter()
        camera = frame.observation.camera
        x0, x1, y0, y1 = camera.visible_cells(grid.width, grid.height)
        if x1 <= x0 or y1 <= y0:
            return OccupancyUpdate(0, 0, 0, (x0, x1, y0, y1))
        ppc = frame.pixels_per_cell
        # Pixel window covering the visible cells (rows go north -> south).
        u0, v0 = camera.world_to_pixel(x0, y1)
        u0, v0 = (u0 - frame.offset[0]) * frame.scale, (v0 - frame.offset[1]) * frame.scale
        cols, rows = x1 - x0, y1 - y0
        pu0, pv0 = int(round(u0)), int(round(v0))
        pu1, pv1 = pu0 + int(round(cols * ppc)), pv0 + int(round(rows * ppc))
        mask = frame.mask("obstacle")[max(0, pv0):pv1, max(0, pu0):pu1]
        if mask.shape[0] < rows or mask.shape[1] < cols:
            return OccupancyUpdate(0, 0, 0, (x0, x1, y0, y1))
        fraction = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_AREA)
        fraction = fraction[::-1].astype(np.float32) / 255.0   # flip to south-up rows

        window = grid.cells[y0:y1, x0:x1]
        before = window.copy()
        blocked = fraction >= BLOCKED_THRESHOLD
        free = fraction <= FREE_THRESHOLD
        window[blocked] = BLOCKED
        window[free] = FREE
        grid.seen[y0:y1, x0:x1] = np.minimum(grid.seen[y0:y1, x0:x1].astype(np.uint32) + 1,
                                             65535).astype(np.uint16)
        grid.last_seen[y0:y1, x0:x1] = frame.observation.timestamp
        newly_blocked = int(np.count_nonzero(blocked & (before != BLOCKED)))
        newly_free = int(np.count_nonzero(free & (before != FREE)))
        if newly_blocked or newly_free:
            grid.version += 1
        self.calls += 1
        self.total_seconds += time.perf_counter() - started
        return OccupancyUpdate(rows * cols, newly_blocked, newly_free, (x0, x1, y0, y1))
