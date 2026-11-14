"""Interaction cues in third-person frames: the "E" prompt and the progress bar.

Both are world-space overlays drawn above the chest the character can
interact with, so they are searched for only in a window above the target's
bounding box:

* **prompt**: a near-white square (low saturation, very bright) with dark
  pixels inside (the letter);
* **progress bar**: a wide, saturated green fill (distinct from grass and
  foliage by hue and saturation).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WHITE = ((0, 0, 225), (180, 40, 255))
PROGRESS_GREEN = ((64, 160, 110), (80, 255, 255))


@dataclass(frozen=True, slots=True)
class Cues:
    prompt: bool
    progress: bool
    prompt_bbox: tuple[int, int, int, int] | None = None


def _window(bbox: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    size = max(w, h)
    x0 = max(0, int(x - 1.2 * size))
    x1 = min(shape[1], int(x + w + 1.2 * size))
    y0 = max(0, int(y - 5.0 * size))
    y1 = min(shape[0], int(y + 0.4 * h))
    return x0, y0, x1, y1


def detect_cues(frame: np.ndarray, target_bbox: tuple[int, int, int, int] | None) -> Cues:
    """Look for the prompt and the progress bar above ``target_bbox``."""
    if target_bbox is None:
        return Cues(False, False)
    x0, y0, x1, y1 = _window(target_bbox, frame.shape)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return Cues(False, False)
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array(WHITE[0]), np.array(WHITE[1]))
    count, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    prompt_box = None
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if w < 5 or h < 5 or not 0.6 <= w / h <= 1.7:
            continue
        inner = hsv[y:y + h, x:x + w, 2]
        dark = np.count_nonzero(inner < 90) / inner.size
        if 0.03 <= dark <= 0.5 and area >= 0.45 * w * h:
            prompt_box = (x0 + x, y0 + y, w, h)
            break
    green = cv2.inRange(hsv, np.array(PROGRESS_GREEN[0]), np.array(PROGRESS_GREEN[1]))
    progress = False
    if np.count_nonzero(green) >= 3:
        count, _, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
        for i in range(1, count):
            _, _, w, h, area = (int(v) for v in stats[i])
            if area >= 3 and w >= 2 * max(1, h):
                progress = True
                break
    return Cues(prompt_box is not None, progress, prompt_box)
