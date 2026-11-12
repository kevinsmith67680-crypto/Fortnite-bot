"""Detection of on-screen interaction cues.

* **Prompt**: a hollow white square drawn above a chest the agent can open
  right now. Its presence near the target means "in range and in sight". Its
  disappearance after an interaction is evidence the chest changed.
* **Progress bar**: a white frame with a green fill above the agent while
  the interact key is held. The fill fraction shows hold progress.
* **Reward flash**: a cyan diamond next to the agent after a fictional token
  is granted.

All positions are reported in original-frame pixels.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from chest_hunter.vision.preprocessing import PreparedFrame


@dataclass(frozen=True)
class Prompt:
    bbox: tuple[int, int, int, int]
    chest_pixel: tuple[float, float]   # where the prompted chest is (below the icon)
    confidence: float


@dataclass(frozen=True)
class InteractionCues:
    prompts: tuple[Prompt, ...] = ()
    progress_visible: bool = False
    progress_fraction: float | None = None
    reward_visible: bool = False

    def prompt_near(self, u: float, v: float, radius: float) -> Prompt | None:
        best = None
        for prompt in self.prompts:
            d = float(np.hypot(prompt.chest_pixel[0] - u, prompt.chest_pixel[1] - v))
            if d <= radius and (best is None or prompt.confidence > best.confidence):
                best = prompt
        return best


class InteractionDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.total_seconds = 0.0

    def detect(self, frame: PreparedFrame,
               agent_pixel: tuple[float, float] | None) -> InteractionCues:
        started = time.perf_counter()
        library = frame.library
        ppc = frame.pixels_per_cell
        agent_local = None
        if agent_pixel is not None:
            agent_local = ((agent_pixel[0] - frame.offset[0]) * frame.scale,
                           (agent_pixel[1] - frame.offset[1]) * frame.scale)
        prompts = self._prompts(frame, ppc,
                                library.size("prompt_size", ppc),
                                library.size("prompt_offset", ppc), agent_local)
        progress_visible, fraction, reward = False, None, False
        if agent_pixel is not None:
            au = (agent_pixel[0] - frame.offset[0]) * frame.scale
            av = (agent_pixel[1] - frame.offset[1]) * frame.scale
            progress_visible, fraction = self._progress(frame, ppc, au, av)
            reward = self._reward(frame, ppc, au, av)
        self.calls += 1
        self.total_seconds += time.perf_counter() - started
        return InteractionCues(tuple(prompts), progress_visible, fraction, reward)

    def _prompts(self, frame: PreparedFrame, ppc: float, size: float, offset: float,
                 agent: tuple[float, float] | None) -> list[Prompt]:
        """Square white icons of the prompt's size (outline thickness widens them).

        The agent's white heading "nose" is similar at low resolution, so blobs
        inside the agent marker are ignored.
        """
        white = frame.mask("prompt")
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected = size + max(1.0, round(ppc / 10))
        prompts = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (0.7 * expected <= w <= 1.6 * expected
                    and 0.7 * expected <= h <= 1.6 * expected and 0.7 <= w / h <= 1.4):
                continue
            if agent is not None and math.hypot(x + w / 2 - agent[0],
                                                y + h / 2 - agent[1]) < 0.6 * ppc:
                continue
            fill = cv2.countNonZero(white[y:y + h, x:x + w]) / float(w * h)
            if fill < 0.4:
                continue
            cu, cv_ = frame.to_original(x + w / 2.0, y + h / 2.0 + offset)
            u0, v0 = frame.to_original(x, y)
            u1, v1 = frame.to_original(x + w, y + h)
            confidence = float(np.clip(1.0 - abs(1.0 - w / expected) - abs(1.0 - h / expected),
                                       0, 1))
            prompts.append(Prompt((int(u0), int(v0), max(1, int(u1 - u0)), max(1, int(v1 - v0))),
                                  (cu, cv_), confidence))
        return prompts

    def _progress(self, frame: PreparedFrame, ppc: float, au: float,
                  av: float) -> tuple[bool, float | None]:
        lib = frame.library
        width = lib.size("progress_width", ppc)
        height = lib.size("progress_height", ppc)
        centre_v = av - lib.size("progress_offset", ppc)
        x0, x1 = int(au - width / 2 - 2), int(au + width / 2 + 3)
        y0, y1 = int(centre_v - height / 2 - 2), int(centre_v + height / 2 + 3)
        img_h, img_w = frame.shape
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)
        if x1 <= x0 or y1 <= y0:
            return False, None
        green = frame.mask("progress_fill")[y0:y1, x0:x1]
        frame_px = frame.mask("prompt")[y0:y1, x0:x1]
        columns = np.count_nonzero(green, axis=0) > 0
        filled = int(np.count_nonzero(columns))
        outline = cv2.countNonZero(frame_px) >= width * 0.8
        if not outline and filled < 2:
            return False, None
        return True, float(np.clip(filled / width, 0.0, 1.0))

    def _reward(self, frame: PreparedFrame, ppc: float, au: float, av: float) -> bool:
        x0, x1 = int(au), int(au + 1.2 * ppc)
        y0, y1 = int(av - 1.2 * ppc), int(av)
        img_h, img_w = frame.shape
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)
        if x1 <= x0 or y1 <= y0:
            return False
        cyan = frame.mask("reward")[y0:y1, x0:x1]
        return cv2.countNonZero(cyan) >= 0.1 * ppc * ppc
