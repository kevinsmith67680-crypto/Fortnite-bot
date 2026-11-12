"""Sprite appearance for the simulator camera (BGR colours, OpenCV drawing).

This module defines how things *look*. It is used by the simulator's frame
renderer and by ``scripts/generate_templates.py`` to produce labelled
reference templates. Perception code never imports it at run time: it only
sees rendered frames and the template images on disk.

Sizes are in grid cells and scaled by ``ppc`` (pixels per cell).
"""

from __future__ import annotations

import math
from typing import Final

import cv2
import numpy as np

# BGR colours
GROUND: Final = (40, 40, 40)
OUT_OF_BOUNDS: Final = (0, 0, 0)
OBSTACLE: Final = (139, 116, 100)          # slate grey
CHEST_BODY: Final = (24, 197, 245)         # gold
CHEST_LID: Final = (20, 72, 120)           # dark brown lid band / outline
OPEN_BODY: Final = (40, 95, 150)           # brown body of an opened chest
OPEN_INSIDE: Final = (160, 210, 230)       # pale interior of an opened chest
DECORATION: Final = (40, 190, 230)         # gold-ish hay bale (distractor)
DECORATION_CORE: Final = (30, 140, 180)
AGENT: Final = (246, 130, 59)              # blue
AGENT_NOSE: Final = (255, 255, 255)
PROMPT: Final = (255, 255, 255)            # interaction prompt icon
PROGRESS_FILL: Final = (94, 197, 34)       # green progress bar fill
PROGRESS_FRAME: Final = (255, 255, 255)
REWARD: Final = (238, 211, 34)             # cyan token flash

CHEST_WIDTH: Final = 0.8   # cells
CHEST_HEIGHT: Final = 0.6
AGENT_RADIUS: Final = 0.35
PROMPT_OFFSET: Final = 0.75  # prompt centre above the chest, cells
PROMPT_SIZE: Final = 0.45
PROGRESS_OFFSET: Final = 0.8
PROGRESS_WIDTH: Final = 1.2
PROGRESS_HEIGHT: Final = 0.22

_SHIFT: Final = 4            # sub-pixel precision for OpenCV drawing
_SCALE: Final = 1 << _SHIFT


def _fx(value: float) -> int:
    """Continuous pixel coordinate -> OpenCV fixed-point coordinate.

    Our convention: pixel ``i`` covers ``[i, i+1)``, so its centre is ``i + 0.5``.
    OpenCV addresses pixel centres with integers, hence the ``- 0.5`` shift.
    Sizes (radii) must use :func:`_len` instead.
    """
    return int(round((value - 0.5) * _SCALE))


def _len(value: float) -> int:
    return int(round(value * _SCALE))


def _rect(img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
          colour: tuple[int, int, int], thickness: int = -1) -> None:
    cv2.rectangle(img, (_fx(x0 + 0.5), _fx(y0 + 0.5)), (_fx(x1 - 0.5), _fx(y1 - 0.5)), colour,
                  thickness, cv2.LINE_8, _SHIFT)


def chest_box(u: float, v: float, ppc: float) -> tuple[float, float, float, float]:
    """Pixel box ``(x0, y0, x1, y1)`` of a chest sprite centred at ``(u, v)``."""
    w, h = CHEST_WIDTH * ppc, CHEST_HEIGHT * ppc
    return (u - w / 2, v - h / 2, u + w / 2, v + h / 2)


def draw_chest(img: np.ndarray, u: float, v: float, ppc: float, *, opened: bool) -> None:
    """Draw a chest centred at pixel ``(u, v)`` (top-down view)."""
    x0, y0, x1, y1 = chest_box(u, v, ppc)
    h = y1 - y0
    if opened:
        _rect(img, x0, y0, x1, y1, OPEN_BODY)
        _rect(img, x0 + 0.08 * ppc, y0 + 0.06 * ppc, x1 - 0.08 * ppc, y0 + 0.45 * h, OPEN_INSIDE)
    else:
        _rect(img, x0, y0, x1, y1, CHEST_BODY)
        _rect(img, x0, y0, x1, y0 + 0.35 * h, CHEST_LID)
    thickness = max(1, round(ppc / 16))
    cv2.rectangle(img, (_fx(x0 + 0.5), _fx(y0 + 0.5)), (_fx(x1 - 0.5), _fx(y1 - 0.5)),
                  CHEST_LID, thickness, cv2.LINE_8, _SHIFT)


def draw_decoration(img: np.ndarray, u: float, v: float, ppc: float, radius: float) -> None:
    cv2.circle(img, (_fx(u), _fx(v)), _len(radius * ppc), DECORATION, -1, cv2.LINE_8, _SHIFT)
    cv2.circle(img, (_fx(u), _fx(v)), _len(radius * 0.4 * ppc), DECORATION_CORE, -1,
               cv2.LINE_8, _SHIFT)


def draw_agent(img: np.ndarray, u: float, v: float, ppc: float, heading: float) -> None:
    """Blue disc with a white "nose" in the heading direction (image y points down)."""
    cv2.circle(img, (_fx(u), _fx(v)), _len(AGENT_RADIUS * ppc), AGENT, -1, cv2.LINE_8, _SHIFT)
    nu = u + math.cos(heading) * 0.2 * ppc
    nv = v - math.sin(heading) * 0.2 * ppc
    cv2.circle(img, (_fx(nu), _fx(nv)), _len(0.11 * ppc), AGENT_NOSE, -1, cv2.LINE_8, _SHIFT)


def draw_prompt(img: np.ndarray, u: float, v: float, ppc: float) -> None:
    """Interaction prompt icon above a chest at pixel ``(u, v)``."""
    cu, cv_ = u, v - PROMPT_OFFSET * ppc
    half = PROMPT_SIZE * ppc / 2
    thickness = max(1, round(ppc / 10))
    cv2.rectangle(img, (_fx(cu - half), _fx(cv_ - half)), (_fx(cu + half), _fx(cv_ + half)),
                  PROMPT, thickness, cv2.LINE_8, _SHIFT)
    _rect(img, cu - half * 0.5, cv_ - half * 0.2, cu + half * 0.5, cv_ + half * 0.2, PROMPT)


def draw_progress(img: np.ndarray, u: float, v: float, ppc: float, fraction: float) -> None:
    """Progress bar above the agent at pixel ``(u, v)``."""
    w, h = PROGRESS_WIDTH * ppc, PROGRESS_HEIGHT * ppc
    x0, y0 = u - w / 2, v - PROGRESS_OFFSET * ppc - h / 2
    if fraction > 0:
        _rect(img, x0, y0, x0 + w * min(1.0, fraction), y0 + h, PROGRESS_FILL)
    cv2.rectangle(img, (_fx(x0), _fx(y0)), (_fx(x0 + w), _fx(y0 + h)), PROGRESS_FRAME, 1,
                  cv2.LINE_8, _SHIFT)


def draw_reward(img: np.ndarray, u: float, v: float, ppc: float) -> None:
    """Fictional token flash: a cyan diamond near the agent."""
    cu, cv_ = u + 0.6 * ppc, v - 0.6 * ppc
    r = 0.25 * ppc
    pts = (np.array([[cu, cv_ - r], [cu + r, cv_], [cu, cv_ + r], [cu - r, cv_]]) - 0.5) * _SCALE
    cv2.fillPoly(img, [pts.astype(np.int32)], REWARD, cv2.LINE_8, _SHIFT)


def chest_template(ppc: float, *, opened: bool, pad: float = 0.15) -> np.ndarray:
    """A chest sprite on ground background, cropped with ``pad`` cells of margin."""
    w = math.ceil((CHEST_WIDTH + 2 * pad) * ppc)
    h = math.ceil((CHEST_HEIGHT + 2 * pad) * ppc)
    img = np.full((h, w, 3), GROUND, np.uint8)
    draw_chest(img, w / 2, h / 2, ppc, opened=opened)
    return img
