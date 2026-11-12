"""Simulator camera: geometry, a following camera rig and a NumPy frame renderer.

Camera geometry (documented, and reported with every observation):

* World coordinates are grid cells; ``y`` points north.
* Pixel coordinates ``(u, v)``: ``u`` to the right, ``v`` downward, origin at
  the top-left corner of the top-left pixel. Pixel ``(i, j)`` covers
  ``[i, i+1) x [j, j+1)``. Bounding boxes are ``(x, y, w, h)`` in pixels with
  ``(x, y)`` the top-left corner.
* ``u = (x - origin_x) * ppc`` and ``v = (origin_top - y) * ppc`` where
  ``ppc`` is pixels per cell and ``origin_top = origin_y + view_height``.

The camera is top-down and orthographic and follows the agent with a dead
zone, moving in whole-cell steps. It sees ``view_width_cells`` cells across
(40 by default); the vertical extent follows from the aspect ratio
(40 x 22.5 cells at 16:9).

The renderer is part of the *simulator*: it reads true state to draw the
picture. Perception only ever receives the resulting pixels plus the camera
geometry. Absent chests are not drawn at all.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

from chest_hunter.simulation import sprites
from chest_hunter.simulation.collision import GridMap

SUPPORTED_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (320, 180), (480, 270), (640, 360), (800, 450),
)


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Orthographic top-down camera geometry for one frame."""

    origin_x: float          # world x at the left image edge (cells)
    origin_y: float          # world y at the bottom image edge (cells)
    pixels_per_cell: float
    width: int
    height: int

    @property
    def view_width(self) -> float:
        return self.width / self.pixels_per_cell

    @property
    def view_height(self) -> float:
        return self.height / self.pixels_per_cell

    @property
    def origin_top(self) -> float:
        return self.origin_y + self.view_height

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        ppc = self.pixels_per_cell
        return ((x - self.origin_x) * ppc, (self.origin_top - y) * ppc)

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        ppc = self.pixels_per_cell
        return (self.origin_x + u / ppc, self.origin_top - v / ppc)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.origin_x + margin <= x < self.origin_x + self.view_width - margin
                and self.origin_y + margin <= y < self.origin_top - margin)

    def visible_cells(self, map_width: int, map_height: int) -> tuple[int, int, int, int]:
        """``(x0, x1, y0, y1)`` (half-open) of map cells fully inside the frame."""
        x0 = max(0, math.ceil(self.origin_x - 1e-9))
        x1 = min(map_width, math.floor(self.origin_x + self.view_width + 1e-9))
        y0 = max(0, math.ceil(self.origin_y - 1e-9))
        y1 = min(map_height, math.floor(self.origin_top + 1e-9))
        return x0, max(x0, x1), y0, max(y0, y1)


class FollowCamera:
    """Keeps the agent inside a central dead zone, moving in whole-cell steps."""

    def __init__(self, map_width: int, map_height: int, width: int, height: int,
                 view_width_cells: int = 40, dead_zone: float = 0.25) -> None:
        if width % view_width_cells:
            raise ValueError(
                f"Frame width {width} must be a multiple of view_width_cells "
                f"{view_width_cells} so pixels per cell is an integer.")
        self.map_width = map_width
        self.map_height = map_height
        self.width = width
        self.height = height
        self.ppc = width // view_width_cells
        self.view_w = width / self.ppc
        self.view_h = height / self.ppc
        self.margin_x = self.view_w * dead_zone
        self.margin_y = self.view_h * dead_zone
        self.origin_x = 0
        self.origin_y = 0
        self._placed = False

    def _clamp(self, value: int, limit: float, view: float) -> int:
        """Keep the view on the map; it may overhang by < 1 cell so edge cells are fully seen."""
        if limit <= view:
            return math.floor((limit - view) / 2)
        return int(min(max(value, 0), math.ceil(limit - view)))

    def update(self, x: float, y: float) -> CameraModel:
        if not self._placed:
            self.origin_x = math.floor(x - self.view_w / 2)
            self.origin_y = math.floor(y - self.view_h / 2)
            self._placed = True
        else:
            if x < self.origin_x + self.margin_x:
                self.origin_x = math.floor(x - self.margin_x)
            elif x > self.origin_x + self.view_w - self.margin_x:
                self.origin_x = math.ceil(x + self.margin_x - self.view_w)
            if y < self.origin_y + self.margin_y:
                self.origin_y = math.floor(y - self.margin_y)
            elif y > self.origin_y + self.view_h - self.margin_y:
                self.origin_y = math.ceil(y + self.margin_y - self.view_h)
        self.origin_x = self._clamp(self.origin_x, self.map_width, self.view_w)
        self.origin_y = self._clamp(self.origin_y, self.map_height, self.view_h)
        return CameraModel(float(self.origin_x), float(self.origin_y), float(self.ppc),
                           self.width, self.height)


@dataclass(frozen=True)
class SensorModel:
    """Deterministic camera imperfections."""

    noise_std: float = 6.0             # Gaussian pixel noise (intensity levels)
    lighting_variation: float = 0.08   # +/- relative brightness per frame
    frame_dropout: float = 0.0         # probability that a capture fails


class FrameRenderer:
    """Renders camera frames of an environment into reusable NumPy buffers."""

    NOISE_BANK = 8

    def __init__(self, width: int, height: int, *, view_width_cells: int = 40,
                 sensor: SensorModel | None = None, seed: int = 0) -> None:
        if (width, height) not in SUPPORTED_RESOLUTIONS and (width < 64 or height < 36):
            raise ValueError(f"Unsupported frame size {width}x{height}.")
        if width % view_width_cells:
            raise ValueError("width must be a multiple of view_width_cells.")
        self.width = width
        self.height = height
        self.view_width_cells = view_width_cells
        self.ppc = width // view_width_cells
        self.sensor = sensor or SensorModel()
        self.seed = seed
        self._frame = np.zeros((height, width, 3), np.uint8)
        self._background: np.ndarray | None = None
        self._background_key: tuple[int, int, int] | None = None
        rng = np.random.default_rng(seed)
        self._noise_pos: list[np.ndarray] = []
        self._noise_neg: list[np.ndarray] = []
        if self.sensor.noise_std > 0:
            for _ in range(self.NOISE_BANK):
                noise = rng.normal(0.0, self.sensor.noise_std, (height, width, 3))
                self._noise_pos.append(np.clip(noise, 0, 255).astype(np.uint8))
                self._noise_neg.append(np.clip(-noise, 0, 255).astype(np.uint8))

    # -- static background ---------------------------------------------------------

    def _background_for(self, env: object) -> np.ndarray:
        from chest_hunter.simulation.environment import Environment  # simulator side

        assert isinstance(env, Environment)
        grid: GridMap = env.grid
        key = (id(grid), grid.version, self.ppc)
        if self._background is not None and self._background_key == key:
            return self._background
        ppc = self.ppc
        mask = np.frombuffer(grid.cells, dtype=np.uint8).reshape(grid.height, grid.width)[::-1]
        big = np.repeat(np.repeat(mask, ppc, axis=0), ppc, axis=1).astype(bool)
        background = np.empty((grid.height * ppc, grid.width * ppc, 3), np.uint8)
        background[:] = sprites.GROUND
        background[big] = sprites.OBSTACLE
        for deco in env.world.decorations:
            sprites.draw_decoration(background, deco.position_x * ppc,
                                    (grid.height - deco.position_y) * ppc, ppc, deco.radius)
        self._background = background
        self._background_key = key
        return background

    # -- frames -------------------------------------------------------------------

    def dropped(self, frame_id: int) -> bool:
        if self.sensor.frame_dropout <= 0:
            return False
        return random.Random(f"chest-hunter:dropout:{self.seed}:{frame_id}").random() \
            < self.sensor.frame_dropout

    def render(self, env: object, camera: CameraModel, frame_id: int, *,
               apply_sensor: bool = True) -> np.ndarray:
        """Render the frame for ``camera`` into the reused buffer and return it."""
        from chest_hunter.simulation.environment import Environment

        assert isinstance(env, Environment)
        frame = self._frame
        frame[:] = sprites.OUT_OF_BOUNDS
        background = self._background_for(env)
        ppc = self.ppc
        bg_h, bg_w = background.shape[:2]
        col0 = int(round(camera.origin_x * ppc))
        row0 = int(round((env.grid.height - camera.origin_top) * ppc))
        src_c0, src_r0 = max(0, col0), max(0, row0)
        src_c1, src_r1 = min(bg_w, col0 + self.width), min(bg_h, row0 + self.height)
        if src_c1 > src_c0 and src_r1 > src_r0:
            frame[src_r0 - row0:src_r1 - row0, src_c0 - col0:src_c1 - col0] = \
                background[src_r0:src_r1, src_c0:src_c1]

        visible = [c for c in env.chests.values()
                   if c.present and c.id not in env.hidden_chests
                   and camera.contains(c.position_x, c.position_y, margin=-1.0)]
        for chest in sorted(visible, key=lambda c: (not c.opened, c.id)):  # opened first
            u, v = camera.world_to_pixel(chest.position_x, chest.position_y)
            sprites.draw_chest(frame, u, v, ppc, opened=chest.opened)
        prompt = env.interactable_chest()
        if prompt is not None and prompt.id not in env.hidden_chests:
            u, v = camera.world_to_pixel(prompt.position_x, prompt.position_y)
            sprites.draw_prompt(frame, u, v, ppc)
        agent = env.agent
        au, av = camera.world_to_pixel(agent.position_x, agent.position_y)
        sprites.draw_agent(frame, au, av, ppc, agent.heading)
        if env.hold_fraction > 0:
            sprites.draw_progress(frame, au, av, ppc, env.hold_fraction)
        if env.tick <= env.reward_flash_until:
            sprites.draw_reward(frame, au, av, ppc)

        if apply_sensor:
            self._apply_sensor(frame, frame_id)
        return frame

    def _apply_sensor(self, frame: np.ndarray, frame_id: int) -> None:
        variation = self.sensor.lighting_variation
        if variation > 0:
            u = random.Random(f"chest-hunter:light:{self.seed}:{frame_id}").uniform(-1, 1)
            cv2.convertScaleAbs(frame, frame, alpha=1.0 + variation * u)
        if self._noise_pos:
            k = frame_id % self.NOISE_BANK
            cv2.add(frame, self._noise_pos[k], frame)
            cv2.subtract(frame, self._noise_neg[k], frame)
