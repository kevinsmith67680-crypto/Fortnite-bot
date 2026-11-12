"""Observation interface between the simulator and perception.

* :class:`VisualObservation` - one validated camera frame plus permitted
  metadata: the simulation timestamp and the documented camera geometry.
* :class:`ObservationProvider` - anything that can ``capture()`` frames.
* :class:`SimulatorObservationProvider` - renders the simulator's own camera
  directly into a NumPy buffer (no desktop screenshots, no other processes).

Buffer ownership: with ``reuse_buffer=True`` (default) the provider renders
every frame into the same array, so a frame is valid until the next
``capture()``; callers that need to keep one must ``.copy()`` it. This avoids
a 0.7 MB allocation per 640x360 frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from chest_hunter.exceptions import ChestHunterError
from chest_hunter.simulation.camera import (
    SUPPORTED_RESOLUTIONS,
    CameraModel,
    FollowCamera,
    FrameRenderer,
    SensorModel,
)


class ObservationError(ChestHunterError):
    """An observation was malformed."""


@dataclass(frozen=True, slots=True)
class VisualObservation:
    frame: np.ndarray        # (height, width, 3) uint8, BGR
    timestamp: float         # simulated seconds since episode start
    width: int
    height: int
    frame_id: int
    camera: CameraModel      # documented camera geometry for this frame

    def __post_init__(self) -> None:
        frame = self.frame
        if not isinstance(frame, np.ndarray):
            raise ObservationError("frame must be a NumPy array.")
        if frame.dtype != np.uint8:
            raise ObservationError(f"frame dtype must be uint8, got {frame.dtype}.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ObservationError(f"frame must have shape (H, W, 3), got {frame.shape}.")
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ObservationError(
                f"frame shape {frame.shape[:2]} does not match {self.height}x{self.width}.")
        if self.width < 1 or self.height < 1:
            raise ObservationError("frame dimensions must be positive.")
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ObservationError(f"invalid timestamp {self.timestamp!r}.")
        if self.frame_id < 0:
            raise ObservationError("frame_id must not be negative.")
        if (self.camera.width, self.camera.height) != (self.width, self.height):
            raise ObservationError("camera geometry does not match the frame size.")


class ObservationProvider(ABC):
    """Source of visual observations."""

    @abstractmethod
    def capture(self) -> VisualObservation | None:
        """Return the latest rendered observation, or ``None`` if the capture failed."""
        raise NotImplementedError


class SimulatorObservationProvider(ObservationProvider):
    """Renders the simulator's camera into a NumPy image buffer.

    This class sits on the simulator side of the boundary: it reads the true
    environment to draw the picture, but returns only pixels, a timestamp and
    the camera geometry.
    """

    def __init__(self, env: Any, *, width: int = 640, height: int = 360,
                 view_width_cells: int = 40, sensor: SensorModel | None = None,
                 seed: int = 0, reuse_buffer: bool = True) -> None:
        if (width, height) not in SUPPORTED_RESOLUTIONS:
            sizes = ", ".join(f"{w}x{h}" for w, h in SUPPORTED_RESOLUTIONS)
            raise ObservationError(f"Unsupported resolution {width}x{height}; use {sizes}.")
        self._env = env
        self.width = width
        self.height = height
        self.reuse_buffer = reuse_buffer
        self.camera_rig = FollowCamera(env.grid.width, env.grid.height, width, height,
                                       view_width_cells)
        self.renderer = FrameRenderer(width, height, view_width_cells=view_width_cells,
                                      sensor=sensor, seed=seed)
        self.frames_captured = 0
        self.frames_failed = 0
        self._next_id = 0
        self.last_camera: CameraModel | None = None

    @property
    def pixels_per_cell(self) -> int:
        return self.renderer.ppc

    def current_camera(self) -> CameraModel:
        agent = self._env.agent
        camera = self.camera_rig.update(agent.position_x, agent.position_y)
        self.last_camera = camera
        return camera

    def capture(self) -> VisualObservation | None:
        frame_id = self._next_id
        self._next_id += 1
        camera = self.current_camera()
        if self.renderer.dropped(frame_id):
            self.frames_failed += 1
            return None
        frame = self.renderer.render(self._env, camera, frame_id)
        if not self.reuse_buffer:
            frame = frame.copy()
        self.frames_captured += 1
        return VisualObservation(frame, self._env.time, self.width, self.height, frame_id,
                                 camera)
