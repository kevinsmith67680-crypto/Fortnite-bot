"""Configurable, cached image preprocessing.

A :class:`PreparedFrame` wraps one observation and computes derived images
lazily, at most once per frame: colour masks, grayscale, HSV and edges are
only built if a detector asks for them. :class:`Preprocessor` skips frames it
has already processed (same ``frame_id``) and records timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from chest_hunter.vision.observation import VisualObservation
from chest_hunter.vision.templates import TemplateLibrary


@dataclass(frozen=True)
class PreprocessConfig:
    contrast_alpha: float = 1.0      # 1.0 = unchanged
    contrast_beta: float = 0.0
    resize_factor: float = 1.0       # < 1 processes a smaller image
    roi: tuple[int, int, int, int] | None = None   # (x, y, w, h) crop, pixels


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def to_hsv(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def resize(image: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-9:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))),
                      interpolation=cv2.INTER_AREA)


def adjust_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if alpha == 1.0 and beta == 0.0:
        return image
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def normalise(image: np.ndarray) -> np.ndarray:
    """Stretch intensities to the full 0-255 range."""
    out = np.empty_like(image)
    cv2.normalize(image, out, 0, 255, cv2.NORM_MINMAX)
    return out


def crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``(x, y, w, h)`` (clipped to the image) without copying."""
    x, y, w, h = roi
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    return image[y0:max(y0, y1), x0:max(x0, x1)]


def edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    return cv2.Canny(gray, low, high)


@dataclass
class PreparedFrame:
    """An observation plus lazily computed, cached derived images."""

    observation: VisualObservation
    bgr: np.ndarray
    library: TemplateLibrary
    scale: float = 1.0
    offset: tuple[int, int] = (0, 0)   # (x, y) of an ROI crop in the original frame
    _masks: dict[str, np.ndarray] = field(default_factory=dict)
    _gray: np.ndarray | None = None
    _hsv: np.ndarray | None = None
    _edges: np.ndarray | None = None

    def to_original(self, u: float, v: float) -> tuple[float, float]:
        """Map working-image pixel coordinates back to original-frame pixels."""
        return (self.offset[0] + u / self.scale, self.offset[1] + v / self.scale)

    @property
    def pixels_per_cell(self) -> float:
        return self.observation.camera.pixels_per_cell * self.scale

    @property
    def shape(self) -> tuple[int, int]:
        return self.bgr.shape[0], self.bgr.shape[1]

    def mask(self, colour: str) -> np.ndarray:
        cached = self._masks.get(colour)
        if cached is None:
            colour_range = self.library.colour_range(colour)
            cached = cv2.inRange(self.bgr, colour_range.lower, colour_range.upper)
            self._masks[colour] = cached
        return cached

    def mask_any(self, *colours: str) -> np.ndarray:
        key = "|".join(colours)
        cached = self._masks.get(key)
        if cached is None:
            cached = self.mask(colours[0]).copy()
            for colour in colours[1:]:
                cv2.bitwise_or(cached, self.mask(colour), cached)
            self._masks[key] = cached
        return cached

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = to_gray(self.bgr)
        return self._gray

    @property
    def hsv(self) -> np.ndarray:
        if self._hsv is None:
            self._hsv = to_hsv(self.bgr)
        return self._hsv

    @property
    def edge_map(self) -> np.ndarray:
        if self._edges is None:
            self._edges = edges(self.gray)
        return self._edges


class Preprocessor:
    """Turns observations into :class:`PreparedFrame` objects."""

    def __init__(self, library: TemplateLibrary, config: PreprocessConfig | None = None) -> None:
        self.library = library
        self.config = config or PreprocessConfig()
        self.frames_processed = 0
        self.frames_skipped = 0
        self.total_seconds = 0.0
        self._last: PreparedFrame | None = None

    def prepare(self, observation: VisualObservation) -> PreparedFrame:
        if self._last is not None and self._last.observation.frame_id == observation.frame_id \
                and self._last.observation.timestamp == observation.timestamp:
            self.frames_skipped += 1
            return self._last
        started = time.perf_counter()
        cfg = self.config
        image = observation.frame
        offset = (0, 0)
        if cfg.roi is not None:
            image = crop(image, cfg.roi)
            offset = (max(0, cfg.roi[0]), max(0, cfg.roi[1]))
        image = adjust_contrast(image, cfg.contrast_alpha, cfg.contrast_beta)
        image = resize(image, cfg.resize_factor)
        prepared = PreparedFrame(observation, image, self.library, scale=cfg.resize_factor,
                                 offset=offset)
        self.frames_processed += 1
        self.total_seconds += time.perf_counter() - started
        self._last = prepared
        return prepared
