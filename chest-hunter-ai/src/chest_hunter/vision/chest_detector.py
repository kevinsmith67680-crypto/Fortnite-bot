"""Classical chest detection on rendered frames.

Three interchangeable methods:

* :class:`ColourShapeDetector` (method A): colour segmentation, contours and
  geometric checks (size, aspect ratio, fill, and the lid band or open
  interior).
* :class:`TemplateDetector` (method B): normalised cross-correlation
  (``TM_CCOEFF_NORMED``) with reference templates over the whole frame.
* :class:`HybridDetector` (method C, default): colour candidates confirmed by
  template matching inside small regions of interest. The opened/unopened
  class comes from colour composition when that is decisive (gold body vs pale
  interior), because correlation separates the two classes poorly at low
  resolution. A candidate up to ``RELAXED_MARGIN`` below the threshold is
  accepted only if its colour/shape check also passes. Chests cut off by the
  frame border fall back to the shape score.

Template matching runs on lightly blurred images (3x3 Gaussian on both the
image region and the cached templates), which suppresses sensor noise. This
mattered for opened chests: their low internal contrast made raw scores hover
around the threshold.

Pixel convention: ``bbox = (x, y, w, h)``, top-left corner, ``u`` right and
``v`` down, in original-frame pixels.

Confidence is a *confidence-like score* in [0, 1]: the template correlation
or a geometric quality score. It is not a calibrated probability.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from chest_hunter.vision.preprocessing import PreparedFrame
from chest_hunter.vision.templates import CHEST_CATEGORIES, TemplateLibrary

_KERNEL3 = np.ones((3, 3), np.uint8)
_BLUR = (3, 3)
RELAXED_MARGIN = 0.15
STRONG_SHAPE = 0.8   # shape score accepted without template support (occlusion)


@dataclass(frozen=True, slots=True)
class Detection:
    category: str                         # "chest" (unopened) or "chest_opened"
    bbox: tuple[int, int, int, int]       # (x, y, w, h) pixels, top-left origin
    confidence: float                     # 0-1 confidence-like score
    timestamp: float                      # simulated seconds of the frame
    method: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.bbox[2] <= 0 or self.bbox[3] <= 0:
            raise ValueError(f"bbox must have positive size, got {self.bbox}")

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection], iou_threshold: float = 0.3) -> list[Detection]:
    """Greedy NMS across categories (one physical chest yields one detection)."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: (-d.confidence, d.bbox)):
        if all(iou(det.bbox, k.bbox) <= iou_threshold for k in kept):
            kept.append(det)
    return kept


def _touches_border(x: int, y: int, w: int, h: int, width: int, height: int) -> bool:
    return x <= 0 or y <= 0 or x + w >= width or y + h >= height


class ChestDetector(ABC):
    name = "base"

    def __init__(self, library: TemplateLibrary, *, confidence_threshold: float = 0.85) -> None:
        self.library = library
        self.confidence_threshold = confidence_threshold
        self.calls = 0
        self.total_seconds = 0.0

    def detect(self, frame: PreparedFrame) -> list[Detection]:
        started = time.perf_counter()
        detections = non_max_suppression(self._detect(frame))
        self.calls += 1
        self.total_seconds += time.perf_counter() - started
        return detections

    @abstractmethod
    def _detect(self, frame: PreparedFrame) -> list[Detection]:
        raise NotImplementedError

    # -- shared geometry scoring -----------------------------------------------------

    def _shape_score(self, frame: PreparedFrame, box: tuple[int, int, int, int],
                     category: str) -> float:
        """Geometric quality 0-1 of a candidate box for ``category`` (0 = reject)."""
        x, y, w, h = box
        ppc = frame.pixels_per_cell
        exp_w = self.library.size("chest_width", ppc)
        exp_h = self.library.size("chest_height", ppc)
        img_h, img_w = frame.shape
        border = _touches_border(x, y, w, h, img_w, img_h)
        area_ratio = (w * h) / (exp_w * exp_h)
        if border:
            if not 0.2 <= area_ratio <= 2.2 or w > exp_w * 1.4 or h > exp_h * 2.2:
                return 0.0
        else:
            if not 0.55 <= area_ratio <= 2.2 or not 0.9 <= w / h <= 2.6:
                return 0.0
        if category == "chest":
            body = frame.mask("chest_body")[y:y + h, x:x + w]
            band = frame.mask("chest_lid")[y:y + max(1, round(h * 0.4)), x:x + w]
            lower = body[round(h * 0.45):]
            body_frac = cv2.countNonZero(lower) / max(1, lower.size)
            band_frac = cv2.countNonZero(band) / max(1, band.size)
            if body_frac < 0.45 or band_frac < 0.3:
                return 0.0
            quality = 0.5 * min(1.0, body_frac / 0.8) + 0.5 * min(1.0, band_frac / 0.6)
        else:
            inside = frame.mask("open_inside")[y:y + max(1, round(h * 0.55)), x:x + w]
            gold = frame.mask("chest_body")[y:y + h, x:x + w]
            inside_frac = cv2.countNonZero(inside) / max(1, inside.size)
            gold_frac = cv2.countNonZero(gold) / max(1, gold.size)
            if inside_frac < 0.25 or gold_frac > 0.25:
                return 0.0
            quality = min(1.0, inside_frac / 0.5)
        size_quality = 1.0 if border else max(0.0, 1.0 - abs(1.0 - area_ratio))
        return float(np.clip(0.7 * quality + 0.3 * max(size_quality, 0.5), 0.0, 1.0))

    def _candidates(self, frame: PreparedFrame) -> list[tuple[int, int, int, int]]:
        mask = frame.mask_any("chest_body", "chest_lid", "open_body", "open_inside")
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ppc = frame.pixels_per_cell
        min_area = 0.15 * self.library.size("chest_width", ppc) * \
            self.library.size("chest_height", ppc)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= min_area:
                boxes.append((x, y, w, h))
        return boxes

    def _detection(self, frame: PreparedFrame, box: tuple[int, int, int, int], category: str,
                   confidence: float, method: str) -> Detection:
        x, y, w, h = box
        u0, v0 = frame.to_original(x, y)
        u1, v1 = frame.to_original(x + w, y + h)
        bbox = (int(round(u0)), int(round(v0)), max(1, int(round(u1 - u0))),
                max(1, int(round(v1 - v0))))
        return Detection(category, bbox, float(np.clip(confidence, 0.0, 1.0)),
                         frame.observation.timestamp, method)


class ColourShapeDetector(ChestDetector):
    """Method A: colour segmentation + contour geometry."""

    name = "color"

    def _detect(self, frame: PreparedFrame) -> list[Detection]:
        detections = []
        for box in self._candidates(frame):
            for category in CHEST_CATEGORIES:
                score = self._shape_score(frame, box, category)
                if score >= min(self.confidence_threshold, 0.6):
                    detections.append(self._detection(frame, box, category, score, self.name))
        return detections


class TemplateDetector(ChestDetector):
    """Method B: full-frame normalised template matching with peak extraction."""

    name = "template"

    def _blurred_templates(self, category: str, ppc: float) -> list[np.ndarray]:
        key = (category, round(ppc, 3))
        cache: dict[tuple[str, float], list[np.ndarray]] = self.__dict__.setdefault(
            "_blur_cache", {})
        if key not in cache:
            cache[key] = [cv2.GaussianBlur(t, _BLUR, 0)
                          for t in self.library.templates(category, ppc)]
        return cache[key]

    def _match(self, image: np.ndarray, category: str, ppc: float,
               offset: tuple[int, int] = (0, 0), *, threshold: float | None = None,
               blurred: bool = False) -> list[tuple[float, tuple[int, int, int, int]]]:
        """Template peaks >= ``threshold``. ``image`` must already be blurred if ``blurred``."""
        threshold = self.confidence_threshold if threshold is None else threshold
        if not blurred:
            image = cv2.GaussianBlur(image, _BLUR, 0)
        results = []
        for template in self._blurred_templates(category, ppc):
            th, tw = template.shape[:2]
            if image.shape[0] < th or image.shape[1] < tw:
                continue
            scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            peaks = cv2.dilate(scores, np.ones((max(3, th // 2), max(3, tw // 2)), np.uint8))
            ys, xs = np.nonzero((scores >= threshold) & (scores >= peaks - 1e-6))
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
                results.append((float(scores[y, x]), (x + offset[0], y + offset[1], tw, th)))
        return results

    def _detect(self, frame: PreparedFrame) -> list[Detection]:
        detections = []
        for category in CHEST_CATEGORIES:
            for score, box in self._match(frame.bgr, category, frame.pixels_per_cell):
                detections.append(self._detection(frame, box, category, score, self.name))
        return detections


class HybridDetector(TemplateDetector):
    """Method C: colour candidates confirmed by template matching in ROIs."""

    name = "hybrid"

    def _detect(self, frame: PreparedFrame) -> list[Detection]:
        ppc = frame.pixels_per_cell
        pad = int(round(0.5 * ppc))
        img_h, img_w = frame.shape
        detections = []
        relaxed = max(0.0, self.confidence_threshold - RELAXED_MARGIN)
        for box in self._candidates(frame):
            x, y, w, h = box
            rx0, ry0 = max(0, x - pad), max(0, y - pad)
            rx1, ry1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
            roi = cv2.GaussianBlur(frame.bgr[ry0:ry1, rx0:rx1], _BLUR, 0)
            best_by_category: dict[str, tuple[float, tuple[int, int, int, int]]] = {}
            for category in CHEST_CATEGORIES:
                for score, match_box in self._match(roi, category, ppc, (rx0, ry0),
                                                    threshold=relaxed, blurred=True):
                    if category not in best_by_category or score > best_by_category[category][0]:
                        best_by_category[category] = (score, match_box)
            shape = {c: self._shape_score(frame, box, c) for c in CHEST_CATEGORIES}
            decisive = [c for c in CHEST_CATEGORIES if shape[c] >= 0.6]
            if len(decisive) == 1 and decisive[0] in best_by_category:
                category = decisive[0]
            elif best_by_category:
                category = max(best_by_category, key=lambda c: best_by_category[c][0])
            else:
                category = ""
            if category:
                score, match_box = best_by_category[category]
                if score >= self.confidence_threshold or shape[category] >= 0.6:
                    detections.append(self._detection(frame, match_box, category, score,
                                                      self.name))
                    continue
            if _touches_border(x, y, w, h, img_w, img_h):
                for category in CHEST_CATEGORIES:  # partially visible: shape fallback
                    score = self._shape_score(frame, box, category)
                    if score >= 0.6:
                        detections.append(self._detection(frame, box, category, score,
                                                          "hybrid-partial"))
            elif len(decisive) == 1 and shape[decisive[0]] >= STRONG_SHAPE:
                # Partly occluded (e.g. by the agent): the template score drops, but the
                # colour composition (gold body + lid band, or pale open interior) is
                # specific to chests, so strong shape evidence alone is accepted.
                detections.append(self._detection(frame, box, decisive[0], shape[decisive[0]],
                                                  "hybrid-shape"))
        return detections


def make_detector(kind: str, library: TemplateLibrary, *,
                  confidence_threshold: float = 0.85) -> ChestDetector:
    detectors: dict[str, type[ChestDetector]] = {
        "color": ColourShapeDetector, "template": TemplateDetector, "hybrid": HybridDetector,
    }
    if kind not in detectors:
        raise ValueError(f"Unknown detector {kind!r}; expected one of {sorted(detectors)}.")
    return detectors[kind](library, confidence_threshold=confidence_threshold)
