"""Evaluation of visual perception against simulator ground truth.

This module is *evaluation code*: it reads the true environment to score
what the policy perceived. The policy never sees these numbers.

Per analysed frame:

* **Detection**: ground-truth chests whose sprite is inside the frame are
  projected to pixel boxes. Overlapping sprites (chests stacked at almost the
  same 2D position) are merged into one visible object, and chests less than
  50% visible are *optional* (matching them counts as a true positive,
  missing them is not a false negative). Detections match ground truth
  greedily at IoU >= 0.4. Class errors count opened/unopened confusions.
* **Localisation**: error of the raw marker measurement and of the filtered
  pose estimate against the true agent position, in grid cells.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from chest_hunter.simulation import sprites
from chest_hunter.vision.chest_detector import iou
from chest_hunter.vision.perception_pipeline import PerceptionResult

Box = tuple[float, float, float, float]


@dataclass
class VisionEpisodeMetrics:
    frames: int = 0
    frames_failed: int = 0
    detections: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    class_errors: int = 0
    marker_found: int = 0
    measurement_errors: list[float] = field(default_factory=list)
    estimate_errors: list[float] = field(default_factory=list)
    inference_seconds: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else None

    @property
    def recall(self) -> float | None:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else None

    @property
    def localisation_success_rate(self) -> float | None:
        return self.marker_found / self.frames if self.frames else None

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "max": None}
        return {"mean": statistics.fmean(values), "median": statistics.median(values),
                "max": max(values)}

    def summary(self) -> dict[str, Any]:
        inference_ms = [s * 1000 for s in self.inference_seconds]
        return {
            "frames_analysed": self.frames,
            "frames_failed": self.frames_failed,
            "detections": self.detections,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "class_errors": self.class_errors,
            "precision": self.precision,
            "recall": self.recall,
            "localisation_success_rate": self.localisation_success_rate,
            "marker_error_cells": self._stats(self.measurement_errors),
            "estimate_error_cells": self._stats(self.estimate_errors),
            "inference_ms": {**self._stats(inference_ms),
                             "peak": max(inference_ms) if inference_ms else None},
        }


@dataclass(frozen=True)
class _Truth:
    box: Box
    category: str
    visible_fraction: float


class VisionEvaluator:
    """Scores perception results against the environment (evaluation only)."""

    def __init__(self, env: Any, controller: Any) -> None:
        self._env = env
        self._controller = controller
        self.metrics = VisionEpisodeMetrics()

    def _ground_truth(self, result: PerceptionResult) -> list[_Truth]:
        camera = result.observation.camera
        width, height = result.observation.width, result.observation.height
        ppc = camera.pixels_per_cell
        raw: list[tuple[Box, str, float]] = []
        for chest in self._env.chests.values():
            if not chest.present or chest.id in self._env.hidden_chests:
                continue
            if not camera.contains(chest.position_x, chest.position_y, margin=-1.0):
                continue
            u, v = camera.world_to_pixel(chest.position_x, chest.position_y)
            x0, y0, x1, y1 = sprites.chest_box(u, v, ppc)
            cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(width, x1), min(height, y1)
            if cx1 - cx0 < 1 or cy1 - cy0 < 1:
                continue
            visible = (cx1 - cx0) * (cy1 - cy0) / ((x1 - x0) * (y1 - y0))
            raw.append(((cx0, cy0, cx1 - cx0, cy1 - cy0),
                        "chest_opened" if chest.opened else "chest", visible))
        merged: list[_Truth] = []
        for box, category, visible in raw:   # merge stacked sprites
            for i, other in enumerate(merged):
                if iou(_int(box), _int(other.box)) > 0.3:
                    top = "chest" if "chest" in (category, other.category) else category
                    merged[i] = _Truth(other.box, top, max(visible, other.visible_fraction))
                    break
            else:
                merged.append(_Truth(box, category, visible))
        return merged

    def observe(self, result: PerceptionResult) -> None:
        m = self.metrics
        m.frames += 1
        m.inference_seconds.append(result.total_seconds)
        truth = self._ground_truth(result)
        m.detections += len(result.detections)
        used: set[int] = set()
        for det in sorted(result.detections, key=lambda d: -d.confidence):
            best_j, best_iou = None, 0.4
            for j, item in enumerate(truth):
                if j in used:
                    continue
                overlap = iou(det.bbox, _int(item.box))
                if overlap >= best_iou:
                    best_j, best_iou = j, overlap
            if best_j is None:
                m.false_positives += 1
            else:
                used.add(best_j)
                m.true_positives += 1
                if truth[best_j].category != det.category:
                    m.class_errors += 1
        m.false_negatives += sum(1 for j, item in enumerate(truth)
                                 if j not in used and item.visible_fraction >= 0.5)
        agent = self._env.agent
        if result.pose is not None:
            m.marker_found += 1
            m.measurement_errors.append(math.dist((result.pose.x, result.pose.y), agent.position))
        estimate = self._controller.world.pose.position
        if estimate is not None:
            m.estimate_errors.append(math.dist(estimate, agent.position))

    def finish(self) -> None:
        self.metrics.frames_failed = self._controller.failed_frames

    def summary(self) -> dict[str, Any]:
        return self.metrics.summary()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self.metrics)
        for key in ("measurement_errors", "estimate_errors", "inference_seconds"):
            data.pop(key)
        return data


def _int(box: Box) -> tuple[int, int, int, int]:
    return (int(round(box[0])), int(round(box[1])), max(1, int(round(box[2]))),
            max(1, int(round(box[3]))))
