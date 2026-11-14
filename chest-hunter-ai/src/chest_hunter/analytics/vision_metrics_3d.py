"""Evaluation of 3D perception against ground-truth labels (evaluation code only).

The labels come from the server's privileged evaluation channel
(``GET_EVALUATION``), which exists only when the server was started in
evaluation mode. The policy never sees them: the runner calls this evaluator
*after* the agent has processed a frame.

Per frame:

* **Detection**: a labelled chest is *required* when at least 40 pixels and at
  least half of its expected area are visible within 30 m; smaller or more
  occluded chests are optional (a match counts, a miss does not). Matching is
  greedy at IoU >= 0.3 (boxes of partially hidden objects are small).
* **Distance**: for matched detections with an estimate, the absolute error
  between the estimated and true camera-to-chest distance, binned by range.
* **Pose**: odometry position against the true character position.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from chest_hunter.vision.chest_detector import iou
from chest_hunter.vision.chest_detector_3d import Detection3D
from chest_hunter.vision.depth_estimation import RangeEstimate

RANGE_BINS = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0))


@dataclass(slots=True)
class Evaluator3D:
    fx: float
    fy: float
    frames: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    class_errors: int = 0
    estimate_failures: int = 0
    distance_errors: list[tuple[float, float]] = field(default_factory=list)   # (true, abs err)
    pose_errors: list[float] = field(default_factory=list)

    def observe(self, labels: dict[str, Any], detections: list[Detection3D],
                estimates: list[RangeEstimate], odometry: tuple[float, float, float]) -> None:
        self.frames += 1
        truth = []
        for lab in labels["chests"]:
            d = max(0.5, float(lab["distance"]))
            expected = (self.fx * 0.95 / d) * (self.fy * 0.75 / d) * 0.85
            required = lab["pixels"] >= 40 and d <= 30.0 and lab["pixels"] >= 0.5 * expected
            truth.append((lab, required))
        used: set[int] = set()
        for k, det in enumerate(detections):
            best, bi = 0.0, None
            for j, (lab, _) in enumerate(truth):
                if j in used:
                    continue
                v = iou(det.bbox, tuple(lab["bbox"]))
                if v > best:
                    best, bi = v, j
            if bi is None or best < 0.3:
                self.fp += 1
                continue
            used.add(bi)
            self.tp += 1
            lab = truth[bi][0]
            if det.category != lab["category"]:
                self.class_errors += 1
            est = estimates[k] if k < len(estimates) else None
            if est is None:
                self.estimate_failures += 1
            else:
                self.distance_errors.append((float(lab["distance"]),
                                             abs(est.distance - float(lab["distance"]))))
        self.fn += sum(1 for j, (_, req) in enumerate(truth) if req and j not in used)
        ax, ay, az = labels["agent_position"]
        self.pose_errors.append(math.dist((ax, ay, az), odometry))

    def summary(self) -> dict[str, Any]:
        errs = [e for _, e in self.distance_errors]
        bins = {}
        for lo, hi in RANGE_BINS:
            sub = [e for d, e in self.distance_errors if lo <= d < hi]
            bins[f"{lo:g}-{hi:g}m"] = {"n": len(sub),
                                      "mean_abs_error_m": _r(statistics.fmean(sub)) if sub else None,
                                      "median_abs_error_m": _r(statistics.median(sub)) if sub else None}
        matched = self.tp
        return {
            "frames": self.frames,
            "true_positives": self.tp, "false_positives": self.fp, "false_negatives": self.fn,
            "class_errors": self.class_errors,
            "precision": _r(self.tp / (self.tp + self.fp)) if self.tp + self.fp else None,
            "recall": _r(self.tp / (self.tp + self.fn)) if self.tp + self.fn else None,
            "distance_mean_abs_error_m": _r(statistics.fmean(errs)) if errs else None,
            "distance_median_abs_error_m": _r(statistics.median(errs)) if errs else None,
            "distance_by_range": bins,
            "distance_failure_rate": _r(self.estimate_failures / matched) if matched else None,
            "pose_error_mean_m": _r(statistics.fmean(self.pose_errors)) if self.pose_errors else None,
            "pose_error_max_m": _r(max(self.pose_errors)) if self.pose_errors else None,
        }


def _r(value: float) -> float:
    return round(float(value), 4)
