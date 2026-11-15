"""Reliability estimation for the learned policy (Phase 6).

Hybrid control uses the learned approach/align commands when the network's auxiliary phase
output says ``P(approach or align) >= 0.6`` (Phase 5). A softmax output is not a
calibrated confidence by construction, so this module measures, on held-out demonstration
episodes, two things:

1. **Phase calibration** - is ``P(approach or align)`` the frequency with which the expert
   really was approaching or aligning? (expected calibration error, ECE)
2. **Action reliability** - in the approach/align situations hybrid control cares about,
   how often is the learned command close to the expert's (largest absolute error of the
   four continuous commands at most ``tolerance``), and how well does ``P`` predict that?

:class:`ReliabilityCalibrator` maps ``P`` to an estimated probability that the command is
reliable by histogram binning (Laplace-smoothed bin frequencies), fitted on the validation
episodes and evaluated on the test episodes, never on training data. Using it in the hybrid
gate is opt-in (``PolicySelector(reliability=...)``); the Phase 5 rule stays the default.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GATE_PHASES = ("approach", "align")


@dataclass(frozen=True, slots=True)
class ReliabilityCalibrator:
    edges: tuple[float, ...]            # bin edges over [0, 1]
    rates: tuple[float, ...]            # smoothed reliable fraction per bin
    counts: tuple[int, ...]
    tolerance: float

    @classmethod
    def fit(cls, scores: np.ndarray, reliable: np.ndarray, *, bins: int = 10,
            tolerance: float = 0.25) -> ReliabilityCalibrator:
        if len(scores) != len(reliable) or len(scores) == 0:
            raise ValueError("scores and labels must be non-empty and of equal length")
        edges = np.linspace(0.0, 1.0, bins + 1)
        idx = np.clip(np.digitize(scores, edges[1:-1]), 0, bins - 1)
        overall = float(np.mean(reliable))
        rates, counts = [], []
        for b in range(bins):
            mask = idx == b
            n = int(mask.sum())
            k = float(reliable[mask].sum())
            rates.append((k + overall) / (n + 1.0))          # shrink sparse bins to the mean
            counts.append(n)
        return cls(tuple(float(e) for e in edges), tuple(rates), tuple(counts), tolerance)

    def predict(self, score: float | np.ndarray) -> np.ndarray:
        edges = np.asarray(self.edges)
        idx = np.clip(np.digitize(np.asarray(score, dtype=np.float64), edges[1:-1]), 0,
                      len(self.rates) - 1)
        return np.asarray(self.rates)[idx]

    def to_dict(self) -> dict[str, Any]:
        return {"edges": list(self.edges), "rates": list(self.rates),
                "counts": list(self.counts), "tolerance": self.tolerance}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReliabilityCalibrator:
        return cls(tuple(data["edges"]), tuple(data["rates"]), tuple(data["counts"]),
                   float(data["tolerance"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ReliabilityCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def expected_calibration_error(prob: np.ndarray, outcome: np.ndarray, bins: int = 10
                               ) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(float(prob[mask].mean()) - float(outcome[mask].mean()))
    return total


def brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((prob - outcome.astype(np.float64)) ** 2))


def auc(score: np.ndarray, outcome: np.ndarray) -> float | None:
    """Area under the ROC curve (rank statistic, ties averaged)."""
    pos = outcome.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def collect(policy: Any, samples: Sequence[Any], *, batch_size: int = 128
            ) -> dict[str, np.ndarray]:
    """Model outputs on demonstration samples: P(approach or align), the largest absolute
    command error against the expert, and whether the expert was approaching/aligning."""
    import cv2

    from chest_hunter.learning.demonstration_recorder import PHASES

    gate = [PHASES.index(p) for p in GATE_PHASES]
    scores, errors, in_gate = [], [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        frames: list[np.ndarray] = []
        for s in chunk:
            img = cv2.imread(s.frame_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"unreadable demonstration frame {s.frame_path}")
            frames.append(img)
        images = np.stack(frames)
        proprio = np.stack([s.proprio for s in chunk]).astype(np.float32)
        a, _b, p = _raw_batched(policy, policy.norm.apply(images), proprio)
        expert = np.stack([s.action[:4] for s in chunk])
        scores.append(p[:, gate].sum(axis=1))
        errors.append(np.abs(a[:, :4] - expert).max(axis=1))
        in_gate.append(np.array([s.phase in gate for s in chunk]))
    return {"score": np.concatenate(scores), "error": np.concatenate(errors),
            "in_gate": np.concatenate(in_gate)}


def _raw_batched(policy: Any, images: np.ndarray, proprio: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        a, b, p = policy.raw(images, proprio)
        return np.asarray(a), np.asarray(b), np.asarray(p)
    except Exception:                        # an export without a dynamic batch dimension
        outs = [policy.raw(images[i:i + 1], proprio[i:i + 1]) for i in range(len(images))]
        return tuple(np.concatenate([np.asarray(o[k]) for o in outs]) for k in range(3))  # type: ignore[return-value]


def evaluate(policy: Any, validation: Sequence[Any], test: Sequence[Any], *,
             tolerance: float = 0.25, gate: float = 0.6, bins: int = 10
             ) -> tuple[ReliabilityCalibrator, dict[str, Any]]:
    """Fit the calibrator on ``validation`` and report raw vs calibrated on ``test``."""
    val, tst = collect(policy, validation), collect(policy, test)
    report: dict[str, Any] = {"tolerance": tolerance, "gate": gate, "bins": bins,
                              "validation_samples": int(len(val["score"])),
                              "test_samples": int(len(tst["score"]))}
    report["phase_calibration_ece_test"] = round(expected_calibration_error(
        tst["score"], tst["in_gate"], bins), 4)
    vm, tm = val["in_gate"], tst["in_gate"]
    v_rel = (val["error"][vm] <= tolerance).astype(np.float64)
    t_rel = (tst["error"][tm] <= tolerance).astype(np.float64)
    t_score = tst["score"][tm]
    cal = ReliabilityCalibrator.fit(val["score"][vm], v_rel, bins=bins, tolerance=tolerance)
    calibrated = cal.predict(t_score)
    above = t_score >= gate
    report.update({
        "approach_align_samples_test": int(tm.sum()),
        "reliable_fraction_test": round(float(t_rel.mean()), 4) if len(t_rel) else None,
        "raw_as_reliability": {"ece": round(expected_calibration_error(t_score, t_rel, bins), 4),
                               "brier": round(brier(t_score, t_rel), 4)},
        "calibrated": {"ece": round(expected_calibration_error(calibrated, t_rel, bins), 4),
                       "brier": round(brier(calibrated, t_rel), 4)},
        "auc_raw_score_for_reliability": None if auc(t_score, t_rel) is None
        else round(float(auc(t_score, t_rel) or 0.0), 4),
        "gate_above": {"share": round(float(above.mean()), 4) if len(above) else None,
                       "reliable_fraction": round(float(t_rel[above].mean()), 4)
                       if above.any() else None},
        "gate_below": {"share": round(float((~above).mean()), 4) if len(above) else None,
                       "reliable_fraction": round(float(t_rel[~above].mean()), 4)
                       if (~above).any() else None},
    })
    return cal, report
