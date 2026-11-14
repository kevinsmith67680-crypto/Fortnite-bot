"""Objective tracking of the learning pipeline: training history, validation and test
metrics against trivial baselines, export checks, and spawn-estimate accuracy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def training_report(model_dir: str | Path) -> dict[str, Any]:
    """Summary of a trained model directory (policy.json, history.json, export.json)."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "policy.json").read_text(encoding="utf-8"))
    history_path = model_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() \
        else []
    export_path = model_dir / "export.json"
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.exists() else {}
    metrics = meta.get("metrics", {})
    test, base = metrics.get("test", {}), metrics.get("baseline_test", {})
    improvement = {k: round(1.0 - test[k] / base[k], 4)
                   for k in base if k.startswith("mae_") and k in test and base[k] > 0}
    dataset = meta.get("dataset", {})
    return {
        "model_version": meta.get("model_version"), "parameters": meta.get("parameters"),
        "dataset": {"episodes": dataset.get("episodes"), "samples": dataset.get("samples"),
                    "maps": dataset.get("maps"), "splits": dataset.get("splits"),
                    "synthetic_only": dataset.get("synthetic_only")},
        "epochs_run": meta.get("epochs_run"), "best_epoch": meta.get("best_epoch"),
        "stopped_early": meta.get("stopped_early"),
        "loss_curve": [{"epoch": h["epoch"], "train": h["train_loss"],
                        "validation": h["validation_loss"]} for h in history],
        "validation": metrics.get("validation"), "test": test, "baseline_test": base,
        "mae_reduction_vs_mean_baseline": improvement,
        "export": export,
    }


def spawn_estimate_error(estimates: dict[int, Any], truth: dict[str, float],
                         names: dict[int, str]) -> dict[str, float]:
    """Mean absolute error of posterior means against known true probabilities (only
    available for generated maps, where the generator knows them)."""
    errs, prior_errs = [], []
    for lid, e in estimates.items():
        name = names.get(lid)
        if name is None or name not in truth:
            continue
        errs.append(abs(e.mean - truth[name]))
        prior_errs.append(abs(e.configured_probability - truth[name]))
    if not errs:
        return {"locations": 0.0}
    return {"locations": float(len(errs)), "posterior_mae": round(sum(errs) / len(errs), 4),
            "configured_prior_mae": round(sum(prior_errs) / len(prior_errs), 4)}


def progress_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Throughput per strategy and episode index (learning progress across experiments)."""
    out: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        out.setdefault((r["strategy"], int(r["order"])), []).append(float(r["throughput_per_min"]))
    return [{"strategy": s, "order": o, "mean_throughput": round(sum(v) / len(v), 4),
             "maps": len(v)} for (s, o), v in sorted(out.items())]
