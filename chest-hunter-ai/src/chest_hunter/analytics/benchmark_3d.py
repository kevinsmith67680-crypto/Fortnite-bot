"""Phase 4 benchmark: oracle vs hybrid visual vs visual-navigation on paired 3D episodes.

Every episode index runs all modes with the same seed, so chest spawns,
start position, map and episode budget are identical across modes. Episodes
are distributed over worker processes (each runs its own loopback reference
server); results are deterministic per episode regardless of the worker.

Outputs (``output_dir``): ``navigation_3d.csv`` (one row per episode and
mode), ``vision_3d.csv`` (one row per episode for the visual modes) and
``summary_3d.json`` (aggregates, paired differences, run cost, the
configuration and the environment the numbers come from).

Time basis: chests per minute, seconds and travel use simulated time; CPU
and wall seconds are real time on the benchmark machine.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chest_hunter.agent.runner_3d import MODES_3D, Runner3D, peak_rss_mb
from chest_hunter.config import Settings, load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ValidationError
from chest_hunter.scenarios import seed_arena_3d

NAVIGATION_FIELDS_3D = (
    "episode", "mode", "outcome", "end_reason", "sim_seconds", "chests_present",
    "chests_available", "chests_seen", "chests_opened", "chests_opened_truth",
    "percent_available_opened", "percent_seen_opened", "chests_per_minute", "travel_distance_m",
    "recoveries", "navigation_failures", "unreachable_targets", "abandoned_targets",
    "failed_interactions", "targets_selected", "targets_reached", "navigation_success_rate",
    "analyses", "perception_ms_mean", "bridge_rtt_ms", "cpu_seconds", "wall_seconds",
    "cpu_utilisation", "peak_rss_mb",
)
VISION_FIELDS_3D = (
    "episode", "mode", "frames", "precision", "recall", "class_errors",
    "distance_mean_abs_error_m", "distance_median_abs_error_m", "distance_failure_rate",
    "distance_err_0_5m", "distance_err_5_10m", "distance_err_10_20m", "distance_err_20_40m",
    "pose_error_mean_m", "pose_error_max_m",
)


def _episode(settings_path: str, map_id: str, seed: int, episode: int, modes: Sequence[str],
             max_ticks: int, width: int, height: int) -> list[dict[str, Any]]:
    settings = load_config(settings_path)
    rows = []
    for mode in modes:
        with DatabaseManager(":memory:") as db:
            db.initialize()
            seed_arena_3d(db)
            runner = Runner3D.from_settings(db, settings, mode=mode, seed=seed, episode=episode,
                                            backend="reference", evaluate=True,
                                            max_ticks=max_ticks, width=width, height=height,
                                            map_id=map_id)
            r = runner.run()
        m, t = r.metrics, r.truth
        available = set(t.get("available_names", []))
        seen = set(t.get("seen_names", [])) & available
        opened = set(t.get("opened_names", []))
        row: dict[str, Any] = {
            "episode": episode, "mode": mode, "outcome": r.outcome, "end_reason": r.end_reason,
            "sim_seconds": round(r.sim_seconds, 3), "chests_present": t.get("present"),
            "chests_available": len(available),
            "chests_seen": len(seen) if mode != "oracle" else None,
            "chests_opened": r.chests_opened, "chests_opened_truth": t.get("opened_truth"),
            "percent_available_opened": round(100.0 * len(opened & available) / len(available), 2)
            if available else None,
            "percent_seen_opened": round(100.0 * len(opened & seen) / len(seen), 2)
            if seen and mode != "oracle" else None,
            "chests_per_minute": round(r.chests_per_minute or 0.0, 4),
            "travel_distance_m": round(m.travel_distance, 2), "recoveries": m.recoveries,
            "navigation_failures": m.navigation_failures,
            "unreachable_targets": m.unreachable_targets, "abandoned_targets": m.abandoned_targets,
            "failed_interactions": m.failed_interactions, "targets_selected": m.targets_selected,
            "targets_reached": m.targets_reached,
            "navigation_success_rate": round(min(1.0, m.targets_reached / m.targets_selected), 4)
            if m.targets_selected else None,
            "analyses": m.analyses, "perception_ms_mean": round(m.perception_ms_mean, 3),
            "bridge_rtt_ms": round(r.bridge_round_trip_ms, 3),
            "cpu_seconds": round(r.cpu_seconds, 3), "wall_seconds": round(r.wall_seconds, 3),
            "cpu_utilisation": round(r.cpu_seconds / r.wall_seconds, 3) if r.wall_seconds else None,
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
        if r.vision is not None:
            v = r.vision
            bins = v.get("distance_by_range", {})
            row["_vision"] = {
                "episode": episode, "mode": mode, "frames": v["frames"],
                "precision": v["precision"], "recall": v["recall"],
                "class_errors": v["class_errors"],
                "distance_mean_abs_error_m": v["distance_mean_abs_error_m"],
                "distance_median_abs_error_m": v["distance_median_abs_error_m"],
                "distance_failure_rate": v["distance_failure_rate"],
                "distance_err_0_5m": bins.get("0-5m", {}).get("mean_abs_error_m"),
                "distance_err_5_10m": bins.get("5-10m", {}).get("mean_abs_error_m"),
                "distance_err_10_20m": bins.get("10-20m", {}).get("mean_abs_error_m"),
                "distance_err_20_40m": bins.get("20-40m", {}).get("mean_abs_error_m"),
                "pose_error_mean_m": v["pose_error_mean_m"],
                "pose_error_max_m": v["pose_error_max_m"],
                "_tp": v["true_positives"], "_fp": v["false_positives"],
                "_fn": v["false_negatives"],
            }
        rows.append(row)
    return rows


def _agg(values: list[float]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": round(statistics.fmean(vals), 4), "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4), "max": round(max(vals), 4)}


def run_benchmark_3d(settings: Settings, *, map_id: str = "research_arena_3d",
                     modes: Sequence[str] = MODES_3D, episodes: int = 50, seed: int = 42,
                     max_ticks: int = 18000, width: int = 640, height: int = 360,
                     workers: int = 1, output_dir: Path | None = None,
                     progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    modes = tuple(m.replace("-", "_") for m in modes)
    bad = [m for m in modes if m not in MODES_3D]
    if bad:
        raise ValidationError(f"unknown mode(s) {bad}; choose from {', '.join(MODES_3D)}")
    if episodes < 1:
        raise ValidationError("episodes must be at least 1")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    rows: list[dict[str, Any]] = []
    args = [(str(settings.config_path), map_id, seed, e, modes, max_ticks, width, height)
            for e in range(episodes)]
    if workers <= 1:
        for a in args:
            rows += _episode(*a)
            if progress:
                progress(f"  ... episode {a[3] + 1}/{episodes}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, result in enumerate(pool.map(_episode, *zip(*args, strict=True))):
                rows += result
                if progress:
                    progress(f"  ... episode {k + 1}/{episodes}")
    rows.sort(key=lambda r: (r["episode"], modes.index(r["mode"])))
    vision = [r.pop("_vision") for r in rows if "_vision" in r]
    summary = _summarise(rows, vision, modes, episodes)
    summary.update({
        "map_id": map_id, "seed": seed, "episodes_requested": episodes,
        "episodes_completed": len({r["episode"] for r in rows}), "modes": list(modes),
        "max_ticks": max_ticks, "observation": [width, height], "workers": workers,
        "environment": "Python reference arena (protocol-identical mock of the Unreal "
                       "research server); NOT Unreal Engine",
        "time_basis": "rates use simulated time; cpu/wall seconds are real time",
        "gpu": "not measured: no GPU on the benchmark machine; the reference renderer is "
               "CPU-only software rasterisation",
        "run": {"wall_seconds": round(time.perf_counter() - started_wall, 2),
                "cpu_seconds_main": round(time.process_time() - started_cpu, 2),
                "cpu_seconds_episodes": round(sum(r["cpu_seconds"] for r in rows), 2),
                "peak_rss_mb_max_worker": max((r["peak_rss_mb"] for r in rows), default=None),
                "machine": f"{platform.machine()} {os.cpu_count()} CPU(s), "
                           f"Python {platform.python_version()} on {sys.platform}"},
        "configuration": {"simulation_3d": asdict(settings.simulation_3d)},
    })
    if output_dir is not None:
        write_outputs_3d(output_dir, rows, vision, summary)
    summary["_rows"] = rows
    summary["_vision_rows"] = vision
    return summary


def _summarise(rows: list[dict[str, Any]], vision: list[dict[str, Any]], modes: Sequence[str],
               episodes: int) -> dict[str, Any]:
    per_mode: dict[str, Any] = {}
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        entry: dict[str, Any] = {k: _agg([r[k] for r in mr]) for k in (
            "chests_per_minute", "percent_available_opened", "percent_seen_opened", "sim_seconds",
            "travel_distance_m", "recoveries", "navigation_failures", "navigation_success_rate",
            "perception_ms_mean", "bridge_rtt_ms", "cpu_seconds", "wall_seconds",
            "cpu_utilisation", "peak_rss_mb")}
        entry["outcomes"] = {o: sum(1 for r in mr if r["outcome"] == o)
                             for o in sorted({r["outcome"] for r in mr})}
        entry["opened_total"] = sum(r["chests_opened"] for r in mr)
        entry["opened_truth_total"] = sum(r["chests_opened_truth"] or 0 for r in mr)
        entry["available_total"] = sum(r["chests_available"] for r in mr)
        entry["seen_total"] = sum(r["chests_seen"] or 0 for r in mr)
        entry["over_reported_total"] = sum(
            max(0, r["chests_opened"] - (r["chests_opened_truth"] or 0)) for r in mr)
        entry["failed_interactions_total"] = sum(r["failed_interactions"] for r in mr)
        vr = [v for v in vision if v["mode"] == mode]
        if vr:
            tp, fp, fn = (sum(v[k] for v in vr) for k in ("_tp", "_fp", "_fn"))
            entry["vision"] = {
                "precision_pooled": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall_pooled": round(tp / (tp + fn), 4) if tp + fn else None,
                "class_errors_total": sum(v["class_errors"] for v in vr),
                **{k: _agg([v[k] for v in vr]) for k in (
                    "distance_mean_abs_error_m", "distance_median_abs_error_m",
                    "distance_failure_rate", "distance_err_0_5m", "distance_err_5_10m",
                    "distance_err_10_20m", "distance_err_20_40m", "pose_error_mean_m",
                    "pose_error_max_m")},
            }
        per_mode[mode] = entry
    paired: dict[str, Any] = {}
    if "oracle" in modes:
        base = {r["episode"]: r for r in rows if r["mode"] == "oracle"}
        for mode in modes:
            if mode == "oracle":
                continue
            diffs = [r["chests_per_minute"] - base[r["episode"]]["chests_per_minute"]
                     for r in rows if r["mode"] == mode and r["episode"] in base]
            if diffs:
                paired[mode] = {"mean_difference_chests_per_minute":
                                    round(statistics.fmean(diffs), 4),
                                "median_difference_chests_per_minute":
                                    round(statistics.median(diffs), 4),
                                "better": sum(d > 1e-9 for d in diffs),
                                "worse": sum(d < -1e-9 for d in diffs),
                                "tied": sum(abs(d) <= 1e-9 for d in diffs)}
    return {"per_mode": per_mode, "paired_vs_oracle": paired}


def write_outputs_3d(output_dir: Path, rows: list[dict[str, Any]], vision: list[dict[str, Any]],
                     summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fields, data in (("navigation_3d.csv", NAVIGATION_FIELDS_3D, rows),
                               ("vision_3d.csv", VISION_FIELDS_3D, vision)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    clean = {k: v for k, v in summary.items() if not k.startswith("_")}
    (output_dir / "summary_3d.json").write_text(json.dumps(clean, indent=2) + "\n",
                                                encoding="utf-8")


def format_summary_3d(summary: dict[str, Any]) -> str:
    def f(x: Any, d: int = 3) -> str:
        return "-" if x is None else f"{x:.{d}f}"

    lines = [f"3D benchmark on {summary['map_id']} ({summary['environment']})",
             f"{summary['episodes_completed']} of {summary['episodes_requested']} paired episodes, "
             f"seed {summary['seed']}, modes {', '.join(summary['modes'])}", "",
             f"{'Mode':11s} {'Chests/min':>10s} {'median':>7s} {'%avail':>7s} {'%seen':>6s} "
             f"{'NavFail':>7s} {'Precision':>9s} {'Recall':>6s} {'DistErr m':>9s} "
             f"{'Percep ms':>9s} {'CPU s/ep':>8s}"]
    for mode, e in summary["per_mode"].items():
        v = e.get("vision", {})
        lines.append(
            f"{mode:11s} {f(e['chests_per_minute']['mean']):>10s} "
            f"{f(e['chests_per_minute']['median']):>7s} "
            f"{f(e['percent_available_opened']['mean'], 1):>7s} "
            f"{f(e['percent_seen_opened']['mean'], 1):>6s} "
            f"{f(e['navigation_failures']['mean'], 1):>7s} {f(v.get('precision_pooled')):>9s} "
            f"{f(v.get('recall_pooled')):>6s} "
            f"{f((v.get('distance_median_abs_error_m') or {}).get('mean'), 2):>9s} "
            f"{f(e['perception_ms_mean']['mean'], 2):>9s} {f(e['cpu_seconds']['mean'], 1):>8s}")
    for mode, e in summary["per_mode"].items():
        lines.append(f"{mode} openings: recorded {e['opened_total']}, simulator "
                     f"{e['opened_truth_total']} (over-reported {e['over_reported_total']}); "
                     f"outcomes {e['outcomes']}")
    for mode, c in summary.get("paired_vs_oracle", {}).items():
        lines.append(f"{mode} vs oracle: mean {c['mean_difference_chests_per_minute']:+.3f} "
                     f"chests/min; better {c['better']}, worse {c['worse']}, tied {c['tied']}")
    run = summary["run"]
    lines.append(f"Run: {run['wall_seconds']:.0f} s wall, {run['cpu_seconds_episodes']:.0f} s "
                 f"episode CPU, peak worker RSS {run['peak_rss_mb_max_worker']} MiB; "
                 f"{run['machine']}. GPU: {summary['gpu']}.")
    return "\n".join(lines)
