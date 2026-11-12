"""Paired benchmark of oracle, chest-vision and full visual agents.

Modes (same map, obstacles, chest spawns, start position, speed and tick
budget per episode index):

* ``oracle`` (A): Phase 2 agent with true spawn points and obstacles;
* ``chest_vision`` (B): chests and pose from camera frames, obstacles given;
* ``visual`` (C): chests, pose *and* obstacles from camera frames.

Outputs (in ``output_dir``):

* ``navigation_benchmark.csv``: one row per (episode, mode): outcome, chests
  opened vs available, chests per *simulated* minute, time per chest,
  recoveries, navigation success rate, travel, CPU time, and so on;
* ``vision_benchmark.csv``: one row per (episode, visual mode): detection
  precision/recall/false positives, localisation errors, inference time;
* ``summary.json``: configuration, units, per-mode aggregates, paired
  differences against the oracle, a detector-method comparison and
  run-time/memory measurements.

Time basis: all rates use **simulated** seconds (fixed timestep). CPU and
wall-clock seconds measure the real cost of running the benchmark.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import resource
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chest_hunter.agent.runner import AGENT_KINDS, SimulationRunner, load_world, make_oracle
from chest_hunter.analytics.benchmark import clone_map_chests
from chest_hunter.config import Settings
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ChestHunterError, ValidationError
from chest_hunter.models import utc_now
from chest_hunter.navigation.astar import distance_field
from chest_hunter.scenarios import SCENARIOS, seed_scenario

NAVIGATION_FIELDS = (
    "episode", "mode", "outcome", "end_reason", "simulated_seconds", "ticks",
    "spawn_locations", "chests_spawned", "chests_available", "chests_opened",
    "chests_opened_truth", "percent_available_opened", "chests_per_minute", "seconds_per_chest",
    "failed_interactions", "recoveries", "unreachable_targets", "abandoned_targets",
    "targets_selected", "targets_reached", "navigation_success_rate",
    "locations_recorded", "travel_distance_cells", "astar_searches", "cpu_seconds",
    "wall_seconds",
)
VISION_FIELDS = (
    "episode", "mode", "frames_analysed", "frames_failed", "analyses_per_sim_second",
    "detections", "true_positives", "false_positives", "false_negatives", "class_errors",
    "precision", "recall", "localisation_success_rate", "marker_error_mean_cells",
    "marker_error_max_cells", "estimate_error_mean_cells", "estimate_error_median_cells",
    "estimate_error_max_cells", "inference_ms_mean", "inference_ms_median", "inference_ms_peak",
)
UNITS = {
    "simulated_seconds": "simulated seconds (fixed 1/tick_rate timestep)",
    "chests_per_minute": "chests opened per simulated minute",
    "seconds_per_chest": "simulated seconds per opened chest",
    "travel_distance_cells": "grid cells travelled (true path)",
    "cpu_seconds": "process CPU seconds on the benchmark machine",
    "wall_seconds": "real elapsed seconds on the benchmark machine",
    "*_error_*_cells": "Euclidean position error in grid cells",
    "inference_ms_*": "milliseconds of perception per analysed frame",
    "chests_available": "spawned chests reachable from the start (ground truth)",
    "chests_opened": "openings the agent verified and recorded (its own evidence)",
    "chests_opened_truth": "chests the simulator marks as opened (ground truth, evaluation only)",
    "navigation_success_rate": "selected chest targets that were reached / selected",
    "peak_rss_mb": "peak resident memory of the benchmark process (MiB)",
}


def _reachable_available(runner: SimulationRunner) -> tuple[int, int]:
    """(spawned chests, spawned chests reachable from the start), from ground truth."""
    world, env = runner.world, runner.env
    field = distance_field(world.grid, world.start_cell)
    spawned = reachable = 0
    for chest in env.chests.values():
        if not chest.present:
            continue
        spawned += 1
        cx, cy = chest.cell
        if math.isfinite(field[cy * world.width + cx]):
            reachable += 1
    return spawned, reachable


def _navigation_row(runner: SimulationRunner, episode: int, mode: str) -> dict[str, Any]:
    m = runner.metrics
    controller = runner.controller
    assert controller is not None
    spawned, available = _reachable_available(runner)
    selected: int
    reached: int
    if runner.is_visual:
        visual = runner.visual_controller
        selected, reached = visual.targets_selected, visual.targets_reached
    else:
        selected = m.target_selections
        reached = sum(1 for t in controller.history if t.target.value == "VERIFYING_CHEST")
    return {
        "episode": episode,
        "mode": mode,
        "outcome": m.outcome,
        "end_reason": controller.end_reason,
        "simulated_seconds": round(m.simulated_seconds, 4),
        "ticks": m.ticks,
        "spawn_locations": m.candidate_locations,
        "chests_spawned": spawned,
        "chests_available": available,
        "chests_opened": m.chests_opened,
        "chests_opened_truth": sum(1 for c in runner.env.chests.values() if c.opened),
        "percent_available_opened": round(100.0 * m.chests_opened / available, 3)
        if available else None,
        "chests_per_minute": round(m.chests_per_minute or 0.0, 4),
        "seconds_per_chest": round(m.simulated_seconds / m.chests_opened, 4)
        if m.chests_opened else None,
        "failed_interactions": m.failed_interactions,
        "recoveries": m.recoveries,
        "unreachable_targets": m.unreachable_targets,
        "abandoned_targets": m.abandoned_targets,
        "targets_selected": selected,
        "targets_reached": reached,
        "navigation_success_rate": round(reached / selected, 4) if selected else None,
        "locations_recorded": m.locations_visited,
        "travel_distance_cells": round(m.travel_distance, 3),
        "astar_searches": m.astar_searches,
        "cpu_seconds": round(m.cpu_seconds, 4),
        "wall_seconds": round(m.wall_seconds, 4),
    }


def _vision_row(runner: SimulationRunner, episode: int, mode: str) -> dict[str, Any]:
    summary = runner.vision_evaluator.summary()
    sim_seconds = runner.metrics.simulated_seconds or 1.0

    def r(value: Any, digits: int = 4) -> Any:
        return None if value is None else round(value, digits)

    return {
        "episode": episode,
        "mode": mode,
        "frames_analysed": summary["frames_analysed"],
        "frames_failed": summary["frames_failed"],
        "analyses_per_sim_second": r(summary["frames_analysed"] / sim_seconds),
        "detections": summary["detections"],
        "true_positives": summary["true_positives"],
        "false_positives": summary["false_positives"],
        "false_negatives": summary["false_negatives"],
        "class_errors": summary["class_errors"],
        "precision": r(summary["precision"]),
        "recall": r(summary["recall"]),
        "localisation_success_rate": r(summary["localisation_success_rate"]),
        "marker_error_mean_cells": r(summary["marker_error_cells"]["mean"], 5),
        "marker_error_max_cells": r(summary["marker_error_cells"]["max"], 5),
        "estimate_error_mean_cells": r(summary["estimate_error_cells"]["mean"], 5),
        "estimate_error_median_cells": r(summary["estimate_error_cells"]["median"], 5),
        "estimate_error_max_cells": r(summary["estimate_error_cells"]["max"], 5),
        "inference_ms_mean": r(summary["inference_ms"]["mean"], 3),
        "inference_ms_median": r(summary["inference_ms"]["median"], 3),
        "inference_ms_peak": r(summary["inference_ms"]["peak"], 3),
    }


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def compare_detectors(settings: Settings, world: Any, *, frames: int = 60,
                      seed: int = 0) -> dict[str, Any]:
    """Methods A/B/C on the same labelled frames (all spawn points hold a chest)."""
    from chest_hunter.analytics.vision_metrics import VisionEvaluator
    from chest_hunter.simulation.camera import SensorModel
    from chest_hunter.simulation.environment import Environment
    from chest_hunter.vision.observation import SimulatorObservationProvider
    from chest_hunter.vision.perception_pipeline import PerceptionPipeline
    from chest_hunter.vision.templates import TemplateLibrary

    vis = settings.vision
    env = Environment(world, settings.simulation, seed=seed)
    rng = random.Random(f"detector-comparison:{seed}")
    for chest in env.chests.values():
        chest.present = True
        chest.opened = rng.random() < 0.4
    library = TemplateLibrary.load(settings.template_dir(), noise_std=vis.noise_std,
                                   lighting_variation=vis.lighting_variation)
    poses = []
    chests = list(env.chests.values())
    for _ in range(frames):
        chest = rng.choice(chests)
        poses.append((min(max(chest.position_x + rng.uniform(-12, 12), 1), world.width - 1),
                      min(max(chest.position_y + rng.uniform(-6, 6), 1), world.height - 1)))
    results: dict[str, Any] = {}

    class _Probe:   # minimal stand-in for the controller (no pose estimate)
        failed_frames = 0

        class world:  # noqa: N801
            class pose:  # noqa: N801
                position = None

    for method in ("color", "template", "hybrid"):
        provider = SimulatorObservationProvider(
            env, width=vis.default_width, height=vis.default_height,
            view_width_cells=vis.view_width_cells,
            sensor=SensorModel(vis.noise_std, vis.lighting_variation, 0.0), seed=seed)
        pipeline = PerceptionPipeline(library, detector=method,
                                      confidence_threshold=vis.confidence_threshold,
                                      update_occupancy=False)
        evaluator = VisionEvaluator(env, _Probe())
        from chest_hunter.vision.obstacle_detector import OccupancyGrid

        grid = OccupancyGrid(world.width, world.height)
        for x, y in poses:
            env.agent.position_x, env.agent.position_y = x, y
            observation = provider.capture()
            assert observation is not None
            evaluator.observe(pipeline.process(observation, grid))
        s = evaluator.summary()
        results[method] = {
            "frames": s["frames_analysed"], "precision": s["precision"], "recall": s["recall"],
            "false_positives": s["false_positives"], "class_errors": s["class_errors"],
            "detector_ms_mean": 1000 * pipeline.chest_detector.total_seconds
            / max(1, pipeline.chest_detector.calls),
        }
    return results


def run_visual_benchmark(
    source_db: DatabaseManager,
    settings: Settings,
    *,
    map_id: str,
    modes: Sequence[str] = AGENT_KINDS,
    episodes: int = 100,
    seed: int = 42,
    max_ticks: int | None = None,
    output_dir: Path | None = None,
    detector_frames: int = 60,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    modes = tuple(dict.fromkeys(m.replace("-", "_") for m in modes))
    if not modes or any(m not in AGENT_KINDS for m in modes):
        raise ValidationError(f"modes must be chosen from {', '.join(AGENT_KINDS)}.")
    if episodes < 1:
        raise ValidationError("episodes must be at least 1.")
    budget = settings.simulation.max_ticks if max_ticks is None else max_ticks
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    db = DatabaseManager(":memory:")
    db.initialize()
    try:
        copied = clone_map_chests(source_db, db, map_id)
        if not copied and map_id in SCENARIOS:
            copied = seed_scenario(db, map_id).inserted
        if not copied:
            raise ChestHunterError(f"Map '{map_id}' has no chest locations to benchmark.")
        world = load_world(db, settings, map_id=map_id, map_seed=seed)
        oracle = make_oracle(settings)
        oracle.warm(world.grid, sorted({c.cell for c in world.chests}))
        clock_base = utc_now()
        navigation: list[dict[str, Any]] = []
        vision: list[dict[str, Any]] = []
        total, done = episodes * len(modes), 0
        for episode in range(episodes):
            for mode in modes:
                runner = SimulationRunner(db, settings, map_id=map_id, seed=seed,
                                          episode=episode, max_ticks=budget, world=world,
                                          oracle=oracle, clock_base=clock_base, agent=mode,
                                          evaluate=mode != "oracle")
                runner.run()
                navigation.append(_navigation_row(runner, episode, mode))
                if runner.is_visual:
                    vision.append(_vision_row(runner, episode, mode))
                done += 1
                if progress is not None:
                    progress(done, total)
        detectors = compare_detectors(settings, world, frames=detector_frames, seed=seed) \
            if detector_frames > 0 else {}
    finally:
        db.close()

    summary: dict[str, Any] = {
        "map_id": map_id,
        "episodes_requested": episodes,
        "episodes_completed": len(navigation) // len(modes),
        "modes": list(modes),
        "seed": seed,
        "max_ticks": budget,
        "time_basis": "rates use simulated time; cpu/wall seconds are real time",
        "units": UNITS,
        "configuration": {
            "simulation": asdict(settings.simulation),
            "vision": asdict(settings.vision),
            "localization": asdict(settings.localization),
            "visual_navigation": asdict(settings.visual_navigation),
            "navigation": asdict(settings.navigation),
        },
        "per_mode": {},
        "paired_vs_oracle": {},
        "detector_comparison": detectors,
        "machine": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    for mode in modes:
        rows = [r for r in navigation if r["mode"] == mode]
        entry: dict[str, Any] = {key: _aggregate(rows, key) for key in (
            "chests_per_minute", "percent_available_opened", "seconds_per_chest",
            "simulated_seconds", "recoveries", "failed_interactions",
            "navigation_success_rate", "travel_distance_cells", "cpu_seconds", "wall_seconds")}
        entry["outcomes"] = {o: sum(1 for r in rows if r["outcome"] == o)
                             for o in sorted({r["outcome"] for r in rows})}
        entry["total_opened"] = sum(r["chests_opened"] for r in rows)
        entry["total_available"] = sum(r["chests_available"] for r in rows)
        # VERIFY accuracy: what the agent recorded vs what really happened.
        entry["verification"] = {
            "opened_recorded_total": entry["total_opened"],
            "opened_truth_total": sum(r["chests_opened_truth"] for r in rows),
            "over_reported_total": sum(max(0, r["chests_opened"] - r["chests_opened_truth"])
                                       for r in rows),
            "under_reported_total": sum(max(0, r["chests_opened_truth"] - r["chests_opened"])
                                        for r in rows),
            "episodes_with_mismatch": sum(1 for r in rows
                                          if r["chests_opened"] != r["chests_opened_truth"]),
        }
        vrows = [r for r in vision if r["mode"] == mode]
        if vrows:
            tp = sum(r["true_positives"] for r in vrows)
            fp = sum(r["false_positives"] for r in vrows)
            fn = sum(r["false_negatives"] for r in vrows)
            entry["vision"] = {
                "precision_pooled": tp / (tp + fp) if tp + fp else None,
                "recall_pooled": tp / (tp + fn) if tp + fn else None,
                "false_positives_per_episode": _aggregate(vrows, "false_positives"),
                "class_errors_total": sum(r["class_errors"] for r in vrows),
                "localisation_success_rate": _aggregate(vrows, "localisation_success_rate"),
                "estimate_error_mean_cells": _aggregate(vrows, "estimate_error_mean_cells"),
                "estimate_error_max_cells": _aggregate(vrows, "estimate_error_max_cells"),
                "inference_ms_mean": _aggregate(vrows, "inference_ms_mean"),
                "inference_ms_peak": _aggregate(vrows, "inference_ms_peak"),
                "analyses_per_sim_second": _aggregate(vrows, "analyses_per_sim_second"),
            }
        summary["per_mode"][mode] = entry
    if "oracle" in modes:
        base = {r["episode"]: r for r in navigation if r["mode"] == "oracle"}
        for mode in modes:
            if mode == "oracle":
                continue
            diffs = [r["chests_per_minute"] - base[r["episode"]]["chests_per_minute"]
                     for r in navigation if r["mode"] == mode]
            summary["paired_vs_oracle"][mode] = {
                "mean_difference_chests_per_minute": statistics.fmean(diffs),
                "median_difference_chests_per_minute": statistics.median(diffs),
                "better": sum(1 for d in diffs if d > 1e-9),
                "worse": sum(1 for d in diffs if d < -1e-9),
                "tied": sum(1 for d in diffs if abs(d) <= 1e-9),
            }
    summary["run"] = {
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    if output_dir is not None:
        write_outputs(output_dir, navigation, vision, summary)
        summary["output_dir"] = str(output_dir)
    summary["_navigation_rows"] = navigation
    summary["_vision_rows"] = vision
    return summary


def write_outputs(output_dir: Path, navigation: list[dict[str, Any]],
                  vision: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in (("navigation_benchmark.csv", NAVIGATION_FIELDS, navigation),
                               ("vision_benchmark.csv", VISION_FIELDS, vision)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    clean = {k: v for k, v in summary.items() if not k.startswith("_")}
    (output_dir / "summary.json").write_text(json.dumps(clean, indent=2, default=str) + "\n",
                                             encoding="utf-8")


def format_summary(summary: dict[str, Any]) -> str:
    lines = []
    header = ("Mode", "Chests/min (mean)", "median", "% avail. opened", "s/chest",
              "Recoveries", "Nav success", "Precision", "Recall", "Loc. err (cells)",
              "Infer ms", "CPU s/ep")
    rows = []
    for mode, entry in summary["per_mode"].items():
        v = entry.get("vision", {})

        def f(x: Any, digits: int = 3) -> str:
            return "-" if x is None else f"{x:.{digits}f}"

        rows.append((
            mode, f(entry["chests_per_minute"]["mean"]), f(entry["chests_per_minute"]["median"]),
            f(entry["percent_available_opened"]["mean"], 1),
            f(entry["seconds_per_chest"]["mean"], 2),
            f(entry["recoveries"]["mean"], 2), f(entry["navigation_success_rate"]["mean"]),
            f(v.get("precision_pooled")), f(v.get("recall_pooled")),
            f((v.get("estimate_error_mean_cells") or {}).get("mean"), 4),
            f((v.get("inference_ms_mean") or {}).get("mean"), 2),
            f(entry["cpu_seconds"]["mean"], 2),
        ))
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    lines.append("  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                           for i, (h, w) in enumerate(zip(header, widths, strict=True))))
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                               for i, (c, w) in enumerate(zip(row, widths, strict=True))))
    for mode, entry in summary["per_mode"].items():
        ver = entry["verification"]
        lines.append(f"{mode} verification: {ver['opened_recorded_total']} openings recorded, "
                     f"{ver['opened_truth_total']} in the simulator "
                     f"({ver['over_reported_total']} over-reported, "
                     f"{ver['under_reported_total']} under-reported, "
                     f"{ver['episodes_with_mismatch']} episode(s) with a mismatch)")
    for mode, cmp in summary.get("paired_vs_oracle", {}).items():
        lines.append(f"{mode} vs oracle: mean {cmp['mean_difference_chests_per_minute']:+.3f} "
                     f"chests/min (median {cmp['median_difference_chests_per_minute']:+.3f}); "
                     f"better {cmp['better']}, worse {cmp['worse']}, tied {cmp['tied']}")
    if summary.get("detector_comparison"):
        lines.append("Detector comparison (same labelled frames):")
        for method, d in summary["detector_comparison"].items():
            lines.append(f"  {method:8s} precision {d['precision'] or 0:.3f}  recall "
                         f"{d['recall'] or 0:.3f}  false positives {d['false_positives']}  "
                         f"class errors {d['class_errors']}  {d['detector_ms_mean']:.2f} ms/frame")
    run = summary["run"]
    lines.append(f"Episodes completed: {summary['episodes_completed']} of "
                 f"{summary['episodes_requested']} requested; wall {run['wall_seconds']:.1f} s, "
                 f"CPU {run['cpu_seconds']:.1f} s, peak RSS {run['peak_rss_mb']:.0f} MiB.")
    return "\n".join(lines)
