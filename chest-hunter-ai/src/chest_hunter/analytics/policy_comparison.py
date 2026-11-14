"""Paired comparison of control strategies (Phase 5 benchmark).

Every strategy plays exactly the same (map, seed, episode) triples on the Python reference
arena. Per episode it records throughput (opened chests per *simulated* minute), the share
of available chests opened, discovery, navigation and interaction success, exploration
coverage, travel, recoveries by class, model latency and fallbacks, CPU time and memory.
Openings are checked against ground truth through the evaluation channel (never visible
to the policy).

Adaptive strategies read their history (spawn estimates, route memory) from a copy of the
experience database. In ``online`` mode the episodes of one (strategy, map) pair run in
sequence and each episode's observations are added to that copy before the next one,
which measures adaptation across repeated experiments.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chest_hunter.analytics.stats import summarise
from chest_hunter.config import Settings, load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    policy: str
    adaptive: bool
    smart_recovery: bool
    description: str


STRATEGIES: dict[str, Strategy] = {s.name: s for s in (
    Strategy("deterministic", "deterministic", False, False,
             "Phase 4 deterministic visual controller (baseline)"),
    Strategy("learned", "learned", False, False,
             "imitation policy drives; deterministic perception and bookkeeping"),
    Strategy("hybrid", "hybrid", False, False,
             "deterministic controller, learned visual approach when confident"),
    Strategy("hybrid_adaptive", "hybrid", True, True,
             "hybrid + adaptive route planning, route memory, classified recovery"),
    Strategy("deterministic_adaptive", "deterministic", True, True,
             "ablation: Phase 4 controller + adaptive planning and recovery"),
)}
DEFAULT_STRATEGIES = ("deterministic", "learned", "hybrid", "hybrid_adaptive")

FIELDS = (
    "map_id", "seed", "episode", "order", "strategy", "outcome", "end_reason", "sim_seconds",
    "chests_present", "chests_available", "chests_seen", "chests_opened", "chests_opened_truth",
    "over_reported", "throughput_per_min", "percent_available_opened", "discovery_rate",
    "navigation_success_rate", "interaction_success_rate", "exploration_coverage",
    "travel_distance_m", "travel_per_chest_m", "recoveries", "navigation_failures",
    "credited_by_notification", "learned_share", "inference_mean_ms", "inference_p95_ms",
    "fallbacks", "recovery_classes", "perception_ms_mean", "cpu_seconds", "wall_seconds",
    "peak_rss_mb",
)


@dataclass(frozen=True, slots=True)
class Job:
    strategy: str
    map_id: str
    seed: int
    episodes: tuple[int, ...]           # run in order in one worker (online adaptation)
    settings_path: str
    experience_db: str | None
    model_dir: str | None
    quantized: bool
    max_ticks: int
    width: int
    height: int
    online: bool


def _memory_copy(path: str | None, map_id: str) -> DatabaseManager:
    """In-memory copy of the experience database (or an empty one) with the map seeded."""
    from chest_hunter.scenarios import ARENA_3D_ID, seed_arena_3d

    db = DatabaseManager(":memory:")
    if path is not None and Path(path).exists():
        src = sqlite3.connect(path)
        try:
            src.backup(db.connection)
        finally:
            src.close()
    db.initialize()
    if map_id == ARENA_3D_ID:
        seed_arena_3d(db)                    # idempotent; procedural maps seed themselves
    return db


def _row(job: Job, episode: int, order: int, runner: Any, result: Any) -> dict[str, Any]:
    from chest_hunter.agent.runner_3d import peak_rss_mb
    from chest_hunter.localization.world_model_3d import PriorStatus3D

    m, t = result.metrics, result.truth
    available = set(t.get("available_names", []))
    seen = set(t.get("seen_names", [])) & available
    opened_truth = set(t.get("opened_names", []))
    minutes = result.sim_seconds / 60.0
    priors = list(runner.controller.world.priors.values())
    resolved = sum(1 for p in priors if p.status is not PriorStatus3D.UNKNOWN)
    selector = runner.selector
    sel = selector.stats.to_dict() if selector is not None else {}
    attempts = m.interaction_attempts if selector is None or selector.mode != "learned" \
        else sel.get("interaction_holds", 0)
    return {
        "map_id": job.map_id, "seed": job.seed, "episode": episode, "order": order,
        "strategy": job.strategy, "outcome": result.outcome, "end_reason": result.end_reason,
        "sim_seconds": round(result.sim_seconds, 3), "chests_present": t.get("present"),
        "chests_available": len(available), "chests_seen": len(seen),
        "chests_opened": result.chests_opened, "chests_opened_truth": t.get("opened_truth"),
        "over_reported": max(0, result.chests_opened - (t.get("opened_truth") or 0)),
        "throughput_per_min": round(result.chests_opened / minutes, 4) if minutes > 0 else 0.0,
        "percent_available_opened": round(100.0 * len(opened_truth & available) / len(available),
                                          2) if available else None,
        "discovery_rate": round(len(seen) / len(available), 4) if available else None,
        "navigation_success_rate": round(min(1.0, m.targets_reached / m.targets_selected), 4)
        if m.targets_selected else None,
        "interaction_success_rate": round(min(1.0, m.chests_opened / attempts), 4)
        if attempts else None,
        "exploration_coverage": round(resolved / len(priors), 4) if priors else None,
        "travel_distance_m": round(m.travel_distance, 2),
        "travel_per_chest_m": round(m.travel_distance / m.chests_opened, 2)
        if m.chests_opened else None,
        "recoveries": m.recoveries, "navigation_failures": m.navigation_failures,
        "credited_by_notification": m.credited_by_notification,
        "learned_share": sel.get("learned_share", 0.0),
        "inference_mean_ms": sel.get("latency_mean_ms"),
        "inference_p95_ms": sel.get("latency_p95_ms"),
        "fallbacks": json.dumps(sel.get("fallbacks", {}), sort_keys=True),
        "recovery_classes": json.dumps(m.recovery_log.to_dict(), sort_keys=True),
        "perception_ms_mean": round(m.perception_ms_mean, 3),
        "cpu_seconds": round(result.cpu_seconds, 3), "wall_seconds": round(result.wall_seconds, 3),
        "peak_rss_mb": round(peak_rss_mb(), 1),
    }


def run_job(job: Job) -> list[dict[str, Any]]:
    from chest_hunter.agent.controller_3d import Agent3DConfig
    from chest_hunter.agent.runner_3d import Runner3D, _layout_hash
    from chest_hunter.exploration.adaptive_planner import AdaptivePlanner
    from chest_hunter.learning.learned_policy import LearnedPolicy

    settings = load_config(job.settings_path)
    strategy = STRATEGIES[job.strategy]
    policy = None
    if strategy.policy != "deterministic":
        if job.model_dir is None:
            raise ValidationError(f"strategy {job.strategy} needs --model")
        policy = LearnedPolicy(job.model_dir, quantized=job.quantized)
    layout = _layout_hash(job.map_id)
    rows = []
    db = _memory_copy(job.experience_db, job.map_id)
    try:
        for order, episode in enumerate(job.episodes):
            if not job.online and order > 0:
                db.close()
                db = _memory_copy(job.experience_db, job.map_id)
            planner = AdaptivePlanner.from_database(db, job.map_id, layout) \
                if strategy.adaptive else None
            config = Agent3DConfig(decision_ticks=settings.simulation_3d.decision_ticks,
                                   smart_recovery=strategy.smart_recovery)
            runner = Runner3D.from_settings(
                db, settings, mode="hybrid", seed=job.seed, episode=episode,
                backend="reference", map_id=job.map_id, collect_truth=True, config=config,
                max_ticks=job.max_ticks, width=job.width, height=job.height,
                policy=strategy.policy, learned_policy=policy, planner=planner)
            result = runner.run()
            if planner is not None:
                planner.flush(result.session_id)
            rows.append(_row(job, episode, order, runner, result))
    finally:
        db.close()
    return rows


def plan_jobs(strategies: Sequence[str], map_ids: Sequence[str], *, seed: int, episodes: int,
              settings_path: str, experience_db: str | None, model_dir: str | None,
              quantized: bool, max_ticks: int, width: int, height: int,
              online: bool) -> list[Job]:
    jobs = []
    for map_id in map_ids:
        for name in strategies:
            if online:
                jobs.append(Job(name, map_id, seed, tuple(range(episodes)), settings_path,
                                experience_db, model_dir, quantized, max_ticks, width, height,
                                True))
            else:
                jobs += [Job(name, map_id, seed, (e,), settings_path, experience_db, model_dir,
                             quantized, max_ticks, width, height, False)
                         for e in range(episodes)]
    # Interleave strategies so partial results stay paired if a run is interrupted.
    jobs.sort(key=lambda j: (j.episodes[0], j.map_id, list(strategies).index(j.strategy)))
    return jobs


def run_policy_benchmark(settings: Settings, *, strategies: Sequence[str] = DEFAULT_STRATEGIES,
                         map_ids: Sequence[str] = ("research_arena_3d",), seed: int = 42,
                         episodes: int = 100, model_dir: str | Path | None = None,
                         experience_db: str | Path | None = None, quantized: bool = False,
                         workers: int = 1, max_ticks: int = 18000, width: int = 640,
                         height: int = 360, online: bool = False,
                         output_dir: str | Path | None = None, baseline: str = "deterministic",
                         progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ValidationError(f"unknown strategies {unknown}; choose from "
                              f"{', '.join(STRATEGIES)}")
    if episodes < 1:
        raise ValidationError("episodes must be at least 1")
    if any(STRATEGIES[s].policy != "deterministic" for s in strategies) and model_dir is None:
        raise ValidationError("learned and hybrid strategies need --model")
    jobs = plan_jobs(strategies, map_ids, seed=seed, episodes=episodes,
                     settings_path=str(settings.config_path),
                     experience_db=str(experience_db) if experience_db else None,
                     model_dir=str(model_dir) if model_dir else None, quantized=quantized,
                     max_ticks=max_ticks, width=width, height=height, online=online)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if workers <= 1:
        for k, job in enumerate(jobs):
            rows += run_job(job)
            if progress:
                progress(f"  ... job {k + 1}/{len(jobs)} ({job.strategy}, {job.map_id})")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for k, fut in enumerate(futures):
                try:
                    rows += fut.result()
                except Exception as exc:                  # report, keep the other results
                    errors.append(f"{jobs[k].strategy}/{jobs[k].map_id}/{jobs[k].episodes}: "
                                  f"{type(exc).__name__}: {exc}")
                if progress:
                    progress(f"  ... job {k + 1}/{len(jobs)} ({jobs[k].strategy}, "
                             f"{jobs[k].map_id})")
    rows.sort(key=lambda r: (r["map_id"], r["episode"], list(strategies).index(r["strategy"])))
    summary = summarise_rows(rows, strategies, baseline=baseline)
    summary.update({
        "map_ids": list(map_ids), "seed": seed, "episodes_requested": episodes,
        "strategies": {s: STRATEGIES[s].description for s in strategies},
        "online_adaptation": online, "max_ticks": max_ticks, "observation": [width, height],
        "model_dir": str(model_dir) if model_dir else None, "quantized_model": quantized,
        "experience_db": str(experience_db) if experience_db else None, "workers": workers,
        "wall_seconds": round(time.perf_counter() - started, 1), "errors": errors,
        "environment": "Python reference arena (protocol-identical mock of the Unreal "
                       "research server); NOT Unreal Engine",
        "time_basis": "throughput uses simulated minutes; cpu/wall seconds are real time",
    })
    if output_dir is not None:
        write_outputs(rows, summary, Path(output_dir))
    return summary


SUMMARY_METRICS = ("throughput_per_min", "percent_available_opened", "discovery_rate",
                   "navigation_success_rate", "interaction_success_rate", "exploration_coverage",
                   "travel_distance_m", "travel_per_chest_m", "recoveries", "sim_seconds",
                   "inference_mean_ms", "learned_share", "cpu_seconds", "peak_rss_mb")


def summarise_rows(rows: Sequence[dict[str, Any]], strategies: Sequence[str], *,
                   baseline: str = "deterministic") -> dict[str, Any]:
    per: dict[str, Any] = {}
    for s in strategies:
        rs = [r for r in rows if r["strategy"] == s]
        entry: dict[str, Any] = {"episodes": len(rs)}
        for key in SUMMARY_METRICS:
            entry[key] = summarise([r[key] for r in rs if r[key] is not None])
        entry["outcomes"] = dict(Counter(r["outcome"] for r in rs))
        entry["end_reasons"] = dict(Counter(str(r["end_reason"]) for r in rs).most_common(6))
        entry["over_reported_total"] = sum(r["over_reported"] for r in rs)
        entry["opened_total"] = sum(r["chests_opened"] for r in rs)
        entry["opened_truth_total"] = sum(r["chests_opened_truth"] or 0 for r in rs)
        entry["credited_by_notification_total"] = sum(r["credited_by_notification"] for r in rs)
        rec: Counter[str] = Counter()
        fb: Counter[str] = Counter()
        for r in rs:
            for cls, v in json.loads(r["recovery_classes"]).items():
                rec[cls] += v["attempts"]
            fb.update(json.loads(r["fallbacks"]))
        entry["recovery_attempts_by_class"] = dict(rec)
        entry["fallbacks"] = dict(fb)
        per[s] = entry
    paired: dict[str, Any] = {}
    base = {(r["map_id"], r["episode"]): r for r in rows if r["strategy"] == baseline}
    for s in strategies:
        if s == baseline:
            continue
        diffs = [r["throughput_per_min"] - base[(r["map_id"], r["episode"])]["throughput_per_min"]
                 for r in rows if r["strategy"] == s and (r["map_id"], r["episode"]) in base]
        stats = summarise(diffs)
        base_mean = per.get(baseline, {}).get("throughput_per_min", {}).get("mean")
        paired[s] = {"throughput_difference": stats,
                     "percent_change_of_mean": round(100.0 * stats["mean"] / base_mean, 2)
                     if diffs and base_mean else None,
                     "better": sum(d > 1e-9 for d in diffs), "worse": sum(d < -1e-9 for d in diffs),
                     "tied": sum(abs(d) <= 1e-9 for d in diffs),
                     "significant_at_95": bool(diffs) and stats["n"] > 1 and (
                         stats["ci_low"] > 0 or stats["ci_high"] < 0)}
    return {"per_strategy": per, "paired_vs_" + baseline: paired,
            "episodes_completed": len({(r["map_id"], r["episode"]) for r in rows})}


def write_outputs(rows: Sequence[dict[str, Any]], summary: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in FIELDS})
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out / "report.md").write_text(format_report(summary) + "\n", encoding="utf-8")


def _fmt(stats: dict[str, Any], digits: int = 3) -> str:
    if not stats or stats.get("mean") is None:
        return "-"
    return f"{stats['mean']:.{digits}f}"


def format_report(summary: dict[str, Any]) -> str:
    per = summary["per_strategy"]
    lines = ["| Strategy | Episodes | Throughput /min mean (median, sd) [95 % CI] "
             "| % avail. opened "
             "| Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries "
             "| Inference ms | Learned share |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in per.items():
        t = e["throughput_per_min"]
        ci = f"[{t['ci_low']:.3f}, {t['ci_high']:.3f}]" if t.get("mean") is not None else ""
        lines.append(
            f"| {name} | {e['episodes']} | "
            + (f"{t['mean']:.3f} ({t['median']:.3f}, {t['std']:.3f}) {ci}" if t.get("mean")
               is not None else "-")
            + f" | {_fmt(e['percent_available_opened'], 1)} | {_fmt(e['navigation_success_rate'])}"
            f" | {_fmt(e['interaction_success_rate'])} | {_fmt(e['exploration_coverage'])}"
            f" | {_fmt(e['travel_per_chest_m'], 1)} | {_fmt(e['recoveries'], 1)}"
            f" | {_fmt(e['inference_mean_ms'], 2)} | {_fmt(e['learned_share'])} |")
    key = next((k for k in summary if k.startswith("paired_vs_")), None)
    if key:
        lines += ["", f"Paired differences in throughput ({key[10:]} baseline):", ""]
        for name, p in summary[key].items():
            d = p["throughput_difference"]
            if d.get("mean") is None:
                continue
            lines.append(f"- {name}: mean {d['mean']:+.3f} /min [95 % CI {d['ci_low']:+.3f}, "
                         f"{d['ci_high']:+.3f}], {p['percent_change_of_mean']:+.1f} %, better "
                         f"{p['better']}, worse {p['worse']}, tied {p['tied']}"
                         + (" (significant)" if p["significant_at_95"] else ""))
    if summary.get("errors"):
        lines += ["", "Errors:"] + [f"- {e}" for e in summary["errors"]]
    return "\n".join(lines)


def throughput_ci_width(stats: dict[str, Any]) -> float:
    if stats.get("ci_low") is None:
        return math.nan
    return float(stats["ci_high"]) - float(stats["ci_low"])
