"""Phase 6 paired benchmark of agent configurations.

Every configuration (``runtime.configurations.PHASE6_SETUPS``) plays exactly the same
(map, seed, episode) triples on the Python reference arena, so differences are paired by
episode. Per episode it records, besides the Phase 5 metrics (``policy_comparison``):

* outcome: ``completed`` (every spawn point resolved), ``tick_limit`` (time ran out) or
  ``failed`` (aborted by the supervisor, recovery budget, communication) - the episode
  completion rate is the share of ``completed``;
* recovery success: recoveries whose target was afterwards reached or opened, over all
  recorded recovery attempts;
* simulated seconds between agent failures (Phase 4 recoveries plus supervisor
  interventions);
* navigation failure rate: failed plans, stuck routes and blocked approaches over routes
  started plus failed plans;
* supervisor detections, interventions and aborts; frames analysed and frames skipped;
* agent-side CPU: CPU time of the agent's own thread in the profiled perception, planning,
  navigation, inference, supervision and persistence sections (the reference simulator runs
  in another thread of the same process and is excluded); process CPU (agent + simulator);
  current and peak resident memory.

The primary metric is chests actually opened per *simulated* minute (ground truth through
the evaluation channel; never visible to the agent).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chest_hunter.analytics.policy_comparison import FIELDS as PHASE5_FIELDS
from chest_hunter.analytics.policy_comparison import Job, _memory_copy, _row
from chest_hunter.analytics.stats import summarise
from chest_hunter.config import Settings, load_config
from chest_hunter.exceptions import ValidationError
from chest_hunter.runtime.configurations import (
    DEFAULT_PHASE6_SETUPS,
    PHASE6_SETUPS,
    agent_config,
    make_planner,
    partition,
    policy_options,
    supervisor_config,
)

PRIMARY_METRIC = "throughput_truth_per_min"
AGENT_SECTIONS = ("preprocessing", "chest_detection", "tracking_localisation",
                  "interaction_cues", "route_selection", "navigation_query",
                  "learned_inference", "action_generation", "supervision",
                  "database_persistence")

PHASE6_FIELDS = (
    "completed", "failed", "recovery_attempts", "recovery_resolved", "recovery_success_rate",
    "seconds_between_failures", "navigation_failure_rate", "routes_started",
    "supervisor_detections", "supervisor_interventions", "supervisor_aborted",
    "frames_analysed", "frames_skipped", "planning_latency_mean_ms", "navigation_queries",
    "agent_cpu_seconds", "agent_cpu_per_sim_second", "rss_end_mb",
)
FIELDS = (*PHASE5_FIELDS, *PHASE6_FIELDS)

SUMMARY_METRICS = (
    "throughput_truth_per_min", "throughput_per_min", "percent_available_opened",
    "travel_distance_m", "interaction_success_rate", "navigation_failure_rate",
    "recovery_success_rate", "seconds_between_failures", "inference_mean_ms",
    "agent_cpu_seconds", "agent_cpu_per_sim_second", "cpu_seconds", "peak_rss_mb",
    "rss_end_mb", "sim_seconds", "frames_analysed", "planning_latency_mean_ms",
    "supervisor_interventions",
)


@dataclass(frozen=True, slots=True)
class Phase6Job:
    setup: str
    map_id: str
    seed: int
    episode: int
    settings_path: str
    experience_db: str | None
    model_dir: str | None
    max_ticks: int
    width: int
    height: int


def _rss_now_mb() -> float:
    from chest_hunter.analytics.resource_monitor import process_rss_mb

    return process_rss_mb()[0]


def run_phase6_job(job: Phase6Job) -> dict[str, Any]:
    from chest_hunter.agent.runner_3d import Runner3D, _layout_hash
    from chest_hunter.learning.learned_policy import LearnedPolicy
    from chest_hunter.runtime.profiler import Profiler

    settings = load_config(job.settings_path)
    setup = PHASE6_SETUPS[job.setup]
    policy = None
    if setup.policy != "deterministic":
        if job.model_dir is None:
            raise ValidationError(f"configuration {job.setup} needs --model")
        policy = _policy_cache(job.model_dir, LearnedPolicy)
    db = _memory_copy(job.experience_db, job.map_id)
    try:
        layout = _layout_hash(job.map_id)
        c3 = settings.simulation_3d
        env = {"odometry_drift": c3.odometry_drift, "noise_std": c3.noise_std,
               "reward_probability": c3.reward_probability,
               "interaction_failure_probability": c3.interaction_failure_probability}
        part = partition(map_id=job.map_id, env_overrides=env, setup=setup,
                         model_version=getattr(policy, "version", None), width=job.width,
                         height=job.height, fov_deg=c3.fov_deg,
                         decision_ticks=c3.decision_ticks)
        planner = make_planner(settings, setup, db, job.map_id, layout,
                               partition_key=part["key"])
        profiler = Profiler()
        runner = Runner3D.from_settings(
            db, settings, mode="hybrid", seed=job.seed, episode=job.episode,
            backend="reference", map_id=job.map_id, collect_truth=True,
            config=agent_config(settings, setup), max_ticks=job.max_ticks, width=job.width,
            height=job.height, policy=setup.policy, learned_policy=policy, planner=planner,
            policy_options=policy_options(settings, setup) or None,
            supervisor=supervisor_config(settings, setup), profiler=profiler, partition=part)
        result = runner.run()
        if planner is not None:
            planner.flush(result.session_id)
        base_job = Job(job.setup, job.map_id, job.seed, (job.episode,), job.settings_path,
                       job.experience_db, job.model_dir, False, job.max_ticks, job.width,
                       job.height, False)
        row = _row(base_job, job.episode, 0, runner, result)
        row.update(_phase6_fields(runner, result, profiler, planner))
        return row
    finally:
        db.close()


_POLICIES: dict[str, Any] = {}


def _policy_cache(model_dir: str, cls: Any) -> Any:
    """One model per worker process (the ONNX session is reused across episodes)."""
    if model_dir not in _POLICIES:
        _POLICIES[model_dir] = cls(model_dir)
    return _POLICIES[model_dir]


def _phase6_fields(runner: Any, result: Any, profiler: Any, planner: Any) -> dict[str, Any]:
    m = result.metrics
    log = m.recovery_log
    attempts = sum(log.attempts.values())
    resolved = sum(log.resolved.values())
    sup = result.supervisor or {}
    interventions = int(sup.get("interventions_total", 0))
    failures = m.recoveries + interventions
    report = {r["component"]: r for r in profiler.report()}
    agent_cpu = sum(report[c]["cpu_total_s"] for c in AGENT_SECTIONS
                    if c in report and c != "action_generation")
    # action_generation contains route selection and navigation queries: count only the rest
    if "action_generation" in report:
        nested = sum(report[c]["cpu_total_s"] for c in ("route_selection",) if c in report)
        agent_cpu += max(0.0, report["action_generation"]["cpu_total_s"] - nested)
    plan = planner.stats.to_dict() if planner is not None and hasattr(planner, "stats") \
        else {}
    agent = getattr(runner.controller, "agent", runner.controller)
    nav = getattr(agent, "navigator", None)
    return {
        "completed": int(result.outcome == "completed"),
        "failed": int(result.outcome == "failed"),
        "recovery_attempts": attempts, "recovery_resolved": resolved,
        "recovery_success_rate": round(resolved / attempts, 4) if attempts else None,
        "seconds_between_failures": round(result.sim_seconds / failures, 2) if failures else None,
        "navigation_failure_rate": round(m.navigation_failures / (m.routes_started
                                                                  + m.navigation_failures), 4)
        if m.routes_started + m.navigation_failures else None,
        "routes_started": m.routes_started,
        "supervisor_detections": json.dumps(sup.get("detections", {}), sort_keys=True),
        "supervisor_interventions": interventions,
        "supervisor_aborted": sup.get("aborted"),
        "frames_analysed": m.analyses, "frames_skipped": m.frames_skipped_unchanged,
        "planning_latency_mean_ms": plan.get("latency_mean_ms"),
        "navigation_queries": getattr(nav, "queries", None),
        "agent_cpu_seconds": round(agent_cpu, 3),
        "agent_cpu_per_sim_second": round(agent_cpu / result.sim_seconds, 5)
        if result.sim_seconds else None,
        "rss_end_mb": round(_rss_now_mb(), 1),
    }


def plan_phase6_jobs(setups: Sequence[str], map_ids: Sequence[str], *, seed: int,
                     episodes: int, settings_path: str, experience_db: str | None,
                     model_dir: str | None, max_ticks: int, width: int,
                     height: int) -> list[Phase6Job]:
    jobs = [Phase6Job(s, m, seed, e, settings_path, experience_db, model_dir, max_ticks, width,
                      height) for m in map_ids for s in setups for e in range(episodes)]
    # Interleave configurations so partial results stay paired if a run is interrupted.
    jobs.sort(key=lambda j: (j.episode, j.map_id, list(setups).index(j.setup)))
    return jobs


def run_phase6_benchmark(settings: Settings, *, setups: Sequence[str] = DEFAULT_PHASE6_SETUPS,
                         map_ids: Sequence[str] = ("research_arena_3d",), seed: int = 42,
                         episodes: int = 100, model_dir: str | Path | None = None,
                         experience_db: str | Path | None = None, workers: int = 1,
                         max_ticks: int = 18000, width: int = 640, height: int = 360,
                         output_dir: str | Path | None = None,
                         baseline: str = "phase4_deterministic", resume: bool = False,
                         progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    unknown = [s for s in setups if s not in PHASE6_SETUPS]
    if unknown:
        raise ValidationError(f"unknown configurations {unknown}; choose from "
                              f"{', '.join(PHASE6_SETUPS)}")
    if episodes < 1:
        raise ValidationError("episodes must be at least 1")
    if baseline not in setups:
        raise ValidationError(f"the baseline {baseline!r} must be one of the configurations")
    if any(PHASE6_SETUPS[s].policy != "deterministic" for s in setups) and model_dir is None:
        raise ValidationError("learned and hybrid configurations need --model")
    jobs = plan_phase6_jobs(setups, map_ids, seed=seed, episodes=episodes,
                            settings_path=str(settings.config_path),
                            experience_db=str(experience_db) if experience_db else None,
                            model_dir=str(model_dir) if model_dir else None,
                            max_ticks=max_ticks, width=width, height=height)
    started = time.perf_counter()
    errors: list[str] = []
    journal = None
    if output_dir is not None:
        from chest_hunter.analytics.journal import RowJournal

        journal = RowJournal(Path(output_dir) / "rows.partial.jsonl", {
            "setups": list(setups), "maps": list(map_ids), "seed": seed, "episodes": episodes,
            "max_ticks": max_ticks, "observation": [width, height],
            "model_dir": str(model_dir) if model_dir else None}, resume=resume)

    def key(job: Phase6Job) -> str:
        return f"{job.setup}|{job.map_id}|{job.episode}"

    pending = [j for j in jobs if journal is None or not journal.done(key(j))]
    if journal is not None and progress and len(pending) < len(jobs):
        progress(f"  resuming: {len(jobs) - len(pending)} job(s) already done")
    done = 0

    def record(job: Phase6Job, row: dict[str, Any] | None, exc: BaseException | None) -> None:
        nonlocal done
        done += 1
        if row is not None and journal is not None:
            journal.add(key(job), row)
        elif row is not None:
            fresh.append(row)
        if exc is not None:                                   # report, keep the others
            errors.append(f"{job.setup}/{job.map_id}/{job.episode}: "
                          f"{type(exc).__name__}: {exc}")
        if progress:
            progress(f"  ... job {done}/{len(pending)} ({job.setup}, episode {job.episode})")

    fresh: list[dict[str, Any]] = []
    if workers <= 1:
        for job in pending:
            try:
                record(job, run_phase6_job(job), None)
            except Exception as exc:
                record(job, None, exc)
    else:
        from concurrent.futures import as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_phase6_job, job): job for job in pending}
            for fut in as_completed(futures):
                try:
                    record(futures[fut], fut.result(), None)
                except Exception as exc:
                    record(futures[fut], None, exc)
    rows = list(journal.rows.values()) if journal is not None else fresh
    rows.sort(key=lambda r: (r["map_id"], r["episode"], list(setups).index(r["strategy"])))
    summary = summarise_phase6(rows, setups, baseline=baseline)
    summary.update({
        "map_ids": list(map_ids), "seed": seed, "episodes_requested": episodes,
        "configurations": {s: PHASE6_SETUPS[s].description for s in setups},
        "max_ticks": max_ticks, "observation": [width, height], "workers": workers,
        "model_dir": str(model_dir) if model_dir else None,
        "experience_db": str(experience_db) if experience_db else None,
        "wall_seconds": round(time.perf_counter() - started, 1), "errors": errors,
        "environment": "Python reference arena (protocol-identical mock of the Unreal "
                       "research server); NOT Unreal Engine",
        "time_basis": "throughput uses simulated minutes; CPU and wall seconds are real time",
    })
    if output_dir is not None:
        write_phase6_outputs(rows, summary, Path(output_dir))
    return summary


def summarise_phase6(rows: Sequence[dict[str, Any]], setups: Sequence[str], *,
                     baseline: str) -> dict[str, Any]:
    per: dict[str, Any] = {}
    for s in setups:
        rs = [r for r in rows if r["strategy"] == s]
        entry: dict[str, Any] = {"episodes": len(rs)}
        for key in SUMMARY_METRICS:
            entry[key] = summarise([float(r[key]) for r in rs if r.get(key) is not None])
        entry["outcomes"] = dict(Counter(r["outcome"] for r in rs))
        entry["completion_rate"] = round(sum(r["completed"] for r in rs) / len(rs), 4) \
            if rs else None
        entry["failed_episodes"] = sum(r["failed"] for r in rs)
        attempts = sum(r["recovery_attempts"] for r in rs)
        entry["recovery_success_rate_pooled"] = round(
            sum(r["recovery_resolved"] for r in rs) / attempts, 4) if attempts else None
        fail = sum(r["recoveries"] + r["supervisor_interventions"] for r in rs)
        entry["seconds_between_failures_pooled"] = round(
            sum(r["sim_seconds"] for r in rs) / fail, 2) if fail else None
        det: Counter[str] = Counter()
        for r in rs:
            det.update(json.loads(r["supervisor_detections"] or "{}"))
        entry["supervisor_detections_total"] = dict(det)
        entry["supervisor_aborts"] = sum(1 for r in rs if r.get("supervisor_aborted"))
        entry["over_reported_total"] = sum(r["over_reported"] for r in rs)
        entry["under_reported_total"] = sum(r["under_reported"] for r in rs)
        entry["frames_skipped_total"] = sum(r["frames_skipped"] for r in rs)
        per[s] = entry
    paired: dict[str, Any] = {}
    base = {(r["map_id"], r["episode"]): r for r in rows if r["strategy"] == baseline}
    for s in setups:
        if s == baseline:
            continue
        paired[s] = {}
        for metric in (PRIMARY_METRIC, "percent_available_opened", "agent_cpu_seconds",
                       "cpu_seconds", "sim_seconds", "travel_distance_m"):
            diffs = [float(r[metric]) - float(base[(r["map_id"], r["episode"])][metric])
                     for r in rows if r["strategy"] == s
                     and (r["map_id"], r["episode"]) in base and r.get(metric) is not None
                     and base[(r["map_id"], r["episode"])].get(metric) is not None]
            st = summarise(diffs)
            base_mean = per.get(baseline, {}).get(metric, {}).get("mean")
            paired[s][metric] = {
                "difference": st,
                "percent_change_of_mean": round(100.0 * st["mean"] / base_mean, 2)
                if diffs and base_mean else None,
                "better": sum(d > 1e-9 for d in diffs), "worse": sum(d < -1e-9 for d in diffs),
                "tied": sum(abs(d) <= 1e-9 for d in diffs),
                "significant_at_95": bool(diffs) and st["n"] > 1 and (
                    st["ci_low"] > 0 or st["ci_high"] < 0)}
    return {"per_configuration": per, "paired_vs_" + baseline: paired,
            "primary_metric": PRIMARY_METRIC,
            "episodes_completed": len({(r["map_id"], r["episode"]) for r in rows})}


def write_phase6_outputs(rows: Sequence[dict[str, Any]], summary: dict[str, Any],
                         out: Path) -> None:
    import csv

    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in FIELDS})
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out / "report.md").write_text(format_phase6_report(summary) + "\n", encoding="utf-8")


def _m(stats: dict[str, Any], digits: int = 3) -> str:
    if not stats or stats.get("mean") is None:
        return "-"
    return f"{stats['mean']:.{digits}f}"


def format_phase6_report(summary: dict[str, Any]) -> str:
    per = summary["per_configuration"]
    lines = ["| Configuration | Episodes | Opened /min mean (median, sd) [95 % CI] | Completed "
             "| Failed | % reachable opened | Interaction success | Navigation failure rate "
             "| Recovery success | Sim s between failures | Agent CPU s | Process CPU s "
             "| Peak RSS MiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in per.items():
        t = e[PRIMARY_METRIC]
        ci = f"[{t['ci_low']:.3f}, {t['ci_high']:.3f}]" if t.get("mean") is not None else ""
        lines.append(
            f"| {name} | {e['episodes']} | "
            + (f"{t['mean']:.3f} ({t['median']:.3f}, {t['std']:.3f}) {ci}"
               if t.get("mean") is not None else "-")
            + f" | {e['completion_rate']} | {e['failed_episodes']}"
            f" | {_m(e['percent_available_opened'], 1)} | {_m(e['interaction_success_rate'])}"
            f" | {_m(e['navigation_failure_rate'])} | {e['recovery_success_rate_pooled']}"
            f" | {e['seconds_between_failures_pooled']} | {_m(e['agent_cpu_seconds'], 1)}"
            f" | {_m(e['cpu_seconds'], 1)} | {_m(e['peak_rss_mb'], 0)} |")
    key = next((k for k in summary if k.startswith("paired_vs_")), None)
    if key:
        lines += ["", f"Paired differences ({key[10:]} baseline), mean [95 % CI]:", ""]
        for name, cmp in summary[key].items():
            parts = []
            for metric, label in ((PRIMARY_METRIC, "opened/min"),
                                  ("agent_cpu_seconds", "agent CPU s"),
                                  ("cpu_seconds", "process CPU s")):
                p = cmp.get(metric, {})
                d = p.get("difference", {})
                if d.get("mean") is None:
                    continue
                parts.append(f"{label} {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
                             f" ({p['percent_change_of_mean']:+.1f} %"
                             + (", significant" if p["significant_at_95"] else "") + ")")
            lines.append(f"- {name}: " + "; ".join(parts))
    if summary.get("errors"):
        lines += ["", "Errors:"] + [f"- {e}" for e in summary["errors"]]
    return "\n".join(lines)
