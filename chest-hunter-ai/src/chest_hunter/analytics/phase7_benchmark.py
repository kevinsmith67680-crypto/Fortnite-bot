"""Phase 7 benchmarks: generalisation to held-out maps, exploration strategies, training of
the exploration memory.

All episodes run on the Python reference arena (the protocol-identical mock of the Unreal
research server) with the procedural maps of :mod:`chest_hunter.simulation3d.generator7`.

**Evaluation categories** (Phase 7 brief, section 12). Episode ``e`` of every category uses
map family ``CATEGORIES[e % 4]`` and layout index ``e // 4``:

=================  ==================================  =======================================
category           map                                 compared with ``familiar``
=================  ==================================  =======================================
``familiar``       ``p7_<cat>_train_NNN`` (new seeds)  -
``obstacles``      ``..._train_NNN_obstacles``         paired (same layout, same seed)
``unseen_layout``  ``p7_<cat>_test_NNN``               unpaired (other layouts)
``lighting``       ``..._train_NNN_light``             paired
``appearance``     ``..._train_NNN_appearance``        paired
``start``          ``..._train_NNN_start``             paired
``elevation``      ``..._train_NNN_elevation``         paired
``combined``       ``p7_<cat>_test_NNN_combined``      unpaired
=================  ==================================  =======================================

Familiar layouts are the training layouts (the exploration memory is trained on them) played
with *evaluation* seeds: :func:`evaluation_seed` maps a benchmark seed into a range that
training refuses (:func:`assert_training_seed`), so no chest-spawn outcome of an evaluation
episode is ever seen in training. Variant maps and test maps are held out
(``generator7.is_held_out``) and refused for every training write.

**Generalisation gap** ``G = (E_familiar - E_unseen) / E_familiar`` with ``E`` the mean chest
throughput (ground-truth openings per simulated minute); undefined (``None``, with the reason)
when ``E_familiar`` is zero. Its 95 % interval is a percentile bootstrap over episodes (pairs
resampled together for paired categories).

Ground truth (evaluation labels, the final oracle state) is requested by the runner for
scoring only; the agent never receives it.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from chest_hunter.analytics.stats import summarise
from chest_hunter.config import Settings, load_config
from chest_hunter.exceptions import ValidationError
from chest_hunter.exploration.active import STRATEGIES
from chest_hunter.simulation3d.generator7 import CATEGORIES, GENERATOR_VERSION, map_id

PRIMARY_METRIC: Final = "throughput_truth_per_min"

#: evaluation category -> (split, variant); see the module docstring
EVAL_CATEGORIES: Final[dict[str, tuple[str, str | None]]] = {
    "familiar": ("train", None),
    "obstacles": ("train", "obstacles"),
    "unseen_layout": ("test", None),
    "lighting": ("train", "light"),
    "appearance": ("train", "appearance"),
    "start": ("train", "start"),
    "elevation": ("train", "elevation"),
    "combined": ("test", "combined"),
}
PAIRED_CATEGORIES: Final = frozenset(k for k, (split, _) in EVAL_CATEGORIES.items()
                                     if split == "train" and k != "familiar")

#: benchmark seeds are moved into [EVAL_SEED_OFFSET, ...); training seeds stay below it
EVAL_SEED_OFFSET: Final = 900_000


def evaluation_seed(seed: int) -> int:
    if not 0 <= seed < EVAL_SEED_OFFSET:
        raise ValidationError(f"seed must be in 0..{EVAL_SEED_OFFSET - 1}")
    return EVAL_SEED_OFFSET + seed


def assert_training_seed(seed: int) -> None:
    """Training never plays an evaluation seed (the spawn outcomes it decides stay unseen)."""
    if seed >= EVAL_SEED_OFFSET:
        raise ValidationError(f"seed {seed} is an evaluation seed (>= {EVAL_SEED_OFFSET}); "
                              "training must use seeds below it")


def evaluation_map(category: str, episode: int) -> tuple[str, str]:
    """(map family, map id) of an evaluation episode."""
    if category not in EVAL_CATEGORIES:
        raise ValidationError(f"evaluation category must be one of {', '.join(EVAL_CATEGORIES)}")
    split, variant = EVAL_CATEGORIES[category]
    family = CATEGORIES[episode % len(CATEGORIES)]
    return family, map_id(family, split, episode // len(CATEGORIES), variant)


def training_map(episode: int) -> tuple[str, str]:
    family = CATEGORIES[episode % len(CATEGORIES)]
    return family, map_id(family, "train", episode // len(CATEGORIES))


# -- configurations ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Phase7Setup:
    name: str
    base: str                 # runtime.configurations.PHASE6_SETUPS entry (policy, supervisor)
    exploration: bool         # False: the Phase 6 agent with the map's spawn points
    strategy: str | None      # an exploration strategy, or "memory" (the trained bandit's best)
    description: str


PHASE7_SETUPS: Final[dict[str, Phase7Setup]] = {s.name: s for s in (
    Phase7Setup("phase6_priors", "phase6_hybrid_supervised", False, None,
                "Phase 6 hybrid + supervisor; given the map's spawn points (not their "
                "outcomes)"),
    Phase7Setup("phase7_active", "phase6_hybrid_supervised", True, "memory",
                "Phase 7 active exploration, no spawn points; strategy from the trained "
                "exploration memory (max_information when it has none)"),
    Phase7Setup("phase7_active_deterministic", "phase6_deterministic_supervised", True,
                "memory", "ablation: active exploration with the Phase 4 controller only"),
    *(Phase7Setup(f"phase7_{s}", "phase6_hybrid_supervised", True, s,
                  f"active exploration with the fixed strategy {s}") for s in STRATEGIES),
)}
DEFAULT_GENERALIZATION_SETUPS: Final = ("phase7_active", "phase6_priors")
STRATEGY_SETUPS: Final = (*(f"phase7_{s}" for s in STRATEGIES), "phase7_active")
DEFAULT_STRATEGY: Final = "max_information"


def memory_strategy(memory: Any, family: str) -> tuple[str, str]:
    """(strategy, why) chosen from the exploration memory for one map family."""
    if memory is None:
        return DEFAULT_STRATEGY, "no exploration memory"
    if not memory.bandit.stats.get(family):
        return DEFAULT_STRATEGY, f"no training episodes for {family} maps"
    return memory.bandit.best(family), "best posterior mean (exploit only)"


# -- one episode -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class P7Job:
    setup: str
    category: str             # evaluation category, or "training"
    family: str
    map_id: str
    seed: int
    episode: int
    settings_path: str
    model_dir: str | None
    memory_path: str | None
    max_ticks: int
    width: int
    height: int
    strategy_override: str | None = None    # training: the arm the bandit sampled
    fault_kind: str | None = None
    fault_seed: int = 0
    evaluate: bool = True


_POLICIES: dict[str, Any] = {}
_MEMORIES: dict[str, Any] = {}


def _policy(model_dir: str) -> Any:
    from chest_hunter.learning.learned_policy import LearnedPolicy

    if model_dir not in _POLICIES:
        _POLICIES[model_dir] = LearnedPolicy(model_dir)
    return _POLICIES[model_dir]


def _memory(path: str | None, *, reload: bool = False) -> Any:
    from chest_hunter.exploration.self_improving import ExplorationMemory
    from chest_hunter.runtime.configurations import AGENT_VERSION

    if path is None:
        return None
    if reload or path not in _MEMORIES:
        _MEMORIES[path] = ExplorationMemory.load(path, agent_version=AGENT_VERSION,
                                                 generator_version=GENERATOR_VERSION)
    return _MEMORIES[path]


def run_p7_job(job: P7Job) -> dict[str, Any]:
    """Play one episode and return its row (agent measurements + evaluation-only truth)."""
    from chest_hunter.agent.runner_3d import Runner3D, peak_rss_mb
    from chest_hunter.cli_phase7 import thresholds_from_settings
    from chest_hunter.database import DatabaseManager
    from chest_hunter.exploration.active import ActiveExplorer, ExplorationConfig
    from chest_hunter.runtime.configurations import (
        PHASE6_SETUPS,
        agent_config,
        policy_options,
        supervisor_config,
    )
    from chest_hunter.runtime.faults import FaultInjector, FaultPlan
    from chest_hunter.runtime.profiler import Profiler

    settings = load_config(job.settings_path)
    setup = PHASE7_SETUPS[job.setup]
    base = PHASE6_SETUPS[setup.base]
    policy = None
    if base.policy != "deterministic":
        if job.model_dir is None:
            raise ValidationError(f"configuration {job.setup} needs --model")
        policy = _policy(job.model_dir)
    memory = _memory(job.memory_path, reload=job.category == "training")
    strategy, why = None, None
    exploration: Any = None
    if setup.exploration:
        if job.strategy_override is not None:
            strategy, why = job.strategy_override, "sampled by the bandit (training)"
        elif setup.strategy == "memory":
            strategy, why = memory_strategy(memory, job.family)
        else:
            strategy, why = setup.strategy, "fixed"
        rates = memory.discovery_rates(job.family) if memory is not None and \
            memory.categories.get(job.family) else None
        cfg = ExplorationConfig(strategy=strategy or DEFAULT_STRATEGY)

        def exploration() -> ActiveExplorer:
            return ActiveExplorer(cfg, discovery_rates=rates)

    injector = None
    if job.fault_kind is not None:
        injector = FaultInjector(FaultPlan.random(job.fault_kind, job.fault_seed),
                                 seed=job.fault_seed)
    db = DatabaseManager(":memory:")
    db.initialize()
    profiler = Profiler()
    try:
        runner = Runner3D.from_settings(
            db, settings, mode="hybrid", seed=job.seed, episode=job.episode,
            backend="reference", map_id=job.map_id, collect_truth=True, evaluate=job.evaluate,
            config=agent_config(settings, base), max_ticks=job.max_ticks, width=job.width,
            height=job.height, policy=base.policy, learned_policy=policy,
            policy_options=policy_options(settings, base) or None,
            supervisor=supervisor_config(settings, base), profiler=profiler,
            exploration=exploration, faults=injector,
            thresholds=thresholds_from_settings(settings) if setup.exploration else None)
        result = runner.run()
        row = episode_row(job, runner, result, profiler)
        row.update({"strategy_used": strategy, "strategy_reason": why,
                    "peak_rss_mb": round(peak_rss_mb(), 1)})
        if injector is not None:
            a = injector.assess(result)
            row.update({"fault_injected": a.get("injected"), "fault_applied": a.get("applied"),
                        "fault_detected": a.get("detected"), "fault_recovered": a.get("recovered"),
                        "fault_terminated_cleanly": a.get("terminated_cleanly"),
                        "fault_start_s": a.get("start_s"),
                        "fault_details": json.dumps({k: v for k, v in a.items()
                                                     if k in ("details", "signals", "reason")},
                                                    sort_keys=True, default=float)})
        if job.category == "training":
            from chest_hunter.exploration.self_improving import outcome_from_episode

            row["_outcome"] = asdict(outcome_from_episode(job.family, strategy or "",
                                                          runner, result))
        return row
    finally:
        db.close()


AGENT_SECTIONS: Final = ("preprocessing", "chest_detection", "tracking_localisation",
                         "interaction_cues", "route_selection", "navigation_query",
                         "learned_inference", "supervision", "exploration_update",
                         "frontier_selection", "database_persistence")


def episode_row(job: P7Job, runner: Any, result: Any, profiler: Any) -> dict[str, Any]:
    m, t = result.metrics, result.truth or {}
    minutes = result.sim_seconds / 60.0
    available = set(t.get("available_names", []))
    opened = set(t.get("opened_names", []))
    seen = set(t.get("seen_names", [])) & available
    v = result.vision or {}
    log = m.recovery_log
    attempts, resolved = sum(log.attempts.values()), sum(log.resolved.values())
    report = {r["component"]: r for r in profiler.report()}
    agent_cpu = sum(report[c]["cpu_total_s"] for c in AGENT_SECTIONS if c in report)
    sel = runner.selector.stats.to_dict() if runner.selector is not None else {}
    sup = result.supervisor or {}
    explorer = runner.explorer
    routes = m.routes_started + m.navigation_failures
    return {
        "setup": job.setup, "category": job.category, "family": job.family,
        "map_id": job.map_id, "seed": job.seed, "episode": job.episode,
        "outcome": result.outcome, "end_reason": result.end_reason,
        "sim_seconds": round(result.sim_seconds, 2),
        "chests_present": t.get("present"), "chests_reachable": len(available),
        "chests_opened_truth": t.get("opened_truth"), "chests_opened_reported": m.chests_opened,
        "over_reported": max(0, m.chests_opened - (t.get("opened_truth") or 0)),
        PRIMARY_METRIC: round((t.get("opened_truth") or 0) / minutes, 4) if minutes else 0.0,
        "percent_reachable_opened": round(100.0 * len(opened & available) / len(available), 2)
        if available else None,
        "percent_reachable_found": round(100.0 * len(seen) / len(available), 2)
        if available else None,
        "interaction_attempts": m.interaction_attempts,
        "interaction_success_rate": round(min(1.0, m.chests_opened / m.interaction_attempts), 4)
        if m.interaction_attempts else None,
        "exploration_coverage": t.get("true_coverage"),
        "detection_tp": v.get("true_positives"), "detection_fp": v.get("false_positives"),
        "detection_fn": v.get("false_negatives"),
        "detection_precision": v.get("precision"), "detection_recall": v.get("recall"),
        "chest_distance_error_m": v.get("distance_mean_abs_error_m"),
        "pose_error_mean_m": v.get("pose_error_mean_m"),
        "navigation_success_rate": round(m.routes_started / routes, 4) if routes else None,
        "navigation_failures": m.navigation_failures, "routes_started": m.routes_started,
        "recovery_attempts": attempts, "recovery_resolved": resolved,
        "recovery_success_rate": round(resolved / attempts, 4) if attempts else None,
        "recoveries": m.recoveries, "abandoned_targets": m.abandoned_targets,
        "supervisor_interventions": int(sup.get("interventions_total", 0)),
        "supervisor_aborted": bool(sup.get("aborted")),
        "travel_distance_m": round(m.travel_distance, 1),
        "frontiers_selected": explorer.stats.selections if explorer else None,
        "frontier_selection_ms_mean": round(statistics.fmean(explorer.stats.latencies_ms), 3)
        if explorer and explorer.stats.latencies_ms else None,
        "camera_scans": explorer.stats.scans_planned if explorer else None,
        "inference_mean_ms": sel.get("latency_mean_ms"),
        "learned_share": sel.get("learned_share"),
        "frames_analysed": m.analyses,
        "agent_cpu_seconds": round(agent_cpu, 3),
        "agent_cpu_per_sim_second": round(agent_cpu / result.sim_seconds, 5)
        if result.sim_seconds else None,
        "process_cpu_seconds": round(result.cpu_seconds, 2),
        "wall_seconds": round(result.wall_seconds, 2),
    }


FIELDS: Final = (
    "setup", "category", "family", "map_id", "seed", "episode", "outcome", "end_reason",
    "sim_seconds", "chests_present", "chests_reachable", "chests_opened_truth",
    "chests_opened_reported", "over_reported", PRIMARY_METRIC, "percent_reachable_opened",
    "percent_reachable_found", "interaction_attempts", "interaction_success_rate",
    "exploration_coverage", "detection_tp", "detection_fp", "detection_fn",
    "detection_precision", "detection_recall", "chest_distance_error_m", "pose_error_mean_m",
    "navigation_success_rate", "navigation_failures", "routes_started", "recovery_attempts",
    "recovery_resolved", "recovery_success_rate", "recoveries", "abandoned_targets",
    "supervisor_interventions", "supervisor_aborted", "travel_distance_m",
    "frontiers_selected", "frontier_selection_ms_mean", "camera_scans", "inference_mean_ms",
    "learned_share", "frames_analysed", "agent_cpu_seconds", "agent_cpu_per_sim_second",
    "process_cpu_seconds", "wall_seconds", "peak_rss_mb", "strategy_used", "strategy_reason",
    "fault_kind", "fault_injected", "fault_applied", "fault_detected", "fault_recovered",
    "fault_terminated_cleanly", "fault_start_s", "fault_details",
)


def run_jobs(jobs: Sequence[P7Job], *, workers: int = 1,
             progress: Callable[[str], None] | None = None
             ) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    def label(j: P7Job) -> str:
        return f"{j.setup}/{j.category}/{j.map_id}/{j.episode}" + (
            f"/{j.fault_kind}" if j.fault_kind else "")

    if workers <= 1:
        for k, job in enumerate(jobs):
            try:
                rows.append(run_p7_job(job) | {"fault_kind": job.fault_kind})
            except Exception as exc:                       # report, keep the other results
                errors.append(f"{label(job)}: {type(exc).__name__}: {exc}")
            if progress:
                progress(f"  ... job {k + 1}/{len(jobs)} ({label(job)})")
        return rows, errors
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_p7_job, job) for job in jobs]
        for k, fut in enumerate(futures):
            try:
                rows.append(fut.result() | {"fault_kind": jobs[k].fault_kind})
            except Exception as exc:
                errors.append(f"{label(jobs[k])}: {type(exc).__name__}: {exc}")
            if progress:
                progress(f"  ... job {k + 1}/{len(jobs)} ({label(jobs[k])})")
    return rows, errors


# -- statistics ------------------------------------------------------------------------------


def generalization_gap(familiar: float | None, unseen: float | None) -> float | None:
    """``G = (E_familiar - E_unseen) / E_familiar``; ``None`` when undefined (no familiar
    throughput, or a missing mean)."""
    if familiar is None or unseen is None or familiar == 0:
        return None
    return (familiar - unseen) / familiar


def bootstrap_gap(familiar: Sequence[float], unseen: Sequence[float], *, paired: bool,
                  resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    fam, uns = [float(x) for x in familiar], [float(x) for x in unseen]
    point = generalization_gap(statistics.fmean(fam) if fam else None,
                               statistics.fmean(uns) if uns else None)
    out: dict[str, Any] = {"G": round(point, 4) if point is not None else None,
                           "ci_low": None, "ci_high": None, "n_familiar": len(fam),
                           "n_unseen": len(uns), "paired": paired}
    if point is None:
        out["reason"] = "undefined: familiar throughput is zero" if fam and \
            statistics.fmean(fam) == 0 else "undefined: no episodes"
        return out
    if paired and len(fam) != len(uns):
        raise ValueError("paired samples must have equal length")
    rng = random.Random(seed)
    draws: list[float] = []
    n_f, n_u = len(fam), len(uns)
    for _ in range(resamples):
        if paired:
            idx = [rng.randrange(n_f) for _ in range(n_f)]
            f = statistics.fmean(fam[i] for i in idx)
            u = statistics.fmean(uns[i] for i in idx)
        else:
            f = statistics.fmean(fam[rng.randrange(n_f)] for _ in range(n_f))
            u = statistics.fmean(uns[rng.randrange(n_u)] for _ in range(n_u))
        g = generalization_gap(f, u)
        if g is not None:
            draws.append(g)
    if len(draws) >= 0.9 * resamples:
        draws.sort()
        out["ci_low"] = round(draws[int(0.025 * (len(draws) - 1))], 4)
        out["ci_high"] = round(draws[int(0.975 * (len(draws) - 1))], 4)
    else:
        out["reason"] = "interval undefined: familiar throughput zero in many resamples"
    return out


SUMMARY_METRICS: Final = (
    PRIMARY_METRIC, "percent_reachable_opened", "percent_reachable_found",
    "exploration_coverage", "interaction_success_rate", "navigation_success_rate",
    "recovery_success_rate", "detection_precision", "detection_recall",
    "chest_distance_error_m", "pose_error_mean_m", "travel_distance_m",
    "agent_cpu_per_sim_second", "frontier_selection_ms_mean", "inference_mean_ms",
    "peak_rss_mb", "sim_seconds", "wall_seconds",
)


def _num(r: dict[str, Any], key: str) -> float | None:
    v = r.get(key)
    if v is None or v == "":
        return None
    return float(v)


def summarise_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entry: dict[str, Any] = {"episodes": len(rows)}
    for key in SUMMARY_METRICS:
        entry[key] = summarise([x for r in rows if (x := _num(r, key)) is not None])
    tp = sum(int(r.get("detection_tp") or 0) for r in rows)
    fp = sum(int(r.get("detection_fp") or 0) for r in rows)
    fn = sum(int(r.get("detection_fn") or 0) for r in rows)
    entry["detection_precision_pooled"] = round(tp / (tp + fp), 4) if tp + fp else None
    entry["detection_recall_pooled"] = round(tp / (tp + fn), 4) if tp + fn else None
    att = sum(int(r.get("recovery_attempts") or 0) for r in rows)
    entry["recovery_success_pooled"] = round(
        sum(int(r.get("recovery_resolved") or 0) for r in rows) / att, 4) if att else None
    ia = sum(int(r.get("interaction_attempts") or 0) for r in rows)
    entry["interaction_success_pooled"] = round(
        sum(int(r.get("chests_opened_reported") or 0) for r in rows) / ia, 4) if ia else None
    reach = sum(int(r.get("chests_reachable") or 0) for r in rows)
    entry["reachable_opened_pooled"] = round(
        sum(int(r.get("chests_opened_truth") or 0) for r in rows) / reach, 4) if reach else None
    entry["outcomes"] = dict(Counter(str(r.get("outcome")) for r in rows))
    entry["over_reported_total"] = sum(int(r.get("over_reported") or 0) for r in rows)
    entry["supervisor_aborts"] = sum(1 for r in rows if r.get("supervisor_aborted")
                                     in (True, "True", 1, "1"))
    entry["strategies_used"] = dict(Counter(str(r.get("strategy_used")) for r in rows))
    return entry


def _paired_diff(a_rows: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]],
                 metric: str) -> dict[str, Any]:
    """``a - b`` over episodes present in both (matched by family and episode)."""
    b = {(r["family"], int(r["episode"])): r for r in b_rows}
    diffs = []
    for r in a_rows:
        other = b.get((r["family"], int(r["episode"])))
        x, y = _num(r, metric), (_num(other, metric) if other else None)
        if x is not None and y is not None:
            diffs.append(x - y)
    st = summarise(diffs)
    return {"difference": st, "better": sum(d > 1e-9 for d in diffs),
            "worse": sum(d < -1e-9 for d in diffs), "tied": sum(abs(d) <= 1e-9 for d in diffs),
            "significant_at_95": st["n"] > 1 and (st["ci_low"] > 0 or st["ci_high"] < 0)}


def summarise_generalization(rows: Sequence[dict[str, Any]], setups: Sequence[str],
                             categories: Sequence[str]) -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = {}
    gaps: dict[str, dict[str, Any]] = {}
    for s in setups:
        per[s], gaps[s] = {}, {}
        fam = [r for r in rows if r["setup"] == s and r["category"] == "familiar"]
        for c in categories:
            rs = [r for r in rows if r["setup"] == s and r["category"] == c]
            per[s][c] = summarise_group(rs)
            per[s][c]["by_family"] = {
                f: summarise([x for r in rs if r["family"] == f
                              and (x := _num(r, PRIMARY_METRIC)) is not None])
                for f in CATEGORIES}
            if c == "familiar" or not fam or not rs:
                continue
            paired = c in PAIRED_CATEGORIES
            if paired:
                keyed = {(r["family"], int(r["episode"])): r for r in fam}
                pairs = [(keyed[(r["family"], int(r["episode"]))], r) for r in rs
                         if (r["family"], int(r["episode"])) in keyed]
                fv = [float(a[PRIMARY_METRIC]) for a, _ in pairs]
                uv = [float(b[PRIMARY_METRIC]) for _, b in pairs]
            else:
                fv = [float(r[PRIMARY_METRIC]) for r in fam]
                uv = [float(r[PRIMARY_METRIC]) for r in rs]
            gaps[s][c] = bootstrap_gap(fv, uv, paired=paired)
            if paired:
                gaps[s][c]["paired_throughput_difference"] = _paired_diff(rs, fam,
                                                                          PRIMARY_METRIC)
    comparison: dict[str, Any] = {}
    if "phase7_active" in setups and "phase6_priors" in setups:
        for c in categories:
            a = [r for r in rows if r["setup"] == "phase7_active" and r["category"] == c]
            b = [r for r in rows if r["setup"] == "phase6_priors" and r["category"] == c]
            comparison[c] = {m: _paired_diff(a, b, m) for m in
                             (PRIMARY_METRIC, "percent_reachable_opened",
                              "agent_cpu_per_sim_second", "travel_distance_m")}
    return {"per_setup": per, "generalization_gap": gaps,
            "phase7_active_minus_phase6_priors": comparison, "primary_metric": PRIMARY_METRIC}


# -- benchmark entry points ------------------------------------------------------------------------


def plan_generalization_jobs(setups: Sequence[str], categories: Sequence[str], *, episodes: int,
                             seed: int, settings_path: str, model_dir: str | None,
                             memory_path: str | None, max_ticks: int, width: int, height: int,
                             start_episode: int = 0) -> list[P7Job]:
    eval_seed = evaluation_seed(seed)
    jobs = []
    for e in range(start_episode, start_episode + episodes):
        for c in categories:
            family, mid = evaluation_map(c, e)
            for s in setups:
                jobs.append(P7Job(s, c, family, mid, eval_seed, e, settings_path, model_dir,
                                  memory_path, max_ticks, width, height))
    return jobs


def _validate(setups: Sequence[str], episodes: int, model_dir: Any) -> None:
    from chest_hunter.runtime.configurations import PHASE6_SETUPS

    unknown = [s for s in setups if s not in PHASE7_SETUPS]
    if unknown:
        raise ValidationError(f"unknown configurations {unknown}; choose from "
                              f"{', '.join(PHASE7_SETUPS)}")
    if episodes < 1:
        raise ValidationError("episodes must be at least 1")
    if model_dir is None and any(PHASE6_SETUPS[PHASE7_SETUPS[s].base].policy != "deterministic"
                                 for s in setups):
        raise ValidationError("hybrid configurations need --model")


def _meta(settings: Settings, **extra: Any) -> dict[str, Any]:
    from chest_hunter.runtime.configurations import AGENT_VERSION

    return {"agent_version": AGENT_VERSION, "generator_version": GENERATOR_VERSION,
            "environment": "Python reference arena (protocol-identical mock of the Unreal "
                           "research server); NOT Unreal Engine",
            "time_basis": "throughput per simulated minute; CPU and wall seconds are real "
                          "time", "config": str(settings.config_path), **extra}


def run_generalization_benchmark(
        settings: Settings, *, setups: Sequence[str] = DEFAULT_GENERALIZATION_SETUPS,
        categories: Sequence[str] = tuple(EVAL_CATEGORIES), episodes: int = 100, seed: int = 42,
        model_dir: str | Path | None = None, memory_path: str | Path | None = None,
        workers: int = 1, max_ticks: int = 6000, width: int = 320, height: int = 180,
        start_episode: int = 0, output_dir: str | Path | None = None,
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    _validate(setups, episodes, model_dir)
    bad = [c for c in categories if c not in EVAL_CATEGORIES]
    if bad:
        raise ValidationError(f"unknown evaluation categories {bad}")
    if "familiar" not in categories:
        raise ValidationError("the generalisation gap needs the 'familiar' category")
    jobs = plan_generalization_jobs(
        setups, categories, episodes=episodes, seed=seed,
        settings_path=str(settings.config_path), model_dir=str(model_dir) if model_dir else None,
        memory_path=str(memory_path) if memory_path else None, max_ticks=max_ticks, width=width,
        height=height, start_episode=start_episode)
    started = time.perf_counter()
    rows, errors = run_jobs(jobs, workers=workers, progress=progress)
    rows.sort(key=lambda r: (int(r["episode"]), list(categories).index(r["category"]),
                             list(setups).index(r["setup"])))
    summary = summarise_generalization(rows, setups, categories)
    summary.update(_meta(
        settings, setups={s: PHASE7_SETUPS[s].description for s in setups},
        categories={c: EVAL_CATEGORIES[c] for c in categories}, episodes_requested=episodes,
        episodes_completed=len(rows), start_episode=start_episode, seed=seed,
        evaluation_seed=evaluation_seed(seed), max_ticks=max_ticks, observation=[width, height],
        workers=workers, model_dir=str(model_dir) if model_dir else None,
        memory_path=str(memory_path) if memory_path else None,
        wall_seconds=round(time.perf_counter() - started, 1), errors=errors))
    if output_dir is not None:
        write_outputs(rows, summary, Path(output_dir), format_generalization_report(summary))
    return summary


def run_strategy_comparison(
        settings: Settings, *, setups: Sequence[str] = STRATEGY_SETUPS,
        categories: Sequence[str] = ("unseen_layout",), episodes: int = 20, seed: int = 42,
        model_dir: str | Path | None = None, memory_path: str | Path | None = None,
        workers: int = 1, max_ticks: int = 6000, width: int = 320, height: int = 180,
        output_dir: str | Path | None = None,
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Fixed strategies against the bandit-selected one on the same held-out episodes."""
    _validate(setups, episodes, model_dir)
    jobs = plan_generalization_jobs(
        setups, categories, episodes=episodes, seed=seed,
        settings_path=str(settings.config_path), model_dir=str(model_dir) if model_dir else None,
        memory_path=str(memory_path) if memory_path else None, max_ticks=max_ticks, width=width,
        height=height)
    started = time.perf_counter()
    rows, errors = run_jobs(jobs, workers=workers, progress=progress)
    per = {s: summarise_group([r for r in rows if r["setup"] == s]) for s in setups}
    ref = "phase7_active" if "phase7_active" in setups else setups[0]
    paired = {s: {m: _paired_diff([r for r in rows if r["setup"] == s],
                                  [r for r in rows if r["setup"] == ref], m)
                  for m in (PRIMARY_METRIC, "exploration_coverage", "travel_distance_m")}
              for s in setups if s != ref}
    memory = _memory(str(memory_path)) if memory_path else None
    summary = {"per_setup": per, f"paired_minus_{ref}": paired, "reference": ref,
               "bandit_choice": {f: memory_strategy(memory, f) for f in CATEGORIES},
               "bandit_posteriors": {f: {a: [round(x, 4) for x in memory.bandit.posterior(f, a)]
                                         for a in memory.bandit.arms}
                                     for f in CATEGORIES} if memory is not None else None,
               "primary_metric": PRIMARY_METRIC}
    summary.update(_meta(settings, categories=list(categories), episodes_requested=episodes,
                         episodes_completed=len(rows), seed=seed,
                         evaluation_seed=evaluation_seed(seed), max_ticks=max_ticks,
                         observation=[width, height], workers=workers,
                         wall_seconds=round(time.perf_counter() - started, 1), errors=errors))
    if output_dir is not None:
        write_outputs(rows, summary, Path(output_dir), format_strategy_report(summary))
    return summary


def train_exploration_memory(
        settings: Settings, *, episodes: int = 80, seed: int = 1, start_episode: int = 0,
        model_dir: str | Path | None = None, memory_path: str | Path,
        workers: int = 1, max_ticks: int = 6000, width: int = 320, height: int = 180,
        setup: str = "phase7_active", progress: Callable[[str], None] | None = None
        ) -> dict[str, Any]:
    """Train the exploration memory on training maps with training seeds only.

    The bandit samples one strategy per episode (fixed for the whole episode); the memory
    is updated *between* episodes (after each batch of ``workers`` parallel episodes) and
    saved atomically after every update.
    """
    from chest_hunter.exploration.self_improving import EpisodeOutcome, ExplorationMemory
    from chest_hunter.runtime.configurations import AGENT_VERSION

    assert_training_seed(seed)
    _validate((setup,), episodes, model_dir)
    if not PHASE7_SETUPS[setup].exploration:
        raise ValidationError("training needs an active-exploration configuration")
    path = Path(memory_path)
    memory = ExplorationMemory.load(path, agent_version=AGENT_VERSION,
                                    generator_version=GENERATOR_VERSION)
    rng = random.Random(f"bandit:{seed}:{start_episode}:{memory.updates}")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    started = time.perf_counter()
    batch = max(1, workers)
    for first in range(start_episode, start_episode + episodes, batch):
        jobs = []
        for e in range(first, min(first + batch, start_episode + episodes)):
            family, mid = training_map(e)
            arm = memory.bandit.select(family, rng)
            jobs.append(P7Job(setup, "training", family, mid, seed, e, str(settings.config_path),
                              str(model_dir) if model_dir else None, str(path), max_ticks, width,
                              height, strategy_override=arm, evaluate=False))
        memory.save(path)            # workers read the memory as of this batch
        got, errs = run_jobs(jobs, workers=min(batch, len(jobs)))
        errors += errs
        for row in sorted(got, key=lambda r: int(r["episode"])):
            outcome = EpisodeOutcome(**row.pop("_outcome"))
            memory.update(outcome, map_id=row["map_id"], training=True)
            rows.append(row)
        memory.save(path)
        if progress:
            progress(f"  ... trained on {len(rows)}/{episodes} episodes")
    return {"memory_path": str(path), "episodes": len(rows), "errors": errors,
            "updates": memory.updates, "seed": seed,
            "arms_played": dict(Counter(f"{r['family']}:{r['strategy_used']}" for r in rows)),
            "throughput_mean": summarise([float(r[PRIMARY_METRIC]) for r in rows]),
            "bandit_choice": {f: memory_strategy(memory, f) for f in CATEGORIES},
            "wall_seconds": round(time.perf_counter() - started, 1),
            "rows": rows}


# -- outputs --------------------------------------------------------------------------------------


def write_outputs(rows: Sequence[dict[str, Any]], summary: dict[str, Any], out: Path,
                  report: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in FIELDS})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n",
                                      encoding="utf-8")
    (out / "report.md").write_text(report + "\n", encoding="utf-8")


def _ci(st: dict[str, Any] | None, digits: int = 3) -> str:
    if not st or st.get("mean") is None:
        return "n/a"
    return (f"{st['mean']:.{digits}f} [{st['ci_low']:.{digits}f}, {st['ci_high']:.{digits}f}]"
            f" (n={st['n']})")


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f} %"


def format_generalization_report(summary: dict[str, Any]) -> str:
    lines = ["# Phase 7 generalisation benchmark", "",
             f"Environment: {summary.get('environment')}. Agent {summary.get('agent_version')}, "
             f"generator {summary.get('generator_version')}. Seed {summary.get('seed')} "
             f"(evaluation seed {summary.get('evaluation_seed')}), "
             f"{summary.get('episodes_requested')} episodes per category and configuration, "
             f"{summary.get('max_ticks')} ticks, observation {summary.get('observation')}.",
             f"Completed episode rows: {summary.get('episodes_completed')}; errors: "
             f"{len(summary.get('errors') or [])}.", "",
             "Throughput = ground-truth chests opened per simulated minute (mean, 95 % CI).", ""]
    for s, cats in summary["per_setup"].items():
        lines += [f"## {s}", "",
                  "| category | n | throughput | reachable opened | coverage | precision | "
                  "recall | chest dist. err (m) | pose err (m) | nav success | recovery | G "
                  "[95 % CI] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, e in cats.items():
            g = summary["generalization_gap"].get(s, {}).get(c)
            gtxt = "-" if c == "familiar" else (
                "n/a" if not g or g.get("G") is None else
                f"{g['G']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]"
                if g.get("ci_low") is not None else f"{g['G']:+.3f}")
            lines.append(
                f"| {c} | {e['episodes']} | {_ci(e[PRIMARY_METRIC])} | "
                f"{_pct(e['reachable_opened_pooled'])} | {_ci(e['exploration_coverage'])} | "
                f"{_pct(e['detection_precision_pooled'])} | {_pct(e['detection_recall_pooled'])}"
                f" | {_ci(e['chest_distance_error_m'], 2)} | {_ci(e['pose_error_mean_m'], 2)} | "
                f"{_ci(e['navigation_success_rate'])} | {_pct(e['recovery_success_pooled'])} | "
                f"{gtxt} |")
        lines.append("")
    comp = summary.get("phase7_active_minus_phase6_priors") or {}
    if comp:
        lines += ["## Phase 7 active exploration minus Phase 6 (spawn points given), paired",
                  "", "| category | throughput diff | better / worse / tied | significant |",
                  "|---|---|---|---|"]
        for c, d in comp.items():
            t = d[PRIMARY_METRIC]
            lines.append(f"| {c} | {_ci(t['difference'])} | {t['better']} / {t['worse']} / "
                         f"{t['tied']} | {t['significant_at_95']} |")
        lines.append("")
    return "\n".join(lines)


def format_strategy_report(summary: dict[str, Any]) -> str:
    lines = ["# Exploration strategies: fixed vs bandit-selected", "",
             f"Environment: {summary.get('environment')}. Categories "
             f"{summary.get('categories')}, {summary.get('episodes_requested')} episodes, seed "
             f"{summary.get('seed')}.", "",
             "| configuration | n | throughput | coverage | travel (m) | strategies used |",
             "|---|---|---|---|---|---|"]
    for s, e in summary["per_setup"].items():
        lines.append(f"| {s} | {e['episodes']} | {_ci(e[PRIMARY_METRIC])} | "
                     f"{_ci(e['exploration_coverage'])} | {_ci(e['travel_distance_m'], 1)} | "
                     f"{e['strategies_used']} |")
    ref = summary["reference"]
    lines += ["", f"Paired difference to {ref} (throughput):", "",
              "| configuration | difference | better / worse / tied | significant |",
              "|---|---|---|---|"]
    for s, d in summary[f"paired_minus_{ref}"].items():
        t = d[PRIMARY_METRIC]
        lines.append(f"| {s} | {_ci(t['difference'])} | {t['better']} / {t['worse']} / "
                     f"{t['tied']} | {t['significant_at_95']} |")
    lines += ["", f"Bandit choice per map family: {summary.get('bandit_choice')}"]
    return "\n".join(lines)
