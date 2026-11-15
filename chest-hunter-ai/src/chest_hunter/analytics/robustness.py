"""Phase 7 robustness benchmark: controlled failure injection on held-out maps.

Episode ``e`` injects fault ``kinds[e % len(kinds)]`` (``runtime.faults``) into an episode on
the held-out layout ``evaluation_map("unseen_layout", e)``, at a start time drawn from the
fault's window (``FaultPlan.random(kind, e)``). Every faulted episode has a paired baseline:
the same configuration, map, seed and episode number without the fault, so the throughput
the fault cost is measured episode by episode.

Per fault category it reports: injected and applied counts (a fault planned after the end of
an episode, or one whose trigger never occurred, is not counted as tested), detection rate
(kind-specific agent/supervisor signals; not applicable to faults the agent cannot tell apart
from the world), recovery success rate (no failure, and progress within 30 simulated seconds
after the fault), clean-termination rate, episodes that failed, and the paired throughput
difference to the baseline.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from chest_hunter.analytics.phase7_benchmark import (
    PHASE7_SETUPS,
    PRIMARY_METRIC,
    P7Job,
    _ci,
    _journal,
    _meta,
    _paired_diff,
    _validate,
    evaluation_map,
    evaluation_seed,
    run_jobs,
    write_outputs,
)
from chest_hunter.analytics.stats import summarise
from chest_hunter.config import Settings
from chest_hunter.exceptions import ValidationError
from chest_hunter.runtime.faults import DETECTION_SIGNALS, FAULT_KINDS


def plan_robustness_jobs(*, setup: str, kinds: Sequence[str], episodes: int, seed: int,
                         settings_path: str, model_dir: str | None, memory_path: str | None,
                         max_ticks: int, width: int, height: int,
                         baseline: bool = True) -> list[P7Job]:
    eval_seed = evaluation_seed(seed)
    jobs = []
    for e in range(episodes):
        family, mid = evaluation_map("unseen_layout", e)
        kind = kinds[e % len(kinds)]
        jobs.append(P7Job(setup, "fault", family, mid, eval_seed, e, settings_path, model_dir,
                          memory_path, max_ticks, width, height, fault_kind=kind, fault_seed=e,
                          evaluate=False))
        if baseline:
            jobs.append(P7Job(setup, "baseline", family, mid, eval_seed, e, settings_path,
                              model_dir, memory_path, max_ticks, width, height, evaluate=False))
    return jobs


def _rate(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    hits = sum(1 for v in vals if v in (True, "True", 1, "1"))
    return {"rate": round(hits / len(vals), 4) if vals else None, "of": len(vals), "yes": hits}


def summarise_robustness(rows: Sequence[dict[str, Any]], kinds: Sequence[str]
                         ) -> dict[str, Any]:
    faulted = [r for r in rows if r["category"] == "fault"]
    base = [r for r in rows if r["category"] == "baseline"]
    per: dict[str, Any] = {}
    for k in kinds:
        rs = [r for r in faulted if r.get("fault_kind") == k]
        tested = [r for r in rs if r.get("fault_applied") in (True, "True", 1, "1")]
        entry: dict[str, Any] = {
            "episodes": len(rs),
            "injected": sum(1 for r in rs if r.get("fault_injected") in (True, "True")),
            "applied": len(tested),
            "detection": _rate(tested, "fault_detected") if DETECTION_SIGNALS[k] is not None
            else {"rate": None, "note": "not applicable: the agent cannot tell this fault "
                                        "apart from the world"},
            "recovery": _rate(tested, "fault_recovered"),
            "terminated_cleanly": _rate(tested, "fault_terminated_cleanly"),
            "failed_episodes": sum(1 for r in tested if r["outcome"] == "failed"),
            "end_reasons_of_failures": sorted({str(r.get("end_reason")) for r in tested
                                               if r["outcome"] == "failed"}),
            PRIMARY_METRIC: summarise([float(r[PRIMARY_METRIC]) for r in tested]),
            "paired_throughput_minus_baseline": _paired_diff(tested, base, PRIMARY_METRIC),
        }
        per[k] = entry
    tested_all = [r for r in faulted if r.get("fault_applied") in (True, "True", 1, "1")]
    return {"per_fault": per,
            "overall": {"faulted_episodes": len(faulted), "applied": len(tested_all),
                        "recovery": _rate(tested_all, "fault_recovered"),
                        "terminated_cleanly": _rate(tested_all, "fault_terminated_cleanly"),
                        "failed_episodes": sum(1 for r in tested_all
                                               if r["outcome"] == "failed")},
            "baseline": {"episodes": len(base),
                         PRIMARY_METRIC: summarise([float(r[PRIMARY_METRIC]) for r in base]),
                         "failed_episodes": sum(1 for r in base if r["outcome"] == "failed")},
            "primary_metric": PRIMARY_METRIC}


def run_robustness_benchmark(
        settings: Settings, *, setup: str = "phase7_active",
        kinds: Sequence[str] = FAULT_KINDS, episodes: int = 100, seed: int = 42,
        model_dir: str | Path | None = None, memory_path: str | Path | None = None,
        workers: int = 1, max_ticks: int = 4500, width: int = 320, height: int = 180,
        baseline: bool = True, output_dir: str | Path | None = None, resume: bool = False,
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    _validate((setup,), episodes, model_dir)
    bad = [k for k in kinds if k not in FAULT_KINDS]
    if bad:
        raise ValidationError(f"unknown fault kinds {bad}; choose from {', '.join(FAULT_KINDS)}")
    jobs = plan_robustness_jobs(setup=setup, kinds=kinds, episodes=episodes, seed=seed,
                                settings_path=str(settings.config_path),
                                model_dir=str(model_dir) if model_dir else None,
                                memory_path=str(memory_path) if memory_path else None,
                                max_ticks=max_ticks, width=width, height=height,
                                baseline=baseline)
    started = time.perf_counter()
    journal = _journal(output_dir, {"kind": "robustness", "setup": setup, "kinds": list(kinds),
                                    "episodes": episodes, "seed": seed, "max_ticks": max_ticks,
                                    "baseline": baseline, "memory": str(memory_path)}, resume)
    rows, errors = run_jobs(jobs, workers=workers, progress=progress, journal=journal)
    rows.sort(key=lambda r: (int(r["episode"]), r["category"]))
    summary = summarise_robustness(rows, kinds)
    summary.update(_meta(settings, setup=setup, description=PHASE7_SETUPS[setup].description,
                         kinds=list(kinds), episodes_requested=episodes, seed=seed,
                         evaluation_seed=evaluation_seed(seed), max_ticks=max_ticks,
                         observation=[width, height], workers=workers, baseline=baseline,
                         wall_seconds=round(time.perf_counter() - started, 1), errors=errors))
    if output_dir is not None:
        write_outputs(rows, summary, Path(output_dir), format_robustness_report(summary))
    return summary


def format_robustness_report(summary: dict[str, Any]) -> str:
    def rate(d: dict[str, Any]) -> str:
        if d.get("rate") is None:
            return "n/a"
        return f"{100 * d['rate']:.0f} % ({d['yes']}/{d['of']})"

    lines = ["# Phase 7 robustness benchmark (controlled failure injection)", "",
             f"Environment: {summary.get('environment')}. Configuration {summary.get('setup')}; "
             f"{summary.get('episodes_requested')} faulted episodes on held-out layouts, "
             f"each with a paired fault-free baseline; {summary.get('max_ticks')} ticks.", "",
             "| fault | applied / episodes | detected | recovered | clean end | failed | "
             "throughput | paired diff vs baseline |",
             "|---|---|---|---|---|---|---|---|"]
    for k, e in summary["per_fault"].items():
        d = e["paired_throughput_minus_baseline"]["difference"]
        lines.append(f"| {k} | {e['applied']} / {e['episodes']} | {rate(e['detection'])} | "
                     f"{rate(e['recovery'])} | {rate(e['terminated_cleanly'])} | "
                     f"{e['failed_episodes']} | {_ci(e[PRIMARY_METRIC])} | {_ci(d)} |")
    o, b = summary["overall"], summary["baseline"]
    lines += ["", f"Overall: {o['applied']} faults applied; recovered {rate(o['recovery'])}; "
              f"clean termination {rate(o['terminated_cleanly'])}; failed episodes "
              f"{o['failed_episodes']}.",
              f"Baseline (no fault): {b['episodes']} episodes, throughput "
              f"{_ci(b[PRIMARY_METRIC])}, failed {b['failed_episodes']}."]
    return "\n".join(lines)
