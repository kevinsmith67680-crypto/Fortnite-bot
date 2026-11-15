"""Phase 8 ``audit-system``: one reproducible entry point for the full-system audit.

Runs every evaluation the environment permits and writes::

    component_inventory.json  integration_results.json  vision_metrics.json
    navigation_metrics.json   interaction_metrics.json  policy_comparison.csv
    planner_comparison.csv    endurance_results.csv     generalisation_results.csv
    resource_profile.json     database_integrity.json   failures.json
    audit_summary.json

An evaluation that cannot run is recorded with status ``BLOCKED`` (and why) - never with
invented numbers or zeros. Long benchmark results already produced by the Phase 6/7
commands (``benchmark_results/phase7``, ``.../robustness``, ``.../phase6/endurance_*``) are
*reused* rather than recomputed when present; the summary names their source and the
commit that produced them.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

STATUSES = ("VERIFIED", "PARTIALLY VERIFIED", "MOCK ONLY", "UNVERIFIED", "BROKEN")


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                             for k, v in r.items()})


class Audit:
    def __init__(self, settings: Any, out: Path, *, episodes: int, workers: int,
                 held_out: bool, profile: bool, verify_db: bool, model_dir: Path | None,
                 progress: Callable[[str], None] | None = None) -> None:
        self.settings = settings
        self.root = Path(settings.base_dir)
        self.out = out
        self.episodes = episodes
        self.workers = workers
        self.held_out = held_out
        self.profile = profile
        self.verify_db = verify_db
        self.model_dir = model_dir
        self.progress = progress or (lambda m: None)
        self.sections: dict[str, dict[str, Any]] = {}
        self.failures: list[dict[str, Any]] = []

    def section(self, name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        self.progress(f"[audit] {name} ...")
        try:
            result = fn()
            status = "MEASURED"
            error = None
        except Exception as exc:                       # recorded, the audit continues
            result, status = None, "BLOCKED"
            error = f"{type(exc).__name__}: {exc}"
            self.failures.append({"section": name, "kind": "evaluation_error",
                                  "detail": error, "trace": traceback.format_exc()[-2000:]})
        self.sections[name] = {"status": status, "seconds": round(time.perf_counter()
                                                                  - started, 1),
                               "error": error}
        self.progress(f"[audit] {name}: {status} ({self.sections[name]['seconds']} s)")
        return result

    # -- evaluations ---------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        from chest_hunter.audit import (
            bridge_security,
            db_integrity,
            interaction_eval,
            leakage,
            loop_trace,
            navigation_eval,
            static_scan,
            unreal_static,
            vision_eval,
        )

        self.out.mkdir(parents=True, exist_ok=True)
        s = self.settings
        scan = self.section("static_scan", lambda: static_scan.scan(self.root).to_dict())
        unreal = self.section("unreal_static", lambda: unreal_static.audit(self.root))
        trace = self.section("loop_trace", lambda: loop_trace.run_trace(
            s, self.out / "loop_trace", model_dir=self.model_dir))
        leak = self.section("ground_truth_isolation", lambda: leakage.evaluate(
            s, self.root / "src"))
        security = self.section("bridge_security", bridge_security.evaluate)
        _write_json(self.out / "integration_results.json", {
            "loop_trace": trace, "ground_truth_isolation": leak, "bridge_security": security,
            "unreal": unreal})
        vision = self.section("vision", lambda: vision_eval.evaluate(
            s, failures_dir=self.out / "vision_failures"))
        _write_json(self.out / "vision_metrics.json", vision)
        nav = self.section("navigation", lambda: navigation_eval.evaluate(
            s, workers=self.workers))
        _write_json(self.out / "navigation_metrics.json", nav)
        inter = self.section("interaction", lambda: {
            "environment_rules": interaction_eval.environment_rules(),
            "agent_crosscheck": interaction_eval.agent_crosscheck(
                s, episodes=max(2, self.episodes // 10), workers=self.workers)})
        _write_json(self.out / "interaction_metrics.json", inter)
        policies = self.section("policy_comparison", lambda: self._comparison(
            ("audit_policy_deterministic", "audit_policy_learned", "audit_policy_hybrid"),
            "policy_comparison"))
        planners = self.section("planner_comparison", lambda: self._comparison(
            ("audit_planner_nearest", "audit_planner_adaptive", "audit_planner_greedy",
             "audit_planner_short_horizon", "audit_planner_frontier"), "planner_comparison"))
        endurance = self.section("endurance", self._endurance)
        general = self.section("generalisation", self._generalisation) \
            if self.held_out else None
        robustness = self.section("robustness", self._robustness)
        resources = self.section("resources", self._resources) if self.profile else None
        _write_json(self.out / "resource_profile.json", resources if resources is not None
                    else {"status": "BLOCKED" if self.profile else "NOT REQUESTED"})
        database = self.section("database", lambda: db_integrity.evaluate(
            s, self.out / "db_work")) if self.verify_db else None
        _write_json(self.out / "database_integrity.json", database if database is not None
                    else {"status": "NOT REQUESTED"})
        from chest_hunter.audit.components import assess_components

        components = assess_components({"inter": inter})
        inventory = self._inventory(scan, unreal, components)
        _write_json(self.out / "component_inventory.json", inventory)
        summary = self._summary(components, trace=trace, leak=leak, security=security,
                                vision=vision, nav=nav, inter=inter, policies=policies,
                                planners=planners,
                                endurance=endurance, general=general, robustness=robustness,
                                resources=resources, database=database, unreal=unreal,
                                scan=scan)
        _write_json(self.out / "failures.json", self.failures)
        _write_json(self.out / "audit_summary.json", summary)
        return summary

    def _comparison(self, setups: tuple[str, ...], name: str) -> dict[str, Any]:
        from chest_hunter.analytics.phase7_benchmark import run_strategy_comparison

        summary = run_strategy_comparison(
            self.settings, setups=setups, categories=("unseen_layout",),
            episodes=self.episodes, model_dir=self.model_dir, workers=self.workers,
            max_ticks=self.settings.exploration.benchmark_max_ticks,
            output_dir=self.out / name, resume=True)
        rows = []
        for setup, e in summary["per_setup"].items():
            rows.append({"configuration": setup, "episodes": e["episodes"],
                         **{f"{k}_mean": (e[k] or {}).get("mean") for k in (
                             "throughput_truth_per_min", "percent_reachable_opened",
                             "navigation_success_rate", "interaction_success_rate",
                             "travel_distance_m", "agent_cpu_per_sim_second",
                             "inference_mean_ms", "frontier_selection_ms_mean")},
                         "throughput_ci_low": e["throughput_truth_per_min"]["ci_low"],
                         "throughput_ci_high": e["throughput_truth_per_min"]["ci_high"],
                         "interaction_success_pooled": e["interaction_success_pooled"],
                         "failed_episodes": e["outcomes"].get("failed", 0),
                         "over_reported_total": e["over_reported_total"]})
        _write_csv(self.out / f"{name}.csv", rows)
        return summary

    def _endurance(self) -> dict[str, Any]:
        from chest_hunter.analytics.endurance import EnduranceConfig, EnduranceRunner

        found = sorted((self.root / "benchmark_results" / "phase6")
                       .glob("endurance_*/summary.json"))
        rows = []
        for path in found:
            rows.append({"run": path.parent.name, "source": "reused",
                         **endurance_row(json.loads(path.read_text(encoding="utf-8")))})
        smoke_dir = self.out / "endurance_smoke"
        if smoke_dir.exists():
            import shutil

            shutil.rmtree(smoke_dir)
        cfg = EnduranceConfig(episodes=10, max_ticks_per_episode=1500,
                              maps=("research_arena_3d", "p7_mixed_test_005"))
        smoke = EnduranceRunner(self.settings, cfg, smoke_dir,
                                install_signal_handlers=False).run()
        rows.append({"run": "level1_smoke_10", "source": "this audit", **endurance_row(smoke)})
        _write_csv(self.out / "endurance_results.csv", rows)
        return {"runs": rows, "smoke": smoke}

    def _generalisation(self) -> dict[str, Any]:
        path = self.root / "benchmark_results" / "phase7" / "summary.json"
        if not path.exists():
            raise FileNotFoundError("run 'benchmark-generalization --held-out-maps' first "
                                    "(about 1600 episodes; not repeated by the audit)")
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for setup, cats in data["per_setup"].items():
            for cat, e in cats.items():
                g = data["generalization_gap"].get(setup, {}).get(cat) or {}
                rows.append({"configuration": setup, "category": cat,
                             "episodes": e["episodes"],
                             "throughput_mean": e["throughput_truth_per_min"]["mean"],
                             "throughput_ci_low": e["throughput_truth_per_min"]["ci_low"],
                             "throughput_ci_high": e["throughput_truth_per_min"]["ci_high"],
                             "detection_precision": e["detection_precision_pooled"],
                             "detection_recall": e["detection_recall_pooled"],
                             "navigation_success_mean": e["navigation_success_rate"]["mean"],
                             "reachable_opened": e["reachable_opened_pooled"],
                             "coverage_mean": e["exploration_coverage"]["mean"],
                             "recovery_success": e["recovery_success_pooled"],
                             "outcomes": e["outcomes"], "gap_G": g.get("G"),
                             "gap_ci_low": g.get("ci_low"), "gap_ci_high": g.get("ci_high")})
        _write_csv(self.out / "generalisation_results.csv", rows)
        return {"source": str(path), "episodes_completed": data.get("episodes_completed"),
                "errors": len(data.get("errors") or []), "rows": rows}

    def _robustness(self) -> dict[str, Any]:
        path = self.root / "benchmark_results" / "robustness" / "summary.json"
        if not path.exists():
            raise FileNotFoundError("run 'benchmark-robustness --failure-injection' first")
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"source": str(path), "per_fault": data["per_fault"],
                "overall": data["overall"], "baseline": data["baseline"]}

    def _resources(self) -> dict[str, Any]:
        """Sequential profiles (one process) at the four configurations of the brief."""
        from chest_hunter.agent.controller_3d import Agent3DConfig, apply_vision_profile
        from chest_hunter.agent.runner_3d import Runner3D
        from chest_hunter.analytics.phase7_benchmark import evaluation_map
        from chest_hunter.analytics.resource_monitor import process_rss_mb, resource_report
        from chest_hunter.database import DatabaseManager
        from chest_hunter.exploration.active import ExplorationConfig
        from chest_hunter.runtime.profiler import Profiler

        configs = {"320x180_5fps": ((320, 180), 5.0), "480x270_5fps": ((480, 270), 5.0),
                   "640x360_10fps": ((640, 360), 10.0), "320x180_adaptive": ((320, 180), None)}
        results: dict[str, Any] = {}
        for name, ((w, h), fps) in configs.items():
            prof = Profiler()
            eps = []
            for ep in range(max(3, self.episodes // 20)):
                mid = evaluation_map("unseen_layout", ep)[1]
                if fps is None:
                    cfg = apply_vision_profile(Agent3DConfig(), "efficient",
                                               skip_unchanged=True)
                else:
                    cfg = Agent3DConfig(rate_travel=fps, rate_search=fps, rate_verify=fps,
                                        rate_idle=fps)
                db = DatabaseManager(":memory:")
                db.initialize()
                try:
                    runner = Runner3D.from_settings(
                        db, self.settings, mode="hybrid", seed=900042, episode=ep,
                        backend="reference", map_id=mid, collect_truth=True,
                        max_ticks=self.settings.exploration.benchmark_max_ticks, width=w,
                        height=h, config=cfg, exploration=ExplorationConfig(), profiler=prof)
                    res = runner.run()
                finally:
                    db.close()
                minutes = res.sim_seconds / 60.0
                eps.append({"episode": ep, "map_id": mid, "outcome": res.outcome,
                            "sim_seconds": round(res.sim_seconds, 2),
                            "throughput_truth_per_min": round(
                                (res.truth.get("opened_truth") or 0) / minutes, 4)
                            if minutes else None,
                            "frames_analysed": res.metrics.analyses,
                            "bridge_round_trip_ms": round(res.bridge_round_trip_ms, 3),
                            "cpu_seconds": round(res.cpu_seconds, 3),
                            "wall_seconds": round(res.wall_seconds, 3),
                            "rss_mb": round(process_rss_mb()[0], 1)})
            results[name] = {"components": prof.report(), "episodes": eps}
        report = resource_report(results, map_id="held-out unseen_layout maps", seed=900042,
                                 policy="deterministic", exploration="active",
                                 observation=(0, 0))
        report["machine"]["platform_detail"] = platform.platform()
        report["note_configurations"] = ("fps = analysed frames per simulated second in every "
                                         "state; adaptive = the 'efficient' vision profile "
                                         "with unchanged-frame skipping")
        return report

    # -- inventory and summary ---------------------------------------------------------------------

    def _inventory(self, scan: Any, unreal: Any, components: list[dict[str, Any]]
                   ) -> dict[str, Any]:
        return {"commit": _git_commit(self.root), "python": sys.version.split()[0],
                "static_scan": {k: (scan or {}).get(k) for k in ("files", "lines", "counts")},
                "static_findings": (scan or {}).get("findings"),
                "modules": (scan or {}).get("modules"),
                "components": components, "unreal": unreal}

    def _summary(self, components: list[dict[str, Any]], **parts: Any) -> dict[str, Any]:
        from chest_hunter.audit.components import readiness

        return {"commit": _git_commit(self.root),
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "environment": "Python reference arena (protocol-identical mock of the Unreal "
                               "research server). Unreal Engine was not available: every "
                               "Unreal-side component is UNVERIFIED.",
                "sections": self.sections,
                "components": components,
                "readiness": readiness(parts),
                "key_metrics": _key_metrics(parts)}


def endurance_row(s: dict[str, Any]) -> dict[str, Any]:
    a, m = s.get("aggregate", {}), s.get("memory", {})
    return {"episodes_requested": s.get("episodes_requested"),
            "episodes_completed": s.get("episodes_completed_in_run"),
            "outcomes": a.get("outcomes"), "failed": a.get("failed"),
            "communication_failures": a.get("communication_failures"),
            "supervisor_interventions": a.get("supervisor_interventions"),
            "supervisor_aborts": a.get("supervisor_aborts"),
            "recovery_success_rate": a.get("recovery_success_rate"),
            "over_reported_total": a.get("over_reported_total"),
            "throughput_mean": (s.get("throughput_truth_per_min") or {}).get("mean"),
            "episode_sim_seconds_mean": (s.get("episode_sim_seconds") or {}).get("mean"),
            "episode_sim_seconds_max": s.get("episode_sim_seconds_max"),
            "cpu_seconds_per_episode": (s.get("cpu_seconds_per_episode") or {}).get("mean"),
            "rss_first_mb": m.get("rss_mb_first"), "rss_last_mb": m.get("rss_mb_last"),
            "rss_max_mb": m.get("rss_mb_max"),
            "rss_growth_mb_per_100": m.get("rss_growth_mb_per_100_episodes"),
            "gc_objects_growth_per_100": m.get("gc_objects_growth_per_100_episodes"),
            "threads_min_max": m.get("threads_min_max"),
            "open_fds_min_max": m.get("open_fds_min_max"), "memory_bounded": m.get("bounded"),
            "stopped_early": s.get("stopped_early")}


def _key_metrics(p: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    v = p.get("vision") or {}
    if v:
        out["detection"] = {k: {m: s.get(m) for m in ("precision", "recall", "f1",
                                                     "latency_mean_ms", "latency_p95_ms")}
                            for k, s in v.get("per_resolution", {}).items()}
    n = p.get("nav") or {}
    if n:
        out["navigation_success_rate"] = n.get("overall_success_rate")
    i = p.get("inter") or {}
    if i:
        out["interaction"] = {k: {m: s.get(m) for m in (
            "interaction_success_rate", "false_success_rate", "false_failure_rate")}
            for k, s in i["agent_crosscheck"]["per_scenario"].items()}
    g = p.get("general") or {}
    if g:
        out["generalisation_rows"] = len(g.get("rows", []))
    return out
