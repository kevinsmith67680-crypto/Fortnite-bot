"""Phase 6 command-line commands: endurance-test, profile-agent, benchmark-phase6, and the
reliability / planning / compute options of ``simulate-3d``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chest_hunter.cli_learning import AddCommand, _experience_db, _model_dir, load_policy

PLANNER_CHOICES = ("phase4", "adaptive", "greedy", "short-horizon")


def _positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _non_negative(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def _positive_float(value: str) -> float:
    x = float(value)
    if not x > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def planner_name(value: str | None) -> str:
    """CLI planner name -> configuration name."""
    return {"phase4": "none", "adaptive": "adaptive", "greedy": "greedy",
            "short-horizon": "short_horizon", None: "none"}[value]


def add_phase6_simulate_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--planner", choices=PLANNER_CHOICES,
                     help="target selection: phase4 rule (default), Phase 5 adaptive, or the "
                          "Phase 6 greedy / short-horizon planner")
    sub.add_argument("--supervisor", choices=("enabled", "disabled"), default="disabled",
                     help="Phase 6 reliability supervisor (default disabled)")
    sub.add_argument("--adaptive-compute", action="store_true",
                     help="Phase 6 adaptive computation (performance.vision_profile, "
                          "unchanged frames skipped)")
    sub.add_argument("--profile", action="store_true",
                     help="print a per-component processing-time profile at the end")


def phase6_extras(ctx: Any, args: argparse.Namespace, map_id: str,
                  extras: dict[str, Any]) -> dict[str, Any]:
    """Apply the Phase 6 options on top of the Phase 5 ``simulate_extras``."""
    from chest_hunter.agent.runner_3d import _layout_hash
    from chest_hunter.runtime.configurations import (
        agent_config,
        custom_setup,
        make_planner,
        partition,
        supervisor_config,
    )

    settings = ctx.settings
    planner = planner_name(getattr(args, "planner", None))
    if planner == "none" and extras.get("planner") is not None:
        planner = "adaptive"                                # --adaptive-planning (Phase 5)
    setup = custom_setup(policy=extras.get("policy", "deterministic"), planner=planner,
                         supervisor=getattr(args, "supervisor", "disabled") == "enabled",
                         adaptive_compute=bool(getattr(args, "adaptive_compute", False)),
                         smart_recovery=bool(getattr(args, "smart_recovery", False))
                         or settings.learning.smart_recovery or planner in ("greedy",
                                                                            "short_horizon"))
    out = dict(extras)
    c3 = settings.simulation_3d
    width = args.observation_width or c3.observation_width
    height = args.observation_height or c3.observation_height
    part = partition(map_id=map_id, env_overrides={
        "odometry_drift": c3.odometry_drift, "noise_std": c3.noise_std,
        "reward_probability": c3.reward_probability,
        "interaction_failure_probability": c3.interaction_failure_probability},
        setup=setup, model_version=getattr(extras.get("learned_policy"), "version", None),
        width=width, height=height, fov_deg=c3.fov_deg, decision_ticks=c3.decision_ticks)
    out["partition"] = part
    out["config"] = agent_config(settings, setup)
    if planner in ("greedy", "short_horizon"):
        out["planner"] = make_planner(settings, setup, ctx.db, map_id, _layout_hash(map_id),
                                      partition_key=part["key"])
    if setup.supervisor:
        out["supervisor"] = supervisor_config(settings, setup)
    if setup.selector_monitors and "policy_options" in out:
        from dataclasses import replace

        limits = out["policy_options"]["limits"]
        out["policy_options"] = {"limits": replace(limits, oscillation_flips=8,
                                                   max_failed_holds=5)}
    if getattr(args, "profile", False):
        from chest_hunter.runtime.profiler import Profiler

        out["profiler"] = Profiler()
    return out


def register(add: AddCommand) -> None:
    sub = add("endurance-test", "Run many consecutive autonomous episodes against one research "
                                "environment (resets between episodes, checkpoints, memory "
                                "tracking).", cmd_endurance_test)
    sub.add_argument("--episodes", type=_positive, help="default endurance.default_episodes")
    sub.add_argument("--max-total-seconds", type=_positive_float, metavar="S",
                     help="wall-clock budget for the whole run")
    sub.add_argument("--max-ticks-per-episode", type=_positive, metavar="N",
                     help="default 18000 (600 simulated seconds)")
    sub.add_argument("--seed", type=_non_negative, default=42)
    sub.add_argument("--policy", choices=("deterministic", "learned", "hybrid"),
                     default="deterministic")
    sub.add_argument("--strategy", "--planner", dest="planner", choices=PLANNER_CHOICES,
                     default="short-horizon", help="target-selection strategy")
    sub.add_argument("--supervisor", choices=("enabled", "disabled"), default="enabled")
    sub.add_argument("--no-adaptive-compute", dest="adaptive_compute", action="store_false")
    sub.add_argument("--map", action="append", metavar="MAP_ID",
                     help="maps played in rotation (repeatable; default research_arena_3d)")
    sub.add_argument("--observation-width", type=_positive, default=320)
    sub.add_argument("--observation-height", type=_positive, default=180)
    sub.add_argument("--backend", choices=("reference", "unreal"), default="reference")
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--preserve-estimates", action="store_true",
                     help="carry spawn estimates and route statistics across the episodes of "
                          "this run (stored in the run's own database)")
    sub.add_argument("--allow-test-maps", action="store_true",
                     help="permit --preserve-estimates on held-out test maps")
    sub.add_argument("--checkpoint-every", type=_positive, metavar="N")
    sub.add_argument("--output", required=True, metavar="DIR")
    sub.add_argument("--resume", action="store_true", help="continue the run in --output")
    sub.add_argument("--dashboard", action="store_true",
                     help="open the operations panel (start, pause, resume, stop, select "
                          "policy / planner / vision profile, export metrics)")

    sub = add("profile-agent", "Profile the agent's processing time per component over several "
                               "episodes and rank the bottlenecks.", cmd_profile_agent)
    sub.add_argument("--episodes", type=_positive, default=20)
    sub.add_argument("--map", metavar="MAP_ID", default="research_arena_3d")
    sub.add_argument("--seed", type=_non_negative, default=42)
    sub.add_argument("--policy", choices=("deterministic", "learned", "hybrid"),
                     default="deterministic")
    sub.add_argument("--planner", choices=PLANNER_CHOICES, default="phase4")
    sub.add_argument("--supervisor", choices=("enabled", "disabled"), default="disabled")
    sub.add_argument("--vision-profiles", default="phase5",
                     help="comma-separated profiles to compare (phase5,efficient,dense); "
                          "each one plays the same episodes")
    sub.add_argument("--skip-unchanged-frames", action="store_true")
    sub.add_argument("--max-ticks", type=_positive, default=18000)
    sub.add_argument("--observation-width", type=_positive, default=640)
    sub.add_argument("--observation-height", type=_positive, default=360)
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--exploration", choices=("spawn-points", "active"),
                     default="spawn-points",
                     help="Phase 7: profile active exploration (with --map procedural: a "
                          "different held-out procedural map every episode)")
    sub.add_argument("--output", metavar="DIR",
                     help="directory for profile.json and the per-profile tables")
    sub.add_argument("--report", metavar="PATH",
                     help="Phase 7 resource report (JSON): Python decision pipeline and "
                          "simulator workload measured separately, memory, per component")

    sub = add("calibrate-policy", "Measure how well the learned policy's phase output predicts "
                                  "reliable commands on held-out episodes and fit a reliability "
                                  "calibrator (validation episodes; reported on test episodes).",
              cmd_calibrate_policy)
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--dataset", metavar="DIR", help="demonstrations with splits.json "
                     "(default: learning.dataset_dir/synthetic_expert_v1)")
    sub.add_argument("--tolerance", type=_positive_float, default=0.25,
                     help="largest command error counted as reliable (default 0.25)")
    sub.add_argument("--output", required=True, metavar="DIR")

    sub = add("benchmark-phase6", "Paired comparison of the Phase 4 baseline, the Phase 5 hybrid "
                                  "and the Phase 6 configurations.", cmd_benchmark_phase6)
    sub.add_argument("--episodes", type=_positive, default=100)
    sub.add_argument("--seed", type=_non_negative, default=42)
    sub.add_argument("--configurations", metavar="LIST",
                     help="comma-separated (default: the five required configurations)")
    sub.add_argument("--map", action="append", metavar="MAP_ID")
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--experience-db", metavar="PATH")
    sub.add_argument("--workers", type=_positive, default=1)
    sub.add_argument("--max-ticks", type=_positive, default=18000)
    sub.add_argument("--output-dir", metavar="DIR", default="benchmark_results/phase6/benchmark")
    sub.add_argument("--resume", action="store_true",
                     help="continue an interrupted run in --output-dir (rows.partial.jsonl)")


# -- commands --------------------------------------------------------------------------------------


def cmd_endurance_test(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.endurance import EnduranceConfig, EnduranceRunner
    from chest_hunter.runtime.checkpoint import CheckpointError

    settings = ctx.settings
    en = settings.endurance
    model = str(_model_dir(ctx, args.model)) if args.policy != "deterministic" else None
    config = EnduranceConfig(
        episodes=args.episodes or en.default_episodes, max_total_seconds=args.max_total_seconds,
        max_ticks_per_episode=args.max_ticks_per_episode or min(18000, en.max_ticks_per_episode),
        seed=args.seed, policy=args.policy, planner=planner_name(args.planner),
        supervisor=args.supervisor == "enabled", adaptive_compute=args.adaptive_compute,
        maps=tuple(args.map or ["research_arena_3d"]), width=args.observation_width,
        height=args.observation_height, backend=args.backend,
        preserve_estimates=args.preserve_estimates, allow_test_maps=args.allow_test_maps,
        checkpoint_every=args.checkpoint_every or en.save_checkpoint_every,
        max_reset_attempts=en.max_reset_attempts, model_dir=model)
    panel = None
    control = None
    on_decision = None
    on_episode = None
    if args.dashboard:
        from chest_hunter.agent.operations_panel import OperationsPanel
        from chest_hunter.runtime.operations import OperationsControl

        control = OperationsControl(start_paused=True)
        panel = OperationsPanel(control, title=f"Endurance run -> {args.output}")
        on_decision = panel.on_decision
        on_episode = panel.on_episode
    where = ("the built-in Python reference arena (NOT Unreal Engine)"
             if config.backend == "reference" else "the Unreal research server")
    print(f"Endurance test: {config.episodes} episode(s) on {', '.join(config.maps)} using "
          f"{where}; policy {config.policy}, planner {args.planner}, supervisor "
          f"{args.supervisor}, adaptive compute {'on' if config.adaptive_compute else 'off'}, "
          f"{config.width}x{config.height}, max {config.max_ticks_per_episode} ticks/episode, "
          f"output {args.output}" + (" (resuming)" if args.resume else ""))
    runner = EnduranceRunner(settings, config, args.output, resume=args.resume, control=control,
                             on_decision=on_decision, on_episode=on_episode, progress=print)
    if panel is not None and control is not None:
        panel.attach(runner)
        control.wait_while_paused(panel.pump)               # until Start (or Stop) is pressed
    try:
        summary = runner.run()
    except CheckpointError as exc:
        print(f"error: {exc}")
        return 2
    finally:
        if panel is not None:
            panel.close()
    from chest_hunter.analytics.endurance import format_endurance_report

    print(format_endurance_report(summary))
    print(f"Results written to {args.output}")
    return 0 if not summary["aggregate"].get("failed") else 1


def cmd_profile_agent(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.agent.runner_3d import Runner3D, _layout_hash
    from chest_hunter.analytics.policy_comparison import _memory_copy
    from chest_hunter.analytics.resource_monitor import process_rss_mb
    from chest_hunter.runtime.configurations import (
        custom_setup,
        make_planner,
        policy_options,
        supervisor_config,
    )
    from chest_hunter.runtime.profiler import Profiler, format_report

    settings = ctx.settings
    profiles = [p.strip() for p in args.vision_profiles.split(",") if p.strip()]
    from chest_hunter.agent.controller_3d import (
        VISION_PROFILES,
        Agent3DConfig,
        apply_vision_profile,
    )

    bad = [p for p in profiles if p not in VISION_PROFILES]
    if bad:
        print(f"error: unknown vision profile(s) {bad}; choose from {', '.join(VISION_PROFILES)}")
        return 2
    if not args.output and not args.report:
        print("error: give --output DIR and/or --report PATH")
        return 2
    policy = load_policy(ctx, args.model) if args.policy != "deterministic" else None
    active = args.exploration == "active"
    setup = custom_setup(policy=args.policy, planner="none" if active
                         else planner_name(args.planner),
                         supervisor=args.supervisor == "enabled", adaptive_compute=False)
    out = Path(args.output) if args.output else Path(args.report).parent
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    print(f"Profiling {args.episodes} episode(s) per vision profile on {args.map} (Python "
          "reference arena, NOT Unreal; one process, sequential) ...")
    for name in profiles:
        prof = Profiler()
        episodes = []
        for ep in range(args.episodes):
            map_id = args.map
            if map_id == "procedural":
                from chest_hunter.analytics.phase7_benchmark import evaluation_map

                map_id = evaluation_map("unseen_layout", ep)[1]
            db = _memory_copy(None, map_id)
            try:
                cfg = apply_vision_profile(
                    Agent3DConfig(decision_ticks=settings.simulation_3d.decision_ticks,
                                  smart_recovery=setup.smart_recovery), name,
                    skip_unchanged=args.skip_unchanged_frames)
                planner = None if active else make_planner(settings, setup, db, map_id,
                                                           _layout_hash(map_id))
                extra: dict[str, Any] = {}
                if active:
                    from chest_hunter.cli_phase7 import thresholds_from_settings
                    from chest_hunter.exploration.active import ExplorationConfig

                    extra = {"exploration": ExplorationConfig(),
                             "thresholds": thresholds_from_settings(settings)}
                runner = Runner3D.from_settings(
                    db, settings, mode="hybrid", seed=args.seed, episode=ep, backend="reference",
                    map_id=map_id, collect_truth=True, config=cfg, max_ticks=args.max_ticks,
                    width=args.observation_width, height=args.observation_height,
                    policy=args.policy, learned_policy=policy, planner=planner,
                    policy_options=policy_options(settings, setup) or None,
                    supervisor=supervisor_config(settings, setup), profiler=prof, **extra)
                res = runner.run()
            finally:
                db.close()
            minutes = res.sim_seconds / 60.0
            episodes.append({"episode": ep, "map_id": map_id, "outcome": res.outcome,
                             "rss_mb": round(process_rss_mb()[0], 1),
                             "decisions": res.metrics.decisions
                             if hasattr(res.metrics, "decisions") else None,
                             "coverage": res.truth.get("true_coverage"),
                             "sim_seconds": round(res.sim_seconds, 2),
                             "opened_truth": res.truth.get("opened_truth"),
                             "throughput_truth_per_min": round(
                                 (res.truth.get("opened_truth") or 0) / minutes, 4)
                             if minutes else None,
                             "frames_analysed": res.metrics.analyses,
                             "frames_skipped": res.metrics.frames_skipped_unchanged,
                             "cpu_seconds": round(res.cpu_seconds, 3),
                             "wall_seconds": round(res.wall_seconds, 3)})
            print(f"  [{name}] episode {ep + 1}/{args.episodes}: {res.outcome}, "
                  f"{episodes[-1]['opened_truth']} opened, {res.metrics.analyses} frames")
        report = prof.report()
        results[name] = {"components": report, "episodes": episodes}
        (out / f"profile_{name}.md").write_text(format_report(report) + "\n", encoding="utf-8")
        print(f"\nVision profile {name}:")
        print(format_report(report))
    if args.report:
        from chest_hunter.analytics.resource_monitor import resource_report

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(resource_report(
            results, map_id=args.map, seed=args.seed, policy=args.policy,
            exploration=args.exploration,
            observation=(args.observation_width, args.observation_height)), indent=1) + "\n",
            encoding="utf-8")
        print(f"Resource report written to {report_path}")
    (out / "profile.json").write_text(json.dumps({
        "map": args.map, "seed": args.seed, "episodes": args.episodes, "policy": args.policy,
        "planner": args.planner, "observation": [args.observation_width, args.observation_height],
        "environment": "Python reference arena; NOT Unreal Engine",
        "note": "latency = wall clock of each section; cpu_total_s = CPU time of the agent's own "
                "thread (the reference simulator runs in another thread of the same process); "
                "action_generation contains route_selection, which contains navigation_query",
        "profiles": results}, indent=1) + "\n", encoding="utf-8")
    print(f"Profile written to {out}")
    return 0


def cmd_benchmark_phase6(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.phase6_benchmark import format_phase6_report, run_phase6_benchmark
    from chest_hunter.runtime.configurations import DEFAULT_PHASE6_SETUPS

    setups = tuple(s.strip() for s in args.configurations.split(",")) if args.configurations \
        else DEFAULT_PHASE6_SETUPS
    exp = _experience_db(ctx, args.experience_db)
    print(f"Phase 6 benchmark: {len(setups)} configuration(s) x {args.episodes} paired "
          f"episode(s) on {', '.join(args.map or ['research_arena_3d'])} (Python reference "
          f"arena, NOT Unreal), seed {args.seed}, {args.workers} worker(s) ...")
    summary = run_phase6_benchmark(
        ctx.settings, setups=setups, map_ids=tuple(args.map or ["research_arena_3d"]),
        seed=args.seed, episodes=args.episodes, model_dir=_model_dir(ctx, args.model),
        experience_db=exp if exp.exists() else None, workers=args.workers,
        max_ticks=args.max_ticks, output_dir=args.output_dir, resume=args.resume,
        progress=print)
    print(format_phase6_report(summary))
    print(f"Results written to {args.output_dir}")
    return 0 if not summary["errors"] else 1


def cmd_calibrate_policy(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.learning.dataset import DemonstrationDataset
    from chest_hunter.learning.reliability import evaluate

    settings = ctx.settings
    policy = load_policy(ctx, args.model)
    root = Path(args.dataset) if args.dataset else \
        Path(settings.learning_path(settings.learning.dataset_dir)) / "synthetic_expert_v1"
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))["episodes"]
    data = DemonstrationDataset(root, check_images=False)
    validation, test = data.subset(splits["validation"]), data.subset(splits["test"])
    print(f"Calibrating {policy.version} on {len(validation)} validation and {len(test)} test "
          f"samples of {root.name} (synthetic demonstrations) ...")
    calibrator, report = evaluate(policy, validation, test, tolerance=args.tolerance)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    calibrator.save(out / "reliability.json")
    report.update({"model": policy.version, "dataset": root.name,
                   "note": "fitted on validation episodes, evaluated on test episodes; the "
                           "demonstrations are synthetic (scripted expert)"})
    (out / "reliability_report.json").write_text(json.dumps(report, indent=1) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0
