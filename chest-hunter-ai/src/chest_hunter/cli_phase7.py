"""Phase 7 command-line commands and options.

* ``simulate-3d --exploration active [--map procedural]`` - active exploration on a generated
  (by default held-out) map, optionally with ``--inject-fault``;
* ``benchmark-generalization --held-out-maps`` - the Phase 7 generalisation benchmark (without
  ``--held-out-maps`` the command keeps its Phase 5 behaviour);
* ``benchmark-robustness --failure-injection``, ``benchmark-exploration`` (fixed strategies vs
  the bandit), ``train-exploration`` (exploration memory, training maps and seeds only);
* ``check-runtime`` - deployment startup checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chest_hunter.cli_learning import AddCommand, _model_dir


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


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip().replace("-", "_") for v in value.split(",") if v.strip())


def _progress(quiet: bool) -> Any:
    def progress(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr, flush=True)

    return progress


# -- simulate-3d --------------------------------------------------------------------------------


def add_phase7_simulate_options(sub: argparse.ArgumentParser) -> None:
    from chest_hunter.exploration.active import STRATEGIES
    from chest_hunter.runtime.faults import FAULT_KINDS
    from chest_hunter.simulation3d.generator7 import CATEGORIES

    sub.add_argument("--exploration", choices=("spawn-points", "active"),
                     help="spawn-points: Phase 4-6 targets from the map's spawn points; active: "
                          "Phase 7 frontier exploration without spawn points (default: "
                          "exploration.mode)")
    sub.add_argument("--exploration-strategy", choices=("memory", *STRATEGIES),
                     help="memory = the trained exploration memory's choice for the map family "
                          "(default: exploration.strategy)")
    sub.add_argument("--exploration-memory", metavar="PATH",
                     help="exploration memory (default: exploration.memory_path); read only")
    sub.add_argument("--map-category", choices=CATEGORIES,
                     help="with --map procedural: map family (default: chosen by the seed)")
    sub.add_argument("--map-split", choices=("test", "train"), default="test",
                     help="with --map procedural: held-out test layouts (default) or training "
                          "layouts")
    sub.add_argument("--inject-fault", choices=FAULT_KINDS, metavar="KIND",
                     help="robustness test: inject one controlled failure "
                          f"({', '.join(FAULT_KINDS)})")
    sub.add_argument("--fault-seed", type=_non_negative, default=0, metavar="N")


def resolve_map(args: argparse.Namespace, seed: int) -> str | None:
    """``--map procedural``: a generated map chosen deterministically by the seed."""
    from chest_hunter.simulation3d.generator7 import CATEGORIES, map_id

    if getattr(args, "map", None) != "procedural":
        return getattr(args, "map", None)
    family = args.map_category or CATEGORIES[seed % len(CATEGORIES)]
    return map_id(family, args.map_split, seed % 1000)


def thresholds_from_settings(settings: Any) -> Any:
    from chest_hunter.exploration.active import ConfidenceThresholds

    e = settings.exploration
    return ConfidenceThresholds(approach_confidence=e.approach_confidence,
                                approach_hits=e.approach_hits,
                                inspect_confidence=e.inspect_confidence)


def phase7_extras(ctx: Any, args: argparse.Namespace, map_id: str,
                  extras: dict[str, Any]) -> dict[str, Any]:
    from chest_hunter.analytics.phase7_benchmark import DEFAULT_STRATEGY, _memory, memory_strategy
    from chest_hunter.exceptions import ValidationError
    from chest_hunter.exploration.active import ActiveExplorer, ExplorationConfig
    from chest_hunter.simulation3d.generator7 import is_generated, parse

    settings = ctx.settings
    out = dict(extras)
    mode = (args.exploration or settings.exploration.mode).replace("-", "_")
    if mode == "active":
        if out.get("planner") is not None:
            raise ValidationError("--exploration active chooses its own targets; it cannot be "
                                  "combined with a spawn-point planner (--planner / "
                                  "--adaptive-planning)")
        path = Path(args.exploration_memory) if args.exploration_memory else \
            settings.exploration_memory_path()
        memory = _memory(str(path)) if path.exists() else None
        family = parse(map_id).category if is_generated(map_id) else None
        choice = args.exploration_strategy or settings.exploration.strategy
        if choice == "memory":
            strategy, why = memory_strategy(memory, family) if family else (
                DEFAULT_STRATEGY, "not a generated map")
        else:
            strategy, why = choice, "chosen on the command line / in the configuration"
        rates = memory.discovery_rates(family) if memory is not None and family \
            and memory.categories.get(family) else None
        cfg = ExplorationConfig(strategy=strategy)
        out["exploration"] = lambda: ActiveExplorer(cfg, discovery_rates=rates)
        out["thresholds"] = thresholds_from_settings(settings)
        print(f"Active exploration: strategy {strategy} ({why}); memory "
              f"{path if memory is not None else 'none'}")
    if getattr(args, "inject_fault", None):
        from chest_hunter.runtime.faults import FaultInjector, FaultPlan

        plan = FaultPlan.random(args.inject_fault, args.fault_seed)
        out["faults"] = FaultInjector(plan, seed=args.fault_seed)
        print(f"Fault injection: {plan.kind} at {plan.start_s:.1f} s simulated "
              f"(duration {plan.duration_s:.0f} s, intensity {plan.intensity})")
    return out


def print_phase7_result(runner: Any, result: Any) -> None:
    from chest_hunter.cli import format_key_values

    rows = []
    if runner.explorer is not None:
        s = runner.explorer.summary()
        rows += [("Exploration", f"strategy {s['strategy']}, {s['selections']} frontier "
                                 f"selection(s), {s['scans_planned']} camera scan(s), "
                                 f"{s['seen_cells']} of {s['walkable_cells']} known walkable "
                                 f"cells seen, done={s['done']}"),
                 ("Frontier selection", f"{s.get('latency_mean_ms')} ms mean, "
                                        f"{s.get('latency_p95_ms')} ms p95")]
        cov = (result.truth or {}).get("true_coverage")
        if cov is not None:
            rows.append(("Coverage of reachable area", f"{100 * cov:.1f} % (evaluation only)"))
    if runner.faults is not None:
        a = runner.faults.assess(result)
        rows.append(("Injected fault", json.dumps({k: a.get(k) for k in (
            "kind", "injected", "applied", "start_s", "detected", "recovered",
            "terminated_cleanly")})))
    if rows:
        print(format_key_values(rows))


# -- commands --------------------------------------------------------------------------------


def register(add: AddCommand) -> None:
    from chest_hunter.analytics.phase7_benchmark import PHASE7_SETUPS
    from chest_hunter.runtime.faults import FAULT_KINDS

    sub = add("benchmark-robustness", "Inject controlled failures into research episodes on "
                                      "held-out maps and measure detection and recovery per "
                                      "failure category (paired with fault-free baselines).",
              cmd_benchmark_robustness)
    sub.add_argument("--episodes", type=_positive, default=100,
                     help="faulted episodes (the fault kinds are played in rotation)")
    sub.add_argument("--failure-injection", action="store_true",
                     help="required: confirms that failures are injected on purpose")
    sub.add_argument("--faults", type=_csv, default=FAULT_KINDS, metavar="LIST",
                     help=f"comma-separated subset of {','.join(FAULT_KINDS)}")
    sub.add_argument("--configuration", choices=tuple(PHASE7_SETUPS), default="phase7_active")
    sub.add_argument("--no-baseline", dest="baseline", action="store_false",
                     help="skip the paired fault-free episodes")
    _common_benchmark(sub, "benchmark_results/robustness", max_ticks=4500)

    sub = add("benchmark-exploration", "Fixed exploration strategies against the "
                                       "bandit-selected strategy on the same held-out episodes.",
              cmd_benchmark_exploration)
    sub.add_argument("--episodes", type=_positive, default=20)
    sub.add_argument("--categories", type=_csv, default=("unseen_layout",), metavar="LIST")
    _common_benchmark(sub, "benchmark_results/phase7/strategies")

    sub = add("train-exploration", "Train the exploration memory (category statistics and the "
                                   "strategy bandit) on training maps with training seeds.",
              cmd_train_exploration)
    sub.add_argument("--episodes", type=_positive, default=80)
    sub.add_argument("--seed", type=_non_negative, default=1,
                     help="training seed (must be below the evaluation seed range)")
    sub.add_argument("--start-episode", type=_non_negative, default=0)
    sub.add_argument("--exploration-memory", metavar="PATH",
                     help="memory file to create or extend (default: exploration.memory_path)")
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--workers", type=_positive, default=1)
    sub.add_argument("--max-ticks", type=_positive)
    sub.add_argument("--quiet", action="store_true")

    sub = add("acceptance-phase7", "End-to-end acceptance on a held-out procedural map: the "
                                   "Phase 7 and Phase 6 agents, with and without injected "
                                   "failures, plus a reproducibility re-run.",
              cmd_acceptance)
    sub.add_argument("--episodes", type=_positive, default=10)
    sub.add_argument("--map", metavar="MAP_ID", help="a held-out map (default "
                     "p7_complex_test_000_unfamiliar)")
    _common_benchmark(sub, "benchmark_results/phase7/acceptance", max_ticks=18000)

    sub = add("check-runtime", "Startup checks for a deployment: configuration, database schema, "
                               "model compatibility, observation size, bridge protocol and "
                               "environment version, output directories.", cmd_check_runtime)
    sub.add_argument("--backend", choices=("reference", "unreal"), default="reference",
                     help="reference: start the built-in arena and handshake with it; unreal: "
                          "connect to the research server on unreal_bridge.host/port")
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--json", action="store_true", help="machine-readable output")


def _common_benchmark(sub: argparse.ArgumentParser, output: str, *,
                      max_ticks: int | None = None) -> None:
    sub.add_argument("--seed", type=_non_negative, default=42,
                     help="benchmark seed (moved into the evaluation seed range)")
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--exploration-memory", metavar="PATH")
    sub.add_argument("--workers", type=_positive, default=1)
    sub.add_argument("--max-ticks", type=_positive, default=max_ticks,
                     help="default: exploration.benchmark_max_ticks")
    sub.add_argument("--output", "--output-dir", dest="output", default=output, metavar="DIR")
    sub.add_argument("--quiet", action="store_true")


def add_generalization_options(sub: argparse.ArgumentParser) -> None:
    """Phase 7 options of the (Phase 5) ``benchmark-generalization`` command."""
    sub.add_argument("--held-out-maps", action="store_true",
                     help="Phase 7: evaluate on held-out procedural maps in the eight "
                          "evaluation categories (familiar, obstacles, unseen_layout, lighting, "
                          "appearance, start, elevation, combined)")
    sub.add_argument("--categories", type=_csv, metavar="LIST",
                     help="Phase 7: subset of the evaluation categories (must include familiar)")
    sub.add_argument("--configurations", type=_csv, metavar="LIST",
                     help="Phase 7: default phase7_active,phase6_priors")
    sub.add_argument("--exploration-memory", metavar="PATH")
    sub.add_argument("--start-episode", type=_non_negative, default=0)
    sub.add_argument("--quiet", action="store_true")


def _memory_arg(ctx: Any, value: str | None) -> Path | None:
    path = Path(value) if value else ctx.settings.exploration_memory_path()
    return path if path.exists() else None


def cmd_generalization_p7(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.phase7_benchmark import (
        DEFAULT_GENERALIZATION_SETUPS,
        EVAL_CATEGORIES,
        run_generalization_benchmark,
    )

    s = ctx.settings
    e = s.exploration
    out = Path(args.output_dir) if args.output_dir else s.base_dir / "benchmark_results/phase7"
    memory = _memory_arg(ctx, args.exploration_memory)
    setups = args.configurations or DEFAULT_GENERALIZATION_SETUPS
    cats = args.categories or tuple(EVAL_CATEGORIES)
    episodes = args.episodes or 100
    print(f"Phase 7 generalisation benchmark: {episodes} episode(s) x {len(cats)} categories x "
          f"{len(setups)} configuration(s), {args.workers} worker(s), Python reference arena "
          f"(NOT Unreal); exploration memory {memory or 'none'}.")
    summary = run_generalization_benchmark(
        s, setups=setups, categories=cats, episodes=episodes,
        seed=42 if args.seed is None else args.seed, model_dir=_model_dir(ctx, args.model),
        memory_path=memory, workers=args.workers, max_ticks=args.max_ticks or e.benchmark_max_ticks,
        width=e.benchmark_observation_width, height=e.benchmark_observation_height,
        start_episode=args.start_episode, output_dir=out, progress=_progress(args.quiet))
    print((out / "report.md").read_text(encoding="utf-8"))
    if summary["errors"]:
        print(f"{len(summary['errors'])} episode(s) failed with an exception; see summary.json")
    print(f"Results written to {out}/episodes.csv, summary.json, report.md")
    return 0 if not summary["errors"] else 1


def cmd_benchmark_robustness(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.robustness import run_robustness_benchmark

    if not args.failure_injection:
        print("error: benchmark-robustness injects failures on purpose; pass "
              "--failure-injection to confirm", file=sys.stderr)
        return 2
    s = ctx.settings
    out = Path(args.output)
    memory = _memory_arg(ctx, args.exploration_memory)
    print(f"Robustness benchmark: {args.episodes} faulted episode(s) ({', '.join(args.faults)}), "
          f"{'with' if args.baseline else 'without'} paired baselines, {args.workers} "
          "worker(s), Python reference arena (NOT Unreal).")
    summary = run_robustness_benchmark(
        s, setup=args.configuration, kinds=args.faults, episodes=args.episodes, seed=args.seed,
        model_dir=_model_dir(ctx, args.model), memory_path=memory, workers=args.workers,
        max_ticks=args.max_ticks or 4500, width=s.exploration.benchmark_observation_width,
        height=s.exploration.benchmark_observation_height, baseline=args.baseline,
        output_dir=out, progress=_progress(args.quiet))
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"Results written to {out}")
    return 0 if not summary["errors"] else 1


def cmd_benchmark_exploration(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.phase7_benchmark import run_strategy_comparison

    s = ctx.settings
    out = Path(args.output)
    memory = _memory_arg(ctx, args.exploration_memory)
    summary = run_strategy_comparison(
        s, categories=args.categories, episodes=args.episodes, seed=args.seed,
        model_dir=_model_dir(ctx, args.model), memory_path=memory, workers=args.workers,
        max_ticks=args.max_ticks or s.exploration.benchmark_max_ticks,
        width=s.exploration.benchmark_observation_width,
        height=s.exploration.benchmark_observation_height, output_dir=out,
        progress=_progress(args.quiet))
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0 if not summary["errors"] else 1


def cmd_train_exploration(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.phase7_benchmark import train_exploration_memory

    s = ctx.settings
    path = Path(args.exploration_memory) if args.exploration_memory else \
        s.exploration_memory_path()
    print(f"Training the exploration memory {path} on {args.episodes} training episode(s) "
          f"(training maps, seed {args.seed}; held-out maps and evaluation seeds are refused).")
    result = train_exploration_memory(
        s, episodes=args.episodes, seed=args.seed, start_episode=args.start_episode,
        model_dir=_model_dir(ctx, args.model), memory_path=path, workers=args.workers,
        max_ticks=args.max_ticks or s.exploration.benchmark_max_ticks,
        width=s.exploration.benchmark_observation_width,
        height=s.exploration.benchmark_observation_height, progress=_progress(args.quiet))
    rows = result.pop("rows")
    log = path.with_suffix(".training.jsonl")
    with log.open("a", encoding="utf-8") as handle:
        for r in rows:
            handle.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=1, default=str))
    print(f"Memory saved to {path}; per-episode training log appended to {log}")
    return 0 if not result["errors"] else 1


def cmd_acceptance(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.phase7_benchmark import ACCEPTANCE_MAP, run_acceptance

    s = ctx.settings
    out = Path(args.output)
    summary = run_acceptance(
        s, episodes=args.episodes, seed=args.seed, map_id=args.map or ACCEPTANCE_MAP,
        model_dir=_model_dir(ctx, args.model),
        memory_path=_memory_arg(ctx, args.exploration_memory), workers=args.workers,
        max_ticks=args.max_ticks or 18000, width=s.exploration.benchmark_observation_width,
        height=s.exploration.benchmark_observation_height, output_dir=out,
        progress=_progress(args.quiet))
    print((out / "report.md").read_text(encoding="utf-8"))
    return 0 if not summary["errors"] else 1


def cmd_check_runtime(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.runtime.startup import format_checks, run_startup_checks

    checks = run_startup_checks(ctx.settings, backend=args.backend,
                                model_dir=_model_dir(ctx, args.model))
    if args.json:
        print(json.dumps([c.to_dict() for c in checks], indent=1))
    else:
        print(format_checks(checks))
    return 0 if all(c.ok for c in checks) else 1
