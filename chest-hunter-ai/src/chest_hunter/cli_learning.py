"""Phase 5 command-line commands (registered by ``chest_hunter.cli``)."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chest_hunter.exceptions import ValidationError

EXIT_OK = 0
EXIT_FAILURE = 1
BENCHMARK_SEED = 42
DEMO_SEED = 1000

AddCommand = Callable[[str, str, Callable[..., int]], argparse.ArgumentParser]


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


def register(add: AddCommand) -> None:
    sub = add("record-demo", "Record demonstrations: a person through the teleoperation window "
                             "(--controller human) or the deterministic controller "
                             "(--controller scripted, labelled synthetic).", cmd_record_demo)
    sub.add_argument("--controller", choices=("scripted", "human"), default="scripted")
    sub.add_argument("--map", action="append", metavar="MAP_ID",
                     help="map to record on (repeatable; default research_arena_3d)")
    sub.add_argument("--episodes", type=_positive, default=1, help="episodes per map")
    sub.add_argument("--seed", type=_non_negative, default=DEMO_SEED,
                     help=f"episode seed (default {DEMO_SEED}; the policy benchmark uses "
                          f"{BENCHMARK_SEED})")
    sub.add_argument("--output", metavar="DIR", help="dataset directory (default: "
                                                    "learning.dataset_dir/demo_01)")
    sub.add_argument("--experience-db", metavar="PATH",
                     help="database receiving the episodes' observations (scripted; default "
                          "learning.experience_db)")
    sub.add_argument("--workers", type=_positive, default=1)
    sub.add_argument("--max-ticks", type=_positive, metavar="N")
    sub.add_argument("--backend", choices=("reference", "unreal"), default="reference",
                     help="human mode: environment to control")
    sub.add_argument("--port", type=_positive, help="Unreal research server port (human mode)")

    sub = add("train-policy", "Train the compact behaviour-cloning policy on a demonstration "
                              "dataset and export it (ONNX FP32 + INT8).", cmd_train_policy)
    sub.add_argument("--dataset", required=True, metavar="DIR")
    sub.add_argument("--output", metavar="DIR", help="model directory (default "
                                                    "learning.model_dir)")
    sub.add_argument("--train-config", dest="train_config", metavar="JSON",
                     help="training configuration (default training/configs/bc_default.json)")
    sub.add_argument("--epochs", type=_positive)
    sub.add_argument("--batch-size", type=_positive)
    sub.add_argument("--learning-rate", type=float)
    sub.add_argument("--seed", type=_non_negative)
    sub.add_argument("--split-by", choices=("episode", "map"))
    sub.add_argument("--max-train-samples", type=_positive)
    sub.add_argument("--no-export", action="store_true", help="skip the ONNX export")

    sub = add("export-policy", "Export a trained policy to ONNX (FP32 and INT8), verify the "
                               "export and benchmark CPU latency.", cmd_export_policy)
    sub.add_argument("--model", metavar="DIR", help="model directory (default learning.model_dir)")
    sub.add_argument("--dataset", metavar="DIR", help="frames for calibration and checks "
                                                     "(default: the training dataset)")
    sub.add_argument("--no-int8", action="store_true")

    sub = add("evaluate-policy", "Evaluate a trained policy: held-out test metrics and paired "
                                 "episodes against the deterministic controller.",
              cmd_evaluate_policy)
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--episodes", type=_positive, default=20)
    sub.add_argument("--policy", choices=("learned", "hybrid"), default="learned")
    sub.add_argument("--map", action="append", metavar="MAP_ID")
    sub.add_argument("--seed", type=_non_negative, default=BENCHMARK_SEED)
    sub.add_argument("--workers", type=_positive, default=1)
    sub.add_argument("--max-ticks", type=_positive, default=18000)
    sub.add_argument("--quantized", action="store_true", help="use the INT8 model")
    sub.add_argument("--output-dir", metavar="DIR")

    sub = add("benchmark-policies", "Paired benchmark of control strategies (deterministic, "
                                    "learned, hybrid, hybrid with adaptive planning).",
              cmd_benchmark_policies)
    sub.add_argument("--episodes", type=_positive, default=100)
    sub.add_argument("--strategies", default="deterministic,learned,hybrid,hybrid_adaptive")
    sub.add_argument("--map", action="append", metavar="MAP_ID")
    sub.add_argument("--seed", type=_non_negative, default=BENCHMARK_SEED)
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--experience-db", metavar="PATH")
    sub.add_argument("--quantized", action="store_true")
    sub.add_argument("--workers", type=_positive, default=1)
    sub.add_argument("--max-ticks", type=_positive, default=18000)
    sub.add_argument("--online", action="store_true",
                     help="run each strategy's episodes in sequence, updating its history")
    sub.add_argument("--output-dir", metavar="DIR",
                     help="default benchmark_results/phase5/policies")

    sub = add("benchmark-generalization", "Familiar (training) vs unseen (test) procedural "
                                          "maps, with online adaptation.",
              cmd_benchmark_generalization)
    sub.add_argument("--familiar", type=_non_negative, default=4)
    sub.add_argument("--unseen", type=_non_negative, default=4)
    sub.add_argument("--episodes", type=_positive,
                     help="Phase 5: episodes per map (default 6); with --held-out-maps: "
                          "episodes per evaluation category and configuration (default 100)")
    sub.add_argument("--strategies", default="deterministic,learned,hybrid,hybrid_adaptive")
    sub.add_argument("--seed", type=_non_negative,
                     help="default 5000 (Phase 5) or 42 (--held-out-maps)")
    sub.add_argument("--model", metavar="DIR")
    sub.add_argument("--experience-db", metavar="PATH")
    sub.add_argument("--workers", type=_positive, default=1)
    sub.add_argument("--max-ticks", type=_positive,
                     help="default 18000 (Phase 5) or exploration.benchmark_max_ticks")
    sub.add_argument("--output-dir", "--output", dest="output_dir", metavar="DIR",
                     help="default benchmark_results/phase5/generalization, or "
                          "benchmark_results/phase7 with --held-out-maps")
    from chest_hunter.cli_phase7 import add_generalization_options

    add_generalization_options(sub)

    sub = add("generate-maps", "Generate procedural 3D maps (training or held-out test split), "
                               "add their spawn points to the database and optionally export "
                               "them for Unreal.", cmd_generate_maps)
    sub.add_argument("--split", choices=("train", "test"), required=True)
    sub.add_argument("--count", type=_positive, default=4)
    sub.add_argument("--start", type=_non_negative, default=0)
    sub.add_argument("--export-unreal", action="store_true",
                     help="write the layouts to unreal/ChestHunterResearch/Content/ResearchData")
    sub.add_argument("--export-dir", metavar="DIR",
                     help="write the layout JSON files here instead (implies --export-unreal)")

    sub = add("estimate-spawns", "Bayesian spawn-probability estimates from the recorded "
                                 "observations (configured probabilities are not changed).",
              cmd_estimate_spawns)
    sub.add_argument("--map", default="research_arena_3d")
    sub.add_argument("--prior", choices=("configured", "uniform"), default="configured")
    sub.add_argument("--prior-strength", type=float)
    sub.add_argument("--store", action="store_true", help="save to the spawn_estimates table")


def add_simulate_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--policy", choices=("deterministic", "learned", "hybrid"),
                     help="control policy (default learning.policy)")
    sub.add_argument("--model", metavar="DIR", help="trained policy (default learning.model_dir)")
    sub.add_argument("--quantized", action="store_true", help="use the INT8 ONNX model")
    sub.add_argument("--adaptive-planning", action="store_true",
                     help="adaptive target selection from spawn estimates and route memory "
                          "learned from this database's history")
    sub.add_argument("--smart-recovery", action="store_true",
                     help="classified, bounded recovery strategies")


# -- helpers -------------------------------------------------------------------------------------


def _model_dir(ctx: Any, value: str | None) -> Path:
    return Path(ctx.settings.learning_path(value or ctx.settings.learning.model_dir))


def _experience_db(ctx: Any, value: str | None) -> Path:
    return Path(ctx.settings.learning_path(value or ctx.settings.learning.experience_db))


def load_policy(ctx: Any, model: str | None, *, quantized: bool = False) -> Any:
    from chest_hunter.learning.learned_policy import LearnedPolicy, PolicyLoadError

    try:
        return LearnedPolicy(_model_dir(ctx, model), quantized=quantized,
                             threads=ctx.settings.learning.onnx_threads)
    except PolicyLoadError as exc:
        raise ValidationError(str(exc)) from exc


def simulate_extras(ctx: Any, args: argparse.Namespace, map_id: str) -> dict[str, Any]:
    """Runner3D keyword arguments for the Phase 5 options of ``simulate-3d``."""
    from chest_hunter.agent.controller_3d import Agent3DConfig
    from chest_hunter.learning.policy_selector import SafetyLimits

    learning = ctx.settings.learning
    policy = args.policy or learning.policy
    extras: dict[str, Any] = {"policy": policy}
    if policy != "deterministic" or args.model:
        extras["learned_policy"] = load_policy(ctx, args.model,
                                               quantized=args.quantized or learning.quantized_model)
        extras["policy_options"] = {"limits": SafetyLimits(
            timeout_ms=learning.inference_timeout_ms, confidence=learning.hybrid_confidence,
            learned_observation_rate=learning.learned_observation_rate)}
    if args.adaptive_planning or learning.adaptive_planning:
        from chest_hunter.agent.runner_3d import _layout_hash
        from chest_hunter.exploration.adaptive_planner import AdaptivePlanner
        from chest_hunter.exploration.frontier_scoring import ScoringWeights

        extras["planner"] = AdaptivePlanner.from_database(
            ctx.db, map_id, _layout_hash(map_id), prior_strength=learning.spawn_prior_strength,
            weights=ScoringWeights(exploration=learning.exploration_weight,
                                   cluster=learning.cluster_weight))
    smart = args.smart_recovery or learning.smart_recovery
    extras["config"] = Agent3DConfig(decision_ticks=ctx.settings.simulation_3d.decision_ticks,
                                     smart_recovery=smart)
    return extras


# -- commands --------------------------------------------------------------------------------


def cmd_record_demo(args: argparse.Namespace, ctx: Any) -> int:
    maps = args.map or ["research_arena_3d"]
    output = ctx.settings.learning_path(args.output) if args.output else \
        ctx.settings.learning_path(ctx.settings.learning.dataset_dir) / "demo_01"
    if args.seed == BENCHMARK_SEED:
        print(f"warning: seed {BENCHMARK_SEED} is the policy benchmark seed; episodes recorded "
              "with it must not be used for training.")
    if args.controller == "human":
        return _record_human(args, ctx, maps[0], output)
    from chest_hunter.learning.demo_generation import record_scripted_demos

    db_path = _experience_db(ctx, args.experience_db)
    print(f"Recording {args.episodes} SYNTHETIC demonstration episode(s) per map with the "
          f"deterministic Phase 4 controller on {', '.join(maps)} (seed {args.seed}) into "
          f"{output}; observations go to {db_path}.")
    started = time.perf_counter()
    results = record_scripted_demos(output, db_path, ctx.settings, maps=maps,
                                    episodes_per_map=args.episodes, seed=args.seed,
                                    workers=args.workers, max_ticks=args.max_ticks,
                                    progress=lambda r: print(f"  episode {r['episode_id']} on "
                                                             f"{r['map_id']}: {r['outcome']}, "
                                                             f"{r['chests_opened']} opened, "
                                                             f"{r['sim_seconds']} s"))
    mb = sum(r["bytes"] for r in results) / 1e6
    print(f"Recorded {len(results)} episode(s), {mb:.1f} MB of frames, in "
          f"{time.perf_counter() - started:.0f} s.")
    return EXIT_OK


def _record_human(args: argparse.Namespace, ctx: Any, map_id: str, output: Path) -> int:
    from chest_hunter.learning.demonstration_recorder import DemonstrationRecorder, next_episode_id
    from chest_hunter.learning.teleop import TeleopSession
    from chest_hunter.unreal_bridge.client import BridgeClient

    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        raise ValidationError("human demonstrations need Tkinter and a display") from exc
    server = None
    port = args.port or ctx.settings.unreal_bridge.port
    if args.backend == "reference":
        from chest_hunter.unreal_bridge.mock_server import MockUnrealServer

        server = MockUnrealServer(port=0).start()
        port = server.port
    output.mkdir(parents=True, exist_ok=True)
    recorder = DemonstrationRecorder(output)
    episode_id = next_episode_id(output)
    try:
        with BridgeClient(ctx.settings.unreal_bridge.host, port,
                          connect_attempts=ctx.settings.unreal_bridge.connect_attempts) as client:
            print("Teleoperation: W/S/A/D move, arrows look, E interact (hold), Space jump, "
                  "P pause, Esc stop and save. Keys act on the recording window only.")
            session = TeleopSession(client, recorder, episode_id=episode_id, map_id=map_id,
                                    seed=args.seed, max_ticks=args.max_ticks or 18000)
            try:
                session.run()
            finally:
                session.close()
    finally:
        if server is not None:
            server.stop()
    print(f"Saved human demonstration episode {episode_id} to {output}")
    return EXIT_OK


def cmd_train_policy(args: argparse.Namespace, ctx: Any) -> int:
    from dataclasses import replace

    from chest_hunter.learning.behavior_cloning import TrainConfig, train_policy

    cfg_path = Path(args.train_config) if args.train_config else \
        ctx.settings.base_dir / "training" / "configs" / "bc_default.json"
    cfg = TrainConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8"))) \
        if cfg_path.exists() else TrainConfig()
    overrides = {k: v for k, v in (("epochs", args.epochs), ("batch_size", args.batch_size),
                                   ("learning_rate", args.learning_rate), ("seed", args.seed),
                                   ("split_by", args.split_by),
                                   ("max_train_samples", args.max_train_samples))
                 if v is not None}
    cfg = replace(cfg, **overrides)
    dataset = ctx.settings.learning_path(args.dataset)
    output = _model_dir(ctx, args.output)
    print(f"Training on {dataset} -> {output} ({cfg.epochs} epochs max, batch {cfg.batch_size}, "
          f"lr {cfg.learning_rate}, split by {cfg.split_by}, seed {cfg.seed})")
    result = train_policy(dataset, output, cfg, progress=lambda e: print(
        f"  epoch {e['epoch']:3d}: train {e['train_loss']:.4f}  validation "
        f"{e['validation_loss']:.4f}  ({e['seconds']:.0f} s)"))
    meta = result.metadata
    if meta["dataset"].get("synthetic_only"):
        print("Note: trained on SYNTHETIC demonstrations only (scripted controller, not human).")
    print(f"Best epoch {result.best_epoch} of {result.epochs_run}"
          f"{' (early stop)' if result.stopped_early else ''}; test metrics: "
          f"{json.dumps(meta['metrics']['test'])}")
    if not args.no_export:
        return cmd_export_policy(argparse.Namespace(model=str(output), dataset=str(dataset),
                                                    no_int8=False), ctx)
    return EXIT_OK


def cmd_export_policy(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.learning.dataset import DemonstrationDataset
    from chest_hunter.learning.model_export import ExportError, export_policy

    model = _model_dir(ctx, args.model)
    meta = json.loads((model / "policy.json").read_text(encoding="utf-8"))
    dataset_dir = Path(args.dataset) if args.dataset else Path(meta["dataset"]["root"])
    ds = DemonstrationDataset(ctx.settings.learning_path(str(dataset_dir)))
    splits = meta["dataset"]["split_episodes"]
    try:
        report = export_policy(model, ds.subset(splits["test"]), ds.subset(splits["train"]),
                               int8=not args.no_int8)
    except ExportError as exc:
        raise ValidationError(str(exc)) from exc
    print(f"Exported {model / 'policy.onnx'}" + ("" if args.no_int8 else " and policy_int8.onnx"))
    print(json.dumps(report, indent=1))
    return EXIT_OK


def cmd_evaluate_policy(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.learning_metrics import training_report
    from chest_hunter.analytics.policy_comparison import format_report, run_policy_benchmark

    model = _model_dir(ctx, args.model)
    load_policy(ctx, str(model), quantized=args.quantized)          # validates the files
    report = training_report(model)
    print("Held-out test split (episodes never used for training or model selection):")
    print(json.dumps({"test": report["test"], "baseline_test": report["baseline_test"],
                      "mae_reduction_vs_mean_baseline": report["mae_reduction_vs_mean_baseline"]},
                     indent=1))
    out = Path(args.output_dir) if args.output_dir else \
        ctx.settings.base_dir / "benchmark_results" / "phase5" / f"evaluate_{args.policy}"
    summary = run_policy_benchmark(ctx.settings, strategies=("deterministic", args.policy),
                                   map_ids=args.map or ["research_arena_3d"], seed=args.seed,
                                   episodes=args.episodes, model_dir=model,
                                   quantized=args.quantized, workers=args.workers,
                                   max_ticks=args.max_ticks, output_dir=out)
    print(format_report(summary))
    print(f"Results written to {out}")
    return EXIT_OK


def _strategies(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def cmd_benchmark_policies(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.analytics.policy_comparison import (
        STRATEGIES,
        format_report,
        run_policy_benchmark,
    )

    strategies = _strategies(args.strategies)
    needs_model = any(STRATEGIES[s].policy != "deterministic" for s in strategies
                      if s in STRATEGIES)
    model = _model_dir(ctx, args.model) if needs_model else None
    if model is not None:
        load_policy(ctx, str(model), quantized=args.quantized)
    out = Path(args.output_dir) if args.output_dir else \
        ctx.settings.base_dir / "benchmark_results" / "phase5" / "policies"
    print(f"Policy benchmark: {args.episodes} paired episode(s) x {', '.join(strategies)} on "
          f"{', '.join(args.map or ['research_arena_3d'])}, seed {args.seed}, Python reference "
          "arena (NOT Unreal Engine).")
    summary = run_policy_benchmark(
        ctx.settings, strategies=strategies, map_ids=args.map or ["research_arena_3d"],
        seed=args.seed, episodes=args.episodes, model_dir=model,
        experience_db=_experience_db(ctx, args.experience_db), quantized=args.quantized,
        workers=args.workers, max_ticks=args.max_ticks, online=args.online, output_dir=out,
        progress=print)
    print(format_report(summary))
    print(f"Results written to {out}")
    return EXIT_OK if not summary["errors"] else EXIT_FAILURE


def cmd_benchmark_generalization(args: argparse.Namespace, ctx: Any) -> int:
    if getattr(args, "held_out_maps", False):
        from chest_hunter.cli_phase7 import cmd_generalization_p7

        return cmd_generalization_p7(args, ctx)
    from chest_hunter.analytics.generalization_benchmark import run_generalization_benchmark
    from chest_hunter.analytics.policy_comparison import format_report

    out = Path(args.output_dir) if args.output_dir else \
        ctx.settings.base_dir / "benchmark_results" / "phase5" / "generalization"
    summary = run_generalization_benchmark(
        ctx.settings, familiar=args.familiar, unseen=args.unseen, episodes=args.episodes or 6,
        strategies=_strategies(args.strategies), model_dir=_model_dir(ctx, args.model),
        experience_db=_experience_db(ctx, args.experience_db), workers=args.workers,
        max_ticks=args.max_ticks or 18000, output_dir=out,
        seed=5000 if args.seed is None else args.seed, progress=print)
    for group, result in summary["results"].items():
        print(f"\n{group} maps: {', '.join(summary['groups'][group])}")
        print(format_report(result))
    print("\nUnseen minus familiar (per simulated minute; truth = chests really opened, "
          "throughput_per_min = as counted by the agent): "
          + json.dumps(summary["gap_unseen_minus_familiar"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return EXIT_OK


def cmd_generate_maps(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.scenarios import seed_procedural_map
    from chest_hunter.simulation3d.arena import arena_to_json
    from chest_hunter.simulation3d.procedural import build_procedural, layout_hash, map_id

    target = Path(args.export_dir) if args.export_dir else \
        ctx.settings.base_dir / "unreal" / "ChestHunterResearch" / "Content" / "ResearchData"
    for i in range(args.start, args.start + args.count):
        mid = map_id(args.split, i)
        arena, _world, _nav = build_procedural(mid)
        seeded = seed_procedural_map(ctx.db, mid)
        reachable = sum(s.reachable for s in arena.spawns)
        print(f"{mid}: {len(arena.boxes)} boxes, {len(arena.spawns)} spawn points "
              f"({reachable} reachable), {len(arena.regions)} regions, start {arena.start}, "
              f"layout {layout_hash(arena)}, {seeded.inserted} new database location(s)")
        if args.export_unreal or args.export_dir:
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{mid}.json").write_text(arena_to_json(arena) + "\n", encoding="utf-8")
    return EXIT_OK


def cmd_estimate_spawns(args: argparse.Namespace, ctx: Any) -> int:
    from chest_hunter.agent.runner_3d import _layout_hash
    from chest_hunter.exploration.spawn_estimator import SpawnEstimator

    strength = args.prior_strength or ctx.settings.learning.spawn_prior_strength
    estimator = SpawnEstimator(ctx.db, prior=args.prior, prior_strength=strength)
    layout = _layout_hash(args.map)
    estimates = estimator.estimate(args.map, layout)
    if not estimates:
        raise ValidationError(f"map {args.map!r} has no chest locations in this database")
    print(f"{'Location':45s} {'config':>6s} {'pres':>4s} {'abs':>4s} {'mean':>6s}  95 % interval")
    for e in estimates.values():
        lo, hi = e.credible_interval()
        print(f"{e.location_name[:45]:45s} {e.configured_probability:6.2f} {e.present:4d} "
              f"{e.absent:4d} {e.mean:6.3f}  [{lo:.3f}, {hi:.3f}]")
    if args.store:
        n = estimator.store(estimates, layout)
        print(f"Stored {n} estimate(s) in spawn_estimates (configured probabilities unchanged).")
    return EXIT_OK
