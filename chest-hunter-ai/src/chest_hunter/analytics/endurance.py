"""Long-duration autonomous testing (Phase 6).

``EnduranceRunner`` plays many consecutive episodes in one process against **one** research
server and **one** bridge connection, the way a persistent Unreal research environment would
be used: between episodes the environment is reset (``RESET_EPISODE``) and the agent is
rebuilt from scratch, so no target, track, route or interaction state can leak from one
episode into the next. Estimates learned from earlier episodes (spawn probabilities, route
statistics) are carried over only with ``preserve_estimates``, in the run's own database and
within one experience partition; held-out test maps are refused unless explicitly allowed.
Without it the planner starts every episode without history.

Per episode it writes one JSON line (``episodes.jsonl``, fsync-ed) with the outcome,
throughput (ground truth via the evaluation channel, used for scoring only), supervisor
interventions, recovery results, CPU time, the *current* resident memory, live Python
objects, threads and open file descriptors, and whether the previous episode's agent was
released. Every ``checkpoint_every`` episodes, and at the end, it writes an atomic checkpoint
(``runtime.checkpoint``); ``resume`` continues after the last finished episode.

Failures: an episode that raises, loses the connection or is aborted by the supervisor is
logged as ``failed``. Before the next episode the environment must answer a ping and accept
a reset (the reference server is restarted if its thread died); after ``max_reset_attempts``
unsuccessful resets the run stops with that reason instead of continuing on a broken
environment. ``SIGINT``/``SIGTERM`` or the operations panel stop the run gracefully: the
current episode ends as ``cancelled`` (it is played again on resume), results and the
checkpoint are written.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import os
import signal
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chest_hunter.analytics.resource_monitor import ResourceMonitor, process_rss_mb
from chest_hunter.analytics.stats import summarise
from chest_hunter.config import Settings
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ValidationError
from chest_hunter.runtime.checkpoint import (
    Checkpoint,
    CheckpointError,
    append_result,
    check_compatible,
    load_checkpoint,
    read_results,
    rewrite_results,
    save_checkpoint,
)
from chest_hunter.runtime.configurations import (
    AGENT_VERSION,
    agent_config,
    custom_setup,
    guard_experience_write,
    make_planner,
    partition,
    policy_options,
    supervisor_config,
)
from chest_hunter.runtime.operations import OperationsControl

logger = logging.getLogger(__name__)

DONE_OUTCOMES = ("completed", "tick_limit", "failed")


@dataclass(frozen=True, slots=True)
class EnduranceConfig:
    episodes: int = 100
    max_total_seconds: float | None = None        # wall clock budget for the whole run
    max_ticks_per_episode: int = 18000
    seed: int = 42
    policy: str = "deterministic"
    planner: str = "short_horizon"
    supervisor: bool = True
    adaptive_compute: bool = True
    maps: tuple[str, ...] = ("research_arena_3d",)
    width: int = 320
    height: int = 180
    backend: str = "reference"
    preserve_estimates: bool = False
    allow_test_maps: bool = False
    checkpoint_every: int = 10
    max_reset_attempts: int = 3
    model_dir: str | None = None

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValidationError("episodes must be at least 1")
        if self.max_total_seconds is not None and self.max_total_seconds <= 0:
            raise ValidationError("max_total_seconds must be positive")
        if self.max_ticks_per_episode < 100:
            raise ValidationError("max_ticks_per_episode must be at least 100")
        if not self.maps:
            raise ValidationError("at least one map is needed")
        if self.checkpoint_every < 1 or self.max_reset_attempts < 1:
            raise ValidationError("checkpoint_every and max_reset_attempts must be >= 1")
        if self.backend not in ("reference", "unreal"):
            raise ValidationError("backend must be 'reference' or 'unreal'")
        if self.policy != "deterministic" and not self.model_dir:
            raise ValidationError(f"policy {self.policy!r} needs a model directory")
        if self.preserve_estimates:
            for m in self.maps:
                guard_experience_write(m, allow_test_maps=self.allow_test_maps)

    def checkpoint_config(self) -> dict[str, Any]:
        """Everything that must match for a resume (not the run's output location)."""
        out = asdict(self)
        out["maps"] = list(self.maps)
        out.pop("max_total_seconds")                 # a resumed run may get a new budget
        out.pop("episodes")                          # ... or more episodes
        return out


class _ResetFailed(RuntimeError):
    pass


class EnduranceRunner:
    def __init__(self, settings: Settings, config: EnduranceConfig, output_dir: str | Path, *,
                 resume: bool = False, control: OperationsControl | None = None,
                 on_episode: Callable[[dict[str, Any]], None] | None = None,
                 on_decision: Callable[[Any, Any], None] | None = None,
                 install_signal_handlers: bool = True,
                 progress: Callable[[str], None] | None = None) -> None:
        self.settings = settings
        self.config = config
        self.out = Path(output_dir)
        self.resume = resume
        self.control = control or OperationsControl()
        self.on_episode = on_episode
        self.on_decision = on_decision
        self.install_signal_handlers = install_signal_handlers
        self.progress = progress or (lambda _m: None)
        self.paths = {"checkpoint": self.out / "checkpoint.json",
                      "results": self.out / "episodes.jsonl",
                      "events": self.out / "events.jsonl",
                      "resources": self.out / "resources.jsonl",
                      "db": self.out / "endurance.db",
                      "summary": self.out / "summary.json", "report": self.out / "report.md",
                      "metrics": self.out / "metrics_export.json"}
        self.server: Any = None
        self.client: Any = None
        self.policy: Any = None
        self.monitor = ResourceMonitor(interval_seconds=settings.performance
                                       .metrics_interval_seconds, history=100_000)
        self.records: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.stop_reason: str | None = None
        self.reset_failures = 0
        self._previous_agent: weakref.ReferenceType[Any] | None = None
        self.current_runner: Any = None

    # -- set-up -----------------------------------------------------------------------------

    def _policy_version(self) -> str:
        return getattr(self.policy, "version", None) or "deterministic"

    def _prepare(self) -> int:
        """Load or create the checkpoint; returns the first episode to play."""
        self.out.mkdir(parents=True, exist_ok=True)
        cp, warns = load_checkpoint(self.paths["checkpoint"])
        self.warnings += warns
        cfg = self.config.checkpoint_config()
        if cp is not None and not self.resume:
            raise CheckpointError(f"{self.out} already holds an endurance run (checkpoint "
                                  f"after episode {cp.last_completed_episode}); use --resume "
                                  "to continue it or choose another output directory")
        if cp is None and self.resume and self.paths["results"].exists():
            raise CheckpointError("results exist but no usable checkpoint was found; refusing "
                                  "to resume without one (nothing was deleted)")
        if cp is not None:
            check_compatible(cp, config=cfg, agent_version=AGENT_VERSION)
        records, warns = read_results(self.paths["results"])
        self.warnings += warns
        if warns:                                     # drop a torn line, keep every record
            rewrite_results(self.paths["results"], records)
        self.records = [r for r in records if r.get("outcome") in DONE_OUTCOMES]
        if len(self.records) != len(records):
            # interrupted (cancelled) episodes are played again
            rewrite_results(self.paths["results"], self.records)
        done = sorted(int(r["episode"]) for r in self.records)
        if done != list(range(len(done))):
            raise CheckpointError(f"results are not a contiguous run of episodes: {done[:10]}")
        if cp is not None and cp.last_completed_episode >= len(done):
            raise CheckpointError(f"checkpoint says episode {cp.last_completed_episode} "
                                  f"finished but only {len(done)} results are on disk")
        self._check_database(len(done))
        return len(done)

    def _check_database(self, next_episode: int) -> None:
        """Abort sessions of an interrupted episode; refuse duplicate successful records."""
        db = DatabaseManager(self.paths["db"])
        try:
            db.initialize()
            rows = db.query_all(
                "SELECT r.episode AS ep, s.status AS status, s.id AS sid "
                "FROM simulation_runs r JOIN simulation_sessions s ON s.id = r.session_id")
            ok: dict[int, int] = {}
            for r in rows:
                if r["status"] == "completed":
                    ok[int(r["ep"])] = ok.get(int(r["ep"]), 0) + 1
            dup = sorted(e for e, n in ok.items() if n > 1)
            if dup:
                raise CheckpointError(f"database holds duplicate successful records for "
                                      f"episodes {dup[:10]}")
            beyond = sorted(e for e in ok if e >= next_episode)
            if beyond:
                self.warnings.append(f"database has completed sessions for episodes {beyond} "
                                     "without results on disk; those episodes are played again "
                                     "and their old sessions marked aborted")
            stale = [int(r["sid"]) for r in rows if r["status"] == "running"
                     or (r["status"] == "completed" and int(r["ep"]) >= next_episode)]
            if stale:
                from chest_hunter.models import utc_now

                with db.transaction() as conn:
                    for sid in stale:
                        conn.execute("UPDATE simulation_sessions SET status = 'aborted', "
                                     "ended_at = COALESCE(ended_at, ?) WHERE id = ?",
                                     (utc_now().isoformat(), sid))
                self.warnings.append(f"{len(stale)} session(s) of interrupted episodes marked "
                                     "aborted")
        finally:
            db.close()

    def _start_environment(self) -> None:
        from chest_hunter.unreal_bridge.client import BridgeClient

        br = self.settings.unreal_bridge
        if self.config.backend == "reference":
            from chest_hunter.simulation3d.environment3d import EnvSettings
            from chest_hunter.unreal_bridge.mock_server import MockUnrealServer

            base = EnvSettings(**{**{f: getattr(EnvSettings(), f)
                                     for f in EnvSettings.__dataclass_fields__},
                                  **self._env_overrides()})
            if self.server is not None:
                with contextlib.suppress(Exception):
                    self.server.stop()
            # oracle=True only lets the *runner* read end-of-episode ground truth for scoring;
            # the agent's AgentPort cannot reach the privileged commands.
            self.server = MockUnrealServer(port=0, oracle=True, evaluation=False,
                                           base_settings=base).start()
            port = int(self.server.port)
        else:
            port = br.port
        if self.client is not None:
            with contextlib.suppress(Exception):
                self.client.close()
        self.client = BridgeClient(br.host, port, timeout=br.timeout_seconds,
                                   connect_attempts=br.connect_attempts)
        self.client.connect()

    def _env_overrides(self) -> dict[str, Any]:
        c3 = self.settings.simulation_3d
        return {"odometry_drift": c3.odometry_drift, "noise_std": c3.noise_std,
                "reward_probability": c3.reward_probability,
                "interaction_failure_probability": c3.interaction_failure_probability}

    def _reset_environment(self) -> None:
        """Make sure the environment can be reset safely before the next episode."""
        limit = self.config.max_reset_attempts
        last: Exception | None = None
        for attempt in range(1, limit + 1):
            try:
                server_dead = self.server is not None and not self.server.running
                if server_dead or self.client is None or not self.client.connected:
                    self._start_environment()
                self.client.ping(attempt)
                return
            except Exception as exc:                  # bounded: never an endless restart
                last = exc
                self.reset_failures += 1
                logger.warning("environment reset attempt %d/%d failed: %s", attempt, limit, exc)
                time.sleep(min(2.0, 0.2 * 2 ** attempt))
                with contextlib.suppress(Exception):
                    self._start_environment()
        raise _ResetFailed(f"the research environment could not be reset safely after "
                           f"{limit} attempts ({type(last).__name__}: {last})")

    # -- the run --------------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        cfg = self.config
        first = self._prepare()
        if cfg.model_dir:
            from chest_hunter.learning.learned_policy import LearnedPolicy

            self.policy = LearnedPolicy(cfg.model_dir)     # one session for the whole run
        old_handlers = self._install_handlers()
        started = time.perf_counter()
        self.monitor.sample()
        db = DatabaseManager(self.paths["db"])
        try:
            db.initialize()
            self._start_environment()
            for episode in range(first, cfg.episodes):
                self.control.wait_while_paused()
                if self.control.stop_requested:
                    self.stop_reason = self.control.stop_reason or "stopped"
                    break
                if cfg.max_total_seconds is not None \
                        and time.perf_counter() - started >= cfg.max_total_seconds:
                    self.stop_reason = (f"time budget of {cfg.max_total_seconds:.0f} s "
                                        "(wall clock) reached")
                    break
                try:
                    self._reset_environment()
                except _ResetFailed as exc:
                    self.stop_reason = str(exc)
                    break
                record = self._episode(db, episode)
                if record["outcome"] == "cancelled":
                    self.stop_reason = self.control.stop_reason or "interrupted"
                    break
                append_result(self.paths["results"], record)
                self.records.append(record)
                if self.on_episode is not None:
                    self.on_episode(record)
                if (episode + 1) % cfg.checkpoint_every == 0:
                    self._checkpoint()
                self.progress(f"  episode {episode + 1}/{cfg.episodes}: {record['outcome']}, "
                              f"{record['opened_truth']} opened, rss {record['rss_mb']} MB")
        finally:
            db.close()
            self._checkpoint()
            self._restore_handlers(old_handlers)
            with contextlib.suppress(Exception):
                if self.client is not None:
                    self.client.close()
            with contextlib.suppress(Exception):
                if self.server is not None:
                    self.server.stop()
        summary = self.summary(time.perf_counter() - started)
        self.paths["summary"].write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        self.paths["report"].write_text(format_endurance_report(summary) + "\n",
                                        encoding="utf-8")
        return summary

    def _episode(self, db: DatabaseManager, episode: int) -> dict[str, Any]:
        from chest_hunter.agent.runner_3d import Runner3D, _layout_hash
        from chest_hunter.scenarios import ARENA_3D_ID, seed_arena_3d

        cfg = self.config
        map_id = cfg.maps[episode % len(cfg.maps)]
        if map_id == ARENA_3D_ID:
            seed_arena_3d(db)
        planner_name = self.control.planner or cfg.planner
        setup = custom_setup(policy=cfg.policy, planner=planner_name,
                             supervisor=cfg.supervisor, adaptive_compute=cfg.adaptive_compute)
        part = partition(map_id=map_id, env_overrides=self._env_overrides(), setup=setup,
                         model_version=getattr(self.policy, "version", None), width=cfg.width,
                         height=cfg.height, fov_deg=self.settings.simulation_3d.fov_deg,
                         decision_ticks=self.settings.simulation_3d.decision_ticks)
        layout = _layout_hash(map_id)
        # History must use this run's spawn-point ids: with preserve_estimates it is the run's
        # own database (earlier episodes of this run); otherwise the planner starts every
        # episode without history (configured spawn probabilities, default reliabilities).
        if cfg.preserve_estimates:
            history_db = db
        else:
            history_db = DatabaseManager(":memory:")
            history_db.initialize()
        planner = make_planner(self.settings, setup, history_db, map_id, layout,
                               partition_key=part["key"])
        cpu0, wall0 = time.process_time(), time.perf_counter()
        control = self.control

        def hook(runner: Any, result: Any) -> None:
            if control.stop_requested and not runner.controller.is_terminal():
                runner.controller.stop("cancelled", control.stop_reason or "interrupted")
            control.apply_to_runner(runner)
            if control.paused:
                control.wait_while_paused()
            if control.take_export():
                self.export_metrics(runner)
            if self.on_decision is not None:
                self.on_decision(runner, result)

        runner = Runner3D.from_settings(
            db, self.settings, mode="hybrid", seed=cfg.seed, episode=episode,
            backend=cfg.backend, map_id=map_id, collect_truth=True,
            config=agent_config(self.settings, setup), max_ticks=cfg.max_ticks_per_episode,
            width=cfg.width, height=cfg.height, policy=setup.policy,
            learned_policy=self.policy, planner=planner,
            policy_options=policy_options(self.settings, setup) or None,
            supervisor=supervisor_config(self.settings, setup), client=self.client,
            partition=part, on_decision=hook)
        self.current_runner = runner
        error = None
        result = None
        try:
            result = runner.run()
        except Exception as exc:                       # logged as a failed episode
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("episode %d failed: %s", episode, error)
        if result is not None and planner is not None and cfg.preserve_estimates:
            planner.flush(result.session_id)      # experience carried to later episodes
        if not cfg.preserve_estimates:
            history_db.close()
        record = self._record(episode, map_id, runner, result, error,
                              time.process_time() - cpu0, time.perf_counter() - wall0)
        if runner.supervisor is not None:
            runner.supervisor.events.write_jsonl(self.paths["events"])
        agent = getattr(runner.controller, "agent", runner.controller)
        self.current_runner = None
        del runner, planner, result
        released = None
        if self._previous_agent is not None:
            gc.collect()
            released = self._previous_agent() is None
        record["previous_agent_released"] = released
        self._previous_agent = weakref.ref(agent) if agent is not None else None
        del agent
        sample = self.monitor.sample(episode=episode)
        with self.paths["resources"].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, default=str) + "\n")
        record["process_cpu_percent"] = sample["process_cpu_percent"]
        return record

    def _record(self, episode: int, map_id: str, runner: Any, result: Any, error: str | None,
                cpu: float, wall: float) -> dict[str, Any]:
        rss, source = process_rss_mb()
        base = {"episode": episode, "map_id": map_id, "seed": self.config.seed,
                "cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3),
                "rss_mb": round(rss, 2), "rss_source": source,
                "gc_objects": len(gc.get_objects()), "threads": threading.active_count(),
                "open_fds": _open_fds()}
        if result is None:
            return {**base, "outcome": "failed", "end_reason": error, "sim_seconds": None,
                    "opened": 0, "opened_truth": None, "throughput_truth_per_min": None,
                    "available": None, "supervisor_interventions": 0,
                    "supervisor_aborted": None, "communication_failure": None,
                    "recoveries": None, "recovery_attempts": 0, "recovery_resolved": 0,
                    "exception": error}
        m, t = result.metrics, result.truth
        minutes = result.sim_seconds / 60.0
        sup = result.supervisor or {}
        log = m.recovery_log
        outcome = result.outcome
        return {**base, "outcome": outcome, "end_reason": result.end_reason,
                "sim_seconds": round(result.sim_seconds, 2), "opened": result.chests_opened,
                "opened_truth": t.get("opened_truth"),
                "throughput_truth_per_min": round((t.get("opened_truth") or 0) / minutes, 4)
                if minutes > 0 and t else None,
                "available": t.get("available"),
                "over_reported": max(0, result.chests_opened - (t.get("opened_truth") or 0))
                if t else None,
                "supervisor_interventions": int(sup.get("interventions_total", 0)),
                "supervisor_detections": sup.get("detections", {}),
                "supervisor_aborted": sup.get("aborted"),
                "communication_failure": result.communication_failure,
                "recoveries": m.recoveries, "recovery_attempts": sum(log.attempts.values()),
                "recovery_resolved": sum(log.resolved.values()),
                "frames_analysed": m.analyses, "frames_skipped": m.frames_skipped_unchanged,
                "exception": None}

    def _checkpoint(self) -> None:
        done = [int(r["episode"]) for r in self.records]
        cp = Checkpoint(kind="endurance", config=self.config.checkpoint_config(),
                        agent_version=AGENT_VERSION, policy_version=self._policy_version(),
                        seed=self.config.seed,
                        seed_sequence="episode i uses (seed, i) on map maps[i % len(maps)]",
                        last_completed_episode=max(done) if done else -1, completed=len(done),
                        results_file=self.paths["results"].name,
                        aggregate=self._aggregate())
        old, _ = (None, None)
        with contextlib.suppress(CheckpointError):
            old, _ = load_checkpoint(self.paths["checkpoint"])
        if old is not None:
            cp.created = old.created
        save_checkpoint(self.paths["checkpoint"], cp)

    def export_metrics(self, runner: Any = None) -> Path:
        from chest_hunter.runtime.operations import snapshot

        payload = {"aggregate": self._aggregate(), "resources": self.monitor.summary(),
                   "current": snapshot(runner, monitor=self.monitor) if runner is not None
                   and runner.controller is not None else None}
        self.paths["metrics"].write_text(json.dumps(payload, indent=1, default=str) + "\n",
                                         encoding="utf-8")
        return self.paths["metrics"]

    # -- signals ----------------------------------------------------------------------------

    def _install_handlers(self) -> dict[int, Any]:
        if not self.install_signal_handlers \
                or threading.current_thread() is not threading.main_thread():
            return {}
        old: dict[int, Any] = {}

        def handler(signum: int, _frame: Any) -> None:
            self.control.request_stop(f"interrupted by signal {signal.Signals(signum).name}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, handler)
        return old

    @staticmethod
    def _restore_handlers(old: dict[int, Any]) -> None:
        for sig, h in old.items():
            signal.signal(sig, h)

    # -- results ----------------------------------------------------------------------------------

    def _aggregate(self) -> dict[str, Any]:
        rs = self.records
        n = len(rs)
        outcomes: dict[str, int] = {}
        for r in rs:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        attempts = sum(int(r.get("recovery_attempts") or 0) for r in rs)
        resolved = sum(int(r.get("recovery_resolved") or 0) for r in rs)
        return {"episodes": n, "outcomes": outcomes,
                "completion_rate": round(outcomes.get("completed", 0) / n, 4) if n else None,
                "failed": outcomes.get("failed", 0),
                "communication_failures": sum(1 for r in rs if r.get("communication_failure")),
                "supervisor_interventions": sum(int(r.get("supervisor_interventions") or 0)
                                                for r in rs),
                "supervisor_aborts": sum(1 for r in rs if r.get("supervisor_aborted")),
                "recovery_success_rate": round(resolved / attempts, 4) if attempts else None,
                "opened_truth_total": sum(int(r.get("opened_truth") or 0) for r in rs),
                "over_reported_total": sum(int(r.get("over_reported") or 0) for r in rs)}

    def summary(self, wall_seconds: float) -> dict[str, Any]:
        rs = self.records
        sims = [float(r["sim_seconds"]) for r in rs if r.get("sim_seconds") is not None]
        out = {"config": self.config.checkpoint_config() | {"episodes": self.config.episodes},
               "agent_version": AGENT_VERSION, "policy_version": self._policy_version(),
               "output_dir": str(self.out), "stopped_early": self.stop_reason,
               "episodes_completed_in_run": len(rs), "episodes_requested": self.config.episodes,
               "wall_seconds_this_session": round(wall_seconds, 1),
               "environment_resets_failed": self.reset_failures, "warnings": self.warnings,
               "aggregate": self._aggregate(),
               "throughput_truth_per_min": summarise(
                   [float(r["throughput_truth_per_min"]) for r in rs
                    if r.get("throughput_truth_per_min") is not None]),
               "episode_sim_seconds": summarise(sims),
               "episode_sim_seconds_max": max(sims) if sims else None,
               "cpu_seconds_per_episode": summarise([float(r["cpu_seconds"]) for r in rs]),
               "memory": memory_trend(rs),
               "previous_agent_released": _released(rs),
               "resources": self.monitor.summary(),
               "environment": ("Python reference arena (protocol-identical mock of the Unreal "
                               "research server); NOT Unreal Engine"
                               if self.config.backend == "reference" else
                               "Unreal research server"),
               "time_basis": "throughput per simulated minute; CPU and wall seconds are "
                             "real time"}
        return out


def _open_fds() -> int | None:
    path = Path("/proc/self/fd")
    try:
        return len(os.listdir(path))
    except OSError:
        return None


def _released(rs: list[dict[str, Any]]) -> dict[str, int]:
    vals = [r.get("previous_agent_released") for r in rs]
    return {"checked": sum(v is not None for v in vals), "released": sum(v is True for v in vals),
            "not_released": sum(v is False for v in vals)}


def _slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def memory_trend(rs: list[dict[str, Any]], *, warmup: int | None = None,
                 threshold_mb_per_100: float = 2.0) -> dict[str, Any]:
    """Least-squares growth of resident memory and live objects after a warm-up."""
    if not rs:
        return {"episodes": 0}
    w = warmup if warmup is not None else max(5, len(rs) // 10)
    tail = rs[w:] if len(rs) > w + 2 else rs
    xs = [float(r["episode"]) for r in tail]
    rss = [float(r["rss_mb"]) for r in tail]
    objs = [float(r["gc_objects"]) for r in tail]
    s_rss, s_obj = _slope(xs, rss), _slope(xs, objs)
    fds = [r.get("open_fds") for r in rs if r.get("open_fds") is not None]
    threads = [r.get("threads") for r in rs if r.get("threads") is not None]
    enough = len(tail) >= 20
    bounded = (s_rss is not None and s_rss * 100 < threshold_mb_per_100) if enough else None
    return {"warmup_episodes": w, "episodes_fitted": len(tail),
            "rss_mb_first": rs[0]["rss_mb"], "rss_mb_after_warmup": tail[0]["rss_mb"],
            "rss_mb_last": rs[-1]["rss_mb"], "rss_mb_max": max(float(r["rss_mb"]) for r in rs),
            "rss_growth_mb_per_100_episodes": round(s_rss * 100, 3) if s_rss is not None
            else None,
            "gc_objects_growth_per_100_episodes": round(s_obj * 100, 1) if s_obj is not None
            else None,
            "open_fds_min_max": [min(fds), max(fds)] if fds else None,  # type: ignore[type-var]
            "threads_min_max": [min(threads), max(threads)] if threads else None,  # type: ignore[type-var]
            "criterion": f"growth below {threshold_mb_per_100} MB per 100 episodes after warm-up "
                         "(at least 20 episodes fitted)",
            "bounded": bounded}


def format_endurance_report(summary: dict[str, Any]) -> str:
    a = summary["aggregate"]
    mem = summary["memory"]
    t = summary["throughput_truth_per_min"]

    def f(x: Any, d: int = 3) -> str:
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else (
            f"{x:.{d}f}" if isinstance(x, float) else str(x))

    lines = [
        f"Endurance run: {summary['episodes_completed_in_run']} of "
        f"{summary['episodes_requested']} episodes "
        f"({summary['environment']}).",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Outcomes | {a['outcomes']} |",
        f"| Completion rate (all spawn points resolved) | {f(a['completion_rate'], 4)} |",
        f"| Failed episodes | {a['failed']} |",
        f"| Communication failures | {a['communication_failures']} |",
        f"| Supervisor interventions / aborts | {a['supervisor_interventions']} / "
        f"{a['supervisor_aborts']} |",
        f"| Recovery success rate | {f(a['recovery_success_rate'], 4)} |",
        f"| Chests opened per simulated minute (mean [95 % CI]) | {f(t.get('mean'))} "
        f"[{f(t.get('ci_low'))}, {f(t.get('ci_high'))}] |",
        f"| Episode simulated seconds (mean / max) | "
        f"{f(summary['episode_sim_seconds'].get('mean'), 1)} / "
        f"{f(summary['episode_sim_seconds_max'], 1)} |",
        f"| CPU seconds per episode (mean) | "
        f"{f(summary['cpu_seconds_per_episode'].get('mean'), 2)} |",
        f"| RSS first / after warm-up / last / max (MB) | {mem.get('rss_mb_first')} / "
        f"{mem.get('rss_mb_after_warmup')} / {mem.get('rss_mb_last')} / "
        f"{mem.get('rss_mb_max')} |",
        f"| RSS growth after warm-up (MB per 100 episodes) | "
        f"{mem.get('rss_growth_mb_per_100_episodes')} |",
        f"| Live Python objects growth (per 100 episodes) | "
        f"{mem.get('gc_objects_growth_per_100_episodes')} |",
        f"| Open file descriptors (min, max) | {mem.get('open_fds_min_max')} |",
        f"| Threads (min, max) | {mem.get('threads_min_max')} |",
        f"| Previous agent released | {summary['previous_agent_released']} |",
        f"| Stopped early | {summary['stopped_early'] or 'no'} |",
    ]
    if summary.get("warnings"):
        lines += ["", "Warnings:"] + [f"- {w}" for w in summary["warnings"]]
    return "\n".join(lines)
