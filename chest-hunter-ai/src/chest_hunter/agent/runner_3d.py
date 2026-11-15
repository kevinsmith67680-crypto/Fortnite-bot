"""Run one autonomous 3D episode against a research server (Unreal or reference).

The runner owns everything the policy must not touch:

* the SQLite session, run row and batched observation recording,
* the bridge connection (the policy only gets an :class:`AgentPort`),
* the optional embedded reference server (``backend="reference"``),
* evaluation: labels and final ground truth through the privileged channel,
  used only *after* the policy has acted, for metrics.

Modes:

* ``oracle``: exact chest positions (privileged channel), NavMesh paths.
* ``hybrid``: vision for chests + NavMesh navigation service (Phase 4 target).
* ``visual_nav``: vision for chests + only local navigation probes; the agent
  builds its own map and plans its own paths.
"""

from __future__ import annotations

import logging
import math
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from chest_hunter.agent.controller_3d import (
    Agent3DConfig,
    Metrics3D,
    OracleAgent3D,
    VisualAgent3D,
)
from chest_hunter.agent.recorder import ObservationRecorder
from chest_hunter.analytics.vision_metrics_3d import Evaluator3D
from chest_hunter.chest_repository import ChestRepository
from chest_hunter.config import Settings
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ValidationError
from chest_hunter.learning.demonstration_recorder import phase_of
from chest_hunter.localization.world_model_3d import SpawnPrior3D
from chest_hunter.models import utc_now
from chest_hunter.run_repository import RunRepository
from chest_hunter.session_repository import SessionRepository
from chest_hunter.unreal_bridge.client import AgentPort, BridgeClient
from chest_hunter.unreal_bridge.protocol import DEFAULT_PORT
from chest_hunter.vision.chest_detector_3d import ChestDetector3D, TemplateSet3D

logger = logging.getLogger(__name__)

MODES_3D = ("oracle", "hybrid", "visual_nav")
STRATEGY_3D = {"oracle": "3d-oracle", "hybrid": "3d-hybrid-visual", "visual_nav": "3d-visual-nav"}
_TEMPLATES: dict[str, TemplateSet3D] = {}


@dataclass(slots=True)
class Result3D:
    mode: str
    seed: int
    episode: int
    outcome: str
    end_reason: str | None
    session_id: int
    sim_seconds: float
    ticks: int
    metrics: Metrics3D
    chests_opened: int
    chests_per_minute: float | None
    truth: dict[str, Any] = field(default_factory=dict)       # evaluation only
    vision: dict[str, Any] | None = None                       # evaluation only
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    bridge_round_trip_ms: float = 0.0
    backend: str = "reference"
    #: Phase 6: supervisor summary (None when no supervisor watched the episode).
    supervisor: dict[str, Any] | None = None
    communication_failure: str | None = None


class _SimClock:
    """Session timestamps on the simulated timeline (base = real start time)."""

    def __init__(self) -> None:
        self.base = utc_now()
        self.time = 0.0

    def __call__(self) -> datetime:
        return self.base + timedelta(seconds=self.time)


def _templates(settings: Settings) -> TemplateSet3D:
    path = str(settings.template_dir_3d())
    if path not in _TEMPLATES:
        _TEMPLATES[path] = TemplateSet3D.load(path)
    return _TEMPLATES[path]


class Runner3D:
    def __init__(self, db: DatabaseManager, settings: Settings, *, mode: str = "hybrid",
                 map_id: str = "research_arena_3d", seed: int = 42, episode: int = 0,
                 backend: str = "reference", host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                 width: int = 640, height: int = 360, fov_deg: float = 90.0,
                 graphics_quality: str = "low", max_ticks: int = 30 * 60 * 10,
                 detector: str = "hybrid", evaluate: bool = False,
                 config: Agent3DConfig | None = None, env_overrides: dict[str, Any] | None = None,
                 on_decision: Any = None, demo_recorder: Any = None, demo_episode_id: int = 0,
                 policy: str = "deterministic", learned_policy: Any = None,
                 planner: Any = None, policy_options: dict[str, Any] | None = None,
                 collect_truth: bool = False, supervisor: Any = None, profiler: Any = None,
                 client: BridgeClient | None = None,
                 partition: dict[str, Any] | None = None, exploration: Any = None,
                 thresholds: Any = None, faults: Any = None) -> None:
        mode = mode.replace("-", "_")
        if mode not in MODES_3D:
            raise ValidationError(f"mode must be one of {', '.join(MODES_3D)}; got {mode!r}")
        if backend not in ("reference", "unreal"):
            raise ValidationError("backend must be 'reference' or 'unreal'")
        self.db, self.settings = db, settings
        self.mode, self.map_id, self.seed, self.episode = mode, map_id, seed, episode
        self.backend, self.host, self.port = backend, host, port
        self.width, self.height, self.fov = width, height, fov_deg
        self.graphics_quality, self.max_ticks = graphics_quality, max_ticks
        self.detector_method = detector
        self.evaluate = evaluate
        self.config = config or Agent3DConfig()
        self.env_overrides = dict(env_overrides or {})
        self.on_decision = on_decision
        #: Phase 5: optional DemonstrationRecorder (records every step of this episode).
        self.demo_recorder = demo_recorder
        self.demo_episode_id = demo_episode_id
        #: Phase 5 control policy for the visual agent: deterministic | learned | hybrid.
        if policy not in ("deterministic", "learned", "hybrid"):
            raise ValidationError("policy must be deterministic, learned or hybrid")
        if policy != "deterministic" and learned_policy is None:
            raise ValidationError(f"policy '{policy}' needs a trained model (--model)")
        if policy != "deterministic" and mode == "oracle":
            raise ValidationError("the oracle baseline only runs the deterministic policy")
        self.policy = policy
        self.learned_policy = learned_policy
        self.policy_options = dict(policy_options or {})
        #: Phase 5 adaptive target planner (exploration.adaptive_planner.AdaptivePlanner).
        self.planner = planner
        #: Ground truth at the end of the episode only (runner-side, privileged channel; no
        #: per-frame labels). The policy's AgentPort cannot request it in any case.
        self.collect_truth = collect_truth
        #: Phase 6 reliability supervisor (runtime.supervisor.SupervisorConfig or Supervisor).
        self.supervisor_config = supervisor
        self.supervisor: Any = None
        #: Phase 6 component profiler (runtime.profiler.Profiler); None = not profiled.
        self.profiler = profiler
        #: Phase 6: a connected client to reuse (endurance runs keep one research server and
        #: one connection for many episodes); the runner then neither starts a server nor
        #: closes the connection.
        self.external_client = client
        #: Phase 6 experience partition, stored with the run (runtime.configurations).
        self.partition = partition
        #: Phase 7: active exploration (an exploration.active.ExplorationConfig or a
        #: callable returning an ActiveExplorer). The agent then gets *no* spawn points:
        #: it explores frontiers and turns the chests it sees into its own targets.
        self.exploration = exploration
        self.thresholds = thresholds
        #: Phase 7 robustness tests: a runtime.faults.FaultInjector for this episode.
        self.faults = faults
        self.explorer: Any = None
        self.selector: Any = None
        self.controller: Any = None
        self.evaluator: Evaluator3D | None = None
        self.server: Any = None
        self.client: BridgeClient | None = None

    @classmethod
    def from_settings(cls, db: DatabaseManager, settings: Settings, **overrides: Any) -> Runner3D:
        """Defaults from the ``simulation_3d`` and ``unreal_bridge`` config sections."""
        c3, br = settings.simulation_3d, settings.unreal_bridge
        kwargs: dict[str, Any] = {
            "map_id": c3.default_map, "backend": c3.backend, "host": br.host, "port": br.port,
            "width": c3.observation_width, "height": c3.observation_height, "fov_deg": c3.fov_deg,
            "graphics_quality": c3.graphics_quality, "max_ticks": c3.max_ticks,
            "detector": c3.detector, "config": Agent3DConfig(decision_ticks=c3.decision_ticks),
            "env_overrides": {"odometry_drift": c3.odometry_drift, "noise_std": c3.noise_std,
                              "reward_probability": c3.reward_probability,
                              "interaction_failure_probability":
                                  c3.interaction_failure_probability},
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(db, settings, **kwargs)

    # -- setup ---------------------------------------------------------------------------------

    def _priors(self) -> list[SpawnPrior3D]:
        from chest_hunter.simulation3d.generator7 import is_generated
        from chest_hunter.simulation3d.procedural import is_procedural

        if is_procedural(self.map_id) or is_generated(self.map_id):
            from chest_hunter.scenarios import seed_procedural_map

            seed_procedural_map(self.db, self.map_id)       # idempotent; uninformative prior
        chests = ChestRepository(self.db).get_chests_by_map(self.map_id)
        priors = [SpawnPrior3D(c.id, c.location_name, c.x, c.y, c.z, c.spawn_probability)
                  for c in chests if c.enabled and c.id is not None]
        if not priors:
            raise ValidationError(
                f"Map '{self.map_id}' has no enabled chest locations. Run "
                "'python -m chest_hunter seed-demo --scenario research_arena_3d'.")
        return priors

    def _start_server(self) -> int:
        from chest_hunter.simulation3d.environment3d import EnvSettings
        from chest_hunter.unreal_bridge.mock_server import MockUnrealServer

        defaults = {f: getattr(EnvSettings(), f) for f in EnvSettings.__dataclass_fields__}
        base = EnvSettings(**{**defaults, **self.env_overrides})
        self.server = MockUnrealServer(port=0, oracle=self.mode == "oracle" or self.collect_truth,
                                       evaluation=self.evaluate, base_settings=base).start()
        return int(self.server.port)

    # -- episode -------------------------------------------------------------------------------

    def run(self) -> Result3D:
        from chest_hunter.runtime.profiler import NULL_PROFILER
        from chest_hunter.unreal_bridge.client import BridgeError

        cpu0, wall0 = time.process_time(), time.perf_counter()
        prof = self.profiler or NULL_PROFILER
        priors = self._priors()
        if self.external_client is not None:
            client = self.external_client
            if not client.connected:
                client.connect()
        else:
            port = self._start_server() if self.backend == "reference" else self.port
            client = BridgeClient(self.host, port,
                                  timeout=self.settings.unreal_bridge.timeout_seconds,
                                  connect_attempts=self.settings.unreal_bridge.connect_attempts)
            try:                                   # connect before any session is opened
                client.connect()
            except BaseException:
                if self.server is not None:
                    self.server.stop()
                raise
        self.client = client
        comm_failure: str | None = None
        clock = _SimClock()
        sessions = SessionRepository(self.db, clock=clock)
        runs = RunRepository(self.db, clock=clock)
        session = sessions.start_session(self.map_id, notes=f"3D {self.mode} episode")
        assert session.id is not None
        runs.create_run(session.id, strategy=STRATEGY_3D[self.mode], seed=self.seed,
                        episode=self.episode, parameters={
                            "environment": "3d", "backend": self.backend, "mode": self.mode,
                            "observation": [self.width, self.height], "fov_deg": self.fov,
                            "graphics_quality": self.graphics_quality,
                            "detector": self.detector_method,
                            "max_ticks": self.max_ticks, "time_basis": "simulated",
                            "decision_ticks": self.config.decision_ticks,
                            "layout_hash": _layout_hash(self.map_id), "policy": self.policy,
                            "model_version": getattr(self.learned_policy, "version", None),
                            "adaptive_planning": self.planner is not None,
                            **({"partition": self.partition} if self.partition else {}),
                            **({"supervisor": True} if self.supervisor_config else {})})
        recorder = ObservationRecorder(self.db, sessions, session.id,
                                       batch_size=self.settings.agent.observation_batch_size)

        def record(location_id: int, *, present: bool, opened: bool, seconds: float,
                   reward: bool | None, sim_time: float) -> None:
            with prof.section("database_persistence"):
                recorder.record(location_id, present=present, opened=opened,
                                interaction_seconds=round(seconds, 4),
                                observed_at=clock.base + timedelta(seconds=sim_time),
                                reward_received=reward)

        def record_discovered(xyz: tuple[float, float, float], *, seconds: float,
                              reward: bool | None, sim_time: float) -> None:
            with prof.section("database_persistence"), self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO discovered_openings (session_id, x, y, z, observed_at, "
                    "interaction_seconds, reward_received) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session.id, float(xyz[0]), float(xyz[1]), float(xyz[2]),
                     (clock.base + timedelta(seconds=sim_time)).isoformat(),
                     round(max(0.0, seconds), 4), None if reward is None else int(reward)))

        outcome, reason = "failed", None
        try:
            navigation = {"oracle": "navmesh", "hybrid": "navmesh",
                          "visual_nav": "local"}[self.mode]
            telemetry = client.reset_episode(self.map_id, seed=self.seed, episode=self.episode,
                                             width=self.width, height=self.height,
                                             fov_deg=self.fov,
                                             graphics_quality=self.graphics_quality,
                                             max_ticks=self.max_ticks, navigation=navigation)
            agent_port = AgentPort(client)
            controller: Any
            if self.mode == "oracle":
                names = {p.name: p.location_id for p in priors}
                state0 = client.oracle_state()
                index_to_loc = {c["spawn_index"]: names.get(c["name"]) for c in state0["chests"]}
                controller = OracleAgent3D(agent_port, client.oracle_state, rules=client.rules,
                                           location_of=index_to_loc.get, recorder=record,
                                           decision_ticks=self.config.decision_ticks)
            else:
                detector = ChestDetector3D(_templates(self.settings), method=self.detector_method)
                agent_priors = priors
                if self.exploration is not None:
                    from chest_hunter.exploration.active import (
                        ActiveExplorer,
                        ConfidenceThresholds,
                        ExplorationConfig,
                    )

                    self.explorer = self.exploration() if callable(self.exploration) \
                        else ActiveExplorer(self.exploration if isinstance(
                            self.exploration, ExplorationConfig) else None)
                    agent_priors = []                 # no prior chest coordinates
                    if self.thresholds is None:
                        self.thresholds = ConfidenceThresholds()
                visual = VisualAgent3D(agent_port, agent_priors, rules=client.rules,
                                       detector=detector, navigation=navigation,
                                       config=self.config, recorder=record,
                                       explorer=self.explorer, thresholds=self.thresholds)
                visual.discovery_recorder = record_discovered
                if self.planner is not None:
                    self.planner.attach(visual, map_id=self.map_id)
                controller = visual
                if self.policy != "deterministic" or self.learned_policy is not None:
                    from chest_hunter.learning.policy_selector import PolicySelector

                    controller = PolicySelector(visual, self.learned_policy, mode=self.policy,
                                                **self.policy_options)
                    self.selector = controller
                if self.evaluate:
                    from chest_hunter.vision.perspective import CameraIntrinsics
                    intr = CameraIntrinsics(self.width, self.height, self.fov)
                    self.evaluator = Evaluator3D(intr.fx, intr.fy)
            self.controller = controller
            if self.profiler is not None and self.mode != "oracle":
                visual.set_profiler(self.profiler)
                if self.selector is not None:
                    self.selector.profiler = self.profiler
            supervisor = None
            if self.supervisor_config is not None and self.mode != "oracle":
                from chest_hunter.runtime.supervisor import Supervisor

                supervisor = self.supervisor_config if isinstance(
                    self.supervisor_config, Supervisor) else Supervisor(
                    self.supervisor_config, episode=self.episode)
                supervisor.attach(controller)
                self.supervisor = supervisor
            controller.initialize(telemetry)
            if self.faults is not None and self.mode != "oracle":
                self.faults.attach(self)
            if self.demo_recorder is not None:
                self.demo_recorder.start(
                    episode_id=self.demo_episode_id, map_id=self.map_id, seed=self.seed,
                    episode=self.episode, source=_demo_source(self.policy), telemetry=telemetry,
                    settings={"width": self.width, "height": self.height, "fov_deg": self.fov,
                              "graphics_quality": self.graphics_quality,
                              "max_ticks": self.max_ticks, "navigation": navigation,
                              "decision_ticks": self.config.decision_ticks,
                              "controller": f"{self.mode}/{self.policy}",
                              "layout_hash": _layout_hash(self.map_id)})
            agent = getattr(controller, "agent", controller)
            ticks = 0
            last_obs = None                      # the frame the controller acts on, if fresh
            while not controller.is_terminal():
                if supervisor is not None:
                    supervisor.before_tick()
                with prof.section("action_generation"):
                    controller.tick()
                if controller.is_terminal():
                    break
                if supervisor is not None:
                    with prof.section("supervision"):
                        supervisor.validate_action()
                observe = controller.wants_observation()
                action = controller.action.to_dict()
                action_time = clock.time
                phase = phase_of(getattr(agent, "state", None))
                try:
                    with prof.section("simulator_step"):
                        result = client.step(action, ticks=self.config.decision_ticks,
                                             observe=observe)
                except BridgeError as exc:
                    if supervisor is None:
                        raise
                    comm_failure = f"{type(exc).__name__}: {exc}"
                    supervisor.communication_failure(exc)
                    break
                if supervisor is not None:
                    result = supervisor.filter_step(result, observe=observe, width=self.width,
                                                    height=self.height)
                ticks = result.telemetry.tick
                clock.time = result.telemetry.sim_time
                controller.on_step(result)
                if supervisor is not None:
                    with prof.section("supervision"):
                        supervisor.after_step(result)
                if self.demo_recorder is not None:
                    self.demo_recorder.record_step(
                        action=action, ticks=self.config.decision_ticks, action_time=action_time,
                        telemetry_after=result.telemetry, observation=last_obs,
                        observed=observe, phase=phase)
                last_obs = result.observation
                if self.evaluator is not None and result.observation is not None \
                        and isinstance(agent, VisualAgent3D):
                    labels = client.evaluation_labels(result.observation.frame_id)
                    self.evaluator.observe(labels, agent.last_detections,
                                           agent.last_estimates,
                                           result.observation.odometry.position)
                if self.on_decision is not None:
                    self.on_decision(self, result)
                if result.ended or ticks >= self.max_ticks:
                    controller.stop("tick_limit", f"reached max_ticks={self.max_ticks}")
            outcome = controller.outcome or "completed"
            reason = controller.end_reason
            if self.demo_recorder is not None and self.demo_recorder.active:
                self.demo_recorder.stop(outcome={"outcome": outcome, "reason": reason,
                                                 "chests_opened": controller.metrics.chests_opened})
            truth: dict[str, Any] = {}
            if comm_failure is None:
                if self.evaluate or self.mode == "oracle" or self.collect_truth:
                    truth = self._truth(client, priors)
                summary = client.stop_episode(f"{self.mode} agent finished")
                truth["server_summary"] = summary.get("summary", {})
        except BaseException as exc:
            outcome, reason = "failed", f"{type(exc).__name__}: {exc}"
            truth = {}
            if self.demo_recorder is not None and self.demo_recorder.active:
                self.demo_recorder.stop(outcome={"outcome": "failed", "reason": reason})
            raise
        finally:
            if self.faults is not None:
                self.faults.detach()
            recorder.flush()
            m = self.controller.metrics if self.controller is not None else Metrics3D()
            final_time = clock.time
            if outcome in ("completed", "tick_limit"):
                sessions.complete_session(session.id)
            else:
                sessions.abort_session(session.id)
            runs.finish_run(session.id, outcome=outcome if outcome in
                            ("completed", "tick_limit", "failed", "cancelled") else "failed",
                            failure_reason=reason if outcome in ("failed", "cancelled") else None,
                            ticks=int(round(final_time * 30)), simulated_seconds=final_time,
                            travel_distance=m.travel_distance, recoveries=m.recoveries,
                            unreachable_targets=m.unreachable_targets,
                            abandoned_targets=m.abandoned_targets, rewards=m.rewards,
                            navigation_failures=m.navigation_failures)
            agent_now = getattr(self.controller, "agent", self.controller)
            if isinstance(agent_now, VisualAgent3D):
                try:                                    # Phase 6 experience summary
                    from chest_hunter.exploration.experience import record_episode

                    record_episode(self.db, session_id=session.id, map_id=self.map_id,
                                   partition_key=(self.partition or {}).get("key", ""),
                                   outcome=outcome if outcome in ("completed", "tick_limit",
                                                                  "failed", "cancelled")
                                   else "failed", sim_seconds=final_time, agent=agent_now,
                                   supervisor=self.supervisor.summary()
                                   if self.supervisor is not None else None)
                except Exception as exc:             # never mask the episode's own outcome
                    logger.warning("episode experience not recorded: %s", exc)
            if self.external_client is None:
                client.close()
                if self.server is not None:
                    self.server.stop()
        m = self.controller.metrics
        per_min = m.chests_opened / (final_time / 60.0) if final_time > 0 else None
        return Result3D(self.mode, self.seed, self.episode, outcome, reason, session.id, final_time,
                        ticks, m, m.chests_opened, per_min, truth,
                        self.evaluator.summary() if self.evaluator else None,
                        time.process_time() - cpu0, time.perf_counter() - wall0,
                        1000.0 * client.round_trip_seconds / max(1, client.requests), self.backend,
                        self.supervisor.summary() if self.supervisor is not None else None,
                        comm_failure)

    def _truth(self, client: BridgeClient, priors: list[SpawnPrior3D]) -> dict[str, Any]:
        """Final ground truth via the privileged channel (evaluation only)."""
        state = client.oracle_state()
        from chest_hunter.simulation3d.environment3d import cached_arena

        arena, _, nav = cached_arena(self.map_id)       # shared layout (also used by Unreal)
        start = arena.start
        present = [c for c in state["chests"] if c["present"]]
        reachable = [c for c in present
                     if nav.find_path(start, tuple(c["position"]), goal_tolerance=1.5).found]
        names = {p.name: p.location_id for p in priors}
        # the visual agent itself, also when a PolicySelector wraps it
        agent = getattr(self.controller, "agent", self.controller)
        seen = sorted(p.name for p in priors if isinstance(agent, VisualAgent3D)
                      and p.location_id in agent.world.seen_locations)
        if isinstance(agent, VisualAgent3D) and agent.world.discover:
            # Phase 7: the agent's own (dynamic) spawn points matched to real chests, for
            # scoring only (evaluation channel, after the episode).
            from chest_hunter.localization.world_model_3d import PriorStatus3D

            found = [d for d in agent.world.priors.values() if d.location_id < 0 and (
                d.location_id in agent.world.seen_locations or d.status in (
                    PriorStatus3D.CHEST_SEEN, PriorStatus3D.OPENED, PriorStatus3D.FAILED))]
            seen = sorted({c["name"] for c in present for d in found
                           if math.dist(c["position"][:2], d.xyz[:2]) <= 2.0
                           and abs(c["position"][2] - d.z) <= 1.0})
        coverage = None
        if self.explorer is not None:
            coverage = _true_coverage(nav, start, self.explorer)
        return {"present": len(present), "available": len(reachable),
                "opened_truth": sum(1 for c in present if c["opened"]),
                "available_names": sorted(c["name"] for c in reachable),
                "opened_names": sorted(c["name"] for c in present if c["opened"]),
                "seen_names": seen, "true_coverage": coverage,
                "location_ids_opened": sorted(names[c["name"]] for c in present
                                              if c["opened"] and c["name"] in names)}


def _true_coverage(nav: Any, start: tuple[float, float, float], explorer: Any) -> float | None:
    """Share of the reachable walkable area (evaluation-only navigation graph) that the
    agent's camera covered, on the explorer's 2 m grid (scoring only)."""
    nodes = nav.reachable_from(start)
    if not nodes:
        return None
    cmap = explorer.map
    cells: set[tuple[int, int, int]] = set()
    for node in nodes:
        x, y, z = nav.node_position(node)
        cells.add(cmap.key(x, y, z))
    seen = 0
    for key in cells:
        c = cmap.cells.get(key)
        if c is not None and c.views > 0:
            seen += 1
    return round(seen / len(cells), 4)


def _layout_hash(map_id: str) -> str | None:
    """Version of the map layout (statistics from another version are never mixed)."""
    from chest_hunter.simulation3d.environment3d import cached_arena
    from chest_hunter.simulation3d.procedural import layout_hash

    try:
        return layout_hash(cached_arena(map_id)[0])
    except ValueError:
        return None


def _demo_source(policy: str) -> str:
    from chest_hunter.learning.demonstration_recorder import SOURCE_SCRIPTED

    return SOURCE_SCRIPTED


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0 if sys.platform != "darwin" else usage / (1024.0 * 1024.0)

