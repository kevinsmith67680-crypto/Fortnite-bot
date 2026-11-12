"""Wires the database, world, navigation and agent into one runnable episode.

``SimulationRunner`` is UI-agnostic: headless runs call :meth:`run`, the GUI
calls :meth:`step` from Tkinter's ``after()`` loop, and the benchmark creates
many runners that share one :class:`World` and :class:`DistanceOracle`.

Database interaction per episode: one session + one run row at the start, one
candidate query, observation writes in batches, and one completion at the end.
Nothing touches SQLite on an ordinary simulation tick.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from chest_hunter.agent.controller import AgentController
from chest_hunter.agent.fsm import StateMachine
from chest_hunter.agent.recorder import ObservationRecorder, SimulationClock
from chest_hunter.agent.states import AgentState
from chest_hunter.analytics.metrics import EpisodeMetrics
from chest_hunter.chest_repository import ChestRepository
from chest_hunter.config import Settings
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ChestHunterError, ValidationError
from chest_hunter.models import STRATEGY_NAMES, SessionStatistics
from chest_hunter.navigation.astar import DistanceOracle, PathFinder
from chest_hunter.navigation.route_planner import Candidate, RoutePlanner
from chest_hunter.run_repository import RunRepository
from chest_hunter.scenarios import scenario_obstacles
from chest_hunter.session_repository import SessionRepository
from chest_hunter.simulation.environment import Environment, World, build_world

if TYPE_CHECKING:
    from chest_hunter.agent.visual_controller import VisualAgentController

logger = logging.getLogger(__name__)

AGENT_KINDS: tuple[str, ...] = ("oracle", "visual", "chest_vision")
"""``oracle``: Phase 2 agent with true spawn points and obstacles.
``visual``: Phase 3 agent; chests, obstacles and pose all come from camera frames.
``chest_vision``: visual chests and pose, but the true obstacle map (benchmark mode B)."""

_TEMPLATE_CACHE: dict[tuple[str, float, float], Any] = {}


def load_world(db: DatabaseManager, settings: Settings, *, map_id: str, map_seed: int) -> World:
    """Build the static world for ``map_id`` from the enabled chest locations in SQLite."""
    locations = ChestRepository(db).get_chests_by_map(map_id, enabled=True)
    if not locations:
        raise ChestHunterError(
            f"Map '{map_id}' has no enabled chest locations. Run "
            "'python -m chest_hunter seed-demo' or add chests first."
        )
    world = build_world(locations, settings.simulation, map_id=map_id, map_seed=map_seed,
                        extra_obstacles=scenario_obstacles(map_id))
    if not world.chests:
        raise ChestHunterError(
            f"None of the chest locations on map '{map_id}' fit inside the "
            f"{world.width}x{world.height} grid; check simulation.world_units_per_cell."
        )
    return world


def make_oracle(settings: Settings) -> DistanceOracle:
    nav = settings.navigation
    return DistanceOracle(allow_diagonal=nav.allow_diagonal,
                          prevent_corner_cutting=nav.prevent_corner_cutting)


@dataclass(frozen=True)
class RunResult:
    session_id: int | None
    outcome: str
    state: AgentState
    end_reason: str | None
    metrics: EpisodeMetrics
    session_statistics: SessionStatistics | None


class SimulationRunner:
    """One autonomous episode, persisted as a session + simulation run."""

    def __init__(
        self,
        db: DatabaseManager,
        settings: Settings,
        *,
        map_id: str | None = None,
        strategy: str | None = None,
        seed: int | None = None,
        episode: int = 0,
        max_ticks: int | None = None,
        world: World | None = None,
        oracle: DistanceOracle | None = None,
        clock_base: datetime | None = None,
        agent: str = "oracle",
        evaluate: bool = False,
    ) -> None:
        sim, nav = settings.simulation, settings.navigation
        self.db = db
        self.settings = settings
        self.map_id = map_id or sim.default_map
        self.agent_kind = agent.replace("-", "_").lower()
        if self.agent_kind not in AGENT_KINDS:
            raise ValidationError(
                f"agent must be one of {', '.join(AGENT_KINDS)}; got {agent!r}.")
        if self.agent_kind != "oracle" and not settings.vision.enabled:
            raise ValidationError(
                f"The {self.agent_kind} agent needs vision.enabled = true in the configuration.")
        self.strategy = (strategy or nav.default_strategy).lower()
        if self.strategy not in STRATEGY_NAMES:
            raise ValidationError(
                f"strategy must be one of {', '.join(STRATEGY_NAMES)}; got {strategy!r}."
            )
        self.route_strategy = self.strategy   # used by the oracle planner
        if self.agent_kind != "oracle":   # label stored with the run
            self.strategy = ("visual" if self.agent_kind == "visual" else "chest-vision") + \
                "-" + settings.visual_navigation.exploration_strategy
        self.evaluate = evaluate
        self.seed = sim.default_seed if seed is None else seed
        if self.seed < 0:
            raise ValidationError("seed must not be negative.")
        if episode < 0:
            raise ValidationError("episode must not be negative.")
        self.episode = episode
        self.max_ticks = sim.max_ticks if max_ticks is None else max_ticks
        if self.max_ticks < 1:
            raise ValidationError("max_ticks must be at least 1.")
        self.world = world if world is not None else load_world(
            db, settings, map_id=self.map_id, map_seed=self.seed)
        if self.world.map_id != self.map_id:
            raise ValidationError(
                f"World is for map '{self.world.map_id}', not '{self.map_id}'.")
        self.env = Environment(self.world, sim, seed=self.seed, episode=episode)
        self.oracle = oracle if oracle is not None else make_oracle(settings)
        self.pathfinder = PathFinder(allow_diagonal=nav.allow_diagonal,
                                     prevent_corner_cutting=nav.prevent_corner_cutting,
                                     cache_paths=nav.cache_paths)
        self.planner = RoutePlanner(self.oracle, strategy=self.route_strategy,
                                    speed=sim.player_speed,
                                    acceleration=sim.player_acceleration,
                                    lookahead_depth=nav.lookahead_depth)
        self.metrics = EpisodeMetrics(strategy=self.strategy, seed=self.seed, episode=episode)
        self.clock = SimulationClock(self.env, base=clock_base)
        self.controller: AgentController | VisualAgentController | None = None
        self.interface: Any = None
        self.vision_evaluator: Any = None
        self.recorder: ObservationRecorder | None = None
        self.session_id: int | None = None
        self.finished = False
        self._cpu_start = 0.0
        self._wall_start = 0.0

    # -- lifecycle ----------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self.controller is not None

    @property
    def state(self) -> Any:
        """Current FSM state (AgentState or VisualAgentState)."""
        return self.controller.state if self.controller else AgentState.INITIALIZING

    @property
    def is_visual(self) -> bool:
        return self.agent_kind != "oracle"

    @property
    def visual_controller(self) -> VisualAgentController:
        """The running visual controller (raises for oracle or unstarted runners)."""
        from chest_hunter.agent.visual_controller import VisualAgentController

        controller = self.controller
        if not isinstance(controller, VisualAgentController):
            raise RuntimeError("This runner is not running a visual agent.")
        return controller

    def start(self) -> int:
        """INITIALIZING: open the session, record the run, load candidates."""
        if self.controller is not None:
            raise RuntimeError("This runner has already been started.")
        self._cpu_start = time.process_time()
        self._wall_start = time.perf_counter()
        sessions = SessionRepository(self.db, clock=self.clock)
        session = sessions.start_session(
            self.map_id,
            notes=f"autonomous agent: strategy={self.strategy} seed={self.seed} "
                  f"episode={self.episode}",
        )
        assert session.id is not None
        self.session_id = session.id
        RunRepository(self.db, clock=self.clock).create_run(
            session.id, strategy=self.strategy, seed=self.seed, episode=self.episode,
            parameters=self._parameters(),
        )
        unvisited = ChestRepository(self.db).get_unvisited_chests(session.id)
        specs = {spec.location_id: spec for spec in self.world.chests}
        candidates = [
            Candidate(loc.id, specs[loc.id].cell, (specs[loc.id].x, specs[loc.id].y),
                      loc.spawn_probability, loc.estimated_open_seconds)
            for loc in unvisited if loc.id is not None and loc.id in specs
        ]
        self.recorder = ObservationRecorder(
            self.db, sessions, session.id,
            batch_size=self.settings.agent.observation_batch_size,
        )
        self.metrics.chests_spawned = sum(
            1 for c in candidates if self.env.chests[c.chest_id].present)  # evaluation only
        if self.is_visual:
            visual = self._build_visual_controller(unvisited)
            self.controller = visual
            logger.info("Session %d: %s agent, seed %d, episode %d, %d spawn prior(s)",
                        session.id, self.agent_kind, self.seed, self.episode, len(candidates))
            visual.initialize()
            return session.id
        self.controller = AgentController(
            self.env, planner=self.planner, pathfinder=self.pathfinder,
            settings=self.settings, metrics=self.metrics, recorder=self.recorder,
            clock=self.clock, finalizer=self._finalize,
        )
        logger.info("Session %d: %s strategy, seed %d, episode %d, %d candidate location(s)",
                    session.id, self.strategy, self.seed, self.episode, len(candidates))
        self.controller.initialize(candidates)
        return session.id

    def step(self, ticks: int = 1) -> bool:
        """Advance up to ``ticks`` fixed steps; return False once the episode has ended."""
        if self.controller is None:
            self.start()
        controller = self.controller
        assert controller is not None
        for _ in range(ticks):
            if self.finished:
                break
            controller.tick()
            if controller.is_terminal():
                break
            if self.interface is not None:
                self.interface.end_tick()
            self.env.advance()
            if self.env.tick >= self.max_ticks:
                controller.stop("tick_limit", f"reached max_ticks={self.max_ticks}")
        return not self.finished

    def run(self) -> RunResult:
        """Run headless until the agent completes, fails or hits ``max_ticks``."""
        while self.step(2048):
            pass
        return self.result()

    def cancel(self, reason: str = "cancelled by user") -> None:
        """Stop a running episode and abort its session (no-op if not running)."""
        if self.controller is not None and not self.finished:
            self.controller.stop("cancelled", reason)

    def result(self) -> RunResult:
        stats = None
        if self.session_id is not None:
            stats = SessionRepository(self.db).get_session_statistics(self.session_id)
        controller = self.controller
        return RunResult(
            session_id=self.session_id,
            outcome=self.metrics.outcome or "running",
            state=self.state,
            end_reason=controller.end_reason if controller else None,
            metrics=self.metrics,
            session_statistics=stats,
        )

    # -- internals ----------------------------------------------------------------

    def _parameters(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "map_seed": self.world.map_seed,
            "max_ticks": self.max_ticks,
            "grid_blocked_fraction": round(self.world.grid.blocked_fraction, 6),
            "chest_locations_on_grid": len(self.world.chests),
            "excluded_location_ids": list(self.world.excluded_location_ids),
            "time_basis": "simulated",
            "simulation": asdict(self.settings.simulation),
            "navigation": asdict(self.settings.navigation),
            "agent": asdict(self.settings.agent),
            "agent_kind": self.agent_kind,
            **({"vision": asdict(self.settings.vision),
                "localization": asdict(self.settings.localization),
                "visual_navigation": asdict(self.settings.visual_navigation)}
               if self.is_visual else {}),
        }

    def _build_visual_controller(self, unvisited: list[Any]) -> VisualAgentController:
        """Wire the Phase 3 stack. Only the interface touches the environment."""
        from chest_hunter.agent.visual_controller import VisualAgentController
        from chest_hunter.localization.pose_estimator import PoseEstimator
        from chest_hunter.localization.world_state import SpawnPrior, WorldModel
        from chest_hunter.navigation.local_controller import EstimatedBody, LocalController
        from chest_hunter.navigation.movement import MovementController
        from chest_hunter.navigation.visual_planner import VisualPlanner
        from chest_hunter.simulation.camera import SensorModel
        from chest_hunter.simulation.interface import VisualSimulatorInterface
        from chest_hunter.vision.perception_pipeline import PerceptionPipeline
        from chest_hunter.vision.templates import TemplateLibrary
        from chest_hunter.vision.visual_tracker import VisualTracker

        settings = self.settings
        sim, vis, loc, vnav = (settings.simulation, settings.vision, settings.localization,
                               settings.visual_navigation)
        self.interface = VisualSimulatorInterface(
            self.env, width=vis.default_width, height=vis.default_height,
            view_width_cells=vis.view_width_cells,
            sensor=SensorModel(vis.noise_std, vis.lighting_variation, vis.frame_dropout),
            seed=self.seed * 1000 + self.episode)
        key = (str(settings.template_dir()), vis.noise_std, vis.lighting_variation)
        library = _TEMPLATE_CACHE.get(key)
        if library is None:
            library = TemplateLibrary.load(settings.template_dir(), noise_std=vis.noise_std,
                                           lighting_variation=vis.lighting_variation)
            _TEMPLATE_CACHE[key] = library
        pipeline = PerceptionPipeline(library, detector=vis.detector,
                                      confidence_threshold=vis.confidence_threshold,
                                      update_occupancy=self.agent_kind == "visual")
        transform = self.world.transform
        priors = []
        for loc_row in unvisited:
            if loc_row.id is None or loc_row.id in self.world.excluded_location_ids:
                continue
            gx, gy = transform.to_grid(loc_row.x, loc_row.y)
            priors.append(SpawnPrior(loc_row.id, loc_row.location_name, gx, gy,
                                     loc_row.spawn_probability))
        pose = PoseEstimator(smoothing_alpha=loc.smoothing_alpha,
                             max_missing_observations=loc.max_missing_observations,
                             dead_reckoning_seconds=loc.dead_reckoning_seconds)
        tracker = VisualTracker(expiration_seconds=vis.track_expiration_seconds,
                                memory_seconds=vis.track_memory_seconds,
                                max_tracks=vis.max_tracks)
        oracle_grid = self.env.grid if self.agent_kind == "chest_vision" else None
        world = WorldModel(self.world.width, self.world.height, priors, pose=pose,
                           tracker=tracker, min_confidence=loc.min_confidence,
                           empty_confirmations=vnav.empty_confirmations,
                           oracle_grid=oracle_grid)
        ppc = vis.default_width / vis.view_width_cells
        planner = VisualPlanner(world, speed=sim.player_speed,
                                acceleration=sim.player_acceleration,
                                view_width=vis.view_width_cells,
                                view_height=vis.default_height / ppc,
                                open_seconds=sim.chest_interaction_seconds,
                                prior_weight=vnav.prior_weight,
                                use_priors=vnav.use_spawn_priors,
                                unknown_cost=vnav.unknown_cell_cost,
                                allow_diagonal=settings.navigation.allow_diagonal,
                                prevent_corner_cutting=settings.navigation.prevent_corner_cutting)
        body = EstimatedBody(pose, self.interface, max_speed=sim.player_speed,
                             acceleration=sim.player_acceleration, radius=sim.player_radius)
        local = LocalController(body, MovementController(
            dt=self.env.dt, stuck_timeout=settings.navigation.stuck_timeout_seconds),
            min_confidence=loc.min_confidence)
        controller = VisualAgentController(
            self.interface, pipeline=pipeline, world=world, planner=planner, local=local,
            settings=settings, metrics=self.metrics, recorder=self.recorder, clock=self.clock,
            finalizer=self._finalize)
        if self.evaluate:
            from chest_hunter.analytics.vision_metrics import VisionEvaluator

            self.vision_evaluator = VisionEvaluator(self.interface.evaluation_environment,
                                                    controller)
            controller.perception_listeners.append(self.vision_evaluator.observe)
        return controller

    def _finalize(self, controller: StateMachine[Any]) -> None:
        """Called once when the FSM reaches COMPLETED or FAILED."""
        m = self.metrics
        m.ticks = self.env.tick
        m.simulated_seconds = self.env.time
        m.travel_distance = self.env.agent.distance_travelled
        pathfinder = controller.planner.pathfinder if self.is_visual else self.pathfinder  # type: ignore[attr-defined]
        m.astar_searches = pathfinder.searches
        m.astar_cache_hits = pathfinder.cache_hits
        m.outcome = controller.outcome or (
            "failed" if str(controller.state.value) == "FAILED" else "completed")
        if self.is_visual:
            m.rewards = len(self.env.rewards.awarded)  # ground truth for reporting
            if self.vision_evaluator is not None:
                self.vision_evaluator.finish()
        if self.recorder is not None:
            m.observations_recorded = self.recorder.recorded
            m.db_flushes = self.recorder.flushes
        assert self.session_id is not None
        sessions = SessionRepository(self.db, clock=self.clock)
        if m.outcome in ("completed", "tick_limit"):
            sessions.complete_session(self.session_id)
        else:
            sessions.abort_session(self.session_id)
        RunRepository(self.db, clock=self.clock).finish_run(
            self.session_id,
            outcome=m.outcome,
            failure_reason=controller.end_reason if m.outcome in ("failed", "cancelled") else None,
            ticks=m.ticks,
            simulated_seconds=m.simulated_seconds,
            travel_distance=m.travel_distance,
            recoveries=m.recoveries,
            unreachable_targets=m.unreachable_targets,
            abandoned_targets=m.abandoned_targets,
            rewards=m.rewards,
        )
        m.cpu_seconds = time.process_time() - self._cpu_start
        m.wall_seconds = time.perf_counter() - self._wall_start
        self.finished = True
        logger.info("Session %d finished: %s (%s); opened %d chest(s) in %.1f simulated s",
                    self.session_id, m.outcome, controller.end_reason, m.chests_opened,
                    m.simulated_seconds)
