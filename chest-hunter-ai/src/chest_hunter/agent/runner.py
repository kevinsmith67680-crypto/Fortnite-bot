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
from typing import Any

from chest_hunter.agent.controller import AgentController
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
from chest_hunter.session_repository import SessionRepository
from chest_hunter.simulation.environment import Environment, World, build_world

logger = logging.getLogger(__name__)


def load_world(db: DatabaseManager, settings: Settings, *, map_id: str, map_seed: int) -> World:
    """Build the static world for ``map_id`` from the enabled chest locations in SQLite."""
    locations = ChestRepository(db).get_chests_by_map(map_id, enabled=True)
    if not locations:
        raise ChestHunterError(
            f"Map '{map_id}' has no enabled chest locations. Run "
            "'python -m chest_hunter seed-demo' or add chests first."
        )
    world = build_world(locations, settings.simulation, map_id=map_id, map_seed=map_seed)
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
    ) -> None:
        sim, nav = settings.simulation, settings.navigation
        self.db = db
        self.settings = settings
        self.map_id = map_id or sim.default_map
        self.strategy = (strategy or nav.default_strategy).lower()
        if self.strategy not in STRATEGY_NAMES:
            raise ValidationError(
                f"strategy must be one of {', '.join(STRATEGY_NAMES)}; got {strategy!r}."
            )
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
        self.planner = RoutePlanner(self.oracle, strategy=self.strategy,
                                    speed=sim.player_speed,
                                    acceleration=sim.player_acceleration,
                                    lookahead_depth=nav.lookahead_depth)
        self.metrics = EpisodeMetrics(strategy=self.strategy, seed=self.seed, episode=episode)
        self.clock = SimulationClock(self.env, base=clock_base)
        self.controller: AgentController | None = None
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
    def state(self) -> AgentState:
        return self.controller.state if self.controller else AgentState.INITIALIZING

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
            if controller.state.is_terminal:
                break
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
        }

    def _finalize(self, controller: AgentController) -> None:
        """Called once when the FSM reaches COMPLETED or FAILED."""
        m = self.metrics
        m.ticks = self.env.tick
        m.simulated_seconds = self.env.time
        m.travel_distance = self.env.agent.distance_travelled
        m.astar_searches = self.pathfinder.searches
        m.astar_cache_hits = self.pathfinder.cache_hits
        m.outcome = controller.outcome or ("completed" if controller.state is AgentState.COMPLETED
                                           else "failed")
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
