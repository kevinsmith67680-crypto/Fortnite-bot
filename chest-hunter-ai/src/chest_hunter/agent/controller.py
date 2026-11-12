"""The autonomous chest-hunting agent: a finite-state machine.

One call to :meth:`AgentController.tick` handles one fixed simulation step.
Instant states (selecting, planning, verifying, confirming, recovering) may
chain within a tick; timed states (NAVIGATING, INTERACTING) advance the
agent's physics and end the tick. A cap on transitions per tick and a bounded
recovery budget guarantee the machine can never loop forever.

State flow::

    INITIALIZING -> SELECTING_TARGET -> PLANNING_ROUTE -> NAVIGATING
        -> VERIFYING_CHEST -(absent)-> SELECTING_TARGET
                           -(present)-> INTERACTING -> CONFIRMING_OPEN
                                                    -> SELECTING_TARGET
    any problem -> RECOVERING -> (retry | replan | abandon target)
    no targets left -> COMPLETED;   unrecoverable -> FAILED

Persistence: every visit outcome becomes a Phase 1 ``chest_observations``
row (via a batched recorder). Unreachable locations are *not* recorded as
observations, because nothing was observed; they are counted in the run
metrics and reported instead.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from chest_hunter.agent.objectives import ObjectiveTracker
from chest_hunter.agent.recorder import ObservationRecorder, SimulationClock
from chest_hunter.agent.states import AgentState, is_valid_transition
from chest_hunter.analytics.metrics import EpisodeMetrics
from chest_hunter.config import Settings
from chest_hunter.navigation.astar import FailureReason, PathFinder
from chest_hunter.navigation.movement import MovementController, MoveStatus, build_waypoints
from chest_hunter.navigation.route_planner import Candidate, RoutePlanner, TargetSelection
from chest_hunter.simulation.environment import Environment, InteractionResult

logger = logging.getLogger(__name__)

MAX_TRANSITIONS_PER_TICK = 16
MAX_TIGHTEN_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class Transition:
    tick: int
    source: AgentState
    target: AgentState
    reason: str | None


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    tick: int
    chest_id: int
    present: bool
    opened: bool
    interaction_seconds: float


class AgentController:
    """Drives one :class:`Environment` episode without human input."""

    def __init__(
        self,
        env: Environment,
        *,
        planner: RoutePlanner,
        pathfinder: PathFinder,
        settings: Settings,
        metrics: EpisodeMetrics,
        recorder: ObservationRecorder | None = None,
        clock: SimulationClock | None = None,
        finalizer: Callable[[AgentController], None] | None = None,
    ) -> None:
        self.env = env
        self.planner = planner
        self.pathfinder = pathfinder
        self.settings = settings
        self.metrics = metrics
        self.recorder = recorder
        self.clock = clock if clock is not None else SimulationClock(env)
        self._finalizer = finalizer

        self.state = AgentState.INITIALIZING
        self.target: Candidate | None = None
        self.last_selection: TargetSelection | None = None
        self.tracker = ObjectiveTracker([])
        self.movement = MovementController(
            dt=env.dt, stuck_timeout=settings.navigation.stuck_timeout_seconds
        )
        self.recovery_reason: str | None = None
        self.outcome: str | None = None
        self.end_reason: str | None = None
        self.history: deque[Transition] = deque(maxlen=1000)
        self.transition_counts: Counter[AgentState] = Counter()
        self.observations: list[ObservationEvent] = []
        self.recovery_budget = 10
        self.routes_planned = 0

        self._handlers: dict[AgentState, Callable[[], AgentState | None]] = {
            AgentState.SELECTING_TARGET: self._select_target,
            AgentState.PLANNING_ROUTE: self._plan_route,
            AgentState.NAVIGATING: self._navigate,
            AgentState.VERIFYING_CHEST: self._verify_chest,
            AgentState.INTERACTING: self._interact,
            AgentState.CONFIRMING_OPEN: self._confirm_open,
            AgentState.RECOVERING: self._recover,
        }
        self._interaction_ticks = 0
        self._interaction_result: InteractionResult | None = None
        self._path_grid = env.grid
        self._path_version = env.grid.version
        self._found: set[int] = set()
        self._recorded: set[int] = set()
        self._total_recoveries = 0
        self._tighten_count = 0

    # -- public API -----------------------------------------------------------

    def initialize(self, candidates: Sequence[Candidate]) -> None:
        """INITIALIZING: load the session's eligible targets and start selecting."""
        if self.state is not AgentState.INITIALIZING:
            raise RuntimeError("The controller has already been initialised.")
        self.tracker = ObjectiveTracker(candidates)
        self.metrics.candidate_locations = len(self.tracker)
        self.metrics.chests_spawned = sum(
            1 for c in candidates if self.env.chests[c.chest_id].present
        )
        max_attempts = self.settings.agent.max_recovery_attempts
        self.recovery_budget = max(10, (max_attempts + 1) * max(1, len(self.tracker)))
        self._transition(AgentState.SELECTING_TARGET, f"{len(self.tracker)} candidate(s)")

    def tick(self) -> AgentState:
        """Process one simulation step and return the resulting state."""
        consumed = False
        for _ in range(MAX_TRANSITIONS_PER_TICK):
            state = self.state
            if state.is_terminal or state is AgentState.INITIALIZING:
                return state
            if state.consumes_time and consumed:
                return state
            next_state = self._handlers[state]()
            if state.consumes_time:
                consumed = True
            if next_state is None:
                return self.state
            self._transition(next_state, self._reason_for(next_state))
            if self.state.is_terminal:
                return self.state
        self.outcome = "failed"
        self.end_reason = "state machine did not settle within one tick"
        self._transition(AgentState.FAILED, self.end_reason)
        return self.state

    def stop(self, outcome: str, reason: str) -> None:
        """End the episode from outside (tick limit, user cancel)."""
        if self.state.is_terminal:
            return
        self.outcome = outcome
        self.end_reason = reason
        ends_normally = outcome in ("completed", "tick_limit")
        self._transition(AgentState.COMPLETED if ends_normally else AgentState.FAILED, reason)

    @property
    def path(self) -> list[tuple[float, float]]:
        """Remaining navigation waypoints, for rendering."""
        return self.movement.remaining_waypoints() if self.state is AgentState.NAVIGATING else []

    # -- state handlers ---------------------------------------------------------

    def _select_target(self) -> AgentState:
        agent = self.env.agent
        if self.env.grid.is_blocked(*agent.cell):
            return self._fail("the agent is inside an obstacle")
        pending = self.tracker.pending()
        if not pending:
            return self._complete("all reachable chest locations visited")
        selection = self.planner.select(self.env.grid, agent.cell, pending)
        self.metrics.target_selections += 1
        unreachable = selection.unreachable if selection else tuple(c.chest_id for c in pending)
        for chest_id in unreachable:
            self._mark_unreachable(chest_id)
        if selection is None:
            return self._complete("no reachable chest locations remain")
        self.last_selection = selection
        self.target = selection.target
        agent.current_target = selection.target.chest_id
        self._tighten_count = 0
        return AgentState.PLANNING_ROUTE

    def _plan_route(self) -> AgentState:
        target = self._require_target()
        agent = self.env.agent
        if self.env.can_interact(target.chest_id):
            self.movement.clear()
            return AgentState.VERIFYING_CHEST
        result = self.pathfinder.find_path(self.env.grid, agent.cell, target.cell)
        if not result.found:
            if result.reason == FailureReason.START_BLOCKED:
                self.recovery_reason = "start_blocked"
                return AgentState.RECOVERING
            self._mark_unreachable(target.chest_id)
            self._clear_target()
            return AgentState.SELECTING_TARGET
        sim = self.settings.simulation
        waypoints = build_waypoints(
            self.env.grid, result.path, agent.position, target.position,
            radius=agent.radius, smooth=self.settings.navigation.smooth_paths,
        )
        self.movement.set_path(
            waypoints,
            goal=target.position,
            stop_radius=max(0.0, sim.interaction_radius - self.settings.agent.arrival_tolerance),
            arrive_radius=sim.interaction_radius,
            time=self.env.time,
        )
        self._path_grid = self.env.grid
        self._path_version = self.env.grid.version
        self.routes_planned += 1
        return AgentState.NAVIGATING

    def _navigate(self) -> AgentState | None:
        target = self._require_target()
        grid = self.env.grid
        if grid is not self._path_grid or grid.version != self._path_version:
            if not self.movement.path_is_clear(grid, self.env.agent):
                self.recovery_reason = "path_blocked"
                return AgentState.RECOVERING
            self._path_grid, self._path_version = grid, grid.version
        status = self.movement.update(self.env.agent, grid, self.env.time)
        if status is MoveStatus.ARRIVED:
            if self.env.can_interact(target.chest_id):
                return AgentState.VERIFYING_CHEST
            self._tighten_count += 1
            if self._tighten_count > MAX_TIGHTEN_ATTEMPTS:
                self.recovery_reason = "no_line_of_sight"
                return AgentState.RECOVERING
            self.movement.tighten()
            return None
        if status is MoveStatus.STUCK:
            self.recovery_reason = "stuck"
            return AgentState.RECOVERING
        if status is MoveStatus.IDLE:
            self.recovery_reason = "no_path"
            return AgentState.RECOVERING
        return None

    def _verify_chest(self) -> AgentState:
        target = self._require_target()
        present = self.env.observe_chest(target.chest_id)
        if present is None:
            self.recovery_reason = "out_of_range"
            return AgentState.RECOVERING
        if not present:
            self._record(target.chest_id, present=False, opened=False, seconds=0.0)
            self.tracker.mark_visited(target.chest_id)
            self._clear_target()
            return AgentState.SELECTING_TARGET
        if target.chest_id not in self._found:
            self._found.add(target.chest_id)
            self.metrics.chests_found += 1
        if self.env.chests[target.chest_id].opened:  # defensive: never opened twice
            self.tracker.mark_visited(target.chest_id)
            self._clear_target()
            return AgentState.SELECTING_TARGET
        self._interaction_ticks = 0
        self._interaction_result = None
        return AgentState.INTERACTING

    def _interact(self) -> AgentState | None:
        target = self._require_target()
        agent = self.env.agent
        agent.brake(self.env.dt)
        agent.step(self.env.grid, self.env.dt)
        if not self.env.can_interact(target.chest_id):
            self.recovery_reason = "left_interaction_range"
            return AgentState.RECOVERING
        self._interaction_ticks += 1
        elapsed = self._interaction_ticks * self.env.dt
        if elapsed + 1e-9 < self.settings.simulation.chest_interaction_seconds:
            return None
        self._interaction_result = self.env.try_open(target.chest_id, elapsed)
        return AgentState.CONFIRMING_OPEN

    def _confirm_open(self) -> AgentState:
        target = self._require_target()
        chest_id = target.chest_id
        seconds = self._interaction_ticks * self.env.dt
        result = self._interaction_result
        self._interaction_result = None
        if result is InteractionResult.OPENED and self.env.chests[chest_id].opened:
            self._record(chest_id, present=True, opened=True, seconds=seconds)
            self.metrics.chests_opened += 1
            if self.env.grant_reward(chest_id):
                self.metrics.rewards += 1
            self.tracker.mark_visited(chest_id)
            self._clear_target()
            return AgentState.SELECTING_TARGET
        if result in (InteractionResult.FAILED, InteractionResult.INCOMPLETE,
                      InteractionResult.OUT_OF_RANGE):
            self._record(chest_id, present=True, opened=False, seconds=seconds)
            self.metrics.failed_interactions += 1
            self.recovery_reason = "interaction_failed"
            return AgentState.RECOVERING
        logger.warning("Unexpected interaction result %s for chest %d", result, chest_id)
        self.tracker.mark_visited(chest_id)
        self._clear_target()
        return AgentState.SELECTING_TARGET

    def _recover(self) -> AgentState:
        self.metrics.recoveries += 1
        self._total_recoveries += 1
        reason = self.recovery_reason or "unknown"
        self.recovery_reason = None
        if self._total_recoveries > self.recovery_budget:
            return self._fail(f"recovery budget of {self.recovery_budget} exhausted "
                              f"(last problem: {reason})")
        target = self.target
        if target is None:
            return AgentState.SELECTING_TARGET
        attempts = self.tracker.record_attempt(target.chest_id)
        if attempts > self.settings.agent.max_recovery_attempts:
            logger.debug("Abandoning chest %d after %d recovery attempts (%s)",
                         target.chest_id, attempts - 1, reason)
            self.tracker.mark_abandoned(target.chest_id)
            self.metrics.abandoned_targets += 1
            self.movement.clear()
            self._clear_target()
            return AgentState.SELECTING_TARGET
        if reason == "interaction_failed":
            self._interaction_ticks = 0
            return AgentState.INTERACTING
        # Physical problems: replan from where the agent is now. The new path
        # starts at the agent's position and current cell centre (which also
        # frees it from wall corners); speed changes stay within the limits.
        self.movement.clear()
        return AgentState.PLANNING_ROUTE

    # -- helpers ------------------------------------------------------------------

    def _require_target(self) -> Candidate:
        if self.target is None:  # pragma: no cover - guarded by the transition table
            raise RuntimeError(f"{self.state} requires a target")
        return self.target

    def _clear_target(self) -> None:
        self.target = None
        self.env.agent.current_target = None
        self.movement.clear()

    def _mark_unreachable(self, chest_id: int) -> None:
        self.tracker.mark_unreachable(chest_id)
        self.metrics.unreachable_targets += 1
        logger.debug("Chest location %d is unreachable from the agent's position", chest_id)

    def _record(self, chest_id: int, *, present: bool, opened: bool, seconds: float) -> None:
        if chest_id not in self._recorded:
            self._recorded.add(chest_id)
            self.metrics.locations_visited += 1
        self.observations.append(
            ObservationEvent(self.env.tick, chest_id, present, opened, seconds)
        )
        if self.recorder is not None:
            self.recorder.record(chest_id, present=present, opened=opened,
                                 interaction_seconds=seconds,
                                 observed_at=self.clock.at(self.env.time))

    def _complete(self, reason: str) -> AgentState:
        self.outcome = "completed"
        self.end_reason = reason
        return AgentState.COMPLETED

    def _fail(self, reason: str) -> AgentState:
        self.outcome = "failed"
        self.end_reason = reason
        return AgentState.FAILED

    def _reason_for(self, state: AgentState) -> str | None:
        if state is AgentState.RECOVERING:
            return self.recovery_reason
        if state.is_terminal:
            return self.end_reason
        return None

    def _transition(self, target: AgentState, reason: str | None = None) -> None:
        if not is_valid_transition(self.state, target):
            raise RuntimeError(f"Invalid agent transition {self.state} -> {target}")
        self.history.append(Transition(self.env.tick, self.state, target, reason))
        self.transition_counts[target] += 1
        logger.debug("tick %d: %s -> %s%s", self.env.tick, self.state, target,
                     f" ({reason})" if reason else "")
        self.state = target
        if target.is_terminal:
            self._on_terminal()

    def _on_terminal(self) -> None:
        self.env.agent.stop()
        self.env.agent.current_target = None
        self.movement.clear()
        self.target = None
        if self.recorder is not None:
            self.recorder.flush()
        if self.state is AgentState.FAILED:
            self.metrics.failure_reason = self.end_reason
            if self.outcome == "cancelled":
                logger.info("Episode cancelled: %s", self.end_reason)
            else:
                logger.warning("Agent failed: %s", self.end_reason)
        if self._finalizer is not None:
            self._finalizer(self)
