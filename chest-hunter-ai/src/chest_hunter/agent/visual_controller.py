"""The Phase 3 vision-based policy: OBSERVE -> DETECT -> TRACK -> PLAN -> ACT -> VERIFY.

Built on the same :class:`~chest_hunter.agent.fsm.StateMachine` engine as the
Phase 2 oracle controller (a second policy, not a second engine).

Information access: this controller receives a :class:`SimulatorPort`
(camera frames, velocity commands, the interact key, map size and time),
the estimated :class:`WorldModel`, database spawn *priors* and the
configuration (game rules such as interaction radius and duration). It never
receives the environment, chest objects or the true obstacle grid; an import
audit and a hidden-chest test in the test suite check this.

Each tick: (1) if a visual analysis is due under the adaptive schedule,
capture a frame, run perception and update the world model; (2) run the FSM
(at most one timed state per tick); (3) the runner advances the simulator.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Protocol

import numpy as np

from chest_hunter.agent.fsm import StateMachine
from chest_hunter.agent.recorder import ObservationRecorder, SimulationClock
from chest_hunter.agent.states import (
    VISUAL_TIMED_STATES,
    VISUAL_TRANSITIONS,
    VisualAgentState,
)
from chest_hunter.analytics.metrics import EpisodeMetrics
from chest_hunter.config import Settings
from chest_hunter.localization.world_state import SpawnStatus, WorldModel
from chest_hunter.navigation.local_controller import LocalController
from chest_hunter.navigation.movement import MoveStatus, build_waypoints
from chest_hunter.navigation.visual_planner import Goal, VisualPlanner
from chest_hunter.vision.observation import VisualObservation
from chest_hunter.vision.obstacle_detector import UNKNOWN
from chest_hunter.vision.perception_pipeline import PerceptionPipeline, PerceptionResult

logger = logging.getLogger(__name__)

S = VisualAgentState
MATCH_RADIUS = 0.8          # cells: detection <-> target association during verification
ALIGN_SECONDS = 0.5         # wait for the interaction prompt before moving closer
VERIFY_SECONDS = 0.6        # time to observe the result of an interaction
MAX_ALIGN_APPROACHES = 2
MAX_INSPECTIONS = 2
MAX_UNPRODUCTIVE_GOALS = 20  # exploration goals in a row that reveal nothing new
APPROACH_DISTANCE = 1.0      # cells between the standing point and the chest
# Standing directions (unit vectors). North is excluded: the interaction prompt
# is drawn above (north of) the chest and the agent's own sprite would hide it.
APPROACH_DIRECTIONS = ((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0),
                       (0.7071, -0.7071), (-0.7071, -0.7071))


class SimulatorPort(Protocol):
    """What the visual policy may use (implemented by VisualSimulatorInterface)."""

    @property
    def map_width(self) -> int: ...
    @property
    def map_height(self) -> int: ...
    @property
    def dt(self) -> float: ...
    @property
    def time(self) -> float: ...
    def capture(self) -> VisualObservation | None: ...
    def move(self, vx: float, vy: float) -> None: ...
    def hold_interact(self) -> None: ...


class VisualAgentController(StateMachine[VisualAgentState]):
    def __init__(
        self,
        port: SimulatorPort,
        *,
        pipeline: PerceptionPipeline,
        world: WorldModel,
        planner: VisualPlanner,
        local: LocalController,
        settings: Settings,
        metrics: EpisodeMetrics,
        recorder: ObservationRecorder | None = None,
        clock: SimulationClock | None = None,
        finalizer: Callable[[VisualAgentController], None] | None = None,
    ) -> None:
        self._ticks = 0
        super().__init__(initial=S.INITIALIZING, completed=S.COMPLETED, failed=S.FAILED,
                         transitions=VISUAL_TRANSITIONS, timed=VISUAL_TIMED_STATES,
                         tick_source=lambda: self._ticks, recovering=S.RECOVERING)
        self.port = port
        self.pipeline = pipeline
        self.world = world
        self.planner = planner
        self.local = local
        self.settings = settings
        self.metrics = metrics
        self.recorder = recorder
        self.clock = clock
        self._finalizer = finalizer
        self.dt = port.dt
        sim, vis, vnav = settings.simulation, settings.vision, settings.visual_navigation
        self.interaction_seconds = sim.chest_interaction_seconds
        self.interaction_radius = sim.interaction_radius
        self.stop_radius = max(0.0, sim.interaction_radius - settings.agent.arrival_tolerance)
        self.min_confidence = settings.localization.min_confidence
        self.max_attempts = vnav.maximum_recovery_attempts
        self.rates = {"travel": vis.rate_travel, "explore": vis.rate_explore,
                      "tracking": vis.rate_tracking, "verify": vis.rate_verify,
                      "idle": vis.rate_idle}
        self.tick_rate = round(1.0 / self.dt)

        self.goal: Goal | None = None
        self.last_result: PerceptionResult | None = None
        self.perception_listeners: list[Callable[[PerceptionResult], None]] = []
        self.analyses = 0
        self.failed_frames = 0
        self.replans = 0
        self.routes_planned = 0
        self.low_confidence_pauses = 0
        self.targets_selected = 0
        self.targets_reached = 0
        self.targets_lost = 0
        self.recorded_locations: set[int] = set()
        self._seen_chest_priors: set[int] = set()
        self._ticks_since_analysis = 10**9
        self._force_observe = True
        self._analysis_tick = -1
        self._new_target = False
        self._known_targets: set[int] = set()
        self._near_unknown = True
        self._replan_requested = False
        self._attempts: dict[object, int] = {}
        # Spawn points where a full hold completed but no opened chest was seen yet
        # (location id -> hold seconds). Credited later only on visual evidence.
        self._unverified: dict[int, float] = {}
        self.late_verifications = 0
        self._inspections: dict[int, int] = {}
        self._total_recoveries = 0
        self.recovery_budget = 20
        self._phase_start = 0.0
        self._phase_analyses = 0
        self._approaches = 0
        self._hold_ticks = 0
        self._loc_failures = 0
        self._unproductive_goals = 0
        self._knowledge_at_goal: tuple[int, int] = (-1, -1)
        self._observe_needed = 1
        self._handlers = {
            S.OBSERVING: self._observe,
            S.LOCALIZING: self._localize,
            S.EXPLORING: self._explore,
            S.TARGET_DETECTED: self._target_detected,
            S.PLANNING_ROUTE: self._plan_route,
            S.NAVIGATING: self._navigate,
            S.ALIGNING: self._align,
            S.INTERACTING: self._interact,
            S.VERIFYING_OPEN: self._verify_open,
            S.RECOVERING: self._recover,
        }

    # -- public API -----------------------------------------------------------------

    def initialize(self) -> None:
        if self.state is not S.INITIALIZING:
            raise RuntimeError("The controller has already been initialised.")
        self.metrics.candidate_locations = len(self.world.priors)
        self.recovery_budget = max(20, (self.max_attempts + 1) * (len(self.world.priors) + 5))
        self._enter_phase()
        self._transition(S.OBSERVING, f"{len(self.world.priors)} spawn prior(s)")

    def tick(self) -> VisualAgentState:
        if not self.is_terminal() and self.state is not S.INITIALIZING:
            self._maybe_perceive()
        state = super().tick()
        self._ticks += 1
        return state

    @property
    def path(self) -> list[tuple[float, float]]:
        if self.state is S.NAVIGATING:
            return self.local.movement.remaining_waypoints()
        return []

    @property
    def time(self) -> float:
        return self.port.time

    # -- perception schedule ------------------------------------------------------------

    def _situation(self) -> str:
        state = self.state
        if self.is_terminal():
            return "idle"
        if state in (S.OBSERVING, S.LOCALIZING, S.ALIGNING, S.INTERACTING, S.VERIFYING_OPEN):
            return "verify"
        if self.goal is not None and self.goal.kind == "chest":
            return "tracking"
        return "explore" if self._near_unknown else "travel"

    def _maybe_perceive(self) -> None:
        self._ticks_since_analysis += 1
        rate = self.rates[self._situation()]
        due = self._force_observe or (
            rate > 0 and self._ticks_since_analysis >= self.tick_rate / rate)
        if not due:
            return
        self._force_observe = False
        self._ticks_since_analysis = 0
        observation = self.port.capture()
        if observation is None:
            self.failed_frames += 1
            self.world.pose.miss(self.port.time)
            return
        hint = None   # region of interest for the marker search (full frame if None)
        if self.settings.vision.use_regions_of_interest and self.world.pose.position is not None:
            hint = observation.camera.world_to_pixel(*self.world.pose.position)
        result = self.pipeline.process(observation, self.world.occupancy, hint)
        update = self.world.integrate(result)
        self.last_result = result
        self.analyses += 1
        self._phase_analyses += 1
        self._analysis_tick = self._ticks
        for prior in update.newly_empty:
            self._record(prior.location_id, present=False, opened=False, seconds=0.0)
        self._reconcile_unverified(result)
        for prior in self.world.priors.values():
            if prior.status is SpawnStatus.CHEST_SEEN and \
                    prior.location_id not in self._seen_chest_priors:
                self._seen_chest_priors.add(prior.location_id)
                self.metrics.chests_found += 1
        targets = {t.track_id for t in self.world.chest_targets()}
        if targets - self._known_targets:
            self._new_target = True
        self._known_targets |= targets
        self._near_unknown = self._unknown_nearby(observation)
        if update.obstacles_discovered and self.state is S.NAVIGATING \
                and self.settings.visual_navigation.replan_on_obstacle_discovery \
                and not self.local.movement.path_is_clear(self.world.planning_grid,
                                                          self.local.body):  # type: ignore[arg-type]
            self._replan_requested = True
        for listener in self.perception_listeners:
            listener(result)

    def _unknown_nearby(self, observation: VisualObservation) -> bool:
        if self.world.pose.position is None:
            return True
        cam = observation.camera
        x, y = self.world.pose.position
        occ = self.world.occupancy.cells
        x0 = max(0, int(x - cam.view_width / 2 - 4))
        x1 = min(occ.shape[1], int(x + cam.view_width / 2 + 5))
        y0 = max(0, int(y - cam.view_height / 2 - 4))
        y1 = min(occ.shape[0], int(y + cam.view_height / 2 + 5))
        return bool(np.any(occ[y0:y1, x0:x1] == UNKNOWN))

    def _fresh_analysis(self) -> bool:
        return self._analysis_tick == self._ticks and self.last_result is not None

    # -- state handlers ------------------------------------------------------------------

    def _enter_phase(self) -> None:
        self._phase_start = self.port.time
        self._phase_analyses = 0

    def _go(self, state: VisualAgentState) -> VisualAgentState:
        """Return ``state`` after resetting per-phase counters for timed states."""
        if state in VISUAL_TIMED_STATES and state is not self.state:
            self._enter_phase()
            if state in (S.OBSERVING, S.ALIGNING, S.VERIFYING_OPEN):
                self._force_observe = True
        return state

    def _observe(self) -> VisualAgentState | None:
        self.local.hold(self.dt)
        if self._phase_analyses < self._observe_needed and \
                self.port.time - self._phase_start < 1.0 + 0.2 * self._observe_needed:
            return None   # wait (bounded) for enough frames
        self._observe_needed = 1
        return S.LOCALIZING

    def _localize(self) -> VisualAgentState:
        pose = self.world.pose
        if pose.initialized and not pose.is_lost and pose.confidence() >= self.min_confidence:
            self._loc_failures = 0
            return S.TARGET_DETECTED if self.world.chest_targets() else S.EXPLORING
        self._loc_failures += 1
        if pose.missing > pose.max_missing or self._loc_failures > 3 * pose.max_missing:
            self.recovery_reason = "localisation_lost"
            return S.RECOVERING
        return self._go(S.OBSERVING)

    def _explore(self) -> VisualAgentState:
        self.goal = None
        self.world.objective = "explore"
        if self.world.chest_targets():
            return S.TARGET_DETECTED
        vnav = self.settings.visual_navigation
        if vnav.stop_when_spawn_points_resolved and self.world.priors \
                and self.world.all_priors_resolved:
            return self._complete("all spawn points resolved")
        agent = self.world.pose.position
        if agent is None:
            return self._go(S.OBSERVING)
        knowledge = (int((self.world.occupancy.cells != UNKNOWN).sum()),
                     sum(1 for p in self.world.priors.values() if p.resolved))
        if knowledge == self._knowledge_at_goal:
            self._unproductive_goals += 1
            if self._unproductive_goals > MAX_UNPRODUCTIVE_GOALS:
                return self._complete("exploration stopped revealing new information")
        else:
            self._unproductive_goals = 0
        self._knowledge_at_goal = knowledge
        choice = self.planner.choose_exploration(agent, self.port.time)
        if choice is None:
            return self._complete("no reachable unexplored regions or spawn points remain")
        goal, _ = choice
        if goal.kind == "inspect" and goal.location_id is not None \
                and self._inspections.get(goal.location_id, 0) >= MAX_INSPECTIONS:
            prior = self.world.priors[goal.location_id]
            prior.status = SpawnStatus.ABANDONED
            self.metrics.abandoned_targets += 1
            return self._explore()   # bounded: each call resolves one prior
        self.goal = goal
        self.world.objective = f"{goal.kind} {goal.cell}"
        return S.PLANNING_ROUTE

    def _target_detected(self) -> VisualAgentState:
        agent = self.world.pose.position
        self._new_target = False
        if agent is None:
            return S.EXPLORING
        choice = self.planner.choose_chest(agent, exclude=set())
        if choice is None:
            return S.EXPLORING
        goal, _ = choice
        self.goal = goal
        self.targets_selected += 1
        self.metrics.target_selections += 1
        self.world.objective = f"chest track {goal.track_id}"
        self._approaches = 0
        return S.PLANNING_ROUTE

    def _plan_route(self) -> VisualAgentState:
        goal = self.goal
        agent = self.world.pose.position
        if goal is None or agent is None:
            return S.EXPLORING
        if goal.kind == "chest":
            track = self.world.tracker.get(goal.track_id or -1)
            if track is None:
                return S.EXPLORING
            goal = Goal("chest", (math.floor(track.estimated_position[0]),
                                  math.floor(track.estimated_position[1])),
                        track.estimated_position, goal.score, goal.track_id, goal.location_id)
            self.goal = goal
            stand = self._approach_point(goal, agent)
            if (stand is None or math.dist(agent, stand) <= 0.35) \
                    and math.dist(agent, goal.position) <= self.interaction_radius:
                return self._go(S.ALIGNING)
            target_cell = goal.cell if stand is None else (math.floor(stand[0]),
                                                           math.floor(stand[1]))
            target_pos = goal.position if stand is None else stand
        else:
            target_cell, target_pos, stand = goal.cell, goal.position, None
        start = self.planner.start_cell(agent)
        result = self.planner.plan(start, target_cell)
        self.metrics.astar_searches = self.planner.pathfinder.searches
        if not result.found:
            if goal.kind == "chest":
                track = self.world.tracker.get(goal.track_id or -1)
                if track is not None:
                    self.planner.mark_unreachable(track)
                self.goal = None
                return S.TARGET_DETECTED if self.world.chest_targets() else S.EXPLORING
            self.planner.blacklist(goal.cell, self.port.time)
            self.goal = None
            return S.EXPLORING
        waypoints = build_waypoints(self.world.planning_grid, result.path, agent, target_pos,
                                    radius=self.local.body.radius,
                                    smooth=self.settings.navigation.smooth_paths)
        if goal.kind == "chest" and stand is not None:
            stop, arrive = 0.0, 0.35            # park on the standing point
        elif goal.kind == "chest":
            stop, arrive = self.stop_radius, self.interaction_radius
        elif goal.kind == "inspect":
            stop, arrive = 2.5, 3.0   # look from nearby; do not stand on the spot
        else:
            stop, arrive = 0.0, 1.0
        self.local.set_path(waypoints, goal=target_pos, stop_radius=stop,
                            arrive_radius=arrive, time=self.port.time)
        self.routes_planned += 1
        self._replan_requested = False
        return self._go(S.NAVIGATING)

    def _navigate(self) -> VisualAgentState | None:
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        pose = self.world.pose
        confidence = pose.confidence(self.port.time)
        if pose.is_lost or confidence < self.min_confidence:
            self.local.hold(self.dt)
            self.low_confidence_pauses += 1
            return self._go(S.OBSERVING)
        if self._replan_requested:
            self._replan_requested = False
            self.replans += 1
            return S.PLANNING_ROUTE
        if goal.kind == "chest":
            track = self.world.tracker.get(goal.track_id or -1)
            if track is None or track.category != "chest":
                self.targets_lost += 1
                self.goal = None
                return S.EXPLORING
            if math.dist(track.estimated_position, goal.position) > 1.0:
                return S.PLANNING_ROUTE   # the target estimate moved: re-plan
        elif self._fresh_analysis():
            if self._new_target and self.world.chest_targets():
                return S.TARGET_DETECTED
            if goal.kind == "frontier" and not self._still_frontier(goal):
                return S.EXPLORING
            if goal.kind == "inspect" and self.planner.prior_resolved(goal.location_id):
                return S.EXPLORING
        status = self.local.update(self.world.planning_grid, self.port.time, confidence)
        if status is MoveStatus.ARRIVED:
            if goal.kind == "chest":
                self.targets_reached += 1
                return self._go(S.ALIGNING)
            if goal.kind == "inspect" and goal.location_id is not None:
                self._inspections[goal.location_id] = \
                    self._inspections.get(goal.location_id, 0) + 1   # counted on arrival
                self._observe_needed = 3      # stand still and look before deciding
                self.goal = None
                return self._go(S.OBSERVING)
            else:
                self.planner.blacklist(goal.cell, self.port.time)   # reached: do not re-pick
            return S.EXPLORING
        if status is MoveStatus.STUCK:
            self.recovery_reason = "stuck"
            return S.RECOVERING
        if status is MoveStatus.IDLE:
            self.recovery_reason = "no_path"
            return S.RECOVERING
        if status == LocalController.LOW_CONFIDENCE:
            self.low_confidence_pauses += 1
            return self._go(S.OBSERVING)
        return None

    def _approach_point(self, goal: Goal, agent: tuple[float, float]) -> tuple[float, float] | None:
        """A free standing point next to the chest that keeps its prompt visible.

        Candidates are ordered by distance from the agent; each alignment retry
        moves on to the next one. Returns ``None`` if no candidate is known free.
        """
        occ = self.world.occupancy
        cx, cy = goal.position
        options = []
        for dx, dy in APPROACH_DIRECTIONS:
            px, py = cx + dx * APPROACH_DISTANCE, cy + dy * APPROACH_DISTANCE
            if occ.state(math.floor(px), math.floor(py)) == 0:   # FREE
                options.append((math.dist(agent, (px, py)), (px, py)))
        if not options:
            return None
        options.sort()
        return options[self._approaches % len(options)][1]

    def _still_frontier(self, goal: Goal) -> bool:
        occ = self.world.occupancy
        x, y = goal.cell
        return any(occ.state(x + dx, y + dy) == UNKNOWN
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= x + dx < occ.width and 0 <= y + dy < occ.height)

    def _target_pixel(self) -> tuple[float, float] | None:
        if self.goal is None or self.last_result is None:
            return None
        track = self.world.tracker.get(self.goal.track_id or -1)
        if track is None:
            return None
        return self.last_result.observation.camera.world_to_pixel(*track.estimated_position)

    def _align(self) -> VisualAgentState | None:
        self.local.hold(self.dt)
        goal = self.goal
        track = self.world.tracker.get(goal.track_id or -1) if goal else None
        if goal is None or track is None:
            self.targets_lost += 1
            return S.EXPLORING
        if self._fresh_analysis():
            result = self.last_result
            assert result is not None
            opened, unopened = self._target_appearance(result, goal)
            if opened and not unopened and self._attempts.get(goal.track_id, 0) > 0:
                # An earlier attempt did open it; now we can see the evidence.
                self._succeed(goal, self._hold_ticks * self.dt, result)
                return S.EXPLORING
            pixel = self._target_pixel()
            ppc = result.observation.camera.pixels_per_cell
            if pixel is not None and result.cues.prompt_near(pixel[0], pixel[1], 0.6 * ppc):
                self._hold_ticks = 0
                return self._go(S.INTERACTING)
        if self.port.time - self._phase_start >= ALIGN_SECONDS:
            self._approaches += 1
            if self._approaches <= MAX_ALIGN_APPROACHES:
                return S.PLANNING_ROUTE   # move closer with a tighter stopping radius
            self.recovery_reason = "alignment_failed"
            return S.RECOVERING
        return None

    def _interact(self) -> VisualAgentState | None:
        self.local.hold(self.dt)
        self.port.hold_interact()
        self._hold_ticks += 1
        elapsed = self._hold_ticks * self.dt
        if self._fresh_analysis() and 0.4 <= elapsed <= self.interaction_seconds - 0.2:
            assert self.last_result is not None
            if not self.last_result.cues.progress_visible:
                self.recovery_reason = "interaction_interrupted"
                return S.RECOVERING
        if elapsed + 1e-9 >= self.interaction_seconds + self.dt:
            if self.goal is not None and self.goal.location_id is not None:
                self._unverified[self.goal.location_id] = elapsed
            return self._go(S.VERIFYING_OPEN)
        return None

    def _target_appearance(self, result: PerceptionResult, goal: Goal) -> tuple[bool, bool]:
        """(opened chest seen at the target, unopened chest seen at the target)."""
        opened = unopened = False
        for det in result.world_detections:
            if math.dist(det.position, goal.position) <= MATCH_RADIUS:
                opened |= det.category == "chest_opened"
                unopened |= det.category == "chest"
        return opened, unopened

    def _back_off(self, goal: Goal) -> None:
        """Step away from the target so the agent's own sprite does not hide it."""
        agent = self.world.pose.position
        if agent is None:
            self.local.hold(self.dt)
            return
        dx, dy = agent[0] - goal.position[0], agent[1] - goal.position[1]
        norm = math.hypot(dx, dy) or 1.0
        body = self.local.body
        body.steer(dx / norm * 2.0, dy / norm * 2.0, self.dt)
        body.step(None, self.dt)

    def _verify_open(self) -> VisualAgentState | None:
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        agent = self.world.pose.position
        too_close = agent is not None and math.dist(agent, goal.position) < 0.9
        if too_close and self._phase_analyses >= 1:
            self._back_off(goal)          # self-occlusion: get a clear view
        else:
            self.local.hold(self.dt)
        if not self._fresh_analysis() or self._phase_analyses == 0:
            if self.port.time - self._phase_start > VERIFY_SECONDS + 1.0:
                self.recovery_reason = "verification_inconclusive"
                return S.RECOVERING
            return None
        result = self.last_result
        assert result is not None
        opened, unopened = self._target_appearance(result, goal)
        seconds = self._hold_ticks * self.dt
        if opened and not unopened:
            self._succeed(goal, seconds, result)
            return S.EXPLORING
        if self.port.time - self._phase_start >= VERIFY_SECONDS and self._phase_analyses >= 3 \
                and not too_close:
            if unopened:
                self._record(goal.location_id, present=True, opened=False, seconds=seconds)
                self.metrics.failed_interactions += 1
                if goal.location_id is not None:
                    self._unverified.pop(goal.location_id, None)   # still closed
                self.recovery_reason = "interaction_failed"
            else:
                self.recovery_reason = "verification_inconclusive"
            return S.RECOVERING
        return None

    def _succeed(self, goal: Goal, seconds: float, result: PerceptionResult) -> None:
        self._credit_open(goal.location_id, seconds, result)
        self.goal = None

    def _credit_open(self, location_id: int | None, seconds: float,
                     result: PerceptionResult) -> None:
        self._record(location_id, present=True, opened=True, seconds=seconds)
        self.metrics.chests_opened += 1
        if result.cues.reward_visible:
            self.metrics.rewards += 1
        if location_id is not None:
            self._unverified.pop(location_id, None)
            self.world.priors[location_id].status = SpawnStatus.OPENED

    def _reconcile_unverified(self, result: PerceptionResult) -> None:
        """Late verification: a hold completed but VERIFY was inconclusive (e.g. at low
        resolution the opened sprite was not recognised). If a later frame clearly shows
        an opened chest (and no unopened one) at that spawn point, credit it then.
        Interaction states verify their own target and are left alone."""
        if not self._unverified or self.state in (S.ALIGNING, S.INTERACTING,
                                                  S.VERIFYING_OPEN):
            return
        for location_id, seconds in list(self._unverified.items()):
            prior = self.world.priors[location_id]
            near = [d.category for d in result.world_detections
                    if math.dist(d.position, (prior.x, prior.y)) <= MATCH_RADIUS]
            if "chest_opened" not in near or "chest" in near:
                continue
            self.late_verifications += 1
            self._credit_open(location_id, seconds, result)
            if self.goal is not None and self.goal.location_id == location_id:
                self.goal = None
                self.local.clear()

    def _recover(self) -> VisualAgentState:
        self.metrics.recoveries += 1
        self._total_recoveries += 1
        reason = self.recovery_reason or "unknown"
        self.recovery_reason = None
        if self._total_recoveries > self.recovery_budget:
            return self._fail(f"recovery budget of {self.recovery_budget} exhausted "
                              f"(last problem: {reason})")
        if reason == "localisation_lost":
            self._loc_failures = 0
            key: object = "localisation"
            self._attempts[key] = self._attempts.get(key, 0) + 1
            if self._attempts[key] > self.max_attempts:
                return self._fail("localisation lost: the agent marker could not be found")
            self.world.pose.missing = 0
            return self._go(S.OBSERVING)
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        key = goal.track_id if goal.kind == "chest" else goal.cell
        self._attempts[key] = self._attempts.get(key, 0) + 1
        interaction = reason in ("interaction_failed", "interaction_interrupted",
                                 "verification_inconclusive", "alignment_failed")
        if self._attempts[key] > self.max_attempts:
            if goal.kind == "chest" and goal.location_id is not None:
                prior = self.world.priors[goal.location_id]
                prior.status = SpawnStatus.FAILED if interaction else SpawnStatus.ABANDONED
                self.metrics.abandoned_targets += 1
            else:
                self.planner.blacklist(goal.cell, self.port.time)
            self.goal = None
            self.local.clear()
            return S.EXPLORING
        if interaction and reason != "alignment_failed":
            return self._go(S.ALIGNING)
        self._approaches = 0
        self.local.clear()
        return S.PLANNING_ROUTE

    # -- helpers ---------------------------------------------------------------------

    def _record(self, location_id: int | None, *, present: bool, opened: bool,
                seconds: float) -> None:
        if location_id is None:
            return
        if location_id not in self.recorded_locations:
            self.recorded_locations.add(location_id)
            self.metrics.locations_visited += 1
        if self.recorder is not None and self.clock is not None:
            self.recorder.record(location_id, present=present, opened=opened,
                                 interaction_seconds=seconds,
                                 observed_at=self.clock.at(self.port.time))

    def _on_terminal(self) -> None:
        self.local.clear()
        self.port.move(0.0, 0.0)
        self.goal = None
        self.metrics.unreachable_targets = sum(
            1 for p in self.world.priors.values() if p.status is SpawnStatus.UNREACHABLE)
        if self.recorder is not None:
            self.recorder.flush()
        if self.state is S.FAILED:
            self.metrics.failure_reason = self.end_reason
            if self.outcome == "cancelled":
                logger.info("Visual episode cancelled: %s", self.end_reason)
            else:
                logger.warning("Visual agent failed: %s", self.end_reason)
        if self._finalizer is not None:
            self._finalizer(self)
