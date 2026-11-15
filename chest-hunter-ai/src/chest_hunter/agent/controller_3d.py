"""The Phase 4 3D agents: the visual agent (hybrid / visual-navigation) and the oracle.

:class:`VisualAgent3D` closes the loop DETECT -> TRACK -> APPROACH -> ALIGN ->
INTERACT -> CONFIRM using only what :class:`~chest_hunter.unreal_bridge.client.AgentPort`
provides: camera frames, its own camera rig, odometry, reward notifications
and the (permitted) navigation service.

Each decision step the runner calls :meth:`tick` (the FSM chooses
``self.action``), sends one ``STEP`` (``decision_ticks`` fixed ticks), and
passes the result to :meth:`on_step`, which runs perception when an
observation was requested. Perception runs at adaptive rates: sparse while
travelling, dense while searching, approaching and verifying.

:class:`OracleAgent3D` is the benchmark baseline: it is given exact chest
positions through the privileged oracle channel and never uses vision.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from chest_hunter.agent.fsm import StateMachine
from chest_hunter.agent.recovery_3d import RecoveryLog, classify
from chest_hunter.localization.world_model_3d import (
    PriorStatus3D,
    SpawnPrior3D,
    WorldModel3D,
)
from chest_hunter.navigation.navigation_3d import (
    LocalNavigator,
    NavMeshNavigator,
    PathFollower,
    ScanPlanner,
)
from chest_hunter.runtime.profiler import NULL_PROFILER
from chest_hunter.unreal_bridge.action_adapter import ActionCommand, look_towards, move_towards
from chest_hunter.unreal_bridge.client import BridgeError, StepResult
from chest_hunter.unreal_bridge.observation_adapter import Observation3D, Telemetry
from chest_hunter.unreal_bridge.protocol import MAX_RAYS
from chest_hunter.vision.chest_detector_3d import (
    ChestDetector3D,
    Detection3D,
    segment_classes,
)
from chest_hunter.vision.depth_estimation import CHEST_HEIGHT, RangeEstimate
from chest_hunter.vision.interaction_cues_3d import Cues, detect_cues
from chest_hunter.vision.perspective import angle_difference_deg, bearing_deg, project


class Agent3DState(StrEnum):
    INITIALIZING = "INITIALIZING"
    OBSERVING = "OBSERVING"
    SCANNING = "SCANNING"
    EXPLORING = "EXPLORING"
    TARGET_DETECTED = "TARGET_DETECTED"
    PLANNING_ROUTE = "PLANNING_ROUTE"
    NAVIGATING = "NAVIGATING"
    APPROACHING = "APPROACHING"
    ALIGNING = "ALIGNING"
    INTERACTING = "INTERACTING"
    VERIFYING_OPEN = "VERIFYING_OPEN"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


S = Agent3DState
_END = frozenset({S.COMPLETED, S.FAILED})
TRANSITIONS_3D: dict[Agent3DState, frozenset[Agent3DState]] = {
    S.INITIALIZING: frozenset({S.OBSERVING}) | _END,
    S.OBSERVING: frozenset({S.SCANNING, S.EXPLORING, S.TARGET_DETECTED, S.PLANNING_ROUTE}) | _END,
    S.SCANNING: frozenset({S.EXPLORING, S.TARGET_DETECTED}) | _END,
    S.EXPLORING: frozenset({S.TARGET_DETECTED, S.PLANNING_ROUTE, S.SCANNING, S.OBSERVING}) | _END,
    S.TARGET_DETECTED: frozenset({S.PLANNING_ROUTE, S.APPROACHING, S.EXPLORING}) | _END,
    S.PLANNING_ROUTE: frozenset({S.NAVIGATING, S.APPROACHING, S.EXPLORING, S.SCANNING,
                                 S.OBSERVING}) | _END,
    S.NAVIGATING: frozenset({S.APPROACHING, S.EXPLORING, S.TARGET_DETECTED, S.SCANNING,
                             S.PLANNING_ROUTE, S.RECOVERING}) | _END,
    S.APPROACHING: frozenset({S.ALIGNING, S.PLANNING_ROUTE, S.RECOVERING, S.EXPLORING}) | _END,
    S.ALIGNING: frozenset({S.INTERACTING, S.APPROACHING, S.RECOVERING, S.EXPLORING}) | _END,
    # EXPLORING: the goal vanished mid-hold (credited elsewhere, e.g. late verification).
    S.INTERACTING: frozenset({S.VERIFYING_OPEN, S.RECOVERING, S.EXPLORING}) | _END,
    S.VERIFYING_OPEN: frozenset({S.EXPLORING, S.RECOVERING}) | _END,
    S.RECOVERING: frozenset({S.PLANNING_ROUTE, S.ALIGNING, S.APPROACHING, S.EXPLORING,
                             S.OBSERVING}) | _END,
    S.COMPLETED: frozenset(),
    S.FAILED: frozenset(),
}
TIMED_3D = frozenset({S.OBSERVING, S.SCANNING, S.NAVIGATING, S.APPROACHING, S.ALIGNING,
                      S.INTERACTING, S.VERIFYING_OPEN})


@dataclass(frozen=True, slots=True)
class Agent3DConfig:
    decision_ticks: int = 2
    rate_travel: float = 3.0           # analyses per simulated second while travelling
    rate_search: float = 8.0           # scanning / approaching
    rate_verify: float = 10.0          # aligning / interacting / verifying
    rate_idle: float = 2.0
    scan_step_deg: float = 50.0
    scan_half_width_deg: float = 75.0
    approach_switch_m: float = 4.5     # start visual approach within this distance
    stand_distance_m: float = 1.1      # where to stand from the chest centre
    arrive_margin_m: float = 0.3       # arrived when within stand distance + margin
    close_pitch_min_deg: float = -32.0 # a spring-arm camera aimed at a chest underfoot would
                                       # otherwise converge to ~45 degrees looking down
    max_attempts: int = 3
    max_inspections: int = 2
    inspect_range_m: float = 6.0
    vote_range_m: float = 9.0
    cruise_pitch_deg: float = -12.0
    #: Phase 5: classified, bounded recovery strategies (off = Phase 4 behaviour).
    smart_recovery: bool = False
    max_lookbacks: int = 2
    max_rescans: int = 1
    uncertain_sigma_m: float = 1.0
    #: Phase 6 adaptive computation: request no frame while nothing can have changed (the
    #: agent stood still and kept its camera still since the last analysed frame), except in
    #: aligning/interacting/verifying, and at least one frame per ``max_frame_skip_s``.
    skip_unchanged_frames: bool = False
    max_frame_skip_s: float = 1.0


#: Phase 6 visual-processing profiles: analyses per simulated second while travelling far
#: from the goal, searching/approaching, aligning-interacting-verifying, and otherwise.
#: an opened chest was within interaction reach (1.8 m) of the agent; with odometry noise
REACH_CHECK_M = 2.6

VISION_PROFILES: dict[str, dict[str, float]] = {
    "phase5": {"rate_travel": 3.0, "rate_search": 8.0, "rate_verify": 10.0, "rate_idle": 2.0},
    "efficient": {"rate_travel": 2.0, "rate_search": 6.0, "rate_verify": 10.0,
                  "rate_idle": 1.0},
    "dense": {"rate_travel": 5.0, "rate_search": 10.0, "rate_verify": 10.0, "rate_idle": 5.0},
}


def apply_vision_profile(config: Agent3DConfig, profile: str, *,
                         skip_unchanged: bool | None = None) -> Agent3DConfig:
    """``config`` with the perception rates of a named profile (see VISION_PROFILES)."""
    if profile not in VISION_PROFILES:
        raise ValueError(f"vision profile must be one of {', '.join(VISION_PROFILES)}")
    from dataclasses import replace

    rates = VISION_PROFILES[profile]
    out = replace(config, rate_travel=rates["rate_travel"], rate_search=rates["rate_search"],
                  rate_verify=rates["rate_verify"], rate_idle=rates["rate_idle"])
    if skip_unchanged is not None:
        out = replace(out, skip_unchanged_frames=skip_unchanged)
    return out


@dataclass(slots=True)
class Goal3D:
    kind: str                                   # "chest" or "inspect"
    position: tuple[float, float, float]
    location_id: int | None
    track_id: int | None = None


@dataclass(slots=True)
class Metrics3D:
    decisions: int = 0
    analyses: int = 0
    perception_seconds: float = 0.0
    perception_peak: float = 0.0
    detections: int = 0
    chests_opened: int = 0
    rewards: int = 0
    failed_interactions: int = 0
    recoveries: int = 0
    navigation_failures: int = 0
    unreachable_targets: int = 0
    abandoned_targets: int = 0
    targets_selected: int = 0
    targets_reached: int = 0
    targets_lost: int = 0
    late_verifications: int = 0
    scans: int = 0
    locations_recorded: int = 0
    travel_distance: float = 0.0
    recovery_reasons: dict[str, int] = field(default_factory=dict)
    interaction_attempts: int = 0
    credited_by_notification: int = 0
    reattributed_openings: int = 0      # opening credited to the spawn point actually reached
    service_errors: int = 0             # navigation-service requests that failed (skipped)
    frames_skipped_unchanged: int = 0
    routes_started: int = 0
    recovery_log: RecoveryLog = field(default_factory=RecoveryLog)

    @property
    def perception_ms_mean(self) -> float:
        return 1000.0 * self.perception_seconds / max(1, self.analyses)


class Recorder(Protocol):
    def __call__(self, location_id: int, *, present: bool, opened: bool, seconds: float,
                 reward: bool | None, sim_time: float) -> None: ...


class VisualAgent3D(StateMachine[Agent3DState]):
    """Vision-based 3D chest hunter (hybrid NavMesh or local navigation)."""

    def __init__(self, port: Any, priors: list[SpawnPrior3D], *, rules: dict[str, Any],
                 detector: ChestDetector3D, navigation: str = "navmesh",
                 config: Agent3DConfig | None = None, recorder: Recorder | None = None,
                 explorer: Any = None, thresholds: Any = None) -> None:
        self.decisions = 0
        super().__init__(initial=S.INITIALIZING, completed=S.COMPLETED, failed=S.FAILED,
                         transitions=TRANSITIONS_3D, timed=TIMED_3D,
                         tick_source=lambda: self.decisions, recovering=S.RECOVERING)
        self.port = port
        self.cfg = config or Agent3DConfig()
        self.rules = rules
        self.detector = detector
        self.recorder = recorder
        #: Phase 8: where openings of discovered (negative-id) spawn points are stored
        self.discovery_recorder: Callable[..., None] | None = None
        #: Phase 7 active exploration (exploration.active.ActiveExplorer): no prior chest
        #: coordinates; frontiers, camera sectors and dynamic spawn points instead.
        self.explorer = explorer
        self.world = WorldModel3D(priors, vote_range=self.cfg.vote_range_m,
                                  discover=explorer is not None, thresholds=thresholds)
        #: Phase 7 localisation monitor (odometry only: no absolute pose in the 3D arena).
        self.localisation_jumps = 0
        self.localisation_confidence = 1.0
        self._explore_check_at = 0.0
        self._look_yaw = 0.0
        self._look_until = -1.0
        self.on_localisation_jump: Callable[[float], None] | None = None
        self.navigator: NavMeshNavigator | LocalNavigator = (
            NavMeshNavigator(port) if navigation == "navmesh"
            else LocalNavigator(port, floors=tuple(round(p.z, 2) for p in priors)))
        self.metrics = Metrics3D()
        self.tick_rate = int(rules.get("tick_rate", 30))
        self.step_seconds = self.cfg.decision_ticks / self.tick_rate
        self.interaction_seconds = float(rules.get("interaction_seconds", 2.0))
        self.action = ActionCommand()
        self.telemetry: Telemetry | None = None
        self.obs: Observation3D | None = None
        self.classes: np.ndarray | None = None
        self.last_detections: list[Detection3D] = []
        self.last_estimates: list[RangeEstimate | None] = []
        self.cues = Cues(False, False)
        self.goal: Goal3D | None = None
        self.follower: PathFollower | None = None
        self.now = 0.0
        self._last_analysis = -math.inf
        self._phase_start = 0.0
        self._phase_analyses = 0
        self._scan: list[float] = []
        self._scan_pitch: float | None = None
        self._settled_since = -1.0
        self._after_observe: Agent3DState = S.EXPLORING
        self._maneuver_until = -1.0
        self._maneuver_bearing = 0.0
        self._attempts: dict[object, int] = {}
        self._align_failures: dict[object, int] = {}
        self._total_recoveries = 0
        self.recovery_budget = (self.cfg.max_attempts + 1) * (len(priors) + 5)
        self._unreachable_keys: set[int] = set()
        self._approach_stand = self.cfg.stand_distance_m
        self._hold_started = -1.0
        self._verify = {"opened": 0, "closed": 0, "reward": 0}
        self._unverified: dict[int, float] = {}
        self._known_targets: set[int] = set()
        self._new_target = False
        self._last_position: tuple[float, float, float] | None = None
        self._approach_progress = (math.inf, 0.0)
        self._approach_goal: tuple[float, float, float] | None = None
        self._off_mesh_waits = 0
        self._approach_side = 0.0
        self._plan_failures: dict[int, int] = {}
        self._trail: deque[tuple[float, tuple[float, float, float]]] = deque(maxlen=90)
        self._approach_path: PathFollower | None = None
        #: Phase 5 adaptive planner (exploration.adaptive_planner); None = Phase 4 rules.
        self.planner: Any = None
        self._lookbacks: dict[object, int] = {}
        self._last_notification_reward = False
        self._rescans: dict[object, int] = {}
        self._approach_losses: dict[object, int] = {}
        self._uncertain_checks: dict[object, int] = {}
        #: Phase 6: component profiler (runtime.profiler); the null profiler costs nothing.
        self.profiler: Any = NULL_PROFILER
        self._analysed_pose: tuple[float, ...] | None = None
        self._handlers = {
            S.OBSERVING: self._observe, S.SCANNING: self._scanning,
            S.EXPLORING: lambda: self._profiled("route_selection", self._explore),
            S.TARGET_DETECTED: lambda: self._profiled("route_selection", self._target_detected),
            S.PLANNING_ROUTE: self._plan_route,
            S.NAVIGATING: self._navigate, S.APPROACHING: self._approach, S.ALIGNING: self._align,
            S.INTERACTING: self._interact, S.VERIFYING_OPEN: self._verify_open,
            S.RECOVERING: self._recover,
        }

    def set_profiler(self, profiler: Any) -> None:
        self.profiler = profiler
        self.navigator.profiler = profiler

    def _profiled(self, name: str, handler: Callable[[], Agent3DState]) -> Agent3DState:
        with self.profiler.section(name):
            return handler()

    # -- runner interface ----------------------------------------------------------------------

    def initialize(self, telemetry: Telemetry) -> None:
        self.telemetry = telemetry
        self.now = telemetry.sim_time
        self._scan = ScanPlanner.full_turn(telemetry.camera.yaw_deg, self.cfg.scan_step_deg)
        self._after_observe = S.SCANNING
        self._enter()
        self._transition(S.OBSERVING, f"{len(self.world.priors)} spawn prior(s)")

    def tick(self) -> Agent3DState:
        self.decisions += 1
        self.metrics.decisions += 1
        self.action = ActionCommand()
        state = super().tick()
        if self._maneuver_until > self.now and not self.is_terminal():
            f, r = move_towards(self._maneuver_bearing, self._cam_yaw, 0.8)
            self.action.move_forward, self.action.move_right = f, r
        return state

    def wants_observation(self) -> bool:
        if self.is_terminal():
            return False
        if self.state in (S.ALIGNING, S.INTERACTING, S.VERIFYING_OPEN):
            rate = self.cfg.rate_verify
        elif self.state in (S.SCANNING, S.APPROACHING, S.OBSERVING):
            rate = self.cfg.rate_search
        elif self.state is S.NAVIGATING:
            rate = self.cfg.rate_search if self._near_goal(10.0) else self.cfg.rate_travel
        else:
            rate = self.cfg.rate_idle
        due = self.now + self.step_seconds - self._last_analysis >= 1.0 / rate - 1e-6
        if due and self.cfg.skip_unchanged_frames and self._unchanged_view():
            self.metrics.frames_skipped_unchanged += 1
            return False
        return due

    def _unchanged_view(self) -> bool:
        """Phase 6: the next frame would show what the last analysed one showed (agent and
        camera still, no command to move or look, no opening in progress)."""
        if self._analysed_pose is None or self.telemetry is None \
                or self.state in (S.ALIGNING, S.INTERACTING, S.VERIFYING_OPEN) \
                or self.now - self._last_analysis >= self.cfg.max_frame_skip_s \
                or self._maneuver_until > self.now:
            return False
        a = self.action
        if abs(a.move_forward) + abs(a.move_right) + abs(a.look_yaw) + abs(a.look_pitch) > 1e-6 \
                or a.jump or a.interact:
            return False
        return _pose_close(self._pose_now(), self._analysed_pose)

    def _pose_now(self) -> tuple[float, ...]:
        assert self.telemetry is not None
        x, y, z = self.telemetry.odometry.position
        return (x, y, z, self.telemetry.camera.yaw_deg, self.telemetry.camera.pitch_deg)

    def on_step(self, result: StepResult) -> None:
        tel = result.telemetry
        if self.telemetry is not None:
            a, b = self.telemetry.odometry.position, tel.odometry.position
            moved = math.hypot(b[0] - a[0], b[1] - a[1])
            dt = max(1e-6, tel.sim_time - self.telemetry.sim_time)
            limit = float(self.rules.get("sprint_speed", 8.0)) * dt * 1.5 + 0.3
            if moved > limit:                  # odometry jumped further than the agent can move
                self.localisation_jumps += 1
                self.localisation_confidence *= 0.3
                if self.on_localisation_jump is not None:
                    self.on_localisation_jump(moved)
            else:
                self.metrics.travel_distance += moved
                self.localisation_confidence = min(1.0, self.localisation_confidence + 0.002)
            if self.explorer is not None and self.state is S.NAVIGATING:
                self.explorer.travel_seconds += dt
        self.telemetry = tel
        self.now = tel.sim_time
        self._trail.append((self.now, tel.odometry.position))
        for event in tel.events:
            if event.get("type") == "chest_opened_notification":
                self._verify["reward"] += 1
                self._last_notification_reward = bool(event.get("reward"))
                if event.get("reward"):
                    self.metrics.rewards += 1
        if result.observation is not None:
            self.perceive(result.observation)
        if isinstance(self.navigator, LocalNavigator):
            self.navigator.observe(self.position)

    # -- perception ------------------------------------------------------------------------------

    def perceive(self, obs: Observation3D) -> None:
        started = time.perf_counter()
        self.obs = obs
        prof = self.profiler
        with prof.section("preprocessing"):
            mask = self.world.self_mask(obs)
            hsv_classes = segment_classes(_hsv(obs.frame))
        with prof.section("chest_detection"):
            # The detector reuses this segmentation (it used to compute the same one again).
            dets = self.detector.detect(obs.frame, obs.sim_time, exclude=mask,
                                        camera_pitch_deg=obs.camera.pitch_deg,
                                        fy=obs.intrinsics.fy, classes=hsv_classes)
        with prof.section("tracking_localisation"):
            update = self.world.integrate(obs, dets, hsv_classes, self._clear_line)
        self.classes = hsv_classes
        self.last_detections = dets
        self.last_estimates = update.estimates
        for p in update.newly_empty:
            self._record(p.location_id, present=False, opened=False, seconds=0.0, reward=None)
        targets = {t.track_id for t in self.world.chest_targets()}
        if targets - self._known_targets:
            self._new_target = True
        self._known_targets |= targets
        with prof.section("interaction_cues"):
            self.cues = detect_cues(obs.frame, self._target_bbox())
        if self.explorer is not None:
            with prof.section("exploration_update"):
                self._update_exploration(obs, dets, update.estimates)
        self._late_verification()
        if self.telemetry is not None:
            self._analysed_pose = self._pose_now()
        elapsed = time.perf_counter() - started
        self.metrics.analyses += 1
        self.metrics.detections += len(dets)
        self.metrics.perception_seconds += elapsed
        self.metrics.perception_peak = max(self.metrics.perception_peak, elapsed)
        self._last_analysis = obs.sim_time
        self._phase_analyses += 1

    def _clear_line(self, prior: SpawnPrior3D) -> bool:
        try:
            return self.navigator.clear_line(self.position, prior.xyz)
        except BridgeError:                    # navigation service unavailable: be conservative
            self.metrics.service_errors += 1
            return False

    def _target_bbox(self) -> tuple[int, int, int, int] | None:
        """Pixel box of the current chest goal in the latest frame (if visible)."""
        if self.goal is None or self.goal.kind != "chest" or self.obs is None:
            return None
        gx, gy, gz = self.goal.position
        pose = self.obs.camera_pose()
        best = None
        for det, est in zip(self.last_detections, self.last_estimates, strict=False):
            if est is None:
                continue
            d = math.hypot(est.position[0] - gx, est.position[1] - gy)
            if d <= 1.5 and (best is None or d < best[0]):
                best = (d, det.bbox)
        if best is not None:
            return best[1]
        u, v, depth = project(np.array([gx, gy, gz + CHEST_HEIGHT / 2]), pose, self.obs.intrinsics)
        if not float(depth) > 0.3:
            return None
        size = max(4, int(self.obs.intrinsics.fx * 0.9 / float(depth)))
        return (int(u - size / 2), int(v - size / 2), size, size)

    def _target_appearance(self) -> tuple[bool, bool]:
        """(opened chest seen at the goal, unopened chest seen at the goal)."""
        if self.goal is None:
            return False, False
        gx, gy, _ = self.goal.position
        opened = closed = False
        for det, est in zip(self.last_detections, self.last_estimates, strict=False):
            if est is None:
                continue
            if math.hypot(est.position[0] - gx, est.position[1] - gy) <= 1.2:
                opened |= det.category == "chest_opened"
                closed |= det.category == "chest"
        return opened, closed

    def _late_verification(self) -> None:
        if not self._unverified or self.state in (S.ALIGNING, S.INTERACTING, S.VERIFYING_OPEN):
            return
        for loc, seconds in list(self._unverified.items()):
            prior = self.world.priors[loc]
            near = [det.category for det, est in zip(self.last_detections, self.last_estimates,
                                                      strict=False)
                    if est is not None
                    and math.hypot(est.position[0] - prior.x, est.position[1] - prior.y) <= 1.5
                    and est.distance <= 8.0]
            if "chest_opened" in near and "chest" not in near:
                self.metrics.late_verifications += 1
                self._credit_open(loc, seconds, reward=None)

    # -- helpers ---------------------------------------------------------------------------------

    @property
    def position(self) -> tuple[float, float, float]:
        assert self.telemetry is not None
        return self.telemetry.odometry.position

    @property
    def _cam_yaw(self) -> float:
        return self.telemetry.camera.yaw_deg if self.telemetry else 0.0

    @property
    def _cam_pitch(self) -> float:
        return self.telemetry.camera.pitch_deg if self.telemetry else 0.0

    def _enter(self) -> None:
        self._approach_progress = (math.inf, self.now)
        self._phase_start = self.now
        self._phase_analyses = 0
        self._settled_since = -1.0

    def _go(self, state: Agent3DState) -> Agent3DState:
        self._enter()
        return state

    def _look(self, yaw: float, pitch: float) -> float:
        pitch = max(float(self.rules.get("pitch_min_deg", -55)) + 1,
                    min(float(self.rules.get("pitch_max_deg", 25)) - 1, pitch))
        ly, lp = look_towards(self._cam_yaw, self._cam_pitch, yaw, pitch,
                              step_seconds=self.step_seconds,
                              yaw_rate=float(self.rules.get("yaw_rate_deg", 120)),
                              pitch_rate=float(self.rules.get("pitch_rate_deg", 60)))
        self.action.look_yaw, self.action.look_pitch = ly, lp
        return abs(angle_difference_deg(yaw, self._cam_yaw)) + abs(pitch - self._cam_pitch)

    def _move(self, bearing: float, speed: float) -> None:
        f, r = move_towards(bearing, self._cam_yaw, speed)
        self.action.move_forward, self.action.move_right = f, r

    #: Close to a chest the over-the-shoulder camera would look *through* the agent's own
    #: body (it stands between camera and chest). Looking this far to the left of the
    #: chest bearing puts the chest near the image centre and the body beside it, while
    #: the chest stays well inside the interaction cone.
    CLOSE_AIM_OFFSET_DEG = 30.0

    def _aim_at(self, point: tuple[float, float, float]) -> float:
        """Point the camera at a world point; returns the remaining angular error."""
        cam = self.obs.camera_pose() if self.obs is not None else None
        if cam is None:
            return 0.0
        yaw = self._view_yaw(point)
        dx, dy = point[0] - cam.x, point[1] - cam.y
        dz = point[2] + CHEST_HEIGHT / 2 - cam.z
        pitch = math.degrees(math.atan2(dz, max(0.3, math.hypot(dx, dy))))
        if math.hypot(point[0] - self.position[0], point[1] - self.position[1]) <= 4.5:
            pitch = max(pitch, self.cfg.close_pitch_min_deg)
        return self._look(yaw, pitch)

    def _view_yaw(self, point: tuple[float, float, float]) -> float:
        """Camera yaw that shows ``point`` without the agent's own body in the way."""
        pos = self.position
        dist = math.hypot(point[0] - pos[0], point[1] - pos[1])
        bearing = bearing_deg(point[0] - pos[0], point[1] - pos[1])
        if dist <= 4.5:
            return (bearing - self.CLOSE_AIM_OFFSET_DEG) % 360.0
        if dist <= 10.0:
            return (bearing - self.CLOSE_AIM_OFFSET_DEG / 2) % 360.0
        cam = self.obs.camera_pose() if self.obs is not None else None
        return bearing if cam is None else bearing_deg(point[0] - cam.x, point[1] - cam.y)

    def _near_goal(self, radius: float) -> bool:
        if self.goal is None or self.telemetry is None:
            return False
        return math.dist(self.position[:2], self.goal.position[:2]) <= radius

    def _opened_location(self, location_id: int) -> int:
        """The spawn point the opening belongs to. A chest is opened only within reach
        (1.8 m), so when the agent stands far from the spawn point its goal was associated
        with, but next to another known spawn point, the chest it opened is that one (Phase 8
        audit F14: a misplaced track had been associated with a spawn point 4.8 m away).
        Discovered spawn points (negative ids) move to the opening instead."""
        if location_id < 0 or self.telemetry is None:
            return location_id
        pos = self.position
        here = self.world.priors.get(location_id)
        if here is None or math.hypot(here.x - pos[0], here.y - pos[1]) <= REACH_CHECK_M:
            return location_id
        near = [(math.hypot(p.x - pos[0], p.y - pos[1]), p.location_id)
                for p in self.world.priors.values()
                if p.location_id >= 0 and p.status is not PriorStatus3D.OPENED
                and abs(p.z - pos[2]) <= 1.0]
        best = min(near, default=None)
        if best is not None and best[0] <= REACH_CHECK_M:
            self.metrics.reattributed_openings += 1
            return best[1]
        return location_id

    def _record(self, location_id: int | None, *, present: bool, opened: bool, seconds: float,
                reward: bool | None) -> None:
        if location_id is not None and location_id < 0:
            # Phase 7 discovered spawn points have no chest_locations row: their openings go
            # to discovered_openings (Phase 8 audit F12; before, they were not stored at all)
            prior = self.world.priors.get(location_id)
            if opened and prior is not None and self.discovery_recorder is not None:
                self.metrics.locations_recorded += 1
                self.discovery_recorder(prior.xyz, seconds=seconds, reward=reward,
                                        sim_time=self.now)
            return
        if location_id is None:
            return
        self.metrics.locations_recorded += 1
        if self.recorder is not None:
            self.recorder(location_id, present=present, opened=opened, seconds=seconds,
                          reward=reward, sim_time=self.now)

    def _credit_open(self, location_id: int, seconds: float, reward: bool | None) -> bool:
        """Count one opening; ``False`` when this spawn point was already credited (one
        spawn point holds at most one chest per episode, so it is never counted twice)."""
        resolved = self._opened_location(location_id)
        # Both ids leave the unverified set: when the opening was reattributed (F14), the
        # interaction's own spawn point stayed "unverified" and was credited a second time
        # when the agent aligned on it and saw the neighbouring opened chest (Phase 8 audit
        # F19: one phantom opening per affected episode).
        self._unverified.pop(location_id, None)
        self._unverified.pop(resolved, None)
        location_id = resolved
        prior = self.world.priors[location_id]
        if prior.status is PriorStatus3D.OPENED:
            if self.goal is not None and self.goal.location_id == location_id \
                    and self.state not in (S.VERIFYING_OPEN,):
                self.goal = None
            return False
        prior.status = PriorStatus3D.OPENED
        if location_id < 0 and self.goal is not None and self.goal.location_id == location_id:
            # a discovered spawn point moves to where the chest was actually opened (its
            # first estimate may have come from far away), so later sightings link to it
            prior.x, prior.y, prior.z = self.goal.position
        self.metrics.chests_opened += 1
        # every recovery on the way to this spawn point (whatever track or goal it had then)
        self.metrics.recovery_log.resolve((None, location_id, None))
        self.metrics.recovery_log.resolve(("reach", (None, location_id, None)))
        if self.planner is not None:
            self.planner.opened(location_id)
        self._record(location_id, present=True, opened=True, seconds=seconds, reward=reward)
        if self.goal is not None and self.goal.location_id == location_id \
                and self.state not in (S.VERIFYING_OPEN,):
            self.goal = None
        return True

    def credit_opening(self, location_id: int, seconds: float, reward: bool | None) -> bool:
        """Record a verified opening decided outside the state machine (Phase 5 learned
        control: the agent's own opening notification next to a spawn point)."""
        if location_id in self.world.priors:
            return self._credit_open(location_id, seconds, reward)
        return False

    def release_interaction(self) -> bool:
        """Drop an interaction or opening verification that is no longer current.

        Phase 5 learned control: another controller moved the agent since the state machine
        last ran, so its hold timer, aim point and verification counters are stale (the
        reward counter may even hold notifications for other chests). Resuming such a phase
        could credit a spawn point that was never opened. Returns ``True`` when a phase was
        dropped; pending late verifications are kept (they need fresh visual evidence)."""
        if self.state not in (S.INTERACTING, S.VERIFYING_OPEN):
            return False
        self._end_interaction("stale interaction released (another controller drove)")
        return True

    def close_credited_interaction(self) -> bool:
        """End the state machine's current interaction because its opening notification was
        credited outside it (Phase 5 learned control: the policy selector credits every
        notification at the spawn point next to the agent). The interaction's goal may be a
        neighbouring spawn point; its verification, or a late verification of it, would count
        the same chest a second time, so both are dropped. Returns ``True`` when a phase was
        ended."""
        if self.state not in (S.INTERACTING, S.VERIFYING_OPEN):
            return False
        if self.goal is not None and self.goal.location_id is not None:
            self._unverified.pop(self.goal.location_id, None)
        self._end_interaction("opening credited by the policy selector")
        return True

    def _end_interaction(self, reason: str) -> None:
        self.goal = None
        self._verify = {"opened": 0, "closed": 0, "reward": 0}
        self._hold_started = -1.0
        self._enter()
        self._transition(S.EXPLORING, reason)

    # -- Phase 6 supervisor interventions ----------------------------------------------------
    # Each returns True when it was applicable and applied. They only use transitions of
    # the table (EXPLORING is reachable from every non-terminal state), never end an
    # interaction's late verification (an opening already held for may still be confirmed
    # visually) and never touch simulator state.

    def _drop_goal(self, *, failure: str, status: PriorStatus3D | None) -> None:
        goal = self.goal
        if goal is not None:
            self._route_end(False, failure)
            if status is not None and goal.location_id is not None:
                prior = self.world.priors.get(goal.location_id)
                if prior is not None and not prior.resolved:
                    prior.status = status
                    self.metrics.abandoned_targets += 1
        self.goal = None
        self.follower = None
        self._approach_goal = None
        self._approach_path = None
        self._verify = {"opened": 0, "closed": 0, "reward": 0}
        self._hold_started = -1.0
        self._maneuver_until = -1.0

    def supervisor_replan(self, reason: str) -> bool:
        """Recalculate the route to the current goal (navigation cache cleared)."""
        if self.goal is None or self.state not in (S.NAVIGATING, S.APPROACHING):
            return False
        self._route_end(False, "path_blocked")
        self.follower = None
        self._approach_goal = None
        self._approach_path = None
        if isinstance(self.navigator, NavMeshNavigator):
            self.navigator.clear_cache()
        self._enter()
        self._transition(S.PLANNING_ROUTE, f"supervisor: {reason}")
        return True

    def supervisor_abandon_target(self, reason: str, *, failed: bool = False) -> bool:
        """Give up the current goal for this episode and select another target."""
        if self.goal is None or self.is_terminal() or self.state is S.INITIALIZING:
            return False
        self._drop_goal(failure="abandoned",
                        status=PriorStatus3D.FAILED if failed else PriorStatus3D.ABANDONED)
        self._enter()
        self._transition(S.EXPLORING, f"supervisor: {reason}")
        return True

    def supervisor_relocalise(self, reason: str, *, stale_seconds: float = 2.0) -> bool:
        """Discard stale tracks and re-observe with a full look-around (the goal, if any,
        is dropped without being abandoned: it may be selected again)."""
        if self.is_terminal() or self.state is S.INITIALIZING:
            return False
        dropped = self.world.tracker.discard_stale(self.now, stale_seconds)
        self._known_targets = {t.track_id for t in self.world.chest_targets()}
        self._drop_goal(failure="relocalise", status=None)
        self._scan = ScanPlanner.full_turn(self._cam_yaw, self.cfg.scan_step_deg)
        self._scan_pitch = None
        self._after_observe = S.SCANNING
        self.metrics.scans += 1
        self._enter()
        detail = f"supervisor: {reason} ({dropped} stale track(s) dropped)"
        if S.OBSERVING not in TRANSITIONS_3D[self.state]:
            self._transition(S.EXPLORING, detail)
        self._transition(S.OBSERVING, detail)
        return True

    def supervisor_unstick(self, reason: str, *, attempt: int = 1) -> bool:
        """Move a short distance in a direction the agent's own back-off does not use (its
        retrace bearing turned by 90, 180 or 270 degrees), then re-observe."""
        if self.is_terminal() or self.state is S.INITIALIZING or self.telemetry is None:
            return False
        self._maneuver_bearing = (self._retrace_bearing() + 90.0 * (1 + (attempt - 1) % 3)) \
            % 360.0
        self._maneuver_until = self.now + 1.0
        self.follower = None
        self._approach_goal = None
        self._approach_path = None
        self._after_observe = S.PLANNING_ROUTE if self.goal is not None else S.EXPLORING
        self._enter()
        detail = f"supervisor: {reason} (unstick towards {self._maneuver_bearing:.0f} deg)"
        if S.OBSERVING not in TRANSITIONS_3D[self.state]:
            self._transition(S.EXPLORING, detail)
            self._after_observe = S.EXPLORING
        self._transition(S.OBSERVING, detail)
        return True

    def supervisor_redirect(self, reason: str) -> bool:
        """Leave the current state for EXPLORING (goal dropped, not abandoned)."""
        if self.is_terminal() or self.state is S.INITIALIZING or self.state is S.EXPLORING:
            return False
        self._drop_goal(failure="redirect", status=None)
        self._enter()
        self._transition(S.EXPLORING, f"supervisor: {reason}")
        return True

    # -- state handlers -------------------------------------------------------------------------

    def _observe(self) -> Agent3DState | None:
        if self._maneuver_until > self.now:
            return None                                   # the tick applies the maneuver
        self._look(self._cam_yaw, self.cfg.cruise_pitch_deg)
        if self._phase_analyses >= 1 or self.now - self._phase_start > 1.5:
            nxt = self._after_observe
            self._after_observe = S.EXPLORING
            if nxt is S.SCANNING:
                self.metrics.scans += 1
            return self._go(nxt)
        return None

    def _scanning(self) -> Agent3DState | None:
        if self.world.chest_targets() and self._new_target:
            self._scan.clear()
            return self._go(S.TARGET_DETECTED)
        if not self._scan:
            return self._go(S.EXPLORING)
        pitch = self._scan_pitch if self._scan_pitch is not None else self.cfg.cruise_pitch_deg
        error = self._look(self._scan[0], pitch)
        if error < 8.0:
            if self._settled_since < 0:
                self._settled_since = self.now
            if self._last_analysis > self._settled_since + 1e-6:
                self._scan.pop(0)
                self._settled_since = -1.0
        if self.now - self._phase_start > 3.0 * (len(self._scan) + 2):
            self._scan.clear()                            # never get stuck scanning
        return None

    def _explore(self) -> Agent3DState:
        self.goal = None
        self.follower = None
        self._scan_pitch = None
        self.world.objective = "explore"
        if self.world.chest_targets():
            return S.TARGET_DETECTED
        if self.world.all_resolved:
            return self._complete("all spawn points resolved")
        pos = self.position
        speed = float(self.rules.get("max_speed", 5.0))
        candidates = []
        pool = [p for p in self.world.unresolved() if p.status is PriorStatus3D.UNKNOWN
                and p.location_id not in self._unreachable_keys] if self.planner else []
        for p in self.world.unresolved():
            if p.status is not PriorStatus3D.UNKNOWN or p.location_id in self._unreachable_keys:
                continue
            if p.inspections >= self.cfg.max_inspections:
                p.status = PriorStatus3D.ABANDONED
                self.metrics.abandoned_targets += 1
                continue
            straight = math.dist(pos, p.xyz)
            if self.planner is not None:
                key = self.planner.preselect_score(p, pos, straight, speed, pool)
            else:
                key = p.probability / (straight * 1.25 / speed + 1.0)
            candidates.append((key, p))
        if not candidates and self.explorer is not None:
            return self._explore_frontier()
        if not candidates:
            if self.world.all_resolved:
                return self._complete("all spawn points resolved")
            # Remaining priors are CHEST_SEEN without a live target: look around once.
            stale = [p for p in self.world.unresolved() if p.status is PriorStatus3D.CHEST_SEEN]
            for p in stale:
                p.status = PriorStatus3D.UNKNOWN
                p.inspections += 1
            if stale:
                return self._complete_or_scan()
            return self._complete("no reachable unresolved spawn points remain")
        candidates.sort(key=lambda c: (-c[0], c[1].location_id))
        best = None
        for _, p in candidates[:4]:
            length = self.navigator.path_length(pos, p.xyz, tolerance=2.5)
            if length is None:
                if self.navigator.last_status == "start_off_navmesh":
                    return self._wait_to_land(S.EXPLORING)
                if isinstance(self.navigator, LocalNavigator) and abs(p.z - pos[2]) <= 0.8:
                    continue                              # unknown to the local map: try later
                self._mark_unreachable(p)
                continue
            if self.planner is not None:
                score = self.planner.inspection_score(p, pos, length, speed, pool)
            else:
                score = p.probability / (length / speed + 1.0)
            if best is None or score > best[0]:
                best = (score, p)
        if best is None:
            return self._explore()           # bounded: each pass marks candidates unreachable
        prior = best[1]
        self.goal = Goal3D("inspect", prior.xyz, prior.location_id)
        self.world.objective = f"inspect {prior.name}"
        return S.PLANNING_ROUTE

    # -- Phase 7 active exploration ----------------------------------------------------------

    def _hfov(self) -> float:
        if self.obs is None:
            return 90.0
        intr = self.obs.intrinsics
        return math.degrees(2.0 * math.atan(intr.width / 2.0 / intr.fx))

    def _update_exploration(self, obs: Observation3D, dets: list[Detection3D],
                            estimates: list[RangeEstimate | None]) -> None:
        ex = self.explorer
        pos = obs.odometry.position
        self.world.note_floor(pos[2])
        if ex.needs_fan(pos):
            try:
                distances = self.port.nav_raycast(pos, ex.bearings(),
                                                  max_distance=ex.cfg.ray_reach_m)
            except BridgeError:                # the service may be unavailable: skip a fan
                self.metrics.service_errors += 1
                distances = None
            if distances is not None:
                ex.add_fan(pos, distances)
        cam = obs.camera_pose()
        ex.observe((cam.x, cam.y, cam.z), pos[2], cam.yaw_deg, self._hfov(), obs.sim_time)
        while ex.discoveries_noted < self.world.dynamic_created:
            prior = self.world.priors.get(-(ex.discoveries_noted + 1))
            if prior is None:
                break
            ex.note_discovery(prior.xyz)
        for det, est in zip(dets, estimates, strict=False):
            if est is not None and det.category == "chest":
                ex.note_detection(est.position, pos, det.confidence, confirmed=False)

    def _explore_frontier(self) -> Agent3DState:
        """No spawn point to inspect: choose the next exploration frontier (or finish)."""
        ex = self.explorer
        pos = self.position
        speed = float(self.rules.get("max_speed", 5.0))

        class _OffMesh(Exception):
            pass

        def length(point: tuple[float, float, float]) -> float | None:
            value = self.navigator.path_length(pos, point, tolerance=2.0)
            if value is None and self.navigator.last_status == "start_off_navmesh":
                raise _OffMesh
            return value

        tentative = [t.xyz for t in self.world.tracker.tracks
                     if t.category == "chest" and self.world.classify(t) != "approach"]
        try:
            with self.profiler.section("frontier_selection"):
                target = ex.select(pos, speed, length, tentative)
                if target is None and not ex.dense_fan_done(pos):
                    ex.add_dense_fan(pos, *self._dense_fan(pos))
                    target = ex.select(pos, speed, length, tentative)
        except _OffMesh:
            return self._wait_to_land(S.EXPLORING)
        if target is None:
            self.world.exploration_done = True
            if self.world.all_resolved:
                return self._complete("exploration complete: no reachable unexplored area left")
            stale = [p for p in self.world.unresolved() if p.status is PriorStatus3D.CHEST_SEEN]
            for p in stale:
                p.status = PriorStatus3D.UNKNOWN
                p.inspections += 1
            if stale:
                return self._complete_or_scan()
            return self._complete("exploration complete; remaining detections unreachable")
        self.goal = Goal3D("explore", target.point, None)
        self.world.objective = f"explore {target.region} ({target.gain:.0f} cells)"
        return S.PLANNING_ROUTE

    def _dense_fan(self, pos: tuple[float, float, float]
                   ) -> tuple[list[float], list[float] | None]:
        """The denser fan for the completion check, in requests of at most MAX_RAYS rays."""
        ex = self.explorer
        assert ex is not None
        bearings = ex.dense_bearings()
        distances: list[float] = []
        try:
            for i in range(0, len(bearings), MAX_RAYS):
                distances += self.port.nav_raycast(pos, bearings[i:i + MAX_RAYS],
                                                   max_distance=ex.cfg.ray_reach_m)
        except BridgeError:
            self.metrics.service_errors += 1
            return bearings, None
        return bearings, distances

    def _navigate_explore(self, goal: Goal3D, follower: PathFollower,
                          pos: tuple[float, float, float]) -> Agent3DState | None:
        ex = self.explorer
        if self.world.chest_targets() and self._new_target:
            return S.TARGET_DETECTED
        if ex is None:
            return S.EXPLORING
        if self.now - self._phase_start > 1.5 and self.now >= self._explore_check_at:
            self._explore_check_at = self.now + 1.5
            if not ex.target_still_useful():
                self._route_end(True)
                return S.EXPLORING                  # the area was seen on the way: replan
        if math.dist(pos[:2], goal.position[:2]) <= ex.cfg.arrive_radius_m or follower.done:
            self._route_end(True)
            if ex.target is not None:
                ex.reached(ex.target)
            self._scan = ex.scan_plan(pos, self._cam_yaw, self.now)
            self._scan_pitch = None
            self.goal = None
            if self._scan:
                self.metrics.scans += 1
                return self._go(S.SCANNING)
            return S.EXPLORING
        return None

    def _complete_or_scan(self) -> Agent3DState:
        self._scan = ScanPlanner.full_turn(self._cam_yaw, self.cfg.scan_step_deg)
        self.metrics.scans += 1
        return S.SCANNING

    def _wait_to_land(self, then: Agent3DState) -> Agent3DState:
        """The agent is momentarily off the navigation mesh (falling, squeezed against
        geometry): wait briefly and retry instead of declaring goals unreachable."""
        self._off_mesh_waits += 1
        if self._off_mesh_waits > 12:
            self._off_mesh_waits = 0
            self._maneuver_bearing = self._retrace_bearing()
            self._maneuver_until = self.now + 0.8
        self._after_observe = then
        return self._go(S.OBSERVING)

    def _retrace_bearing(self) -> float:
        """Direction back along the agent's own recent trail (known to be walkable)."""
        pos = self.position
        for t, p in reversed(self._trail):
            if self.now - t >= 1.2 and math.dist(p[:2], pos[:2]) > 0.6:
                return bearing_deg(p[0] - pos[0], p[1] - pos[1])
        return (self._cam_yaw + 180.0) % 360.0

    def _mark_unreachable(self, prior: SpawnPrior3D) -> None:
        if prior.location_id in self._unreachable_keys:
            return
        self._unreachable_keys.add(prior.location_id)
        prior.status = PriorStatus3D.UNREACHABLE
        self.metrics.unreachable_targets += 1

    def _target_detected(self) -> Agent3DState:
        self._new_target = False
        targets = self.world.chest_targets()
        if not targets:
            return S.EXPLORING
        pos = self.position
        if self.planner is not None:
            track = self.planner.choose_track(targets, pos, float(self.rules.get("max_speed", 5.0)))
        else:
            track = min(targets, key=lambda t: (math.dist(pos, t.xyz), t.track_id))
        self.goal = Goal3D("chest", track.xyz, track.location_id, track.track_id)
        self.metrics.targets_selected += 1
        self._approach_side = 0.0
        self._approach_stand = self.cfg.stand_distance_m
        self._approach_goal = None
        self._approach_path = None
        self.world.objective = f"chest track T{track.track_id}"
        if math.dist(pos[:2], track.xyz[:2]) <= self.cfg.approach_switch_m \
                and abs(track.xyz[2] - pos[2]) <= 0.6:
            return self._go(S.APPROACHING)
        return S.PLANNING_ROUTE

    def _plan_route(self) -> Agent3DState:
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        tolerance = {"chest": 1.4, "explore": 2.0}.get(goal.kind, 2.5)
        plan = self.navigator.plan(self.position, goal.position, tolerance=tolerance)
        if plan is None and goal.kind == "explore" \
                and self.navigator.last_status != "start_off_navmesh":
            self.metrics.navigation_failures += 1
            if self.explorer is not None and self.explorer.target is not None:
                self.explorer.unreachable(self.explorer.target)
            self.goal = None
            self._after_observe = S.EXPLORING
            return self._go(S.OBSERVING)
        if plan is None and self.navigator.last_status == "start_off_navmesh":
            return self._wait_to_land(S.PLANNING_ROUTE)
        if plan is None:
            self.metrics.navigation_failures += 1
            if self.planner is not None:
                self.planner.route_failed_immediately(goal, self.position, self.now,
                                                      "target_unreachable")
            prior = self.world.priors.get(goal.location_id) if goal.location_id else None
            if prior is not None:
                # The agent's own map (local mode) is incomplete: a failed plan is weak
                # evidence. Only a goal on another floor is out of reach for that mode.
                n = self._plan_failures.get(prior.location_id, 0) + 1
                self._plan_failures[prior.location_id] = n
                definitive = isinstance(self.navigator, NavMeshNavigator) \
                    or self.navigator.last_status == "other_floor" or n >= 3
                if definitive:
                    self.metrics.recovery_log.record(self._target_key(goal),
                                                     "target_unreachable")
                    self._mark_unreachable(prior)
                else:
                    prior.inspections += 1
            self.goal = None
            # Go through a timed state so a run of failed plans spends simulated time
            # instead of chaining instant transitions within one tick.
            self._after_observe = S.EXPLORING
            return self._go(S.OBSERVING)
        self._off_mesh_waits = 0
        self.metrics.routes_started += 1
        self.follower = PathFollower(plan.waypoints, arrive_radius=0.7)
        self.follower.start(self.now)
        if self.planner is not None:
            self.planner.route_started(goal, self.position, plan.length,
                                       float(self.rules.get("max_speed", 5.0)), self.now)
        return self._go(S.NAVIGATING)

    def _navigate(self) -> Agent3DState | None:
        goal, follower = self.goal, self.follower
        if goal is None or follower is None:
            return S.EXPLORING
        pos = self.position
        if goal.kind == "chest":
            track = self.world.tracker.get(goal.track_id or -1)
            if track is None:
                self.metrics.targets_lost += 1
                self._route_end(False, "target_lost")
                if self._smart_target_lost(goal):
                    return self._go(S.SCANNING)
                return S.EXPLORING
            if math.dist(track.xyz, goal.position) > 1.5:
                goal.position = track.xyz
                return S.PLANNING_ROUTE                     # target estimate moved: replan
            goal.position = track.xyz
            if math.dist(pos[:2], goal.position[:2]) <= self.cfg.approach_switch_m \
                    and abs(goal.position[2] - pos[2]) <= 0.6:
                self._route_end(True)
                return self._go(S.APPROACHING)
        elif goal.kind == "explore":
            nxt = self._navigate_explore(goal, follower, pos)
            if nxt is not None:
                return nxt
        else:
            prior = self.world.priors.get(goal.location_id) if goal.location_id else None
            if prior is None or prior.resolved or prior.status is PriorStatus3D.CHEST_SEEN:
                return S.EXPLORING if not self.world.chest_targets() else S.TARGET_DETECTED
            if self._new_target and self.world.chest_targets():
                return S.TARGET_DETECTED
            near = math.dist(pos[:2], prior.xyz[:2]) <= self.cfg.inspect_range_m \
                and abs(prior.z - pos[2]) <= 0.6
            if near or follower.done:
                self._route_end(True)
                prior.inspections += 1
                self._scan = [self._view_yaw(prior.xyz)] * 2
                cam = self.obs.camera_pose() if self.obs is not None else None
                if cam is not None:
                    horizontal = max(0.5, math.dist(cam.position[:2], prior.xyz[:2]))
                    self._scan_pitch = math.degrees(math.atan2(prior.z + 0.3 - cam.z, horizontal))
                return self._go(S.SCANNING)
        if follower.stuck(self.now):
            self.recovery_reason = "stuck"
            self._route_end(False, "path_blocked")
            self.metrics.navigation_failures += 1
            self.navigator.note_obstacle(pos, self._cam_yaw)
            return S.RECOVERING
        bearing, speed = follower.steer(pos, self.now)
        if follower.done and goal.kind == "chest":
            self._route_end(True)
            return self._go(S.APPROACHING)
        self._move(bearing, speed)
        if goal.kind == "chest" and self._near_goal(12.0):
            self._aim_at(goal.position)
        elif goal.kind == "explore" and self.explorer is not None:
            self._look(self._explore_look_yaw(bearing), self.cfg.cruise_pitch_deg)
        else:
            self._look(bearing, self.cfg.cruise_pitch_deg)
        return None

    def _explore_look_yaw(self, bearing: float) -> float:
        """Phase 7 visual inspection while travelling: look towards the most informative
        sector within 75 degrees of the travel direction (movement is relative to the camera,
        so the agent can walk one way and look another). The choice is held for 1.5 s."""
        if self.now < self._look_until:
            return self._look_yaw
        ex = self.explorer
        sectors = ex.sectors
        gains = ex.unseen_by_sector(self.position)
        best, best_key = bearing, (-1.0, 0.0)
        for k in range(sectors.n):
            centre = sectors.centre(k)
            off = abs(angle_difference_deg(centre, bearing))
            if off > 75.0 or gains[k] < 2.0:
                continue
            key = (gains[k], -off)
            if key > best_key:
                best, best_key = centre, key
        self._look_yaw = best
        self._look_until = self.now + 1.5
        return best

    def _approach(self) -> Agent3DState | None:
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        track = self.world.tracker.get(goal.track_id or -1)
        if track is None:
            self.metrics.targets_lost += 1
            if self._may_rescan_lost_approach(goal):
                self.recovery_reason = "target_lost"
                return S.RECOVERING
            return S.EXPLORING
        goal.position = track.xyz
        pos = self.position
        tx, ty, _ = goal.position
        dist = math.hypot(tx - pos[0], ty - pos[1])
        self._aim_at(goal.position)
        if self.cfg.smart_recovery and track.sigma > self.cfg.uncertain_sigma_m and dist < 6.0:
            key = self._target_key(goal)
            if self._uncertain_checks.get(key, 0) < 2 \
                    and self.metrics.recovery_log.can_use("localisation_uncertainty"):
                self._uncertain_checks[key] = self._uncertain_checks.get(key, 0) + 1
                self.recovery_reason = "localisation_uncertainty"
                return S.RECOVERING
        if dist <= self._approach_stand + self.cfg.arrive_margin_m:
            self.metrics.targets_reached += 1
            self.metrics.recovery_log.resolve(("reach", self._target_key(goal)))
            return self._go(S.ALIGNING)
        if dist < self._approach_progress[0] - 0.15:
            self._approach_progress = (dist, self.now)
        elif self.now - self._approach_progress[1] > 2.5:
            self._approach_progress = (math.inf, self.now)
            self.recovery_reason = "approach_blocked"
            self.metrics.navigation_failures += 1
            return S.RECOVERING
        if self.now - self._phase_start > 15.0:
            self.recovery_reason = "approach_timeout"
            return S.RECOVERING
        # Stand on the line from the chest towards the agent, at stand distance. If the
        # straight line there is obstructed (bushes, crates), follow a short planned path.
        ux, uy = (pos[0] - tx) / max(dist, 1e-6), (pos[1] - ty) / max(dist, 1e-6)
        if self._approach_side:                       # retry from another side (occluders)
            rad = math.radians(self._approach_side)
            cr, sr = math.cos(rad), math.sin(rad)
            ux, uy = ux * cr - uy * sr, ux * sr + uy * cr
        sx, sy = tx + ux * self._approach_stand, ty + uy * self._approach_stand
        stand = (sx, sy, pos[2])
        if self._approach_goal is None or math.dist(self._approach_goal[:2], stand[:2]) > 0.7:
            self._approach_goal = stand
            self._approach_path = None
            if not self.navigator.clear_line(pos, stand):
                plan = self.navigator.plan(pos, stand, tolerance=0.8)
                if plan is not None:
                    self._approach_path = PathFollower(plan.waypoints, arrive_radius=0.4)
                    self._approach_path.start(self.now)
        if self._approach_path is not None and not self._approach_path.done:
            bearing, speed = self._approach_path.steer(pos, self.now)
            self._move(bearing, max(0.4, speed))
            return None
        remaining = math.hypot(sx - pos[0], sy - pos[1])
        self._move(bearing_deg(sx - pos[0], sy - pos[1]), min(1.0, max(0.3, remaining / 2.0)))
        return None

    def _align(self) -> Agent3DState | None:
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        track = self.world.tracker.get(goal.track_id or -1)
        if track is not None:
            goal.position = track.xyz
        error = self._aim_at(goal.position)
        opened, closed = self._target_appearance()
        if opened and not closed and goal.location_id in self._unverified:
            self._credit_open(goal.location_id, self._unverified[goal.location_id], None)
            return S.EXPLORING
        if error < 10.0 and self.cues.prompt and self._phase_analyses >= 1:
            self._hold_started = self.now
            self._verify = {"opened": 0, "closed": 0, "reward": 0}
            self.metrics.interaction_attempts += 1
            return self._go(S.INTERACTING)
        if self.now - self._phase_start > 1.6:
            pos = self.position
            dist = math.dist(pos[:2], goal.position[:2])
            if self._approach_stand > 0.8 and dist > 0.9:
                self._approach_stand -= 0.3                # step closer and try again
                self._approach_progress = (math.inf, self.now)
                return self._go(S.APPROACHING)
            self.recovery_reason = "alignment_failed"
            return S.RECOVERING
        return None

    def _interact(self) -> Agent3DState | None:
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        self.action.interact = True
        self._aim_at(goal.position)
        self.action.look_yaw *= 0.3                        # keep the view steady while holding
        self.action.look_pitch *= 0.3
        held = self.now - self._hold_started
        if 0.6 <= held <= self.interaction_seconds - 0.3 and self._phase_analyses >= 3 \
                and not self.cues.progress and not self.cues.prompt:
            self.recovery_reason = "interaction_interrupted"
            return S.RECOVERING
        if held + 1e-6 >= self.interaction_seconds + self.step_seconds:
            if goal.location_id is not None:
                self._unverified[goal.location_id] = held
            return self._go(S.VERIFYING_OPEN)
        return None

    def _verify_open(self) -> Agent3DState | None:
        goal = self.goal
        if goal is None:
            return S.EXPLORING
        self._aim_at(goal.position)
        if self._phase_analyses == 0:
            return None
        if self._last_analysis >= self._phase_start:
            opened, closed = self._target_appearance()
            if self._settled_since != self._last_analysis:
                self._settled_since = self._last_analysis
                self._verify["opened"] += int(opened and not closed)
                self._verify["closed"] += int(closed and not opened)
        seconds = self.now - self._hold_started
        v = self._verify
        # Visual evidence alone must be unanimous: two "opened" looks against two "closed"
        # ones, without the opening notification, credited a chest whose interaction had
        # failed (Phase 8 audit F20, opened/closed confusion of single frames).
        if (v["opened"] >= 2 and v["closed"] == 0) or (v["opened"] >= 1 and v["reward"] >= 1):
            if goal.location_id is not None:
                reward = v["reward"] > 0 and self._last_notification_reward
                self._credit_open(goal.location_id, seconds, reward=reward)
            self.goal = None
            return S.EXPLORING
        elapsed = self.now - self._phase_start
        if v["closed"] >= 3 and v["reward"] == 0 and elapsed >= 0.8:
            self._record(goal.location_id, present=True, opened=False, seconds=seconds, reward=None)
            self.metrics.failed_interactions += 1
            if goal.location_id is not None:
                self._unverified.pop(goal.location_id, None)
            self.recovery_reason = "interaction_failed"
            return S.RECOVERING
        if elapsed > 2.0:
            # The agent's own "chest opened" notification arrived but the opened lid was not
            # seen clearly (Phase 8 audit F11: some opened chests read as closed from the
            # verification pose): look again from a step back, bounded, then credit the
            # opening on the notification - counted separately, never as visually verified.
            if v["reward"] >= 1 and goal.location_id is not None:
                return self._look_back_or_credit(goal, seconds)
            self.recovery_reason = "verification_inconclusive"
            return S.RECOVERING
        return None

    # -- Phase 5 helpers -------------------------------------------------------------------

    @staticmethod
    def _target_key(goal: Goal3D) -> object:
        return (goal.kind, goal.location_id, goal.track_id)

    def _route_end(self, success: bool, kind: str | None = None) -> None:
        if self.planner is not None:
            self.planner.route_finished(self.now, success=success, failure_kind=kind)

    def _may_rescan_lost_approach(self, goal: Goal3D) -> bool:
        """Smart recovery: a chest lost during the approach gets a bounded rescan.

        The limit is per spawn point, not per track: a chest that keeps dropping out of
        tracking is re-detected under a new track id every time, so a per-track limit never
        triggers (seen on an unseen procedural map, where one chest was lost 100 times in a
        row and the episode failed on the global recovery budget). Beyond the limit, or once
        the ``target_lost`` class budget is spent, the Phase 4 behaviour applies: explore on.
        """
        if not self.cfg.smart_recovery or not self.metrics.recovery_log.can_use("target_lost"):
            return False
        key = ("lost", goal.location_id) if goal.location_id is not None \
            else self._target_key(goal)
        if self._approach_losses.get(key, 0) >= self.cfg.max_rescans:
            return False
        self._approach_losses[key] = self._approach_losses.get(key, 0) + 1
        return True

    def _smart_target_lost(self, goal: Goal3D) -> bool:
        """Search near the last reliable estimate of a lost chest (bounded)."""
        if not self.cfg.smart_recovery:
            return False
        key = self._target_key(goal)
        if self._rescans.get(key, 0) >= self.cfg.max_rescans \
                or not self.metrics.recovery_log.can_use("target_lost"):
            return False
        self._rescans[key] = self._rescans.get(key, 0) + 1
        self.metrics.recovery_log.record(key, "target_lost")
        self.metrics.recovery_log.use("target_lost")
        centre = self._view_yaw(goal.position)
        self._scan = [centre, (centre + 40.0) % 360.0, (centre - 40.0) % 360.0]
        self._scan_pitch = None
        self.goal = None
        return True

    def _look_back_or_credit(self, goal: Goal3D, seconds: float) -> Agent3DState | None:
        """The agent's own opening notification arrived but the opened chest was not seen
        clearly: back off a little, look again (bounded), then credit the opening on the
        notification (counted separately as ``credited_by_notification``)."""
        key = self._target_key(goal)
        n = self._lookbacks.get(key, 0)
        if n < self.cfg.max_lookbacks and self.metrics.recovery_log.can_use("interaction_failure"):
            self._lookbacks[key] = n + 1
            self.metrics.recovery_log.record(key, "interaction_failure")
            self.metrics.recovery_log.use("interaction_failure")
            pos = self.position
            self._maneuver_bearing = bearing_deg(pos[0] - goal.position[0],
                                                 pos[1] - goal.position[1])
            self._maneuver_until = self.now + 0.4
            self._phase_start = self.now
            self._verify["opened"] = self._verify["closed"] = 0
            return None
        assert goal.location_id is not None
        if self._credit_open(goal.location_id, seconds, reward=self._last_notification_reward):
            self.metrics.credited_by_notification += 1
        self.goal = None
        return S.EXPLORING

    def _smart_recover(self, goal: Goal3D, reason: str, cls: str) -> Agent3DState | None:
        log = self.metrics.recovery_log
        if not log.can_use(cls):
            return None
        if cls == "target_lost":
            log.use(cls)
            centre = self._view_yaw(goal.position)
            self._scan = [centre, (centre + 40.0) % 360.0, (centre - 40.0) % 360.0]
            self._after_observe = S.SCANNING
            self.goal = None
            return self._go(S.OBSERVING)
        if cls == "localisation_uncertainty":
            log.use(cls)
            self._scan = [self._view_yaw(goal.position)] * 2
            self._after_observe = S.SCANNING
            return self._go(S.OBSERVING)
        if reason == "interaction_failed":
            # Recheck distance and alignment before trying again.
            log.use(cls)
            dist = math.dist(self.position[:2], goal.position[:2])
            if dist > self.cfg.stand_distance_m + 0.2:
                self._approach_stand = max(0.8, self.cfg.stand_distance_m - 0.2)
                self._approach_progress = (math.inf, self.now)
                key: object = (goal.kind, goal.location_id, goal.track_id)
                self._attempts[key] = self._attempts.get(key, 0) + 1
                return self._go(S.APPROACHING)
        return None

    def _recover(self) -> Agent3DState:
        self.metrics.recoveries += 1
        self._total_recoveries += 1
        reason = self.recovery_reason or "unknown"
        self.recovery_reason = None
        self.metrics.recovery_reasons[reason] = self.metrics.recovery_reasons.get(reason, 0) + 1
        # Discovery mode starts with no spawn points: the budget grows with the chests found.
        self.recovery_budget = max(self.recovery_budget,
                                   (self.cfg.max_attempts + 1) * (len(self.world.priors) + 5))
        if self._total_recoveries > self.recovery_budget:
            return self._fail(f"recovery budget of {self.recovery_budget} exhausted ({reason})")
        goal = self.goal
        cls = classify(reason)
        if cls == "path_blocked" and isinstance(self.navigator, NavMeshNavigator):
            # a cached path the agent could not follow must not be handed out again (the
            # world may have changed: a new obstacle, a closed door)
            self.navigator.clear_cache()
        if goal is not None:
            self.metrics.recovery_log.record(("reach", self._target_key(goal))
                                             if cls == "path_blocked" else self._target_key(goal),
                                             cls)
        if self.planner is not None:
            self.planner.recovery()
        if goal is None:
            return S.EXPLORING
        if self.cfg.smart_recovery:
            smart = self._smart_recover(goal, reason, cls)
            if smart is not None:
                return smart
        key: object = (goal.kind, goal.location_id, goal.track_id)
        self._attempts[key] = self._attempts.get(key, 0) + 1
        interaction = reason in ("interaction_failed", "interaction_interrupted",
                                 "verification_inconclusive", "alignment_failed")
        give_up = self._attempts[key] > self.cfg.max_attempts
        if reason == "alignment_failed" and goal.location_id is not None \
                and goal.location_id < 0:
            # A discovered chest whose prompt never appears from two sides is most likely
            # not an unopened chest (a misdetection, or an opened one seen again).
            self._align_failures[key] = self._align_failures.get(key, 0) + 1
            give_up = give_up or self._align_failures[key] >= 2
        if give_up:
            if goal.kind == "explore" and self.explorer is not None:
                self.explorer.note_failure(goal.position)
                if self.explorer.target is not None:
                    self.explorer.unreachable(self.explorer.target)
            prior = self.world.priors.get(goal.location_id) if goal.location_id else None
            if prior is not None and not prior.resolved:
                prior.status = PriorStatus3D.FAILED if interaction else PriorStatus3D.ABANDONED
                self.metrics.abandoned_targets += 1
            self.goal = None
            return S.EXPLORING
        if reason in ("stuck", "approach_blocked", "approach_timeout"):
            # Back off along the way we came: unlike a blind side-step it cannot walk
            # the agent off a staircase or platform edge.
            self._maneuver_bearing = self._retrace_bearing()
            self._maneuver_until = self.now + 0.7
            self._after_observe = S.PLANNING_ROUTE
            return self._go(S.OBSERVING)
        if reason == "alignment_failed":
            # Something may block the interaction from this side (a hay bale, a crate):
            # walk round to another side of the chest.
            self._approach_side = (100.0, -100.0, 180.0)[(self._attempts[key] - 1) % 3]
            self._approach_stand = self.cfg.stand_distance_m
            self._approach_goal = None
            self._approach_path = None
            self._approach_progress = (math.inf, self.now)
            return self._go(S.APPROACHING)
        if interaction:
            return self._go(S.ALIGNING)
        return S.PLANNING_ROUTE


def _pose_close(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return math.dist(a[:3], b[:3]) <= 0.05 and abs(angle_difference_deg(a[3], b[3])) <= 0.5 \
        and abs(a[4] - b[4]) <= 0.5


def _hsv(frame: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2HSV)


# -- oracle baseline ----------------------------------------------------------------------------


class OracleAgent3D:
    """Benchmark baseline with exact chest positions (privileged oracle channel)."""

    def __init__(self, port: Any, oracle: Callable[[], dict[str, Any]], *, rules: dict[str, Any],
                 location_of: Callable[[int], int | None], recorder: Recorder | None = None,
                 decision_ticks: int = 2) -> None:
        self.port = port
        self.oracle = oracle
        self.rules = rules
        self.location_of = location_of
        self.recorder = recorder
        self.step_seconds = decision_ticks / int(rules.get("tick_rate", 30))
        self.interaction_seconds = float(rules.get("interaction_seconds", 2.0))
        self.metrics = Metrics3D()
        self.navigator = NavMeshNavigator(port)
        self.action = ActionCommand()
        self.telemetry: Telemetry | None = None
        self.state = "PLANNING"
        self.outcome: str | None = None
        self.end_reason: str | None = None
        self.target: dict[str, Any] | None = None
        self.follower: PathFollower | None = None
        self.skipped: set[int] = set()
        self._hold_start = -1.0
        self.now = 0.0
        self.decisions = 0

    def initialize(self, telemetry: Telemetry) -> None:
        self.telemetry = telemetry

    def is_terminal(self) -> bool:
        return self.state in ("COMPLETED", "FAILED")

    def stop(self, outcome: str, reason: str) -> None:
        if not self.is_terminal():
            self.outcome, self.end_reason = outcome, reason
            self.state = "COMPLETED" if outcome in ("completed", "tick_limit") else "FAILED"

    def wants_observation(self) -> bool:
        return False

    def on_step(self, result: StepResult) -> None:
        if self.telemetry is not None:
            a, b = self.telemetry.odometry.position, result.telemetry.odometry.position
            self.metrics.travel_distance += math.hypot(b[0] - a[0], b[1] - a[1])
        self.telemetry = result.telemetry
        self.now = result.telemetry.sim_time
        for e in result.telemetry.events:
            if e.get("type") == "chest_opened_notification" and e.get("reward"):
                self.metrics.rewards += 1

    def tick(self) -> str:
        self.decisions += 1
        self.action = ActionCommand()
        if self.is_terminal():
            return self.state
        state = self.oracle()
        me = state["agent"]["position"]
        cam_yaw = state["agent"]["camera_yaw_deg"]
        if self.state == "PLANNING":
            todo = [c for c in state["chests"] if c["present"] and not c["opened"]
                    and c["spawn_index"] not in self.skipped]
            if not todo:
                self.stop("completed", "all present chests opened")
                return self.state
            todo.sort(key=lambda c: math.dist(me, c["position"]))
            for c in todo:
                plan = self.navigator.plan(tuple(me), tuple(c["position"]), tolerance=1.3)
                if plan is not None:
                    self.target = c
                    self.follower = PathFollower(plan.waypoints, arrive_radius=0.5)
                    self.follower.start(self.now)
                    self.metrics.targets_selected += 1
                    self.state = "MOVING"
                    break
                self.skipped.add(c["spawn_index"])
                self.metrics.unreachable_targets += 1
            else:
                return self.state
        assert self.target is not None and self.follower is not None
        tx, ty, tz = self.target["position"]
        dist = math.hypot(tx - me[0], ty - me[1])
        aim_yaw = bearing_deg(tx - me[0], ty - me[1])
        if self.state == "MOVING":
            if dist <= 1.35 or self.follower.done:
                self.state, self._hold_start = "OPENING", self.now
                self.metrics.targets_reached += 1
            elif self.follower.stuck(self.now):
                self.metrics.navigation_failures += 1
                self.follower.start(self.now)
                f, r = move_towards((cam_yaw + 150) % 360, cam_yaw, 0.8)
                self.action.move_forward, self.action.move_right = f, r
                return self.state
            else:
                bearing, speed = self.follower.steer(tuple(me), self.now)
                f, r = move_towards(bearing, cam_yaw, speed)
                self.action.move_forward, self.action.move_right = f, r
                ly, lp = look_towards(cam_yaw, self.telemetry.camera.pitch_deg if self.telemetry
                                      else 0.0, bearing, -12.0, step_seconds=self.step_seconds,
                                      yaw_rate=float(self.rules.get("yaw_rate_deg", 120)),
                                      pitch_rate=float(self.rules.get("pitch_rate_deg", 60)))
                self.action.look_yaw, self.action.look_pitch = ly, lp
        if self.state == "OPENING":
            pitch = self.telemetry.camera.pitch_deg if self.telemetry else 0.0
            ly, lp = look_towards(cam_yaw, pitch, aim_yaw, -25.0, step_seconds=self.step_seconds,
                                  yaw_rate=float(self.rules.get("yaw_rate_deg", 120)),
                                  pitch_rate=float(self.rules.get("pitch_rate_deg", 60)))
            self.action.look_yaw, self.action.look_pitch = ly, lp
            current = next(c for c in state["chests"]
                           if c["spawn_index"] == self.target["spawn_index"])
            if current["opened"]:
                loc = self.location_of(self.target["spawn_index"])
                self.metrics.chests_opened += 1
                if loc is not None and self.recorder is not None:
                    self.recorder(loc, present=True, opened=True,
                                  seconds=self.now - self._hold_start, reward=None,
                                  sim_time=self.now)
                    self.metrics.locations_recorded += 1
                self.state = "PLANNING"
                self.target = None
                return self.state
            if abs(angle_difference_deg(aim_yaw, cam_yaw)) < 30.0:
                self.action.interact = True
            if self.now - self._hold_start > self.interaction_seconds + 4.0:
                self.metrics.failed_interactions += 1
                self.skipped.add(self.target["spawn_index"])
                self.state = "PLANNING"
        return self.state
