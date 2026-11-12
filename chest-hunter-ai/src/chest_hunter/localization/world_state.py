"""The visual agent's estimated world model.

Strict separation of information:

* **Ground truth** (simulator only): true obstacles, chest presence,
  collisions, rewards.
* **Estimated state** (this module, the only thing the visual policy uses):
  pose from marker fixes plus dead reckoning, an occupancy grid built from
  frames, chest tracks from detections, and spawn *priors* from the
  database. A prior is a place where a chest *might* be, never proof that
  one spawned.

The model is updated incrementally after each analysed frame and its memory
is bounded: a fixed-size grid, a capped tracker, and one record per spawn
prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from chest_hunter.localization.pose_estimator import PoseEstimator
from chest_hunter.simulation.collision import GridMap
from chest_hunter.vision.obstacle_detector import BLOCKED, UNKNOWN, OccupancyGrid
from chest_hunter.vision.perception_pipeline import PerceptionResult
from chest_hunter.vision.visual_tracker import ObjectTrack, TrackState, VisualTracker

ASSOCIATION_RADIUS = 1.0     # cells between a chest track and a spawn prior
OCCLUSION_RADIUS = 0.9       # cells around the agent marker that may hide a chest


class SpawnStatus(StrEnum):
    UNKNOWN = "unknown"
    EMPTY = "empty"              # seen clearly, no chest
    CHEST_SEEN = "chest_seen"    # an unopened chest is tracked here
    OPENED = "opened"            # opened by this agent and visually verified
    FAILED = "failed"            # interaction attempts exhausted
    UNREACHABLE = "unreachable"  # no path on the estimated map
    ABANDONED = "abandoned"      # navigation recovery exhausted


RESOLVED_STATUSES = frozenset({SpawnStatus.EMPTY, SpawnStatus.OPENED, SpawnStatus.FAILED,
                               SpawnStatus.UNREACHABLE, SpawnStatus.ABANDONED})


@dataclass
class SpawnPrior:
    location_id: int
    name: str
    x: float
    y: float
    probability: float
    status: SpawnStatus = SpawnStatus.UNKNOWN
    empty_votes: int = 0
    track_id: int | None = None

    @property
    def resolved(self) -> bool:
        return self.status in RESOLVED_STATUSES


@dataclass(frozen=True)
class EstimatedWorldState:
    """Read-only snapshot used by planning, the GUI and tests."""

    agent_position: tuple[float, float] | None
    agent_heading: float | None
    pose_confidence: float
    chest_tracks: tuple[ObjectTrack, ...]
    occupancy_grid: np.ndarray
    timestamp: float
    explored_fraction: float
    objective: str | None


@dataclass
class WorldUpdate:
    pose_fixed: bool
    obstacles_discovered: int
    newly_empty: list[SpawnPrior] = field(default_factory=list)
    new_tracks: list[ObjectTrack] = field(default_factory=list)


class WorldModel:
    def __init__(self, width: int, height: int, priors: list[SpawnPrior], *,
                 pose: PoseEstimator, tracker: VisualTracker, min_confidence: float = 0.5,
                 empty_confirmations: int = 2, oracle_grid: GridMap | None = None) -> None:
        self.width = width
        self.height = height
        self.pose = pose
        self.tracker = tracker
        self.min_confidence = min_confidence
        self.empty_confirmations = empty_confirmations
        self.priors = {p.location_id: p for p in sorted(priors, key=lambda p: p.location_id)}
        self.occupancy = OccupancyGrid(width, height)
        self.planning_grid = GridMap(width, height)
        self.unknown_cells = bytearray(width * height)
        self._blocked_view = np.frombuffer(self.planning_grid.cells, dtype=np.uint8).reshape(
            height, width)
        self._unknown_view = np.frombuffer(self.unknown_cells, dtype=np.uint8).reshape(
            height, width)
        self.oracle_obstacles = oracle_grid is not None
        if oracle_grid is not None:  # "chest vision" benchmark mode: obstacles are given
            truth = np.frombuffer(oracle_grid.cells, dtype=np.uint8).reshape(height, width)
            self.occupancy.cells[:] = np.where(truth == 1, BLOCKED, 0).astype(np.int8)
            self.occupancy.version += 1
        self._sync_planning_grid((0, width, 0, height))
        self.objective: str | None = None
        self.timestamp = 0.0
        self.frames_integrated = 0

    # -- incremental update ------------------------------------------------------------

    def integrate(self, result: PerceptionResult) -> WorldUpdate:
        obs = result.observation
        self.timestamp = obs.timestamp
        if result.pose is not None:
            self.pose.correct(result.pose)
        else:
            self.pose.miss(obs.timestamp)
        discovered = 0
        if result.occupancy is not None and result.occupancy.changed:
            discovered = result.occupancy.newly_blocked
            self._sync_planning_grid(result.occupancy.window)

        agent = self.pose.position
        camera = obs.camera

        def in_view(x: float, y: float) -> bool:
            if not camera.contains(x, y, margin=0.5):
                return False
            return agent is None or math.dist((x, y), agent) > OCCLUSION_RADIUS

        new_tracks = self.tracker.update(result.world_detections, obs.timestamp, in_view)
        self._associate_tracks()
        newly_empty = self._vote_empty(in_view, result.pose is not None)
        self.frames_integrated += 1
        return WorldUpdate(result.pose is not None, discovered, newly_empty, new_tracks)

    def _sync_planning_grid(self, window: tuple[int, int, int, int]) -> None:
        x0, x1, y0, y1 = window
        occ = self.occupancy.cells[y0:y1, x0:x1]
        blocked = (occ == BLOCKED).astype(np.uint8)
        unknown = (occ == UNKNOWN).astype(np.uint8)
        if not (np.array_equal(self._blocked_view[y0:y1, x0:x1], blocked)
                and np.array_equal(self._unknown_view[y0:y1, x0:x1], unknown)):
            self._blocked_view[y0:y1, x0:x1] = blocked
            self._unknown_view[y0:y1, x0:x1] = unknown
            self.planning_grid.version += 1

    def _associate_tracks(self) -> None:
        # A spawn point whose chest track expired is unresolved again (look again).
        for prior in self.priors.values():
            if prior.status is SpawnStatus.CHEST_SEEN and (
                    prior.track_id is None or self.tracker.get(prior.track_id) is None):
                prior.status = SpawnStatus.UNKNOWN
                prior.track_id = None
                prior.empty_votes = 0
        for track in self.tracker.tracks:
            if track.location_id is None:
                nearest = self.nearest_prior(track.estimated_position, ASSOCIATION_RADIUS,
                                             exclude_tracked=True)
                if nearest is not None:
                    track.location_id = nearest.location_id
                    nearest.track_id = track.track_id
            linked = self.priors.get(track.location_id) if track.location_id else None
            if linked is not None and track.state is not TrackState.TENTATIVE:
                linked.empty_votes = 0
                if linked.status in (SpawnStatus.UNKNOWN, SpawnStatus.EMPTY) \
                        and track.category == "chest":
                    linked.status = SpawnStatus.CHEST_SEEN

    def _vote_empty(self, in_view: object, pose_ok: bool) -> list[SpawnPrior]:
        assert callable(in_view)
        if not pose_ok or self.pose.confidence() < self.min_confidence:
            return []
        newly: list[SpawnPrior] = []
        tracks = [t for t in self.tracker.tracks if t.state is not TrackState.EXPIRED]
        for prior in self.priors.values():
            if prior.status is not SpawnStatus.UNKNOWN or not in_view(prior.x, prior.y):
                continue
            if any(math.dist(t.estimated_position, (prior.x, prior.y)) <= ASSOCIATION_RADIUS
                   for t in tracks):
                prior.empty_votes = 0
                continue
            prior.empty_votes += 1
            if prior.empty_votes >= self.empty_confirmations:
                prior.status = SpawnStatus.EMPTY
                newly.append(prior)
        return newly

    # -- queries ---------------------------------------------------------------------

    def nearest_prior(self, position: tuple[float, float], radius: float, *,
                      exclude_tracked: bool = False) -> SpawnPrior | None:
        best, best_d = None, radius
        for prior in self.priors.values():
            if exclude_tracked and prior.track_id is not None \
                    and self.tracker.get(prior.track_id) is not None:
                continue
            d = math.dist(position, (prior.x, prior.y))
            if d <= best_d:
                best, best_d = prior, d
        return best

    def chest_targets(self) -> list[ObjectTrack]:
        """Confirmed unopened chest tracks at unresolved spawn points."""
        targets = []
        for track in self.tracker.confirmed():
            if track.category != "chest" or track.location_id is None:
                continue
            prior = self.priors.get(track.location_id)
            if prior is not None and not prior.resolved:
                targets.append(track)
        return sorted(targets, key=lambda t: t.track_id)

    def unresolved_priors(self) -> list[SpawnPrior]:
        return [p for p in self.priors.values() if not p.resolved]

    @property
    def all_priors_resolved(self) -> bool:
        return all(p.resolved for p in self.priors.values())

    def snapshot(self) -> EstimatedWorldState:
        view = self.occupancy.cells.view()
        view.flags.writeable = False
        return EstimatedWorldState(
            agent_position=self.pose.position,
            agent_heading=self.pose.heading,
            pose_confidence=self.pose.confidence(),
            chest_tracks=tuple(self.tracker.tracks),
            occupancy_grid=view,
            timestamp=self.timestamp,
            explored_fraction=self.occupancy.known_fraction,
            objective=self.objective,
        )
