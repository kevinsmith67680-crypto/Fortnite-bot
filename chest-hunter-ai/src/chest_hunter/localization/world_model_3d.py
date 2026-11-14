"""The 3D agent's estimated world: spawn priors, chest tracks and its own pose.

Everything here is built from observations and the database priors (spawn
point, probability); nothing comes from simulator ground truth.

* **Pose** is the odometry estimate (it drifts slowly) plus the camera rig.
* **Chest tracks** come from :class:`~chest_hunter.vision.tracker_3d.Tracker3D`.
* **Spawn priors** carry a status: ``unknown``, ``empty`` (looked at twice and
  no chest), ``chest_seen``, ``opened``, ``failed``, ``unreachable`` or
  ``abandoned``. A prior is a hint, never proof: a chest must be seen before
  it is targeted, and an empty spot needs two clear looks.

A look only counts as "clear" when the agent is on the same floor, closer
than 9 m, the spot is well inside the frame and not behind the agent's own
body, the navigation service's local ray reaches the spot (no wall in
between), and no chest colours appear where the chest would be.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from chest_hunter.unreal_bridge.observation_adapter import Observation3D
from chest_hunter.vision.chest_detector_3d import BROWN_ID, GOLD_ID, Detection3D
from chest_hunter.vision.depth_estimation import CHEST_HEIGHT, DistanceEstimator, RangeEstimate
from chest_hunter.vision.perspective import CameraPose, project
from chest_hunter.vision.tracker_3d import Track3D, Tracker3D, TrackState3D

ASSOCIATION_RADIUS = 1.6        # metres, horizontal, track <-> spawn prior
ASSOCIATION_HEIGHT = 1.0


class PriorStatus3D(str, Enum):
    UNKNOWN = "unknown"
    EMPTY = "empty"
    CHEST_SEEN = "chest_seen"
    OPENED = "opened"
    FAILED = "failed"
    UNREACHABLE = "unreachable"
    ABANDONED = "abandoned"


RESOLVED = frozenset({PriorStatus3D.EMPTY, PriorStatus3D.OPENED, PriorStatus3D.FAILED,
                      PriorStatus3D.UNREACHABLE, PriorStatus3D.ABANDONED})


@dataclass(slots=True)
class SpawnPrior3D:
    location_id: int
    name: str
    x: float
    y: float
    z: float
    probability: float
    status: PriorStatus3D = PriorStatus3D.UNKNOWN
    track_id: int | None = None
    empty_votes: int = 0
    last_vote: float = -1.0
    inspections: int = 0

    @property
    def resolved(self) -> bool:
        return self.status in RESOLVED

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(slots=True)
class PerceptionUpdate:
    detections: list[Detection3D]
    estimates: list[RangeEstimate | None]        # aligned with ``detections``
    new_tracks: list[Track3D]
    newly_empty: list[SpawnPrior3D]
    self_mask: tuple[int, int, int, int] | None


class WorldModel3D:
    def __init__(self, priors: Sequence[SpawnPrior3D], *, estimator: DistanceEstimator | None = None,
                 tracker: Tracker3D | None = None, empty_confirmations: int = 2,
                 vote_range: float = 9.0) -> None:
        self.priors = {p.location_id: p for p in priors}
        self.estimator = estimator or DistanceEstimator()
        self.tracker = tracker or Tracker3D()
        self.empty_confirmations = empty_confirmations
        self.vote_range = vote_range
        self.floors = sorted({round(p.z, 2) for p in priors} | {0.0})
        self.position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.camera_pose: CameraPose | None = None
        self.objective = "-"

    # -- perception integration ----------------------------------------------------------------

    @staticmethod
    def self_mask(obs: Observation3D) -> tuple[int, int, int, int] | None:
        """Screen box covering the agent's own body, from its known camera rig."""
        pose = obs.camera_pose()
        fx, fy, fz = obs.odometry.position
        r = 0.4
        corners = np.array([[fx + dx, fy + dy, fz + dz] for dx in (-r, r) for dy in (-r, r)
                            for dz in (0.0, 1.75)])
        u, v, d = project(corners, pose, obs.intrinsics)
        if not np.all(d > 0.05):
            return None
        x0, x1 = int(np.floor(np.nanmin(u))) - 3, int(np.ceil(np.nanmax(u))) + 3
        y0, y1 = int(np.floor(np.nanmin(v))) - 3, int(np.ceil(np.nanmax(v))) + 3
        w, h = obs.intrinsics.width, obs.intrinsics.height
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def integrate(self, obs: Observation3D, detections: list[Detection3D],
                  classes: np.ndarray | None,
                  clear_line: Callable[[SpawnPrior3D], bool]) -> PerceptionUpdate:
        self.position = obs.odometry.position
        pose = obs.camera_pose()
        self.camera_pose = pose
        mask = self.self_mask(obs)
        anchors = [p.xyz for p in self.priors.values()]
        measurements: list[tuple[Detection3D, RangeEstimate]] = []
        estimates: list[RangeEstimate | None] = []
        for det in detections:
            est = self.estimator.estimate(det, pose, obs.intrinsics, self.floors,
                                          anchors=anchors, own_floor=self.position[2])
            estimates.append(est)
            if est is not None:
                measurements.append((det, est))

        def expected_visible(t: Track3D) -> bool:
            return self._in_clear_view(t.xyz, pose, obs, mask, max_range=10.0)

        new_tracks = self.tracker.update(measurements, obs.sim_time, expected_visible)
        self._associate()
        newly_empty = self._vote_empty(obs, pose, mask, classes, clear_line)
        return PerceptionUpdate(detections, estimates, new_tracks, newly_empty, mask)

    def _in_clear_view(self, point: tuple[float, float, float], pose: CameraPose,
                       obs: Observation3D, mask: tuple[int, int, int, int] | None, *,
                       max_range: float) -> bool:
        x, y, z = point
        if abs(z - self.position[2]) > 0.6:
            return False
        centre = np.array([x, y, z + CHEST_HEIGHT / 2])
        u, v, d = project(centre, pose, obs.intrinsics)
        if not float(d) > 0.5 or float(d) > max_range:
            return False
        w, h = obs.intrinsics.width, obs.intrinsics.height
        if not (0.1 * w <= u <= 0.9 * w and 0.1 * h <= v <= 0.9 * h):
            return False
        if mask is not None:
            mx, my, mw, mh = mask
            if mx - 10 <= u <= mx + mw + 10 and my - 10 <= v <= my + mh + 10:
                return False
        return True

    def _associate(self) -> None:
        for prior in self.priors.values():
            if prior.status is PriorStatus3D.CHEST_SEEN and (
                    prior.track_id is None or self.tracker.get(prior.track_id) is None):
                prior.status = PriorStatus3D.UNKNOWN          # evidence expired: look again
                prior.track_id = None
        for track in self.tracker.tracks:
            if track.location_id is None:
                prior = self.nearest_prior(track.xyz, exclude_tracked=True)
                if prior is not None:
                    track.location_id = prior.location_id
                    prior.track_id = track.track_id
            linked = self.priors.get(track.location_id) if track.location_id else None
            if linked is None or track.state is TrackState3D.TENTATIVE:
                continue
            linked.empty_votes = 0
            if linked.status in (PriorStatus3D.UNKNOWN, PriorStatus3D.EMPTY) \
                    and track.category == "chest":
                linked.status = PriorStatus3D.CHEST_SEEN

    def nearest_prior(self, point: tuple[float, float, float], *,
                      exclude_tracked: bool = False) -> SpawnPrior3D | None:
        best: tuple[float, SpawnPrior3D] | None = None
        for p in self.priors.values():
            if exclude_tracked and p.track_id is not None and self.tracker.get(p.track_id):
                continue
            d = math.hypot(p.x - point[0], p.y - point[1])
            if d <= ASSOCIATION_RADIUS and abs(p.z - point[2]) <= ASSOCIATION_HEIGHT:
                if best is None or d < best[0]:
                    best = (d, p)
        return None if best is None else best[1]

    def _vote_empty(self, obs: Observation3D, pose: CameraPose,
                    mask: tuple[int, int, int, int] | None, classes: np.ndarray | None,
                    clear_line: Callable[[SpawnPrior3D], bool]) -> list[SpawnPrior3D]:
        newly = []
        tracks = [t for t in self.tracker.tracks if t.state is not TrackState3D.TENTATIVE
                  or t.hits >= 1]
        for p in self.priors.values():
            if p.status is not PriorStatus3D.UNKNOWN:
                continue
            if math.hypot(p.x - self.position[0], p.y - self.position[1]) > self.vote_range:
                continue
            if not self._in_clear_view(p.xyz, pose, obs, mask, max_range=self.vote_range + 2.0):
                continue
            if any(math.hypot(t.position[0] - p.x, t.position[1] - p.y) <= ASSOCIATION_RADIUS
                   for t in tracks):
                p.empty_votes = 0
                continue
            if classes is not None and self._chest_colours_near(p, pose, obs, classes):
                continue
            if obs.sim_time - p.last_vote < 0.4:
                continue
            if not clear_line(p):
                continue
            p.empty_votes += 1
            p.last_vote = obs.sim_time
            if p.empty_votes >= self.empty_confirmations:
                p.status = PriorStatus3D.EMPTY
                newly.append(p)
        return newly

    @staticmethod
    def _chest_colours_near(p: SpawnPrior3D, pose: CameraPose, obs: Observation3D,
                            classes: np.ndarray) -> bool:
        """Any gold/brown pixels where a chest at this spot would appear?"""
        pts = np.array([[p.x + dx, p.y + dy, p.z + dz] for dx in (-0.6, 0.6) for dy in (-0.6, 0.6)
                        for dz in (0.0, CHEST_HEIGHT + 0.6)])
        u, v, d = project(pts, pose, obs.intrinsics)
        if not np.all(d > 0.1):
            return True
        h, w = classes.shape
        x0, x1 = max(0, int(np.nanmin(u))), min(w, int(np.nanmax(u)) + 1)
        y0, y1 = max(0, int(np.nanmin(v))), min(h, int(np.nanmax(v)) + 1)
        if x1 <= x0 or y1 <= y0:
            return True
        region = classes[y0:y1, x0:x1]
        chesty = np.count_nonzero((region == GOLD_ID) | (region == BROWN_ID))
        return chesty > max(3, 0.02 * region.size)

    # -- queries --------------------------------------------------------------------------------

    def chest_targets(self) -> list[Track3D]:
        """Confirmed unopened chest tracks at unresolved spawn points."""
        out = []
        for t in self.tracker.confirmed():
            if t.category != "chest" or t.location_id is None:
                continue
            prior = self.priors.get(t.location_id)
            if prior is not None and not prior.resolved:
                out.append(t)
        return out

    def unresolved(self) -> list[SpawnPrior3D]:
        return [p for p in self.priors.values() if not p.resolved]

    @property
    def all_resolved(self) -> bool:
        return all(p.resolved for p in self.priors.values())

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.priors.values():
            out[p.status.value] = out.get(p.status.value, 0) + 1
        return out
