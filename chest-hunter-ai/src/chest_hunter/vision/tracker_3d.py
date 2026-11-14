"""Tracking chests in 3D world coordinates across frames (Phase 4).

Each track keeps a position estimate with an isotropic uncertainty. New
measurements (detection + :class:`RangeEstimate`) are associated by a gate
that widens with the combined uncertainty, then fused by inverse variance.
Camera rays from different viewpoints are kept so the position can also be
triangulated (method C) once the baseline is wide enough.

States: ``tentative`` (not yet confirmed), ``confirmed``, ``lost`` (not seen
for a while: kept, still worth revisiting) and ``expired`` (removed). Misses
are only counted when the caller says the track *should* be visible (in the
frame, close enough, not hidden by the agent itself); walls can still hide a
track, so confirmed tracks only expire after several such misses at close
range. Memory is bounded.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from chest_hunter.vision.chest_detector_3d import Detection3D
from chest_hunter.vision.depth_estimation import RangeEstimate
from chest_hunter.vision.perspective import triangulate_rays


class TrackState3D(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    EXPIRED = "expired"


@dataclass(slots=True)
class Track3D:
    track_id: int
    category: str
    position: np.ndarray
    sigma: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    state: TrackState3D = TrackState3D.TENTATIVE
    confidence: float = 0.0
    opened_score: float = 0.0
    rays: deque[tuple[np.ndarray, np.ndarray]] = field(default_factory=lambda: deque(maxlen=8))
    triangulated: bool = False
    location_id: int | None = None
    last_distance: float = math.inf

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (float(self.position[0]), float(self.position[1]), float(self.position[2]))

    @property
    def active(self) -> bool:
        return self.state is not TrackState3D.EXPIRED


class Tracker3D:
    def __init__(self, *, max_tracks: int = 100, gate: float = 1.2, confirm_hits: int = 2,
                 tentative_timeout: float = 1.5, lost_after: float = 3.0,
                 memory_seconds: float = 900.0, expire_misses: int = 6) -> None:
        self.max_tracks = max_tracks
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.tentative_timeout = tentative_timeout
        self.lost_after = lost_after
        self.memory_seconds = memory_seconds
        self.expire_misses = expire_misses
        self._tracks: dict[int, Track3D] = {}
        self._next_id = 1
        self.created = 0
        self.expired = 0
        self.triangulations = 0

    @property
    def tracks(self) -> list[Track3D]:
        return [t for t in self._tracks.values() if t.active]

    def get(self, track_id: int) -> Track3D | None:
        t = self._tracks.get(track_id)
        return t if t is not None and t.active else None

    def confirmed(self) -> list[Track3D]:
        return [t for t in self.tracks if t.state in (TrackState3D.CONFIRMED, TrackState3D.LOST)]

    def update(self, measurements: Sequence[tuple[Detection3D, RangeEstimate]], timestamp: float,
               expected_visible: Callable[[Track3D], bool]) -> list[Track3D]:
        """Associate, fuse and age tracks. Returns the tracks created in this update."""
        pairs: list[tuple[float, int, int]] = []
        for mi, (_, est) in enumerate(measurements):
            m = np.array(est.position)
            for t in self.tracks:
                d = float(np.linalg.norm((t.position - m)[:2]))
                dz = abs(float(t.position[2] - m[2]))
                combined = math.hypot(t.sigma, est.sigma)
                if dz <= 1.2 and d <= max(self.gate, 2.5 * combined):
                    pairs.append((d / max(0.05, combined), mi, t.track_id))
        pairs.sort()
        used_m: set[int] = set()
        used_t: set[int] = set()
        for _, mi, tid in pairs:
            if mi in used_m or tid in used_t:
                continue
            used_m.add(mi)
            used_t.add(tid)
            self._apply(self._tracks[tid], *measurements[mi], timestamp)
        created = []
        for mi, (det, est) in enumerate(measurements):
            if mi not in used_m:
                new = self._create(det, est, timestamp)
                if new is not None:
                    created.append(new)
                    used_t.add(new.track_id)
        for t in self.tracks:
            if t.track_id in used_t:
                continue
            self._age(t, timestamp, expected_visible(t))
        return created

    def _create(self, det: Detection3D, est: RangeEstimate, timestamp: float) -> Track3D | None:
        if len(self.tracks) >= self.max_tracks and not self._evict():
            return None
        t = Track3D(self._next_id, det.category, np.array(est.position, dtype=np.float64),
                    est.sigma, timestamp, timestamp, confidence=det.confidence,
                    opened_score=1.0 if det.category == "chest_opened" else -1.0,
                    last_distance=est.distance)
        t.rays.append((np.array(est.ray_origin), np.array(est.ray_direction)))
        self._tracks[t.track_id] = t
        self._next_id += 1
        self.created += 1
        return t

    def _apply(self, t: Track3D, det: Detection3D, est: RangeEstimate, timestamp: float) -> None:
        prior_var = t.sigma ** 2 + 0.01                    # small process noise
        meas_var = est.sigma ** 2
        m = np.array(est.position)
        if est.distance < 5.0 and float(np.linalg.norm((t.position - m)[:2])) > 3.0 * est.sigma:
            # A close, accurate measurement contradicts a track built from distant views
            # (e.g. a box merged with a neighbouring object): trust the close look.
            t.position = m
            t.sigma = max(0.05, est.sigma)
            t.rays.clear()
        else:
            t.position = (t.position * meas_var + m * prior_var) / (prior_var + meas_var)
            t.sigma = max(0.05, math.sqrt(prior_var * meas_var / (prior_var + meas_var)))
        t.rays.append((np.array(est.ray_origin), np.array(est.ray_direction)))
        self._triangulate(t)
        t.hits += 1
        t.misses = 0
        t.last_seen = timestamp
        t.last_distance = est.distance
        t.confidence = 0.7 * t.confidence + 0.3 * det.confidence
        # Opened/closed: exponentially weighted vote, closer views count more.
        weight = det.confidence * (2.0 if est.distance < 6.0 else 1.0)
        t.opened_score = 0.6 * t.opened_score + (weight if det.category == "chest_opened"
                                                 else -weight)
        t.category = "chest_opened" if t.opened_score > 0 else "chest"
        if t.hits >= self.confirm_hits:
            t.state = TrackState3D.CONFIRMED

    def _triangulate(self, t: Track3D) -> None:
        if len(t.rays) < 3:
            return
        origins = np.array([r[0] for r in t.rays])
        dirs = np.array([r[1] for r in t.rays])
        spread = float(np.linalg.norm(origins.max(axis=0) - origins.min(axis=0)))
        if spread < 1.0:
            return
        point, conditioning = triangulate_rays(origins, dirs)
        if conditioning < 0.002:
            return
        centre = t.position + np.array([0.0, 0.0, 0.325])
        if float(np.linalg.norm(point - centre)) > max(1.5, 3 * t.sigma):
            return
        tri_sigma = max(0.1, 0.05 / math.sqrt(conditioning))
        base = point - np.array([0.0, 0.0, 0.325])
        var_t, var_c = t.sigma ** 2, tri_sigma ** 2
        t.position = (t.position * var_c + base * var_t) / (var_t + var_c)
        t.sigma = max(0.05, math.sqrt(var_t * var_c / (var_t + var_c)))
        t.triangulated = True
        self.triangulations += 1

    def _age(self, t: Track3D, timestamp: float, should_see: bool) -> None:
        age = timestamp - t.last_seen
        if t.state is TrackState3D.TENTATIVE:
            if age > self.tentative_timeout or should_see:
                t.misses += 1
                if age > self.tentative_timeout or t.misses >= 2:
                    self._expire(t)
            return
        if should_see:
            t.misses += 1
            if t.misses >= self.expire_misses:
                self._expire(t)
                return
        if age > self.lost_after and t.state is TrackState3D.CONFIRMED:
            t.state = TrackState3D.LOST
        if age > self.memory_seconds:
            self._expire(t)

    def _expire(self, t: Track3D) -> None:
        t.state = TrackState3D.EXPIRED
        del self._tracks[t.track_id]
        self.expired += 1

    def _evict(self) -> bool:
        candidates = sorted(self.tracks, key=lambda t: (t.state is not TrackState3D.TENTATIVE,
                                                        t.last_seen))
        if not candidates:
            return False
        self._expire(candidates[0])
        return True

    def reactivate(self, t: Track3D) -> None:
        if t.state is TrackState3D.LOST:
            t.state = TrackState3D.CONFIRMED
