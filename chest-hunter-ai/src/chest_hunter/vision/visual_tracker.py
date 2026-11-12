"""Lightweight multi-object tracking of chests in world coordinates.

Detections are first converted to world positions (using the camera
geometry), then associated with existing tracks by greedy nearest-neighbour
matching inside a gate. A chest that changes from unopened to opened keeps
its track. Lifecycle:

``TENTATIVE`` (fewer than ``confirm_hits`` sightings) -> ``CONFIRMED`` ->
``LOST`` (in view but not detected) -> ``EXPIRED`` (removed).

Misses are only counted while the track's position is inside the camera
view, so chests out of view are remembered (up to ``memory_seconds``)
instead of being forgotten. The number of tracks is capped at ``max_tracks``;
when full, the stalest unconfirmed (then oldest) track is evicted. Memory is
therefore bounded.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class TrackState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class WorldDetection:
    category: str
    position: tuple[float, float]      # world (grid-cell) coordinates
    confidence: float
    bbox: tuple[int, int, int, int]    # pixels in the frame it came from


@dataclass
class ObjectTrack:
    track_id: int
    category: str
    estimated_position: tuple[float, float]
    confidence: float
    last_seen: float
    observations: int
    state: TrackState = TrackState.TENTATIVE
    first_seen: float = 0.0
    misses: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    location_id: int | None = None
    category_history: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state is not TrackState.EXPIRED


class VisualTracker:
    def __init__(self, *, gate_distance: float = 0.9, confirm_hits: int = 2,
                 expiration_seconds: float = 3.0, memory_seconds: float = 600.0,
                 max_tracks: int = 100, position_alpha: float = 0.5) -> None:
        self.gate_distance = gate_distance
        self.confirm_hits = confirm_hits
        self.expiration_seconds = expiration_seconds
        self.memory_seconds = memory_seconds
        self.max_tracks = max_tracks
        self.position_alpha = position_alpha
        self._tracks: dict[int, ObjectTrack] = {}
        self._next_id = 1
        self.expired_total = 0
        self.evicted_total = 0

    @property
    def tracks(self) -> list[ObjectTrack]:
        return list(self._tracks.values())

    def get(self, track_id: int) -> ObjectTrack | None:
        return self._tracks.get(track_id)

    def confirmed(self) -> list[ObjectTrack]:
        return [t for t in self._tracks.values()
                if t.state in (TrackState.CONFIRMED, TrackState.LOST)]

    def update(self, detections: Sequence[WorldDetection], timestamp: float,
               in_view: Callable[[float, float], bool]) -> list[ObjectTrack]:
        """Integrate one frame's detections; return tracks created this frame."""
        pairs = []
        for di, det in enumerate(detections):
            for track in self._tracks.values():
                d = math.dist(det.position, track.estimated_position)
                if d <= self.gate_distance:
                    pairs.append((d, di, track.track_id))
        pairs.sort()
        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        for _, di, tid in pairs:
            if di in matched_dets or tid in matched_tracks:
                continue
            matched_dets.add(di)
            matched_tracks.add(tid)
            self._apply(self._tracks[tid], detections[di], timestamp)

        created = []
        for di, det in enumerate(detections):
            if di not in matched_dets:
                new_track = self._create(det, timestamp)
                if new_track is not None:
                    created.append(new_track)
                    matched_tracks.add(new_track.track_id)

        for track in list(self._tracks.values()):
            if track.track_id in matched_tracks:
                continue
            visible = in_view(*track.estimated_position)
            unseen_for = timestamp - track.last_seen
            if visible:
                track.misses += 1
                if track.state is TrackState.CONFIRMED:
                    track.state = TrackState.LOST
                limit = self.expiration_seconds if track.state is not TrackState.TENTATIVE \
                    else min(self.expiration_seconds, 1.0)
                if unseen_for > limit:
                    self._expire(track)
            elif unseen_for > self.memory_seconds:
                self._expire(track)
        return created

    def _apply(self, track: ObjectTrack, det: WorldDetection, timestamp: float) -> None:
        a = self.position_alpha
        tx, ty = track.estimated_position
        track.estimated_position = (tx + a * (det.position[0] - tx),
                                    ty + a * (det.position[1] - ty))
        track.confidence = max(det.confidence, 0.7 * track.confidence + 0.3 * det.confidence)
        if det.category != track.category:
            track.category_history.append(track.category)
            track.category = det.category
        track.last_seen = timestamp
        track.observations += 1
        track.misses = 0
        track.bbox = det.bbox
        if track.state is TrackState.LOST or (
                track.state is TrackState.TENTATIVE
                and track.observations >= self.confirm_hits):
            track.state = TrackState.CONFIRMED

    def _create(self, det: WorldDetection, timestamp: float) -> ObjectTrack | None:
        if len(self._tracks) >= self.max_tracks and not self._evict():
            return None
        track = ObjectTrack(self._next_id, det.category, det.position, det.confidence,
                            timestamp, 1, TrackState.TENTATIVE, timestamp, 0, det.bbox)
        if self.confirm_hits <= 1:
            track.state = TrackState.CONFIRMED
        self._next_id += 1
        self._tracks[track.track_id] = track
        return track

    def _evict(self) -> bool:
        if not self._tracks:
            return False
        victim = min(self._tracks.values(),
                     key=lambda t: (t.state is not TrackState.TENTATIVE, t.last_seen))
        del self._tracks[victim.track_id]
        self.evicted_total += 1
        return True

    def _expire(self, track: ObjectTrack) -> None:
        track.state = TrackState.EXPIRED
        del self._tracks[track.track_id]
        self.expired_total += 1

    def forget(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
