"""OBSERVE -> DETECT: one call turns a frame into structured perception output.

The pipeline runs, in order: preprocessing, agent-marker localisation, chest
detection (+ NMS), conversion of detections to world coordinates,
occupancy-grid update and interaction-cue detection. It consumes only the
observation (pixels, timestamp, camera geometry) and the template library.
It has no access to simulator state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from chest_hunter.localization.pose_estimator import AgentMarkerDetector, PoseMeasurement
from chest_hunter.vision.chest_detector import Detection, make_detector
from chest_hunter.vision.interaction_detector import InteractionCues, InteractionDetector
from chest_hunter.vision.observation import VisualObservation
from chest_hunter.vision.obstacle_detector import ObstacleDetector, OccupancyGrid, OccupancyUpdate
from chest_hunter.vision.preprocessing import PreprocessConfig, Preprocessor
from chest_hunter.vision.templates import TemplateLibrary
from chest_hunter.vision.visual_tracker import WorldDetection


@dataclass
class PerceptionResult:
    observation: VisualObservation
    detections: list[Detection]
    world_detections: list[WorldDetection]
    pose: PoseMeasurement | None
    occupancy: OccupancyUpdate | None
    cues: InteractionCues
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())


class PerceptionPipeline:
    def __init__(self, library: TemplateLibrary, *, detector: str = "hybrid",
                 confidence_threshold: float = 0.85, update_occupancy: bool = True,
                 preprocess: PreprocessConfig | None = None) -> None:
        self.library = library
        self.preprocessor = Preprocessor(library, preprocess)
        self.chest_detector = make_detector(detector, library,
                                            confidence_threshold=confidence_threshold)
        self.marker_detector = AgentMarkerDetector()
        self.obstacle_detector = ObstacleDetector()
        self.interaction_detector = InteractionDetector()
        self.update_occupancy = update_occupancy
        self.frames = 0
        self.total_seconds = 0.0
        self.peak_seconds = 0.0

    def process(self, observation: VisualObservation, occupancy: OccupancyGrid,
                pose_hint_pixel: tuple[float, float] | None = None) -> PerceptionResult:
        timings: dict[str, float] = {}
        t = time.perf_counter()
        frame = self.preprocessor.prepare(observation)
        timings["preprocess"] = time.perf_counter() - t

        t = time.perf_counter()
        pose = self.marker_detector.detect(frame, pose_hint_pixel)
        timings["localisation"] = time.perf_counter() - t

        t = time.perf_counter()
        detections = self.chest_detector.detect(frame)
        camera = observation.camera
        world = []
        for det in detections:
            u, v = det.center
            world.append(WorldDetection(det.category, camera.pixel_to_world(u, v),
                                        det.confidence, det.bbox))
        timings["chest_detection"] = time.perf_counter() - t

        update = None
        if self.update_occupancy:
            t = time.perf_counter()
            update = self.obstacle_detector.observe(frame, occupancy)
            timings["occupancy"] = time.perf_counter() - t

        t = time.perf_counter()
        cues = self.interaction_detector.detect(frame, pose.pixel if pose else None)
        timings["interaction"] = time.perf_counter() - t

        total = sum(timings.values())
        self.frames += 1
        self.total_seconds += total
        self.peak_seconds = max(self.peak_seconds, total)
        return PerceptionResult(observation, detections, world, pose, update, cues, timings)

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.frames if self.frames else 0.0
