"""Phase 8 computer-vision audit: the agent's chest detector on a labelled dataset.

The dataset is rendered by the Python reference arena (the Unreal research build would render
its own frames; nothing here is an Unreal measurement). For every sample the harness places
the character and camera at a sampled pose, renders one frame per resolution with pixel
labels, and runs exactly the agent's perception front end on it: the blurred-HSV colour
segmentation, then ``ChestDetector3D`` (hybrid method, the agent's templates), then the
agent's ``DistanceEstimator`` for every matched detection (with the *true* camera pose, so
the distance error is the estimator's own).

Ground truth: a chest counts as *visible* when at least ``MIN_PIXEL_SHARE`` of the frame
shows it (about 35 px at 640x360, 9 px at 320x180). Smaller chest views are *ignored*: they
are neither required (recall) nor counted as false positives when detected. A detection
matches a label when its box overlaps the label's visible-pixel box (IoU >= 0.25) or its
centre lies inside it. A ``chest`` detection on an opened chest is a class error and counts
as a false positive; a ``chest_opened`` detection on an unopened chest likewise.

Scenario tags: ``chest_visible`` / ``chest_absent``, distance bin of the nearest visible
chest, ``centred`` / ``off_centre`` aim, ``partially_occluded`` (visible pixels below half
of the chest's projected silhouette, or cut by the frame edge), ``indoor`` / ``outdoor``
camera, lighting (``lighting_low`` / ``lighting_normal`` / ``lighting_high``),
``decoy_in_view`` (a gold-coloured decoy prop in view), ``opened``, ``multiple``.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MIN_PIXEL_SHARE = 1.5e-4
SIZES: tuple[tuple[int, int], ...] = ((320, 180), (480, 270), (640, 360))
DEFAULT_MAPS: tuple[str, ...] = ("research_arena_3d", "p7_open_test_001",
                                 "p7_indoor_test_001", "p7_mixed_test_001",
                                 "p7_complex_test_001", "p7_open_test_001_light",
                                 "p7_mixed_test_002_light")


@dataclass(frozen=True, slots=True)
class Pose:
    map_id: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    aim: str                       # centred | off_centre | random
    opened: tuple[int, ...] = ()   # spawn indices whose chests are set opened


@dataclass(slots=True)
class Tally:
    frames: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    class_errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    cpu_ms: list[float] = field(default_factory=list)
    distance_errors: list[float] = field(default_factory=list)
    bearing_errors: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
        lat = sorted(self.latencies_ms)
        out = {"frames": self.frames, "true_positives": self.tp, "false_positives": self.fp,
               "missed": self.fn, "class_errors": self.class_errors,
               "precision": _r(p), "recall": _r(r), "f1": _r(f1),
               "false_positives_per_frame": _r(self.fp / self.frames) if self.frames else None,
               "latency_mean_ms": _r(statistics.fmean(lat)) if lat else None,
               "latency_p95_ms": _r(lat[int(0.95 * (len(lat) - 1))]) if lat else None,
               "cpu_mean_ms": _r(statistics.fmean(self.cpu_ms)) if self.cpu_ms else None}
        if self.distance_errors:
            d = sorted(self.distance_errors)
            out["distance_abs_error_mean_m"] = _r(statistics.fmean(d))
            out["distance_abs_error_median_m"] = _r(statistics.median(d))
            out["distance_abs_error_p95_m"] = _r(d[int(0.95 * (len(d) - 1))])
            out["bearing_abs_error_mean_deg"] = _r(statistics.fmean(self.bearing_errors))
        return out


def _r(x: float | None, digits: int = 4) -> float | None:
    return None if x is None else round(float(x), digits)


def sample_poses(map_id: str, rng: random.Random, *, per_chest: int = 6,
                 negatives: int = 12, max_chests: int = 8) -> list[Pose]:
    """Camera poses around the map's present chests (several distances and aims), plus
    random poses (mostly without chests in view) and a few with chests set opened."""
    from chest_hunter.simulation3d.environment3d import cached_arena

    arena, _, nav = cached_arena(map_id)
    poses: list[Pose] = []
    spawns = list(enumerate(arena.spawns))
    rng.shuffle(spawns)
    for idx, sp in spawns[:max_chests]:
        for k in range(per_chest):
            d = rng.choice((2.0, 3.0, 4.5, 7.0, 10.0, 14.0))
            a = rng.uniform(0.0, 2 * math.pi)
            px, py = sp.x + d * math.cos(a), sp.y + d * math.sin(a)
            node = nav.project((px, py, sp.z), horizontal=1.0)
            if node is None:
                continue
            x, y, z = nav.node_position(node)
            bearing = math.degrees(math.atan2(sp.y - y, sp.x - x)) % 360.0
            aim = "centred" if k % 3 == 0 else "off_centre"
            yaw = bearing + (0.0 if aim == "centred" else rng.choice((-25.0, 25.0)))
            pitch = -math.degrees(math.atan2(max(0.0, z + 1.6 - sp.z), max(0.5, d))) * 0.6 - 4
            opened = (idx,) if k == per_chest - 1 else ()
            poses.append(Pose(map_id, x, y, z, yaw % 360.0, max(-35.0, pitch), aim, opened))
    nodes = list(nav.reachable_from(arena.start))
    for _ in range(negatives):
        x, y, z = nav.node_position(rng.choice(nodes))
        poses.append(Pose(map_id, x, y, z, rng.uniform(0, 360), -10.0, "random"))
    return poses


def _projected_area(c: Any, pose: Any, intr: Any) -> float:
    from chest_hunter.simulation3d.arena import CHEST_DEPTH, CHEST_HEIGHT, CHEST_WIDTH
    from chest_hunter.vision.perspective import project

    us, vs = [], []
    r = max(CHEST_WIDTH, CHEST_DEPTH) / 2
    for dx in (-r, r):
        for dy in (-r, r):
            for dz in (0.0, CHEST_HEIGHT):
                u, v, d = project(np.array([c.x + dx, c.y + dy, c.z + dz]), pose, intr)
                if float(d) <= 0.05:
                    return 0.0
                us.append(float(u))
                vs.append(float(v))
    u0, u1 = max(0.0, min(us)), min(float(intr.width), max(us))
    v0, v1 = max(0.0, min(vs)), min(float(intr.height), max(vs))
    return max(0.0, u1 - u0) * max(0.0, v1 - v0) * 0.75   # a box fills ~3/4 of its bbox


def _decoy_in_view(env: Any, pose: Any) -> bool:
    from chest_hunter.vision.perspective import project

    for b in env.world.boxes:
        if b.kind != "decoy" or "gold" not in b.colour:
            continue
        c = np.array([(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, (b.z0 + b.z1) / 2])
        u, v, d = project(c, pose, env.intrinsics)
        if 0.3 < float(d) < 15.0 and 0 <= float(u) < env.intrinsics.width \
                and 0 <= float(v) < env.intrinsics.height \
                and env.world.line_of_sight(tuple(pose.position), tuple(c)):
            return True
    return False


def _indoor(arena: Any, x: float, y: float) -> bool:
    for name, (x0, y0, x1, y1) in getattr(arena, "regions", {}).items():
        if any(w in name for w in ("Room", "Building", "Complex", "Shelter", "Sealed")) \
                and x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match(det_bbox: Sequence[float], label_bbox: Sequence[float]) -> bool:
    x, y, w, h = det_bbox
    lx, ly, lw, lh = label_bbox
    cx, cy = x + w / 2, y + h / 2
    return _iou(det_bbox, label_bbox) >= 0.25 or (lx <= cx <= lx + lw and ly <= cy <= ly + lh)


def evaluate(settings: Any, *, maps: Sequence[str] = DEFAULT_MAPS,
             sizes: Sequence[tuple[int, int]] = SIZES, seed: int = 8,
             per_chest: int = 6, negatives: int = 12, failures_dir: Path | None = None,
             max_failure_examples: int = 4) -> dict[str, Any]:
    import cv2

    from chest_hunter.agent.controller_3d import _hsv
    from chest_hunter.agent.runner_3d import _templates
    from chest_hunter.simulation3d.arena import chest_boxes
    from chest_hunter.simulation3d.environment3d import Environment3D, EnvSettings
    from chest_hunter.vision.chest_detector_3d import ChestDetector3D, segment_classes
    from chest_hunter.vision.depth_estimation import DistanceEstimator
    from chest_hunter.vision.perspective import CameraPose

    rng = random.Random(seed)
    poses = [p for m in maps for p in sample_poses(m, rng, per_chest=per_chest,
                                                   negatives=negatives)]
    detector = ChestDetector3D(_templates(settings), method="hybrid")
    estimator = DistanceEstimator()
    per_size: dict[str, Tally] = {}
    per_tag: dict[str, dict[str, Tally]] = {}
    failures: dict[str, int] = {}
    if failures_dir is not None:
        failures_dir.mkdir(parents=True, exist_ok=True)
    for w, h in sizes:
        size = f"{w}x{h}"
        tally = per_size.setdefault(size, Tally())
        envs: dict[str, Any] = {}
        for pose in poses:
            env = envs.get(pose.map_id)
            if env is None:
                env = envs[pose.map_id] = Environment3D(
                    pose.map_id, seed=seed, episode=0,
                    settings=EnvSettings(width=w, height=h))
            for c in env.chests:                     # restore, then apply this sample
                opened = c.spawn_index in pose.opened and c.present
                if c.opened != opened and c.present:
                    c.opened = opened
                    c.boxes = tuple(chest_boxes(c.x, c.y, c.z, c.yaw_deg, opened=opened))
            ch = env.character
            ch.x, ch.y, ch.z = pose.x, pose.y, pose.z
            ch.body_yaw = ch.cam_yaw = pose.yaw
            ch.cam_pitch = pose.pitch
            frame, header = env.observe(labels=True)
            labels = env.last_labels or {"chests": []}
            cam = labels["camera_pose"]
            cpose = CameraPose(cam[0], cam[1], cam[2], cam[3], cam[4])
            area = w * h
            visible, ignored = [], []
            for lab in labels["chests"]:
                (visible if lab["pixels"] >= MIN_PIXEL_SHARE * area else ignored).append(lab)
            started, cpu0 = time.perf_counter(), time.thread_time()
            classes = segment_classes(_hsv(frame))
            dets = detector.detect(frame, header["sim_time"],
                                   camera_pitch_deg=header["camera"]["pitch_deg"],
                                   fy=env.intrinsics.fy, classes=classes)
            latency = 1000 * (time.perf_counter() - started)
            cpu = 1000 * (time.thread_time() - cpu0)
            tags = _tags(env, pose, cpose, visible)
            groups = [tally] + [per_tag.setdefault(size, {}).setdefault(t, Tally())
                                for t in tags]
            matched_labels: set[int] = set()
            fp = tp = cls_err = 0
            for d in dets:
                hit = next((lab for lab in visible + ignored
                            if _match(d.bbox, lab["bbox"])), None)
                if hit is None:
                    fp += 1
                    continue
                if (d.category == "chest") != (hit["category"] == "chest"):
                    fp += 1
                    cls_err += 1
                    continue
                if hit in visible and id(hit) not in matched_labels:
                    matched_labels.add(id(hit))
                    tp += 1
                    est = estimator.estimate(d, cpose, env.intrinsics,
                                             floors=sorted({0.0, pose.z,
                                                            hit["position"][2]}))
                    if est is not None:
                        derr = abs(est.distance - hit["distance"])
                        true_bearing = math.degrees(math.atan2(hit["position"][1] - cam[1],
                                                               hit["position"][0] - cam[0]))
                        berr = abs((est.bearing_deg - true_bearing + 180) % 360 - 180)
                        for g in groups:
                            g.distance_errors.append(derr)
                            g.bearing_errors.append(berr)
            fn = sum(1 for lab in visible if id(lab) not in matched_labels)
            for g in groups:
                g.frames += 1
                g.tp += tp
                g.fp += fp
                g.fn += fn
                g.class_errors += cls_err
                g.latencies_ms.append(latency)
                g.cpu_ms.append(cpu)
            if failures_dir is not None and (fp or fn) and size == "640x360":
                kind = ("false_positive" if fp else "missed") + "_" + (
                    next((t for t in tags if t in ("partially_occluded", "decoy_in_view",
                                                   "opened", "lighting_low",
                                                   "lighting_high")), tags[1]))
                n = failures.get(kind, 0)
                if n < max_failure_examples:
                    failures[kind] = n + 1
                    img = frame.copy()
                    for lab in visible:
                        x, y, bw, bh = lab["bbox"]
                        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
                    for d in dets:
                        x, y, bw, bh = d.bbox
                        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
                    cv2.imwrite(str(failures_dir / f"{kind}_{n}.png"), img)
                else:
                    failures[kind] = n + 1
    return {
        "dataset": {"maps": list(maps), "poses": len(poses), "seed": seed,
                    "renderer": "Python reference arena (NOT Unreal)",
                    "min_pixel_share": MIN_PIXEL_SHARE},
        "per_resolution": {s: t.summary() for s, t in per_size.items()},
        "per_scenario": {s: {tag: t.summary() for tag, t in sorted(tags.items())}
                         for s, tags in per_tag.items()},
        "failure_frames_at_640x360": failures,
    }


def _tags(env: Any, pose: Pose, cpose: Any, visible: list[dict[str, Any]]) -> list[str]:
    tags = ["chest_visible" if visible else "chest_absent", pose.aim]
    if visible:
        d = min(lab["distance"] for lab in visible)
        tags.append("dist_0_3m" if d <= 3 else "dist_3_8m" if d <= 8 else
                    "dist_8_15m" if d <= 15 else "dist_15m_plus")
        for lab in visible:
            c = env.chests[lab["spawn_index"]]
            area = _projected_area(c, cpose, env.intrinsics)
            x, y, bw, bh = lab["bbox"]
            edge = x <= 0 or y <= 0 or x + bw >= env.intrinsics.width \
                or y + bh >= env.intrinsics.height
            if edge or (area > 0 and lab["pixels"] < 0.5 * area):
                tags.append("partially_occluded")
                break
        if any(lab["category"] == "chest_opened" for lab in visible):
            tags.append("opened")
        if len(visible) >= 2:
            tags.append("multiple")
    tags.append("indoor" if _indoor(env.arena, pose.x, pose.y) else "outdoor")
    light = env.base_lighting
    tags.append("lighting_low" if light < 0.8 else "lighting_high" if light > 1.2
                else "lighting_normal")
    if _decoy_in_view(env, cpose):
        tags.append("decoy_in_view")
    return tags
