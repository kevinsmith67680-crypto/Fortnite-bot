#!/usr/bin/env python3
"""Generate the multi-view chest reference templates for the 3D detector.

Dataset generation is allowed to use simulator ground truth: each chest
model (closed and opened) is rendered alone, from several yaw angles and
camera elevations, and cropped with the renderer's label buffer. For every
view we store

* ``<name>.png``: the BGR crop (for the pure template-matching detector),
* ``<name>_labels.png``: per-pixel classes 0 background, 1 gold body,
  2 brown lid, 3 cream inside (for the hybrid detector),
* an entry in ``templates.json`` with the view angles and the chest's
  apparent width/height in metres for that view (used to estimate distance
  from apparent size).

Usage::

    python scripts/generate_templates_3d.py [--output assets/templates3d]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import chest_hunter  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from chest_hunter.simulation3d.arena import chest_boxes  # noqa: E402
from chest_hunter.simulation3d.geometry import World3D, box  # noqa: E402
from chest_hunter.simulation3d.renderer3d import Renderer3D, box_faces  # noqa: E402
from chest_hunter.vision.chest_detector_3d import segment_classes  # noqa: E402
from chest_hunter.vision.perspective import CameraIntrinsics, CameraPose  # noqa: E402

YAWS = (0, 30, 60, 90, 135, 180)              # camera bearing relative to the chest front
ELEVATIONS = (-8, 8, 20, 35, 55)              # degrees the camera looks down at the chest
DISTANCE = 4.0


def render_view(state: str, yaw: float, elevation: float) -> tuple[np.ndarray, np.ndarray, dict]:
    world = World3D([box(-50, -50, -0.2, 50, 50, 0.0, "ground", "grass")],
                    bounds=(-50.0, -50.0, 50.0, 50.0))
    intr = CameraIntrinsics(640, 360, 90.0)
    renderer = Renderer3D(world, intr, noise_std=0.0, seed=0)
    chest = chest_boxes(0.0, 0.0, 0.0, 0.0, opened=(state == "opened"))
    faces = [f for b in chest for f in box_faces(b, owner=1)]
    centre_z = 0.33
    # Camera placed in front of the chest (chest front faces +X) at the given bearing.
    bearing = math.radians(yaw)
    horizontal = DISTANCE * math.cos(math.radians(elevation))
    cx, cy = horizontal * math.cos(bearing), horizontal * math.sin(bearing)
    cz = centre_z + DISTANCE * math.sin(math.radians(elevation))
    look_yaw = math.degrees(math.atan2(-cy, -cx))
    pose = CameraPose(cx, cy, cz, look_yaw, -elevation)
    frame, ids = renderer.render(pose, faces, [], labels=True, noise=False)
    assert ids is not None
    ys, xs = np.nonzero(ids == 1)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = frame[y0:y1, x0:x1].copy()
    labels = segment_classes(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    labels[ids[y0:y1, x0:x1] != 1] = 0
    meta = {"width_m": (x1 - x0) * DISTANCE / intr.fx, "height_m": (y1 - y0) * DISTANCE / intr.fy,
            "aspect": float((x1 - x0) / (y1 - y0))}
    return crop, labels, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "templates3d")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for state in ("closed", "opened"):
        for yaw in YAWS:
            for elevation in ELEVATIONS:
                crop, labels, meta = render_view(state, yaw, elevation)
                name = f"{state}_y{yaw:03d}_e{elevation:+03d}"
                cv2.imwrite(str(args.output / f"{name}.png"), crop)
                cv2.imwrite(str(args.output / f"{name}_labels.png"), labels)
                index.append({"name": name, "category": "chest" if state == "closed"
                              else "chest_opened", "view_yaw_deg": yaw,
                              "elevation_deg": elevation, **{k: round(v, 4) for k, v in meta.items()}})
    (args.output / "templates.json").write_text(json.dumps(
        {"format": "chest-templates-3d/1", "render_distance_m": DISTANCE, "templates": index},
        indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(index)} templates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
