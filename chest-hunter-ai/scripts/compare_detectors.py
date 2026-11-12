#!/usr/bin/env python3
"""Compare chest detectors A (colour), B (template) and C (hybrid) at every resolution.

``benchmark-vision`` compares the three methods at the configured frame size
only. This script repeats that comparison at 320x180, 480x270, 640x360 and
800x450 on the acceptance arena, with the same labelled frames for every method::

    python scripts/compare_detectors.py [--frames 60] [--seed 42] [--output PATH]

Ground truth is used only to label the frames and score the detections.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
try:
    import chest_hunter  # noqa: F401
except ModuleNotFoundError:  # running from a checkout without installing
    sys.path.insert(0, str(ROOT / "src"))

from chest_hunter.agent.runner import load_world  # noqa: E402
from chest_hunter.analytics.visual_benchmark import compare_detectors  # noqa: E402
from chest_hunter.config import SUPPORTED_FRAME_SIZES, load_config  # noqa: E402
from chest_hunter.database import DatabaseManager  # noqa: E402
from chest_hunter.scenarios import VISUAL_ARENA_ID, seed_scenario  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=60, help="labelled frames per resolution")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmark_results" / "detector_by_resolution.json")
    args = parser.parse_args(argv)

    base = load_config()
    with DatabaseManager(":memory:") as db:
        db.initialize()
        seed_scenario(db)
        world = load_world(db, base, map_id=VISUAL_ARENA_ID, map_seed=args.seed)
    results: dict[str, Any] = {}
    print(f"{'resolution':10s}  {'method':8s}  precision  recall  false+  class err  ms/frame")
    for width, height in SUPPORTED_FRAME_SIZES:
        vision = dataclasses.replace(base.vision, default_width=width, default_height=height)
        settings = dataclasses.replace(base, vision=vision)
        key = f"{width}x{height}"
        results[key] = compare_detectors(settings, world, frames=args.frames, seed=args.seed)
        for method, r in results[key].items():
            print(f"{key:10s}  {method:8s}  {r['precision'] or 0:9.3f}  {r['recall'] or 0:6.3f}  "
                  f"{r['false_positives']:6d}  {r['class_errors']:9d}  "
                  f"{r['detector_ms_mean']:8.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"map_id": VISUAL_ARENA_ID, "seed": args.seed,
                                       "frames_per_resolution": args.frames,
                                       "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
