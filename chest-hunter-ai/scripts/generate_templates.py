#!/usr/bin/env python3
"""Generate the reference templates and colour palette used by perception.

Writes ``assets/templates/``:

* ``chests/unopened.png``, ``chests/opened.png``: chest sprites at 16 px/cell
* ``interaction/prompt.png``: the interaction prompt icon
* ``obstacles/wall.png``, ``obstacles/ground.png``: one-cell texture samples
* ``agent/marker.png``: the agent marker
* ``palette.json``: reference BGR colours of every rendered element

These files are the perception system's *appearance model*, the equivalent
of a labelled dataset. They are generated from the simulator's sprite
definitions; perception code reads only these files and never imports the
simulator's drawing code. Re-run after changing ``simulation/sprites.py``::

    python scripts/generate_templates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import chest_hunter  # noqa: F401
except ModuleNotFoundError:  # running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from chest_hunter.simulation import sprites

BASE_PPC = 16
ROOT = Path(__file__).resolve().parents[1] / "assets" / "templates"


def main() -> int:
    for sub in ("chests", "interaction", "obstacles", "agent"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "chests" / "unopened.png"),
                sprites.chest_template(BASE_PPC, opened=False))
    cv2.imwrite(str(ROOT / "chests" / "opened.png"),
                sprites.chest_template(BASE_PPC, opened=True))

    prompt = np.full((BASE_PPC, BASE_PPC, 3), sprites.GROUND, np.uint8)
    sprites.draw_prompt(prompt, BASE_PPC / 2, BASE_PPC / 2 + sprites.PROMPT_OFFSET * BASE_PPC,
                        BASE_PPC)
    cv2.imwrite(str(ROOT / "interaction" / "prompt.png"), prompt)

    cv2.imwrite(str(ROOT / "obstacles" / "wall.png"),
                np.full((BASE_PPC, BASE_PPC, 3), sprites.OBSTACLE, np.uint8))
    cv2.imwrite(str(ROOT / "obstacles" / "ground.png"),
                np.full((BASE_PPC, BASE_PPC, 3), sprites.GROUND, np.uint8))

    marker = np.full((BASE_PPC, BASE_PPC, 3), sprites.GROUND, np.uint8)
    sprites.draw_agent(marker, BASE_PPC / 2, BASE_PPC / 2, BASE_PPC, 0.0)
    cv2.imwrite(str(ROOT / "agent" / "marker.png"), marker)

    palette = {
        "base_pixels_per_cell": BASE_PPC,
        "colour_space": "BGR",
        "colours": {
            "ground": sprites.GROUND,
            "out_of_bounds": sprites.OUT_OF_BOUNDS,
            "obstacle": sprites.OBSTACLE,
            "chest_body": sprites.CHEST_BODY,
            "chest_lid": sprites.CHEST_LID,
            "open_body": sprites.OPEN_BODY,
            "open_inside": sprites.OPEN_INSIDE,
            "agent": sprites.AGENT,
            "agent_nose": sprites.AGENT_NOSE,
            "prompt": sprites.PROMPT,
            "progress_fill": sprites.PROGRESS_FILL,
            "reward": sprites.REWARD,
        },
        "geometry_cells": {
            "chest_width": sprites.CHEST_WIDTH,
            "chest_height": sprites.CHEST_HEIGHT,
            "agent_radius": sprites.AGENT_RADIUS,
            "prompt_offset": sprites.PROMPT_OFFSET,
            "prompt_size": sprites.PROMPT_SIZE,
            "progress_offset": sprites.PROGRESS_OFFSET,
            "progress_width": sprites.PROGRESS_WIDTH,
            "progress_height": sprites.PROGRESS_HEIGHT,
        },
    }
    (ROOT / "palette.json").write_text(json.dumps(palette, indent=2) + "\n", encoding="utf-8")
    print(f"Templates written to {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
