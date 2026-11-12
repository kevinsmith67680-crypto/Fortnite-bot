"""Appearance model loaded from ``assets/templates`` (templates + colour palette).

The palette gives the reference colour of each rendered element; colour
ranges are widened by the expected sensor noise and lighting variation so
segmentation stays reliable on noisy frames. Templates are rescaled to the
current pixels-per-cell once and cached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from chest_hunter.exceptions import ConfigError

CHEST_CATEGORIES: tuple[str, ...] = ("chest", "chest_opened")
_TEMPLATE_FILES = {"chest": "chests/unopened", "chest_opened": "chests/opened"}


@dataclass(frozen=True)
class ColourRange:
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class TemplateLibrary:
    """Reference templates and colour ranges for perception."""

    directory: Path
    base_ppc: float
    colours: dict[str, tuple[int, int, int]]
    geometry: dict[str, float]
    raw_templates: dict[str, list[np.ndarray]]
    noise_std: float = 6.0
    lighting_variation: float = 0.08
    _scaled: dict[tuple[str, float], list[np.ndarray]] = field(default_factory=dict)
    _ranges: dict[str, ColourRange] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: Path, *, noise_std: float = 6.0,
             lighting_variation: float = 0.08) -> TemplateLibrary:
        palette_path = directory / "palette.json"
        if not palette_path.is_file():
            raise ConfigError(
                f"Perception templates not found in {directory}. Run "
                "'python scripts/generate_templates.py' or set vision.template_dir.")
        palette = json.loads(palette_path.read_text(encoding="utf-8"))
        templates: dict[str, list[np.ndarray]] = {}
        for category, stem in _TEMPLATE_FILES.items():
            files = sorted(directory.glob(f"{stem}*.png"))
            images = [img for img in (cv2.imread(str(f), cv2.IMREAD_COLOR) for f in files)
                      if img is not None]
            if not images:
                raise ConfigError(f"No templates for '{category}' in {directory / stem}*.png")
            templates[category] = images
        return cls(
            directory=directory,
            base_ppc=float(palette["base_pixels_per_cell"]),
            colours={k: tuple(v) for k, v in palette["colours"].items()},
            geometry={k: float(v) for k, v in palette["geometry_cells"].items()},
            raw_templates=templates,
            noise_std=noise_std,
            lighting_variation=lighting_variation,
        )

    def templates(self, category: str, ppc: float) -> list[np.ndarray]:
        """Templates for ``category`` scaled to ``ppc`` pixels per cell (cached)."""
        key = (category, round(ppc, 3))
        cached = self._scaled.get(key)
        if cached is not None:
            return cached
        factor = ppc / self.base_ppc
        scaled = []
        for image in self.raw_templates[category]:
            if abs(factor - 1.0) < 1e-9:
                scaled.append(image)
            else:
                h, w = image.shape[:2]
                size = (max(3, round(w * factor)), max(3, round(h * factor)))
                interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_NEAREST
                scaled.append(cv2.resize(image, size, interpolation=interp))
        self._scaled[key] = scaled
        return scaled

    def colour_range(self, name: str) -> ColourRange:
        """BGR range for a palette colour, widened for noise and lighting."""
        cached = self._ranges.get(name)
        if cached is not None:
            return cached
        colour = np.array(self.colours[name], dtype=np.float32)
        light = self.lighting_variation + 0.04
        spread = 3.5 * self.noise_std + 6.0
        lower = np.clip(colour * (1.0 - light) - spread, 0, 255).astype(np.uint8)
        upper = np.clip(colour * (1.0 + light) + spread, 0, 255).astype(np.uint8)
        result = ColourRange(lower, upper)
        self._ranges[name] = result
        return result

    def size(self, key: str, ppc: float) -> float:
        """A palette geometry value (cells) converted to pixels."""
        return self.geometry[key] * ppc
