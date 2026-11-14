"""Demonstration dataset: validation, normalisation, episode-level splits and lazy batches.

Splits are made by complete **episode** (default) or complete **map**, never by frame:
neighbouring frames of one episode are nearly identical, so a frame-level split would put
near-duplicates in training and validation and overstate validation accuracy. The split is
written to ``splits.json`` the first time and reused afterwards, which keeps the test
episodes untouched while models are tuned on the validation split.

Images are read from disk one batch at a time; only file paths and the small per-sample
arrays are kept in memory.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from chest_hunter.learning.demonstration_recorder import (
    ACTION_FIELDS,
    PHASES,
    SOURCE_HUMAN,
    load_episode,
)

SPLIT_NAMES: Final = ("train", "validation", "test")


class DatasetError(ValueError):
    """The dataset cannot be used (missing, empty, or no valid samples)."""


@dataclass(frozen=True, slots=True)
class Sample:
    episode_id: int
    map_id: str
    frame_path: str
    action: np.ndarray            # (6,) float32 in ACTION_FIELDS order
    proprio: np.ndarray           # (2,) float32
    phase: int                    # -1 when unknown (human demonstrations)
    sim_time: float


@dataclass(slots=True)
class ValidationReport:
    episodes: int = 0
    episodes_rejected: list[str] = field(default_factory=list)
    samples: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {"episodes": self.episodes, "episodes_rejected": self.episodes_rejected,
                "valid_samples": self.samples, "rejected_samples": dict(self.rejected)}


class DemonstrationDataset:
    """All valid samples of the episodes under ``root`` (``episode_*/`` directories)."""

    def __init__(self, root: str | Path, *, frame_size: tuple[int, int] | None = None,
                 check_images: bool = True) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise DatasetError(f"dataset directory not found: {self.root}")
        self.report = ValidationReport()
        self.samples: list[Sample] = []
        self.episodes: dict[int, dict[str, Any]] = {}
        self.frame_size = frame_size
        for directory in sorted(self.root.glob("episode_*")):
            self._load(directory, check_images)
        if not self.samples:
            raise DatasetError(f"no valid samples in {self.root} "
                               f"(rejected: {dict(self.report.rejected)})")

    # -- loading and validation ---------------------------------------------------------------

    def _load(self, directory: Path, check_images: bool) -> None:
        try:
            meta, a = load_episode(directory)
        except (OSError, ValueError, KeyError) as exc:
            self.report.episodes_rejected.append(f"{directory.name}: {exc}")
            return
        size = tuple(meta.get("frame_size", ()))
        if self.frame_size is None:
            self.frame_size = (int(size[0]), int(size[1]))
        layout_ok = list(meta.get("action_fields", [])) == list(ACTION_FIELDS)
        if self.frame_size is None or size != self.frame_size or not layout_ok:
            self.report.episodes_rejected.append(f"{directory.name}: incompatible frame size "
                                                 "or action layout")
            return
        self.report.episodes += 1
        episode_id = int(meta["episode_id"])
        self.episodes[episode_id] = meta
        w, h = self.frame_size
        n_frames = int(meta.get("frames", 0))
        for i in np.nonzero(a["frame_ref"] >= 0)[0]:
            reason = _check_sample(a, int(i), n_frames)
            path = directory / "frames" / f"{int(a['frame_ref'][i]):06d}.png"
            if reason is None and check_images:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    reason = "unreadable_image"
                elif img.shape != (h, w, 3):
                    reason = "wrong_image_size"
            if reason is not None:
                self.report.rejected[reason] += 1
                continue
            self.samples.append(Sample(episode_id, str(meta["map_id"]), str(path),
                                       a["action"][i].astype(np.float32),
                                       a["proprio"][i].astype(np.float32), int(a["phase"][i]),
                                       float(a["action_time"][i])))
            self.report.samples += 1

    # -- statistics -------------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        actions = np.stack([s.action for s in self.samples])
        phases = Counter(PHASES[s.phase] if s.phase >= 0 else "unknown" for s in self.samples)
        sources = Counter(str(m.get("source")) for m in self.episodes.values())
        maps = Counter(s.map_id for s in self.samples)
        return {
            "episodes": len(self.episodes), "samples": len(self.samples),
            "maps": len(maps), "samples_per_map": dict(sorted(maps.items())),
            "sources": dict(sources),
            "synthetic_only": SOURCE_HUMAN not in sources,
            "frame_size": list(self.frame_size or ()),
            "action_mean": {k: round(float(actions[:, i].mean()), 4)
                            for i, k in enumerate(ACTION_FIELDS)},
            "action_std": {k: round(float(actions[:, i].std()), 4)
                           for i, k in enumerate(ACTION_FIELDS)},
            "phases": dict(phases),
            "validation": self.report.to_dict(),
        }

    # -- splits ---------------------------------------------------------------------------

    def split(self, *, by: str = "episode", fractions: tuple[float, float, float]
              = (0.7, 0.15, 0.15), seed: int = 0, persist: bool = True) -> dict[str, list[int]]:
        """Episode ids per split. Reuses ``splits.json`` when it covers these episodes."""
        if by not in ("episode", "map"):
            raise DatasetError("split 'by' must be 'episode' or 'map'")
        path = self.root / "splits.json"
        ids = sorted(self.episodes)
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("by") != by:
                raise DatasetError(
                    f"{path} already splits this dataset by {saved.get('by')}; re-splitting by "
                    f"{by} could move test episodes into training. Delete splits.json to "
                    "start a new split deliberately.")
            covered = sorted(i for name in SPLIT_NAMES for i in saved["episodes"][name])
            if covered == ids and saved.get("by") == by:
                return {name: list(saved["episodes"][name]) for name in SPLIT_NAMES}
            if any(i in covered for i in ids) and set(covered) - set(ids) == set():
                # New episodes were added: keep the old assignment, place the new ones.
                assignment = {i: name for name in SPLIT_NAMES for i in saved["episodes"][name]}
                return self._assign(ids, by, fractions, seed, assignment, path if persist else None)
        return self._assign(ids, by, fractions, seed, {}, path if persist else None)

    def _assign(self, ids: list[int], by: str, fractions: tuple[float, float, float], seed: int,
                fixed: dict[int, str], path: Path | None) -> dict[str, list[int]]:
        if abs(sum(fractions) - 1.0) > 1e-6 or min(fractions) < 0:
            raise DatasetError("split fractions must be non-negative and sum to 1")
        rng = random.Random(seed)
        groups: dict[str, list[int]] = {}
        for i in ids:
            key = str(self.episodes[i]["map_id"]) if by == "map" else str(i)
            groups.setdefault(key, []).append(i)
        keys = sorted(groups)
        rng.shuffle(keys)
        out: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
        for i, name in fixed.items():
            if i in self.episodes:
                out[name].append(i)
        todo = [k for k in keys if not any(i in fixed for i in groups[k])]
        n = len(todo)
        n_test = max(1, round(fractions[2] * n)) if n >= 3 and fractions[2] > 0 else 0
        n_val = max(1, round(fractions[1] * n)) if n - n_test >= 2 and fractions[1] > 0 else 0
        for j, key in enumerate(todo):
            name = "test" if j < n_test else ("validation" if j < n_test + n_val else "train")
            out[name].extend(groups[key])
        for name in SPLIT_NAMES:
            out[name].sort()
        if path is not None:
            path.write_text(json.dumps({"by": by, "seed": seed, "fractions": list(fractions),
                                        "episodes": out}, indent=1) + "\n", encoding="utf-8")
        return out

    def subset(self, episode_ids: Sequence[int]) -> list[Sample]:
        keep = set(episode_ids)
        return [s for s in self.samples if s.episode_id in keep]


def _check_sample(a: dict[str, np.ndarray], i: int, n_frames: int) -> str | None:
    action = a["action"][i]
    if not np.all(np.isfinite(action)):
        return "non_finite_action"
    if np.any(np.abs(action[:4]) > 1.0 + 1e-6):
        return "action_out_of_range"
    if not np.all(np.isin(action[4:], (0.0, 1.0))):
        return "binary_action_not_boolean"
    if not 0 <= int(a["frame_ref"][i]) < max(1, n_frames):
        return "missing_frame"
    if not math.isfinite(float(a["frame_time"][i])) \
            or abs(float(a["frame_time"][i]) - float(a["action_time"][i])) > 1e-6:
        return "misaligned_frame"
    if not np.all(np.isfinite(a["proprio"][i])):
        return "non_finite_proprio"
    return None


# -- normalisation and batches ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Normalisation:
    """Per-channel image statistics (on 0-1 scaled BGR) from the training split only."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def apply(self, images_bgr_uint8: np.ndarray) -> np.ndarray:
        """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 normalised."""
        x: np.ndarray = images_bgr_uint8.astype(np.float32) / np.float32(255.0)
        x = (x - np.array(self.mean, np.float32)) / np.array(self.std, np.float32)
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Normalisation:
        mean, std = data["mean"], data["std"]
        return cls((float(mean[0]), float(mean[1]), float(mean[2])),
                   (float(std[0]), float(std[1]), float(std[2])))


def compute_normalisation(samples: Sequence[Sample], *, max_images: int = 2000,
                          seed: int = 0) -> Normalisation:
    if not samples:
        raise DatasetError("cannot compute normalisation without samples")
    rng = random.Random(seed)
    chosen = samples if len(samples) <= max_images else rng.sample(list(samples), max_images)
    total = np.zeros(3)
    total_sq = np.zeros(3)
    count = 0
    for s in chosen:
        img = cv2.imread(s.frame_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        x = img.reshape(-1, 3).astype(np.float64) / 255.0
        total += x.sum(axis=0)
        total_sq += (x * x).sum(axis=0)
        count += x.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-8))
    return Normalisation((float(mean[0]), float(mean[1]), float(mean[2])),
                         (float(std[0]), float(std[1]), float(std[2])))


@dataclass(slots=True)
class Batch:
    images: np.ndarray            # (N, 3, H, W) float32, normalised
    proprio: np.ndarray           # (N, 2)
    actions: np.ndarray           # (N, 6)
    phases: np.ndarray            # (N,) int (-1 unknown)


def iterate_batches(samples: Sequence[Sample], norm: Normalisation, *, batch_size: int,
                    shuffle: bool, seed: int = 0, brightness_jitter: float = 0.0
                    ) -> Iterator[Batch]:
    """Yield batches, reading only that batch's images from disk."""
    if batch_size < 1:
        raise DatasetError("batch_size must be >= 1")
    order = list(range(len(samples)))
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [samples[i] for i in order[start:start + batch_size]]
        images = np.stack([_read(s.frame_path) for s in chunk])
        if brightness_jitter > 0:
            factors = rng.uniform(1 - brightness_jitter, 1 + brightness_jitter, len(chunk))
            images = np.clip(images.astype(np.float32) * factors[:, None, None, None], 0, 255
                             ).astype(np.uint8)
        yield Batch(norm.apply(images), np.stack([s.proprio for s in chunk]),
                    np.stack([s.action for s in chunk]),
                    np.array([s.phase for s in chunk], dtype=np.int64))


def _read(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise DatasetError(f"image disappeared or became unreadable: {path}")
    return img
