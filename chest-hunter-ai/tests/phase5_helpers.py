"""Shared fixtures for the Phase 5 tests: a tiny SYNTHETIC demonstration dataset and a
tiny trained policy, built once per test process in a temporary directory."""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from chest_hunter.config import Settings, load_config

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Path] = {}
_TMP: Path | None = None


def settings() -> Settings:
    return load_config(ROOT / "config" / "settings.json")


def tmp_root() -> Path:
    global _TMP
    if _TMP is None:
        _TMP = Path(tempfile.mkdtemp(prefix="chest_hunter_p5_"))
        atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
    return _TMP


def tiny_dataset() -> Path:
    """Four short scripted (synthetic) episodes on two training maps."""
    if "dataset" not in _CACHE:
        from chest_hunter.learning.demo_generation import record_scripted_demos

        out = tmp_root() / "dataset"
        record_scripted_demos(out, tmp_root() / "experience.db", settings(),
                              maps=["proc_train_000", "proc_train_001"], episodes_per_map=2,
                              seed=1000, max_ticks=360, width=320, height=180)
        _CACHE["dataset"] = out
    return _CACHE["dataset"]


def tiny_model(export: bool = True) -> Path:
    """A policy trained for two epochs on the tiny dataset (exported to ONNX if possible)."""
    if "model" not in _CACHE:
        from chest_hunter.learning.behavior_cloning import TrainConfig, train_policy

        out = tmp_root() / "model"
        cfg = TrainConfig(epochs=2, batch_size=16, patience=3, seed=0,
                          fractions=(0.5, 0.25, 0.25), brightness_jitter=0.0)
        train_policy(tiny_dataset(), out, cfg)
        _CACHE["model"] = out
        if export and onnx_available():
            from chest_hunter.learning.dataset import DemonstrationDataset
            from chest_hunter.learning.model_export import export_policy

            ds = DemonstrationDataset(tiny_dataset())
            samples = ds.samples
            export_policy(out, samples[:64], samples[:32], int8=True)
    return _CACHE["model"]


def onnx_available() -> bool:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True
