"""CPU inference of the exported behaviour-cloning policy.

Backends, in order of preference: ONNX Runtime (FP32 or INT8 model, single thread by
default) and the NumPy reference network (no optional dependency). The policy sees only
what the agent sees: the camera frame (downscaled exactly like the demonstrations) and
its own camera pitch and speed.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chest_hunter.learning.behavior_cloning import Architecture, PolicyNetwork
from chest_hunter.learning.dataset import Normalisation
from chest_hunter.learning.demonstration_recorder import PHASES, downscale, proprioception
from chest_hunter.unreal_bridge.observation_adapter import Telemetry

BACKENDS = ("auto", "onnxruntime", "numpy")


class PolicyLoadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    move_forward: float
    move_right: float
    look_yaw: float
    look_pitch: float
    jump_prob: float
    interact_prob: float
    phase_probs: tuple[float, ...]
    latency_ms: float

    @property
    def phase(self) -> str:
        return PHASES[int(np.argmax(self.phase_probs))]

    def phase_probability(self, *names: str) -> float:
        return float(sum(self.phase_probs[PHASES.index(n)] for n in names))

    def finite(self) -> bool:
        values = (self.move_forward, self.move_right, self.look_yaw, self.look_pitch,
                  self.jump_prob, self.interact_prob, *self.phase_probs)
        return all(math.isfinite(v) for v in values)


class LearnedPolicy:
    def __init__(self, model_dir: str | Path, *, backend: str = "auto", quantized: bool = False,
                 threads: int = 1) -> None:
        if backend not in BACKENDS:
            raise PolicyLoadError(f"backend must be one of {', '.join(BACKENDS)}")
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "policy.json"
        if not meta_path.is_file():
            raise PolicyLoadError(f"no trained policy in {self.model_dir} (policy.json missing); "
                                  "run 'python -m chest_hunter train-policy' first")
        self.metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.arch = Architecture.from_dict(self.metadata["architecture"])
        self.norm = Normalisation.from_dict(self.metadata["normalisation"])
        self.version = str(self.metadata.get("model_version", "unknown"))
        self.calls = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        onnx_file = self.model_dir / ("policy_int8.onnx" if quantized else "policy.onnx")
        self._session: Any = None
        self._net: PolicyNetwork | None = None
        want_ort = backend in ("auto", "onnxruntime")
        if want_ort and onnx_file.is_file():
            try:
                import onnxruntime as ort  # type: ignore[import-untyped]

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = threads
                opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(str(onnx_file), sess_options=opts,
                                                     providers=["CPUExecutionProvider"])
                self.backend = "onnxruntime-int8" if quantized else "onnxruntime-fp32"
            except ImportError:
                if backend == "onnxruntime":
                    raise PolicyLoadError("onnxruntime is not installed") from None
        elif backend == "onnxruntime":
            raise PolicyLoadError(f"{onnx_file.name} not found; run export-policy first")
        if self._session is None:
            if quantized:
                raise PolicyLoadError("the INT8 model needs onnxruntime")
            self._net = PolicyNetwork.load(self.arch, self.model_dir / "policy.npz")
            self.backend = "numpy-fp32"

    # -- metadata ---------------------------------------------------------------------------

    @property
    def synthetic_training_data(self) -> bool:
        return bool(self.metadata.get("dataset", {}).get("synthetic_only", True))

    def validated(self) -> bool:
        """The model beat the "predict the mean action" baseline on its validation split.

        Hybrid control only uses a validated model; otherwise it stays deterministic."""
        metrics = self.metadata.get("metrics", {})
        val, base = metrics.get("validation", {}), metrics.get("baseline_validation", {})
        keys = [k for k in base if k.startswith("mae_")]
        if not keys or any(k not in val for k in keys):
            return False
        return float(np.mean([val[k] for k in keys])) < float(np.mean([base[k] for k in keys]))

    # -- inference --------------------------------------------------------------------------

    def preprocess(self, frame: np.ndarray, telemetry: Telemetry | None
                   ) -> tuple[np.ndarray, np.ndarray]:
        small = downscale(frame, (self.arch.width, self.arch.height))
        return self.norm.apply(small[None]), proprioception(telemetry)[None]

    def raw(self, image: np.ndarray, proprio: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                    np.ndarray]:
        if self._session is not None:
            a, b, p = self._session.run(None, {"image": image.astype(np.float32),
                                               "proprio": proprio.astype(np.float32)})
            return a, b, p
        assert self._net is not None
        out, _ = self._net.forward(image, proprio)
        h = PolicyNetwork.heads(out)
        return h["continuous"], h["binary"], h["phase"]

    def predict(self, frame: np.ndarray, telemetry: Telemetry | None) -> PolicyOutput:
        started = time.perf_counter()
        image, proprio = self.preprocess(frame, telemetry)
        a, b, p = self.raw(image, proprio)
        ms = (time.perf_counter() - started) * 1000.0
        self.calls += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)
        return PolicyOutput(float(a[0, 0]), float(a[0, 1]), float(a[0, 2]), float(a[0, 3]),
                            float(b[0, 0]), float(b[0, 1]), tuple(float(v) for v in p[0]), ms)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0
