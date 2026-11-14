"""Export the behaviour-cloning policy to ONNX (FP32 and INT8) and check the export.

The ONNX graph is built directly from the NumPy weights with ``onnx.helper`` and includes
the output activations, so a consumer gets ready-to-use outputs::

    inputs : image   float32 [N, 3, H, W]  (normalised BGR, see metadata "normalisation")
             proprio float32 [N, 2]        (camera pitch / 60, speed / 7)
    outputs: actions     float32 [N, 4]    tanh: move_forward, move_right, look_yaw, look_pitch
             binary_prob float32 [N, 2]    sigmoid: jump, interact
             phase_prob  float32 [N, 7]    softmax over controller phases

INT8: static post-training quantisation (ONNX Runtime, QDQ format, per-channel weights)
calibrated on training frames. Both models are compared with the NumPy reference model on
held-out frames before they are accepted. ``onnx``/``onnxruntime`` are optional
dependencies: without them the policy still runs on the NumPy backend.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from chest_hunter.learning.behavior_cloning import (
    N_BIN,
    N_CONT,
    N_OUT,
    Architecture,
    PolicyNetwork,
)
from chest_hunter.learning.dataset import Normalisation, Sample, iterate_batches

OPSET = 17


class ExportError(RuntimeError):
    pass


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:                           # pragma: no cover - optional dependency
        raise ExportError("ONNX export needs the 'onnx' package (pip install onnx)") from exc
    return onnx


def build_onnx(net: PolicyNetwork, *, version: str = "") -> Any:
    """ONNX ModelProto equivalent to ``net.forward`` + ``PolicyNetwork.heads``."""
    onnx = _require_onnx()
    from onnx import TensorProto, helper, numpy_helper

    arch = net.arch
    p = net.params
    inits = []

    def init(name: str, value: np.ndarray) -> str:
        inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    nodes = []
    x = "image"
    for i, (_f, k, s, pad) in enumerate(arch.conv):
        nodes.append(helper.make_node("Conv", [x, init(f"conv{i}_w", p[f"conv{i}_w"]),
                                               init(f"conv{i}_b", p[f"conv{i}_b"])],
                                      [f"conv{i}"], kernel_shape=[k, k], strides=[s, s],
                                      pads=[pad, pad, pad, pad]))
        nodes.append(helper.make_node("Relu", [f"conv{i}"], [f"relu{i}"]))
        x = f"relu{i}"
    ho, wo = arch.conv_shapes()[-1]
    nodes.append(helper.make_node("AveragePool", [x], ["pooled"],
                                  kernel_shape=[ho, wo // arch.pool_bins],
                                  strides=[ho, wo // arch.pool_bins]))
    nodes.append(helper.make_node("Flatten", ["pooled"], ["features"], axis=1))
    nodes.append(helper.make_node("Concat", ["features", "proprio"], ["h0"], axis=1))
    nodes.append(helper.make_node("Gemm", ["h0", init("fc0_w", p["fc0_w"]),
                                           init("fc0_b", p["fc0_b"])], ["z1"]))
    nodes.append(helper.make_node("Relu", ["z1"], ["h1"]))
    nodes.append(helper.make_node("Gemm", ["h1", init("fc1_w", p["fc1_w"]),
                                           init("fc1_b", p["fc1_b"])], ["z2"]))
    nodes.append(helper.make_node("Relu", ["z2"], ["h2"]))
    nodes.append(helper.make_node("Gemm", ["h2", init("head_w", p["head_w"]),
                                           init("head_b", p["head_b"])], ["raw"]))
    bounds = {"cont": (0, N_CONT), "bin": (N_CONT, N_CONT + N_BIN),
              "phase": (N_CONT + N_BIN, N_OUT)}
    for name, (a, b) in bounds.items():
        nodes.append(helper.make_node(
            "Slice", ["raw", init(f"{name}_start", np.array([a], np.int64)),
                      init(f"{name}_end", np.array([b], np.int64)),
                      init(f"{name}_axis", np.array([1], np.int64))], [f"{name}_raw"]))
    nodes.append(helper.make_node("Tanh", ["cont_raw"], ["actions"]))
    nodes.append(helper.make_node("Sigmoid", ["bin_raw"], ["binary_prob"]))
    nodes.append(helper.make_node("Softmax", ["phase_raw"], ["phase_prob"], axis=1))
    graph = helper.make_graph(
        nodes, "chest_hunter_policy",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT,
                                       ["N", 3, arch.height, arch.width]),
         helper.make_tensor_value_info("proprio", TensorProto.FLOAT, ["N", arch.proprio])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["N", N_CONT]),
         helper.make_tensor_value_info("binary_prob", TensorProto.FLOAT, ["N", N_BIN]),
         helper.make_tensor_value_info("phase_prob", TensorProto.FLOAT,
                                       ["N", N_OUT - N_CONT - N_BIN])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="chest-hunter-ai")
    model.ir_version = 8
    if version:
        helper.set_model_props(model, {"model_version": version})
    onnx.checker.check_model(model)
    return model


class _Calibration:
    """ONNX Runtime calibration reader over dataset frames."""

    def __init__(self, samples: Sequence[Sample], norm: Normalisation, count: int) -> None:
        self._batches = iterate_batches(list(samples)[:count], norm, batch_size=1, shuffle=False)

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self._batches, None)
        if batch is None:
            return None
        return {"image": batch.images, "proprio": batch.proprio.astype(np.float32)}


def quantise_int8(fp32_path: Path, int8_path: Path, samples: Sequence[Sample],
                  norm: Normalisation, *, calibration_frames: int = 200) -> None:
    try:
        from onnxruntime.quantization import (  # type: ignore[import-untyped]
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:                           # pragma: no cover - optional dependency
        raise ExportError("INT8 quantisation needs onnxruntime") from exc
    source = fp32_path
    prepared = int8_path.with_name(int8_path.stem + "_prep.onnx")
    try:                                    # shape inference + graph optimisation first
        from onnxruntime.quantization.shape_inference import (  # type: ignore[import-untyped]
            quant_pre_process,
        )

        quant_pre_process(str(fp32_path), str(prepared), skip_symbolic_shape=True)
        source = prepared
    except Exception:                       # pragma: no cover - optional optimisation step
        source = fp32_path
    quantize_static(str(source), str(int8_path),
                    _Calibration(samples, norm, calibration_frames),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    if prepared.exists():
        prepared.unlink()


def _session(path: Path, threads: int = 1) -> Any:
    import onnxruntime as ort  # type: ignore[import-untyped]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def compare(net: PolicyNetwork, model_path: Path, samples: Sequence[Sample],
            norm: Normalisation, *, limit: int = 256) -> dict[str, float]:
    """Agreement between the NumPy reference model and an ONNX model."""
    sess = _session(model_path)
    worst = 0.0
    abs_sum = 0.0
    interact_agree = n = 0
    for batch in iterate_batches(list(samples)[:limit], norm, batch_size=32, shuffle=False):
        out, _ = net.forward(batch.images, batch.proprio)
        ref = PolicyNetwork.heads(out)
        got = sess.run(None, {"image": batch.images, "proprio": batch.proprio.astype(np.float32)})
        diff = np.abs(got[0] - ref["continuous"])
        worst = max(worst, float(diff.max()), float(np.abs(got[1] - ref["binary"]).max()),
                    float(np.abs(got[2] - ref["phase"]).max()))
        abs_sum += float(diff.sum())
        interact_agree += int(((got[1][:, 1] > 0.5) == (ref["binary"][:, 1] > 0.5)).sum())
        n += len(batch.images)
    return {"max_abs_diff": round(worst, 6),
            "mean_abs_action_diff": round(abs_sum / (n * N_CONT), 6),
            "interact_decision_agreement": round(interact_agree / n, 4), "samples": float(n)}


def benchmark_latency(fn: Any, runs: int = 300, warmup: int = 20) -> dict[str, float]:
    """Single-observation latency of ``fn()`` in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    return {"mean_ms": round(float(np.mean(times)), 4),
            "median_ms": round(times[len(times) // 2], 4),
            "p95_ms": round(times[int(len(times) * 0.95)], 4), "runs": float(runs)}


def export_policy(model_dir: str | Path, samples_for_checks: Sequence[Sample],
                  calibration_samples: Sequence[Sample], *, int8: bool = True) -> dict[str, Any]:
    """Write policy.onnx (and policy_int8.onnx), verify them, benchmark CPU latency and
    record everything in ``export.json`` next to the model."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "policy.json").read_text(encoding="utf-8"))
    arch = Architecture.from_dict(meta["architecture"])
    norm = Normalisation.from_dict(meta["normalisation"])
    net = PolicyNetwork.load(arch, model_dir / "policy.npz")
    onnx = _require_onnx()
    fp32 = model_dir / "policy.onnx"
    onnx.save(build_onnx(net, version=str(meta.get("model_version", ""))), str(fp32))
    report: dict[str, Any] = {"opset": OPSET, "fp32": {"path": fp32.name,
                                                       "bytes": fp32.stat().st_size}}
    report["fp32"]["agreement"] = compare(net, fp32, samples_for_checks, norm)
    if report["fp32"]["agreement"]["max_abs_diff"] > 1e-3:
        raise ExportError(f"ONNX FP32 output differs from the source model: "
                          f"{report['fp32']['agreement']}")
    if int8:
        q = model_dir / "policy_int8.onnx"
        quantise_int8(fp32, q, calibration_samples, norm)
        report["int8"] = {"path": q.name, "bytes": q.stat().st_size,
                          "agreement": compare(net, q, samples_for_checks, norm)}
    # CPU latency for one observation (batch 1), single thread.
    x = np.zeros((1, 3, arch.height, arch.width), np.float32)
    pr = np.zeros((1, arch.proprio), np.float32)
    latency = {"numpy_fp32": benchmark_latency(lambda: net.forward(x, pr))}
    for key in ("fp32", "int8"):
        if key in report:
            sess = _session(model_dir / report[key]["path"])
            latency[f"onnxruntime_{key}"] = benchmark_latency(
                lambda s=sess: s.run(None, {"image": x, "proprio": pr}))
    report["latency_single_thread"] = latency
    (model_dir / "export.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report
