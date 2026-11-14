"""Behaviour cloning of a compact convolutional policy, trained offline in NumPy.

Architecture (input 3 x 72 x 128 BGR, normalised; plus 2 proprioceptive values)::

    conv 5x5/2, 16 ch, ReLU   -> 16 x 36 x 64
    conv 3x3/2, 32 ch, ReLU   -> 32 x 18 x 32
    conv 3x3/2, 32 ch, ReLU   -> 32 x  9 x 16
    average pool over 4 vertical strips (keeps coarse left/right position) -> 128
    concat [camera pitch, speed] -> 130 -> FC 64 ReLU -> FC 64 ReLU -> FC 13
    heads: tanh(4)  move_forward, move_right, look_yaw, look_pitch
           logit(2) jump, interact
           logit(7) controller phase (auxiliary; its confidence gates hybrid control)

About 28 k parameters. Loss::

    L = lm * Huber(move) + lc * Huber(camera) + li * BCE(interact) + lj * BCE(jump)
        + lp * CE(phase)

PyTorch was not used: its CPU wheel index was unreachable from the development machine
and the default Linux build pulls about 3 GB of CUDA libraries, which a 28 k-parameter
model does not justify. The NumPy implementation is checked against numerical gradients
in the tests, trains on a CPU, and exports to ONNX (``model_export``).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from chest_hunter.learning.dataset import (
    DemonstrationDataset,
    Normalisation,
    Sample,
    compute_normalisation,
    iterate_batches,
)
from chest_hunter.learning.demonstration_recorder import ACTION_FIELDS, PHASES

MODEL_FORMAT: Final = "chest-hunter-policy/1"
N_CONT: Final = 4
N_BIN: Final = 2
N_PHASE: Final = len(PHASES)
N_OUT: Final = N_CONT + N_BIN + N_PHASE


@dataclass(frozen=True, slots=True)
class Architecture:
    width: int = 128
    height: int = 72
    conv: tuple[tuple[int, int, int, int], ...] = ((16, 5, 2, 2), (32, 3, 2, 1), (32, 3, 2, 1))
    pool_bins: int = 4
    hidden: int = 64
    proprio: int = 2

    def conv_shapes(self) -> list[tuple[int, int]]:
        h, w = self.height, self.width
        out = []
        for _f, k, s, p in self.conv:
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
            out.append((h, w))
        return out

    def validate(self) -> None:
        h, w = self.conv_shapes()[-1]
        if w % self.pool_bins:
            raise ValueError(f"final feature width {w} is not divisible by {self.pool_bins} bins")
        if h < 1 or w < 1:
            raise ValueError("input too small for the convolution stack")

    @property
    def features(self) -> int:
        return self.conv[-1][0] * self.pool_bins

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "conv": [list(c) for c in self.conv],
                "pool_bins": self.pool_bins, "hidden": self.hidden, "proprio": self.proprio}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Architecture:
        return cls(int(d["width"]), int(d["height"]),
                   tuple((int(c[0]), int(c[1]), int(c[2]), int(c[3])) for c in d["conv"]),
                   int(d["pool_bins"]), int(d["hidden"]), int(d["proprio"]))


# -- network ---------------------------------------------------------------------------------


def _im2col(x: np.ndarray, k: int, s: int, p: int) -> tuple[np.ndarray, int, int]:
    """x: (N, H, W, C) -> cols (N*Ho*Wo, C*k*k) in (C, kh, kw) order, Ho, Wo."""
    if p:
        x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
    win = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::s, ::s]   # (N, Ho, Wo, C, k, k)
    n, ho, wo = win.shape[:3]
    return win.reshape(n * ho * wo, -1), ho, wo


class PolicyNetwork:
    """Parameters plus forward/backward passes. Weights use ONNX layouts: convolutions
    (out, in, kh, kw); fully connected layers (in, out)."""

    def __init__(self, arch: Architecture, params: dict[str, np.ndarray]) -> None:
        arch.validate()
        self.arch = arch
        self.params = params

    @classmethod
    def initialise(cls, arch: Architecture, seed: int = 0,
                   dtype: type[np.floating[Any]] = np.float32) -> PolicyNetwork:
        rng = np.random.default_rng(seed)
        params: dict[str, np.ndarray] = {}
        c_in = 3
        for i, (f, k, _s, _p) in enumerate(arch.conv):
            fan_in = c_in * k * k
            params[f"conv{i}_w"] = rng.normal(0, math.sqrt(2.0 / fan_in), (f, c_in, k, k))
            params[f"conv{i}_b"] = np.zeros(f)
            c_in = f
        sizes = [arch.features + arch.proprio, arch.hidden, arch.hidden]
        for i in range(2):
            params[f"fc{i}_w"] = rng.normal(0, math.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
            params[f"fc{i}_b"] = np.zeros(sizes[i + 1])
        params["head_w"] = rng.normal(0, 0.01, (arch.hidden, N_OUT))
        params["head_b"] = np.zeros(N_OUT)
        return cls(arch, {k: v.astype(dtype) for k, v in params.items()})

    @property
    def parameter_count(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def forward(self, images_nchw: np.ndarray, proprio: np.ndarray,
                cache: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Raw outputs (N, 13): 4 pre-tanh values, 2 logits, 7 phase logits."""
        p = self.params
        dtype = p["head_w"].dtype
        x = np.ascontiguousarray(images_nchw.transpose(0, 2, 3, 1), dtype=dtype)   # NHWC
        n = x.shape[0]
        saved: dict[str, Any] = {}
        for i, (f, k, s, pad) in enumerate(self.arch.conv):
            cols, ho, wo = _im2col(x, k, s, pad)
            w = p[f"conv{i}_w"].reshape(f, -1)
            z = cols @ w.T + p[f"conv{i}_b"]
            y = np.maximum(z, 0.0)
            if cache:
                saved[f"conv{i}"] = (x.shape, cols, z > 0, ho, wo)
            x = y.reshape(n, ho, wo, f)
        bins = self.arch.pool_bins
        _, ho, wo, f = x.shape
        pooled = x.mean(axis=1).reshape(n, bins, wo // bins, f).mean(axis=2)   # (N, bins, F)
        feat = pooled.transpose(0, 2, 1).reshape(n, f * bins)                  # F-major
        h0 = np.concatenate([feat, proprio.astype(dtype)], axis=1)
        z1 = h0 @ p["fc0_w"] + p["fc0_b"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ p["fc1_w"] + p["fc1_b"]
        h2 = np.maximum(z2, 0.0)
        out = h2 @ p["head_w"] + p["head_b"]
        if cache:
            saved.update(h0=h0, z1=z1, h1=h1, z2=z2, h2=h2, final_shape=(n, ho, wo, f))
        return out, saved

    def backward(self, d_out: np.ndarray, saved: dict[str, Any]) -> dict[str, np.ndarray]:
        p = self.params
        g: dict[str, np.ndarray] = {}
        g["head_w"] = saved["h2"].T @ d_out
        g["head_b"] = d_out.sum(axis=0)
        d = (d_out @ p["head_w"].T) * (saved["z2"] > 0)
        g["fc1_w"] = saved["h1"].T @ d
        g["fc1_b"] = d.sum(axis=0)
        d = (d @ p["fc1_w"].T) * (saved["z1"] > 0)
        g["fc0_w"] = saved["h0"].T @ d
        g["fc0_b"] = d.sum(axis=0)
        d_h0 = d @ p["fc0_w"].T
        n, ho, wo, f = saved["final_shape"]
        bins = self.arch.pool_bins
        d_feat = d_h0[:, :f * bins].reshape(n, f, bins).transpose(0, 2, 1)     # (N, bins, F)
        scale = 1.0 / (ho * (wo // bins))
        d_x = np.repeat(d_feat, wo // bins, axis=1)[:, None, :, :] * scale    # (N, 1, Wo, F)
        d_x = np.broadcast_to(d_x, (n, ho, wo, f))
        for i in range(len(self.arch.conv) - 1, -1, -1):
            f_i, k, s, pad = self.arch.conv[i]
            in_shape, cols, mask, ho_i, wo_i = saved[f"conv{i}"]
            d_z = (d_x.reshape(-1, f_i) * mask)
            g[f"conv{i}_w"] = (d_z.T @ cols).reshape(p[f"conv{i}_w"].shape)
            g[f"conv{i}_b"] = d_z.sum(axis=0)
            if i == 0:
                break
            d_cols = d_z @ p[f"conv{i}_w"].reshape(f_i, -1)
            nb, h_in, w_in, c_in = in_shape
            d_cols6 = d_cols.reshape(nb, ho_i, wo_i, c_in, k, k)
            d_pad = np.zeros((nb, h_in + 2 * pad, w_in + 2 * pad, c_in), dtype=d_cols.dtype)
            for a in range(k):
                for b in range(k):
                    d_pad[:, a:a + s * ho_i:s, b:b + s * wo_i:s, :] += d_cols6[..., a, b]
            d_x = d_pad[:, pad:pad + h_in, pad:pad + w_in, :] if pad else d_pad
        return g

    # -- outputs ----------------------------------------------------------------------------

    @staticmethod
    def heads(out: np.ndarray) -> dict[str, np.ndarray]:
        cont = np.tanh(out[:, :N_CONT])
        bin_p = 1.0 / (1.0 + np.exp(-out[:, N_CONT:N_CONT + N_BIN]))
        logits = out[:, N_CONT + N_BIN:]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return {"continuous": cont, "binary": bin_p, "phase": e / e.sum(axis=1, keepdims=True)}

    def save(self, path: str | Path) -> None:
        arrays: dict[str, Any] = {k: v.astype(np.float32) for k, v in self.params.items()}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, arch: Architecture, path: str | Path) -> PolicyNetwork:
        with np.load(path) as data:
            params = {k: data[k].astype(np.float32) for k in data.files}
        return cls(arch, params)


# -- loss ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LossWeights:
    movement: float = 1.0
    camera: float = 1.0
    interaction: float = 1.0
    jump: float = 0.2
    phase: float = 0.3
    huber_delta: float = 0.25
    interact_pos_weight: float = 1.0


def loss_and_grad(out: np.ndarray, actions: np.ndarray, phases: np.ndarray,
                  w: LossWeights) -> tuple[float, dict[str, float], np.ndarray]:
    """Combined loss, its parts, and dL/d(out) (mean over the batch)."""
    n = out.shape[0]
    d = np.zeros_like(out)
    parts: dict[str, float] = {}
    pred = np.tanh(out[:, :N_CONT])
    err = pred - actions[:, :N_CONT].astype(out.dtype)
    delta = w.huber_delta
    absd = np.abs(err)
    huber = np.where(absd <= delta, 0.5 * err * err, delta * (absd - 0.5 * delta))
    d_huber = np.where(absd <= delta, err, delta * np.sign(err))
    for name, cols, lam in (("movement", slice(0, 2), w.movement),
                            ("camera", slice(2, 4), w.camera)):
        parts[name] = float(huber[:, cols].mean())
        d[:, cols] += lam * d_huber[:, cols] / (n * 2) * (1.0 - pred[:, cols] ** 2)
    for j, (name, lam, pos_w) in enumerate((("jump", w.jump, 1.0),
                                             ("interaction", w.interaction,
                                              w.interact_pos_weight))):
        z = out[:, N_CONT + j]
        y = actions[:, N_CONT + j].astype(out.dtype)
        # numerically stable BCE with logits and a positive-class weight
        log_p = -np.logaddexp(0.0, -z)
        log_1mp = -np.logaddexp(0.0, z)
        bce = -(pos_w * y * log_p + (1 - y) * log_1mp)
        parts[name] = float(bce.mean())
        sig = 1.0 / (1.0 + np.exp(-z))
        d[:, N_CONT + j] += lam * (pos_w * y * (sig - 1) + (1 - y) * sig) / n
    known = phases >= 0
    logits = out[:, N_CONT + N_BIN:]
    if known.any():
        lk = logits[known]
        m = lk.max(axis=1, keepdims=True)
        log_z = m[:, 0] + np.log(np.exp(lk - m).sum(axis=1))
        target = phases[known]
        ce = log_z - lk[np.arange(len(target)), target]
        parts["phase"] = float(ce.mean())
        soft = np.exp(lk - log_z[:, None])
        soft[np.arange(len(target)), target] -= 1.0
        d_phase = np.zeros_like(logits)
        d_phase[known] = soft / known.sum()
        d[:, N_CONT + N_BIN:] += w.phase * d_phase
    else:
        parts["phase"] = 0.0
    total = (w.movement * parts["movement"] + w.camera * parts["camera"]
             + w.interaction * parts["interaction"] + w.jump * parts["jump"]
             + w.phase * parts["phase"])
    return total, parts, d


# -- optimiser -------------------------------------------------------------------------------


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3, betas: tuple[float, float]
                 = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            update = (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
            if self.wd and k.endswith("_w"):
                params[k] -= self.lr * self.wd * params[k]
            params[k] -= (self.lr * update).astype(params[k].dtype)


# -- training ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-4
    seed: int = 0
    split_by: str = "episode"
    split_seed: int = 0
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    brightness_jitter: float = 0.1
    max_train_samples: int | None = None
    weights: LossWeights = field(default_factory=LossWeights)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        d = dict(d)
        if "weights" in d:
            d["weights"] = LossWeights(**d["weights"])
        if "fractions" in d:
            d["fractions"] = tuple(d["fractions"])
        return cls(**d)


def evaluate(net: PolicyNetwork, samples: Sequence[Sample], norm: Normalisation,
             weights: LossWeights, batch_size: int = 128) -> dict[str, float]:
    """Loss parts and interpretable metrics on a set of samples (no augmentation)."""
    if not samples:
        return {}
    totals: dict[str, float] = {}
    abs_err = np.zeros(N_CONT)
    n = 0
    tp = fp = fn = tn = 0
    phase_hits = phase_n = 0
    for batch in iterate_batches(samples, norm, batch_size=batch_size, shuffle=False):
        out, _ = net.forward(batch.images, batch.proprio)
        loss, parts, _ = loss_and_grad(out, batch.actions, batch.phases, weights)
        b = len(batch.actions)
        totals["loss"] = totals.get("loss", 0.0) + loss * b
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * b
        heads = PolicyNetwork.heads(out)
        abs_err += np.abs(heads["continuous"] - batch.actions[:, :N_CONT]).sum(axis=0)
        pred_i = heads["binary"][:, 1] > 0.5
        true_i = batch.actions[:, N_CONT + 1] > 0.5
        tp += int((pred_i & true_i).sum())
        fp += int((pred_i & ~true_i).sum())
        fn += int((~pred_i & true_i).sum())
        tn += int((~pred_i & ~true_i).sum())
        known = batch.phases >= 0
        phase_hits += int((heads["phase"][known].argmax(axis=1) == batch.phases[known]).sum())
        phase_n += int(known.sum())
        n += b
    metrics = {k: v / n for k, v in totals.items()}
    for i, key in enumerate(ACTION_FIELDS[:N_CONT]):
        metrics[f"mae_{key}"] = float(abs_err[i] / n)
    metrics["interact_accuracy"] = (tp + tn) / n
    metrics["interact_precision"] = tp / (tp + fp) if tp + fp else 0.0
    metrics["interact_recall"] = tp / (tp + fn) if tp + fn else 0.0
    metrics["phase_accuracy"] = phase_hits / phase_n if phase_n else 0.0
    metrics["samples"] = float(n)
    return {k: round(float(v), 6) for k, v in metrics.items()}


def baseline_metrics(train: Sequence[Sample], evaluation: Sequence[Sample]) -> dict[str, float]:
    """MAE of always predicting the training mean (what the model must beat)."""
    if not train or not evaluation:
        return {}
    mean = np.mean(np.stack([s.action for s in train]), axis=0)
    actions = np.stack([s.action for s in evaluation])
    out = {f"mae_{k}": round(float(np.abs(actions[:, i] - mean[i]).mean()), 6)
           for i, k in enumerate(ACTION_FIELDS[:N_CONT])}
    out["interact_positive_rate"] = round(float(actions[:, N_CONT + 1].mean()), 6)
    return out


@dataclass(slots=True)
class TrainResult:
    output_dir: Path
    epochs_run: int
    best_epoch: int
    stopped_early: bool
    history: list[dict[str, Any]]
    metadata: dict[str, Any]


def train_policy(dataset_root: str | Path, output_dir: str | Path, config: TrainConfig, *,
                 arch: Architecture | None = None,
                 progress: Callable[[dict[str, Any]], None] | None = None) -> TrainResult:
    """Train with early stopping on the validation split; the test split is only scored
    once at the end with the selected (best-validation) weights."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DemonstrationDataset(dataset_root)
    assert ds.frame_size is not None
    arch = arch or Architecture(width=ds.frame_size[0], height=ds.frame_size[1])
    splits = ds.split(by=config.split_by, fractions=config.fractions, seed=config.split_seed)
    train = ds.subset(splits["train"])
    val = ds.subset(splits["validation"])
    test = ds.subset(splits["test"])
    if not train or not val:
        raise ValueError("training needs at least one training and one validation episode "
                         f"(have {len(splits['train'])} / {len(splits['validation'])})")
    if config.max_train_samples is not None and len(train) > config.max_train_samples:
        rng = np.random.default_rng(config.seed)
        keep = sorted(rng.choice(len(train), config.max_train_samples, replace=False).tolist())
        train = [train[i] for i in keep]
    norm = compute_normalisation(train, seed=config.seed)
    positives = float(np.mean([s.action[N_CONT + 1] for s in train]))
    pos_weight = min(20.0, (1 - positives) / positives) if positives > 0 else 1.0
    weights = LossWeights(**{**asdict(config.weights), "interact_pos_weight":
                             config.weights.interact_pos_weight
                             if config.weights.interact_pos_weight != 1.0 else pos_weight})
    net = PolicyNetwork.initialise(arch, seed=config.seed)
    opt = Adam(net.params, lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    best = (math.inf, -1)
    bad_epochs = 0
    stopped_early = False
    best_path = out_dir / "policy_best.npz"
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        run_loss, seen = 0.0, 0
        for batch in iterate_batches(train, norm, batch_size=config.batch_size, shuffle=True,
                                     seed=config.seed * 1000 + epoch,
                                     brightness_jitter=config.brightness_jitter):
            out, saved = net.forward(batch.images, batch.proprio, cache=True)
            loss, _parts, d_out = loss_and_grad(out, batch.actions, batch.phases, weights)
            grads = net.backward(d_out, saved)
            opt.step(net.params, grads)
            run_loss += loss * len(batch.actions)
            seen += len(batch.actions)
        val_metrics = evaluate(net, val, norm, weights)
        entry = {"epoch": epoch, "train_loss": round(run_loss / seen, 6),
                 "validation_loss": val_metrics["loss"], "validation": val_metrics,
                 "seconds": round(time.perf_counter() - started, 2)}
        history.append(entry)
        if progress is not None:
            progress(entry)
        if val_metrics["loss"] < best[0] - config.min_delta:
            best = (val_metrics["loss"], epoch)
            bad_epochs = 0
            net.save(best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                stopped_early = True
                break
        (out_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n",
                                              encoding="utf-8")
    best_net = PolicyNetwork.load(arch, best_path)
    final = {"validation": evaluate(best_net, val, norm, weights),
             "test": evaluate(best_net, test, norm, weights),
             "baseline_test": baseline_metrics(train, test),
             "baseline_validation": baseline_metrics(train, val)}
    metadata = {
        "format": MODEL_FORMAT,
        "model_version": f"bc-{int(time.time())}",
        "architecture": arch.to_dict(),
        "parameters": best_net.parameter_count,
        "inputs": {"image": [1, 3, arch.height, arch.width], "image_layout": "NCHW BGR",
                   "proprio": [1, arch.proprio], "proprio_fields": ["camera_pitch_deg/60",
                                                                   "speed_mps/7"]},
        "outputs": {"continuous": list(ACTION_FIELDS[:N_CONT]),
                    "continuous_activation": "tanh, range [-1, 1]",
                    "binary_logits": list(ACTION_FIELDS[N_CONT:]),
                    "phase_logits": list(PHASES),
                    "raw_layout": "13 values: 4 pre-tanh, 2 logits, 7 phase logits"},
        "normalisation": norm.to_dict(),
        "training": {**asdict(config), "weights": asdict(weights)},
        "dataset": {**ds.statistics(), "root": str(Path(dataset_root)),
                    "splits": {k: len(v) for k, v in splits.items()},
                    "split_episodes": splits, "train_samples_used": len(train)},
        "epochs_run": len(history), "best_epoch": best[1], "stopped_early": stopped_early,
        "metrics": final,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
    best_net.save(out_dir / "policy.npz")
    (out_dir / "policy.json").write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")
    return TrainResult(out_dir, len(history), best[1], stopped_early, history, metadata)
