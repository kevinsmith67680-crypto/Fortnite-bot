"""Dataset pipeline, behaviour cloning, ONNX export/quantisation and CPU inference."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from chest_hunter.learning.behavior_cloning import (
    Architecture,
    LossWeights,
    PolicyNetwork,
    loss_and_grad,
)
from chest_hunter.learning.dataset import (
    DatasetError,
    DemonstrationDataset,
    Normalisation,
    compute_normalisation,
    iterate_batches,
)
from chest_hunter.learning.learned_policy import LearnedPolicy, PolicyLoadError
from tests.phase5_helpers import onnx_available, tiny_dataset, tiny_model

TINY = Architecture(width=16, height=8, conv=((3, 3, 2, 1), (4, 3, 2, 1)), pool_bins=2,
                    hidden=5, proprio=2)


class DatasetPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data"
        shutil.copytree(tiny_dataset(), self.root)

    def test_statistics_and_labels(self) -> None:
        ds = DemonstrationDataset(self.root)
        st = ds.statistics()
        self.assertEqual(st["episodes"], 4)
        self.assertEqual(st["maps"], 2)
        self.assertTrue(st["synthetic_only"])
        self.assertEqual(st["frame_size"], [128, 72])
        self.assertEqual(st["validation"]["rejected_samples"], {})

    def test_split_is_by_episode_and_persistent(self) -> None:
        ds = DemonstrationDataset(self.root)
        split = ds.split(by="episode", fractions=(0.5, 0.25, 0.25), seed=3)
        groups = [set(v) for v in split.values()]
        self.assertEqual(sum(len(g) for g in groups), 4)
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertFalse(groups[a] & groups[b])             # no episode in two splits
        train_eps = {s.episode_id for s in ds.subset(split["train"])}
        test_eps = {s.episode_id for s in ds.subset(split["test"])}
        self.assertFalse(train_eps & test_eps)
        self.assertTrue((self.root / "splits.json").exists())
        again = DemonstrationDataset(self.root).split(by="episode", seed=99)   # other seed
        self.assertEqual(again, split)                               # persisted assignment

    def test_split_by_map_keeps_maps_together(self) -> None:
        ds = DemonstrationDataset(self.root)
        split = ds.split(by="map", fractions=(0.5, 0.5, 0.0), seed=1, persist=False)
        for name, ids in split.items():
            maps = {ds.episodes[i]["map_id"] for i in ids}
            self.assertLessEqual(len(maps), 1, name)

    def test_corrupted_samples_are_removed(self) -> None:
        ep = sorted(self.root.glob("episode_*"))[0]
        with np.load(ep / "steps.npz") as data:
            arrays = {k: data[k] for k in data.files}
        rows = np.nonzero(arrays["frame_ref"] >= 0)[0]
        arrays["action"][rows[0], 0] = np.nan                       # non-finite action
        arrays["action"][rows[1], 1] = 3.0                          # out of range
        arrays["frame_time"][rows[2]] += 0.5                        # misaligned frame
        arrays["action"][rows[3], 5] = 0.5                          # binary not boolean
        np.savez_compressed(ep / "steps.npz", **arrays)
        frame = ep / "frames" / f"{int(arrays['frame_ref'][rows[4]]):06d}.png"
        bad_size = cv2.imread(str(frame))[:10, :10]
        cv2.imwrite(str(frame), bad_size)                          # wrong image size
        ds = DemonstrationDataset(self.root)
        rejected = ds.report.rejected
        for reason in ("non_finite_action", "action_out_of_range", "misaligned_frame",
                       "binary_action_not_boolean", "wrong_image_size"):
            self.assertGreaterEqual(rejected[reason], 1, reason)

    def test_unreadable_episode_and_empty_dataset(self) -> None:
        (sorted(self.root.glob("episode_*"))[0] / "steps.npz").write_bytes(b"garbage")
        ds = DemonstrationDataset(self.root)
        self.assertEqual(len(ds.report.episodes_rejected), 1)
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(DatasetError):
            DemonstrationDataset(empty)

    def test_normalisation_and_lazy_batches(self) -> None:
        ds = DemonstrationDataset(self.root)
        norm = compute_normalisation(ds.samples, max_images=50)
        batch = next(iterate_batches(ds.samples, norm, batch_size=8, shuffle=True, seed=1))
        self.assertEqual(batch.images.shape, (8, 3, 72, 128))
        self.assertEqual(batch.images.dtype, np.float32)
        self.assertLess(abs(float(batch.images.mean())), 1.0)
        restored = Normalisation.from_dict(norm.to_dict())
        self.assertEqual(restored, norm)
        # Lazy: the dataset keeps paths, not pixels.
        self.assertTrue(all(isinstance(s.frame_path, str) for s in ds.samples))
        self.assertFalse(any(hasattr(s, "image") for s in ds.samples))


class NetworkAndLoss(unittest.TestCase):
    def test_gradients_match_numerical_derivatives(self) -> None:
        net = PolicyNetwork.initialise(TINY, seed=2, dtype=np.float64)
        rng = np.random.default_rng(0)
        x, pp = rng.normal(size=(3, 3, 8, 16)), rng.normal(size=(3, 2))
        acts = np.concatenate([rng.uniform(-1, 1, (3, 4)), (rng.random((3, 2)) > 0.5)], 1)
        phases = np.array([1, 4, -1])
        w = LossWeights(interact_pos_weight=3.0)
        out, saved = net.forward(x, pp, cache=True)
        _, _, d = loss_and_grad(out, acts, phases, w)
        grads = net.backward(d, saved)
        for key in ("conv0_w", "conv1_b", "fc0_w", "head_w"):
            v = net.params[key]
            idx = tuple(int(i) for i in np.unravel_index(1, v.shape))
            old = v[idx]
            v[idx] = old + 1e-6
            lp = loss_and_grad(net.forward(x, pp)[0], acts, phases, w)[0]
            v[idx] = old - 1e-6
            lm = loss_and_grad(net.forward(x, pp)[0], acts, phases, w)[0]
            v[idx] = old
            self.assertAlmostEqual(grads[key][idx], (lp - lm) / 2e-6, delta=1e-5, msg=key)

    def test_loss_parts(self) -> None:
        out = np.zeros((2, 13))
        acts = np.array([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]], dtype=float)
        total, parts, _ = loss_and_grad(out, acts, np.array([0, -1]), LossWeights())
        self.assertAlmostEqual(parts["movement"], 0.0)
        self.assertAlmostEqual(parts["interaction"], float(np.log(2.0)), places=6)
        self.assertAlmostEqual(parts["phase"], float(np.log(7.0)), places=6)   # uniform logits
        self.assertGreater(total, 0)

    def test_outputs_are_in_range(self) -> None:
        net = PolicyNetwork.initialise(Architecture(), seed=0)
        x = np.random.default_rng(1).normal(size=(2, 3, 72, 128)).astype(np.float32) * 5
        h = PolicyNetwork.heads(net.forward(x, np.zeros((2, 2), np.float32))[0])
        self.assertTrue(np.all(np.abs(h["continuous"]) <= 1.0))
        self.assertTrue(np.all((h["binary"] >= 0) & (h["binary"] <= 1)))
        self.assertTrue(np.allclose(h["phase"].sum(axis=1), 1.0, atol=1e-5))

    def test_architecture_validation(self) -> None:
        with self.assertRaises(ValueError):
            PolicyNetwork.initialise(Architecture(width=100), seed=0)   # 13 cols: not /4


class TrainingAndExport(unittest.TestCase):
    model: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_model()

    def test_training_artifacts(self) -> None:
        meta = json.loads((self.model / "policy.json").read_text())
        history = json.loads((self.model / "history.json").read_text())
        self.assertEqual(meta["epochs_run"], len(history))
        self.assertTrue(meta["dataset"]["synthetic_only"])
        self.assertIn("test", meta["metrics"])
        self.assertIn("baseline_test", meta["metrics"])
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"] * 1.5)
        net = PolicyNetwork.load(Architecture.from_dict(meta["architecture"]),
                                 self.model / "policy.npz")
        self.assertEqual(net.parameter_count, meta["parameters"])

    def test_numpy_backend_predicts_valid_actions(self) -> None:
        policy = LearnedPolicy(self.model, backend="numpy")
        frame = np.full((360, 640, 3), 120, np.uint8)
        out = policy.predict(frame, None)
        self.assertTrue(out.finite())
        self.assertLessEqual(abs(out.move_forward), 1.0)
        self.assertAlmostEqual(sum(out.phase_probs), 1.0, places=4)
        self.assertGreater(out.latency_ms, 0.0)
        self.assertEqual(policy.backend, "numpy-fp32")

    @unittest.skipUnless(onnx_available(), "onnx/onnxruntime not installed")
    def test_onnx_export_matches_the_source_model(self) -> None:
        report = json.loads((self.model / "export.json").read_text())
        self.assertLess(report["fp32"]["agreement"]["max_abs_diff"], 1e-3)
        self.assertIn("int8", report)
        self.assertGreaterEqual(report["int8"]["agreement"]["interact_decision_agreement"], 0.8)
        self.assertLess(report["int8"]["bytes"], report["fp32"]["bytes"])
        for key in ("numpy_fp32", "onnxruntime_fp32", "onnxruntime_int8"):
            self.assertGreater(report["latency_single_thread"][key]["mean_ms"], 0)

    @unittest.skipUnless(onnx_available(), "onnx/onnxruntime not installed")
    def test_backends_agree(self) -> None:
        frame = cv2.imread(str(next((tiny_dataset()).rglob("*.png"))))
        frame = cv2.resize(frame, (640, 360))
        a = LearnedPolicy(self.model, backend="numpy").predict(frame, None)
        b = LearnedPolicy(self.model, backend="onnxruntime").predict(frame, None)
        q = LearnedPolicy(self.model, backend="onnxruntime", quantized=True).predict(frame, None)
        self.assertEqual(b.phase, a.phase)
        self.assertAlmostEqual(a.move_forward, b.move_forward, delta=1e-4)
        self.assertAlmostEqual(a.look_yaw, b.look_yaw, delta=1e-4)
        self.assertTrue(q.finite())
        self.assertAlmostEqual(a.move_forward, q.move_forward, delta=0.25)

    def test_load_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(PolicyLoadError):
            LearnedPolicy(tmp)
        with self.assertRaises(PolicyLoadError):
            LearnedPolicy(self.model, backend="cuda")

    def test_validation_gate(self) -> None:
        policy = LearnedPolicy(self.model, backend="numpy")
        meta = policy.metadata["metrics"]
        expected = np.mean([meta["validation"][k] for k in meta["baseline_validation"]
                            if k.startswith("mae_")]) < np.mean(
            [meta["baseline_validation"][k] for k in meta["baseline_validation"]
             if k.startswith("mae_")])
        self.assertEqual(policy.validated(), bool(expected))
        policy.metadata = {"metrics": {}}
        self.assertFalse(policy.validated())


if __name__ == "__main__":
    unittest.main()
