"""Phase 7 generalisation benchmark: evaluation categories, held-out seeds, the zero-safe
generalisation gap, paired statistics, and a tiny end-to-end run."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from chest_hunter.analytics import phase7_benchmark as p7
from chest_hunter.config import load_config
from chest_hunter.exceptions import ValidationError
from chest_hunter.simulation3d.generator7 import is_held_out


class EvaluationDesignTest(unittest.TestCase):
    def test_eight_categories_map_to_the_intended_maps(self) -> None:
        self.assertEqual(len(p7.EVAL_CATEGORIES), 8)
        fam, mid = p7.evaluation_map("familiar", 5)
        self.assertEqual((fam, mid), ("indoor", "p7_indoor_train_001"))
        self.assertFalse(is_held_out(mid))
        for cat in p7.EVAL_CATEGORIES:
            if cat == "familiar":
                continue
            with self.subTest(category=cat):
                self.assertTrue(is_held_out(p7.evaluation_map(cat, 5)[1]))
        # paired categories shift the familiar layout of the same episode
        self.assertEqual(p7.evaluation_map("lighting", 6)[1],
                         p7.evaluation_map("familiar", 6)[1] + "_light")
        self.assertIn("lighting", p7.PAIRED_CATEGORIES)
        self.assertNotIn("unseen_layout", p7.PAIRED_CATEGORIES)
        with self.assertRaises(ValidationError):
            p7.evaluation_map("night", 0)

    def test_evaluation_seeds_are_never_training_seeds(self) -> None:
        self.assertEqual(p7.evaluation_seed(42), p7.EVAL_SEED_OFFSET + 42)
        p7.assert_training_seed(1)
        with self.assertRaises(ValidationError):
            p7.assert_training_seed(p7.evaluation_seed(1))
        with self.assertRaises(ValidationError):
            p7.evaluation_seed(p7.EVAL_SEED_OFFSET)

    def test_training_uses_training_maps_only(self) -> None:
        for e in range(40):
            self.assertFalse(is_held_out(p7.training_map(e)[1]))

    def test_planned_jobs_are_paired_and_use_the_evaluation_seed(self) -> None:
        jobs = p7.plan_generalization_jobs(
            ("phase7_active", "phase6_priors"), ("familiar", "lighting"), episodes=3, seed=5,
            settings_path="x", model_dir=None, memory_path=None, max_ticks=100, width=320,
            height=180)
        self.assertEqual(len(jobs), 3 * 2 * 2)
        self.assertEqual({j.seed for j in jobs}, {p7.evaluation_seed(5)})
        by_key = {(j.setup, j.category, j.episode): j.map_id for j in jobs}
        self.assertEqual(by_key[("phase7_active", "lighting", 2)],
                         by_key[("phase6_priors", "lighting", 2)])


class GapTest(unittest.TestCase):
    def test_gap_definition_and_zero_safety(self) -> None:
        self.assertAlmostEqual(p7.generalization_gap(4.0, 3.0) or 0.0, 0.25)
        self.assertAlmostEqual(p7.generalization_gap(2.0, 3.0) or 0.0, -0.5)
        self.assertIsNone(p7.generalization_gap(0.0, 1.0))
        self.assertIsNone(p7.generalization_gap(None, 1.0))

    def test_bootstrap_interval_contains_the_point_estimate(self) -> None:
        fam = [3.0, 4.0, 5.0, 4.0, 3.5, 4.5] * 5
        uns = [2.0, 3.0, 4.0, 3.0, 2.5, 3.5] * 5
        out = p7.bootstrap_gap(fam, uns, paired=True)
        self.assertAlmostEqual(out["G"], (4.0 - 3.0) / 4.0, places=4)
        self.assertLessEqual(out["ci_low"], out["G"])
        self.assertGreaterEqual(out["ci_high"], out["G"])
        self.assertGreater(out["ci_low"], 0.0)            # a consistent drop is significant
        unpaired = p7.bootstrap_gap(fam, uns[:20], paired=False)
        self.assertEqual(unpaired["n_unseen"], 20)
        with self.assertRaises(ValueError):
            p7.bootstrap_gap(fam, uns[:20], paired=True)

    def test_zero_familiar_throughput_is_reported_not_divided(self) -> None:
        out = p7.bootstrap_gap([0.0, 0.0, 0.0], [1.0, 0.0, 2.0], paired=True)
        self.assertIsNone(out["G"])
        self.assertIn("zero", out["reason"])


def _row(setup: str, category: str, episode: int, throughput: float) -> dict[str, object]:
    return {"setup": setup, "category": category, "family": "open", "episode": episode,
            "map_id": "m", p7.PRIMARY_METRIC: throughput, "outcome": "tick_limit",
            "detection_tp": 8, "detection_fp": 2, "detection_fn": 2, "recovery_attempts": 4,
            "recovery_resolved": 3, "interaction_attempts": 5, "chests_opened_reported": 4,
            "chests_reachable": 6, "chests_opened_truth": 4, "exploration_coverage": 0.7}


class SummaryTest(unittest.TestCase):
    def test_summary_pools_counts_and_pairs_episodes(self) -> None:
        rows = [_row("phase7_active", "familiar", e, 4.0) for e in range(5)]
        rows += [_row("phase7_active", "lighting", e, 3.0) for e in range(5)]
        rows += [_row("phase6_priors", "familiar", e, 5.0) for e in range(5)]
        rows += [_row("phase6_priors", "lighting", e, 5.0) for e in range(5)]
        s = p7.summarise_generalization(rows, ("phase7_active", "phase6_priors"),
                                        ("familiar", "lighting"))
        fam = s["per_setup"]["phase7_active"]["familiar"]
        self.assertEqual(fam["detection_precision_pooled"], 0.8)
        self.assertEqual(fam["recovery_success_pooled"], 0.75)
        self.assertEqual(fam["reachable_opened_pooled"], round(4 / 6, 4))
        gap = s["generalization_gap"]["phase7_active"]["lighting"]
        self.assertAlmostEqual(gap["G"], 0.25)
        self.assertTrue(gap["paired"])
        self.assertEqual(s["generalization_gap"]["phase6_priors"]["lighting"]["G"], 0.0)
        diff = s["phase7_active_minus_phase6_priors"]["familiar"][p7.PRIMARY_METRIC]
        self.assertEqual(diff["worse"], 5)
        self.assertAlmostEqual(diff["difference"]["mean"], -1.0)


class EndToEndTest(unittest.TestCase):
    def test_tiny_benchmark_writes_results(self) -> None:
        settings = load_config("config/settings.json")
        with tempfile.TemporaryDirectory() as tmp:
            summary = p7.run_generalization_benchmark(
                settings, setups=("phase7_active_deterministic",),
                categories=("familiar", "lighting"), episodes=1, max_ticks=450, workers=1,
                output_dir=tmp)
            self.assertEqual(summary["errors"], [])
            self.assertEqual(summary["episodes_completed"], 2)
            with (Path(tmp) / "episodes.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({r["category"] for r in rows}, {"familiar", "lighting"})
            self.assertTrue(all(r["exploration_coverage"] for r in rows))
            self.assertTrue((Path(tmp) / "report.md").read_text().startswith("# Phase 7"))
            self.assertIn("NOT Unreal", summary["environment"])

    def test_unknown_configuration_is_refused(self) -> None:
        settings = load_config("config/settings.json")
        with self.assertRaises(ValidationError):
            p7.run_generalization_benchmark(settings, setups=("phase9",), episodes=1)
        with self.assertRaises(ValidationError):
            p7.run_generalization_benchmark(settings, setups=("phase7_active_deterministic",),
                                            categories=("lighting",), episodes=1)


if __name__ == "__main__":
    unittest.main()
