"""Tests for the oracle-vs-visual benchmark and its exports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from chest_hunter.analytics.visual_benchmark import (
    NAVIGATION_FIELDS,
    VISION_FIELDS,
    format_summary,
    run_visual_benchmark,
)
from chest_hunter.config import load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import ValidationError
from chest_hunter.scenarios import VISUAL_ARENA_ID, seed_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATED_KEYS = ("outcome", "simulated_seconds", "ticks", "chests_spawned", "chests_available",
                  "chests_opened", "recoveries", "locations_recorded")


class VisualBenchmarkTests(unittest.TestCase):
    source: DatabaseManager

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DatabaseManager(":memory:")
        cls.source.initialize()
        seed_scenario(cls.source)
        cls.settings = load_config(PROJECT_ROOT / "config" / "settings.json")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "results"
        cls.summary = run_visual_benchmark(cls.source, cls.settings, map_id=VISUAL_ARENA_ID,
                                           episodes=2, seed=7, max_ticks=2400,
                                           output_dir=cls.out, detector_frames=8)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.source.close()
        cls.tmp.cleanup()

    def test_exports(self) -> None:
        for name in ("navigation_benchmark.csv", "vision_benchmark.csv", "summary.json"):
            self.assertTrue((self.out / name).is_file(), name)
        with (self.out / "navigation_benchmark.csv").open(encoding="utf-8") as handle:
            nav = list(csv.DictReader(handle))
        with (self.out / "vision_benchmark.csv").open(encoding="utf-8") as handle:
            vision = list(csv.DictReader(handle))
        self.assertEqual(len(nav), 6)           # 2 episodes x 3 modes
        self.assertEqual(len(vision), 4)        # 2 episodes x 2 visual modes
        self.assertEqual(tuple(nav[0]), NAVIGATION_FIELDS)
        self.assertEqual(tuple(vision[0]), VISION_FIELDS)
        data = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["episodes_completed"], 2)
        self.assertEqual(set(data["per_mode"]), {"oracle", "chest_vision", "visual"})
        self.assertIn("units", data)
        self.assertIn("simulated", data["time_basis"])
        self.assertEqual(set(data["detector_comparison"]), {"color", "template", "hybrid"})
        self.assertGreater(data["run"]["peak_rss_mb"], 0)
        self.assertIn("vision", data["per_mode"]["visual"])
        self.assertNotIn("vision", data["per_mode"]["oracle"])

    def test_paired_episodes_share_the_environment(self) -> None:
        rows = self.summary["_navigation_rows"]
        for episode in (0, 1):
            pair = [r for r in rows if r["episode"] == episode]
            self.assertEqual(len({r["chests_spawned"] for r in pair}), 1)
            self.assertEqual(len({r["chests_available"] for r in pair}), 1)

    def test_metrics_are_sane(self) -> None:
        for row in self.summary["_vision_rows"]:
            self.assertGreaterEqual(row["precision"], 0.9)
            self.assertEqual(row["localisation_success_rate"], 1.0)
            self.assertGreater(row["inference_ms_mean"], 0)
        for row in self.summary["_navigation_rows"]:
            self.assertLessEqual(row["chests_opened"], row["chests_available"])
            # The agent never claims an opening the simulator did not register.
            self.assertLessEqual(row["chests_opened"], row["chests_opened_truth"])
        for entry in self.summary["per_mode"].values():
            self.assertEqual(entry["verification"]["over_reported_total"], 0)
            self.assertEqual(entry["verification"]["opened_recorded_total"],
                             entry["total_opened"])
        text = format_summary(self.summary)
        self.assertIn("vs oracle", text)
        self.assertIn("visual verification:", text)

    def test_benchmark_is_deterministic(self) -> None:
        again = run_visual_benchmark(self.source, self.settings, map_id=VISUAL_ARENA_ID,
                                     modes=("oracle", "visual"), episodes=1, seed=7,
                                     max_ticks=2400, detector_frames=0)
        first = {(r["episode"], r["mode"]): r for r in self.summary["_navigation_rows"]}
        for row in again["_navigation_rows"]:
            for key in SIMULATED_KEYS:
                self.assertEqual(row[key], first[(row["episode"], row["mode"])][key], key)

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValidationError):
            run_visual_benchmark(self.source, self.settings, map_id=VISUAL_ARENA_ID,
                                 modes=("telepathy",), episodes=1)
        with self.assertRaises(ValidationError):
            run_visual_benchmark(self.source, self.settings, map_id=VISUAL_ARENA_ID, episodes=0)


if __name__ == "__main__":
    unittest.main()
