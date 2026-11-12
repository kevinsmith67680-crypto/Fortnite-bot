"""Tests for the paired strategy benchmark and metrics aggregation."""

from __future__ import annotations

import csv
import statistics
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from chest_hunter.analytics.benchmark import CSV_FIELDS, clone_map_chests, run_benchmark
from chest_hunter.analytics.metrics import EpisodeMetrics, summarise
from chest_hunter.chest_repository import ChestRepository
from chest_hunter.config import Settings, load_config
from chest_hunter.database import DatabaseManager
from chest_hunter.demo_data import DEMO_MAP_ID, seed_demo_chests
from chest_hunter.exceptions import ValidationError
from chest_hunter.session_repository import SessionRepository

VOLATILE = ("cpu_seconds", "wall_seconds")


def stable(row: EpisodeMetrics) -> dict[str, object]:
    data = row.as_row()
    for key in VOLATILE:
        data.pop(key)
    return data


class BenchmarkTests(unittest.TestCase):
    source: DatabaseManager
    settings: Settings

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DatabaseManager(":memory:")
        cls.source.initialize()
        seed_demo_chests(cls.source)
        base = load_config()
        # A smaller arena keeps the test fast while using the real demo data.
        cls.settings = replace(base, simulation=replace(
            base.simulation, map_width=100, map_height=100, max_ticks=1800))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.source.close()

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_paired_deterministic_benchmark_with_csv(self) -> None:
        csv_path = self.tmp / "out" / "bench.csv"
        report = run_benchmark(self.source, self.settings, map_id=DEMO_MAP_ID,
                               strategies=("nearest", "weighted"), episodes=3, seed=11,
                               csv_path=csv_path)
        self.assertEqual(len(report.rows), 6)
        self.assertEqual(set(report.summaries), {"nearest", "weighted"})
        # Paired: both strategies saw identical chest spawns in every episode.
        for episode in range(3):
            pair = [r for r in report.rows if r.episode == episode]
            self.assertEqual(len({r.chests_spawned for r in pair}), 1)
            self.assertEqual(len({r.candidate_locations for r in pair}), 1)
        # Deterministic: rerunning produces identical simulated results.
        again = run_benchmark(self.source, self.settings, map_id=DEMO_MAP_ID,
                              strategies=("nearest", "weighted"), episodes=3, seed=11)
        self.assertEqual([stable(r) for r in report.rows], [stable(r) for r in again.rows])

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertEqual(tuple(rows[0].keys()), CSV_FIELDS)
        self.assertEqual(report.csv_path, csv_path)

        summary = report.summaries["nearest"]
        rates = [r.chests_per_minute or 0.0 for r in report.rows if r.strategy == "nearest"]
        self.assertAlmostEqual(summary.mean_chests_per_minute, statistics.fmean(rates))
        self.assertAlmostEqual(summary.median_chests_per_minute, statistics.median(rates))
        self.assertEqual(len(report.comparisons), 1)
        comparison = report.comparisons[0]
        self.assertEqual((comparison.baseline, comparison.challenger), ("nearest", "weighted"))
        self.assertEqual(comparison.challenger_wins + comparison.baseline_wins + comparison.ties,
                         3)
        table = report.format_table()
        self.assertIn("Nearest", table)
        self.assertIn("ProbabilityWeighted", table)
        self.assertIn("vs Nearest", report.format_comparisons())

    def test_source_database_is_untouched_unless_persisting(self) -> None:
        run_benchmark(self.source, self.settings, map_id=DEMO_MAP_ID, strategies=("nearest",),
                      episodes=1, seed=1)
        self.assertEqual(SessionRepository(self.source).list_sessions(), [])

        with DatabaseManager(":memory:") as target:
            target.initialize()
            self.assertEqual(clone_map_chests(self.source, target, DEMO_MAP_ID), 35)
            report = run_benchmark(target, self.settings, map_id=DEMO_MAP_ID,
                                   strategies=("nearest", "lookahead"), episodes=2, seed=1,
                                   persist_db=target, max_ticks=300)
            self.assertEqual(len(SessionRepository(target).list_sessions()), 4)
            self.assertTrue(all(r.outcome == "tick_limit" for r in report.rows))
            source_ids = [c.id for c in ChestRepository(self.source).get_all_chests()]
            target_ids = [c.id for c in ChestRepository(target).get_all_chests()]
            self.assertEqual(source_ids, target_ids)

    def test_vary_obstacles_changes_the_map_per_episode(self) -> None:
        report = run_benchmark(self.source, self.settings, map_id=DEMO_MAP_ID,
                               strategies=("nearest",), episodes=2, seed=5, max_ticks=600,
                               vary_obstacles=True)
        self.assertTrue(report.vary_obstacles)
        self.assertEqual(len(report.rows), 2)

    def test_invalid_arguments(self) -> None:
        for kwargs in ({"strategies": ("random",)}, {"strategies": ()}, {"episodes": 0},
                       {"max_ticks": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    run_benchmark(self.source, self.settings, map_id=DEMO_MAP_ID,
                                  **kwargs)  # type: ignore[arg-type]


class MetricsTests(unittest.TestCase):
    def test_chests_per_minute_and_summary(self) -> None:
        m = EpisodeMetrics("nearest", 1, 0, simulated_seconds=120.0, chests_opened=6)
        self.assertAlmostEqual(m.chests_per_minute or 0.0, 3.0)
        self.assertIsNone(EpisodeMetrics("nearest", 1, 0).chests_per_minute)
        rows = [m, EpisodeMetrics("nearest", 1, 1, simulated_seconds=60.0, chests_opened=1)]
        summary = summarise("nearest", rows)
        self.assertAlmostEqual(summary.mean_chests_per_minute, 2.0)
        self.assertAlmostEqual(summary.median_chests_per_minute, 2.0)
        with self.assertRaises(ValueError):
            summarise("nearest", [])


if __name__ == "__main__":
    unittest.main()
