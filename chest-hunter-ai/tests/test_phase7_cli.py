"""Phase 7 command-line interface: documented commands parse, earlier commands keep their
behaviour, safety confirmations are required."""

from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from chest_hunter.cli import build_parser, main
from chest_hunter.cli_phase7 import resolve_map


class ParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_parser()

    def test_documented_phase7_commands_parse(self) -> None:
        a = self.parser.parse_args(["simulate-3d", "--agent", "hybrid", "--exploration",
                                    "active", "--map", "procedural", "--seed", "42"])
        self.assertEqual((a.exploration, a.map, a.seed), ("active", "procedural", 42))
        a = self.parser.parse_args(["benchmark-generalization", "--episodes", "100",
                                    "--held-out-maps", "--output", "benchmark_results/phase7"])
        self.assertTrue(a.held_out_maps)
        self.assertEqual(a.output_dir, "benchmark_results/phase7")
        a = self.parser.parse_args(["benchmark-robustness", "--episodes", "100",
                                    "--failure-injection", "--output",
                                    "benchmark_results/robustness"])
        self.assertTrue(a.failure_injection)
        a = self.parser.parse_args(["profile-agent", "--episodes", "20", "--report",
                                    "benchmark_results/resource_profile.json"])
        self.assertEqual(a.report, "benchmark_results/resource_profile.json")
        for cmd in (["train-exploration"], ["benchmark-exploration"], ["check-runtime"],
                    ["acceptance-phase7", "--episodes", "2"]):
            self.parser.parse_args(cmd)

    def test_phase5_generalisation_defaults_are_unchanged(self) -> None:
        a = self.parser.parse_args(["benchmark-generalization"])
        self.assertFalse(a.held_out_maps)
        self.assertIsNone(a.episodes)            # resolved to 6 (Phase 5) at run time
        self.assertEqual(a.familiar, 4)

    def test_unknown_fault_kind_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parser.parse_args(["simulate-3d", "--inject-fault", "meteor"])

    def test_procedural_map_choice_is_deterministic_and_held_out(self) -> None:
        ns = argparse.Namespace(map="procedural", map_category=None, map_split="test")
        self.assertEqual(resolve_map(ns, 42), "p7_mixed_test_042")
        self.assertEqual(resolve_map(ns, 42), resolve_map(ns, 42))
        ns2 = argparse.Namespace(map="procedural", map_category="open", map_split="train")
        self.assertEqual(resolve_map(ns2, 1003), "p7_open_train_003")
        self.assertEqual(resolve_map(argparse.Namespace(map="research_arena_3d"), 1),
                         "research_arena_3d")


class SafetyConfirmationTest(unittest.TestCase):
    def test_robustness_benchmark_needs_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = main(["benchmark-robustness", "--episodes", "1", "--output",
                             str(Path(tmp) / "r")])
        self.assertEqual(code, 2)
        self.assertIn("--failure-injection", err.getvalue())

    def test_active_exploration_cannot_use_a_spawn_point_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "db.sqlite")
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                main(["--db", db, "init-db"])
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = main(["--db", db, "simulate-3d", "--exploration", "active", "--map",
                             "procedural", "--planner", "greedy", "--max-ticks", "30"])
        self.assertNotEqual(code, 0)
        self.assertIn("spawn-point planner", err.getvalue())


if __name__ == "__main__":
    unittest.main()
