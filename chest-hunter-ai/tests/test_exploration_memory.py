"""Phase 7 self-improving exploration: strategy bandit, persistent memory (versioning,
training/evaluation separation, no held-out contamination)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from chest_hunter.exploration.active import REGION_TYPES, STRATEGIES
from chest_hunter.exploration.self_improving import (
    MEMORY_FORMAT,
    EpisodeOutcome,
    ExplorationMemory,
    StrategyBandit,
)


def _outcome(strategy: str = "max_information", throughput: float = 2.0,
             category: str = "mixed") -> EpisodeOutcome:
    return EpisodeOutcome(category=category, strategy=strategy, throughput_per_min=throughput,
                          cells_seen_by_region={"enclosed": 40.0, "open": 100.0},
                          discoveries_by_region={"enclosed": 2.0, "open": 1.0},
                          travel_m=300.0, travel_s=100.0, interaction_attempts=5, openings=4,
                          recovery={"target_lost": {"attempts": 3, "resolved": 2}}, scans=6,
                          cells_from_scans=30.0)


class BanditTest(unittest.TestCase):
    def test_learns_the_better_arm_per_context(self) -> None:
        bandit = StrategyBandit()
        rng = random.Random(0)
        means = {s: 1.0 for s in STRATEGIES} | {"cluster": 3.0}
        for _ in range(300):
            arm = bandit.select("open", rng)
            bandit.update("open", arm, rng.gauss(means[arm], 0.5))
        self.assertEqual(bandit.best("open"), "cluster")
        counts = {a: bandit.stats["open"][a][0] for a in bandit.stats["open"]}
        self.assertGreater(counts["cluster"], max(v for a, v in counts.items()
                                                  if a != "cluster"))
        # another context is unaffected
        self.assertEqual(bandit.best("indoor"), STRATEGIES[0])

    def test_round_trip_and_unknown_arm(self) -> None:
        b = StrategyBandit()
        b.update("mixed", "cluster", 2.5)
        b2 = StrategyBandit.from_dict(json.loads(json.dumps(b.to_dict())))
        self.assertEqual(b2.posterior("mixed", "cluster"), b.posterior("mixed", "cluster"))
        with self.assertRaises(ValueError):
            b.update("mixed", "random_walk", 1.0)

    def test_exploitation_is_deterministic(self) -> None:
        b = StrategyBandit()
        b.update("complex", "risk_adjusted", 4.0)
        self.assertEqual({b.best("complex") for _ in range(20)}, {"risk_adjusted"})


class MemoryTest(unittest.TestCase):
    def test_update_accumulates_category_statistics(self) -> None:
        mem = ExplorationMemory(agent_version="7.0.0", generator_version="7.0")
        mem.update(_outcome(), map_id="p7_mixed_train_003", training=True)
        mem.update(_outcome(), map_id="p7_mixed_train_004", training=True)
        rates = mem.discovery_rates("mixed")
        self.assertEqual(set(rates), set(REGION_TYPES))
        self.assertGreater(rates["enclosed"], rates["open"])
        self.assertAlmostEqual(mem.travel_speed("mixed"), 3.0)
        self.assertAlmostEqual(mem.interaction_success("mixed") or 0.0, 0.8)
        self.assertAlmostEqual(mem.recovery_success("mixed")["target_lost"], 2 / 3)
        self.assertAlmostEqual(mem.scan_gain("mixed") or 0.0, 5.0)
        self.assertEqual(mem.categories["mixed"]["episodes"], 2)
        self.assertIsNone(mem.interaction_success("open"))

    def test_evaluation_mode_and_held_out_maps_are_refused(self) -> None:
        mem = ExplorationMemory()
        with self.assertRaises(ValueError):
            mem.update(_outcome(), map_id="p7_mixed_train_003", training=False)
        for held_out in ("p7_mixed_test_003", "p7_mixed_train_003_light", "proc_test_001"):
            with self.assertRaises(ValueError):
                mem.update(_outcome(), map_id=held_out, training=True)
        self.assertEqual(mem.updates, 0)

    def test_memory_holds_no_map_specific_answers(self) -> None:
        mem = ExplorationMemory()
        mem.update(_outcome(), map_id="p7_open_train_001", training=True)
        text = json.dumps(mem.to_dict())
        self.assertNotIn("p7_open_train_001", text)
        self.assertNotIn("Spawn", text)

    def test_persistence_versioning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            mem = ExplorationMemory(agent_version="7.0.0", generator_version="7.0")
            mem.update(_outcome(), map_id="p7_mixed_train_003", training=True)
            mem.save(path)
            same = ExplorationMemory.load(path, agent_version="7.0.0", generator_version="7.0")
            self.assertEqual(same.updates, 1)
            self.assertEqual(same.bandit.best("mixed"), "max_information")
            # another agent or generator version starts empty (never mixed)
            other = ExplorationMemory.load(path, agent_version="8.0.0", generator_version="7.0")
            self.assertEqual(other.updates, 0)
            self.assertEqual(other.categories, {})
            # a missing file is an empty memory; a foreign file is refused
            self.assertEqual(ExplorationMemory.load(Path(tmp) / "none.json").updates, 0)
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"format": MEMORY_FORMAT, "version": 99}))
            with self.assertRaises(ValueError):
                ExplorationMemory.load(bad)


if __name__ == "__main__":
    unittest.main()
