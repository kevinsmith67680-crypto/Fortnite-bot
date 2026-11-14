"""Procedural maps and execution on unseen layouts."""

from __future__ import annotations

import unittest

from chest_hunter.agent.controller_3d import Agent3DConfig
from chest_hunter.agent.runner_3d import Runner3D
from chest_hunter.database import DatabaseManager
from chest_hunter.learning.learned_policy import LearnedPolicy
from chest_hunter.scenarios import procedural_chests, seed_procedural_map
from chest_hunter.simulation3d.environment3d import Environment3D, EnvSettings
from chest_hunter.simulation3d.procedural import (
    DATABASE_PRIOR,
    SEED_BASE,
    _generate,
    build_procedural,
    is_procedural,
    layout_hash,
    map_id,
    parse_map_id,
)
from tests.phase5_helpers import settings, tiny_model

SMALL = EnvSettings(width=320, height=180)


class ProceduralMaps(unittest.TestCase):
    def test_ids_and_disjoint_splits(self) -> None:
        self.assertEqual(map_id("train", 7), "proc_train_007")
        self.assertEqual(parse_map_id("proc_test_012"), ("test", 12))
        self.assertTrue(is_procedural("proc_train_000"))
        self.assertFalse(is_procedural("research_arena_3d"))
        for bad in ("proc_val_001", "proc_train_1", "proc_train_0001"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_map_id(bad)
        with self.assertRaises(ValueError):
            map_id("validation", 0)
        train = set(range(SEED_BASE["train"], SEED_BASE["train"] + 1000))
        test = set(range(SEED_BASE["test"], SEED_BASE["test"] + 1000))
        self.assertFalse(train & test)

    def test_generation_is_deterministic(self) -> None:
        import random
        a = _generate("proc_test_003", random.Random("proc3d:test:9003"))
        b = _generate("proc_test_003", random.Random("proc3d:test:9003"))
        self.assertEqual(layout_hash(a), layout_hash(b))

    def test_layouts_vary_and_are_valid(self) -> None:
        hashes, starts = set(), set()
        for mid in ("proc_train_000", "proc_train_001", "proc_test_000", "proc_test_001"):
            arena, _world, nav = build_procedural(mid)
            hashes.add(layout_hash(arena))
            starts.add((arena.start, arena.start_yaw_deg))
            self.assertGreaterEqual(len(arena.spawns), 15)
            self.assertGreaterEqual(sum(s.reachable for s in arena.spawns),
                                    0.85 * len(arena.spawns))
            self.assertIsNotNone(nav.project(arena.start))
            self.assertTrue(any(s.z > 0.5 for s in arena.spawns) or len(arena.regions) > 0)
            self.assertTrue(0.8 <= arena.appearance["lighting"] <= 1.2)
            self.assertTrue(all(0.15 <= s.probability <= 0.95 for s in arena.spawns))
            self.assertTrue(any(b.kind == "platform" for b in arena.boxes))
        self.assertEqual(len(hashes), 4)
        self.assertEqual(len(starts), 4)                       # different start poses

    def test_database_gets_an_uninformative_prior(self) -> None:
        db = DatabaseManager(":memory:")
        db.initialize()
        self.addCleanup(db.close)
        result = seed_procedural_map(db, "proc_test_000")
        self.assertGreater(result.inserted, 0)
        self.assertEqual(seed_procedural_map(db, "proc_test_000").inserted, 0)
        self.assertEqual({c.spawn_probability for c in procedural_chests("proc_test_000")},
                         {DATABASE_PRIOR})

    def test_chest_presence_is_randomised_per_seed(self) -> None:
        a = [c.present for c in Environment3D("proc_test_001", seed=1, settings=SMALL).chests]
        b = [c.present for c in Environment3D("proc_test_001", seed=2, settings=SMALL).chests]
        self.assertNotEqual(a, b)

    def test_appearance_changes_the_rendering(self) -> None:
        env = Environment3D("proc_test_001", seed=1, settings=SMALL)
        self.assertNotEqual(env.base_lighting, 1.0)
        frame, _ = env.observe()
        self.assertEqual(frame.shape, (180, 320, 3))


def _db() -> DatabaseManager:
    db = DatabaseManager(":memory:")
    db.initialize()
    return db


class UnseenMapEpisodes(unittest.TestCase):
    def _run(self, map_name: str, **kw: object) -> tuple[Runner3D, object]:
        db = _db()
        self.addCleanup(db.close)
        runner = Runner3D.from_settings(db, settings(), mode="hybrid", seed=21, map_id=map_name,
                                        backend="reference", max_ticks=1200, evaluate=True,
                                        width=320, height=180, **kw)
        return runner, runner.run()

    def test_deterministic_agent_on_an_unseen_map(self) -> None:
        runner, result = self._run("proc_test_002")
        self.assertIn(result.outcome, ("completed", "tick_limit"))
        self.assertLessEqual(result.chests_opened, result.truth["opened_truth"])
        status = runner.db.query_all("SELECT status FROM simulation_sessions WHERE id = ?",
                                     (result.session_id,))[0]["status"]
        self.assertEqual(status, "completed")

    def test_learned_and_hybrid_policies_complete_unseen_episodes(self) -> None:
        policy = LearnedPolicy(tiny_model(), backend="numpy")
        for mode in ("learned", "hybrid"):
            with self.subTest(mode=mode):
                runner, result = self._run("proc_test_003", policy=mode, learned_policy=policy)
                self.assertIn(result.outcome, ("completed", "tick_limit"))
                stats = runner.selector.stats.to_dict()
                self.assertGreater(stats["decisions"], 0)
                if mode == "learned":
                    self.assertGreater(stats["inferences"], 0)
                self.assertLessEqual(result.chests_opened, result.truth["opened_truth"])

    def test_recovery_stays_bounded(self) -> None:
        runner, result = self._run("proc_test_000",
                                   config=Agent3DConfig(smart_recovery=True))
        self.assertNotEqual(result.outcome, "failed")
        agent = runner.controller
        self.assertLessEqual(agent.metrics.recoveries, agent.recovery_budget)


if __name__ == "__main__":
    unittest.main()
