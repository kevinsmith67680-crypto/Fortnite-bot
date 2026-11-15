"""Phase 7 procedural environments: determinism, validity, held-out separation, variants."""

from __future__ import annotations

import hashlib
import json
import unittest

from chest_hunter.simulation3d import generator7 as g7
from chest_hunter.simulation3d.arena import arena_to_json


def _digest(map_id: str) -> str:
    arena = g7.build_arena(map_id)
    return hashlib.sha256(json.dumps(arena_to_json(arena), sort_keys=True).encode()).hexdigest()


class MapIdTest(unittest.TestCase):
    def test_ids_round_trip(self) -> None:
        mid = g7.map_id("complex", "test", 7, "light")
        self.assertEqual(mid, "p7_complex_test_007_light")
        spec = g7.parse(mid)
        self.assertEqual((spec.category, spec.split, spec.index, spec.variant),
                         ("complex", "test", 7, "light"))
        self.assertEqual(spec.base_id, "p7_complex_test_007")

    def test_invalid_ids_are_refused(self) -> None:
        for bad in ("p7_open_dev_000", "p7_caves_train_000", "p7_open_train_0001",
                    "research_arena_3d"):
            self.assertFalse(g7.is_generated(bad) and g7.parse(bad).split in g7.SPLITS)
        with self.assertRaises(ValueError):
            g7.map_id("open", "train", 1000)
        with self.assertRaises(ValueError):
            g7.map_id("open", "train", 0, "fog")

    def test_train_and_test_seeds_are_disjoint(self) -> None:
        train = {g7.parse(g7.map_id(c, "train", i)).seed for c in g7.CATEGORIES
                 for i in range(1000)}
        test = {g7.parse(g7.map_id(c, "test", i)).seed for c in g7.CATEGORIES
                for i in range(1000)}
        self.assertFalse(train & test)


class HeldOutTest(unittest.TestCase):
    def test_test_maps_and_all_variants_are_held_out(self) -> None:
        self.assertFalse(g7.is_held_out("p7_open_train_003"))
        self.assertTrue(g7.is_held_out("p7_open_test_003"))
        for v in g7.VARIANTS:                     # a shift of a familiar layout is unseen too
            self.assertTrue(g7.is_held_out(f"p7_open_train_003_{v}"))
        self.assertFalse(g7.is_held_out("research_arena_3d"))

    def test_training_writes_refuse_held_out_maps(self) -> None:
        g7.assert_training_map("p7_mixed_train_001")
        for mid in ("p7_mixed_test_001", "p7_mixed_train_001_light", "proc_test_003"):
            with self.assertRaises(ValueError):
                g7.assert_training_map(mid)


class GenerationTest(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        mid = "p7_indoor_train_002"
        first = _digest(mid)
        g7.build.cache_clear()
        self.assertEqual(first, _digest(mid))
        self.assertNotEqual(first, _digest("p7_indoor_train_003"))

    def test_every_category_and_split_validates(self) -> None:
        for category in g7.CATEGORIES:
            for split in g7.SPLITS:
                with self.subTest(category=category, split=split):
                    arena, world, nav = g7.build(g7.map_id(category, split, 1))
                    # valid start on the navigation graph
                    self.assertIsNotNone(nav.project(arena.start))
                    # a large reachable navigation area
                    total = (arena.bounds[2] - arena.bounds[0]) * (arena.bounds[3]
                                                                   - arena.bounds[1])
                    self.assertGreaterEqual(arena.meta["reachable_area_m2"], 0.15 * total)
                    # enough chest spawn points, most of them reachable
                    self.assertGreaterEqual(len(arena.spawns), 8)
                    intended = [s for s in arena.spawns if "unreachable" not in s.tags
                                or s.reachable]
                    self.assertGreaterEqual(sum(s.reachable for s in arena.spawns),
                                            0.8 * len(intended) - 1)
                    self.assertEqual(arena.meta["generator_version"], g7.GENERATOR_VERSION)
                    self.assertEqual(arena.meta["category"], category)

    def test_reachable_flags_match_the_navigation_graph(self) -> None:
        arena, _, nav = g7.build("p7_complex_test_004")
        for s in arena.spawns:
            found = nav.find_path(arena.start, (s.x, s.y, s.z), goal_tolerance=1.2).found
            self.assertEqual(found, s.reachable, s.name)

    def test_unreachable_regions_are_flagged_not_dropped(self) -> None:
        sealed = 0
        for i in range(4):                    # (indoor complexes leave no room for one)
            arena = g7.build_arena(g7.map_id("complex", "train", i))
            unreachable = [s for s in arena.spawns if not s.reachable]
            for s in unreachable:
                self.assertIn("unreachable", s.tags)
            sealed += bool(unreachable)
        self.assertGreater(sealed, 0, "no map with an unreachable chest region")

    def test_complex_maps_have_several_elevations(self) -> None:
        arena = g7.build_arena("p7_complex_train_000")
        levels = {round(s.z, 1) for s in arena.spawns}
        self.assertGreaterEqual(len(levels), 2)
        self.assertGreater(max(levels), 1.0)


class VariantTest(unittest.TestCase):
    def test_lighting_variant_changes_only_the_appearance(self) -> None:
        base = g7.build_arena("p7_mixed_train_002")
        light = g7.build_arena("p7_mixed_train_002_light")
        self.assertEqual(base.boxes, light.boxes)
        self.assertEqual(base.start, light.start)
        lo, hi = g7.TRAIN_LIGHTING
        self.assertTrue(lo <= base.appearance["lighting"] <= hi)
        self.assertFalse(lo <= light.appearance["lighting"] <= hi)

    def test_appearance_variant_leaves_the_training_range(self) -> None:
        look = g7.build_arena("p7_open_train_001_appearance").appearance
        lo, hi = g7.TRAIN_CHEST_HUE
        self.assertFalse(lo <= look["chest_hue_shift"] <= hi)
        lo, hi = g7.TRAIN_SURFACE_VALUE
        self.assertFalse(lo <= look["surface_value"] <= hi)

    def test_obstacle_and_start_variants(self) -> None:
        base = g7.build_arena("p7_open_train_004")
        obstacles = g7.build_arena("p7_open_train_004_obstacles")
        start = g7.build_arena("p7_open_train_004_start")
        self.assertNotEqual(base.boxes, obstacles.boxes)
        self.assertNotEqual((base.start, base.start_yaw_deg),
                            (start.start, start.start_yaw_deg))

    def test_unfamiliar_variant_keeps_the_chest_appearance(self) -> None:
        look = g7.build_arena("p7_complex_test_000_unfamiliar").appearance
        lo, hi = g7.TRAIN_LIGHTING
        self.assertFalse(lo <= look["lighting"] <= hi)
        lo, hi = g7.TRAIN_CHEST_HUE
        self.assertTrue(lo <= look["chest_hue_shift"] <= hi)
        base = g7.build_arena("p7_complex_test_000")
        acc = g7.build_arena("p7_complex_test_000_unfamiliar")
        self.assertNotEqual(base.start, acc.start)
        self.assertGreater(len({round(s.z, 1) for s in acc.spawns}), 2)
        self.assertTrue(g7.is_held_out(acc.map_id))

    def test_elevation_variant_adds_raised_levels(self) -> None:
        base = g7.build_arena("p7_open_train_005")
        high = g7.build_arena("p7_open_train_005_elevation")
        self.assertGreater(max(s.z for s in high.spawns), max(s.z for s in base.spawns))


if __name__ == "__main__":
    unittest.main()
