"""Tests for ChestRepository: CRUD, validation, filtering, deletion rules, statistics."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chest_hunter.chest_repository import ChestRepository
from chest_hunter.database import DatabaseManager
from chest_hunter.demo_data import DEMO_CHESTS, DEMO_MAP_ID, DEMO_REGIONS, seed_demo_chests
from chest_hunter.exceptions import (
    ChestInUseError,
    DuplicateChestError,
    RecordNotFoundError,
    ValidationError,
)
from chest_hunter.models import ChestLocation, parse_timestamp
from chest_hunter.session_repository import SessionRepository

MAP = "research_arena_01"
OTHER_MAP = "research_arena_02"


class ManualClock:
    """Deterministic clock for timestamp assertions."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class ChestRepositoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = DatabaseManager(Path(tmp.name) / "chests.db")
        self.addCleanup(self.db.close)
        self.db.initialize()
        self.clock = ManualClock()
        self.repo = ChestRepository(self.db, clock=self.clock)
        self.sessions = SessionRepository(self.db, clock=self.clock)

    def add(self, name: str = "Warehouse Chest", x: float = 120.0, y: float = 340.0,
            map_id: str = MAP, **kwargs: object) -> ChestLocation:
        return self.repo.add_chest(map_id=map_id, location_name=name, x=x, y=y, **kwargs)  # type: ignore[arg-type]


class CreateAndReadTests(ChestRepositoryTestCase):
    def test_add_chest_returns_typed_model_with_id_and_utc_timestamps(self) -> None:
        chest = self.add(z=2.5, spawn_probability=0.85, estimated_open_seconds=1.5)

        self.assertIsInstance(chest, ChestLocation)
        self.assertIsNotNone(chest.id)
        self.assertEqual(chest.map_id, MAP)
        self.assertEqual(chest.location_name, "Warehouse Chest")
        self.assertEqual(chest.position, (120.0, 340.0, 2.5))
        self.assertEqual(chest.spawn_probability, 0.85)
        self.assertEqual(chest.estimated_open_seconds, 1.5)
        self.assertTrue(chest.enabled)
        assert chest.created_at is not None
        self.assertEqual(chest.created_at, chest.updated_at)
        self.assertEqual(parse_timestamp(chest.created_at), self.clock.now)
        self.assertTrue(chest.created_at.endswith("+00:00"))

    def test_get_chest_round_trips_all_fields(self) -> None:
        created = self.add(z=-3.0, spawn_probability=0.6, estimated_open_seconds=0.0, enabled=False)
        assert created.id is not None
        fetched = self.repo.get_chest(created.id)
        self.assertEqual(fetched, created)
        assert fetched is not None
        self.assertIsInstance(fetched.enabled, bool)
        self.assertIsInstance(fetched.x, float)

    def test_integer_coordinates_are_stored_as_floats(self) -> None:
        chest = self.repo.add_chest(MAP, "Int Coords", 10, 20, 0)
        assert chest.id is not None
        fetched = self.repo.get_chest(chest.id)
        assert fetched is not None
        self.assertEqual(fetched.position, (10.0, 20.0, 0.0))
        self.assertIsInstance(fetched.x, float)

    def test_missing_chest_returns_none(self) -> None:
        self.assertIsNone(self.repo.get_chest(999))
        with self.assertRaises(RecordNotFoundError):
            self.repo.require_chest(999)

    def test_names_and_map_ids_are_trimmed(self) -> None:
        chest = self.add(name="  Padded Name  ", map_id="  arena_x  ")
        self.assertEqual(chest.location_name, "Padded Name")
        self.assertEqual(chest.map_id, "arena_x")

    def test_duplicate_position_on_same_map_is_rejected(self) -> None:
        self.add(x=1.0, y=1.0)
        with self.assertRaises(DuplicateChestError):
            self.add(name="Another", x=1.0, y=1.0)
        # Same coordinates on a different map, or a different height, are fine.
        self.add(x=1.0, y=1.0, map_id=OTHER_MAP)
        self.add(x=1.0, y=1.0, z=5.0)

    def test_find_chest_at(self) -> None:
        chest = self.add(x=7.0, y=8.0, z=9.0)
        self.assertEqual(self.repo.find_chest_at(MAP, 7.0, 8.0, 9.0), chest)
        self.assertIsNone(self.repo.find_chest_at(MAP, 7.0, 8.0, 0.0))
        self.assertIsNone(self.repo.find_chest_at(OTHER_MAP, 7.0, 8.0, 9.0))


class ValidationTests(ChestRepositoryTestCase):
    def assert_rejected(self, **kwargs: object) -> None:
        params: dict[str, object] = {"map_id": MAP, "location_name": "Spot", "x": 1.0, "y": 2.0}
        params.update(kwargs)
        with self.assertRaises(ValidationError):
            self.repo.add_chest(**params)  # type: ignore[arg-type]
        self.assertEqual(self.repo.get_all_chests(), [], "nothing may be written on failure")

    def test_rejects_invalid_probability(self) -> None:
        for value in (-0.01, 1.01, 2, float("nan"), float("inf"), True, "0.5", None):
            with self.subTest(value=value):
                self.assert_rejected(spawn_probability=value)

    def test_accepts_probability_bounds(self) -> None:
        self.assertEqual(self.add(x=1.0, spawn_probability=0.0).spawn_probability, 0.0)
        self.assertEqual(self.add(x=2.0, spawn_probability=1.0).spawn_probability, 1.0)

    def test_rejects_invalid_coordinates(self) -> None:
        for field in ("x", "y", "z"):
            for value in (float("nan"), float("inf"), float("-inf"), "12", None, False):
                with self.subTest(field=field, value=value):
                    self.assert_rejected(**{field: value})

    def test_rejects_negative_or_non_finite_open_time(self) -> None:
        for value in (-0.1, -5, float("inf"), float("nan")):
            with self.subTest(value=value):
                self.assert_rejected(estimated_open_seconds=value)

    def test_rejects_invalid_map_ids_and_names(self) -> None:
        for map_id in ("", "   ", "has space", "-leading-dash", "x" * 65, 42):
            with self.subTest(map_id=map_id):
                self.assert_rejected(map_id=map_id)
        for name in ("", "   ", "n" * 201, None):
            with self.subTest(name=name):
                self.assert_rejected(location_name=name)

    def test_rejects_non_boolean_enabled(self) -> None:
        self.assert_rejected(enabled=1)

    def test_model_rejects_invalid_values_directly(self) -> None:
        with self.assertRaises(ValidationError):
            ChestLocation(None, MAP, "Spot", 0.0, 0.0, 0.0, 1.5, 2.0, True)

    def test_invalid_ids_are_rejected(self) -> None:
        for bad in (0, -1, True, "1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    self.repo.get_chest(bad)  # type: ignore[arg-type]


class UpdateTests(ChestRepositoryTestCase):
    def test_update_changes_fields_and_updated_at_only(self) -> None:
        chest = self.add(spawn_probability=0.5)
        assert chest.id is not None
        self.clock.advance(60)

        updated = self.repo.update_chest(
            chest.id, location_name="Renamed", x=5.0, spawn_probability=0.9, enabled=False
        )

        self.assertEqual(updated.location_name, "Renamed")
        self.assertEqual(updated.x, 5.0)
        self.assertEqual(updated.y, chest.y)
        self.assertEqual(updated.spawn_probability, 0.9)
        self.assertFalse(updated.enabled)
        self.assertEqual(updated.created_at, chest.created_at)
        assert updated.updated_at is not None
        self.assertEqual(parse_timestamp(updated.updated_at), self.clock.now)
        self.assertEqual(self.repo.get_chest(chest.id), updated)

    def test_update_missing_chest_raises_not_found(self) -> None:
        with self.assertRaises(RecordNotFoundError):
            self.repo.update_chest(404, location_name="Ghost")

    def test_update_without_fields_is_rejected(self) -> None:
        chest = self.add()
        assert chest.id is not None
        with self.assertRaises(ValidationError):
            self.repo.update_chest(chest.id)

    def test_invalid_update_leaves_record_unchanged(self) -> None:
        chest = self.add()
        assert chest.id is not None
        for kwargs in ({"spawn_probability": 1.2}, {"x": float("nan")},
                       {"estimated_open_seconds": -1.0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    self.repo.update_chest(chest.id, **kwargs)  # type: ignore[arg-type]
        self.assertEqual(self.repo.get_chest(chest.id), chest)

    def test_update_onto_existing_position_is_rejected(self) -> None:
        self.add(name="A", x=1.0, y=1.0)
        other = self.add(name="B", x=2.0, y=2.0)
        assert other.id is not None
        with self.assertRaises(DuplicateChestError):
            self.repo.update_chest(other.id, x=1.0, y=1.0)

    def test_cannot_move_observed_chest_to_another_map(self) -> None:
        chest = self.add()
        assert chest.id is not None
        session = self.sessions.start_session(MAP)
        assert session.id is not None
        self.sessions.record_observation(session.id, chest.id, chest_present=False)

        with self.assertRaises(ChestInUseError):
            self.repo.update_chest(chest.id, map_id=OTHER_MAP)
        # Other edits to an observed chest remain allowed.
        updated = self.repo.update_chest(chest.id, spawn_probability=0.3)
        self.assertEqual(updated.spawn_probability, 0.3)


class DeleteTests(ChestRepositoryTestCase):
    def test_delete_unreferenced_chest(self) -> None:
        chest = self.add()
        assert chest.id is not None
        self.assertTrue(self.repo.delete_chest(chest.id))
        self.assertIsNone(self.repo.get_chest(chest.id))

    def test_delete_missing_chest_returns_false(self) -> None:
        self.assertFalse(self.repo.delete_chest(12345))

    def test_referenced_chest_cannot_be_deleted_and_can_be_disabled(self) -> None:
        chest = self.add()
        assert chest.id is not None
        session = self.sessions.start_session(MAP)
        assert session.id is not None
        self.sessions.record_observation(session.id, chest.id, chest_present=True,
                                         chest_opened=True, interaction_seconds=1.0)
        self.assertEqual(self.repo.count_observations(chest.id), 1)

        with self.assertRaisesRegex(ChestInUseError, "Disable it instead"):
            self.repo.delete_chest(chest.id)
        self.assertIsNotNone(self.repo.get_chest(chest.id))

        disabled = self.repo.update_chest(chest.id, enabled=False)
        self.assertFalse(disabled.enabled)


class QueryTests(ChestRepositoryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.a1 = self.add(name="bravo", x=1.0, spawn_probability=0.5)
        self.b1 = self.add(name="Alpha", x=2.0, spawn_probability=0.9, map_id=OTHER_MAP)
        self.a2 = self.add(name="charlie", x=3.0, spawn_probability=0.7, enabled=False)
        self.a3 = self.add(name="alpha", x=4.0, spawn_probability=0.9)

    def ids(self, chests: list[ChestLocation]) -> list[int | None]:
        return [chest.id for chest in chests]

    def test_get_chests_by_map(self) -> None:
        self.assertEqual(self.ids(self.repo.get_chests_by_map(MAP)),
                         [self.a1.id, self.a2.id, self.a3.id])
        self.assertEqual(self.ids(self.repo.get_chests_by_map(OTHER_MAP)), [self.b1.id])
        self.assertEqual(self.repo.get_chests_by_map("empty_map"), [])

    def test_filter_enabled_and_disabled(self) -> None:
        self.assertEqual(self.ids(self.repo.get_chests_by_map(MAP, enabled=True)),
                         [self.a1.id, self.a3.id])
        self.assertEqual(self.ids(self.repo.get_all_chests(enabled=False)), [self.a2.id])
        self.assertEqual(len(self.repo.get_all_chests()), 4)

    def test_deterministic_sort_orders(self) -> None:
        by_name = self.ids(self.repo.get_all_chests(map_id=MAP, order_by="name"))
        self.assertEqual(by_name, [self.a3.id, self.a1.id, self.a2.id])
        by_probability = self.ids(self.repo.get_all_chests(order_by="probability"))
        # Ties (0.9) are broken by ID, so the order is fully deterministic.
        self.assertEqual(by_probability, [self.b1.id, self.a3.id, self.a2.id, self.a1.id])
        self.assertEqual(self.repo.get_all_chests(order_by="map"),
                         self.repo.get_all_chests(order_by="map"))

    def test_invalid_sort_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self.repo.get_all_chests(order_by="x; DROP TABLE chest_locations")

    def test_list_map_ids(self) -> None:
        self.assertEqual(self.repo.list_map_ids(), [MAP, OTHER_MAP])


class UnvisitedChestTests(ChestRepositoryTestCase):
    def test_unvisited_is_derived_per_session(self) -> None:
        c1 = self.add(name="one", x=1.0)
        c2 = self.add(name="two", x=2.0)
        c3 = self.add(name="three", x=3.0)
        self.add(name="off", x=4.0, enabled=False)
        self.add(name="elsewhere", x=5.0, map_id=OTHER_MAP)
        assert c1.id and c2.id and c3.id

        first = self.sessions.start_session(MAP)
        assert first.id is not None
        self.assertEqual([c.id for c in self.repo.get_unvisited_chests(first.id)],
                         [c1.id, c2.id, c3.id])

        self.sessions.record_observation(first.id, c1.id, chest_present=True, chest_opened=True)
        self.sessions.record_observation(first.id, c2.id, chest_present=False)
        self.assertEqual([c.id for c in self.repo.get_unvisited_chests(first.id)], [c3.id])

        # A new session starts with every enabled location unvisited again.
        self.sessions.complete_session(first.id)
        second = self.sessions.start_session(MAP)
        assert second.id is not None
        self.assertEqual([c.id for c in self.repo.get_unvisited_chests(second.id)],
                         [c1.id, c2.id, c3.id])

    def test_unknown_session_raises_not_found(self) -> None:
        with self.assertRaises(RecordNotFoundError):
            self.repo.get_unvisited_chests(99)


class StatisticsTests(ChestRepositoryTestCase):
    def test_empty_map_statistics(self) -> None:
        stats = self.repo.get_chest_statistics("nothing_here")
        self.assertEqual(stats.total_locations, 0)
        self.assertEqual(stats.enabled_locations, 0)
        self.assertIsNone(stats.average_spawn_probability)
        self.assertEqual(stats.total_visits, 0)
        self.assertEqual(stats.successful_openings, 0)
        self.assertIsNone(stats.observed_spawn_frequency)
        self.assertFalse(stats.spawn_frequency_available)

    def test_statistics_with_observations(self) -> None:
        chests = [self.add(name=f"c{i}", x=float(i), spawn_probability=p)
                  for i, p in enumerate((0.6, 0.8, 1.0))]
        self.add(name="disabled", x=99.0, spawn_probability=0.0, enabled=False)
        self.add(name="other map", x=1.0, map_id=OTHER_MAP)
        ids = [c.id for c in chests]
        assert all(ids)

        s1 = self.sessions.start_session(MAP)
        assert s1.id is not None
        self.sessions.record_observation(s1.id, ids[0], chest_present=True, chest_opened=True)  # type: ignore[arg-type]
        self.sessions.record_observation(s1.id, ids[1], chest_present=True)  # type: ignore[arg-type]
        self.sessions.record_observation(s1.id, ids[2], chest_present=False)  # type: ignore[arg-type]
        # Revisit of an already-opened chest: absent now, but must not count as
        # a failed spawn in the observed spawn frequency.
        self.sessions.record_observation(s1.id, ids[0], chest_present=False)  # type: ignore[arg-type]

        stats = self.repo.get_chest_statistics(MAP, min_spawn_samples=3)
        self.assertEqual(stats.total_locations, 4)
        self.assertEqual(stats.enabled_locations, 3)
        assert stats.average_spawn_probability is not None
        self.assertAlmostEqual(stats.average_spawn_probability, 0.8)  # enabled only
        self.assertEqual(stats.total_visits, 4)
        self.assertEqual(stats.chests_found, 2)
        self.assertEqual(stats.successful_openings, 1)
        self.assertEqual(stats.spawn_samples, 3)
        assert stats.observed_spawn_frequency is not None
        self.assertAlmostEqual(stats.observed_spawn_frequency, 2 / 3)

    def test_spawn_frequency_unavailable_with_insufficient_data(self) -> None:
        chest = self.add()
        assert chest.id is not None
        session = self.sessions.start_session(MAP)
        assert session.id is not None
        self.sessions.record_observation(session.id, chest.id, chest_present=True)

        stats = self.repo.get_chest_statistics(MAP)  # default threshold: 5 samples
        self.assertEqual(stats.spawn_samples, 1)
        self.assertIsNone(stats.observed_spawn_frequency)
        self.assertEqual(self.repo.get_chest_statistics(MAP, min_spawn_samples=1)
                         .observed_spawn_frequency, 1.0)

    def test_min_spawn_samples_must_be_positive(self) -> None:
        with self.assertRaises(ValidationError):
            self.repo.get_chest_statistics(MAP, min_spawn_samples=0)


class DemoDataTests(ChestRepositoryTestCase):
    def test_demo_dataset_shape(self) -> None:
        self.assertGreaterEqual(len(DEMO_CHESTS), 30)
        self.assertEqual(len(DEMO_REGIONS), 5)
        positions = {(c.x, c.y, c.z) for c in DEMO_CHESTS}
        self.assertEqual(len(positions), len(DEMO_CHESTS), "positions must be unique")
        for chest in DEMO_CHESTS:
            self.assertTrue(0.5 <= chest.spawn_probability <= 1.0)
            self.assertTrue(0.0 <= chest.x <= 1000.0 and 0.0 <= chest.y <= 1000.0)

    def test_seeding_is_idempotent(self) -> None:
        first = seed_demo_chests(self.db)
        self.assertEqual((first.inserted, first.skipped), (len(DEMO_CHESTS), 0))
        second = seed_demo_chests(self.db)
        self.assertEqual((second.inserted, second.skipped), (0, len(DEMO_CHESTS)))
        self.assertEqual(len(self.repo.get_chests_by_map(DEMO_MAP_ID)), len(DEMO_CHESTS))

    def test_seeding_does_not_overwrite_edits_and_supports_other_maps(self) -> None:
        seed_demo_chests(self.db)
        first = self.repo.get_chests_by_map(DEMO_MAP_ID)[0]
        assert first.id is not None
        self.repo.update_chest(first.id, spawn_probability=0.51)
        seed_demo_chests(self.db)
        edited = self.repo.get_chest(first.id)
        assert edited is not None
        self.assertEqual(edited.spawn_probability, 0.51)

        other = seed_demo_chests(self.db, OTHER_MAP)
        self.assertEqual(other.inserted, len(DEMO_CHESTS))
        self.assertEqual(self.repo.list_map_ids(), [DEMO_MAP_ID, OTHER_MAP])


if __name__ == "__main__":
    unittest.main()
