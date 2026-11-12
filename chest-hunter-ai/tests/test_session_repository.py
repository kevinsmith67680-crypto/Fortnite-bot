"""Tests for SessionRepository: lifecycle, observations and session statistics."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chest_hunter.chest_repository import ChestRepository
from chest_hunter.database import DatabaseManager
from chest_hunter.exceptions import (
    DuplicateOpeningError,
    RecordNotFoundError,
    SessionStateError,
    ValidationError,
)
from chest_hunter.models import ChestObservation, SessionStatus, SimulationSession, parse_timestamp
from chest_hunter.session_repository import SessionRepository

MAP = "research_arena_01"
OTHER_MAP = "research_arena_02"


class ManualClock:
    """Deterministic clock so durations and rates can be asserted exactly."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class SessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = DatabaseManager(Path(tmp.name) / "sessions.db")
        self.addCleanup(self.db.close)
        self.db.initialize()
        self.clock = ManualClock()
        self.chests = ChestRepository(self.db, clock=self.clock)
        self.sessions = SessionRepository(self.db, clock=self.clock)
        self.chest_ids: list[int] = []
        for index in range(4):
            chest = self.chests.add_chest(MAP, f"Spot {index}", float(index * 10), 0.0)
            assert chest.id is not None
            self.chest_ids.append(chest.id)
        other = self.chests.add_chest(OTHER_MAP, "Elsewhere", 0.0, 0.0)
        assert other.id is not None
        self.other_map_chest = other.id

    def start(self, map_id: str = MAP) -> int:
        session = self.sessions.start_session(map_id)
        assert session.id is not None
        return session.id

    def count_observations(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM chest_observations")
        assert row is not None
        return int(row["n"])


class LifecycleTests(SessionTestCase):
    def test_start_session(self) -> None:
        session = self.sessions.start_session(MAP, notes="  baseline run  ")
        self.assertIsInstance(session, SimulationSession)
        self.assertIsNotNone(session.id)
        self.assertEqual(session.map_id, MAP)
        self.assertIs(session.status, SessionStatus.RUNNING)
        self.assertTrue(session.is_active)
        self.assertIsNone(session.ended_at)
        self.assertEqual(session.notes, "baseline run")
        self.assertEqual(parse_timestamp(session.started_at), self.clock.now)
        assert session.id is not None
        self.assertEqual(self.sessions.get_session(session.id), session)

    def test_start_session_rejects_invalid_map(self) -> None:
        for bad in ("", "  ", "bad map id"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    self.sessions.start_session(bad)
        self.assertEqual(self.sessions.list_sessions(), [])

    def test_complete_session(self) -> None:
        session_id = self.start()
        self.clock.advance(90)
        session = self.sessions.complete_session(session_id)
        self.assertIs(session.status, SessionStatus.COMPLETED)
        self.assertFalse(session.is_active)
        self.assertEqual(session.duration_seconds, 90.0)
        self.assertEqual(self.sessions.get_session(session_id), session)

    def test_abort_session(self) -> None:
        session_id = self.start()
        self.clock.advance(5)
        session = self.sessions.abort_session(session_id)
        self.assertIs(session.status, SessionStatus.ABORTED)
        assert session.ended_at is not None
        self.assertEqual(parse_timestamp(session.ended_at), self.clock.now)

    def test_invalid_state_transitions_are_rejected(self) -> None:
        completed = self.start()
        self.sessions.complete_session(completed)
        aborted = self.start()
        self.sessions.abort_session(aborted)

        for session_id in (completed, aborted):
            with self.subTest(session_id=session_id):
                with self.assertRaises(SessionStateError):
                    self.sessions.complete_session(session_id)
                with self.assertRaises(SessionStateError):
                    self.sessions.abort_session(session_id)
        # The original outcome is preserved.
        self.assertIs(self.sessions.require_session(completed).status, SessionStatus.COMPLETED)
        self.assertIs(self.sessions.require_session(aborted).status, SessionStatus.ABORTED)

    def test_finishing_unknown_session_raises_not_found(self) -> None:
        with self.assertRaises(RecordNotFoundError):
            self.sessions.complete_session(999)
        with self.assertRaises(RecordNotFoundError):
            self.sessions.abort_session(999)
        self.assertIsNone(self.sessions.get_session(999))

    def test_clock_going_backwards_records_zero_duration(self) -> None:
        session_id = self.start()
        self.clock.advance(-30)
        with self.assertLogs("chest_hunter.session_repository", level="WARNING"):
            session = self.sessions.complete_session(session_id)
        self.assertEqual(session.duration_seconds, 0.0)

    def test_list_sessions_filters_and_order(self) -> None:
        s1 = self.start()
        s2 = self.start(OTHER_MAP)
        s3 = self.start()
        self.sessions.complete_session(s1)

        self.assertEqual([s.id for s in self.sessions.list_sessions()], [s1, s2, s3])
        self.assertEqual([s.id for s in self.sessions.list_sessions(map_id=MAP)], [s1, s3])
        self.assertEqual([s.id for s in self.sessions.list_sessions(status="running")], [s2, s3])
        self.assertEqual(
            [s.id for s in self.sessions.list_sessions(newest_first=True, limit=2)], [s3, s2]
        )
        with self.assertRaises(ValidationError):
            self.sessions.list_sessions(status="paused")


class ObservationTests(SessionTestCase):
    def test_record_successful_opening(self) -> None:
        session_id = self.start()
        self.clock.advance(12)
        observation = self.sessions.record_observation(
            session_id, self.chest_ids[0], chest_present=True, chest_opened=True,
            interaction_seconds=1.8,
        )
        self.assertIsInstance(observation, ChestObservation)
        self.assertIsNotNone(observation.id)
        self.assertEqual(observation.session_id, session_id)
        self.assertEqual(observation.chest_location_id, self.chest_ids[0])
        self.assertTrue(observation.chest_present)
        self.assertTrue(observation.chest_opened)
        self.assertEqual(observation.interaction_seconds, 1.8)
        self.assertEqual(parse_timestamp(observation.observed_at), self.clock.now)
        self.assertEqual(self.sessions.get_observations(session_id), [observation])

    def test_record_absent_chest(self) -> None:
        session_id = self.start()
        observation = self.sessions.record_observation(
            session_id, self.chest_ids[1], chest_present=False
        )
        self.assertFalse(observation.chest_present)
        self.assertFalse(observation.chest_opened)
        self.assertEqual(observation.interaction_seconds, 0.0)

    def test_reject_opening_an_absent_chest(self) -> None:
        session_id = self.start()
        with self.assertRaisesRegex(ValidationError, "not present"):
            self.sessions.record_observation(
                session_id, self.chest_ids[0], chest_present=False, chest_opened=True
            )
        self.assertEqual(self.count_observations(), 0)

    def test_reject_negative_or_invalid_interaction_time(self) -> None:
        session_id = self.start()
        for value in (-0.5, float("nan"), float("inf"), "2"):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    self.sessions.record_observation(
                        session_id, self.chest_ids[0], chest_present=True,
                        interaction_seconds=value,  # type: ignore[arg-type]
                    )
        self.assertEqual(self.count_observations(), 0)

    def test_reject_non_boolean_flags(self) -> None:
        session_id = self.start()
        with self.assertRaises(ValidationError):
            self.sessions.record_observation(session_id, self.chest_ids[0], chest_present=1)  # type: ignore[arg-type]

    def test_prevent_duplicate_successful_openings(self) -> None:
        session_id = self.start()
        chest = self.chest_ids[0]
        # A failed attempt followed by a success is a legitimate sequence.
        self.sessions.record_observation(session_id, chest, chest_present=True, chest_opened=False)
        self.sessions.record_observation(session_id, chest, chest_present=True, chest_opened=True)
        with self.assertRaises(DuplicateOpeningError):
            self.sessions.record_observation(session_id, chest, chest_present=True,
                                             chest_opened=True)
        # Revisiting after opening (now empty) is still allowed.
        self.sessions.record_observation(session_id, chest, chest_present=False)
        self.assertEqual(self.count_observations(), 3)

    def test_same_location_can_be_opened_again_in_a_later_session(self) -> None:
        first = self.start()
        self.sessions.record_observation(first, self.chest_ids[0], chest_present=True,
                                         chest_opened=True)
        self.sessions.complete_session(first)
        second = self.start()
        self.sessions.record_observation(second, self.chest_ids[0], chest_present=True,
                                         chest_opened=True)
        self.assertEqual(self.count_observations(), 2)

    def test_prevent_observations_after_completion_or_abort(self) -> None:
        completed = self.start()
        self.sessions.complete_session(completed)
        aborted = self.start()
        self.sessions.abort_session(aborted)
        for session_id in (completed, aborted):
            with self.subTest(session_id=session_id):
                with self.assertRaises(SessionStateError):
                    self.sessions.record_observation(session_id, self.chest_ids[0],
                                                     chest_present=False)
        self.assertEqual(self.count_observations(), 0)

    def test_observation_must_match_session_map(self) -> None:
        session_id = self.start(MAP)
        with self.assertRaisesRegex(ValidationError, "map"):
            self.sessions.record_observation(session_id, self.other_map_chest,
                                             chest_present=True)
        self.assertEqual(self.count_observations(), 0)

    def test_observation_rejected_for_disabled_location(self) -> None:
        session_id = self.start()
        self.chests.update_chest(self.chest_ids[2], enabled=False)
        with self.assertRaisesRegex(ValidationError, "disabled"):
            self.sessions.record_observation(session_id, self.chest_ids[2], chest_present=True)

    def test_unknown_session_or_chest_raises_not_found(self) -> None:
        session_id = self.start()
        with self.assertRaises(RecordNotFoundError):
            self.sessions.record_observation(999, self.chest_ids[0], chest_present=True)
        with self.assertRaises(RecordNotFoundError):
            self.sessions.record_observation(session_id, 999, chest_present=True)
        with self.assertRaises(RecordNotFoundError):
            self.sessions.get_observations(999)


class StatisticsTests(SessionTestCase):
    def test_session_statistics(self) -> None:
        session_id = self.start()
        c0, c1, c2, c3 = self.chest_ids
        self.clock.advance(10)
        self.sessions.record_observation(session_id, c0, chest_present=True, chest_opened=True,
                                         interaction_seconds=2.0)
        self.sessions.record_observation(session_id, c1, chest_present=True, chest_opened=False,
                                         interaction_seconds=3.0)  # failed interaction
        self.sessions.record_observation(session_id, c1, chest_present=True, chest_opened=True,
                                         interaction_seconds=1.5)  # retry succeeds
        self.sessions.record_observation(session_id, c2, chest_present=False)  # empty spawn
        self.sessions.record_observation(session_id, c0, chest_present=False)  # revisit
        self.clock.advance(110)  # total duration: 120 s
        self.sessions.complete_session(session_id)

        stats = self.sessions.get_session_statistics(session_id)
        self.assertEqual(stats.session_id, session_id)
        self.assertEqual(stats.map_id, MAP)
        self.assertIs(stats.status, SessionStatus.COMPLETED)
        self.assertEqual(stats.locations_visited, 3)
        self.assertEqual(stats.total_observations, 5)
        self.assertEqual(stats.chests_found, 2)
        self.assertEqual(stats.chests_opened, 2)
        self.assertEqual(stats.failed_interactions, 1)
        self.assertAlmostEqual(stats.total_interaction_seconds, 6.5)
        self.assertEqual(stats.duration_seconds, 120.0)
        assert stats.chests_opened_per_minute is not None
        self.assertAlmostEqual(stats.chests_opened_per_minute, 1.0)
        observed = {o.chest_location_id for o in self.sessions.get_observations(session_id)}
        self.assertNotIn(c3, observed)

    def test_running_session_has_no_duration_or_rate(self) -> None:
        session_id = self.start()
        self.sessions.record_observation(session_id, self.chest_ids[0], chest_present=True,
                                         chest_opened=True)
        stats = self.sessions.get_session_statistics(session_id)
        self.assertIsNone(stats.duration_seconds)
        self.assertIsNone(stats.chests_opened_per_minute)
        self.assertEqual(stats.chests_opened, 1)

    def test_zero_duration_session_does_not_divide_by_zero(self) -> None:
        session_id = self.start()
        self.sessions.record_observation(session_id, self.chest_ids[0], chest_present=True,
                                         chest_opened=True)
        self.sessions.complete_session(session_id)  # clock did not advance
        stats = self.sessions.get_session_statistics(session_id)
        self.assertEqual(stats.duration_seconds, 0.0)
        self.assertIsNone(stats.chests_opened_per_minute)

    def test_zero_openings_gives_zero_rate(self) -> None:
        session_id = self.start()
        self.clock.advance(60)
        self.sessions.abort_session(session_id)
        stats = self.sessions.get_session_statistics(session_id)
        self.assertEqual(stats.locations_visited, 0)
        self.assertEqual(stats.chests_opened, 0)
        self.assertEqual(stats.failed_interactions, 0)
        self.assertEqual(stats.chests_opened_per_minute, 0.0)

    def test_statistics_for_unknown_session_raise_not_found(self) -> None:
        with self.assertRaises(RecordNotFoundError):
            self.sessions.get_session_statistics(42)

    def test_statistics_are_isolated_per_session(self) -> None:
        first = self.start()
        second = self.start()
        self.sessions.record_observation(first, self.chest_ids[0], chest_present=True,
                                         chest_opened=True)
        stats = self.sessions.get_session_statistics(second)
        self.assertEqual(stats.total_observations, 0)


if __name__ == "__main__":
    unittest.main()
