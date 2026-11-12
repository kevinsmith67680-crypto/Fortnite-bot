"""Tests for DatabaseManager: initialisation, migrations, constraints, transactions."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from chest_hunter.database import EXPECTED_TABLES, SCHEMA_VERSION, DatabaseManager
from chest_hunter.exceptions import DatabaseError, DatabaseIntegrityError

NOW = "2026-01-01T00:00:00.000000+00:00"
LATER = "2026-01-01T00:10:00.000000+00:00"


def insert_chest(conn: sqlite3.Connection, **overrides: object) -> int:
    """Insert a chest row with raw SQL (bypassing repository validation)."""
    values: dict[str, object] = {
        "map_id": "arena",
        "location_name": "Spot",
        "x": 1.0,
        "y": 2.0,
        "z": 0.0,
        "spawn_probability": 0.5,
        "estimated_open_seconds": 2.0,
        "enabled": 1,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    cursor = conn.execute(
        f"INSERT INTO chest_locations ({columns}) VALUES ({placeholders})", tuple(values.values())
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def insert_session(conn: sqlite3.Connection, map_id: str = "arena") -> int:
    cursor = conn.execute(
        "INSERT INTO simulation_sessions (map_id, started_at, status) VALUES (?, ?, 'running')",
        (map_id, NOW),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def insert_observation(
    conn: sqlite3.Connection,
    session_id: int,
    chest_id: int,
    present: int = 1,
    opened: int = 0,
    seconds: float = 1.0,
) -> int:
    cursor = conn.execute(
        "INSERT INTO chest_observations (session_id, chest_location_id, observed_at, "
        "chest_present, chest_opened, interaction_seconds) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, chest_id, NOW, present, opened, seconds),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)
        self.db_path = self.tmp_path / "nested" / "dir" / "test.db"
        self.db = DatabaseManager(self.db_path)
        self.addCleanup(self.db.close)  # runs before tmp.cleanup (LIFO)


class InitialisationTests(DatabaseTestCase):
    def test_fresh_initialisation_creates_file_tables_and_indexes(self) -> None:
        self.assertFalse(self.db_path.exists())
        version = self.db.initialize()

        self.assertEqual(version, SCHEMA_VERSION)
        self.assertTrue(self.db_path.is_file(), "parent directories and file should be created")
        tables = {
            row["name"]
            for row in self.db.query_all("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        self.assertTrue(set(EXPECTED_TABLES).issubset(tables))
        indexes = {
            row["name"]
            for row in self.db.query_all("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        for expected in (
            "uq_chest_locations_map_position",
            "idx_chest_locations_map_enabled",
            "idx_simulation_sessions_map_started",
            "idx_chest_observations_session_chest",
            "uq_chest_observations_one_opening",
        ):
            self.assertIn(expected, indexes)

    def test_schema_version_is_recorded_in_user_version(self) -> None:
        self.assertEqual(self.db.schema_version(), 0)
        self.assertFalse(self.db.is_initialized())
        self.db.initialize()
        self.assertEqual(self.db.schema_version(), SCHEMA_VERSION)
        raw = self.db.query_one("PRAGMA user_version")
        assert raw is not None
        self.assertEqual(raw[0], SCHEMA_VERSION)
        self.assertTrue(self.db.is_initialized())

    def test_repeated_initialisation_is_idempotent_and_preserves_data(self) -> None:
        self.db.initialize()
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn)

        self.assertEqual(self.db.initialize(), SCHEMA_VERSION)
        self.assertEqual(self.db.initialize(), SCHEMA_VERSION)

        rows = self.db.query_all("SELECT id FROM chest_locations")
        self.assertEqual([row["id"] for row in rows], [chest_id])

    def test_initialisation_in_a_new_manager_preserves_data(self) -> None:
        self.db.initialize()
        with self.db.transaction() as conn:
            insert_chest(conn)
        self.db.close()

        with DatabaseManager(self.db_path) as second:
            second.initialize()
            row = second.query_one("SELECT COUNT(*) AS n FROM chest_locations")
            assert row is not None
            self.assertEqual(row["n"], 1)

    def test_newer_schema_version_is_refused_without_modification(self) -> None:
        self.db.initialize()
        with self.db.transaction() as conn:
            insert_chest(conn)
        self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")

        with self.assertRaisesRegex(DatabaseError, "newer|only supports"):
            self.db.initialize()
        row = self.db.query_one("SELECT COUNT(*) AS n FROM chest_locations")
        assert row is not None
        self.assertEqual(row["n"], 1)
        self.assertEqual(self.db.schema_version(), SCHEMA_VERSION + 5)

    def test_conflicting_existing_table_is_reported_not_overwritten(self) -> None:
        self.db.execute("CREATE TABLE chest_locations (something TEXT)")
        self.db.execute("INSERT INTO chest_locations VALUES ('keep me')")

        with self.assertRaisesRegex(DatabaseError, "migration 1"):
            self.db.initialize()
        self.assertEqual(self.db.schema_version(), 0)
        row = self.db.query_one("SELECT something FROM chest_locations")
        assert row is not None
        self.assertEqual(row["something"], "keep me")

    def test_non_database_file_raises_database_error(self) -> None:
        bogus = self.tmp_path / "not_a_db.db"
        bogus.write_bytes(b"this is definitely not an SQLite database file" * 20)
        manager = DatabaseManager(bogus)
        self.addCleanup(manager.close)
        with self.assertRaises(DatabaseError):
            manager.initialize()

    def test_directory_path_raises_database_error(self) -> None:
        manager = DatabaseManager(self.tmp_path)
        with self.assertRaisesRegex(DatabaseError, "directory"):
            manager.connect()

    def test_in_memory_database_is_supported(self) -> None:
        with DatabaseManager(":memory:") as memory:
            self.assertEqual(memory.initialize(), SCHEMA_VERSION)
            self.assertIsNone(memory.path)
            self.assertFalse(memory.exists())


class ForeignKeyAndConstraintTests(DatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.db.initialize()

    def test_foreign_keys_pragma_is_enabled(self) -> None:
        row = self.db.query_one("PRAGMA foreign_keys")
        assert row is not None
        self.assertEqual(row[0], 1)

    def test_observation_with_unknown_session_or_chest_is_rejected(self) -> None:
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn)
            session_id = insert_session(conn)
        with self.assertRaises(DatabaseIntegrityError):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id, chest_id + 999)
        with self.assertRaises(DatabaseIntegrityError):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id + 999, chest_id)

    def test_foreign_key_enforced_on_insert_without_validation_trigger(self) -> None:
        # The validation trigger would reject these rows first; drop it so the
        # assertion exercises SQLite's foreign-key enforcement on its own.
        self.db.execute("DROP TRIGGER trg_chest_observations_validate_insert")
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn)
            session_id = insert_session(conn)
        with self.assertRaisesRegex(DatabaseIntegrityError, "FOREIGN KEY"):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id, chest_id + 999)
        with self.assertRaisesRegex(DatabaseIntegrityError, "FOREIGN KEY"):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id + 999, chest_id)

    def test_foreign_key_prevents_deleting_referenced_rows(self) -> None:
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn)
            session_id = insert_session(conn)
            insert_observation(conn, session_id, chest_id)
        with self.assertRaisesRegex(DatabaseIntegrityError, "FOREIGN KEY"):
            self.db.execute("DELETE FROM chest_locations WHERE id = ?", (chest_id,))
        with self.assertRaisesRegex(DatabaseIntegrityError, "FOREIGN KEY"):
            self.db.execute("DELETE FROM simulation_sessions WHERE id = ?", (session_id,))

    def test_check_constraints_reject_invalid_chest_values(self) -> None:
        invalid: list[dict[str, object]] = [
            {"spawn_probability": 1.5},
            {"spawn_probability": -0.1},
            {"estimated_open_seconds": -1.0},
            {"x": float("inf")},
            {"y": float("nan")},
            {"z": "high"},
            {"enabled": 2},
            {"map_id": "   "},
            {"location_name": ""},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(DatabaseIntegrityError):
                    with self.db.transaction() as conn:
                        insert_chest(conn, **overrides)
        row = self.db.query_one("SELECT COUNT(*) AS n FROM chest_locations")
        assert row is not None
        self.assertEqual(row["n"], 0)

    def test_unique_position_per_map(self) -> None:
        with self.db.transaction() as conn:
            insert_chest(conn, map_id="a", x=5.0)
            insert_chest(conn, map_id="b", x=5.0)  # same position, other map: fine
        with self.assertRaisesRegex(DatabaseIntegrityError, "UNIQUE"):
            with self.db.transaction() as conn:
                insert_chest(conn, map_id="a", x=5.0)

    def test_schema_rejects_opened_without_present_and_negative_duration(self) -> None:
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn)
            session_id = insert_session(conn)
        with self.assertRaises(DatabaseIntegrityError):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id, chest_id, present=0, opened=1)
        with self.assertRaises(DatabaseIntegrityError):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id, chest_id, seconds=-1.0)

    def test_schema_rejects_duplicate_opening_in_same_session(self) -> None:
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn)
            session_id = insert_session(conn)
            insert_observation(conn, session_id, chest_id, opened=1)
            insert_observation(conn, session_id, chest_id, opened=0)  # failed retry is fine
        with self.assertRaisesRegex(DatabaseIntegrityError, "UNIQUE"):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id, chest_id, opened=1)

    def test_trigger_rejects_observation_for_finished_session(self) -> None:
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn)
            session_id = insert_session(conn)
            conn.execute(
                "UPDATE simulation_sessions SET status = 'completed', ended_at = ? WHERE id = ?",
                (LATER, session_id),
            )
        with self.assertRaisesRegex(DatabaseIntegrityError, "running session"):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id, chest_id)

    def test_trigger_rejects_observation_for_other_map(self) -> None:
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn, map_id="other_map")
            session_id = insert_session(conn, map_id="arena")
        with self.assertRaisesRegex(DatabaseIntegrityError, "different map"):
            with self.db.transaction() as conn:
                insert_observation(conn, session_id, chest_id)

    def test_trigger_rejects_reopening_finished_session(self) -> None:
        with self.db.transaction() as conn:
            session_id = insert_session(conn)
            conn.execute(
                "UPDATE simulation_sessions SET status = 'aborted', ended_at = ? WHERE id = ?",
                (LATER, session_id),
            )
        with self.assertRaisesRegex(DatabaseIntegrityError, "already finished"):
            self.db.execute(
                "UPDATE simulation_sessions SET status = 'running', ended_at = NULL WHERE id = ?",
                (session_id,),
            )

    def test_session_status_and_end_time_must_agree(self) -> None:
        with self.assertRaises(DatabaseIntegrityError):
            self.db.execute(
                "INSERT INTO simulation_sessions (map_id, started_at, status) "
                "VALUES ('arena', ?, 'completed')",
                (NOW,),
            )
        with self.assertRaises(DatabaseIntegrityError):
            self.db.execute(
                "INSERT INTO simulation_sessions (map_id, started_at, ended_at, status) "
                "VALUES ('arena', ?, ?, 'aborted')",
                (LATER, NOW),  # ends before it starts
            )


class TransactionTests(DatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.db.initialize()

    def count_chests(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM chest_locations")
        assert row is not None
        return int(row["n"])

    def test_transaction_commits(self) -> None:
        with self.db.transaction() as conn:
            insert_chest(conn, x=1.0)
            insert_chest(conn, x=2.0)
        self.assertEqual(self.count_chests(), 2)

    def test_transaction_rolls_back_every_statement_on_error(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.db.transaction() as conn:
                insert_chest(conn, x=1.0)
                insert_chest(conn, x=2.0)
                raise RuntimeError("boom")
        self.assertEqual(self.count_chests(), 0)
        self.assertFalse(self.db.connection.in_transaction)

    def test_nested_transaction_rolls_back_only_inner_block(self) -> None:
        with self.db.transaction() as conn:
            insert_chest(conn, x=1.0)
            with self.assertRaises(DatabaseIntegrityError):
                with self.db.transaction() as inner:
                    insert_chest(inner, x=2.0)
                    insert_chest(inner, x=1.0)  # duplicate position -> UNIQUE violation
        self.assertEqual(self.count_chests(), 1)

    def test_sqlite_errors_are_translated(self) -> None:
        with self.assertRaises(DatabaseError):
            self.db.execute("SELECT * FROM no_such_table")


class PersistenceTests(DatabaseTestCase):
    def test_data_persists_across_connections(self) -> None:
        self.db.initialize()
        with self.db.transaction() as conn:
            chest_id = insert_chest(conn, location_name="Persistent Spot")
        self.db.close()
        self.assertFalse(self.db.is_open)

        reopened = DatabaseManager(self.db_path)
        self.addCleanup(reopened.close)
        row = reopened.query_one(
            "SELECT location_name FROM chest_locations WHERE id = ?", (chest_id,)
        )
        assert row is not None
        self.assertEqual(row["location_name"], "Persistent Spot")
        fk = reopened.query_one("PRAGMA foreign_keys")
        assert fk is not None
        self.assertEqual(fk[0], 1, "foreign keys must be enabled on every new connection")

    def test_close_is_idempotent_and_reconnects_lazily(self) -> None:
        self.db.initialize()
        self.db.close()
        self.db.close()
        self.assertEqual(self.db.schema_version(), SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
