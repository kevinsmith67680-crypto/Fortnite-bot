"""Phase 6 endurance runs: resets, no stale state, checkpoints, resume, failures, interruption."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from chest_hunter.analytics.endurance import EnduranceConfig, EnduranceRunner, memory_trend
from chest_hunter.exceptions import ValidationError
from chest_hunter.runtime.checkpoint import (
    Checkpoint,
    CheckpointError,
    append_result,
    check_compatible,
    load_checkpoint,
    read_results,
    save_checkpoint,
)
from chest_hunter.runtime.operations import OperationsControl
from tests.phase5_helpers import settings


def config(**kw: Any) -> EnduranceConfig:
    base: dict[str, Any] = {"episodes": 3, "max_ticks_per_episode": 450, "seed": 11,
                            "policy": "deterministic", "planner": "short_horizon",
                            "supervisor": True, "adaptive_compute": True,
                            "maps": ("proc_train_000",), "width": 320, "height": 180,
                            "checkpoint_every": 2}
    base.update(kw)
    return EnduranceConfig(**base)


class Checkpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="ch_cp_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    def _cp(self, last: int = 4) -> Checkpoint:
        return Checkpoint(kind="endurance", config={"seed": 1, "maps": ["m"]},
                          agent_version="6.0.0", policy_version="deterministic", seed=1,
                          seed_sequence="(seed, i)", last_completed_episode=last,
                          completed=last + 1, aggregate={"episodes": last + 1})

    def test_save_and_load_round_trip(self) -> None:
        path = self.dir / "checkpoint.json"
        save_checkpoint(path, self._cp())
        cp, warnings = load_checkpoint(path)
        assert cp is not None
        self.assertEqual((cp.last_completed_episode, cp.completed), (4, 5))
        self.assertEqual(warnings, [])
        self.assertTrue(json.loads(path.read_text())["checksum"])

    def test_a_corrupted_checkpoint_falls_back_to_the_backup(self) -> None:
        path = self.dir / "checkpoint.json"
        save_checkpoint(path, self._cp(3))
        save_checkpoint(path, self._cp(5))          # the 3 becomes the backup
        path.write_text(path.read_text()[:-40])     # torn write
        cp, warnings = load_checkpoint(path)
        assert cp is not None
        self.assertEqual(cp.last_completed_episode, 3)
        self.assertTrue(any("backup" in w for w in warnings))

    def test_an_edited_checkpoint_is_rejected(self) -> None:
        path = self.dir / "checkpoint.json"
        save_checkpoint(path, self._cp())
        doc = json.loads(path.read_text())
        doc["last_completed_episode"] = 99
        path.write_text(json.dumps(doc))
        with self.assertRaises(CheckpointError):
            load_checkpoint(path)                     # no backup to fall back to

    def test_no_checkpoint_is_not_an_error(self) -> None:
        self.assertEqual(load_checkpoint(self.dir / "none.json"), (None, []))

    def test_incompatible_configuration_is_refused(self) -> None:
        cp = self._cp()
        check_compatible(cp, config={"seed": 1, "maps": ["m"]}, agent_version="6.0.0")
        with self.assertRaises(CheckpointError) as err:
            check_compatible(cp, config={"seed": 2, "maps": ["m"]}, agent_version="6.0.0")
        self.assertIn("seed", str(err.exception))
        with self.assertRaises(CheckpointError):
            check_compatible(cp, config={"seed": 1, "maps": ["m"]}, agent_version="7.0.0")

    def test_results_keep_complete_records_only_once(self) -> None:
        path = self.dir / "episodes.jsonl"
        for k in range(3):
            append_result(path, {"episode": k, "outcome": "completed"})
        append_result(path, {"episode": 1, "outcome": "completed"})       # duplicate
        with path.open("a") as handle:
            handle.write('{"episode": 3, "outc')                         # torn last line
        records, warnings = read_results(path)
        self.assertEqual([r["episode"] for r in records], [0, 1, 2])
        self.assertEqual(len(warnings), 2)


class Runs(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="ch_end_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    def _run(self, cfg: EnduranceConfig, *, resume: bool = False,
             control: OperationsControl | None = None, hook: Any = None) -> dict[str, Any]:
        runner = EnduranceRunner(settings(), cfg, self.dir, resume=resume, control=control,
                                 on_decision=hook, install_signal_handlers=False)
        return runner.run()

    def test_repeated_episodes_reset_the_environment_and_the_agent(self) -> None:
        summary = self._run(config())
        self.assertEqual(summary["episodes_completed_in_run"], 3)
        records, _ = read_results(self.dir / "episodes.jsonl")
        self.assertEqual([r["episode"] for r in records], [0, 1, 2])
        self.assertTrue(all(r["outcome"] in ("completed", "tick_limit") for r in records))
        # the previous agent object is gone after every episode: no stale state
        self.assertEqual(summary["previous_agent_released"], {"checked": 2, "released": 2,
                                                               "not_released": 0})
        # one server, one connection: threads and descriptors stay constant
        self.assertEqual(len({r["threads"] for r in records}), 1)
        self.assertEqual(len({r["open_fds"] for r in records}), 1)
        cp, _ = load_checkpoint(self.dir / "checkpoint.json")
        assert cp is not None
        self.assertEqual(cp.last_completed_episode, 2)
        for name in ("summary.json", "report.md", "resources.jsonl", "endurance.db"):
            self.assertTrue((self.dir / name).exists(), name)

    def test_resume_continues_after_the_last_finished_episode(self) -> None:
        self._run(config(episodes=2))
        summary = self._run(config(episodes=4), resume=True)
        records, _ = read_results(self.dir / "episodes.jsonl")
        self.assertEqual([r["episode"] for r in records], [0, 1, 2, 3])
        self.assertEqual(summary["episodes_completed_in_run"], 4)
        with self.assertRaises(CheckpointError):            # not without --resume
            self._run(config(episodes=5))
        with self.assertRaises(CheckpointError):            # not with another configuration
            self._run(config(episodes=5, seed=12), resume=True)

    def test_a_failed_episode_is_logged_and_the_run_continues(self) -> None:
        from chest_hunter.agent import runner_3d

        original = runner_3d.Runner3D.run

        def flaky(self: Any) -> Any:
            if self.episode == 1:
                raise RuntimeError("injected failure (test)")
            return original(self)

        runner_3d.Runner3D.run = flaky                      # type: ignore[method-assign]
        self.addCleanup(setattr, runner_3d.Runner3D, "run", original)
        summary = self._run(config())
        records, _ = read_results(self.dir / "episodes.jsonl")
        self.assertEqual([r["outcome"] == "failed" for r in records], [False, True, False])
        self.assertIn("injected failure", records[1]["exception"])
        self.assertEqual(summary["aggregate"]["failed"], 1)

    def test_graceful_interruption_keeps_results_and_replays_the_episode(self) -> None:
        control = OperationsControl()

        def hook(runner: Any, _result: Any) -> None:
            if runner.episode == 1 and runner.controller.decisions == 50:
                control.request_stop("test interruption")

        summary = self._run(config(), control=control, hook=hook)
        self.assertEqual(summary["stopped_early"], "test interruption")
        records, _ = read_results(self.dir / "episodes.jsonl")
        self.assertEqual([r["episode"] for r in records], [0])      # episode 1 not recorded
        cp, _ = load_checkpoint(self.dir / "checkpoint.json")
        assert cp is not None
        self.assertEqual(cp.last_completed_episode, 0)
        self._run(config(), resume=True)                           # replays episode 1
        records, _ = read_results(self.dir / "episodes.jsonl")
        self.assertEqual([r["episode"] for r in records], [0, 1, 2])
        from chest_hunter.database import DatabaseManager

        db = DatabaseManager(self.dir / "endurance.db")
        self.addCleanup(db.close)
        rows = db.query_all("SELECT r.episode AS ep, s.status AS st FROM simulation_runs r "
                            "JOIN simulation_sessions s ON s.id = r.session_id "
                            "ORDER BY s.id")
        ep1 = [r["st"] for r in rows if r["ep"] == 1]
        self.assertEqual(ep1, ["aborted", "completed"])     # never two successful records

    def test_resume_aborts_sessions_left_running_by_a_crash(self) -> None:
        self._run(config(episodes=1))
        from chest_hunter.database import DatabaseManager

        db = DatabaseManager(self.dir / "endurance.db")
        with db.transaction() as conn:                       # what a killed process leaves
            conn.execute("UPDATE simulation_sessions SET status = 'running', ended_at = NULL")
        db.close()
        # the crashed episode's result line was never written: drop it from the results
        records, _ = read_results(self.dir / "episodes.jsonl")
        (self.dir / "episodes.jsonl").write_text("")
        (self.dir / "checkpoint.json").unlink()
        (self.dir / "checkpoint.json.bak").unlink(missing_ok=True)
        del records
        summary = self._run(config(episodes=1), resume=False)
        self.assertIn("marked aborted", " ".join(summary["warnings"]))

    def test_an_environment_that_cannot_be_reset_stops_the_run(self) -> None:
        cfg = config(episodes=3, max_reset_attempts=2)
        runner = EnduranceRunner(settings(), cfg, self.dir, install_signal_handlers=False)

        def broken() -> None:
            raise ConnectionRefusedError("no research server (test)")

        def episode_then_break(db: Any, episode: int) -> dict[str, Any]:
            record = original_episode(db, episode)
            runner._start_environment = broken              # type: ignore[method-assign]
            runner.server.stop()
            return record

        original_episode = runner._episode
        runner._episode = episode_then_break                 # type: ignore[method-assign]
        summary = runner.run()
        self.assertEqual(summary["episodes_completed_in_run"], 1)
        self.assertIn("could not be reset safely", summary["stopped_early"])
        self.assertEqual(summary["environment_resets_failed"], 2)

    def test_time_budget(self) -> None:
        summary = self._run(config(episodes=50, max_total_seconds=0.001))
        self.assertIn("time budget", summary["stopped_early"])


class Guards(unittest.TestCase):
    def test_test_maps_are_not_stored_in_persistent_experience(self) -> None:
        with self.assertRaises(ValidationError):
            config(maps=("proc_test_000",), preserve_estimates=True)
        config(maps=("proc_test_000",), preserve_estimates=True, allow_test_maps=True)
        config(maps=("proc_test_000",))                     # read-only use is fine

    def test_invalid_configurations(self) -> None:
        for bad in ({"episodes": 0}, {"max_ticks_per_episode": 10}, {"maps": ()},
                    {"policy": "hybrid"}, {"backend": "fortnite"}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                config(**bad)


class MemoryTrend(unittest.TestCase):
    def _rows(self, slope: float, n: int = 60) -> list[dict[str, Any]]:
        return [{"episode": k, "rss_mb": 100.0 + slope * k + (0.3 if k % 3 else -0.3),
                 "gc_objects": 30000 + (k % 5), "open_fds": 8, "threads": 3}
                for k in range(n)]

    def test_flat_memory_is_bounded(self) -> None:
        trend = memory_trend(self._rows(0.0))
        self.assertTrue(trend["bounded"])
        self.assertLess(abs(trend["rss_growth_mb_per_100_episodes"]), 1.0)

    def test_growing_memory_is_flagged(self) -> None:
        trend = memory_trend(self._rows(0.2))
        self.assertFalse(trend["bounded"])
        self.assertAlmostEqual(trend["rss_growth_mb_per_100_episodes"], 20.0, delta=1.0)

    def test_too_few_episodes_give_no_verdict(self) -> None:
        self.assertIsNone(memory_trend(self._rows(0.0, n=10))["bounded"])


if __name__ == "__main__":
    unittest.main()
