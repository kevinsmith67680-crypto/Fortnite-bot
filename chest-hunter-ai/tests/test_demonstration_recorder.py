"""Demonstration recording: controls, alignment, file output, de-duplication, replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chest_hunter.learning.demonstration_recorder import (
    ACTION_FIELDS,
    SOURCE_HUMAN,
    SOURCE_SCRIPTED,
    DemonstrationRecorder,
    RecordingError,
    iter_samples,
    load_episode,
    next_episode_id,
    replay_episode,
)
from chest_hunter.unreal_bridge.client import BridgeClient
from chest_hunter.unreal_bridge.mock_server import MockUnrealServer
from chest_hunter.unreal_bridge.protocol import IDLE_ACTION
from tests.phase5_helpers import tiny_dataset

MAP = "research_arena_3d"


def _act(**kw: object) -> dict[str, object]:
    return {**IDLE_ACTION, **kw}


class RecorderWithServer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = MockUnrealServer().start()
        self.addCleanup(self.server.stop)
        self.client = BridgeClient(port=self.server.port, connect_attempts=1)
        self.client.connect()
        self.addCleanup(self.client.close)
        self.tel = self.client.reset_episode(MAP, seed=3, width=320, height=180)
        self.rec = DemonstrationRecorder(Path(self.tmp.name))

    def _record(self, steps: int, *, observe_every: int = 2) -> Path:
        self.rec.start(episode_id=7, map_id=MAP, seed=3, episode=0, source=SOURCE_HUMAN,
                       telemetry=self.tel, settings={"width": 320, "height": 180})
        t, obs = self.tel.sim_time, None
        for k in range(steps):
            action = _act(move_forward=1.0, look_yaw=0.2 if k % 3 else -0.1)
            observe = k % observe_every == 0
            res = self.client.step(action, ticks=2, observe=observe)
            self.rec.record_step(action=action, ticks=2, action_time=t,
                                 telemetry_after=res.telemetry, observation=obs, observed=observe)
            t, obs = res.telemetry.sim_time, res.observation
        return self.rec.stop(outcome={"outcome": "test"})

    def test_files_and_metadata(self) -> None:
        directory = self._record(20)
        meta, arrays = load_episode(directory)
        self.assertEqual(meta["episode_id"], 7)
        self.assertEqual(meta["steps"], 20)
        self.assertEqual(meta["samples"], 10)              # every other step has a fresh frame
        self.assertFalse(meta["synthetic"])
        self.assertTrue(meta["replayable"])
        self.assertEqual(arrays["action"].shape, (20, len(ACTION_FIELDS)))
        self.assertEqual(len(list((directory / "frames").glob("*.png"))), meta["frames"])
        samples = iter_samples(directory)
        self.assertEqual(len(samples), 10)
        self.assertEqual(set(samples[0]) >= {"episode_id", "frame_id", "observation_path",
                                             "move_forward", "interact"}, True)
        # frame_time == action_time for every training sample (alignment).
        refs = arrays["frame_ref"] >= 0
        self.assertTrue(np.allclose(arrays["frame_time"][refs], arrays["action_time"][refs]))

    def test_replay_is_exact(self) -> None:
        directory = self._record(30)
        report = replay_episode(directory, self.client)
        self.assertTrue(report.exact, report)
        self.assertEqual(report.frames_compared, 15)

    def test_misaligned_observation_is_rejected(self) -> None:
        self.rec.start(episode_id=1, map_id=MAP, seed=3, episode=0, source=SOURCE_HUMAN,
                       telemetry=self.tel)
        res = self.client.step(_act(), ticks=2, observe=True)
        res2 = self.client.step(_act(), ticks=2)
        with self.assertRaises(RecordingError):
            # the observation is from t=2/30 but the action is said to start at t=0
            self.rec.record_step(action=_act(), ticks=2, action_time=0.0,
                                 telemetry_after=res2.telemetry, observation=res.observation)

    def test_invalid_actions_are_rejected(self) -> None:
        self.rec.start(episode_id=1, map_id=MAP, seed=3, episode=0, source=SOURCE_HUMAN,
                       telemetry=self.tel)
        res = self.client.step(_act(), ticks=2)
        for bad in (_act(move_forward=1.5), _act(jump=1), _act(look_yaw=float("nan"))):
            with self.subTest(bad=bad), self.assertRaises(RecordingError):
                self.rec.record_step(action=bad, ticks=2, action_time=0.0,
                                     telemetry_after=res.telemetry)

    def test_pause_resume_and_state_errors(self) -> None:
        with self.assertRaises(RecordingError):
            self.rec.pause()
        self.rec.start(episode_id=2, map_id=MAP, seed=3, episode=0, source=SOURCE_SCRIPTED,
                       telemetry=self.tel)
        with self.assertRaises(RecordingError):
            self.rec.start(episode_id=3, map_id=MAP, seed=3, episode=0, source=SOURCE_SCRIPTED)
        self.rec.pause()
        res = self.client.step(_act(), ticks=2)
        self.rec.record_step(action=_act(), ticks=2, action_time=0.0, telemetry_after=res.telemetry)
        self.rec.resume()
        res = self.client.step(_act(), ticks=2)
        self.rec.record_step(action=_act(), ticks=2, action_time=res.telemetry.sim_time - 2 / 30,
                             telemetry_after=res.telemetry)
        directory = self.rec.stop()
        meta, _ = load_episode(directory)
        self.assertEqual(meta["steps"], 1)
        self.assertEqual(meta["steps_skipped_while_paused"], 1)
        self.assertFalse(meta["replayable"])
        self.assertTrue(meta["synthetic"])
        with self.assertRaises(RecordingError):
            replay_episode(directory, self.client)

    def test_never_overwrites_and_unknown_source(self) -> None:
        self._record(2)
        with self.assertRaises(RecordingError):
            self.rec.start(episode_id=7, map_id=MAP, seed=3, episode=0, source=SOURCE_HUMAN)
        with self.assertRaises(RecordingError):
            self.rec.start(episode_id=8, map_id=MAP, seed=3, episode=0, source="someone")
        self.assertEqual(next_episode_id(self.tmp.name), 8)

    def test_identical_frames_are_stored_once_and_memory_is_released(self) -> None:
        self.rec.start(episode_id=4, map_id=MAP, seed=3, episode=0, source=SOURCE_HUMAN,
                       telemetry=self.tel)
        obs = self.client.get_observation()
        t = obs.sim_time
        for _ in range(5):                                 # same frame, five samples
            res = self.client.step(_act(), ticks=1)
            self.rec.record_step(action=_act(), ticks=1, action_time=t,
                                 telemetry_after=res.telemetry, observation=obs)
            obs = obs.__class__(**{**{f: getattr(obs, f) for f in obs.__dataclass_fields__},
                                   "sim_time": res.telemetry.sim_time})
            t = res.telemetry.sim_time
        self.assertTrue(all(not hasattr(s, "frame") for s in self.rec._steps))
        directory = self.rec.stop()
        meta, _ = load_episode(directory)
        self.assertEqual((meta["samples"], meta["frames"]), (5, 1))
        self.assertEqual(self.rec._steps, [])               # nothing kept after stop


class ScriptedDemonstrations(unittest.TestCase):
    def test_synthetic_episodes_are_labelled_and_valid(self) -> None:
        root = tiny_dataset()
        episodes = sorted(root.glob("episode_*"))
        self.assertEqual(len(episodes), 4)
        for directory in episodes:
            meta = json.loads((directory / "episode.json").read_text())
            self.assertEqual(meta["source"], SOURCE_SCRIPTED)
            self.assertTrue(meta["synthetic"])
            self.assertGreater(meta["samples"], 0)
            self.assertIn(meta["map_id"], ("proc_train_000", "proc_train_001"))


if __name__ == "__main__":
    unittest.main()
