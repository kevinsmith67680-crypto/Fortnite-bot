"""Phase 6 adaptive computation: vision rates, frame skipping, caches, bounded tracking."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from chest_hunter.agent.controller_3d import (
    VISION_PROFILES,
    Agent3DConfig,
    Agent3DState,
    apply_vision_profile,
)
from chest_hunter.navigation.navigation_3d import NavMeshNavigator
from chest_hunter.unreal_bridge.client import NavResult
from tests.phase5_helpers import settings


def _episode(*, config: Agent3DConfig, seed: int = 4, ticks: int = 900, hook: Any = None,
             profiler: Any = None, map_id: str = "research_arena_3d") -> tuple[Any, Any]:
    from chest_hunter.agent.runner_3d import Runner3D
    from chest_hunter.database import DatabaseManager
    from chest_hunter.scenarios import seed_arena_3d

    db = DatabaseManager(":memory:")
    db.initialize()
    if map_id == "research_arena_3d":
        seed_arena_3d(db)
    runner = Runner3D.from_settings(db, settings(), mode="hybrid", seed=seed, map_id=map_id,
                                    backend="reference", max_ticks=ticks, width=320,
                                    height=180, config=config, on_decision=hook,
                                    profiler=profiler)
    try:
        return runner, runner.run()
    finally:
        db.close()


class VisionProfiles(unittest.TestCase):
    def test_profiles_set_the_rates(self) -> None:
        base = Agent3DConfig()
        for name, rates in VISION_PROFILES.items():
            cfg = apply_vision_profile(base, name)
            for key, value in rates.items():
                self.assertEqual(getattr(cfg, key), value, (name, key))
        self.assertTrue(apply_vision_profile(base, "efficient",
                                             skip_unchanged=True).skip_unchanged_frames)
        with self.assertRaises(ValueError):
            apply_vision_profile(base, "turbo")

    def test_the_default_profile_is_the_phase5_behaviour(self) -> None:
        base = Agent3DConfig()
        self.assertEqual(apply_vision_profile(base, "phase5"), base)

    def test_an_efficient_profile_analyses_fewer_frames(self) -> None:
        _r1, phase5 = _episode(config=apply_vision_profile(Agent3DConfig(), "phase5"))
        _r2, efficient = _episode(config=apply_vision_profile(Agent3DConfig(), "efficient",
                                                              skip_unchanged=True))
        _r3, dense = _episode(config=apply_vision_profile(Agent3DConfig(), "dense"))
        self.assertLess(efficient.metrics.analyses, phase5.metrics.analyses)
        self.assertGreater(dense.metrics.analyses, phase5.metrics.analyses)

    def test_rates_follow_the_state(self) -> None:
        cfg = Agent3DConfig()
        counts: dict[str, list[int]] = {}

        def hook(runner: Any, result: Any) -> None:
            agent = runner.controller
            counts.setdefault(str(agent.state), [0, 0])
            counts[str(agent.state)][0] += 1
            counts[str(agent.state)][1] += int(result.observation is not None)

        _episode(config=cfg, hook=hook, ticks=1800)
        verify = counts.get(Agent3DState.VERIFYING_OPEN.value, [0, 0])
        travel = counts.get(Agent3DState.NAVIGATING.value, [0, 0])
        if verify[0] and travel[0]:
            self.assertGreater(verify[1] / verify[0], travel[1] / travel[0])

    def test_perception_continues_after_a_profile_change(self) -> None:
        from chest_hunter.runtime.operations import OperationsControl

        control = OperationsControl()
        seen: dict[str, int] = {}

        def hook(runner: Any, _result: Any) -> None:
            agent = runner.controller
            if agent.decisions == 200:
                seen["before"] = agent.metrics.analyses
                control.select_vision_profile("efficient")
                control.apply_to_runner(runner)
                seen["rate"] = int(agent.cfg.rate_travel)

        runner, res = _episode(config=Agent3DConfig(), hook=hook)
        self.assertEqual(seen["rate"], int(VISION_PROFILES["efficient"]["rate_travel"]))
        self.assertGreater(res.metrics.analyses, seen["before"])
        self.assertIn(res.outcome, ("completed", "tick_limit"))
        self.assertTrue(any("vision profile" in m for m in control.messages))


class UnchangedFrames(unittest.TestCase):
    def _agent(self, state: Agent3DState) -> Any:
        from chest_hunter.agent.controller_3d import VisualAgent3D
        from chest_hunter.localization.world_model_3d import SpawnPrior3D
        from chest_hunter.unreal_bridge.action_adapter import ActionCommand

        agent = VisualAgent3D.__new__(VisualAgent3D)
        agent.cfg = Agent3DConfig(skip_unchanged_frames=True)
        agent.state = state
        agent.now = 10.0
        agent._last_analysis = 9.9
        agent._maneuver_until = -1.0
        agent.action = ActionCommand()
        agent.telemetry = SimpleNamespace(odometry=SimpleNamespace(position=(1.0, 2.0, 0.0)),
                                          camera=SimpleNamespace(yaw_deg=30.0, pitch_deg=-10.0))
        agent._analysed_pose = (1.0, 2.0, 0.0, 30.0, -10.0)
        del SpawnPrior3D
        return agent

    def test_a_still_view_is_not_analysed_again(self) -> None:
        agent = self._agent(Agent3DState.SCANNING)
        self.assertTrue(agent._unchanged_view())
        agent.action.look_yaw = 0.3                     # the camera will move
        self.assertFalse(agent._unchanged_view())

    def test_never_skipped_while_interacting_or_verifying(self) -> None:
        for state in (Agent3DState.ALIGNING, Agent3DState.INTERACTING,
                      Agent3DState.VERIFYING_OPEN):
            self.assertFalse(self._agent(state)._unchanged_view(), state)

    def test_at_least_one_frame_per_interval(self) -> None:
        agent = self._agent(Agent3DState.SCANNING)
        agent.now = agent._last_analysis + agent.cfg.max_frame_skip_s
        self.assertFalse(agent._unchanged_view())

    def test_a_moved_agent_is_analysed(self) -> None:
        agent = self._agent(Agent3DState.OBSERVING)
        agent._analysed_pose = (1.5, 2.0, 0.0, 30.0, -10.0)
        self.assertFalse(agent._unchanged_view())


class NavigationCache(unittest.TestCase):
    class Port:
        def __init__(self) -> None:
            self.calls = 0

        def nav_query(self, start: Any, goal: Any, *, goal_tolerance: float = 0.5) -> NavResult:
            self.calls += 1
            return NavResult("ok", (start, goal), 5.0)

    def test_repeated_queries_use_the_cache(self) -> None:
        port = self.Port()
        nav = NavMeshNavigator(port)
        a, b = (0.0, 0.0, 0.0), (5.0, 0.0, 0.0)
        for _ in range(5):
            self.assertIsNotNone(nav.plan(a, b, tolerance=1.0))
        self.assertEqual(port.calls, 1)
        self.assertEqual(nav.cache_hits, 4)
        nav.clear_cache()                               # the supervisor's replan clears it
        nav.plan(a, b, tolerance=1.0)
        self.assertEqual(port.calls, 2)

    def test_the_cache_is_bounded(self) -> None:
        port = self.Port()
        nav = NavMeshNavigator(port)
        for k in range(1000):
            nav.plan((0.0, 0.0, 0.0), (float(k), 0.0, 0.0))
        self.assertLessEqual(len(nav._cache), 257)


class Perception(unittest.TestCase):
    def test_the_detector_gives_identical_results_with_the_shared_segmentation(self) -> None:
        from chest_hunter.agent.controller_3d import _hsv
        from chest_hunter.agent.runner_3d import _templates
        from chest_hunter.vision.chest_detector_3d import ChestDetector3D, segment_classes

        frames: list[np.ndarray] = []

        def hook(_runner: Any, result: Any) -> None:
            if result.observation is not None and len(frames) < 40:
                frames.append(np.array(result.observation.frame))

        _episode(config=Agent3DConfig(), hook=hook, ticks=600)
        detector = ChestDetector3D(_templates(settings()), method="hybrid")
        for k, frame in enumerate(frames):
            own = detector.detect(frame, float(k))
            shared = detector.detect(frame, float(k), classes=segment_classes(_hsv(frame)))
            self.assertEqual([(d.category, d.bbox, round(d.confidence, 9)) for d in own],
                             [(d.category, d.bbox, round(d.confidence, 9)) for d in shared])

    def test_tracking_memory_is_bounded(self) -> None:
        from chest_hunter.vision.chest_detector_3d import Detection3D
        from chest_hunter.vision.depth_estimation import RangeEstimate
        from chest_hunter.vision.tracker_3d import Tracker3D

        tracker = Tracker3D(max_tracks=5)
        for k in range(50):
            det = Detection3D("chest", (0, 0, 10, 10), 0.9, float(k), "test")
            est = RangeEstimate((10.0 * k, 0.0, 0.0), 5.0, 0.3, "size", 0.0, 5.0)
            tracker.update([(det, est)], float(k), lambda _t: False)
            self.assertLessEqual(len(tracker.tracks), 5)
            self.assertLessEqual(len(tracker._tracks), 5)
        self.assertGreater(tracker.discard_stale(100.0, 5.0), 0)
        self.assertEqual(tracker.tracks, [])


class InferenceSessions(unittest.TestCase):
    def test_the_benchmark_loads_each_model_once_per_process(self) -> None:
        from chest_hunter.analytics import phase6_benchmark

        made: list[str] = []

        class Model:
            def __init__(self, path: str) -> None:
                made.append(path)

        phase6_benchmark._POLICIES.clear()
        first = phase6_benchmark._policy_cache("m", Model)
        second = phase6_benchmark._policy_cache("m", Model)
        self.assertIs(first, second)
        self.assertEqual(made, ["m"])
        phase6_benchmark._POLICIES.clear()

    def test_one_onnx_session_serves_every_prediction(self) -> None:
        from chest_hunter.learning.learned_policy import LearnedPolicy
        from tests.phase5_helpers import onnx_available, tiny_model

        if not onnx_available():
            self.skipTest("onnxruntime is not installed")
        policy = LearnedPolicy(tiny_model(export=True), backend="onnxruntime")
        session = policy._session
        self.assertIsNotNone(session)
        frame = np.zeros((180, 320, 3), np.uint8)
        for _ in range(5):
            policy.predict(frame, None)
        self.assertIs(policy._session, session)               # created once, reused
        self.assertEqual(policy.calls, 5)


if __name__ == "__main__":
    unittest.main()
