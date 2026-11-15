"""Phase 8 ground-truth isolation audit.

The visual agent may use: its camera frames, its odometry and camera rig telemetry, the game
rules, the in-game events (e.g. the "chest opened" toast), and the navigation service.
It must not use hidden chest presence, unseen chest positions, evaluation labels or other
privileged simulator state. Three independent checks:

1. **Static** - the agent-facing port exposes only the permitted calls, and no module of the
   agent's decision path imports the simulator or names the privileged requests.
2. **Runtime call audit** - the bridge client's privileged requests (oracle state, evaluation
   labels) are wrapped during real episodes; every call is recorded with the calling modules
   and the agent's state. Allowed callers: the runner's end-of-episode truth collection
   (after the agent loop) and its evaluator (scoring only, never passed to the agent).
3. **Counterfactual** - the same episode is played twice with one difference the agent must
   not be able to sense until the chest comes into view: a far chest's hidden presence is
   flipped. The two action sequences must be identical up to the first frame in which that
   chest is visible. The same comparison with evaluation labels requested vs not requested
   must give identical sequences for the whole episode.
"""

from __future__ import annotations

import ast
import inspect
import math
import traceback
from pathlib import Path
from typing import Any

AGENT_PATH_MODULES = ("agent/controller_3d.py", "localization/world_model_3d.py",
                      "vision/chest_detector_3d.py", "vision/tracker_3d.py",
                      "vision/depth_estimation.py", "navigation/navigation_3d.py",
                      "exploration/active.py", "exploration/advanced_planner.py",
                      "exploration/adaptive_planner.py", "learning/policy_selector.py",
                      "learning/learned_policy.py", "runtime/supervisor.py")
FORBIDDEN_IMPORTS = ("chest_hunter.simulation3d.environment3d",
                     "chest_hunter.unreal_bridge.mock_server")
PERMITTED_PORT = {"rules", "observe", "step", "nav_query", "nav_raycast"}


def static_checks(src: Path) -> list[dict[str, Any]]:
    from chest_hunter.unreal_bridge.client import AgentPort

    out = []
    public = {n for n, _ in inspect.getmembers(AgentPort) if not n.startswith("_")}
    out.append({"check": "agent port exposes only permitted calls",
                "pass": public <= PERMITTED_PORT, "detail": sorted(public)})
    for rel in AGENT_PATH_MODULES:
        path = src / "chest_hunter" / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        bad = sorted(i for i in imports if any(i.startswith(f) for f in FORBIDDEN_IMPORTS))
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        privileged = sorted(names & {"oracle_state", "evaluation_labels", "last_labels",
                                     "GET_ORACLE_STATE", "GET_EVALUATION"})
        out.append({"check": f"{rel}: no simulator imports, no privileged requests",
                    "pass": not bad and not privileged,
                    "detail": {"forbidden_imports": bad, "privileged_names": privileged}})
    return out


def _actions_and_frames(settings: Any, *, map_id: str, seed: int, max_ticks: int,
                        evaluate: bool, flip: int | None = None, exploration: bool = True,
                        audit_calls: list[dict[str, Any]] | None = None
                        ) -> tuple[list[dict[str, Any]], list[list[int]], Any]:
    """Play one episode; return the action of every decision, the spawn indices visible
    in every frame (from the evaluation channel, requested by this harness only after the
    agent has acted), and the result."""
    from chest_hunter.agent.controller_3d import Agent3DConfig
    from chest_hunter.agent.runner_3d import Runner3D
    from chest_hunter.database import DatabaseManager
    from chest_hunter.exploration.active import ExplorationConfig
    from chest_hunter.simulation3d import environment3d as env_mod
    from chest_hunter.simulation3d.arena import chest_boxes
    from chest_hunter.unreal_bridge import client as client_mod

    original_reset = env_mod.Environment3D.reset

    def reset(self: Any) -> None:
        original_reset(self)
        if flip is not None:
            c = self.chests[flip]
            c.present = not c.present
            c.boxes = tuple(chest_boxes(c.x, c.y, c.z, c.yaw_deg, opened=False)) \
                if c.present else ()
            self.nav_blocked = self.nav.dynamic_cells(self._dynamic_boxes())

    wrapped: dict[str, Any] = {}

    def wrap(name: str) -> None:
        original = getattr(client_mod.BridgeClient, name)
        wrapped[name] = original

        def call(self: Any, *a: Any, **k: Any) -> Any:
            if audit_calls is not None:
                stack = [f.filename.split("chest_hunter/")[-1] for f in
                         traceback.extract_stack()[:-1] if "chest_hunter" in f.filename]
                audit_calls.append({"call": name, "callers": stack[-3:]})
            return original(self, *a, **k)

        setattr(client_mod.BridgeClient, name, call)

    env_mod.Environment3D.reset = reset                   # type: ignore[method-assign]
    for name in ("oracle_state", "evaluation_labels"):
        wrap(name)
    actions: list[dict[str, Any]] = []
    visible: list[list[int]] = []

    def hook(runner: Any, result: Any) -> None:
        agent = getattr(runner.controller, "agent", runner.controller)
        actions.append({"t": round(float(agent.now), 4), "state": agent.state.name,
                        **agent.action.to_dict()})
        env = runner.server.session.env                   # evaluation side, after acting
        labels = env.last_labels if evaluate else None
        if result.observation is not None and evaluate and labels is not None:
            visible.append([c["spawn_index"] for c in labels["chests"]])
        else:
            visible.append([])

    db = DatabaseManager(":memory:")
    db.initialize()
    if not exploration and map_id == "research_arena_3d":
        from chest_hunter.scenarios import seed_arena_3d

        seed_arena_3d(db)
    try:
        runner = Runner3D.from_settings(
            db, settings, mode="hybrid", seed=seed, map_id=map_id, backend="reference",
            collect_truth=True, evaluate=evaluate, max_ticks=max_ticks, width=320, height=180,
            config=Agent3DConfig(smart_recovery=True),
            exploration=ExplorationConfig() if exploration else None, on_decision=hook)
        result = runner.run()
    finally:
        db.close()
        env_mod.Environment3D.reset = original_reset      # type: ignore[method-assign]
        for name, original in wrapped.items():
            setattr(client_mod.BridgeClient, name, original)
    return actions, visible, result


def counterfactual(settings: Any, *, map_id: str = "p7_mixed_test_003", seed: int = 4,
                   max_ticks: int = 2400) -> dict[str, Any]:
    from chest_hunter.simulation3d.environment3d import Environment3D

    env = Environment3D(map_id, seed=seed)
    start = env.character.x, env.character.y
    present = [c for c in env.chests if c.present]
    far = max(present, key=lambda c: math.dist(start, (c.x, c.y)))
    base_actions, base_visible, base = _actions_and_frames(
        settings, map_id=map_id, seed=seed, max_ticks=max_ticks, evaluate=True)
    alt_actions, _, alt = _actions_and_frames(
        settings, map_id=map_id, seed=seed, max_ticks=max_ticks, evaluate=True,
        flip=far.spawn_index)
    first_seen = next((i for i, v in enumerate(base_visible) if far.spawn_index in v), None)
    horizon = first_seen if first_seen is not None else min(len(base_actions),
                                                            len(alt_actions))
    diverged = next((i for i in range(min(horizon, len(alt_actions)))
                     if base_actions[i] != alt_actions[i]), None)
    no_eval_actions, _, _ = _actions_and_frames(
        settings, map_id=map_id, seed=seed, max_ticks=max_ticks, evaluate=False)
    eval_diverged = next((i for i in range(min(len(base_actions), len(no_eval_actions)))
                          if base_actions[i] != no_eval_actions[i]), None)
    return {
        "map_id": map_id, "seed": seed, "flipped_spawn": far.spawn_index,
        "flipped_distance_m": round(math.dist(start, (far.x, far.y)), 1),
        "decisions": len(base_actions),
        "first_decision_with_flipped_chest_in_view": first_seen,
        "first_divergence_before_view": diverged,
        "hidden_presence_isolated": diverged is None,
        "decisions_compared": horizon,
        "evaluation_channel_isolated": eval_diverged is None
        and len(base_actions) == len(no_eval_actions),
        "first_divergence_with_vs_without_labels": eval_diverged,
        "outcomes": [base.outcome, alt.outcome],
    }


def call_audit(settings: Any, *, map_id: str = "research_arena_3d", seed: int = 2,
               max_ticks: int = 900) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    actions, _, _ = _actions_and_frames(settings, map_id=map_id, seed=seed, max_ticks=max_ticks,
                                        evaluate=True, exploration=False, audit_calls=calls)
    allowed = ("agent/runner_3d.py",)
    offenders = [c for c in calls if not any(a in " ".join(c["callers"]) for a in allowed)
                 or any(m in " ".join(c["callers"]) for m in AGENT_PATH_MODULES)]
    by_call: dict[str, int] = {}
    for c in calls:
        by_call[c["call"]] = by_call.get(c["call"], 0) + 1
    return {"privileged_calls": by_call, "decisions": len(actions),
            "callers": sorted({" > ".join(c["callers"]) for c in calls}),
            "calls_from_agent_modules": len(offenders), "pass": not offenders}


def evaluate(settings: Any, src: Path) -> dict[str, Any]:
    static = static_checks(src)
    return {"static": static, "static_pass": all(c["pass"] for c in static),
            "runtime_call_audit": call_audit(settings),
            "counterfactual": counterfactual(settings)}
