"""Phase 8 end-to-end control-loop trace (one recorded research episode).

Runs one complete episode through the production path - environment reset over the bridge,
agent initialisation, camera observations, chest recognition, target selection,
navigation, approach, interaction, verification, database persistence, next target,
termination - into a real SQLite file, and records the evidence the audit asks for:

* episode id, seed, map, agent version, policy version;
* every state transition with its simulated time and observation timestamp;
* every target selection, every movement action (summarised per decision), every
  interaction attempt (interact held) and every credited opening;
* the simulator's own opening log (evaluation side) to cross-check the credited openings;
* the database rows the episode left (session, run, observations);
* a short video of the agent's camera with the state and target overlaid.

Each loop stage is then marked as exercised (with the first decision it happened at) or not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

STAGES = ("environment_reset", "agent_initialisation", "camera_observation",
          "chest_recognition", "target_selection", "navigation", "chest_approach",
          "interaction", "visual_confirmation", "database_persistence",
          "next_target_selection", "episode_termination")


def run_trace(settings: Any, out: Path, *, map_id: str = "p7_mixed_test_003", seed: int = 4,
              max_ticks: int = 3600, video: bool = True, model_dir: Path | None = None
              ) -> dict[str, Any]:
    import cv2

    from chest_hunter.agent.controller_3d import Agent3DConfig
    from chest_hunter.agent.runner_3d import Runner3D
    from chest_hunter.database import DatabaseManager
    from chest_hunter.exploration.active import ExplorationConfig
    from chest_hunter.runtime.configurations import AGENT_VERSION
    from chest_hunter.runtime.supervisor import SupervisorConfig

    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "trace.db"
    db_path.unlink(missing_ok=True)
    db = DatabaseManager(db_path)
    db.initialize()
    policy = None
    if model_dir is not None:
        from chest_hunter.learning.learned_policy import LearnedPolicy

        policy = LearnedPolicy(model_dir)
    transitions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    credits: list[dict[str, Any]] = []
    decisions = {"n": 0, "moving": 0, "looking": 0}
    first: dict[str, int] = {}
    writer: Any = None
    state = {"last_state": None, "last_goal": None, "opened": 0, "interacting": False,
             "obs_time": None, "detections": 0}

    def mark(stage: str) -> None:
        first.setdefault(stage, decisions["n"])

    def hook(runner: Any, result: Any) -> None:
        nonlocal writer
        agent = getattr(runner.controller, "agent", runner.controller)
        decisions["n"] += 1
        mark("environment_reset")
        mark("agent_initialisation")
        t = round(float(agent.now), 3)
        if result.observation is not None:
            state["obs_time"] = result.observation.sim_time
            mark("camera_observation")
        if agent.last_detections:
            state["detections"] += len(agent.last_detections)
            mark("chest_recognition")
        st = agent.state.name
        if st != state["last_state"]:
            transitions.append({"t": t, "from": state["last_state"], "to": st,
                                "observation_time": state["obs_time"]})
            state["last_state"] = st
            if st == "NAVIGATING":
                mark("navigation")
            if st == "APPROACHING":
                mark("chest_approach")
            if st == "VERIFYING_OPEN":
                mark("visual_confirmation")
        goal = agent.goal
        key = (goal.kind, goal.location_id, tuple(round(v, 1) for v in goal.position)) \
            if goal is not None else None
        if key != state["last_goal"] and key is not None:
            targets.append({"t": t, "kind": goal.kind, "location_id": goal.location_id,
                            "position": [round(v, 2) for v in goal.position]})
            mark("target_selection" if len(targets) == 1 else "next_target_selection")
        state["last_goal"] = key
        a = agent.action
        if abs(a.move_forward) > 1e-6 or abs(a.move_right) > 1e-6:
            decisions["moving"] += 1
        if abs(a.look_yaw) > 1e-6 or abs(a.look_pitch) > 1e-6:
            decisions["looking"] += 1
        if a.interact and not state["interacting"]:
            interactions.append({"t": t, "target": key})
            mark("interaction")
        state["interacting"] = bool(a.interact)
        if agent.metrics.chests_opened > state["opened"]:
            state["opened"] = agent.metrics.chests_opened
            credits.append({"t": t, "target": key})
        if video and result.observation is not None and decisions["n"] % 3 == 0:
            frame = result.observation.frame.copy()
            cv2.putText(frame, f"{t:6.1f}s {st}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            cv2.putText(frame, f"opened {state['opened']}", (4, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out / "episode.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
            writer.write(frame)

    try:
        runner = Runner3D.from_settings(
            db, settings, mode="hybrid", seed=seed, map_id=map_id, backend="reference",
            collect_truth=True, evaluate=True, max_ticks=max_ticks, width=320, height=180,
            config=Agent3DConfig(smart_recovery=False), exploration=ExplorationConfig(),
            supervisor=SupervisorConfig(), on_decision=hook,
            policy="hybrid" if policy is not None else "deterministic", learned_policy=policy)
        result = runner.run()
        episode_id = runner.client.episode_id if runner.client is not None else None
        env = runner.server.session.env
        sim_openings = [{"t": round(float(e["sim_time"]), 3), "spawn_index": e["spawn_index"]}
                        for e in env.opened_log]
        rows = {
            "sessions": [dict(r) for r in db.query_all(
                "SELECT id, map_id, status, started_at, ended_at, notes "
                "FROM simulation_sessions")],
            "runs": [dict(r) for r in db.query_all("SELECT * FROM simulation_runs")],
            "observations": db.query_one(
                "SELECT COUNT(*) AS n FROM chest_observations")["n"],
        }
    finally:
        db.close()
        if writer is not None:
            writer.release()
    if rows["sessions"]:
        first.setdefault("database_persistence", decisions["n"])
    if result.outcome in ("completed", "tick_limit"):
        first.setdefault("episode_termination", decisions["n"])
    m = result.metrics
    return {
        "episode_id": episode_id, "seed": seed, "map_id": map_id,
        "agent_version": AGENT_VERSION,
        "policy": "hybrid" if policy is not None else "deterministic",
        "policy_version": getattr(policy, "version", None),
        "environment": "Python reference arena via the localhost bridge (NOT Unreal)",
        "outcome": result.outcome, "end_reason": result.end_reason,
        "sim_seconds": round(result.sim_seconds, 2),
        "stages": {s: {"exercised": s in first, "first_decision": first.get(s)}
                   for s in STAGES},
        "decisions": decisions, "transitions": transitions, "targets": targets,
        "interaction_attempts": interactions, "credited_openings": credits,
        "simulator_openings": sim_openings,
        "credited_vs_simulator": {"credited": len(credits), "simulator": len(sim_openings)},
        "database": rows, "detections": state["detections"],
        "final_metrics": {"chests_opened": m.chests_opened,
                          "opened_truth": result.truth.get("opened_truth"),
                          "travel_distance_m": round(m.travel_distance, 1),
                          "recoveries": m.recoveries, "analyses": m.analyses,
                          "coverage": result.truth.get("true_coverage"),
                          "vision": result.vision},
        "video": str(out / "episode.mp4") if video and writer is not None else None,
    }
