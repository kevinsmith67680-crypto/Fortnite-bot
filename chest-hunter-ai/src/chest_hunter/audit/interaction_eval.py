"""Phase 8 chest-interaction audit.

Two levels, both on the Python reference arena:

1. **Environment interaction rules** (scripted, evaluation side): the character is placed
   next to a chest and the simulator's interaction rule is exercised directly - in range and
   aligned, too far, misaligned, chest absent, already opened, interrupted hold, a duplicate
   (over-long) hold. Each case states the expected outcome and checks the simulator's chest
   state and opening log. (The Unreal research server implements the same rule in C++; that
   implementation has not been executed.)
2. **Agent verification against ground truth**: complete agent episodes (spawn-point mode,
   research arena) in several scenarios - clean, failed interactions, a corrupted view while
   verifying, a lost frame stream while verifying, a dropped bridge connection while
   interacting. After each episode the spawn points the agent credited as opened are compared
   with the simulator's opened chests: a *false success* is a credited opening that did not
   happen, a *false failure* an opening that happened but was not credited.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

SCENARIOS: dict[str, tuple[str, float, float, str | None] | None] = {
    # name: (fault kind, duration s, intensity, trigger state) | None (no fault)
    "clean": None,
    "failed_interactions": ("interaction_failure", 600.0, 3.0, None),
    "corrupted_view_while_verifying": ("corrupted_frame", 1.5, 0.3, "VERIFYING_OPEN"),
    "frames_lost_while_verifying": ("frame_loss", 1.5, 1.0, "VERIFYING_OPEN"),
    "connection_dropped_while_interacting": ("bridge_disconnect", 0.0, 1.0, "INTERACTING"),
}


def environment_rules(map_id: str = "research_arena_3d", seed: int = 3) -> list[dict[str, Any]]:
    from chest_hunter.simulation3d.environment3d import Environment3D

    def fresh() -> tuple[Any, Any, Any]:
        env = Environment3D(map_id, seed=seed, episode=0)
        chest = next(c for c in env.chests if c.present and c.z < 0.5)
        absent = next((c for c in env.chests if not c.present and c.z < 0.5), None)
        return env, chest, absent

    def place(env: Any, chest: Any, dist: float, facing_offset: float = 0.0) -> None:
        # stand on the chest's front side, facing it
        a = math.radians(chest.yaw_deg)
        ch = env.character
        ch.x, ch.y, ch.z = chest.x + dist * math.cos(a), chest.y + dist * math.sin(a), chest.z
        bearing = math.degrees(math.atan2(chest.y - ch.y, chest.x - ch.x))
        ch.body_yaw = ch.cam_yaw = (bearing + facing_offset) % 360.0
        ch.cam_pitch = -20.0
        ch.vx = ch.vy = ch.vz = 0.0

    def hold(env: Any, seconds: float, interact: bool = True) -> None:
        ticks = int(round(seconds * env.settings.tick_rate))
        env.step({"move_forward": 0.0, "move_right": 0.0, "look_yaw": 0.0,
                  "look_pitch": 0.0, "interact": interact}, ticks)

    cases: list[dict[str, Any]] = []

    def case(name: str, expected_open: bool, env: Any, chest: Any, *, openings: int) -> None:
        ok = chest.opened == expected_open and len(env.opened_log) == openings
        cases.append({"case": name, "expected_opened": expected_open,
                      "chest_opened": chest.opened, "opening_events": len(env.opened_log),
                      "expected_events": openings, "pass": ok})

    env, chest, _ = fresh()
    place(env, chest, 1.2)
    hold(env, 2.2)
    case("in range, aligned, 2.2 s hold", True, env, chest, openings=1)

    env, chest, _ = fresh()
    place(env, chest, 2.6)
    hold(env, 2.5)
    case("too far (2.6 m > 1.8 m radius)", False, env, chest, openings=0)

    env, chest, _ = fresh()
    place(env, chest, 1.2, facing_offset=100.0)
    hold(env, 2.5)
    case("camera misaligned (100 deg)", False, env, chest, openings=0)

    env, chest, absent = fresh()
    if absent is not None:
        place(env, absent, 1.2)
        hold(env, 2.5)
        cases.append({"case": "chest absent at the spawn point",
                      "expected_opened": False, "chest_opened": absent.opened,
                      "opening_events": len(env.opened_log), "expected_events": 0,
                      "pass": not absent.opened and not env.opened_log})

    env, chest, _ = fresh()
    place(env, chest, 1.2)
    hold(env, 2.2)
    hold(env, 0.2, interact=False)
    hold(env, 2.5)
    case("already opened, held again", True, env, chest, openings=1)

    env, chest, _ = fresh()
    place(env, chest, 1.2)
    hold(env, 1.0)
    hold(env, 0.3, interact=False)
    hold(env, 1.0)
    case("interrupted hold (1.0 s + 1.0 s)", False, env, chest, openings=0)

    env, chest, _ = fresh()
    place(env, chest, 1.2)
    hold(env, 6.0)
    case("duplicate command (one 6 s hold)", True, env, chest, openings=1)
    return cases


@dataclass(frozen=True, slots=True)
class InteractionJob:
    scenario: str
    episode: int
    seed: int
    settings_path: str
    max_ticks: int


def run_interaction_job(job: InteractionJob) -> dict[str, Any]:
    from chest_hunter.agent.runner_3d import Runner3D
    from chest_hunter.config import load_config
    from chest_hunter.database import DatabaseManager
    from chest_hunter.localization.world_model_3d import PriorStatus3D
    from chest_hunter.runtime.configurations import PHASE6_SETUPS, agent_config, supervisor_config
    from chest_hunter.runtime.faults import FaultInjector, FaultPlan
    from chest_hunter.scenarios import seed_arena_3d

    settings = load_config(job.settings_path)
    setup = PHASE6_SETUPS["phase6_deterministic_supervised"]
    spec = SCENARIOS[job.scenario]
    injector = FaultInjector(FaultPlan(spec[0], 0.0, spec[1], spec[2], spec[3]),
                             seed=job.episode) if spec else None
    db = DatabaseManager(":memory:")
    db.initialize()
    seed_arena_3d(db)
    try:
        runner = Runner3D.from_settings(
            db, settings, mode="hybrid", seed=job.seed, episode=job.episode,
            backend="reference", collect_truth=True, max_ticks=job.max_ticks, width=320,
            height=180, config=agent_config(settings, setup),
            supervisor=supervisor_config(settings, setup), faults=injector)
        result = runner.run()
    finally:
        db.close()
    agent = getattr(runner.controller, "agent", runner.controller)
    credited = {p.location_id for p in agent.world.priors.values()
                if p.status is PriorStatus3D.OPENED}
    truth = set(result.truth.get("location_ids_opened", []))
    m = result.metrics
    a = injector.assess(result) if injector is not None else {}
    return {"scenario": job.scenario, "episode": job.episode, "outcome": result.outcome,
            "interaction_attempts": m.interaction_attempts,
            "failed_interactions": m.failed_interactions,
            "credited": len(credited), "truth_opened": len(truth),
            "false_success": len(credited - truth), "false_failure": len(truth - credited),
            "credited_by_notification": m.credited_by_notification,
            "fault_applied": a.get("applied"), "fault_recovered": a.get("recovered")}


def agent_crosscheck(settings: Any, *, episodes: int = 6, seed: int = 21,
                     max_ticks: int = 3600, workers: int = 1,
                     scenarios: Sequence[str] = tuple(SCENARIOS)) -> dict[str, Any]:
    jobs = [InteractionJob(s, e, seed, str(settings.config_path), max_ticks)
            for s in scenarios for e in range(episodes)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(run_interaction_job, jobs))
    else:
        rows = [run_interaction_job(j) for j in jobs]
    per: dict[str, Any] = {}
    for s in scenarios:
        rs = [r for r in rows if r["scenario"] == s]
        credited = sum(r["credited"] for r in rs)
        truth = sum(r["truth_opened"] for r in rs)
        fs, ff = sum(r["false_success"] for r in rs), sum(r["false_failure"] for r in rs)
        attempts = sum(r["interaction_attempts"] for r in rs)
        per[s] = {"episodes": len(rs), "interaction_attempts": attempts,
                  "credited_openings": credited, "true_openings": truth,
                  "false_successes": fs, "false_failures": ff,
                  "false_success_rate": round(fs / credited, 4) if credited else None,
                  "false_failure_rate": round(ff / truth, 4) if truth else None,
                  "interaction_success_rate": round(truth / attempts, 4) if attempts else None,
                  "credited_by_notification": sum(r["credited_by_notification"] for r in rs),
                  "faults_applied": sum(1 for r in rs if r["fault_applied"]),
                  "failed_episodes": sum(1 for r in rs if r["outcome"] == "failed")}
    return {"per_scenario": per, "rows": rows, "seed": seed, "max_ticks": max_ticks,
            "configuration": "phase6_deterministic_supervised, research_arena_3d, spawn points",
            "environment": "Python reference arena (NOT Unreal)"}
