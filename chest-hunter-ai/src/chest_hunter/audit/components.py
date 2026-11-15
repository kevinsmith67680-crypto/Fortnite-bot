"""Phase 8 component inventory, readiness gating and scorecard weights.

Statuses follow the audit definitions:

* VERIFIED - executed successfully under realistic test conditions;
* PARTIALLY VERIFIED - some functionality tested, important limitations remain;
* MOCK ONLY - works only against mocks, fixtures or simulated interfaces;
* UNVERIFIED - implementation exists but has not been meaningfully tested;
* BROKEN - reproducible failure or missing essential functionality.

"Realistic" for the 2D phases (1-3) is the project's own 2D simulator - their intended
environment. For every 3D component the intended environment is the Unreal research build;
the Python reference arena is a protocol-identical *mock* of it, so a 3D component that has
only run there is at best PARTIALLY VERIFIED (agent-side logic exercised for real, on
simulated frames and physics), and anything that *is* the Unreal side is UNVERIFIED.
Evidence names refer to files in the audit output directory (``benchmark_results/phase8``)
or to test modules.
"""

from __future__ import annotations

from typing import Any

COMPONENTS: list[dict[str, Any]] = [
    {"id": 1, "name": "SQLite data infrastructure", "status": "VERIFIED",
     "evidence": ["database_integrity.json", "tests/test_database.py"],
     "observed": "schema 6, triggers and constraints refuse invalid values, migration from "
                 "v1 keeps data, two concurrent writer processes, integrity_check ok",
     "limitations": "single-file SQLite; no network database",
     "fixes": []},
    {"id": 2, "name": "Chest-location management", "status": "VERIFIED",
     "evidence": ["tests/test_chest_repository.py", "database_integrity.json"],
     "observed": "repository validation (probability, NaN, empty map) and duplicate refusal",
     "limitations": "", "fixes": []},
    {"id": 3, "name": "Simulation sessions", "status": "PARTIALLY VERIFIED",
     "evidence": ["database_integrity.json", "tests/test_session_repository.py"],
     "observed": "sessions finish with ended_at; interrupted episodes end aborted; finished "
                 "sessions immutable",
     "limitations": "active-exploration openings are not persisted (F12); endurance resume "
                    "after a lost result line failed (F9, fixed)",
     "fixes": ["F9 (fixed)", "F12"]},
    {"id": 4, "name": "2D navigation (A*)", "status": "VERIFIED",
     "evidence": ["tests/test_astar.py", "tests/test_route_planner.py"],
     "observed": "optimality against Dijkstra, blocked goals, determinism", "limitations": "",
     "fixes": []},
    {"id": 5, "name": "Route optimisation (2D)", "status": "VERIFIED",
     "evidence": ["tests/test_benchmark.py", "README section 13"],
     "observed": "paired 2D benchmark (Phase 2)", "limitations": "2D only", "fixes": []},
    {"id": 6, "name": "2D autonomous controller", "status": "VERIFIED",
     "evidence": ["tests/test_agent.py"], "observed": "every FSM path in the 2D simulator",
     "limitations": "2D research simulator only", "fixes": []},
    {"id": 7, "name": "Visual chest detection (3D)", "status": "PARTIALLY VERIFIED",
     "evidence": ["vision_metrics.json", "vision_failures/"],
     "observed": "labelled dataset at 320x180 / 480x270 / 640x360 rendered by the reference "
                 "arena",
     "limitations": "frames from the Python renderer only (Unreal frames never seen); "
                    "opened/unopened confusion; chest-appearance shift defeats the colour "
                    "segmentation",
     "fixes": ["F11 (verification side)"]},
    {"id": 8, "name": "Object tracking (3D)", "status": "PARTIALLY VERIFIED",
     "evidence": ["tests/test_vision_3d.py", "integration_results.json (loop trace)"],
     "observed": "bounded tracks, stale expiry, re-acquisition", "limitations": "mock frames",
     "fixes": []},
    {"id": 9, "name": "Localisation (3D)", "status": "PARTIALLY VERIFIED",
     "evidence": ["navigation_metrics.json (localisation)", "benchmark_results/robustness"],
     "observed": "odometry-based (no visual correction); pose error from the reference "
                 "arena's drift model; odometry jumps detected and relocalised",
     "limitations": "the drift model is the simulator's; Unreal odometry untested; no visual "
                    "relocalisation, a persistent offset stays",
     "fixes": []},
    {"id": 10, "name": "Occupancy / coverage estimation", "status": "PARTIALLY VERIFIED",
     "evidence": ["generalisation_results.csv (coverage)", "tests/test_active_exploration.py"],
     "observed": "coverage map from raycasts and the view wedge; true coverage scored on "
                 "the evaluation side",
     "limitations": "coverage within 200 s far below the 85 % target on 90 m maps",
     "fixes": []},
    {"id": 11, "name": "Visual navigation (2D, Phase 3)", "status": "VERIFIED",
     "evidence": ["tests/test_visual_navigation.py", "tests/test_visual_benchmark.py"],
     "observed": "2D visual agent and benchmark", "limitations": "2D only", "fixes": []},
    {"id": 12, "name": "Unreal environment", "status": "UNVERIFIED",
     "evidence": ["integration_results.json (unreal)"],
     "observed": "C++ sources and config only: no binaries, no map asset, never compiled",
     "limitations": "nothing executed", "fixes": ["build and run on an Unreal machine"]},
    {"id": 13, "name": "Unreal communication (bridge)", "status": "MOCK ONLY",
     "evidence": ["integration_results.json (bridge_security)", "tests/test_unreal_bridge.py",
                  "tests/test_unreal_sources.py"],
     "observed": "Python client fully exercised against the protocol-identical Python "
                 "server; C++ server constants and checks match statically",
     "limitations": "the C++ server never ran", "fixes": []},
    {"id": 14, "name": "Third-person camera capture", "status": "MOCK ONLY",
     "evidence": ["tests/test_renderer.py", "loop_trace/episode.mp4"],
     "observed": "Python software renderer frames", "limitations": "Unreal SceneCapture "
                 "never ran", "fixes": []},
    {"id": 15, "name": "3D target localisation", "status": "PARTIALLY VERIFIED",
     "evidence": ["vision_metrics.json (distance/bearing errors)"],
     "observed": "distance error from the agent's estimator on labelled frames",
     "limitations": "mock frames", "fixes": []},
    {"id": 16, "name": "3D navigation", "status": "PARTIALLY VERIFIED",
     "evidence": ["navigation_metrics.json"],
     "observed": "real agent sent to goals by terrain type on held-out maps",
     "limitations": "the reference arena's NavMesh-style grid stands in for Unreal's "
                    "Recast NavMesh", "fixes": []},
    {"id": 17, "name": "Chest interaction", "status": "PARTIALLY VERIFIED",
     "evidence": ["interaction_metrics.json"],
     "observed": "simulator interaction rules (range, facing, absent, opened, interrupted, "
                 "duplicate) all as specified; agent interaction attempts",
     "limitations": "Unreal interaction validation never ran", "fixes": []},
    {"id": 18, "name": "Visual interaction verification", "status": "BROKEN",
     "evidence": ["interaction_metrics.json (false failures)", "failures.json F11"],
     "observed": "real openings judged closed in every look; the agent re-aligns on an "
                 "opened chest and marks it failed",
     "limitations": "", "fixes": ["F11"]},
    {"id": 19, "name": "Imitation-learning pipeline", "status": "PARTIALLY VERIFIED",
     "evidence": ["tests/test_learned_policy.py", "training/datasets"],
     "observed": "dataset, splits, NumPy training, ONNX export and parity tests",
     "limitations": "synthetic demonstrations only (no human data)", "fixes": []},
    {"id": 20, "name": "Learned-policy inference", "status": "PARTIALLY VERIFIED",
     "evidence": ["policy_comparison.csv"], "observed": "ONNX CPU inference in episodes",
     "limitations": "see the policy comparison for its value", "fixes": []},
    {"id": 21, "name": "Adaptive exploration", "status": "PARTIALLY VERIFIED",
     "evidence": ["planner_comparison.csv", "benchmark_results/phase7/strategies"],
     "observed": "Phase 5/6 planners and Phase 7 frontier strategies on identical seeds",
     "limitations": "mock environment", "fixes": []},
    {"id": 22, "name": "Reliability supervisor", "status": "PARTIALLY VERIFIED",
     "evidence": ["benchmark_results/robustness", "tests/test_supervisor.py"],
     "observed": "ten injected fault kinds, detection and bounded recovery",
     "limitations": "mock environment; Unreal timeouts never exercised", "fixes": []},
    {"id": 23, "name": "Endurance execution", "status": "PARTIALLY VERIFIED",
     "evidence": ["endurance_results.csv"],
     "observed": "100 / 500 / 1000-episode runs against one long-lived reference server",
     "limitations": "short episodes (1500 ticks); Unreal never used", "fixes": ["F9 (fixed)"]},
    {"id": 24, "name": "Generalisation testing", "status": "PARTIALLY VERIFIED",
     "evidence": ["generalisation_results.csv"],
     "observed": "8 evaluation categories, held-out seeds and maps, paired gaps with CIs",
     "limitations": "procedural maps of one generator; mock environment", "fixes": []},
    {"id": 25, "name": "Metrics and reporting", "status": "PARTIALLY VERIFIED",
     "evidence": ["audit_summary.json", "tests/test_phase7_benchmark.py"],
     "observed": "ground-truth-scored metrics, journals and resume",
     "limitations": "a recovery-success metric read near zero until Phase 7 (fixed); the "
                    "Phase 6 benchmark lost its first run to a 2 h session limit (journal "
                    "added)", "fixes": []},
]

#: scorecard categories and weights (Phase 8 brief, section 21)
WEIGHTS: dict[str, float] = {
    "Core autonomous functionality": 0.20, "Unreal integration": 0.15,
    "Computer vision": 0.10, "Navigation": 0.10, "Chest interaction": 0.10,
    "Reliability": 0.10, "Performance efficiency": 0.10, "Generalisation": 0.05,
    "Data integrity": 0.05, "Testing and documentation": 0.05,
}


def readiness(parts: dict[str, Any]) -> dict[str, Any]:
    """Gating flags from the measured results (scores are judged in the scorecard)."""
    trace = parts.get("trace") or {}
    inter = parts.get("inter") or {}
    database = parts.get("database") or {}
    endurance = parts.get("endurance") or {}
    unreal = parts.get("unreal") or {}
    stages = trace.get("stages") or {}
    loop_ok = bool(stages) and all(s["exercised"] for s in stages.values()) and \
        trace.get("outcome") in ("completed", "tick_limit")
    clean = (inter.get("agent_crosscheck") or {}).get("per_scenario", {}).get("clean", {})
    rules = inter.get("environment_rules") or []
    interactions_ok = bool(rules) and all(r["pass"] for r in rules) and \
        clean.get("false_success_rate") in (0, 0.0)
    ffr = clean.get("false_failure_rate")
    verification_reliable = ffr is not None and ffr <= 0.05
    db_checks = {c["check"]: c["pass"] for c in database.get("checks", [])}
    critical_db = ("no duplicated openings", "no recorded opening that did not happen",
                   "integrity_check ok", "no foreign-key violations")
    db_ok = bool(db_checks) and all(db_checks.get(c, False) for c in critical_db)
    runs = endurance.get("runs") or []
    leaks = [r for r in runs if r.get("memory_bounded") is False]
    unreal_ok = bool(unreal) and unreal.get("unreal_engine", {}).get("installed") and \
        bool(unreal.get("maps_present"))
    gates = {
        "actual Unreal integration verified": bool(unreal_ok),
        "autonomous control loop completes": loop_ok,
        "chest interactions verified (simulator rules + no false successes)": interactions_ok,
        "visual verification reliable (false failures <= 5 %)": verification_reliable,
        "no critical data-integrity defects": db_ok,
        "no severe resource leaks": bool(runs) and not leaks,
        "not entirely dependent on mocked components": bool(unreal_ok),
    }
    if not unreal_ok:
        verdict = "UNVERIFIED"
        reason = ("the intended Unreal environment has never been built or run; every "
                  "end-to-end result comes from the Python mock of it")
    elif all(gates.values()):
        verdict = "READY FOR CONTROLLED RESEARCH USE"
        reason = "all gates passed"
    else:
        verdict = "NOT READY"
        reason = "failed gates: " + ", ".join(k for k, v in gates.items() if not v)
    return {"gates": gates, "verdict": verdict, "reason": reason, "weights": WEIGHTS}
