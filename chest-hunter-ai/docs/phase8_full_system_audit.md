# Phase 8: full-system audit of chest-hunter-ai

Independent technical audit of Phases 1–7. Every conclusion below rests on inspected code, an
executed test, an observed run or a measured benchmark, with the artifact named.

* **Audit artifacts:** `benchmark_results/phase8/`, produced by
  `python -m chest_hunter audit-system --episodes 40 --held-out-maps --profile-resources
  --verify-database --workers 4`.
* **Reused benchmarks:** `benchmark_results/phase7/` (generalisation),
  `benchmark_results/robustness/`, `benchmark_results/phase7/acceptance/` and
  `benchmark_results/phase6/endurance_*`.
* **Machine:** 4 vCPU Xeon 2.1 GHz, 16 GB RAM, no GPU, Linux. Python 3.11.18 / 3.12.3 /
  3.13.18. **No Unreal Engine installation.**

Where a sentence says "measured", the run took place in the **Python reference arena**. That
arena is the project's protocol-identical mock of the Unreal research server: same bridge
messages and same navigation services, but its own physics and software renderer. It is not
Unreal Engine, and a result from it is not evidence about Unreal.

<<SECTION_A>>

## B. Verified capabilities (demonstrated by execution)

"Verified" for a 3D component means executed in the reference arena. Section C covers what
that does and does not prove.

* **Phases 1–3 (2D research simulator, the intended environment for those phases)**:
  * the SQLite chest database;
  * sessions;
  * A* and route planning;
  * the 2D controller;
  * the Phase 3 visual agent.

  All are covered by the suite (section N) and the earlier paired benchmarks.
* **The complete 3D autonomous loop**, in the reference arena. All twelve stages were
  exercised in one traced episode, recorded on video (`loop_trace/episode.mp4`): reset,
  initialisation, observation, recognition, target selection, navigation, approach,
  interaction, visual confirmation, persistence, next target and termination
  (`integration_results.json`).
* **Ground-truth isolation of the visual agent**: static, runtime and counterfactual checks
  all pass (section F).
* **The bridge's defensive behaviour (Python server)**: 15 of 15 raw-socket checks pass
  (section E).
* **Chest-interaction rules (simulator side)**: 7 of 7 cases as specified (section H).
* **Database integrity**: <<DB_PASS>> (section L).
* **Recovery from injected faults**: <<ROBUST_SHORT>> (section L).
* **Generalisation to held-out procedural maps**, with paired gaps and intervals
  (section K).
* **Startup checks, pinned runtime, JSON logs, SIGTERM handling**: tested (Phase 7 tests;
  clean Python 3.11 and 3.12 virtual environments built from `requirements-lock.txt`).

## C. Unverified capabilities

* **Everything that runs inside Unreal Engine** (section E):
  * the C++ module;
  * the map;
  * character, chest and camera capture;
  * the C++ bridge server;
  * Unreal's interaction validation;
  * Unreal resets;
  * Unreal frame time, GPU and memory.
* **Perception on Unreal-rendered frames.** The detector has only ever seen the Python
  renderer's flat-shaded frames.
* **Recast NavMesh behaviour.** The reference arena's NavMesh-style grid stands in for it.
* **The learned policy's value beyond the synthetic demonstrations it was trained on.** No
  human demonstration exists (section I).
* **Long episodes in endurance runs.** Endurance used 50-second episodes; long-episode
  stability was observed only up to 600 simulated seconds in the benchmarks.

## D. Broken capabilities (as found) and their state now

| # | Severity | Defect (reproduced) | State |
|---|---|---|---|
| F11 | P1 | **Visual verification false negatives.** 5 of 91 real openings (5.7 %) in 10 clean episodes were judged "closed" in every look although the agent's own "chest opened" notification arrived. The agent then re-aligned about 6 times on an already-opened chest (~12 s lost) and marked it failed. | **fixed**: bounded look-back, then credit on the notification (counted separately). Same seeds after the fix: 102 / 102 credited, 0 false failures, and 12 % more real openings |
| F14 | P2 | **Opening credited to the wrong spawn point.** A track misplaced by ~4 m was associated with a spawn point 4.8 m from the chest actually opened (1 false success + 1 false failure in 20 episodes). | **fixed**: credit the known spawn point within reach; 0 / 0 after |
| F12 | P2 | **Active-exploration openings never stored in SQLite.** 0 rows for 5 real openings in the traced episode. | **fixed**: schema 7 `discovered_openings`; rows = credited openings |
| F9 | P2 | **Every endurance resume failed after a crash between "session completed" and "result written"** (finished sessions are immutable; the resume tried to rewrite them). The Phase 6 test meant to cover crashes could never pass. | **fixed** + regression test |
| F8 | P2 | **Experience partitions ignored the supervisor.** Behaviourally different configurations shared experience. | **fixed** |
| F7 | P2 | **Four commands refused to run on a fresh checkout** (database required but unused): `profile-agent`, `benchmark-phase6`, `endurance-test`, `calibrate-policy`. | **fixed** |
| F2 | P1 for Unreal | **Phase 7 maps could not be exported to the Unreal project at all.** The Phase 7 document claimed they could. | **fixed**; the acceptance map is exported |
| F3 | P2 | **The Unreal arena builder ignores `surface_value`.** The appearance shift is only partly reproducible in Unreal. | open (C++ cannot be built here) |
| F1 | P3 | **Two controller handlers caught every exception silently.** | **fixed** (`BridgeError` only, counted) |
| F15 | P1 | **Chest-appearance shift defeats detection.** Precision and recall collapse when chest colours leave the trained range (section F, K). | open: needs a perception change (section O) |
| F16 | P2 | **Opened/unopened chest confusion in single frames.** <<OPENED_CONFUSION>> (section F). | mitigated by F11's look-back and F20; detector unchanged |
| F17 | P1 for long episodes | **False loop alarms in active exploration.** The frontier → scan → frontier cycle of an exploring agent read as a transition loop: 489 detections and 380 interventions in 100 fault-free 150 s episodes, and a 600 s acceptance episode aborted on the intervention budget. | **fixed**: new coverage clears the loop window. Same 100 episodes after: 152 detections, 143 interventions, unchanged throughput; no acceptance episode aborted |
| F18 | P2 | **Exact replay not guaranteed.** Two decisions depend on wall-clock time: the 20 ms frontier-selection budget and the hybrid selector's 20 ms inference timeout. One acceptance episode played twice on the same seed gave 10 and 11 chests. | open: needs work-based limits for lock-step research runs (section O) |
| F19 | P1 (introduced by the F14 fix) | **Phantom openings after a reattribution.** The interaction's own spawn point stayed "unverified" and was credited again when the agent aligned on it and saw the neighbouring opened chest. 1–4 over-reported openings per 40 episodes in every spawn-point configuration. | **fixed**: the 5 affected episodes replay with reported = true |
| F20 | P1 | **False success on mixed visual evidence.** A failed (injected) interaction was credited on two "opened" looks against two "closed" ones, with no opening notification. 1 in 200 robustness episodes. | **fixed**: vision-only credit needs unanimous looks |

The full log, with reproduction steps, is `benchmark_results/phase8/failures.json` plus
section N.

## E. Unreal integration assessment

**Python-only functionality.**

* the agent;
* perception, tracking, world model, planners, exploration, supervisor;
* the learned policy (ONNX, CPU);
* the database;
* the benchmarks.

All executed (sections B, F–M).

**Mock Unreal integration (executed).**

* The Python bridge client against the Python reference server:
  * framing, HELLO, RESET_EPISODE, STEP (with replay protection), observations;
  * NAV_QUERY / NAV_RAYCAST;
  * privileged-request gating, reconnect, busy rule, timeouts.
* `bridge_security`: 15 / 15, including oversized and malformed messages, NaN/∞ and
  out-of-range actions, unknown action keys, stale and out-of-order steps, a stalled partial
  message (dropped after the 1 s read timeout), abrupt disconnects, and clean shutdown.

**Actual Unreal Engine functionality: none executed.** The static audit
(`integration_results.json → unreal`) found:

* `ChestHunterResearch.uproject` is valid (UE 5.4, one runtime module, editor-only Python
  plug-in, online subsystems disabled).
* 17 C++ files (3,143 lines). Build dependencies: Core, CoreUObject, Engine, InputCore, Json,
  JsonUtilities, Sockets, Networking, NavigationSystem, AIModule, RenderCore.
* **No `Binaries/`, no `.umap`, no `.uasset`.** The configured default map
  `/Game/Maps/ResearchArena3D` does not exist. It must be generated by
  `Scripts/create_research_map.py`, which has never been executed.
* Assets referenced: only `/Engine/BasicShapes/Cube` and `BasicShapeMaterial`, which exist in
  every engine installation.
* Layout files present: `research_arena_3d`, `proc_{train,test}_000–003`, and since this
  audit, five Phase 7 maps including the acceptance map.
* **Protocol and parameter consistency** with the Python side: the existing source tests pass,
  including the C++ random roll compiled with the host `g++` and compared bit for bit. This
  audit added checks that `RESET_EPISODE`, `NAV_QUERY` and `NAV_RAYCAST` argument whitelists
  and ranges match.
* **Units and axes are consistent.**
  * The protocol carries metres; the C++ converts with `ResearchUnits::ToCm/ToMetres`.
  * Both sides use X north/forward, Y east/right, Z up, yaw in degrees from +X towards +Y
    (Unreal's convention).
  * Image coordinates are u right, v down, origin top-left.
  * The navigation grid is 0.5 m cells.
* The C++ automation tests (`ResearchProtocolTests.cpp`) have never been compiled.

**Blocked tests** (all need UE 5.4):

* building the module;
* the automation tests;
* creating the map;
* loading map, character and chests;
* SceneCapture frames;
* the C++ server with the Python client;
* Unreal interaction validation and chest state;
* resets and termination in Unreal;
* every benchmark with `--backend unreal`;
* Unreal frame time, GPU and memory.

The exact commands for a machine with Unreal are listed in `integration_results.json → unreal →
commands_for_unreal_machine` and in section O.

**Classification: UNVERIFIED.**

<<SECTION_F>>

<<SECTION_G>>

<<SECTION_H>>

<<SECTION_I>>

<<SECTION_J>>

<<SECTION_K>>

<<SECTION_L>>

<<SECTION_M>>

<<SECTION_N>>

<<SECTION_O>>
