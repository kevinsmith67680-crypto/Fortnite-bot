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

## A. Executive summary

**Classification: UNVERIFIED. Weighted score 50.3 / 100** (scorecard:
`docs/phase8_readiness_scorecard.md`).

* **Why UNVERIFIED.** The intended environment, the Unreal Engine research simulator, has
  never been built or run. No Unreal Engine installation was available to this audit. Every
  end-to-end number below comes from the Python reference arena, the project's mock of the
  Unreal server.
* **Within that mock,** every gate that can be evaluated passes. The two gates that need
  Unreal do not.
* **Not a deployment verdict.** Nothing here concerns live Fortnite matches. The project is
  confined to an authorised research simulator, and readiness for anything else is neither
  assessed nor implied.

**What the audit established by execution** (reference arena, final code):

| Area | Result | Section |
|---|---|---|
| Autonomous loop | all 12 stages exercised in one traced, recorded episode; credited openings equal the simulator's (3 / 3) | B |
| Detection (labelled frames) | precision 0.97 / 0.93 / 0.96, recall 0.48 / 0.65 / 0.75 at 320×180 / 480×270 / 640×360; 0.9–3.0 ms per frame on CPU | F |
| Navigation | 32 / 32 reachable goals reached; 5 / 5 unreachable goals recognised after one query; pose error about 0.01 m | G |
| Interaction | 7 / 7 simulator rules; 0 false successes and 0 false failures in 20 cross-checked episodes, with injected faults | H |
| Best configuration (deterministic, nearest target, spawn points) | 3.84 [3.51, 4.18] chests per simulated minute, 76 % of reachable chests in 200 s, 0 failed episodes | I, J |
| Without spawn points (active exploration) | 1.16 [1.01, 1.31] chests per minute on the held-out acceptance map, 57 % of reachable chests and 74.5 % coverage in 600 s | K |
| Learned policy | **worse** than the deterministic controller: −2.60 [−2.90, −2.31] alone, −0.46 [−0.70, −0.21] gated | I |
| Faults | 85 / 94 recovered, 94 / 94 clean terminations, 0 failed | K |
| Endurance | 100, 500 and 1000 episodes: 0 failures, memory growth ≤ 1.4 MB per 100 episodes | K |
| Python pipeline cost | about 1 % of one core at 320×180 and 5 analysed frames per second | L |
| Database | 13 / 14 integrity checks; the one failure is an opening still being verified when its episode ended | L |
| Bridge security (Python server) | 15 / 15 | L |
| Ground-truth isolation | static, runtime and counterfactual checks all pass | L |

**What the audit could not establish:**

* anything inside Unreal Engine: the C++ module, the map, SceneCapture frames, the C++
  bridge server, Unreal's interaction validation;
* Unreal frame time, GPU load and memory;
* perception on Unreal-rendered frames;
* the learned policy's value with real demonstrations.

**Defects.** 23 findings: 18 defects (section D) and five informational notes (F4–F6, F10,
F13: Unreal static facts, scan results, the first test run and the loop trace).

* **Fixed, each reproduced first and guarded by a regression test:** 14, of which 6 were P1
  (F2, F11, F17, F19, F20, F21).
  * Five behaviour defects (F17, F19, F20, F21 and F22) passed the full unit-test suite.
    They were found only by checking every credited opening, early ending and failed episode
    in the benchmarks against the simulator.
  * F19 was introduced by this audit's own F14 fix, and was caught the same way.
  * F23 was in the audit tool itself: a re-run silently reported rows played by older
    code. It was caught because two sections finished in 0.1 s instead of minutes.
* **Open:**
  * chest-appearance sensitivity of detection (F15, P1);
  * opened/closed confusion in single frames (F16, mitigated);
  * wall-clock-dependent replay (F18);
  * the Unreal builder ignores `surface_value` (F3);
  * one residual early ending in 100 maps (F21 residual).

**Tests.** {{TESTS_A}} Ruff and `mypy --strict` are clean.

**Simplicity.** No component added after Phase 4 improves throughput over the deterministic
nearest-target controller in this environment (section M). That covers the learned policy,
the adaptive and short-horizon planners and the exploration bandit.

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
  all pass (section L).
* **The bridge's defensive behaviour (Python server)**: 15 of 15 raw-socket checks pass
  (section E).
* **Chest-interaction rules (simulator side)**: 7 of 7 cases as specified (section H).
* **Database integrity**: 13 of 14 checks pass. The one failure is an opening still being
  verified when its episode ended, not a persistence defect (section L).
* **Recovery from injected faults** (section K):
  * 85 of 94 applied faults recovered within 30 s (90 %);
  * 94 / 94 clean terminations and 0 failed episodes;
  * 0 over-reported openings in 200 episodes.
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
| F16 | P2 | **Opened/unopened chest confusion in single frames.** On the 21 labelled frames with an opened chest, precision is 0.71–0.88 and recall 0.58–0.75 across the three resolutions; 9 of the 38 failure frames involve opened chests (section F). | mitigated by F11's look-back and F20; detector unchanged |
| F17 | P1 for long episodes | **False loop alarms in active exploration.** The frontier → scan → frontier cycle of an exploring agent read as a transition loop: 489 detections and 380 interventions in 100 fault-free 150 s episodes, and a 600 s acceptance episode aborted on the intervention budget. | **fixed**: new coverage clears the loop window. Same 100 episodes after: 152 detections, 143 interventions, unchanged throughput (final code: 159 and 149); no acceptance episode aborted |
| F18 | P2 | **Exact replay not guaranteed.** Two decisions depend on wall-clock time: the 20 ms frontier-selection budget and the hybrid selector's 20 ms inference timeout. One acceptance episode played twice on the same seed gave 10 and 11 chests. | open: needs work-based limits for lock-step research runs (section O) |
| F19 | P1 (introduced by the F14 fix) | **Phantom openings after a reattribution.** The interaction's own spawn point stayed "unverified" and was credited again when the agent aligned on it and saw the neighbouring opened chest. 1–4 over-reported openings per 40 episodes in every spawn-point configuration. | **fixed**: the 5 affected episodes replay with reported = true |
| F20 | P1 | **False success on mixed visual evidence.** A failed (injected) interaction was credited on two "opened" looks against two "closed" ones, with no opening notification. 1 in 200 robustness episodes. | **fixed**: vision-only credit needs unanimous looks |
| F21 | P1 | **Premature "exploration complete".** When no ray of the 16-ray walkability fan crossed the start room's doorway, the first frontier search after the initial scan found no candidate and the episode ended after 3.9 s with no chest. 18 of 800 Phase 7 generalisation episodes (all indoor) and one resource-profile episode were affected. | **fixed**: one denser 64-ray fan per place before concluding; the reproduced episode now runs to the time limit. One mid-episode case remains (`p7_indoor_test_018`, 77 s, 12.5 % coverage; section O) |
| F22 | P2 | **Off-mesh unstick manoeuvre repeated in a blocked direction.** With no walked trail to retrace, every manoeuvre went straight behind the camera. On a raised spot that direction was blocked, and the agent stood still until the supervisor aborted the episode (Phase 6 benchmark, `phase6_hybrid_optimised` episode 57: 19 manoeuvres at one bearing; the only failed episode in 500). | **fixed**: each further trail-less manoeuvre turns by a quarter turn; the episode now completes at 387.8 s. Found after the first final audit run; the audit was re-run on the fixed code |
| F23 | P2 (audit tool) | **An audit re-run reused comparison rows from older code.** `audit-system` always resumed the policy and planner row journals in its output directory. Re-run after F22, it finished those sections in 0.1 s and reported 320 rows played by the code before the fix as its own. | **fixed**: a journal is resumed only when the recorded git tree of `src/` matches the current, unmodified source. The audit was then re-run from scratch |

Each defect's reproduction (episode, seed and map) is named in its regression test
(section N). `failures.json` logs the audit's own evaluation errors; it is empty in the final
run because every section completed.

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

## F. Computer-vision evaluation (`vision_metrics.json`, `vision_failures/`)

**Dataset.** 410 labelled frames per resolution, rendered by the reference arena.

* Seven maps: the research arena and Phase 7 maps of every family.
* Each frame is labelled from the evaluation channel with every chest in view, its class
  (closed or opened) and its pixel share. Chests below 0.015 % of the image are excluded:
  they are fewer pixels than a detector can be expected to see.
* Scenario tags: centred, off-centre, partially occluded, near (0–3 m), medium (3–8 and
  8–15 m), far (15 m+), multiple chests, opened chests, three lighting levels, indoor and
  outdoor, chest absent, and random poses.
* The same poses are rendered at every resolution, so the resolutions are paired.

The detector is the unchanged production detector, which also feeds the agent. Its scores
are segmentation and shape scores, not calibrated probabilities, so no precision–recall curve
is drawn from them.

| resolution | frames | TP | FP | missed | precision | recall | F1 | FP / frame | latency mean / p95 (ms) | CPU (ms) | distance error mean / median / p95 (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 320x180 | 410 | 198 | 7 | 217 | 0.966 | 0.477 | 0.639 | 0.0171 | 0.92 / 1.48 | 0.92 | 0.35 / 0.17 / 1.38 |
| 480x270 | 410 | 258 | 18 | 141 | 0.935 | 0.647 | 0.764 | 0.0439 | 1.86 / 2.59 | 1.84 | 0.41 / 0.18 / 1.71 |
| 640x360 | 410 | 292 | 13 | 95 | 0.957 | 0.754 | 0.844 | 0.0317 | 3.04 / 3.88 | 3.01 | 0.39 / 0.16 / 1.76 |

Per scenario (precision / recall, frames):

| scenario | 320x180 P / R (frames) | 480x270 P / R (frames) | 640x360 P / R (frames) |
|---|---|---|---|
| centred | 0.97 / 0.52 (107) | 0.93 / 0.63 (107) | 0.95 / 0.75 (107) |
| chest_absent | n/a / n/a (169) | n/a / n/a (172) | n/a / n/a (174) |
| chest_visible | 0.97 / 0.48 (241) | 0.93 / 0.65 (238) | 0.96 / 0.75 (236) |
| dist_0_3m | 1.00 / 0.17 (2) | 1.00 / 0.43 (2) | 1.00 / 0.43 (2) |
| dist_15m_plus | 1.00 / 0.21 (86) | 1.00 / 0.49 (83) | 0.99 / 0.68 (82) |
| dist_3_8m | 0.96 / 0.67 (85) | 0.89 / 0.69 (85) | 0.91 / 0.78 (84) |
| dist_8_15m | 0.96 / 0.58 (68) | 0.94 / 0.77 (68) | 0.98 / 0.83 (68) |
| indoor | 1.00 / 0.77 (67) | 0.96 / 0.87 (67) | 1.00 / 0.88 (67) |
| lighting_high | 0.95 / 0.38 (60) | 0.95 / 0.57 (60) | 0.99 / 0.70 (60) |
| lighting_low | 0.97 / 0.48 (60) | 0.94 / 0.69 (60) | 0.92 / 0.76 (60) |
| lighting_normal | 0.97 / 0.52 (290) | 0.93 / 0.67 (290) | 0.96 / 0.78 (290) |
| multiple | 0.96 / 0.39 (110) | 0.96 / 0.60 (106) | 0.97 / 0.72 (102) |
| off_centre | 0.96 / 0.48 (219) | 0.93 / 0.69 (219) | 0.96 / 0.78 (219) |
| opened | 0.88 / 0.58 (21) | 0.71 / 0.68 (21) | 0.82 / 0.75 (21) |
| outdoor | 0.96 / 0.45 (343) | 0.93 / 0.63 (343) | 0.95 / 0.74 (343) |
| partially_occluded | 0.98 / 0.44 (55) | 0.93 / 0.56 (53) | 0.93 / 0.70 (51) |
| random | 1.00 / 0.35 (84) | 0.96 / 0.47 (84) | 0.97 / 0.65 (84) |

What the numbers say:

* **Precision is high at every resolution** (0.93–0.97), with 0.02–0.04 false positives per
  frame.
* **Recall depends strongly on resolution:**
  * 0.48 at 320×180, 0.65 at 480×270 and 0.75 at 640×360;
  * almost all of the gap comes from chests beyond 15 m (recall 0.21 / 0.49 / 0.68) and from
    multiple chests in view.
  * At 3–15 m, where the agent actually decides, recall is 0.58–0.83.
  * The agent works at 320×180 because the closed loop re-observes: per-frame recall in
    episodes is about 0.88 (loop trace), since the agent turns towards and approaches what it
    has seen.
* **Latency** is 0.9 / 1.9 / 3.0 ms mean and 1.5 / 2.6 / 3.9 ms p95, CPU only.
* **Distance error:** median 0.16–0.18 m, p95 1.4–1.8 m. Bearing error is under 0.6°.
* **Failure frames** at 640×360 are in `vision_failures/` (38 PNGs: missed, false positive
  and wrong class):
  * misses are dominated by bright lighting (15), partial occlusion (15) and off-centre
    chests (14);
  * false positives occur on opened chests (4), partially occluded chests (4) and in low
    lighting (2).
* **Opened chests (F16)** are the weakest class: precision 0.71–0.88 on the 21 frames with
  an opened chest. The detector is not changed by this audit. The F11 look-back, the
  opening notification and F20's unanimous-looks rule keep this from turning into false
  credits (section H).
* **Chest appearance** outside the trained colour range is the main perception weakness
  (F15, section K). It is not visible here because this dataset uses the standard
  appearance.

**Not evaluated:** Unreal-rendered frames. No number in this section transfers to Unreal
without being re-measured.

## G. Localisation and navigation (`navigation_metrics.json`)

**Coordinate conventions** (documented in `docs/phase4_3d_architecture.md` and checked by
the source tests):

* World frame: X north/forward, Y east/right, Z up; metres in the protocol, centimetres
  inside Unreal (`ResearchUnits`).
* Yaw is in degrees from +X towards +Y; pitch is positive up.
* Image coordinates: u right, v down, origin top-left.
* The navigation grid uses 0.5 m cells.

**Localisation.** The agent localises from odometry only; there is no visual correction.
Pose error in the reference arena (evaluation side):

| terrain | runs | pose error mean (m) | pose error max (m, mean / max) | final error (m) | error per 100 m travelled | runs with error > 1 m |
|---|---|---|---|---|---|---|
| open_terrain | 10 | 0.009 | 0.028 / 0.051 | 0.027 | 0.079 | 0 |
| around_obstacles | 2 | 0.007 | 0.024 / 0.030 | 0.024 | 0.092 | 0 |
| indoor_room | 9 | 0.009 | 0.029 / 0.047 | 0.027 | 0.090 | 0 |
| elevated | 6 | 0.013 | 0.036 / 0.045 | 0.033 | 0.074 | 0 |
| unreachable | 5 | 0.000 | 0.000 / 0.000 | 0.000 | n/a | 0 |

* The reference arena's odometry drift is small: 0.08–0.09 m per 100 m travelled.
* Injected drift and odometry jumps are detected and relocalised (section K: 10 of 10).
* A persistent offset would stay, because nothing corrects it visually. Unreal odometry was
  not measured.

**Navigation.** 37 goals with the real agent on six maps (the research arena and five
Phase 7 test maps):

* goals sampled by terrain;
* the agent is sent to each goal through its normal planning, path following, approach and
  recovery;
* scored against the navigation service's shortest path.

| terrain | tasks | reached | time to reach mean / median / max (s) | detour ratio mean / median / max | recoveries per task (mean) | stuck events | supervisor interventions |
|---|---|---|---|---|---|---|---|
| open_terrain | 10 | 100.0 % | 11.2 / 10.2 / 19.1 | 0.978 / 0.973 / 1.011 | 0.00 | 0 | 0 |
| around_obstacles | 2 | 100.0 % | 10.0 / 10.0 / 12.8 | 0.986 / 0.986 / 0.993 | 0.00 | 0 | 0 |
| indoor_room | 9 | 100.0 % | 11.0 / 10.0 / 17.9 | 0.960 / 0.958 / 0.988 | 0.00 | 0 | 0 |
| elevated | 6 | 100.0 % | 13.7 / 14.3 / 16.9 | 0.964 / 0.960 / 0.988 | 0.00 | 0 | 0 |

Unreachable goals: 5 tasks, detected 100.0 %, time to detection mean / median / max 3.9 / 3.9 / 3.9 s, at most 1 path queries per goal, falsely reached 0.

Overall success on reachable goals: 100.0 % (32 tasks, maps research_arena_3d, p7_open_test_002, p7_indoor_test_002, p7_mixed_test_002, p7_complex_test_002, p7_complex_test_003).

* **Every reachable goal was reached**: no stuck events, no recoveries, no interventions.
* **Paths are near-optimal.** Detour ratios below 1 mean the agent's string-pulled path is
  slightly shorter than the service's grid path. The ratio is not an absolute optimality
  measure.
* **Unreachable goals are recognised after one path query** (3.9 s) and never "reached".
* The "around obstacles" class has only 2 goals: the generator places few obstacles directly
  between the start and a goal.
* **Navigation inside full episodes:**
  * per-episode navigation success is 0.96–0.98 (section K);
  * failures there come from dynamic situations (blocked paths, injected obstacles), which
    this static test does not cover.
* **Not evaluated:** Unreal's Recast NavMesh. The reference arena's grid stands in for it.

## H. Chest-interaction audit (`interaction_metrics.json`)

**Simulator rules** (7 / 7 as specified):

| case | expected opened | opened | events (expected) | pass |
|---|---|---|---|---|
| in range, aligned, 2.2 s hold | True | True | 1 (1) | True |
| too far (2.6 m > 1.8 m radius) | False | False | 0 (0) | True |
| camera misaligned (100 deg) | False | False | 0 (0) | True |
| chest absent at the spawn point | False | False | 0 (0) | True |
| already opened, held again | True | True | 1 (1) | True |
| interrupted hold (1.0 s + 1.0 s) | False | False | 0 (0) | True |
| duplicate command (one 6 s hold) | True | True | 1 (1) | True |

**Agent cross-check.**

* Setup: the Phase 6 deterministic agent with supervisor, on the research arena, 4 episodes
  per scenario, 120 s each, seed 21.
* Every interaction attempt is compared with the simulator's opening log.
* A **false success** is a credited opening that did not happen. A **false failure** is a
  real opening that the agent did not credit, excluding openings in the last 6 s of an
  episode, which are still being verified when it ends.

| scenario | episodes | attempts | true openings | credited | false successes | false failures | false-success rate | false-failure rate | uncredited at end | credited on notification | faults applied |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 4 | 40 | 38 | 38 | 0 | 0 | 0.0 % | 0.0 % | 0 | 2 | 0 |
| failed_interactions | 4 | 49 | 36 | 36 | 0 | 0 | 0.0 % | 0.0 % | 0 | 0 | 4 |
| corrupted_view_while_verifying | 4 | 41 | 40 | 40 | 0 | 0 | 0.0 % | 0.0 % | 0 | 3 | 4 |
| frames_lost_while_verifying | 4 | 41 | 40 | 40 | 0 | 0 | 0.0 % | 0.0 % | 0 | 3 | 4 |
| connection_dropped_while_interacting | 4 | 40 | 38 | 38 | 0 | 0 | 0.0 % | 0.0 % | 0 | 2 | 4 |

Configuration: phase6_deterministic_supervised, research_arena_3d, spawn points; seed 21; 3600 ticks.

* **0 false successes and 0 false failures in 20 episodes** after F11 and F14, including:
  * injected interaction failures;
  * corrupted frames while verifying;
  * lost frames while verifying;
  * a dropped connection while interacting.
* **Before the fixes**, the same seeds showed 5.7 % false failures (F11) and one false
  success (F14).
* 2–3 openings per scenario were credited on the agent's opening notification after a
  bounded look-back, not on a visual confirmation. They are counted separately.
* **Over the whole audit, after F19 and F20:**
  * 0 over-reported openings in 320 policy and planner episodes, 200 robustness episodes,
    41 acceptance episodes and the endurance smoke run.
  * Before those fixes, the same runs had 1–4 phantom openings per 40-episode
    configuration.
* **What this does and does not show.**
  * A successful interaction command is not taken as proof of opening: every credit is
    checked against the simulator's own chest state.
  * That state is the reference arena's. Unreal's interaction validation and opened-chest
    mesh never ran.

## I. Learned policy vs deterministic vs hybrid (`policy_comparison.csv`)

**Setup.**

* 40 paired episodes on held-out unseen layouts (evaluation seeds), 200 s each, spawn points
  given.
* The same configuration in all three, except the policy.
* The learned policy is the Phase 5 behaviour-cloning model (ONNX, CPU), trained on
  synthetic demonstrations from the deterministic controller. No human demonstration exists.

| configuration | episodes | throughput (chests / sim min) [95 % CI] | reachable opened % | navigation success | interaction success (pooled) | travel m | agent CPU s / sim s | inference ms | frontier selection ms | failed | over-reported |
|---|---|---|---|---|---|---|---|---|---|---|---|
| audit_policy_deterministic | 40 | 3.844 [3.512, 4.177] | 76.1 | 0.996 | 0.993 | 364.3 | 0.0106 | n/a | n/a | 0 | 0 |
| audit_policy_learned | 40 | 1.241 [1.083, 1.399] | 46.1 | 0.749 | 0.926 | 387.1 | 0.0157 | 0.659 | n/a | 0 | 0 |
| audit_policy_hybrid | 40 | 3.389 [3.097, 3.681] | 74.1 | 0.982 | 0.993 | 366.8 | 0.0125 | 0.675 | n/a | 0 | 0 |

Paired differences against the deterministic controller (throughput, chests per simulated
minute):

| Policy | Difference [95 % CI] | Better / worse / tied |
|---|---|---|
| learned only | **−2.60 [−2.90, −2.31]** | 0 / 40 / 0 |
| hybrid (gated) | **−0.46 [−0.70, −0.21]** | 9 / 30 / 1 |

* **The deterministic controller is the best of the three.** The learned policy alone opens
  a third as many chests and its navigation success is 0.75. The hybrid gate costs 12 % of
  throughput against the deterministic controller.
* A trained model file is not evidence that the learned policy helps: here it does not.
  This agrees with Phases 5 and 6. The learned policy imitates the deterministic controller
  and so cannot exceed it.
* Inference costs 0.7 ms per call on CPU. The learned policy is not a performance problem,
  only an unproductive component (section M).

## J. Planner comparison (`planner_comparison.csv`)

**Setup.** The same 40 paired episodes, deterministic controller, with five target
selectors:

* **nearest:** the Phase 4 rule, probability ÷ travel time. With equal priors on these maps
  it chooses the nearest spawn point, so it is the same configuration as the deterministic
  row of section I.
* **adaptive:** Phase 5, learned spawn estimates.
* **greedy** and **short-horizon:** Phase 6, beam search over a 3-target horizon.
* **frontier:** Phase 7 active exploration, without spawn points.

| configuration | episodes | throughput (chests / sim min) [95 % CI] | reachable opened % | navigation success | interaction success (pooled) | travel m | agent CPU s / sim s | inference ms | frontier selection ms | failed | over-reported |
|---|---|---|---|---|---|---|---|---|---|---|---|
| audit_planner_nearest | 40 | 3.844 [3.512, 4.177] | 76.1 | 0.996 | 0.993 | 364.3 | 0.0109 | n/a | n/a | 0 | 0 |
| audit_planner_adaptive | 40 | 3.651 [3.347, 3.954] | 76.7 | 0.996 | 0.984 | 386.2 | 0.0109 | n/a | n/a | 0 | 0 |
| audit_planner_greedy | 40 | 3.828 [3.490, 4.166] | 77.0 | 0.996 | 0.997 | 364.6 | 0.0109 | n/a | n/a | 0 | 0 |
| audit_planner_short_horizon | 40 | 3.554 [3.205, 3.902] | 74.8 | 0.999 | 0.993 | 397.2 | 0.0108 | n/a | n/a | 0 | 0 |
| audit_planner_frontier | 40 | 1.702 [1.497, 1.908] | 59.9 | 0.998 | 0.966 | 518.8 | 0.0129 | n/a | 15.140 | 0 | 0 |

Paired differences against nearest:

| Planner | Throughput difference [95 % CI] | Better / worse / tied | Extra travel (m) |
|---|---|---|---|
| greedy | −0.02 [−0.16, +0.12] | 8 / 13 / 19 | +0.3 [−5.3, +5.9] |
| adaptive | −0.19 [−0.43, +0.04] | 15 / 19 / 6 | **+21.9 [+9.0, +34.9]** |
| short-horizon | −0.29 [−0.59, +0.00] | 14 / 24 / 2 | **+32.9 [+17.4, +48.4]** |
| frontier (no spawn points) | **−2.14 [−2.45, −1.83]** | 0 / 40 / 0 | **+154.5 [+111.1, +198.0]** |

* **The simplest planner is the best.** No planner beats nearest-target selection.
  * Greedy is indistinguishable from it.
  * Adaptive and short-horizon travel significantly further for no gain.
* **The frontier explorer** solves a harder problem: it is not told where chests can be. It
  is not comparable on equal terms, but it gives the price of not knowing the spawn points.
* **Agent CPU** is the same for all spawn-point planners (0.011 s per simulated second), so
  none of them is a performance problem either.
* **Recommendation:** nearest-target selection with spawn points; active exploration only
  where spawn points are unknown.

## K. Reliability, endurance and generalisation

**Fault injection** (`benchmark_results/robustness/`):

* 100 faulted episodes on held-out layouts, ten kinds × 10, each with a fault-free twin;
* the Phase 7 agent, 150 s per episode;
* full table in the Phase 7 document, section 15.2.

| | Result (commit `3ad12f3`) |
|---|---|
| Faults applied | 94 of 100 (interaction failures need an attempt; slow inference needs the learned selector) |
| Recovered within 30 s | **85 / 94 (90 %)** |
| Clean termination | 94 / 94 (100 %) |
| Failed episodes | 0 |
| Over-reported openings | 0 in 200 episodes |
| Fault-free twins | 1.79 [1.61, 1.96] chests per simulated minute |
| Weakest kinds | slow inference 4 / 8 recovered; missed detection 7 / 10 recovered; unexpected obstacle 7 / 10 detected; unreachable target 7 / 10 detected; corrupted frame 9 / 10 recovered |

No fault kind costs measurable throughput: every paired interval includes 0, with 6–10 pairs per kind.

* **Supervisor false alarms.**
  * Before F17: 489 loop detections in 100 fault-free episodes.
  * After F17: 159 detections in 62 of the 100 twins, with 149 supervisor interventions, and they cost no measurable throughput.
  * Whether each residual detection is a real loop was not established (section O).
* **Unbounded retries:** none. Every recovery ladder is bounded, and no episode hit the
  intervention budget.

**Endurance** (`endurance_results.csv`):

* The level-1 runs are reused from Phase 6: 100, 500 and 1000 episodes, commit `e1124bd`.
* A 10-episode smoke run on the final code was added.

| run | source | episodes completed / requested | outcomes | failed | comm. failures | interventions / aborts | recovery success | over-reported | throughput | CPU s / episode | RSS first / last / max (MB) | RSS growth MB / 100 ep | threads | fds | bounded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| endurance_100 | reused | 100 / 100 | {"tick_limit": 100} | 0 | 0 | 1 / 0 | 0.886 | 0 | 5.184 | 4.63 | 97.0 / 102.1 / 107.0 | 1.360 | [3, 3] | [9, 9] | True |
| endurance_1000 | reused | 1000 / 1000 | {"tick_limit": 1000} | 0 | 0 | 16 / 0 | 0.868 | 0 | 5.366 | 4.38 | 97.0 / 110.2 / 110.6 | 0.466 | [3, 3] | [9, 9] | True |
| endurance_500 | reused | 500 / 500 | {"tick_limit": 500} | 0 | 0 | 8 / 0 | 0.853 | 0 | 5.314 | 4.51 | 97.3 / 105.3 / 109.1 | 0.535 | [3, 3] | [9, 9] | True |
| level1_smoke_10 | this audit | 10 / 10 | {"tick_limit": 10} | 0 | 0 | 1 / 0 | 0.531 | 0 | 5.040 | 3.93 | 230.5 / 239.2 / 239.2 | 0.200 | [3, 3] | [11, 11] |  |

* 1,600 episodes in three runs, each run on one long-lived server connection: no failed
  episode, communication failure or abort.
* Memory growth was 0.47–1.36 MB per 100 episodes after warm-up (bounded). File descriptors
  and threads were constant.
* The 1000-episode run is not a Phase 7 active-exploration endurance run: it uses spawn
  points and 50 s episodes. Long episodes were covered only by the 600 s acceptance
  episodes.

**Generalisation** (`generalisation_results.csv`, reused from Phase 7, 1,600 episodes):

* No significant gap for new obstacle arrangements, unseen layouts, start poses or extra
  elevations.
* **Significant gaps:**
  * lighting: G = +0.11 [+0.04, +0.18] (Phase 7 agent);
  * chest appearance: G = +0.28 [+0.21, +0.36];
  * combined: G = +0.37 [+0.25, +0.49].
* Appearance is a perception failure (F15): precision and recall fall to about 0.58.
* **Code version.** That benchmark ran before the Phase 8 fixes.
  * F21 ended 18 of its 800 Phase 7 episodes at 3.9 s, all indoor. Indoor throughput in
    that table is therefore understated.
  * F11, F19 and F20 do not change ground-truth throughput, only crediting.
  * It was not re-run: 2 h of wall clock. Section O lists it.

**Acceptance** (held-out map, 600 s episodes):

| Group (10 episodes each, commit `3ad12f3`) | Throughput | Reachable opened | Interaction success | Coverage | Failed |
|---|---|---|---|---|---|
| Phase 7 active, clean | 1.16 [1.01, 1.31] | 57.4 % | 95.1 % | 0.745 [0.705, 0.786] | 0 |
| Phase 7 active, one fault each | 1.09 [0.89, 1.29] | 54.0 % | 91.6 % | 0.709 [0.650, 0.767] | 0 |
| Phase 6 with spawn points, clean | 3.14 [2.65, 3.63] | 60.9 % | 96.1 % | n/a | 0 |
| Phase 6 with spawn points, one fault each | 2.91 [2.15, 3.66] | 56.9 % | 95.0 % | n/a | 0 |

* Interaction success target (≥ 90 %): **met** (95.1 %). Coverage target (≥ 85 %): **not met** (74.5 %).
* Exact replay of episode 0: **failed** (10 vs 11 chests, under heavy CPU load): finding F18.
* 0 over-reported openings in 41 episodes; peak worker memory 146.5 MB.
* Full description in the Phase 7 document, section 16.

## L. Performance, data integrity and communication security

**Hardware profile** (`resource_profile.json`):

* One process, sequential, 3 held-out maps per configuration, active exploration,
  deterministic policy.
* Configurations: 320×180 at 5 analysed frames per simulated second, 480×270 at 5,
  640×360 at 10, and 320×180 with the adaptive "efficient" profile.

| configuration | episodes | sim s | wall s | pipeline CPU s / sim s (share of one core in real time) | pipeline CPU ms / analysed frame | frames analysed | bridge client CPU s | simulator CPU s | RSS first / last / max (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 320x180_5fps | 3 | 600.0 | 35.8 | 0.0095 | 1.906 | 3000 | 1.88 | 26.91 | 246.7 / 260.2 / 260.2 |
| 480x270_5fps | 3 | 600.0 | 42.8 | 0.0157 | 3.136 | 3000 | 2.15 | 29.83 | 266.1 / 270.1 / 270.1 |
| 640x360_10fps | 3 | 600.0 | 82.1 | 0.0330 | 4.394 | 4500 | 3.22 | 57.28 | 291.3 / 315.7 / 315.7 |
| 320x180_adaptive | 3 | 600.0 | 31.2 | 0.0087 | 2.029 | 2580 | 1.88 | 22.88 | 316.5 / 317.0 / 317.0 |

* **Python decision pipeline:** the agent's thread uses about 1 % of one core per simulated
  second at 320×180 (1.9 ms per analysed frame), 1.6 % at 480×270 and 3.3 % at 640×360 and
  10 fps (4.4 ms per frame).
* **Simulator:** most of the process CPU is the reference arena's software renderer and
  physics. It stands in for Unreal and says nothing about Unreal's cost.
* **Adaptive profile:** it analysed 2,580 frames instead of 3,000 and used 8 % less pipeline CPU per simulated second (0.0087 vs 0.0095). Its CPU per analysed frame is higher (2.03 ms) only because the work done on every tick is divided by fewer frames. In these three episodes it opened fewer chests (1.1 vs 1.5 per minute on average). Three episodes cannot measure that cost, so the saving should not be assumed to be free. The other configurations were within 0.2 chests per minute of each other on the same three maps
* **Memory:** the RSS values are those of the audit process after many earlier sections.
  They are not an agent's footprint; that is 97–110 MB in the endurance runs and at most
  146.5 MB per acceptance worker.
* **Unreal frame time, GPU load and Unreal memory: BLOCKED.** There is no Unreal and no GPU.
  No value is given or estimated.

**Database integrity** (`database_integrity.json`, 6 episodes in a fresh SQLite file plus
targeted checks): 13 of 14 checks pass.

| check | pass | detail |
|---|---|---|
| one session per episode | True | 6 |
| every session finished with ended_at | True | ["completed"] |
| integrity_check ok | True | null |
| no foreign-key violations | True | null |
| no duplicated openings | True | [] |
| database openings equal the agent's credited openings | True | null |
| no recorded opening that did not happen | True | [] |
| every true opening recorded | False | {"missing": 1, "true_openings": 39} |
| session statistics equal raw observation rows | True | [] |
| an interrupted episode's session ends aborted | True | {"status": "aborted", "ended_at": "2026-11-15T20:14:59.253287+00:00"} |
| active-exploration openings reach the database | True | {"discovered_opening_rows": 4, "credited": 4, "true_openings": 4} |
| invalid values are refused | True | {"probability 1.5": "ValidationError", "NaN coordinate": "ValidationError", "empty map id": "ValidationError", "negative |
| migration from schema 1 keeps data | True | {"schema": 7, "rows": 1} |
| two concurrent writer processes | True | {"sessions": 4, "integrity": true, "errors": [[], []]} |

13 / 14 checks pass; 6 episodes.

* **The failing check is not a persistence defect.** The one missing opening (spawn point
  15, session 1) happened 6.1 s before the end of an 80 s episode. The agent was still in
  VERIFYING_OPEN when the episode ended (state log), so it had not credited it.
* **No recorded opening is false and none is duplicated.** Every credited opening, including
  active-exploration openings (F12), is in the database.

**Communication security** (`integration_results.json → bridge_security`, against the Python
reference server): 15 of 15.

| check | pass | observed |
|---|---|---|
| server refuses to bind a non-loopback address | True | the research server only listens on the loopback interface |
| client refuses a non-loopback server | True | ValidationError |
| oversized header is refused | True | {'type': 'response', 'command': None, 'request_id': None, 'status': 'error', 'error': {'code': 'too_large', 'm |
| oversized payload is refused without reading it | True | {'type': 'response', 'command': None, 'request_id': None, 'status': 'error', 'error': {'code': 'too_large', 'm |
| malformed JSON header is refused | True | {'type': 'response', 'command': None, 'request_id': None, 'status': 'error', 'error': {'code': 'bad_json', 'me |
| wrong magic bytes close the connection | True | {'code': 'bad_magic', 'message': "expected b'CHRB', got b'XXXX'"} |
| protocol version mismatch is refused | True | {'code': 'version_mismatch', 'message': 'client protocol version 999, server 1'} |
| invalid actions are refused by the server | True | {'NaN': 'error', 'infinity': 'error', 'out of range': 'error', 'wrong type': 'error', 'unknown action key': 'e |
| a replayed step is not applied twice | True | {'first': 0.033333, 'replayed': 0.033333} |
| stale and out-of-order steps are refused | True | [{'code': 'stale_step', 'message': 'step 1 is older than 2'}, {'code': 'out_of_order', 'message': 'expected st |
| privileged requests are refused without the server flag | True | [{'code': 'forbidden', 'message': 'the server was not started in oracle or evaluation mode'}, {'code': 'forbid |
| a second concurrent client is refused (busy) | True | {'code': 'busy', 'message': 'another client is connected'} |
| server recovers from an abrupt client disconnect | True | None |
| a stalled partial message is dropped after the read timeout | True | 1.0 |
| server stop releases its thread and listener | True | {'running': False} |

15 / 15

* The bridge listens on loopback only and accepts one client at a time.
* Privileged evaluation and oracle requests are refused unless the server was started for
  evaluation.
* No external game-control service exists.
* The C++ server implements the same checks (static comparison, section E), but it never
  ran.

**Ground-truth isolation** (`integration_results.json → ground_truth_isolation`):

* **Static:** 13 of 13 checks pass. Twelve agent modules import no simulator code and make
  no privileged request, and the agent port exposes only permitted calls.
* **Runtime:** 0 privileged calls from agent modules in 450 decisions. The 210 privileged
  calls in that episode came from the runner's evaluation scoring.
* **Counterfactual:**
  * flipping the presence of a chest 68 m away changed no decision before it came into
    view (138 decisions compared);
  * running with and without the evaluation labels produced identical decisions.

## M. Overengineering analysis

Measured value against complexity, component by component:

| Component | Size | Measured value | Verdict |
|---|---|---|---|
| Learned policy + hybrid gate (Phase 5) | dataset, training, ONNX export, selector, reliability monitors | learned alone −2.60, hybrid −0.46 chests/min vs deterministic (section I) | **no gain**; keep for research only, or remove from default configurations |
| Adaptive planner (Phase 5) | spawn estimator, route memory | −0.19 [−0.43, +0.04], +22 m travel | **no gain** |
| Short-horizon and greedy planners (Phase 6) | beam search | −0.29 / −0.02, more travel | **no gain** over nearest-target |
| Exploration memory + bandit (Phase 7) | statistics, posterior, persistence | not better than the best fixed strategy (Phase 7 doc 15.3) | **no gain**; a fixed `max_information` (or `nearest_frontier`) would do |
| Five frontier strategies | weights per strategy | only `cluster` differs (worse) | 1–2 strategies suffice |
| Supervisor (Phase 6) | 10 detectors, ladders | detects and bounds all injected faults; residual false loop alarms (section K) | **needed** |
| Active exploration (Phase 7) | frontiers, fans, sectors, coverage | the only way to find chests without spawn points | **needed** where spawn points are unknown |
| Visual verification look-back (F11) | bounded | removed 5.7 % false failures | **needed** |

* Nothing is removed in this audit. The brief asks for evidence before broad rewrites, and
  removal should be judged in the intended environment (section O).
* The repeated pattern is that **added sophistication on top of the deterministic
  nearest-target controller has not improved throughput** in this environment.

## N. Tests, defects and fixes

**Test runs.**

| When | Python | Result |
|---|---|---|
| Start of the audit (`6e48740`) | 3.13.18 | 762 tests: **4 failures, 1 error** (F7 ×3, F8, F9). The Phase 6 tests had evidently never been run as one suite |
| After the first fixes (`fc5877d`) | 3.11.18, clean venv from `requirements-lock.txt` | 772 OK, 7 skipped (GUI tests, no display) |
| After the first fixes (`fc5877d`) | 3.12.3, clean venv + Xvfb | 772 OK, 0 skipped |
| `3ad12f3` code (run at `e076232`) | 3.13.18 | **779 OK**, 7 skipped (GUI, no display), 680 s |
| `3ad12f3` code (run at `56f2345`) | 3.12.3, clean venv + Xvfb | **779 OK, 0 skipped**, 726 s |
| `107d52b` (with the F1 and F2 regression tests) | 3.13.18 | **781 OK**, 7 skipped (GUI, no display), 528 s |
{{TESTS_FINAL_ROW}}

Static checks on the final code:

* ruff (E, F, W, B, UP, SIM, I): clean;
* mypy `--strict`: 136 source files, no issues.

**How each defect was handled.** Every defect was reproduced before a change: the exact
episode, seed and map are named in its regression test. The fix was the smallest change that
removed the reproduced failure. The same episodes were then replayed, with the before and
after numbers in section D.

| # | Regression test |
|---|---|
| F1 | `test_phase8_audit.test_f1_only_service_errors_are_absorbed` |
| F2 | `test_phase8_audit.test_f2_a_phase7_map_can_be_exported_for_unreal` (equals the committed layout) |
| F7 | `test_phase6_workflow.CommandLine` (fresh checkout, no database) |
| F8 | `test_phase6_workflow.EpisodeExperience` |
| F9 | `test_endurance.test_a_completed_session_whose_result_was_lost_is_superseded` |
| F11 | `test_phase8_audit.test_f11_notification_confirmed_openings_are_credited` (seed 21, episode 0) |
| F12 | `test_f12_discovered_openings_are_persisted`, `test_f12_openings_only_for_running_sessions` |
| F14 | `test_f14_an_opening_is_credited_to_the_spawn_point_reached` (seed 21, episode 8) |
| F17 | `test_f17_exploration_progress_clears_the_loop_window` (p7_indoor_test_007, episode 29), `test_f17_loop_window_cleared_by_new_coverage` |
| F19 | `test_f19_a_reattributed_opening_is_not_credited_twice` (p7_mixed_test_003, episode 14) |
| F20 | `test_f20_mixed_visual_evidence_alone_does_not_credit_an_opening` (p7_open_test_019, episode 76) |
| F21 | `test_f21_a_closed_looking_start_room_does_not_end_the_episode` (p7_indoor_test_001, episode 5; fails before the fix), `test_f21_the_dense_fan_is_tried_once_per_place` |
| F22 | `test_f22_off_mesh_manoeuvres_without_a_trail_change_direction`, `test_f22_the_stuck_phase6_episode_completes` (Phase 6 benchmark, episode 57; needs the local experience database, skipped without it; both fail before the fix) |
| F23 | `test_f23_comparison_rows_from_other_code_are_not_reused` |

**Lessons from the test record.**

* The unit and integration tests passed while five behaviour defects were present: F17,
  F20, F21, F22, and F19, which one of this audit's own fixes introduced.
* All five were found only by running full benchmarks and checking every credited opening,
  early ending and failed episode against the simulator.
* The episode-replay regression tests now guard those cases. **The general lesson stands:**
  in this project a passing suite is not evidence of correct behaviour. The ground-truth
  cross-checks in the benchmarks (over-reports, false failures, early endings) are what
  found the defects. They should run on every change.

## O. Required next steps

In priority order. Each step names the evidence that would close it.

1. **Build and run the Unreal project (blocking for any readiness claim).** On a machine with
   Unreal Engine 5.4, run the commands in `integration_results.json → unreal →
   commands_for_unreal_machine`:

   ```bash
   python -m chest_hunter export-arena-3d --map research_arena_3d --map p7_complex_test_000_unfamiliar
   <UE>/Engine/Build/BatchFiles/Build.(bat|sh) ChestHunterResearchEditor <Platform> Development \
       -Project=$PWD/unreal/ChestHunterResearch/ChestHunterResearch.uproject
   <UE>/Engine/Binaries/<Platform>/UnrealEditor-Cmd unreal/ChestHunterResearch/ChestHunterResearch.uproject \
       -run=pythonscript -script=$PWD/unreal/ChestHunterResearch/Scripts/create_research_map.py -unattended
   <UE>/Engine/Binaries/<Platform>/UnrealEditor-Cmd unreal/ChestHunterResearch/ChestHunterResearch.uproject \
       -ExecCmds="Automation RunTests ChestHunter; Quit" -unattended -nullrhi
   <UE>/Engine/Binaries/<Platform>/UnrealEditor unreal/ChestHunterResearch/ChestHunterResearch.uproject \
       -game -ResearchEvaluation -windowed -ResX=640 -ResY=360
   python -m chest_hunter check-runtime --backend unreal
   python -m chest_hunter simulate-3d --backend unreal --agent hybrid --map research_arena_3d --evaluate
   ```

   Closing evidence: the module compiles; the automation tests pass; the map exists; one
   traced episode against `--backend unreal` exercises all twelve loop stages; credited
   openings match the Unreal chest state. Then re-run `audit-system` against Unreal.
2. **Measure perception on Unreal frames before trusting any detection number here.**
   Capture a labelled set from SceneCapture (the vision harness takes any frame source with
   labels) and compare it with `vision_metrics.json`. The detector has only seen the
   Python renderer's flat-shaded frames.
3. **Fix F3** (the Unreal arena builder ignores `surface_value`) when the C++ can be
   compiled, so that the appearance shift is reproducible in Unreal.
4. **Perception robustness to chest appearance (F15).** Colour segmentation fails outside
   the trained colour range (G = +0.28 for appearance, +0.37 combined). Candidates, smallest
   first: a shape and edge cue that does not depend on hue; per-map colour calibration from
   the first confirmed chest; a small learned classifier on the existing candidates (only
   with measured latency at 320×180, per the brief). Closing evidence: the appearance gap's
   interval includes 0.
5. **Opened/unopened single-frame confusion (F16).** The F11 look-back mitigates its effect
   on crediting; the classifier itself is unchanged. Closing evidence: opened-class
   precision and recall in `vision_metrics.json` above 0.9.
6. **Reproducible research runs (F18).** Replace the wall-clock frontier budget and the
   inference timeout with work-based limits in lock-step runs, and record overruns and
   timeouts per episode. Closing evidence: the acceptance replay check passes under full CPU
   load.
7. **Exploration completeness (F21 residual).** One map in 100 still ends "complete" in
   mid-episode with 12.5 % coverage. Cast denser fans from earlier fan origins before
   concluding. Closing evidence: no "exploration complete" below 50 % true coverage in the
   robustness set.
8. **Re-run the generalisation benchmark on the final code.** It ran before F11–F21; F21 cut
   18 of its 800 Phase 7 episodes short.
9. **Remaining loop detections in long active-exploration episodes** (after F17). Inspect the
   residual detections in the 600 s acceptance episodes; decide whether each is a true loop.
10. **Coverage within the budget.** Active exploration covers about half of a 90 m map in
   200 s and 74.5 % of the acceptance map in 600 s, below the 85 % engineering target. The
   Phase 6 agent, given spawn points, is two to three times as productive. Any deployment that knows the spawn points should use them.
11. **The learned policy needs human or Unreal demonstrations** before it can be judged. On
   synthetic demonstrations it adds nothing measurable over the deterministic controller
   (section I).
12. **Simplify** where section M shows no benefit, after the Unreal results are in, so the
   simplification is judged on the intended environment.
