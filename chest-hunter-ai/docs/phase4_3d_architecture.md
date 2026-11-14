# Phase 4: 3D chest-hunting agent and Unreal Engine 5 integration

This document describes the third-person 3D extension of chest-hunter-ai: a controlled
research arena, a localhost bridge between the Python agent and the environment, and a
vision-based agent that finds, approaches and opens chests from camera images.

Everything runs in a research environment built for this project. Nothing here touches a
commercial game, its servers, its client process or its anti-cheat; no third-party game
assets are used, and the agent's movement goes through the environment's own control
interface (no operating-system input injection, no gamepad emulation).

## 0. Status: what is verified and what is not

| Component | Where | Status |
|---|---|---|
| Bridge protocol (framing, validation, versioning) | `src/chest_hunter/unreal_bridge/protocol.py` | **Verified** by `tests/test_unreal_protocol.py` |
| Bridge client (loopback-only, timeouts, reconnect, stale frames, step de-duplication) | `unreal_bridge/client.py` | **Verified** against the mock server over real loopback TCP (`tests/test_unreal_bridge.py`) |
| Mock research server (same protocol as the Unreal plugin) | `unreal_bridge/mock_server.py` | **Verified** (same tests) |
| Python reference arena: geometry, capsule physics, spring-arm camera, software renderer, NavMesh-style graph, interaction rules | `src/chest_hunter/simulation3d/` | **Verified** (`tests/test_simulation3d.py`, `tests/test_navigation_3d.py`) |
| 3D perception: detector, distance estimation, tracker, interaction cues | `src/chest_hunter/vision/*_3d.py`, `depth_estimation.py`, `perspective.py` | **Verified on reference-arena frames** (`tests/test_vision_3d.py`, `tests/test_perspective.py`) |
| World model, navigation, controller, runner, database records | `localization/world_model_3d.py`, `navigation/navigation_3d.py`, `agent/controller_3d.py`, `agent/runner_3d.py` | **Verified on the reference arena** (`tests/test_agent_3d.py`, benchmark) |
| Debug dashboard | `agent/dashboard_3d.py` | **Verified** under Xvfb (`tests/test_dashboard_3d.py`) |
| Unreal C++ module (`unreal/ChestHunterResearch/Source`) | C++ | **Prepared, NOT compiled or run.** No Unreal Engine installation was available. Consistency with the Python side is checked statically, and the shared SplitMix64/UnitRoll code is compiled with g++ and compared bit for bit (`tests/test_unreal_sources.py`) |
| Unreal automation tests (`Private/Tests/ResearchProtocolTests.cpp`) | C++ | **Prepared, NOT run** |
| Editor map script (`unreal/ChestHunterResearch/Scripts/create_research_map.py`) | Unreal Python | **Prepared, NOT run** |
| Python agent ⇄ Unreal over the bridge | | **NOT tested.** All end-to-end results in this repository come from the Python reference arena. |

The Python reference arena is **not** Unreal Engine. It exists so that every Python
component (protocol, client, perception, navigation, controller, database, benchmark) can be
built and tested on a machine without Unreal, and so that the Unreal server has an
executable specification. Benchmark numbers from it describe the agent in that arena only;
they say nothing about Unreal's rendering, lighting or physics.

## 1. Architecture

```text
                    ┌──────────────────────── research environment ────────────────────────┐
                    │  Unreal (prepared)                      Python reference arena (verified)│
                    │  AResearchGameMode  ◄─ same rules ─►    simulation3d.Environment3D      │
                    │  AResearchCharacter (spring arm,        Character3D (capsule, spring    │
                    │    SceneCapture2D)                        arm) + Renderer3D (z-buffer)  │
                    │  AResearchChest, arena from JSON        arena.py (same layout JSON)     │
                    │  UNavigationSystemV1 (Recast)           navmesh.NavGrid3D               │
                    │  UResearchBridgeSubsystem ─┐            mock_server.BridgeSession ─┐    │
                    │  FResearchBridgeServer     │            MockUnrealServer           │    │
                    └────────────────────────────┼───────────────────────────────────────┼────┘
                                                 └──── TCP 127.0.0.1:47820, protocol v1 ──┘
                                                                     │
  Python agent ── unreal_bridge.client.BridgeClient ── AgentPort (observe / step / nav_query / nav_raycast)
                    │ privileged calls (oracle_state, evaluation_labels) stay in the runner,
                    │ and only work against a server started in oracle/evaluation mode
                    ▼
  agent.controller_3d.VisualAgent3D (state machine)
     ├─ vision.chest_detector_3d.ChestDetector3D   (colour / template / hybrid)
     ├─ vision.depth_estimation.DistanceEstimator  (apparent size, ground plane, fused)
     ├─ vision.tracker_3d.Tracker3D                (tracks + ray triangulation)
     ├─ vision.interaction_cues_3d.detect_cues     (prompt / progress bar in the image)
     ├─ localization.world_model_3d.WorldModel3D   (spawn priors, statuses, empty votes)
     └─ navigation.navigation_3d                   (NavMeshNavigator | LocalNavigator,
                                                    PathFollower, ScanPlanner)
  agent.runner_3d.Runner3D → SQLite (sessions, chest_observations, simulation_runs)
  analytics.benchmark_3d   → benchmark_results/phase4/{navigation_3d.csv, vision_3d.csv, summary_3d.json}
```

## 2. Unreal Engine project

* **Engine:** Unreal Engine **5.4** (`EngineAssociation` in the `.uproject`). Other 5.x
  versions should work but have not been tried.
* **Tools:** Visual Studio 2022 with the C++ game development workload (Windows), Xcode
  (macOS) or clang with the Unreal Linux toolchain; the Unreal *Python Editor Script Plugin*
  (enabled in the project) for the map script.
* **Content:** no Marketplace or third-party assets. Geometry uses the engine's
  `/Engine/BasicShapes/Cube` and `BasicShapeMaterial` (with a `Color` parameter).

### 2.1 Files

```text
unreal/ChestHunterResearch/
  ChestHunterResearch.uproject
  Config/DefaultEngine.ini         renderer (no Lumen/Nanite/VSM), fixed 30 FPS, Recast NavMesh
                                   (dynamic, AgentRadius 35, MaxStepHeight 55), [ResearchBridge]
  Config/DefaultGame.ini, DefaultInput.ini, DefaultScalability.ini
  Content/ResearchData/research_arena_3d.json   arena layout exported from Python
  Scripts/create_research_map.py   creates /Game/Maps/ResearchArena3D (nav bounds, start, lights)
  Source/ChestHunterResearch/
    Public|Private/ResearchProtocol       framing, validation, SplitMix64/UnitRoll (mirror of protocol.py)
    Public|Private/ResearchChest          chest actor: closed/opened geometry, prompt/progress/reward cues
    Public|Private/ResearchCharacter      third-person character, spring arm, SceneCapture2D, odometry
    Public|Private/ResearchArenaBuilder   spawns the arena boxes from the layout JSON
    Public|Private/ResearchGameMode       episodes, seeded spawns, interaction, nav service, oracle/labels
    Public|Private/ResearchBridgeServer   loopback TCP server thread
    Public|Private/ResearchBridgeSubsystem game-thread command dispatch, synchronous stepping
    Private/Tests/ResearchProtocolTests.cpp automation tests (framing, actions, reference vectors)
```

### 2.2 Build and run (not verified here)

```bash
# 1. Export the layout (already committed; rerun after changing simulation3d/arena.py)
python -m chest_hunter export-arena-3d

# 2. Generate project files and build the editor target (Windows example)
"<UE>/Engine/Build/BatchFiles/Build.bat" ChestHunterResearchEditor Win64 Development \
    -Project="<repo>/chest-hunter-ai/unreal/ChestHunterResearch/ChestHunterResearch.uproject"

# 3. Create the map once
UnrealEditor-Cmd ChestHunterResearch.uproject -run=pythonscript \
    -script="<abs path>/Scripts/create_research_map.py" -unattended -nosplash

# 4. Automation tests
UnrealEditor-Cmd ChestHunterResearch.uproject -ExecCmds="Automation RunTests ChestHunter;Quit" \
    -unattended -nullrhi -nosplash -log

# 5. Start the research server (standard, non-privileged mode) with a small window
UnrealEditor ChestHunterResearch.uproject -game -windowed -ResX=640 -ResY=360 -ResearchPort=47820
#    oracle baseline:      add -ResearchOracle
#    offline evaluation:   add -ResearchEvaluation (also allows the oracle state)

# 6. Run the agent against it
python -m chest_hunter simulate-3d --backend unreal --agent visual
```

Rendering is required for camera images: `-nullrhi` (no rendering) is fine for the protocol
automation tests but **cannot** produce camera observations. An off-screen run still needs a
GPU-backed RHI.

### 2.3 Synchronous stepping

The world is paused between bridge steps. `STEP` sets the character's action, unpauses, lets
the engine advance exactly `ticks` frames of 1/30 s (`FApp::SetUseFixedTimeStep`, fixed
delta 1/30 s, so simulated time does not depend on rendering speed), pauses again, then
optionally renders the agent's view with `USceneCaptureComponent2D::CaptureScene()` and
reads it back (`ReadPixels`, BGRA→BGR). Commands are processed on the game thread by a
`FTickableGameObject` that also ticks while paused. The socket thread hands each request to
the game thread through a lock-free queue and waits for the encoded answer (60 s limit).

## 3. Bridge protocol (version 1)

Transport: one TCP connection to **127.0.0.1** (the server binds the loopback address only
and drops non-loopback peers; the client refuses any host that does not resolve to
loopback). One client at a time; a second one receives `busy` and is disconnected.

Framing (both directions):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `CHRB` |
| 4 | 2 | protocol version (big-endian), currently 1 |
| 6 | 2 | flags (0) |
| 8 | 4 | header length H, 2 ≤ H ≤ 64 KiB |
| 12 | 4 | payload length P ≤ 8 MiB |
| 16 | H | UTF-8 JSON object (no NaN/Infinity) |
| 16+H | P | raw payload (camera image, `bgr8`, rows top to bottom) |

Requests: `{"command", "request_id", "args"}`; responses:
`{"type": "response", "command", "request_id", "status": "ok"|"error", ...}` with
`error: {code, message}` on failure. A malformed frame (`bad_magic`, `version_mismatch`,
`too_large`, `bad_json`, `bad_header`) closes the connection; every other error leaves it
usable. `HELLO` must come first (only `PING` is allowed before it).

| Command | Arguments | Answer |
|---|---|---|
| `HELLO` | `protocol`, `version`, `client?` | server, engine, maps, observation sizes, features (oracle/evaluation flags) |
| `RESET_EPISODE` | `map_id`, `seed`, `episode?`, `observation? {width,height}`, `fov_deg?` (60–110), `graphics_quality?` (low/medium/high), `max_ticks?`, `navigation?` (navmesh/local/none) | `episode_id`, `rules`, `navigation`, `telemetry` |
| `GET_OBSERVATION` | `episode_id` | `observation` metadata + image payload |
| `STEP` | `episode_id`, `step_id` (= last + 1), `action`, `ticks?` (1–30), `observe?` | `telemetry`, `ended`, `duplicate`, optional `observation` + payload |
| `GET_EPISODE_STATUS` | `episode_id` | `state` (running/ended/stopped), `tick`, `sim_time`, `last_step_id` |
| `STOP_EPISODE` | `episode_id`, `reason?` | `state`, `summary` |
| `PING` | `nonce?` | `pong`, `server_time` |
| `NAV_QUERY` | `episode_id`, `start`, `goal` (metres), `goal_tolerance?` | `status_code` (ok/unreachable/start_off_navmesh/goal_off_navmesh), `waypoints`, `length` |
| `NAV_RAYCAST` | `episode_id`, `origin`, `bearings_deg` (1–32), `max_distance?` | walkable `distances` along each bearing |
| `GET_ORACLE_STATE` | `episode_id` | **privileged**: true agent pose, every spawn point's presence/opened state |
| `GET_EVALUATION` | `episode_id`, `frame_id` | **privileged**: labels of the latest rendered frame |

Robustness rules: `STEP` with the previous `step_id` is answered from a cache and never
applied twice (the client resends a step after reconnecting); older ids are `stale_step`,
gaps are `out_of_order`; requests for an old episode are `stale_episode`; the client
rejects frames that are not newer than the last one (`StaleFrameError`). Unknown commands,
unknown argument keys, wrong types and out-of-range values are rejected with
`unknown_command`, `invalid_argument` or `invalid_action`. There is no console, script,
file or arbitrary-execution command.

### 3.1 Coordinates and units

Metres and degrees on the wire, Unreal's axes: **X north/forward, Y east/right, Z up**
(Unreal works in centimetres; the plugin converts). Yaw is the compass bearing from +X
towards +Y; pitch is positive upwards. Images: `u` to the right, `v` down, principal point at
the image centre, square pixels, `fx = (W/2) / tan(hfov/2)`.

### 3.2 Observation

`observation = {frame_id, sim_time, width, height, encoding: "bgr8", intrinsics {width,
height, hfov_deg, near, fx, fy, cx, cy}, camera {yaw_deg, pitch_deg, arm_length,
relative_position}, odometry {position, velocity, grounded}}` and the image as the payload.
Sizes: 320×180, 480×270, 640×360 (default), 800×450. `camera.relative_position` is the
camera relative to the character's feet (the agent knows its own rig). `odometry.position`
is dead reckoning **with seeded random-walk drift** (σ = 0.5 % of the distance moved, per
axis), not the true position. `telemetry.events` carries the player's own notifications
(`chest_opened_notification` with `reward`, `interaction_failed`), never other chests.

### 3.3 Action

`{move_forward, move_right, look_yaw, look_pitch ∈ [-1, 1], jump, interact, sprint?,
crouch?}`. Movement is relative to the camera yaw; `look_*` are rates (fractions of
120°/s yaw and 60°/s pitch); `jump` is an impulse on the first tick; `interact` must be
held for 2 s within 1.8 m, inside a 60° cone of the camera direction and with line of sight
to open the chest (the prompt and a progress bar appear in the world, so the camera sees
them).

## 4. Episodes

Chest presence is deterministic: spawn `i` holds a chest when
`UnitRoll(seed, episode, i, 0) < probability_i` (SplitMix64; identical in Python and C++,
checked bit for bit). Rewards: `UnitRoll(..., salt 999) < 0.3`. The arena has 35 spawn
points in five regions plus a plaza and a field, including indoor rooms, doorways (one
1.3 m wide), stairs to mezzanines and upper floors, obstacles, gold decoys (same colour as
the chests) and two spawns that are unreachable by design (a water-tower platform with no
ladder and a sealed container).

## 5. Perception

* **Detection** (`ChestDetector3D`, methods `color`, `template`, `hybrid`): 3×3 blur, HSV
  classes (chest gold, lid brown, open-chest cream), lid-anchored candidate boxes, then for
  `hybrid` a comparison with 72 label templates (6 yaws × 6 elevations, closed and opened)
  restricted to elevations consistent with the camera pitch; colour/structure acceptance and
  non-maximum suppression. Candidates cut by the frame edge or overlapping the agent's own
  body are marked partial. Templates are generated by `scripts/generate_templates_3d.py`.
* **Distance** (`DistanceEstimator`): method A — apparent size from the matched view's
  physical width/height; method B — the bottom-centre ray intersected with candidate floor
  planes, a plane being plausible when it is the agent's own floor or matches a nearby spawn
  prior's height; the two are fused by inverse variance. Method C — the tracker triangulates
  bearing rays from different viewpoints once the baseline is good enough. Estimates outside
  0.6–40 m or without a valid intersection are rejected.
* **Tracking** (`Tracker3D`): tentative → confirmed (2 hits) → lost → expired, gating at
  1.2 m, a close-range reset when a measurement under 5 m disagrees by more than 3σ.
* **Interaction cues** (`detect_cues`): the white prompt sign and the green progress bar
  above the targeted chest, found in the image.

## 6. Navigation

* **Hybrid (`--agent visual`, benchmark mode B):** the agent asks the environment's
  NavMesh service for a route from its *estimated* position to an *estimated* destination
  (a visually estimated chest position or a spawn prior from the database). The service
  never receives or returns chest positions.
* **Visual navigation (`--agent visual-nav`, mode C):** no route service. The agent builds
  its own occupancy map from `NAV_RAYCAST` probes (16 bearings, 8 m) and plans with the
  Phase 2 A*; it has no knowledge of stairs, so other floors are reported as `other_floor`.
* **Following and recovery:** `PathFollower` steers at each waypoint of the string-pulled
  path, slows before sharp turns and detects lack of progress (2.5 s). When stuck, the
  agent backs off along the way it came (which cannot walk it off a stair or platform edge),
  looks again and replans; after an alignment failure it walks round to another side of the
  chest; a target is given up (prior marked abandoned or failed) after more than three
  recoveries on it. `start_off_navmesh` (mid-air or mid-stairs) is treated as transient.
* **Scanning:** full turns or arcs (`ScanPlanner`) when there is nothing to do, and arcs
  towards unexplored priors.

## 7. Closed-loop controller

States: `INITIALIZING → OBSERVING → SCANNING / EXPLORING → TARGET_DETECTED →
PLANNING_ROUTE → NAVIGATING → APPROACHING → ALIGNING → INTERACTING → VERIFYING_OPEN`, with
`RECOVERING`, `COMPLETED` and `FAILED`. Within 4.5 m the agent switches from the route to a
visual approach, keeps the chest in view over the shoulder (yaw offset −30° within 4.5 m,
−15° within 10 m; pitch not below −32°), stops 1.1 m (+0.3 m) from the chest centre,
aligns until the interaction prompt is visible, holds `interact` while watching the progress
bar, and confirms the opening by seeing the opened chest twice (or once plus the opening
notification). An opening is written to SQLite only after verification
(`chest_observations.reward_received` records the reward; a database trigger forbids a
reward without an opening). Locations are marked empty only after two consistent "nothing
there" views with a clear line of sight.

## 8. Information boundaries

The standard agent (`VisualAgent3D`) receives only an `AgentPort`: camera frames with
their intrinsics, its own camera rig and drifting odometry, notification events, the game
rules, actions, and the route/raycast services that take estimated positions. It has no
method that could request the oracle state or evaluation labels, and the server refuses
those commands unless it was started with `-ResearchOracle` / `-ResearchEvaluation` (the
reference server: `oracle=True` / `evaluation=True`). The camera is a scene capture of the
game view: it shows what a player would see (no hidden-state overlays) and never the
desktop. Ground truth is used only by the oracle baseline (mode A) and by the evaluation
code after or beside the episode, never inside the visual policy. `tests/test_agent_3d.py`
audits the agent's modules (no imports of the simulator or the mock server, no privileged
names) and runs a hybrid episode against a server with every privileged command disabled.

## 9. Performance

Measured on the development container (4 vCPU Xeon 2.1 GHz, no GPU; README section 15.6
has the full tables): the reference renderer takes 16.5 ms per 640×360 frame (median), the
hybrid detector 3.1 ms, a loopback `STEP` returning a 640×360 frame 20.2 ms and one without
a frame 0.9 ms. The agent analyses 2–10 frames per simulated second depending on its state
(2 idle, 3 travelling, 8 searching or approaching, 10 aligning, interacting or verifying).
A hybrid episode costs about 0.19 CPU-seconds per simulated second; peak memory of a worker
process stayed below 117 MiB.

Unreal: not measured (no installation). The project is configured for a low rendering cost
(no Lumen, Nanite or virtual shadow maps, low scalability, 640×360 capture), but a full
UE5 editor needs considerably more than 8 GB of RAM and a dedicated GPU to work comfortably;
**8 GB with integrated graphics is not expected to run the editor reliably**. A packaged
(cooked) development build with a small window is the realistic option on modest machines,
and has not been tried either.

## 10. Benchmark methodology

`python -m chest_hunter benchmark-3d --episodes 50 --workers 3` runs paired episodes: for
each episode number `k` the three modes play the same seed (same chests, same rewards) in
the reference arena for 10 simulated minutes at 640×360. Mode A (oracle) reads the true
chest positions over the privileged channel and plans with the NavMesh service; mode B
(hybrid) sees chests only through the camera and uses the route service; mode C
(visual navigation) sees chests through the camera and has only local probes. Metrics:
chests opened (verified by the agent; checked against ground truth), chests per simulated
minute, availability (present and reachable chests), travel distance, recoveries,
navigation failures, failed interactions, detection precision/recall and distance error
against the reference renderer's labels, CPU time and peak memory. Results are written to
`benchmark_results/phase4/`.

## 11. Limitations

* The Unreal module has not been compiled, run or connected to the Python agent.
* The HSV colour classes and templates are tuned to the reference renderer; Unreal's
  lighting, shadows, tone mapping and anti-aliasing will change colours and edges, so the
  detector will need recalibration (and probably new templates) on Unreal frames.
* Unreal's evaluation labels are approximate (projected boxes plus visibility traces), not
  per-pixel ids like the reference renderer.
* Odometry drift in Unreal uses a different (but equally seeded) random stream from Python,
  so trajectories are not bit-identical across the two environments.
* Opening detection relies on seeing the opened chest; an opened chest seen from behind can
  be misclassified, which delays verification.
* Visual-navigation mode has no stairs knowledge and cannot reach upper floors.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no research server on 127.0.0.1:47820` | Start the Unreal project (section 2.2), or use `--backend reference`. |
| `busy: another client is connected` | Another agent is connected; the client retries with back-off. |
| `forbidden: send HELLO first` | A custom client skipped `HELLO`. |
| `forbidden` on `GET_ORACLE_STATE` | Server not started with `-ResearchOracle`/`-ResearchEvaluation` (by design for the visual agent). |
| `navigation_disabled` | The episode was reset with `navigation: local` or `none`. |
| `version_mismatch` | Python package and Unreal plugin speak different protocol versions. |
| Black images from Unreal | Running with `-nullrhi`; camera observations need rendering. |
| `unknown map 'research_arena_3d'` from Unreal | `Content/ResearchData/research_arena_3d.json` is missing: run `export-arena-3d`. |
