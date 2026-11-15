# chest-hunter-ai

A lightweight research project: an **autonomous agent that navigates a
controlled research environment (2D, and since Phase 4 a third-person 3D
arena), finds simulated chests, opens them, and tries to maximise chests
opened per minute.**

| Phase | Status | What it provides |
|---|---|---|
| 1. Data foundation | Done | SQLite database of chest spawn points, simulation sessions and observations; CLI; tests |
| 2. Autonomous navigation | Done | Grid-world simulator, A* pathfinding, route planning, continuous movement, 10-state agent, Tkinter GUI, headless runs, benchmarking |
| 3. Computer vision and visual navigation | Done | Simulated camera, OpenCV chest/obstacle/prompt detection, object tracking, visual localisation, occupancy mapping, frontier exploration, closed-loop interaction with visual verification, debug overlays, oracle-vs-visual benchmark |
| 4. 3D agent and Unreal Engine 5 integration | Python side done and tested; **Unreal project prepared but not compiled or run** | Localhost bridge protocol, client and protocol-identical reference server; 3D reference arena (capsule physics, spring-arm camera, perspective renderer, NavMesh-style graph); perspective chest detection, distance estimation, 3D tracking; NavMesh-hybrid and NavMesh-free navigation; 14-state closed-loop controller; UE5 C++ project sources; 50-episode benchmark (section 15) |
| 5. Imitation learning, adaptive exploration, generalisation | Done on the Python reference arena; **no human demonstrations recorded, Unreal untested** | Demonstration recorder (teleoperation + synthetic scripted), dataset pipeline, NumPy behaviour cloning, ONNX FP32/INT8 export and CPU inference, policy selector with safety layer, Bayesian spawn estimates, route memory, adaptive planner, classified recovery, procedural maps, paired policy and generalisation benchmarks (section 16) |
| 6. Reliability, planning and optimisation | Done on the Python reference arena; **Unreal untested** | Episode supervisor with state specifications, loop detection and bounded escalation; short-horizon planning; adaptive computation; learned-policy monitors and reliability calibration; partitioned experience; endurance runs with checkpoints; resource monitor and profiler; operations panel; paired benchmark (section 17) |
| 7. Generalisation, self-improving exploration, deployment readiness | Done on the Python reference arena; **Unreal untested** | Procedural environment families with held-out shift variants, active frontier exploration without spawn points, information-aware camera search, uncertainty thresholds, versioned exploration memory and strategy bandit, failure injection and robustness benchmark, generalisation benchmark, startup checks, pinned runtime, JSON logs (section 18) |
| 8. Full-system audit | Done; **verdict UNVERIFIED for the Unreal environment** (never built or run) | Independent audit of Phases 1-7 with executed evidence: inventory, Unreal static audit, traced end-to-end episode, ground-truth isolation, bridge security, vision, navigation, interaction, policies, planners, endurance, generalisation, resources, database; 8 confirmed defects fixed with regression tests (section 19) |

> The simulator is entirely self-contained. The project never talks to a game
> client, game service or network. It never reads or writes game memory or
> injects into processes, never sends keyboard or mouse input to other
> applications, and contains no anti-cheat evasion. The demo map is a
> **fictional** research arena, and its coordinates are not real game data.
> Phase 3 "camera frames" are images the simulator draws for itself. The
> project never captures the screen or windows of any other application.
> Phase 4 adds a self-made 3D research arena: its Unreal Engine project uses
> engine basic shapes only, and its bridge listens on 127.0.0.1 only. Movement
> goes through the environment's own control interface, never through
> operating-system input injection.

![The simulator mid-run: obstacles, the agent following a cyan A* path to its orange-ringed target, opened (green), unopened (gold) and empty (grey) spawn points](docs/gui_screenshot.png)

## Quick start

```bash
cd chest-hunter-ai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -e .

python -m chest_hunter init-db
python -m chest_hunter seed-demo

# Watch the agent (needs Tkinter and a display):
python -m chest_hunter simulate --render

# Or run without a window and print statistics:
python -m chest_hunter simulate --map research_arena_01 --strategy weighted --seed 42 --headless

# Compare strategies on 100 paired episodes:
python -m chest_hunter benchmark --episodes 100 --strategies nearest,weighted

# Phase 3: the vision-based agent on the 20-location acceptance arena
python -m chest_hunter seed-demo --scenario visual_arena
python -m chest_hunter simulate --map visual_arena_01 --agent visual --seed 42 --render
python -m chest_hunter simulate --map visual_arena_01 --agent visual --seed 42 --headless
python -m chest_hunter benchmark-vision --map visual_arena_01 --episodes 100 --seed 42

# Phase 4: the 3D third-person agent on the Python reference arena (no Unreal needed)
python -m chest_hunter seed-demo --scenario research_arena_3d
python -m chest_hunter simulate-3d --backend reference --agent visual --seed 42 --headless-controller
python -m chest_hunter benchmark-3d --episodes 50 --seed 42 --workers 3

# Phase 5: learned and adaptive control (synthetic demonstrations; see section 16)
python -m chest_hunter simulate-3d --backend reference --policy hybrid \
    --model training/checkpoints/bc_synthetic_v1 --adaptive-planning --smart-recovery --seed 42
python -m chest_hunter benchmark-policies --episodes 100 --workers 4

# Phase 6: supervisor, short-horizon planning, adaptive computation; endurance and profiling
python -m chest_hunter simulate-3d --backend reference --agent hybrid --planner short-horizon \
    --supervisor enabled --adaptive-compute --seed 42
python -m chest_hunter endurance-test --episodes 100 --planner short-horizon \
    --output benchmark_results/endurance
python -m chest_hunter profile-agent --episodes 20 --output benchmark_results/profiling
python -m chest_hunter benchmark-phase6 --episodes 100 --seed 42 --workers 4

python -m unittest discover -s tests -v
```

---

## 1. Scope

**In scope:** a controlled 2D grid world built from the Phase 1 chest
locations; procedural obstacles; seeded chest spawns; an autonomous agent that
selects targets, plans A* routes, moves continuously with acceleration
limits, detects absent chests, opens present ones, records every outcome in
SQLite and recovers from problems; GUI and headless execution; paired
benchmarks.

**Phase 3 adds** a vision-based agent that sees the world only through
simulated camera frames (section 14). It uses classical computer vision
(OpenCV + NumPy) on the CPU.

**Still out of scope:** neural networks and other machine learning (including
reinforcement learning), large language models, GPU use and external APIs.

**Never in scope:** live-game automation, matchmaking bots, memory reading,
process injection, input simulation for third-party software, and detection
evasion. Movement looks natural only because of continuous steering and
acceleration. Nothing imitates human input timing.

## 2. Architecture

```text
 cli.py  ──────────────┬───────────────────────────────┬──────────────────────────┐
 (Phase 1 commands)    │ simulate --headless/--render   │ benchmark                │
                       ▼                               ▼                          ▼
              agent/runner.py  SimulationRunner  ◄── simulation/renderer.py   analytics/benchmark.py
              (session + run rows, clock, loop)       (Tkinter, after() loop)  (paired episodes, CSV)
                       │
                       ▼
              agent/controller.py  AgentController (finite-state machine)
                 │            │                 │                     │
                 ▼            ▼                 ▼                     ▼
   navigation/route_planner  navigation/astar  navigation/movement   agent/recorder
   (nearest/weighted/        (A*, path cache,  (smoothing, steering,  (batched observations,
    lookahead scoring)        distance fields)  arrival, stuck)        simulated timestamps)
                 └──────────────┬───────────────────┘                     │
                                ▼                                         ▼
   simulation/environment (World, Environment, spawns, interaction,   Phase 1 repositories
   rewards) · map_generator · collision (GridMap) · entities          (chest, session, run)
                                                                          │
                                                                          ▼
                                                        data/chest_locations.db (SQLite)
```

Design rules:

* **Deterministic.** The simulator uses a fixed timestep (default 1/30
  simulated second) and never reads the wall clock for physics. Every random
  decision uses `random.Random` seeded from `(seed, episode, chest id)`. The
  same inputs give the same episode in headless and GUI mode, at any speed.
* **Plan rarely, move every tick.** A* runs only when a target is selected, a
  path is blocked, the agent is stuck, or the map changes. Paths are cached
  per grid version. Target scoring uses precomputed shortest-path distance
  fields, so choosing among N chests costs N array lookups.
* **No database work on ordinary ticks.** Each session makes one candidate
  query when it starts, writes observations in batches, and finalises once.
  Tests verify that navigation ticks issue zero SQL statements.
* **GUI is optional.** The renderer only drives a runner it is given. Headless
  runs never import Tkinter (also tested).
* **Small dependency footprint.** Phases 1-2 use only the standard
  library. Phase 3 adds NumPy and `opencv-python-headless` (CPU only, no GUI
  backend). They are imported only when a visual agent runs, so headless
  oracle runs, the database commands and the Phase 2 benchmark never load
  them (tested).

### Source layout

```text
chest-hunter-ai/
├── pyproject.toml, README.md, .gitignore
├── config/settings.json          # all settings (Phases 1-3)
├── assets/templates/             # chest, prompt, obstacle and agent templates + palette.json
├── benchmark_results/            # Phase 3 benchmark outputs (CSV + summary.json)
├── data/                         # SQLite DB + Phase 2 benchmark CSVs (git-ignored)
├── docs/gui_screenshot.png, docs/visual_debug_screenshot.png
├── scripts/seed_demo_chests.py, scripts/generate_templates.py
├── src/chest_hunter/
│   ├── cli.py, config.py, database.py, models.py, exceptions.py, logging_config.py
│   ├── chest_repository.py, session_repository.py, run_repository.py, demo_data.py
│   ├── scenarios.py                                    # Phase 3 acceptance arena
│   ├── simulation/   environment.py, map_generator.py, entities.py, collision.py, renderer.py,
│   │                 camera.py, sprites.py, interface.py # Phase 3: frames + action interface
│   ├── vision/       observation.py, preprocessing.py, templates.py, chest_detector.py,
│   │                 obstacle_detector.py, interaction_detector.py, visual_tracker.py,
│   │                 perception_pipeline.py
│   ├── localization/ pose_estimator.py, world_state.py
│   ├── navigation/   astar.py, route_planner.py, movement.py, visual_planner.py, local_controller.py
│   ├── agent/        fsm.py, controller.py, visual_controller.py, states.py, objectives.py,
│   │                 recorder.py, runner.py
│   └── analytics/    metrics.py, benchmark.py, vision_metrics.py, visual_benchmark.py
└── tests/  (20 modules, 316 tests)
```

Additions beyond the requested layout: `run_repository.py` (records the
strategy and seed used by every simulation session), `agent/recorder.py`
(batched observation writes and the simulated clock), `agent/runner.py` (one
runnable episode shared by headless mode, the GUI and the benchmark) and
`tests/test_movement.py`, `tests/test_renderer.py`.

## 3. Hardware requirements and measured performance

Any computer that runs Python 3.11+ is enough. No GPU is used. Measured on a
4-vCPU Intel Xeon 2.1 GHz cloud VM:

| Measurement | Target | Measured |
|---|---|---|
| Headless simulation speed (demo episode, 4,120 ticks) | >= 500 ticks/s | 78,000-128,000 ticks per CPU-second |
| Headless speed including SQLite writes (benchmark average) | - | 59,000-103,000 ticks per CPU-second |
| One-off map setup (generate map + 34 distance fields) | - | 0.55-0.77 s per map |
| GUI simulation rate at 1x | 30 ticks/s | 29.4 ticks/s, 14.7 frames/s (target 15), 1.4% of one CPU core |
| Memory for simulation logic (world, distance fields, episode) | < 100 MB | ~3 MiB Python heap; process RSS grew by 3.6 MiB |
| One full demo episode, headless | - | ~0.05 s CPU after setup |

Phase 3 visual agent, same machine. Acceptance arena `visual_arena_01`, seed
42, one camera frame analysed per perception step:

| Measurement | Measured |
|---|---|
| Full visual episode, headless, 640x360 | 160.1 simulated s in 6.4 CPU s, so about **25x faster than real time** (~750 ticks per CPU-second) |
| Perception steps per episode | 1,210 analyses, 7.6 per simulated second (the rate adapts: 2/s while travelling, 5/s exploring, 10/s near a target or while verifying) |
| Perception time per analysis (detection, marker, occupancy, prompt and progress cues) | 640x360: 3.5 ms mean, 6-9 ms peak · 320x180: 1.4 ms · 480x270: 2.3 ms · 800x450: 5.0 ms |
| Same episode at each resolution | all four open 15 of 15 reachable chests: 320x180 in 162.7 s (60x real time), 480x270 in 140.0 s (35x), 640x360 in 160.1 s (25x), 800x450 in 146.3 s (16x) |
| Memory | Python heap peak 27 MiB during an episode (frames, templates, occupancy grid, tracks). Process RSS 90 MiB after one episode, of which 29 MiB is the OpenCV/NumPy libraries |
| GUI at 1x with every debug overlay on | 26.8 ticks/s (target 30) and 13.4 frames/s (target 15) at 9% of one core. The shortfall is Tk timer scheduling, not CPU. The GUI and headless runs give identical results |

The CPU budget is single-threaded: `cv2.setNumThreads(1)` is set because
OpenCV's thread pool cost more than it saved on these small frames.

## 4. Supported operating systems

* **Linux:** tested with Python 3.11, 3.12 and 3.13. The GUI was tested with
  Python 3.12 + Tk 8.6 under Xvfb.
* **macOS, Windows 10/11:** expected to work but not tested yet. The code is
  pure Python using `pathlib`, with ASCII-only console output. NumPy and
  `opencv-python-headless` publish wheels for all three platforms.

The GUI needs Tkinter. It ships with the python.org installers for Windows
and macOS; on Debian/Ubuntu install `python3-tk`. Headless mode, the
benchmark and every other command work without Tkinter.

## 5. Installation

```bash
python3.11 -m venv .venv          # any Python >= 3.11
source .venv/bin/activate         # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .        # editable install; pulls numpy + opencv-python-headless
```

Dependencies: Phases 1-2 need only the standard library. Phase 3 adds
`numpy>=1.26` and `opencv-python-headless>=4.8` (no GPU, no GUI backend; the
GUI uses Tkinter). Without installing, `pip install numpy
opencv-python-headless` and prefix commands with `PYTHONPATH=src`.

This installs the `chest_hunter` package and a `chest-hunter` console
command, so `chest-hunter ...` is equivalent to `python -m chest_hunter ...`.
Without installing, prefix commands with `PYTHONPATH=src`.

## 6. Running the agent

**GUI**

```bash
python -m chest_hunter simulate --render                    # default map/strategy/seed
python -m chest_hunter simulate --render --strategy nearest --seed 7 --speed 5
```

The window opens and the agent starts immediately. Controls:

* **Buttons:** Start, Pause, Resume, Reset.
* **Speed:** 1x, 2x, 5x or 10x. Faster speeds run more fixed ticks per
  frame, so the results are identical at any speed.
* **Strategy:** applied on the next Start or Reset.
* **Show paths:** toggles the navigation path.
* **Keys:** Space pauses or resumes, S starts, R resets, Q or Escape quits.

Closing or resetting a running episode aborts its session. When the window
closes, the terminal prints the statistics of the last finished episode.

**Headless**

```bash
python -m chest_hunter simulate --map research_arena_01 --strategy weighted --seed 42 --headless
python -m chest_hunter simulate --strategy lookahead --episode 3 --max-ticks 1800
```

| Option | Meaning | Default |
|---|---|---|
| `--map` | map whose enabled chest locations are loaded | `simulation.default_map` |
| `--strategy` | `nearest`, `weighted` or `lookahead` | `navigation.default_strategy` |
| `--seed` | seed for obstacles, spawns, failures and rewards | `simulation.default_seed` |
| `--episode` | different chest spawns on the same map | 0 |
| `--max-ticks` | stop after N ticks (outcome `tick_limit`) | `simulation.max_ticks` |
| `--speed` | initial GUI speed (1, 2, 5, 10). Headless runs are never throttled | 1 |
| `--agent` | `oracle` (Phase 2, perfect map knowledge), `visual` (Phase 3, camera only) or `chest-vision` (camera for chests, true obstacle map) | `oracle` |
| `--headless` / `--render` | no window (default) / Tkinter window | headless |

`simulate` exits with 0 when the episode completes or hits the tick limit,
and with 1 if the agent fails or an error occurs.

## 7. Configuration

`config/settings.json` holds every setting. Phase 1 keys are required. Every
Phase 2 key is optional and defaults to the values below, so Phase 1 config
files still load. Unknown keys inside a known section are rejected (typos),
and values are range-checked.

| Key | Default | Unit / meaning |
|---|---|---|
| `simulation.map_width`, `map_height` | 100 | grid cells (10-1000) |
| `simulation.cell_size` | 8 | screen pixels per cell (GUI only) |
| `simulation.world_units_per_cell` | 10.0 | database coordinate units per cell (demo arena: 1000 units = 100 cells) |
| `simulation.tick_rate` | 30 | fixed ticks per simulated second |
| `simulation.render_rate` | 15 | GUI frames per real second |
| `simulation.default_seed` | 42 | seed for map, spawns, failures and rewards |
| `simulation.max_ticks` | 100000 | tick budget per episode (~55 simulated minutes) |
| `simulation.player_speed` | 5.0 | cells per simulated second |
| `simulation.player_acceleration` | 12.0 | cells/s² (also used for braking) |
| `simulation.player_radius` | 0.3 | cells (collision circle, below 0.5) |
| `simulation.agent_start` | `null` | `[x, y]` in database units; `null` = map centre |
| `simulation.interaction_radius` | 1.5 | cells; the agent must be this close **with line of sight** |
| `simulation.chest_interaction_seconds` | 2.0 | simulated seconds to open a chest |
| `simulation.interaction_failure_probability` | 0.0 | chance (0-1) that an attempt fails, to exercise recovery |
| `simulation.reward_probability` | 0.0 | chance (0-1) of a fictional token per opened chest (try 0.1 in testing) |
| `simulation.obstacle_density` | 0.12 | target fraction of blocked cells (0-0.4) |
| `navigation.allow_diagonal` | true | 8-directional movement (otherwise 4-directional) |
| `navigation.prevent_corner_cutting` | true | diagonal moves need both side cells free |
| `navigation.cache_paths` | true | cache A* results for the static map |
| `navigation.default_strategy` | `weighted` | `nearest`, `weighted` or `lookahead` |
| `navigation.stuck_timeout_seconds` | 5.0 | simulated seconds without progress before recovering |
| `navigation.smooth_paths` | true | line-of-sight shortcutting of A* paths |
| `navigation.lookahead_depth` | 2 | chests per sequence for `lookahead` (1-3) |
| `agent.max_recovery_attempts` | 3 | recoveries per target before abandoning it |
| `agent.arrival_tolerance` | 0.5 | cells inside `interaction_radius` where the agent aims to stop |
| `agent.observation_batch_size` | 10 | observations per SQLite transaction |

Phase 3 sections (all optional, with the defaults shown, so Phase 1-2
config files still load):

| Key | Default | Unit / meaning |
|---|---|---|
| `simulation.decoration_density` | 0.004 | fraction of free cells holding gold-coloured decoy props (false-positive bait) |
| `vision.enabled` | true | `false` refuses to start the `visual` and `chest-vision` agents |
| `vision.default_width`, `default_height` | 640, 360 | camera frame in pixels: 320x180, 480x270, 640x360 or 800x450 |
| `vision.view_width_cells` | 40 | grid cells across the frame; width must be a multiple (640 / 40 = 16 pixels per cell) |
| `vision.detector` | `hybrid` | chest detector: `color`, `template` or `hybrid` (section 14.4) |
| `vision.confidence_threshold` | 0.85 | minimum template-matching score (0-1) |
| `vision.max_tracks` | 100 | upper bound on remembered object tracks |
| `vision.track_expiration_seconds` | 3.0 | simulated seconds an **in-view** track may go undetected before it expires |
| `vision.track_memory_seconds` | 600.0 | simulated seconds an out-of-view track is remembered |
| `vision.use_regions_of_interest` | true | search for the agent marker near its predicted position first (`false` = whole frame) |
| `vision.noise_std` | 6.0 | per-pixel Gaussian camera noise (0-255 intensity levels) |
| `vision.lighting_variation` | 0.08 | per-frame brightness change, ± fraction |
| `vision.frame_dropout` | 0.0 | probability (0-0.9) that a capture fails |
| `vision.rate_travel`, `rate_explore`, `rate_tracking`, `rate_verify`, `rate_idle` | 2, 5, 10, 10, 1 | analyses per simulated second in each situation |
| `vision.template_dir` | `assets/templates` | template images and `palette.json` |
| `localization.smoothing_alpha` | 0.4 | weight of a new visual fix against the motion prediction (0-1) |
| `localization.max_missing_observations` | 5 | consecutive frames without the marker before localisation is lost |
| `localization.min_confidence` | 0.5 | below this pose confidence the agent stops and re-observes |
| `localization.dead_reckoning_seconds` | 2.0 | seconds without a fix until pose confidence reaches 0 |
| `visual_navigation.exploration_strategy` | `frontier` | exploration method (`frontier` is the only one implemented) |
| `visual_navigation.replan_on_obstacle_discovery` | true | replan when a newly seen wall blocks the current path |
| `visual_navigation.maximum_recovery_attempts` | 3 | recoveries per target before it is marked failed or abandoned |
| `visual_navigation.use_spawn_priors` | true | spawn points attract frontier exploration (`false` = pure frontier search; spawn points are still resolved when seen) |
| `visual_navigation.prior_weight` | 300.0 | information-gain units per unit of spawn probability |
| `visual_navigation.stop_when_spawn_points_resolved` | true | finish once every spawn point is resolved, instead of exploring the whole map |
| `visual_navigation.unknown_cell_cost` | 0.5 | extra A* cost for entering an unexplored cell |
| `visual_navigation.empty_confirmations` | 2 | in-view analyses without a chest before a spawn point counts as empty |
| `visual_debug.enabled`, `show_detections`, `show_object_tracks`, `show_occupancy_grid`, `show_localization`, `show_camera_view` | true, true, true, false, true, true | initial state of the GUI debug toggles |

Overrides: `--config PATH` or `CHEST_HUNTER_CONFIG`, `--db PATH`,
`--log-level LEVEL`, `CHEST_HUNTER_HOME`. Relative paths resolve against the
project root, never the current directory.

The later sections are optional too and documented with their phases:

* `unreal_bridge`, `simulation_3d`: Phase 4 (`docs/phase4_3d_architecture.md`);
* `learning`: Phase 5 (`docs/phase5_learning_architecture.md`);
* `supervisor`, `planning`, `performance`, `endurance`: Phase 6 (section 17.6).

## 8. The simulator

* **World mapping.** Each enabled chest location from SQLite is placed at
  `(x, y) / world_units_per_cell`; `z` is ignored in 2D. Locations outside
  the grid are excluded and listed in the run's parameters. Grid `y` points
  north and the renderer flips it.
* **Map generation** (`map_generator.py`). The generator is deterministic
  per seed. It builds walls one cell thick with two doorways, wall segments
  and 1x1-3x3 rocks until `obstacle_density` is reached. It keeps a walkable
  one-cell border, and a 5x5 obstacle-free zone around every chest and the
  start. Obstacles are rectangles rasterised into a `bytearray` grid.
* **Spawning.** At the start of each episode a chest exists with its
  location's `spawn_probability`. Over 5,000 samples at p = 0.3 the observed
  frequency was within ±0.03 (tested). Presence depends only on
  `(seed, episode, chest id)`, so every strategy sees identical spawns.
* **Interaction.** A chest opens only if it exists, is not already open, the
  agent is within `interaction_radius` with line of sight, and
  `chest_interaction_seconds` have elapsed. Presence is only perceived from
  within range: the agent never knows in advance which spawn points are
  empty.
* **Fictional rewards.** A separate `TokenRewardModel` rolls
  `reward_probability` after each successful opening. It is a synthetic
  signal for testing reward reporting and says nothing about any real game.
* **Episode reset.** `Environment.reset(episode)` respawns chests, restores
  the agent, resets the clock and drops any obstacles added at runtime.

## 9. The autonomous agent

| State | What happens |
|---|---|
| `INITIALIZING` | The runner opens a SQLite session and a `simulation_runs` row, loads candidate locations (`get_unvisited_chests`) and spawns the episode |
| `SELECTING_TARGET` | Scores pending candidates. Unreachable ones are marked and reported. Moves to COMPLETED when nothing is left |
| `PLANNING_ROUTE` | A* from the agent's cell to the target, then smoothing. Goes straight to VERIFYING if already in range; an A* failure marks the target unreachable |
| `NAVIGATING` | Continuous steering along waypoints. Detects arrival, being stuck, and paths invalidated by new obstacles |
| `VERIFYING_CHEST` | Perceives presence. An empty spot is recorded as an observation and the agent moves on |
| `INTERACTING` | Brakes and holds position for `chest_interaction_seconds`. Leaving range means recovery |
| `CONFIRMING_OPEN` | A success is recorded (opened, duration), with a reward roll. A failure is recorded as a failed interaction and leads to recovery |
| `RECOVERING` | Bounded: retry the interaction or replan from the current position. Abandons the target after `max_recovery_attempts`. Whole-episode budget: (attempts + 1) x candidates |
| `COMPLETED` | Flushes observations, completes the session and run row, stops the agent |
| `FAILED` | Records the reason, aborts the session safely |

Safety against endless loops: at most 16 state transitions per tick, bounded
recoveries per target and per episode, and the tick limit. Every transition
is checked against an explicit table (`agent/states.py`).

**Movement** (`navigation/movement.py`):

* **Waypoints.** A* cell paths are shortened by line-of-sight smoothing that
  accounts for the agent's radius.
* **Steering.** Desired velocity is limited by `player_speed`, by the turn
  angle at the next waypoint (the agent slows for sharp corners) and by a
  braking profile `v = sqrt(2·a·d)`. Velocity changes are capped at
  `acceleration × dt`.
* **Waypoint progress.** A waypoint counts as passed once the agent crosses
  the line through it, which prevents oscillating around it.
* **Arrival.** The agent stops inside the interaction radius, not on top of
  the chest.
* **Collisions.** Circle-versus-grid collision with axis-separated sliding.
  Tests confirm the agent never overlaps an obstacle on any tick of a full
  demo episode.
* **Stuck detection.** No progress along the remaining path for
  `stuck_timeout_seconds` triggers recovery.

## 10. Route planning

For each reachable candidate *i* (Q_i = 1; unreachable ones get Q_i = 0 and
are excluded):

```text
T_i = path_length_i / player_speed + player_speed / acceleration + P_i · estimated_open_seconds_i
S_i = P_i · Q_i / T_i
```

* **Path length.** `path_length` is the shortest-path cost (identical to
  A*'s cost; tested), never straight-line distance.
* **Leg overhead.** The `speed / acceleration` term is the time lost
  accelerating away and braking on each leg.
* **Interaction time.** The interaction term is the expected time: an empty
  spot costs no interaction time.
* **Estimates vs truth.** `P_i` and `estimated_open_seconds_i` come from the
  database. These are the agent's estimates, while the simulator's true
  opening time is `chest_interaction_seconds`.

| Strategy | Rule |
|---|---|
| `nearest` | minimum travel time (ties: lower chest ID) |
| `weighted` | maximum S_i (expected openings per second) |
| `lookahead` | short-horizon cluster planning: for each first chest, the best rate ΣP / ΣT over sequences of up to `lookahead_depth` chests (beam of 8), then pick the best first chest |

The plan is recalculated after every observation. The strategy, seed,
episode and all parameters are stored in `simulation_runs` for every
session.

## 11. Database structure (schema version 6)

Phase 4 added migration 3 (`reward_received`, `navigation_failures`;
section 15.7) and Phase 5 migration 4 (`spawn_estimates`, `route_memory`; section 16.3).
Phase 6 added migration 5 (`route_memory.partition_key`) and migration 6
(`episode_experience`), both additive (section 17).
The rest of this section describes migrations 1 and 2.

Phase 1 tables are unchanged. Migration 2 adds one table and
`python -m chest_hunter init-db` upgrades a Phase 1 database in place. Every
command except `init-db` also applies pending migrations automatically.
Existing data is preserved (tested).

* **`chest_locations`**: possible spawn points. Columns: `map_id`,
  `location_name`, `x`, `y`, `z`, `spawn_probability` (0-1),
  `estimated_open_seconds` (>= 0), `enabled`, and UTC timestamps.
  `UNIQUE (map_id, x, y, z)`. There is no `visited` column.
* **`simulation_sessions`**: one per run, `running -> completed | aborted`.
  Final states are immutable (trigger).
* **`chest_observations`**: one row per visit outcome: `chest_present`,
  `chest_opened`, `interaction_seconds`, `observed_at`. Enforced rules:
  opened requires present, the session must be running and on the same map,
  and there is at most one successful opening per location per session.
* **`simulation_runs`** (new): one row per agent-driven session. Columns:
  `strategy`, `seed`, `episode`, `parameters_json` (every simulation,
  navigation and agent setting used, plus `time_basis: "simulated"`),
  `outcome` (`completed`, `tick_limit`, `failed` or `cancelled`),
  `failure_reason`, `ticks`, `simulated_seconds`, `travel_distance` (cells),
  `recoveries`, `unreachable_targets`, `abandoned_targets`, `rewards`, and
  timestamps. A finished run is immutable.

How the agent uses Phase 1 records:

* **Observations.** Absent chests are recorded as
  `present=0, opened=0, seconds=0`, successful openings as
  `present=1, opened=1, seconds=<duration>`, and failed attempts as
  `present=1, opened=0`. A failed attempt counts as a "failed interaction",
  as defined in Phase 1.
* **Unreachable locations** are *not* written as observations, because
  nothing was observed. They are counted in `simulation_runs` and in the
  printed report.
* **Timestamps are on the simulated timeline.** `started_at` is the real
  start time, and every later timestamp is `started_at + simulated elapsed
  time`. Session duration and chests per minute therefore reflect simulated
  time. `record_observation` gained an optional keyword-only `observed_at`
  argument (backwards compatible) so batched writes keep each event's own
  time.

## 12. CLI reference

| Command | Purpose |
|---|---|
| `init-db` | Create the database or apply pending migrations (never erases data) |
| `seed-demo [--map MAP] [--scenario research_arena\|visual_arena\|research_arena_3d]` | Insert the 35 fictional demo locations, the 20-location Phase 3 acceptance arena `visual_arena_01`, or the 35 spawn points of the Phase 4 3D arena (idempotent) |
| `list-chests`, `show-chest`, `add-chest`, `update-chest`, `delete-chest` | Manage spawn points |
| `start-session`, `record-observation`, `complete-session`, `abort-session`, `list-sessions`, `unvisited-chests` | Manual sessions (Phase 1) |
| `session-stats --session ID` | Session statistics. Agent-driven sessions also show strategy, seed, outcome, travel and recoveries |
| `map-stats [--map MAP] [--min-samples N]` | Map statistics, including observed spawn frequency |
| `simulate ... [--agent oracle\|visual\|chest-vision]` | Run the autonomous agent (sections 6 and 14) |
| `benchmark ...` | Paired strategy comparison (section 13) |
| `benchmark-vision [--map M] [--episodes N] [--seed S] [--modes LIST] [--max-ticks N] [--output-dir DIR] [--detector-frames N] [--quiet]` | Oracle vs chest-vision vs visual agent on paired episodes, plus a detector comparison (section 14.12) |
| `simulate-3d [--agent visual\|hybrid\|visual-nav\|oracle] [--backend unreal\|reference] [--host H] [--port P] [--seed S] [--episode N] [--observation-width PX] [--observation-height PX] [--max-ticks N] [--detector color\|template\|hybrid] [--graphics-quality low\|medium\|high] [--evaluate] [--headless-controller\|--render-debug]` | One 3D episode against the Unreal research server or the Python reference arena (section 15) |
| `benchmark-3d [--episodes N] [--seed S] [--modes LIST] [--workers N] [--max-ticks N] [--observation-width PX] [--observation-height PX] [--output-dir DIR] [--quiet]` | Oracle vs hybrid vs visual-navigation on paired 3D episodes (section 15.5) |
| `export-arena-3d [--output PATH]` | Write the 3D arena layout JSON used by the Unreal arena builder (needs no database) |
| `record-demo [--controller scripted\|human] [--map M ...] [--episodes N] [--seed S] [--output DIR] [--experience-db PATH] [--workers N] [--max-ticks N] [--backend reference\|unreal]` | Record demonstrations (human teleoperation window, or the scripted controller labelled synthetic) |
| `train-policy --dataset DIR [--output DIR] [--train-config JSON] [--epochs N] [--batch-size N] [--learning-rate X] [--seed S] [--split-by episode\|map] [--no-export]` | Train the behaviour-cloning policy (episode-level splits, early stopping) and export it |
| `export-policy [--model DIR] [--dataset DIR] [--no-int8]` | ONNX FP32/INT8 export, parity checks, CPU latency |
| `evaluate-policy [--model DIR] [--episodes N] [--policy learned\|hybrid] [--map M] [--quantized]` | Held-out test metrics and paired episodes against the deterministic controller |
| `benchmark-policies [--episodes N] [--strategies LIST] [--map M] [--model DIR] [--experience-db PATH] [--workers N] [--online]` | Paired strategy comparison with confidence intervals (section 16.4) |
| `benchmark-generalization [--familiar N] [--unseen N] [--episodes N] [--strategies LIST] [--model DIR]` | Familiar (training) vs unseen (test) procedural maps with online adaptation |
| `generate-maps --split train\|test [--count N] [--start N] [--export-unreal] [--export-dir DIR]` | Procedural maps (database spawn points with an uninformative prior; Unreal layouts) |
| `estimate-spawns [--map M] [--prior configured\|uniform] [--prior-strength S] [--store]` | Bayesian spawn estimates from recorded observations |

Global options (`--config`, `--db`, `--log-level`) may appear before or
after the command name. Exit codes: 0 success, 1 failure, 2 invalid usage,
130 interrupted. Ordinary errors are a single `error: ...` line, never a
traceback.

Definitions carried over from Phase 1:

* **Failed interaction:** a chest was present but not opened.
* **Chests opened per minute:** `opened / (duration / 60)`. Unavailable while
  the session is running or if its duration is 0.
* **Observed spawn frequency:** computed from distinct (session, location)
  samples. Unavailable below `--min-samples` (default 5).

## 13. Benchmarking and results

```bash
python -m chest_hunter benchmark --episodes 100 --strategies nearest,weighted,lookahead
python -m chest_hunter benchmark --episodes 100 --vary-obstacles --max-ticks 1800
```

What the benchmark holds constant and what it reports:

* **Paired design.** For each episode index every strategy runs on the same
  map, obstacles, chest spawns, start position, speed and tick budget.
  `--vary-obstacles` gives every episode index its own layout, still shared
  by all strategies.
* **Storage.** It runs on an in-memory copy of the map's chest locations, so
  the research database receives no benchmark sessions unless `--persist` is
  given.
* **Output.** A terminal table (mean and median chests per minute, standard
  deviation, mean travel distance, openings, failed interactions,
  recoveries, unreachable or abandoned targets, CPU time) plus a paired
  win/loss summary. The CSV (default `data/benchmark_results.csv`) has one
  row per episode.
* **Time basis.** Chests per minute is measured in **simulated time** (fixed
  1/30 s timestep), so it is independent of host speed. CPU time is real
  process time, including SQLite writes.

### Measured results (demo map `research_arena_01`, seed 42, 100 paired episodes per strategy)

**A. Fixed map, full sweep** (every reachable location visited):

| Strategy | Chests/min mean | median | Travel (cells, mean) | Opened | Recoveries | Unreachable | CPU s/episode |
|---|---|---|---|---|---|---|---|
| Nearest | **11.711** | **11.791** | 332.8 | 2581 | 0 | 0 | 0.040 |
| ProbabilityWeighted | 11.110 | 11.188 | 368.4 | 2581 | 0 | 0 | 0.045 |
| Lookahead | 10.587 | 10.661 | 400.6 | 2581 | 0 | 0 | 0.065 |

Paired: Weighted was 0.601/min below Nearest on average and worse in 100 of 100 episodes.

**B. Different obstacles each episode, full sweep:** Nearest 11.416 (median
11.512), Weighted 10.674 (10.723), Lookahead 10.470 (10.494). Weighted was
worse than Nearest in 99 of 100 episodes. There were 1 and 4 recoveries, all
resolved.

**C. Different obstacles each episode, 60 s budget:** Nearest 11.840 (median
12.0), Weighted 11.120 (11.0), Lookahead 11.680 (12.0). Lookahead against
Nearest: better in 28, worse in 37, tied in 35 episodes.

**Control experiment.** The same coordinates with spawn probabilities
alternating between 0.05 and 1.0, 50 paired episodes on varied maps:

* With a 30 s budget, Weighted scored **13.84** chests/min against Nearest's
  11.00 (better in 47 of 50).
* With a 60 s budget, Weighted scored **11.00** against 10.06 (better in 40
  of 50).

**Interpretation.**

* **Full sweeps favour Nearest.** The agent cannot know in advance whether a
  spawn point is empty, so a full sweep must visit everything. Total time is
  then dominated by tour length, and greedy nearest-neighbour gives the
  shortest tour.
* **The demo data blunts the weighted strategy.** Its probabilities only span
  0.5-0.95, so detouring for likely chests rarely pays for the extra travel.
* **Weighted wins where it should.** When probabilities differ strongly and
  time is limited, probability weighting is clearly better. This confirms
  the scoring works as designed.

The default strategy stays `weighted` as the spec requests. For the demo map,
`nearest` gives the highest throughput.

## 14. Phase 3: visual perception and navigation

The Phase 2 agent reads the simulator's state directly: it knows where walls
are, which spawn points exist and exactly where it stands. The Phase 3
**visual agent** knows none of that. It can capture a camera frame (a BGR
image, plus the camera's geometry: see section 14.7), `move(vx, vy)` and
`hold_interact()`. Everything else it has to work out from pixels.

| Agent (`--agent`) | Chests | Obstacles | Own position | Use |
|---|---|---|---|---|
| `oracle` | true positions and presence (once in range) | true grid | true | Phase 2 baseline, benchmark mode A |
| `chest-vision` | camera only | true grid (oracle) | marker in the frame | isolates the cost of seeing chests, mode B |
| `visual` | camera only | camera only (occupancy map) | marker in the frame | full visual agent, mode C |

All three get the same spawn **priors** from SQLite: location and spawn
probability, exactly the Phase 1 data. A prior is a hint, never proof: the
agent must see a chest before it targets it, and must see an empty spot
(twice) before it records the spot as empty.

![Visual debugging: the observer view (left, ground truth for humans), the agent's 320x180 camera thumbnail with detections, its occupancy map (black = unknown), and live perception statistics](docs/visual_debug_screenshot.png)

The large map on the left is the **observer's** view of simulator truth and
is never given to the agent. The panel on the right shows what the agent
actually has.

### 14.1 Closed loop: OBSERVE -> DETECT -> TRACK -> PLAN -> ACT -> VERIFY

```text
  simulation/environment (truth) ──► camera.FrameRenderer + SensorModel (noise, lighting, dropout)
                                                │ BGR frame                    ▲ move / hold_interact
                                                ▼                              │
  1 Observation provider    vision/observation.py   VisualSimulatorInterface (the only door to the simulator)
                                                │ VisualObservation(frame, timestamp, id, camera)
  2 Perception engine       vision/preprocessing ─► chest_detector (A colour / B template / C hybrid + NMS)
                            vision/obstacle_detector · interaction_detector (prompt, progress bar, reward)
                            localization/pose_estimator.AgentMarkerDetector · vision/perception_pipeline
                                                │ detections, occupancy update, marker fix, cues
  3 World-state estimator   vision/visual_tracker (tentative/confirmed/lost/expired)
                            localization/pose_estimator (predict + correct) · localization/world_state
                            (occupancy grid UNKNOWN/FREE/BLOCKED, spawn-prior statuses, planning grid)
                                                │ EstimatedWorldState
  4 Navigation / planning   navigation/visual_planner (chest goals, frontiers S = G / (T + eps), A* with
                            unknown-cell penalty, blacklists) · navigation/local_controller (confidence gate)
                                                │ velocity commands
  5 Action / verification   agent/visual_controller  (13-state FSM: align, interact, verify, recover)
```

* **One way in, one way out.** The policy modules (vision, localisation,
  visual planner, local controller, visual controller) never import the
  simulator. They talk to `VisualSimulatorInterface`, which keeps the
  environment in a name-mangled private attribute and exposes only the map
  size, `dt`, the clock, `capture()`, `move()`, `hold_interact()` and
  `end_tick()`.
* **Ground truth is used only** to draw frames, for the accuracy evaluator
  (`analytics/vision_metrics.py`, which observes results and never feeds
  back), for the benchmark, and for the explicit oracle and chest-vision
  modes. Three tests enforce this: an AST audit of the policy modules'
  imports; a check that the controller holds no simulator objects; and a
  behavioural test where chests that exist in the simulator are simply not
  drawn. The agent then records those spawn points as empty, so it cannot be
  reading truth.
* **Perception runs at a scheduled rate**, not every tick: 2 analyses per
  simulated second while travelling, 5 while exploring, 10 near a target or
  while verifying, 1 when idle. Movement runs every tick on the predicted
  pose.

### 14.2 Camera and frames

* **Orthographic top-down follow camera.** It shows 40 x 22.5 cells, so
  640x360 gives 16 pixels per cell. The camera moves in whole-cell steps
  when the agent leaves a central dead zone, and stops at the map edge.
  Supported frames are 320x180, 480x270, 640x360 (default) and 800x450.
* **Sprites** are drawn by OpenCV at sub-pixel precision (fixed-point
  coordinates). They include dark ground, slate walls, gold chests with a
  brown lid, and opened chests (dark body, bright inside). The agent is a
  blue disc with a white nose. Overlays are an "E" prompt above an
  interactable chest, a green progress bar while holding, and a cyan reward
  flash. Gold **decoy props** (`decoration_density`)
  test false-positive rejection.
* **Sensor model.** Gaussian pixel noise (sigma 6), per-frame brightness
  change of up to ±8%, and optional capture failures. Seeded, so runs
  repeat exactly.
* **Templates** live in `assets/templates/` and are generated from the
  same drawing code by `scripts/generate_templates.py`. They are scaled once
  per pixels-per-cell and cached.

### 14.3 Preprocessing

`Preprocessor.prepare()` builds a `PreparedFrame`. Colour masks (BGR
ranges around the `palette.json` colours, widened for the noise and
lighting model), grey, HSV and edge images are computed lazily, at most once
per frame, so a detector pays only for what it uses. Identical consecutive frames are recognised and not processed
twice.

### 14.4 Chest detection

| Method (`vision.detector`) | How it works | Measured |
|---|---|---|
| A `color` | gold-body blobs (small morphological closing, connected components), then shape checks: size relative to pixels-per-cell, aspect ratio, fill, a lid stripe | recall 0.986-0.993 at 480x270 and above, but **0.386 at 320x180** (the lid is about 1 px tall); 0.25-1.75 ms |
| B `template` | `TM_CCOEFF_NORMED` matching of the unopened and opened templates over the whole frame (3x3 blur against noise), peaks above `confidence_threshold` | recall 0.952-0.959; 11-61 ms, 15-21x slower than hybrid |
| C `hybrid` (default) | colour candidates -> template score in a small window around each -> colour composition decides opened vs unopened. Accepts a slightly lower score when the shape checks pass, a strong shape alone (partial occlusion by the agent), and frame-edge partial chests | **recall 0.972-0.993 at every size; 0.8-3.7 ms** |

All methods return `Detection(category, bbox, confidence, timestamp,
method)` and finish with non-maximum suppression (IoU 0.3). Detections are
converted to world coordinates with the frame's camera geometry
(section 14.7).

The measurements come from `python scripts/compare_detectors.py`, saved to
`benchmark_results/detector_by_resolution.json`. Each resolution gets the
same 60 labelled frames, with every spawn point holding a chest and about
40% of them opened. Every method had precision 1.000 at every resolution:
no false positives and no opened/unopened confusions. Rejection of the gold
decoy props is also tested directly. Times are detector milliseconds per frame on one core, measured
while two benchmark processes used two of the other cores.

### 14.5 Occupancy mapping

The agent keeps an `OccupancyGrid` of the whole map: `UNKNOWN = -1`,
`FREE = 0`, `BLOCKED = 1`. Each analysis takes the wall-coloured mask,
averages it per cell with `cv2.resize(..., INTER_AREA)`, and updates the
visible window: >= 0.5 wall pixels means BLOCKED, <= 0.2 means FREE.
Chests, props and overlays are not wall-coloured, so they count as walkable.
A cell that is only partly wall-coloured keeps its previous value. The grid
starts fully unknown, grows as the agent explores, and picks up walls that
appear at runtime, which triggers a replan when they cut the current path.
Tests check that every observed cell matches ground truth and that unseen
cells stay unknown.

### 14.6 Object tracking

`VisualTracker` associates detections with tracks by greedy
nearest-neighbour matching in **world** coordinates (gate 0.9 cells, so
camera motion does not break identity). A track starts **tentative**,
becomes **confirmed** after repeated hits, is **lost** while out of view
(kept for `track_memory_seconds`) and **expires** after going unseen for
`track_expiration_seconds` *while it should be visible*. Misses are not
counted where the agent's own sprite covers the chest. Opening a chest
keeps its track (category changes from `chest` to `chest_opened`). Memory is
bounded by `max_tracks`: when full, the stalest unconfirmed track (then the
oldest) is evicted.

### 14.7 Localisation

* **Marker.** The agent marker is found by colour (searching near the
  predicted position first when `use_regions_of_interest` is on). Its
  centroid gives the position and its white nose gives the heading.
* **Transform.** Pixels are converted to world coordinates with the camera
  geometry delivered with each frame (origin and pixels per cell), as for a
  calibrated camera whose pose is known. This is the one piece of
  geometric metadata the agent receives. The camera moves in whole-cell
  steps behind a dead zone, so it only coarsely follows the agent. The
  agent's exact position, heading, chests and walls all come from pixels.
  Estimating the camera pose from images alone (visual odometry or
  landmarks) is a next step (section 25). Tests check the round trip and
  that the transform is not an identity.
* **Filter.** The pose is predicted from the agent's own velocity commands
  (dead reckoning) and corrected with each visual fix by exponential
  smoothing (`smoothing_alpha`). Confidence decays to 0 over
  `dead_reckoning_seconds` without a fix, and localisation is lost after
  `max_missing_observations` failed fixes.
* **Reset on disagreement.** When a fix and the prediction differ by more
  than 0.5 cells, the estimate jumps to the fix instead of moving 40% of the
  way. Marker fixes are accurate to about 0.01 cells, so such a gap means
  the prediction failed. Typically the body was pressed against a wall
  while velocity was still commanded. In 30 paired episodes without the
  reset:
  * 12 episodes had pose errors above 1 cell (worst 3.65 cells);
  * there were 4x as many recoveries (36 against 9);
  * the same 412 chests were opened.
* **Confidence gate.** Below `min_confidence` the local controller stops
  until confident fixes return, instead of driving blind.
* **Measured accuracy** (acceptance arena, seed 42, 640x360): a fix in 100%
  of analyses, mean error 0.033 cells, maximum 0.31 cells.

### 14.8 World model, exploration and spawn priors

* **Spawn-point status.** Each spawn point is `UNKNOWN`, `EMPTY`
  (seen twice with no chest), `CHEST_SEEN`, `OPENED`, `FAILED`,
  `UNREACHABLE` or `ABANDONED`. A chest track is linked to the nearest
  spawn point within 1 cell. If its track expires, the spot becomes
  `UNKNOWN` again rather than staying "seen" on stale evidence.
* **Planning grid.** A* runs on the estimated map: BLOCKED cells are walls,
  UNKNOWN cells are passable at `+unknown_cell_cost`, and plans are
  checked again as cells are revealed. If the estimated position falls
  inside a wall cell (it is off by a fraction of a cell), planning starts
  from the nearest walkable cell. Without this, a drifted estimate made
  every goal look unreachable. In 6 of 20 test episodes the agent then
  "completed" after about 40 s with most spawn points unchecked (found by
  the A/B runs above, and fixed).
* **Frontiers.** Frontier cells are FREE cells next to UNKNOWN ones,
  grouped into regions. Each region scores `S = G / (T + eps)`. `G` is the
  number of unknown cells around it plus `prior_weight` x the spawn
  probability of unresolved spawn points nearby. `T` is the A* travel time.
  Up to 6 candidates are checked with A*. Unreachable or unproductive goals
  are blacklisted for 30 simulated seconds.
* **Inspection goals.** A known but unresolved spawn point becomes an
  inspection goal: drive to within 2.5 cells and look, at most twice.
* **Chest goals** score `confidence / (T + eps)` and win over exploration.
  The episode ends when every spawn point is resolved
  (`stop_when_spawn_points_resolved`) or nothing reachable remains.

### 14.9 Closed-loop control and visual interaction

| State | What happens |
|---|---|
| `INITIALIZING` | Session, priors and pipeline set up |
| `OBSERVING` | Stand still and analyse a few frames (start, after inspections, after localisation trouble) |
| `LOCALIZING` | Wait for a confident marker fix |
| `EXPLORING` | Pick the best chest goal, inspection or frontier; complete when everything is resolved |
| `TARGET_DETECTED` | A new confirmed chest track interrupts exploration |
| `PLANNING_ROUTE` | A* to a standing point 1 cell east, west, south, south-east or south-west of the chest (never north, where the prompt is drawn) |
| `NAVIGATING` | Follow the path on the estimated pose. Replan when new walls block it; pause when pose confidence is low |
| `ALIGNING` | Stop and wait until the "E" prompt is visible next to the target chest in the frame |
| `INTERACTING` | Hold interact for `chest_interaction_seconds`. The progress bar must stay visible, or the attempt is treated as interrupted |
| `VERIFYING_OPEN` | Success only if the frame shows an **opened** chest at the target and no unopened one. If the agent's own sprite hides the chest, it backs off first |
| `RECOVERING` | Bounded retries per target (`maximum_recovery_attempts`), then FAILED or ABANDONED for that spawn point. Whole-episode budget |
| `COMPLETED`, `FAILED` | Flush observations, close the session and run row |

* **Late verification.** If a hold completed but verification was
  inconclusive (rare, mostly at 320x180), the spawn point is remembered.
  It is credited as opened only if a later frame clearly shows an opened
  chest there.
* **The recorded outcome is what the agent saw**, not what the simulator
  knows. The `simulate` report and the benchmark compare the two as an
  evaluation-only check: over-reported openings would be a VERIFY error,
  under-reported ones a missed confirmation (section 14.12).

### 14.10 Visual debugging GUI

`simulate --agent visual --render` adds a third column to the Phase 2 GUI:

* **Toggles** (initial state from `visual_debug.*`): enable, chest
  detections, object tracks, occupancy map, localisation estimate, camera
  view.
* **Camera view.** The agent's camera frame (thumbnail) with detection
  boxes, confidences and track IDs.
* **Occupancy map.** Grey = free, slate = blocked, black = unknown, plus
  tracks and the estimated pose.
* **Main canvas.** A ring at the estimated pose, `T<n>` track labels, and
  spawn-point colours from the agent's own status, not from truth.
* **Live statistics.** Analyses per simulated second, perception ms per
  frame, pose confidence, explored percentage, track count and current
  objective.

Overlays are drawn on copies, never on the frames perception uses.

### 14.11 The acceptance arena (`visual_arena_01`)

`seed-demo --scenario visual_arena` inserts 20 fictional spawn points.
Spawn probabilities range from 0.50 to 1.00, so presence is random per
episode. One location sits inside a sealed vault (a closed ring of walls),
so it is visible from above but unreachable. The map adds procedural walls,
rocks and decoys. The agent starts knowing none of it. A full seed-42 run:

```text
$ python -m chest_hunter simulate --map visual_arena_01 --agent visual --seed 42 --headless
  Outcome:               completed (all spawn points resolved)
  Simulated time:        160.133 s (4804 ticks at 30 Hz)
  Chest locations:       20 candidates, 19 visited, 1 unreachable, 0 abandoned
  Chests:                16 spawned, 16 found, 15 opened, 0 failed interaction(s)
  Chests opened/minute:  5.62 (simulated time)
  ...
Visual perception (evaluated against ground truth; the agent never saw it)
  Frames analysed:             1210 (0 failed captures)
  Detection precision:         1.000
  Detection recall:            0.957
  False positives:             0
  Open/closed confusions:      0
  Localisation success:        1.000
  Pose error mean / max:       0.0325 / 0.307 cells
  Perception time:             3.49 ms mean, 8.97 ms peak per frame
  Opened per simulator:        15 (agent recorded 15)
  Targets selected / reached:  16 / 15
  Replans after new walls:     22
```

The 16th chest is the one in the sealed vault: seen, correctly marked
unreachable, and not counted as a failure.

### 14.12 Oracle vs visual benchmark

```bash
python -m chest_hunter benchmark-vision --map visual_arena_01 --episodes 100 --seed 42
python -m chest_hunter benchmark-vision --map research_arena_01 --episodes 100 --seed 42 \
    --output-dir benchmark_results/research_arena_01
python scripts/compare_detectors.py          # detectors A/B/C at every resolution
```

* **Modes.**
  * A `oracle`: the Phase 2 agent, with the true map and positions.
  * B `chest-vision`: chests from the camera, true obstacle grid.
  * C `visual`: everything from the camera.
* **Paired design.** For each episode index, every mode runs with the same:
  * map and obstacle layout (`--seed`);
  * chest spawns (seeded by seed, episode and chest);
  * start position and tick budget.

  Spawns differ between episodes. The benchmark runs on an in-memory copy of
  the map's chest locations, so the research database is never touched.
* **Outputs** (in `--output-dir`, default `benchmark_results/`):
  * `navigation_benchmark.csv`, one row per episode and mode:
    * outcome and simulated time;
    * chests spawned, available and opened, both as recorded by the agent
      and as registered by the simulator;
    * chests per minute and seconds per chest;
    * recoveries, failed interactions, and unreachable or abandoned targets;
    * navigation success, travel, A* searches, and CPU and wall seconds.
  * `vision_benchmark.csv`, one row per episode for modes B and C:
    * frames analysed and failed, and analyses per simulated second;
    * detections, true and false positives and negatives, and open/closed
      confusions;
    * precision and recall;
    * localisation success, and marker and pose error in cells (mean,
      median, max);
    * perception milliseconds (mean, median, peak).
  * `summary.json`:
    * per-mode aggregates (mean, median, min, max);
    * paired differences against the oracle;
    * the VERIFY cross-check (over- and under-reported openings);
    * the detector comparison at the configured resolution;
    * run cost (wall, CPU, peak RSS) and the full configuration.
* **Time basis.** Chests per minute, seconds per chest and simulated seconds
  use simulated time (fixed 1/30 s step), independent of the host. CPU and
  wall seconds are real time on the benchmark machine.
* **"Available" chests** are spawned chests reachable from the start. They
  are computed from ground truth, for scoring only. The sealed-vault chest
  is never available.
* **Detection scoring.**
  * A detection matches a visible ground-truth chest at IoU >= 0.4 with the
    right category.
  * Chests less than half inside the frame are optional: matching one counts
    as a hit, missing one is not a miss.
  * Stacked chests are merged into one ground-truth object.

**Results (measured; 100 of 100 paired episodes per mode on each map, seed 42).**

**Map `visual_arena_01`, seed 42: 100 of 100 paired episodes completed per mode.**

| Mode | Chests/min mean | median | min-max | % of available opened | Sim s/chest | Recoveries/ep | Nav success | Outcomes | Recorded / simulator openings | CPU s/ep |
|---|---|---|---|---|---|---|---|---|---|---|
| A oracle | 7.141 | 7.186 | 4.58-8.65 | 100.0 (1394/1394) | 8.49 | 0.00 | 1.000 | completed 100 | 1394 / 1394 | 0.03 |
| B chest-vision | 7.256 | 7.229 | 4.72-8.37 | 100.0 (1394/1394) | 8.32 | 0.00 | 1.000 | completed 100 | 1394 / 1394 | 3.63 |
| C visual | 5.460 | 5.415 | 3.83-7.00 | 100.0 (1394/1394) | 11.23 | 0.28 | 0.936 | completed 100 | 1394 / 1394 | 6.20 |

| Mode | Precision (pooled) | Recall (pooled) | False positives/ep | Open/closed confusions | Localisation success | Pose error mean (cells) | worst-case max (cells) | Perception ms mean | peak | Analyses / sim s |
|---|---|---|---|---|---|---|---|---|---|---|
| B chest-vision | 1.000 | 0.986 | 0.00 | 0 | 1.000 | 0.0117 | 0.30 | 3.18 | 21.1 | 6.20 |
| C visual | 1.000 | 0.972 | 0.00 | 0 | 1.000 | 0.0338 | 0.31 | 3.47 | 28.3 | 7.45 |

* Paired against oracle, B chest-vision: mean +0.115 chests/min (median +0.108); better in 60, worse in 40, tied in 0 episodes.
* Paired against oracle, C visual: mean -1.681 chests/min (median -1.725); better in 0, worse in 100, tied in 0 episodes.
* Run cost: 16.7 min wall, 16.6 min CPU, peak RSS 463 MiB.
* Detector comparison on 60 labelled frames at 640x360: color recall 0.986 at 1.06 ms, template recall 0.959 at 38.81 ms, hybrid recall 0.986 at 1.81 ms; precision 1.000 for all.

**Map `research_arena_01`, seed 42: 100 of 100 paired episodes completed per mode.**

| Mode | Chests/min mean | median | min-max | % of available opened | Sim s/chest | Recoveries/ep | Nav success | Outcomes | Recorded / simulator openings | CPU s/ep |
|---|---|---|---|---|---|---|---|---|---|---|
| A oracle | 11.184 | 11.191 | 9.09-12.45 | 100.0 (2607/2607) | 5.39 | 0.00 | 1.000 | completed 100 | 2607 / 2607 | 0.03 |
| B chest-vision | 11.400 | 11.502 | 10.04-12.37 | 98.4 (2564/2607) | 5.27 | 0.43 | 0.997 | completed 100 | 2564 / 2607 | 6.15 |
| C visual | 7.685 | 7.556 | 5.12-10.01 | 98.5 (2566/2607) | 8.02 | 0.48 | 0.999 | completed 100 | 2566 / 2607 | 9.20 |

| Mode | Precision (pooled) | Recall (pooled) | False positives/ep | Open/closed confusions | Localisation success | Pose error mean (cells) | worst-case max (cells) | Perception ms mean | peak | Analyses / sim s |
|---|---|---|---|---|---|---|---|---|---|---|
| B chest-vision | 1.000 | 0.984 | 0.00 | 0 | 1.000 | 0.0072 | 0.31 | 3.80 | 33.8 | 8.07 |
| C visual | 1.000 | 0.993 | 0.33 | 0 | 1.000 | 0.0247 | 0.31 | 4.02 | 34.8 | 7.69 |

* Paired against oracle, B chest-vision: mean +0.217 chests/min (median +0.119); better in 59, worse in 41, tied in 0 episodes.
* Paired against oracle, C visual: mean -3.499 chests/min (median -3.559); better in 0, worse in 100, tied in 0 episodes.
* Run cost: 26.0 min wall, 25.8 min CPU, peak RSS 404 MiB.
* Detector comparison on 60 labelled frames at 640x360: color recall 0.941 at 1.17 ms, template recall 0.916 at 38.68 ms, hybrid recall 0.941 at 2.72 ms; precision 1.000 for all.

What the numbers say:

* **Every mode opened every available chest on the acceptance arena** (1,394 of
  1,394), and the agent's own records matched the simulator exactly.
* **Seeing chests is not the bottleneck.**
  * Chest-vision (B) was slightly *faster* than the Phase 2 oracle (A) on both
    maps: better in 60 and 59 of 100 paired episodes.
  * The oracle only learns that a spawn point is empty by walking to it. The
    camera rules out empty spots from up to 20 cells away.
* **Mapping the walls is the cost.**
  * The full visual agent (C) is 24% (acceptance arena) and 31% (demo arena)
    slower than the oracle, and worse in all 100 paired episodes on each map.
  * The time goes into exploring and replanning around walls it has not seen
    yet. Open/closed confusions were zero on both maps.
  * On the acceptance arena, C made no false positives at all. On the demo
    arena it made 0.33 per episode, pooled precision still rounding to
    1.000. None led to a wrong opening (zero over-reports).
* **The only recording errors are the stacked pair.**
  * On the demo arena, B and C missed one opening record in 43 and 41
    episodes respectively.
  * Each one was the Blue House attic/basement pair, 0.28 cells apart: the
    chest is opened, but the spot it was credited to had already been judged
    empty, so the opening is not recorded (reproduced and traced for episode
    0, and consistent with the limitation in section 24).
  * Openings were **never over-reported** in any mode or map.
* **Cost.**
  * Perception averaged 3.2-4.0 ms per analysed frame, at 6-8 analyses per
    simulated second.
  * A full visual episode cost 6-9 CPU seconds for 2-4 simulated minutes.
  * The benchmark processes peaked at 404-463 MiB RSS (three modes, 100
    episodes and all result rows kept in memory).


## 15. Phase 4: 3D third-person agent and Unreal Engine 5 integration

Phase 4 moves the agent into a **3D, third-person** research arena: a character
with a capsule body, an over-the-shoulder spring-arm camera, stairs, upper
floors, doorways and occluders. The agent finds chests in **perspective camera
images**, estimates how far away they are, walks there through a navigation
mesh (or without one), aligns with the chest, holds *interact* while watching
the on-screen progress bar and confirms the opening visually.

The target environment is a small **Unreal Engine 5** project
(`unreal/ChestHunterResearch`). The Python agent talks to it over a
**localhost-only TCP bridge** with a versioned binary protocol. Unreal Engine is
not installed on the development machine, so the project also ships a
**Python reference arena** that serves exactly the same protocol. All Phase 4
results below come from that reference arena.

> **What is verified and what is not.**
> * **Verified (automated tests, benchmark):** the protocol, the bridge client
>   and mock server over real loopback sockets, the reference arena (physics,
>   camera, renderer, navigation graph, interaction rules), 3D perception,
>   navigation, the controller, database records, CLI, dashboard, and the
>   agent end to end on the reference arena.
> * **Prepared, not verified:** the Unreal C++ module, its automation tests
>   and the editor map script. They have **not been compiled or run**, and the
>   Python agent has **never been connected to Unreal**. Static checks compare
>   the C++ sources with the Python protocol, and the shared deterministic
>   random-number code is compiled with g++ and matches Python bit for bit.
>   That is all that could be checked without Unreal.
> * The reference arena is not Unreal Engine. Its numbers say nothing about
>   Unreal's rendering, lighting or physics. The colour detector will need
>   recalibration on Unreal frames.

Full design, protocol reference, build steps and troubleshooting:
[`docs/phase4_3d_architecture.md`](docs/phase4_3d_architecture.md).

### 15.1 Quick start

```bash
python -m chest_hunter init-db
python -m chest_hunter seed-demo --scenario research_arena_3d      # 35 spawn points

# Reference arena (no Unreal needed), standard visual agent, no window:
python -m chest_hunter simulate-3d --backend reference --agent visual --seed 42 --headless-controller
# ... with the debug dashboard (camera feed with detections, map, statistics; needs Tk):
python -m chest_hunter simulate-3d --backend reference --agent visual --seed 42 --render-debug
# Other agents: --agent visual-nav (no route service), --agent oracle (privileged baseline)
# Options: --observation-width/--observation-height (320x180 ... 800x450), --max-ticks,
#          --detector color|template|hybrid, --graphics-quality low|medium|high, --evaluate

# Against a running Unreal research server (see the architecture doc; untested):
python -m chest_hunter simulate-3d --backend unreal --port 47820

# Paired benchmark, three modes:
python -m chest_hunter benchmark-3d --episodes 50 --seed 42 --workers 3
```

### 15.2 Components

| Area | Module(s) | Notes |
|---|---|---|
| Bridge protocol | `unreal_bridge/protocol.py` (C++ mirror: `ResearchProtocol.h/.cpp`) | 16-byte prefix + JSON header + raw `bgr8` payload; size limits; strict validation of every command and action; version negotiation |
| Bridge client | `unreal_bridge/client.py` | Loopback only; timeouts; reconnect with back-off; STEP resent with the same id after a dropped connection (applied at most once); stale-frame detection; `AgentPort` is the only surface the visual agent gets |
| Mock / reference server | `unreal_bridge/mock_server.py`, `simulation3d/` | Same commands and semantics as the Unreal subsystem; privileged commands are off unless started in oracle or evaluation mode |
| Unreal project | `unreal/ChestHunterResearch/` | Chest actor, third-person character with `USceneCaptureComponent2D`, arena built from `Content/ResearchData/research_arena_3d.json`, Recast NavMesh, loopback server thread, game-thread command subsystem, synchronous fixed-step stepping, automation tests (**not compiled**) |
| Perception | `vision/perspective.py`, `chest_detector_3d.py`, `depth_estimation.py`, `tracker_3d.py`, `interaction_cues_3d.py` | Pinhole model; colour / template / hybrid detection with 72 multi-view templates; distance from apparent size (A), ground plane (B), triangulation (C); 3D tracks |
| World model | `localization/world_model_3d.py` | Spawn priors from SQLite, statuses, empty-spot votes with line-of-sight checks |
| Navigation | `navigation/navigation_3d.py` | Hybrid: NavMesh route service queried with *estimated* positions. Visual-nav: own occupancy map from raycast probes plus A*. Path following, stuck detection, scan patterns |
| Controller | `agent/controller_3d.py` | 14-state machine: OBSERVING, SCANNING, EXPLORING, TARGET_DETECTED, PLANNING_ROUTE, NAVIGATING, APPROACHING, ALIGNING, INTERACTING, VERIFYING_OPEN, RECOVERING, ... |
| Runner, benchmark, dashboard | `agent/runner_3d.py`, `analytics/benchmark_3d.py`, `analytics/vision_metrics_3d.py`, `agent/dashboard_3d.py` | Sessions, observations and runs in SQLite; paired benchmark; vision metrics against labels; Tk dashboard |

### 15.3 The 3D arena

`research_arena_3d` (100 m × 100 m) has 35 spawn points in a central warehouse
with a mezzanine, an eastern settlement (houses with doorways, a 1.3 m narrow
door, a two-storey house), a northern forest, a western outpost with a
watchtower, southern docks, a central plaza and a south field. It includes gold
decoys (the same colour as the chests) and two spawns that are **unreachable by
design**: a water-tower platform with no ladder and a sealed container. Chest
presence is a deterministic function of (seed, episode, spawn index), identical
in Python and C++. Opening a chest needs `interact` held for 2 s within 1.8 m,
facing it (60° cone), with line of sight. A notification then reports the
player's own opening and whether it gave a reward (30 %).

### 15.4 What the agent may and may not know

| The standard visual agent receives | It never receives |
|---|---|
| Camera images rendered by the environment's own scene capture of the third-person view, with the camera intrinsics | Chest presence, chest positions or opened state |
| Its own camera rig (yaw, pitch, arm length, camera position relative to its feet) | Its true position (only odometry with random-walk drift) |
| Odometry with seeded drift (0.5 % of distance travelled, per axis) | Evaluation labels or the oracle state (the server refuses them unless started for evaluation, and `AgentPort` cannot send them) |
| Its own notifications (opened / reward / failed) and the game rules | Desktop screenshots or any other application's window |
| Spawn priors from SQLite (location + probability, as in Phases 1 to 3) | Routes to hidden chests: route queries take *estimated* destinations only |

`tests/test_agent_3d.py` checks this statically (the agent's modules import
neither the simulator nor the mock server, and contain no privileged names). It
also checks it at run time, with a full episode against a server that refuses
every privileged command.

### 15.5 Benchmark: oracle vs hybrid vs visual navigation (50 paired episodes)

`benchmark-3d --episodes 50 --seed 42 --workers 3`, reference arena, 640×360,
90° horizontal FOV, 10 simulated minutes maximum per episode. All three modes
play the same seeds. Episodes end early when the agent has resolved every
spawn point (all three modes usually do). Chests per minute is therefore
`opened / episode duration`, in simulated time.

* **A, oracle:** true chest positions over the privileged channel, NavMesh routes. An upper bound, not a vision result.
* **B, hybrid:** chests from the camera only, NavMesh route service with estimated positions.
* **C, visual navigation:** chests from the camera only, local raycast probes and its own A*. No route service and no knowledge of stairs.

| Mode | Chests / min, mean (median, min-max) | % of available chests opened | % of seen chests opened | Episode length (sim.) | Travel | Recoveries | Navigation failures |
|---|---|---|---|---|---|---|---|
| A oracle | **7.81** (7.73, 7.01-8.58) | 100.0 | - | 180 s | 585 m | 0.0 | 0.0 |
| B hybrid | **4.06** (4.12, 2.95-5.45) | 84.9 | 90.4 | 276 s | 816 m | 11.2 | 0.26 |
| C visual-nav | **1.98** (1.98, 0.84-3.15) | 45.7 | 72.0 | 302 s | 668 m | 34.9 | 28.4 |

Means over 50 episodes. "Available" means present and reachable from the
start (1,175 chests in 50 episodes). Paired against the oracle, the hybrid
agent was slower in all 50 episodes (mean −3.75 chests/min) and visual-nav in
all 50 (mean −5.84).

**Openings checked against ground truth.** No episode of any mode counted more
openings than it really made (0 over-reported; the check compares counts per episode). Oracle recorded 1,175 of 1,175 real
openings. Hybrid opened 1,000 chests and recorded 934. Visual-nav opened 537
and recorded 492. The gap is chests the agent opened but could not confirm
visually (for example, seeing the opened chest only from behind), so it did
not write them. Outcomes: oracle and hybrid 50/50 completed. Visual-nav 49
completed and 1 hit the tick limit. One visual-nav episode (episode 19) ended
after 85 s with 2 of 23 available chests opened, because the agent considered
every spawn point resolved.

**Perception against the reference renderer's labels** (every analysed frame;
hybrid 78,057 frames, visual-nav 76,890):

| | Hybrid | Visual-nav |
|---|---|---|
| Detection precision / recall (pooled) | 0.983 / 0.925 | 0.979 / 0.932 |
| Distance error, median of per-episode median absolute error | 0.35 m | 0.38 m |
| Distance error, mean absolute (episode mean) | 0.83 m | 0.73 m |
| ... by true distance 0-5 / 5-10 / 10-20 / 20-40 m | 0.23 / 0.35 / 0.72 / 2.01 m | 0.30 / 0.33 / 0.84 / 1.51 m |
| Matched detections without a usable distance | 3.6 % | 3.1 % |
| Odometry error (mean over the episode, mean of episodes) | 0.042 m | 0.039 m |

The exported counts include 12,577 (hybrid) and 9,024 (visual-nav) matched
detections whose open/closed class disagreed with the label. This run did not
export the matched-detection totals, so a per-detection rate cannot be given,
and the causes were not broken down.

Raw data: `benchmark_results/phase4/navigation_3d.csv` (one row per episode and
mode), `vision_3d.csv` and `summary_3d.json`. Before committing, the first 3
paired episodes were re-run on the final code. They reproduced every
non-timing field exactly.

**Against the proposed acceptance targets** (reference arena only; Unreal
not tested):

| Target | Result |
|---|---|
| ≥ 90 % of visually detected, reachable chests opened | Hybrid: **90.6 %** pooled (1,000 of 1,104; per-episode mean 90.4 %, 26 of 50 episodes ≥ 90 %, minimum 77.8 %). Counting only openings the agent confirmed and recorded: 84.6 % (934). Visual-nav: 72.6 %, **not met** |
| No infinite navigation loops | Met: all 150 episodes ended on their own (149 completed: the oracle after opening every present chest, the visual agents after resolving every spawn point; 1 at the tick limit); a recovery budget bounds every episode |
| No duplicate successful openings | Met: database constraint plus tests; 0 duplicates and 0 over-reported openings in the benchmark |
| No access to hidden chest state during visual runs | Met: static audit of the agent's modules plus a full episode against a server with every privileged command disabled (`tests/test_agent_3d.py`) |
| Stable across repeated seeded episodes | Met: identical results for repeated seeds (test), and a re-run of 3 benchmark episodes on the final code matched every non-timing field |

### 15.6 Measured performance (reference arena)

Same 4-vCPU Intel Xeon 2.1 GHz machine, no GPU, Python 3.13, three benchmark
workers running in parallel:

| Measurement | Hybrid | Visual-nav | Oracle |
|---|---|---|---|
| CPU per episode | 52.1 s for 276 simulated s (**~5.3× faster than real time** on one core) | 47.3 s for 302 s | 2.5 s for 180 s (no images) |
| Perception per analysed frame (detect, estimate, track, world model) | 6.5 ms | 6.0 ms | - |
| Bridge round trip per request (incl. rendering when a frame is requested) | 7.2 ms | 5.8 ms | 0.41 ms |
| Peak memory per worker process (RSS), max over the run | 116 MiB | | |
| Whole benchmark (150 episodes) | 1,802 s wall, 5,098 CPU-s | | |

Single-process micro-benchmark on the same machine (medians of 60 frames,
moving camera):

| Resolution | Render one frame (reference software renderer) | Hybrid chest detection |
|---|---|---|
| 320×180 | 11.4 ms | 0.9 ms |
| 480×270 | 14.3 ms | 2.2 ms |
| 640×360 | 16.5 ms | 3.1 ms |
| 800×450 | 21.2 ms | 5.0 ms |

Bridge over loopback TCP, 640×360: a `STEP` (2 ticks) without a frame takes
0.9 ms, a `STEP` returning a frame (691 KB raw `bgr8` payload, rendering
included) 20.2 ms, and `PING` 0.13 ms. Physics alone is 0.13 ms per 2-tick
step. Rendering, not the bridge or perception, dominates the reference
arena's cost.

**Unreal Engine was not measured** (not installed). The project is configured
for a low rendering cost: no Lumen, Nanite or virtual shadow maps, low
scalability and a 640×360 capture. Even so, a UE5 editor is **not expected to
run reliably on 8 GB of RAM or integrated graphics**. A packaged development
build in a small window is the realistic option on modest hardware, and it has
not been tried. Camera images need rendering: `-nullrhi` runs (fine for the
protocol automation tests) cannot produce observations.

### 15.7 Database (schema version 3)

Migration 3 adds `chest_observations.reward_received` (NULL when unknown) with a
trigger forbidding a reward without an opening, and
`simulation_runs.navigation_failures`. Existing data is kept (tested). 3D runs
use strategies `3d-oracle`, `3d-hybrid-visual` and `3d-visual-nav`, and store
their backend, camera and detector settings in `parameters_json`.

### 15.8 Phase 4 limitations

* **No Unreal verification.** The C++ module has not been compiled. The Unreal
  automation tests and map script have not been run, and the agent has not been
  connected to Unreal.
* **Detector tuned to the reference renderer.** HSV classes and templates
  match its flat shading. Unreal lighting, shadows, tone mapping and
  anti-aliasing will need recalibration and probably new templates.
* **Unconfirmed openings.** About 7 % (hybrid) of real openings were not
  recorded, because the opened chest was never seen clearly afterwards.
* **Visual-nav cannot use stairs**, so upper-floor chests are out of its
  reach (45.7 % of the available chests opened).
* **Remaining hybrid misses** include chests behind occluders where alignment
  keeps failing (for example, the South Field haystack, seen during
  development), and chests whose opened state is misread.
* Unreal evaluation labels are approximate (projected boxes plus visibility
  traces). Unreal odometry drift uses a different seeded random stream, so
  Python and Unreal trajectories are not bit-identical.

## 16. Phase 5: imitation learning, adaptive exploration and generalisation

Phase 5 adds a learning layer to the Phase 4 visual agent:

* a demonstration recorder (human teleoperation, or the scripted controller);
* a dataset pipeline;
* a compact imitation policy, trained offline in NumPy and exported to ONNX for CPU inference;
* a policy selector with a safety layer and a deterministic fallback;
* Bayesian spawn estimates;
* route memory;
* an adaptive target planner;
* classified, bounded recovery;
* procedural maps for testing on layouts the agent has never seen.

The deterministic Phase 4 controller remains the default.

> **What is verified and what is not**
> * **No human demonstrations exist.** The teleoperation recorder works (tested with
>   simulated key presses), but no person recorded with it. The training data is
>   **synthetic**: 30 episodes of the deterministic controller, labelled
>   `synthetic-scripted-expert` everywhere. The learned policy imitates that controller.
>   Nothing here claims human-like behaviour.
> * All results come from the **Python reference arena**, not from Unreal Engine.
>   Unreal is not installed. The small C++ additions (procedural layouts, lighting and
>   chest appearance, the map list) are **not compiled**.
> * Throughput is chests per **simulated** minute. It is not a real-world speed claim.

Design details, data formats and every rule: [`docs/phase5_learning_architecture.md`](docs/phase5_learning_architecture.md).

### 16.1 Commands

```bash
# Demonstrations (human: Tk window, keys act on that window only; or scripted = synthetic)
python -m chest_hunter record-demo --controller human --map research_arena_3d --output training/datasets/demo_01
python -m chest_hunter record-demo --controller scripted --map proc_train_000 --episodes 2 --seed 1000 \
    --output training/datasets/demo_01

# Train (early stopping on the validation episodes) and export ONNX FP32 + INT8
python -m chest_hunter train-policy --dataset training/datasets/demo_01 --epochs 30 --batch-size 32
python -m chest_hunter export-policy --model training/checkpoints/bc_synthetic_v1

# Evaluate: held-out test metrics + paired episodes against the deterministic controller
python -m chest_hunter evaluate-policy --model training/checkpoints/bc_synthetic_v1 --episodes 50

# Run with a policy: deterministic | learned | hybrid, optionally adaptive planning
python -m chest_hunter simulate-3d --backend reference --agent hybrid --policy hybrid \
    --model training/checkpoints/bc_synthetic_v1 --adaptive-planning --smart-recovery --seed 42
python -m chest_hunter simulate-3d --backend reference --policy hybrid --render-debug   # switch policy live

# Compare strategies (paired seeds) and test generalisation on unseen maps
python -m chest_hunter benchmark-policies --episodes 100 --workers 4
python -m chest_hunter benchmark-generalization --familiar 4 --unseen 4 --episodes 6 --workers 4

# Procedural maps and Bayesian spawn estimates
python -m chest_hunter generate-maps --split test --count 4 --export-unreal
python -m chest_hunter estimate-spawns --map research_arena_3d --store
```

`simulate-3d --agent hybrid` keeps its Phase 4 meaning (camera plus NavMesh route
service). The control policy is chosen separately with `--policy`. The default policy comes
from the `learning` configuration section (`deterministic`).

### 16.2 The learned policy

* **Model:** a 28,493-parameter CNN.
  * Input: a 128×72 camera frame, plus camera pitch and speed.
  * Outputs: move, strafe, yaw and pitch rates, jump, interact, and an auxiliary
    controller-phase head.
  * Trained in NumPy with Huber, BCE and cross-entropy losses, Adam and early stopping. The
    backward pass is checked against numerical gradients.
  * PyTorch was not used: its CPU index is blocked here, and the default build pulls about
    3 GB of CUDA libraries.
* **Data:** 29,109 samples from 30 synthetic episodes on 13 maps, split **by episode**
  (22 / 4 / 4).
* **Held-out test episodes:** compared with predicting the mean action, the model's MAE is:
  * move_forward 0.098 (baseline 0.448);
  * move_right 0.091 (0.130);
  * look_pitch 0.053 (0.109);
  * **look_yaw 0.166, worse than the baseline's 0.156.** The expert's turning depends on
    route plans that one frame does not show.

  Interact accuracy is 1.00 and phase accuracy 0.94.
* **Export:**
  * FP32 ONNX (116 KB) matches the NumPy model within 1e-6.
  * INT8 (42 KB) keeps every interact decision but changes the continuous outputs by 0.037
    on average, and is slower than FP32 for this small model.
  * CPU latency per observation (one thread): **0.14 ms** with ONNX Runtime FP32, 0.21 ms
    INT8, 1.3 ms NumPy. The proposed target was 20 ms.
* **Policy selector:**
  * *learned*: the model drives every decision.
  * *hybrid*: the model replaces the controller's movement and camera commands only while
    approaching and aligning with a visible chest, and only when it is validated and
    confident.
  * Safety layer:
    * clamped ranges and speed, and camera pitch limits;
    * bounded output reuse;
    * interaction only with the prompt visible;
    * a 20 ms inference limit;
    * fallback to deterministic control on errors or invalid output, with the model
      disabled after 10 consecutive failures;
    * bounded handovers when a learned policy is stuck;
    * when the deterministic controller takes over again, an interaction or opening
      check it was paused in is dropped (stale timers could credit an unopened chest).
    * in learned mode, each opening notification is credited once, at the spawn point
      next to the agent. If it arrives during an interaction the deterministic controller
      started itself, that interaction ends without its own opening check, so one chest
      cannot be counted at two spawn points (or at none).

### 16.3 Adaptive exploration

* **Spawn estimates:** each location's chest probability is `Beta(4·p0, 4·(1−p0))` updated
  with presence and absence observations.
  * Only completed sessions on the same layout version count, one observation per session.
  * Unvisited spots keep their prior, and rewards are never used.
  * Estimates are stored in `spawn_estimates`; the configured value is never changed.
* **Route memory** (`route_memory`) learns travel-time factors and decayed reliabilities per
  region pair. It ignores other layout versions and can be invalidated.
* **Adaptive planner:** scores candidates with
  `(p̂·q + λ·information gain + μ·cluster value) / (expected time + ε)`, and prefers
  reliable chest clusters.
* **Classified recovery:** seven failure classes, each with a bounded strategy. One of
  these confirms an opening that the agent was notified about but did not see clearly.
  A chest lost during the approach gets at most one rescan per spawn point, keyed by spawn
  point rather than by track, because a re-detected chest gets a new track id.
* **Schema version 4** (migration 4) adds the two tables. Existing data is kept (tested).

### 16.4 Policy benchmark (research arena, 100 paired episodes)

**Setup:**

* `research_arena_3d`, 100 episodes per strategy from base seed 42. Every strategy plays
  the same 100 episodes (the same chest draws and start), so the comparisons are paired.
* Limit of 18,000 ticks (600 simulated seconds) per episode, 640×360 camera.
* Model `bc_synthetic_v1` (trained on synthetic demonstrations only). The adaptive
  strategies read the training-time history in `training/experience.db`.
* 4 worker processes on the 4-vCPU development machine. All 500 episodes completed with
  no errors, in 8,259 s wall time.
* **The two adaptive rows come from a re-run** after a recovery fix (commit `91a8297`;
  see 16.5). The re-run played deterministic, hybrid_adaptive and deterministic_adaptive
  on the same 100 episodes (300 episodes, no errors, 3,907 s wall time). Its deterministic
  episodes matched the first run exactly, in every non-timing field of all 100 rows, so
  the hybrid row from the first run remains paired with the same baseline.
* **The learned rows come from a second re-run** after an opening-bookkeeping fix
  (commit `7675196`; see 16.6). It played learned on the same 100 episodes (no errors,
  2,990 s wall time). The changed code runs in learned mode only, and the deterministic
  episodes in the matching generalisation re-run were identical to the first run.

**Strategies:**

* `deterministic`: the Phase 4 controller (baseline).
* `learned`: the model drives.
* `hybrid`: the model steers the approach to a chest when it is confident.
* `hybrid_adaptive`: hybrid plus adaptive planning and classified recovery.
* `deterministic_adaptive`: an ablation, the Phase 4 controller plus adaptive planning and
  classified recovery.

The **primary metric** is chests actually opened per simulated minute. Openings are
counted from the simulator's ground truth, which is used only for scoring and never shown
to the agent.

| Strategy | Mean /min | Median | SD | 95 % CI | Paired Δ vs deterministic [95 % CI] | % change | Better / worse / tied |
|---|---|---|---|---|---|---|---|
| deterministic | **4.403** | 4.385 | 0.559 | [4.292, 4.514] | baseline | - | - |
| learned | 0.826 | 0.800 | 0.148 | [0.797, 0.855] | −3.577 [−3.683, −3.471] | −81.2 % | 0 / 100 / 0 |
| hybrid | 3.350 | 3.359 | 0.532 | [3.245, 3.456] | −1.053 [−1.170, −0.935] | −23.9 % | 3 / 97 / 0 |
| hybrid_adaptive | 3.281 | 3.257 | 0.611 | [3.160, 3.403] | −1.122 [−1.259, −0.984] | −25.5 % | 3 / 97 / 0 |
| deterministic_adaptive | 4.388 | 4.347 | 0.485 | [4.292, 4.485] | −0.015 [−0.107, +0.078] (not significant) | −0.3 % | 53 / 47 / 0 |

| Strategy | Agent-counted /min | Under- / over-counted (total) | % of available chests opened | Chests found | Spawn points resolved | Recovery attempts / ep | Simulated s / ep | Model share of decisions |
|---|---|---|---|---|---|---|---|---|
| deterministic | 4.133 | 126 / 0 | 85.9 % | 94.3 % | 100 % | 11.7 | 279.0 | 0 |
| learned | 0.826 | 0 / 0 | 34.9 % | 56.2 % | 43.2 % | 10.8 | 600.0 (all at the limit) | 0.878 |
| hybrid | 3.183 | 97 / 0 | 79.2 % | 94.1 % | 99.4 % | 25.5 | 342.9 | 0.248 |
| hybrid_adaptive | 3.281 | 0 / 0 | 79.5 % | 93.3 % | 98.9 % | 25.5 | 353.1 | 0.216 |
| deterministic_adaptive | 4.388 | 0 / 0 | 86.7 % | 93.6 % | 100 % | 10.1 | 282.0 | 0 |

**What the results show:**

* **The deterministic controller remains the best strategy and stays the default.**
* **The learned policy opened 81 % fewer chests per minute.**
  * All 100 episodes ran to the 600 s limit.
  * It found 56 % of the available chests and resolved 43 % of the spawn points.
  * Every episode used all 25 stuck handovers to the deterministic controller.
  * Likely causes: the policy sees a single frame and has no memory or route plan, so it
    cannot head for unexplored spawn points. Its yaw prediction is worse than a constant.
    Small errors lead to states the expert never visited, where it has no training data.
  * The bookkeeping fix did not change this: against the first run the change was
    −0.018 /min [−0.052, +0.016] (−2.1 %, not significant).
* **Hybrid control cost 23.9 %** (worse in 97 of 100 episodes).
  * It opened 1.6 fewer chests per episode (−7.8 %) and took 64 s longer (+22.9 %).
  * Its recovery attempts doubled. Over the 100 episodes, camera-alignment recoveries rose
    from 802 to 1,556, and it had 405 path-blocked recoveries, against none for
    deterministic.
  * The learned approach steering is less precise than the analytic controller it copies.
* **Adaptive planning did not open more chests.**
  * Against deterministic, the change was −0.015 /min [95 % CI −0.107, +0.078], −0.3 %.
    Against hybrid, it was −0.069 /min [−0.214, +0.076], −2.1 %. Neither is significant.
  * deterministic_adaptive's agent-counted throughput is 6.2 % higher (significant), but
    that is bookkeeping only. Classified recovery confirmed 155 openings by notification,
    where the plain controller under-counted 126.
  * It travelled 10.9 % further (+89 m [+77, +101] per episode) in about the same time.
    It entered the Phase 4 RECOVERING state 74 % less often (2.9 against 11.4 times per
    episode), because the classified strategies (look-backs, rescans) handle those cases
    first.
  * The recovery fix barely changed the arena results: −0.009 /min (deterministic_adaptive)
    and −0.015 /min (hybrid_adaptive) against the first run, neither significant. The
    loop it fixes was found on a procedural map (16.5).
* **No over-counting.** In all 900 arena episodes of the three runs, no episode counted
  more openings than it really made. The check compares counts per episode, not which
  spawn point was credited (16.6).
* **Learned inference under benchmark load:** 0.89 ms mean and 1.16 ms p95 per
  observation (averages of the per-episode values). In the 100 learned episodes, one
  inference exceeded the 20 ms limit and was discarded (two in the first run).
* **Speed:** one worker simulated 5–6× faster than real time on one core (45 CPU-s per
  279 simulated s for deterministic, 117 CPU-s per 600 s for learned). Perception
  took about 6.2 ms per analysed frame. Peak memory was at most 143 MiB per worker.

Per-episode rows, the full summaries and the reports are in
[`benchmark_results/phase5/policies`](benchmark_results/phase5/policies) (first run, all five
strategies) and
[`benchmark_results/phase5/policies_recovery_fix`](benchmark_results/phase5/policies_recovery_fix)
(re-run after the recovery fix) and
[`benchmark_results/phase5/policies_learned_fix`](benchmark_results/phase5/policies_learned_fix)
(learned, re-run after the bookkeeping fix).

### 16.5 Generalisation: familiar vs unseen maps

**Setup:**

* **Familiar maps:** `proc_train_000`–`003`. They are in the demonstration data (2
  episodes each, seed 1000), and the experience database holds 2 completed sessions per map.
* **Unseen maps:** `proc_test_000`–`003`, generated from a disjoint seed range. Their
  layouts, obstacles, chest locations, lighting, chest appearance and start poses were
  never used by any trained component, and there is no history for them.
* **Episodes:** 6 per map from seed 5000, played in sequence for each (strategy, map) with
  online adaptation. The adaptive strategy updates its spawn estimates and route memory
  after every episode.
* **Outcome:** 24 episodes per strategy and group, 192 in total. All completed with no
  errors.
* **The hybrid_adaptive row comes from a re-run** after the recovery fix described under
  "Ablation" below. The re-run also repeated deterministic and hybrid. All 96 of their
  episodes matched the first run exactly, so the pairing holds.
* **The learned row comes from a re-run** after the opening-bookkeeping fix (commit
  `7675196`), which also repeated deterministic. All 48 deterministic episodes matched the
  first run exactly. 96 episodes, no errors.

Chests actually opened per simulated minute (ground truth):

| Strategy | Familiar maps, mean [95 % CI] | Unseen maps, mean [95 % CI] | Unseen − familiar [95 % CI] | Paired change vs deterministic, familiar | … unseen |
|---|---|---|---|---|---|
| deterministic | **3.673** [3.292, 4.054] | **3.422** [2.964, 3.880] | −0.251 [−0.832, +0.329] (−6.8 %) | baseline | baseline |
| learned | 0.683 [0.602, 0.765] | 0.650 [0.563, 0.737] | −0.033 [−0.149, +0.083] (−4.9 %) | −81.4 % | −81.0 % |
| hybrid | 3.063 [2.781, 3.345] | 2.686 [2.409, 2.963] | −0.377 [−0.762, +0.008] (−12.3 %) | −16.6 % | −21.5 % |
| hybrid_adaptive | 3.011 [2.733, 3.289] | 2.710 [2.427, 2.993] | −0.301 [−0.687, +0.085] (−10.0 %) | −18.0 % | −20.8 % |

The "Unseen − familiar" column is unpaired (Welch), because the maps differ. Every paired
change against deterministic in the last two columns is significant at 95 %.

**What the results show:**

* **The deterministic controller is the best strategy on both groups of maps.**
* **The unseen maps are harder for every strategy.** They hold fewer chests (9.9 against
  11.1 per episode), and even the deterministic controller, which learned nothing, drops
  6.8 % (not significant). The fair test is therefore each strategy's deficit against
  deterministic on the same maps.
* **That deficit grows on unseen maps, but not significantly.**
  * hybrid: from −16.6 % to −21.5 %. Difference-in-differences −0.126 /min
    [−0.514, +0.262].
  * hybrid_adaptive: from −18.0 % to −20.8 %. Difference-in-differences −0.050 /min
    [−0.434, +0.334].
  * learned: unchanged (−81.4 % and −81.0 %).
  * With 24 episodes per group, only large generalisation gaps would show up.
* **No measurable online improvement over six episodes.**
  * Throughput rises over the sequence for every strategy, including deterministic,
    which does not adapt: the later episodes simply have more chests (8.5 rising to 14 on
    the familiar maps).
  * hybrid_adaptive against hybrid is −1.7 % on familiar and +0.9 % on unseen maps,
    neither significant.
  * The paired difference shows no trend over the episode sequence: −0.027 /min per
    episode [−0.157, +0.103] on familiar maps, −0.012 [−0.146, +0.123] on unseen maps.
* **The learned policy reached the 600 s limit in all 48 episodes.** The procedural maps
  are smaller, so it opened a larger share of their chests (62 % and 67 %) than in the
  arena, but still only 0.65–0.68 per minute.
* **No over-counting.** In none of the 624 procedural-map episodes over all runs did the
  agent count more openings than it really made.

**Ablation: adaptive planning without the learned policy.** To separate the planner from
the model, deterministic and deterministic_adaptive played 12 episodes per map in
sequence (seed 5000, online updates after every episode), 48 per group:

| Group | deterministic, mean [95 % CI] | deterministic_adaptive, mean [95 % CI] | Paired Δ [95 % CI] | % change | Better / worse |
|---|---|---|---|---|---|
| familiar | 3.680 [3.407, 3.954] | 3.500 [3.251, 3.748] | −0.181 [−0.315, −0.047] (significant) | −4.9 % | 16 / 32 |
| unseen | 3.457 [3.163, 3.752] | 3.383 [3.077, 3.689] | −0.074 [−0.244, +0.096] (not significant) | −2.1 % | 22 / 26 |

* **Adaptive planning is slower on the familiar maps.** It opened about as many chests
  (+0.10 per episode [−0.25, +0.46]) but travelled 12.6 % further and took 10.4 s
  [+5.5, +15.4] longer per episode (+6.9 %). The prior history for these maps comes from
  only 2 sessions each.
* **On unseen maps it is not significantly different** (−2.1 %). It still travelled 11.6 %
  further.
* **No learning trend over 12 episodes.** The paired difference changes by −0.008 /min per
  episode [−0.047, +0.031] on familiar and +0.022 [−0.028, +0.071] on unseen maps.
  Episodes 7–12 on unseen maps average +1.4 % against −5.7 % for episodes 1–6, but both
  intervals include zero.
* **The ablation found a bug.** The first ablation run (before commit `91a8297`) failed 3
  of 48 unseen-map episodes. A chest that kept dropping out of tracking during the
  approach came back under a new track id each time. The rescan limit was per track, so
  it never triggered, and the agent retried until the episode's recovery budget was spent
  (100 times for one chest). The limit is now per spawn point and respects the
  `target_lost` class budget, and a regression test covers it. Before the fix the unseen
  result was −0.303 /min [−0.611, +0.005] (−8.8 %) with the 3 failures; after it, −2.1 %
  with none. The familiar result barely moved (−5.5 % before, −4.9 % after).
* The deterministic episodes were identical in both ablation runs and match the first
  6 episodes per map of the generalisation run.

Results: [`benchmark_results/phase5/generalization`](benchmark_results/phase5/generalization)
(first run, four strategies),
[`generalization_recovery_fix`](benchmark_results/phase5/generalization_recovery_fix)
(re-run after the recovery fix),
[`generalization_learned_fix`](benchmark_results/phase5/generalization_learned_fix)
(deterministic and learned, re-run after the bookkeeping fix) and
[`generalization_adaptive`](benchmark_results/phase5/generalization_adaptive) (ablation,
after the fix).

### 16.6 Phase 5 limitations

* **No human demonstrations.** The teleoperation recorder is tested with simulated key
  presses only. The policy was trained on synthetic demonstrations from the deterministic
  controller. Behaviour cloning of that expert is bounded by the expert, so the learned
  policy is not expected to beat it. Its results say nothing about learning from people.
* **Single-frame behaviour cloning.** The policy has no memory and no route plan. It
  cannot know where unexplored spawn points are, and it inherits the usual
  behaviour-cloning problem: small errors lead to states the expert never visited, where it
  has no training data. The yaw rate is predicted worse than a constant.
* **Hybrid control is narrow by design.** The model only steers the local approach and
  alignment. Everything else stays deterministic.
* **INT8 quantisation** changes the continuous outputs (mean 0.037) and is slower than FP32
  for this 28k-parameter model. It is optional, not the default.
* **Spawn estimation helps only when the priors are wrong.** On the hand-made arena the
  configured probabilities are the truth, and six training sessions move the estimates
  *away* from it (MAE 0.097). On procedural maps with an uninformative prior, two sessions
  per map reduced the error from 0.202 to 0.168.
* **History in the arena benchmark is static.** The adaptive strategies read the
  training-time history (6 episodes, seed 1000). They do not update it between the 100
  benchmark episodes, because those run in parallel. Online adaptation is measured in the
  generalisation benchmark.
* **Credited by notification.** With smart recovery, an opening the agent was notified
  about but could not see is credited after two look-backs.
  * This corrects the agent's own count. It does not open more chests.
  * In the arena benchmark it accounts for all of deterministic_adaptive's +6.2 % gain in
    agent-counted throughput. Against ground truth the change is −0.3 % (not
    significant).
  * Such openings are counted separately (`credited_by_notification`). The benchmark
    compares each episode's count with ground truth. In none of the 900 arena episodes
    or 624 procedural-map episodes did the agent count more openings than it made.
* **Procedural maps** have single-storey buildings, with elevation coming from plateaus. True
  spawn probabilities are independent uniform draws, so there is no spatial correlation
  for the planner to exploit.
* **Unreal** was not available. The C++ additions are not compiled, teleoperation against
  Unreal is untested, and exact replay there would need deterministic engine stepping.
* **Training in NumPy** is practical only for small models like this one. A larger model
  would need a framework (PyTorch or similar) and probably a GPU.
* **The learned and hybrid policies are worse than the controller they imitate.**
  * Learned: −81 % in the arena, and −81 % on both familiar and unseen procedural maps.
  * Hybrid: −24 % in the arena, −17 % and −22 % on the procedural maps.
  * The deterministic controller therefore stays the default (`learning.policy:
    deterministic`).
* **Adaptive planning is not shown to help.**
  * Arena: −0.3 % (not significant).
  * Familiar procedural maps: −4.9 % (significant).
  * Unseen procedural maps: −2.1 % (not significant).
  * No improvement over 6 or 12 online episodes.
  * Its only consistent effect is bookkeeping: the agent no longer under-counts its own
    openings.
* **Small samples for generalisation.** There are 4 maps per group and 24 episodes per
  strategy and group (48 in the ablation). Unpaired familiar-vs-unseen gaps below roughly
  0.4 /min cannot be resolved, and the maps differ in how many chests they hold.
* **Five problems were found by the benchmarks and tests, and fixed** before the numbers
  above (details in `docs/phase5_learning_architecture.md`, section 13.2):
  * the discovery metric was read from the wrong object;
  * a stale phase after a handover over-counted openings in learned mode;
  * the agent's own count was used as the primary metric;
  * the approach-time rescan loop;
  * learned-mode opening bookkeeping. Under heavy load a test episode counted one
    opening at two spawn points (3 counted, 2 opened). The first fix (`052d79f`) left
    openings during handovers to the deterministic agent, which then under-counted 103
    real openings in a 148-episode re-run; those results were discarded. The final fix
    (`7675196`) credits every learned-mode opening once, at the spawn point nearest the
    agent, and ends any interaction the deterministic agent had started for it. Its
    re-run counted exactly in all 148 learned episodes.

  The baseline reproduced exactly in every re-run.
* **Learned results depend on machine load.** An inference over 20 ms is discarded, and
  after 10 consecutive failures the model is disabled for the episode, so on a busy
  machine the deterministic agent drives more of the learned episodes. The arena
  benchmark had one timeout in 100 learned episodes.
* **Spawn-point attribution is not checked against ground truth.** The benchmark compares
  how many openings were counted, not where. A chest opened next to the agent's goal can
  be credited at the wrong spawn point with a correct count. A check of 36
  deterministic episodes (300 credited openings) found none; it was seen only in
  learned-mode stress tests with injected inference failures.
* **Simulated time only.** All throughputs are per simulated minute in the Python
  reference arena, on one CPU core per worker. They say nothing about speed in Unreal, in a
  real game, or with GPU rendering.

## 17. Phase 6: reliability, planning and computational optimisation

Phase 6 makes the agent supervise itself during long autonomous runs, plan short sequences of
targets, spend less computation where it can, and report what it does. The baseline it
starts from is recorded in `docs/phase6_baseline_audit.md`; the design is in
`docs/phase6_reliability_and_optimization.md`. **Everything below ran on the Python
reference arena. Unreal Engine is not installed here, so nothing was verified against
Unreal.**

### 17.1 Commands

```bash
# One episode with the supervisor, short-horizon planning and adaptive computation
# (--agent selects the 3D mode; --policy deterministic|learned|hybrid selects the controller)
python -m chest_hunter simulate-3d --backend reference --agent hybrid --planner short-horizon \
    --supervisor enabled --adaptive-compute --seed 42 --profile

# Many consecutive episodes against one environment, with checkpoints
python -m chest_hunter endurance-test --episodes 500 --policy hybrid --planner short-horizon \
    --output benchmark_results/endurance            # --resume continues; Ctrl-C stops safely
python -m chest_hunter endurance-test --episodes 100 --dashboard --output runs/demo   # panel

# Processing time per component, optionally comparing vision profiles
python -m chest_hunter profile-agent --episodes 20 --vision-profiles phase5,efficient \
    --output benchmark_results/profiling

# Paired benchmark of the Phase 4, Phase 5 and Phase 6 configurations
python -m chest_hunter benchmark-phase6 --episodes 100 --seed 42 --workers 4

# How well the learned policy's confidence predicts reliable commands (held-out episodes)
python -m chest_hunter calibrate-policy --output benchmark_results/phase6/reliability
```

### 17.2 What was added

* **Supervisor** (`runtime/supervisor.py`). It watches every decision step for:
  * time in a state beyond its measured limit;
  * a stuck agent;
  * repeated state cycles;
  * a spawn point planned for again and again;
  * repeated interaction failures;
  * no progress;
  * refused transitions;
  * invalid commands, frames or telemetry;
  * model failures and slow inference;
  * a lost connection.

  Each detection becomes a structured event. The response follows a bounded escalation
  ladder: replan → abandon target → unstick → relocalise → deterministic fallback → abort.
  No intervention is repeated for the same incident, there are per-incident and
  per-episode limits, and a monitor-only mode is available.
* **State specifications** (`runtime/state_spec.py`): for every agent state, its entry
  conditions, permitted commands, exits, maximum duration (at least twice the longest
  measured visit) and recovery transition. Invalid transitions are refused and logged
  instead of crashing the episode.
* **Planning** (`exploration/advanced_planner.py`): S = (p q + λ I) / (T + ε) − μ R,
  greedy or short-horizon beam search over 2–3 targets. Work is bounded, there is a 15 ms
  budget per search level, and ties are deterministic.
* **Adaptive computation:**
  * vision profiles (`phase5`, `efficient`, `dense`);
  * skipping frames when nothing can have changed;
  * the detector reuses the agent's colour segmentation instead of computing it again;
  * bounded caches and tracks;
  * one ONNX session per process.
* **Learned-policy monitors:** oscillation, failed interaction holds, logged policy switches.
  Reliability calibration of the hybrid gate is measured; the calibrated gate is opt-in.
* **Experience partitions** by environment version, agent version and simulation
  configuration (migration 5), and one `episode_experience` row per episode (migration 6).
* **Endurance runs:**
  * one research server and connection for the whole run, with a reset between episodes;
  * a new agent for every episode, with a check that the previous one was released;
  * atomic, checksummed checkpoints, with resume;
  * graceful stopping on SIGINT/SIGTERM;
  * bounded environment resets.
* **Monitoring:** a resource monitor (process vs system CPU and memory; GPU and Unreal
  frame time reported as unavailable) and a component profiler (latency and own-thread CPU
  time; mean, median, p95, max).
* **Operations panel** (Tk, optional). It shows the episode, state, policy, target, chests,
  rate, coverage, recoveries, supervisor warnings, frame rate, CPU, memory, inference latency
  and recent transitions. Its controls are start, pause, resume, stop safely, select policy,
  route strategy and vision profile, and export metrics.

### 17.3 Benchmark

`benchmark-phase6 --episodes 100 --seed 42 --workers 4 --model training/checkpoints/bc_synthetic_v1 --experience-db training/experience.db`.

* `research_arena_3d` at 640×360, spawn points given, up to 600 simulated seconds per
  episode.
* Five configurations on the same 100 seeds (paired), 500 episodes, 0 errors.
* Code: Phase 8 fixes up to F20. The F21 fix, committed during the run, does not touch these
  configurations.
* Full results: `docs/phase6_reliability_and_optimization.md`, section 12.1.

| Configuration | Chests / min [95 % CI] | Paired step vs previous row [95 % CI] | Completed / failed | Agent CPU s / episode |
|---|---|---|---|---|
| phase4_deterministic | **4.568 [4.474, 4.662]** | baseline | 100 / 0 | 7.3 |
| phase5_hybrid | 3.442 [3.334, 3.549] | −1.127 [−1.241, −1.012] | 99 / 0 (one hit the time limit) | 11.4 |
| + supervisor | 3.447 [3.351, 3.544] | +0.006 [−0.041, +0.053] | 100 / 0 | 11.7 |
| + short-horizon planning | 3.368 [3.268, 3.468] | −0.079 [−0.188, +0.029] | 100 / 0 | 11.8 |
| + adaptive computation | 2.797 [2.711, 2.882] | **−0.572 [−0.703, −0.440]** | 99 / 1 | 11.1 |

* **No Phase 6 component improves throughput in fault-free episodes.**
  * The supervisor is neutral here; its value is under injected faults (Phases 7 and 8).
  * Planning makes no measurable difference.
  * Adaptive computation saves 5 % of agent CPU but costs 17 % of throughput.
* **The Phase 4 deterministic controller remains the best configuration.**
* **0 over-reported openings** in 500 episodes.
* **The one failed episode** was a stuck agent that repeated its unstick manoeuvre in a
  blocked direction (finding F22). It was fixed afterwards; the replayed episode now
  completes.

### 17.4 Endurance runs

`endurance-test --episodes N --max-ticks-per-episode 1500`.

* Default endurance configuration, reference arena, 50 s episodes, one server connection per
  run.
* Commit `e1124bd`; full table in `docs/phase6_reliability_and_optimization.md`, section 12.2.

| Run | Completed | Failed / comm. failures / aborts | Throughput | RSS growth after warm-up | FDs / threads |
|---|---|---|---|---|---|
| 100 episodes | 100 / 100 | 0 / 0 / 0 | 5.18 chests/min | 1.36 MB per 100 episodes | 9 / 3, constant |
| 500 episodes | 500 / 500 | 0 / 0 / 0 | 5.31 | 0.54 MB per 100 | 9 / 3, constant |
| 1000 episodes | 1000 / 1000 | 0 / 0 / 0 | 5.37 | 0.47 MB per 100 | 9 / 3, constant |

* No leak criterion was exceeded (2 MB per 100 episodes).
* Every previous agent was released.
* These runs are not evidence about Unreal or about long episodes.

### 17.5 Profiling and resource use

Python decision pipeline, measured by the Phase 8 audit's resource profile (final code, 3
held-out maps per configuration, active exploration, deterministic policy, one process):

| configuration | episodes | sim s | wall s | pipeline CPU s / sim s (share of one core in real time) | pipeline CPU ms / analysed frame | frames analysed | bridge client CPU s | simulator CPU s | RSS first / last / max (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 320x180_5fps | 3 | 600.0 | 36.7 | 0.0097 | 1.938 | 3000 | 1.95 | 27.67 | 247.0 / 260.0 / 260.0 |
| 480x270_5fps | 3 | 600.0 | 45.2 | 0.0168 | 3.370 | 3000 | 2.32 | 31.38 | 265.8 / 269.9 / 269.9 |
| 640x360_10fps | 3 | 600.0 | 86.1 | 0.0348 | 4.636 | 4500 | 3.42 | 59.79 | 288.8 / 307.9 / 307.9 |
| 320x180_adaptive | 3 | 600.0 | 32.0 | 0.0092 | 2.147 | 2580 | 2.01 | 22.99 | 313.8 / 314.1 / 314.1 |

* **The agent's own thread** uses about 1 % of one core per simulated second at 320×180 and
  5 analysed frames per second, and 3.5 % at 640×360 and 10 fps.
* **In the Phase 6 benchmark** (640×360, spawn points) agent CPU is 0.027 s per simulated
  second for the deterministic controller and 0.034–0.035 s for the hybrid configurations.
  Learned inference costs 0.95 ms (p95 1.3–1.4 ms) and short-horizon planning 0.47 ms per
  decision.
* **Most process CPU is the Python reference arena** (physics and software renderer). It
  stands in for Unreal and says nothing about Unreal's cost.
* **Memory:** 97–110 MB for an endurance agent; 197–198 MiB peak per Phase 6 benchmark worker,
  including ONNX Runtime and the reference arena.
* **Not measured:** GPU load, Unreal frame time and Unreal memory (no GPU, no Unreal).
* Details: `docs/phase6_reliability_and_optimization.md`, section 12.3.

### 17.6 Configuration

New optional sections in `config/settings.json`, all validated. Simulation-state limits use
simulated seconds; infrastructure limits (inference budget, planning budget, resets,
metrics interval) use wall-clock time.

| Key | Default | Meaning |
|---|---|---|
| `supervisor.enabled`, `mode` | true, `enforce` | `monitor` detects and logs without acting |
| `supervisor.max_state_seconds` | 30 | limit for states without their own (per-state limits in the Phase 6 doc) |
| `supervisor.max_recovery_attempts` | 3 | interventions per incident |
| `supervisor.max_interventions` | 30 | per episode; then the episode is aborted |
| `supervisor.loop_detection_window`, `loop_repetitions` | 12, 3 | recent states examined; repetitions of a cycle |
| `supervisor.max_no_progress_seconds` | 90 | simulated seconds without an opening, resolution, reached target or new detection |
| `supervisor.stationary_seconds` | 6 | movement requested but no displacement |
| `supervisor.max_goal_plans`, `max_interaction_failures`, `max_invalid_observations` | 8, 3, 5 | per spawn point / in a row |
| `supervisor.inference_budget_ms` | 20 | p95 of the last 50 inferences (wall clock) |
| `planning.strategy` | `short_horizon` | or `greedy` (used when a Phase 6 planner is selected) |
| `planning.planning_horizon`, `beam_width`, `maximum_candidates` | 3, 5, 20 | search bounds |
| `planning.planning_budget_ms` | 15 | wall clock per search level |
| `planning.exploration_weight`, `recovery_weight`, `detour_factor` | 0.15, 0.005, 1.25 | λ, μ, straight-line detour |
| `performance.adaptive_vision`, `vision_profile`, `skip_unchanged_frames` | true, `efficient`, true | used by `--adaptive-compute` |
| `performance.max_tracking_objects` | 100 | tracker bound |
| `performance.metrics_interval_seconds` | 2 | resource samples (wall clock) |
| `endurance.default_episodes`, `max_ticks_per_episode`, `save_checkpoint_every` | 100, 100000, 10 | endurance defaults |
| `endurance.max_reset_attempts`, `reset_timeout_seconds` | 3, 30 | environment resets before a run stops |

### 17.7 Phase 6 limitations

* **Reference arena only.** Nothing in Phase 6 has run against Unreal Engine; GPU load and
  Unreal frame time are not measured.
* **No throughput gain from the Phase 6 additions** in fault-free episodes (17.3). Adaptive
  computation costs throughput and should be used only where CPU is the binding limit.
* **Wall-clock budgets** for planning (15 ms) and inference (20 ms) make exact replay under
  CPU load not guaranteed (Phase 8 finding F18).
* **Endurance runs** use 50-second episodes with spawn points. A slow growth of about 200
  live Python objects per 100 episodes was not traced to its source.
* **Benchmark scope:** one map. Generalisation to held-out maps is measured in Phases 7 and 8.

## 18. Phase 7: generalisation, self-improving exploration and deployment readiness

Phase 7 runs the agent on 3D environments it has never seen and does **not** tell it where
chests can appear. The agent explores frontiers, recognises chests visually and opens them,
and recovers from injected failures. The Python runtime is packaged so it can be checked and
reproduced. Everything was measured on the Python reference arena; **Unreal Engine was not
available**, so the Unreal side of these features is unverified (see
`docs/phase7_generalization_and_deployment.md`, section 14).

### 18.1 Commands

```bash
# Deployment checks, then one episode on a held-out procedural map, with no spawn points
python -m chest_hunter check-runtime
python -m chest_hunter simulate-3d --agent hybrid --exploration active --map procedural --seed 42 \
    --backend reference --observation-width 320 --observation-height 180

# The same with an injected failure (robustness test)
python -m chest_hunter simulate-3d --exploration active --map procedural --seed 42 \
    --inject-fault localisation_drift --fault-seed 0

# Train the exploration memory (training maps and training seeds only)
python -m chest_hunter train-exploration --episodes 160 --seed 1 --workers 4

# Benchmarks
python -m chest_hunter benchmark-generalization --episodes 100 --held-out-maps \
    --workers 4 --output benchmark_results/phase7
python -m chest_hunter benchmark-exploration --episodes 20 --workers 4 \
    --output benchmark_results/phase7/strategies
python -m chest_hunter benchmark-robustness --episodes 100 --failure-injection \
    --workers 4 --output benchmark_results/robustness
python -m chest_hunter acceptance-phase7 --episodes 10 --workers 4
python -m chest_hunter profile-agent --episodes 20 --map procedural --exploration active \
    --observation-width 320 --observation-height 180 --max-ticks 6000 \
    --report benchmark_results/resource_profile.json

# Structured logs
python -m chest_hunter --log-format json simulate-3d --exploration active --map procedural
```

* `benchmark-generalization` without `--held-out-maps` keeps its Phase 5 behaviour.
* `--map procedural` picks the held-out layout `p7_<family>_test_<seed mod 1000>` (family =
  seed mod 4). Use `--map-category` / `--map-split` to choose others, or give any map id,
  such as `p7_complex_test_000_unfamiliar`.

### 18.2 What was added

* **Procedural environment families** `open`, `indoor`, `mixed` and `complex`. They have:
  * deterministic seeds and disjoint train/test seed ranges;
  * validated starts and navigable regions;
  * flagged unreachable chests;
  * a stored generator version.
  Six shift variants (obstacles, lighting, chest appearance, start, elevation, combined) and
  the `unfamiliar` acceptance variant are all evaluation-only.
* **Active exploration** without spawn points:
  * a coverage map from walkability raycasts and the camera's view wedge;
  * frontier blocks scored `S = (λg G + λc C) / (T + ε) − λr R` under five strategies;
  * event-driven and budgeted (20 ms, at most 4 path queries per selection);
  * information-aware camera sweeps in one direction;
  * looking at the most promising sector while travelling.
* **Uncertainty-aware decisions.** Approach / inspect / ignore thresholds on detector score,
  hits and position sigma; the score is never treated as a probability. Odometry-jump
  detection with a localisation confidence, and relocalisation by the supervisor.
* **Memory.**
  * Bounded episode memory: coverage cells, candidates, tracks.
  * A persistent, versioned **exploration memory** of category-level statistics, plus a
    Thompson-sampling **strategy bandit**. It is updated only between training episodes on
    training maps and is read-only in evaluation.
* **Robustness.** Ten controlled failure kinds, a robustness benchmark with paired fault-free
  baselines, a blank-band frame check, and a recovery budget that grows with the chests found.
* **Deployment.**
  * `requirements-lock.txt` (verified in fresh Python 3.11 and 3.12 venvs).
  * `config_version: 7`.
  * `check-runtime` startup checks with remedies.
  * `--log-format json`.
  * Graceful SIGTERM handling.
  * Model, agent and generator versions in every benchmark row.
* **Resource report.** `profile-agent --report` measures the Python decision pipeline and the
  simulator separately.
* **Dashboard.** A coverage map and frontier target, plus a panel with detection confidence,
  localisation confidence, world-model uncertainty, recent recovery events, versions, map
  complexity, chests per minute, latency, memory and the latest generalisation gap.

### 18.3 Configuration

New `exploration` section and `config_version` (both optional; a file without them reads
as version 0 with these defaults):

| Key | Default | Meaning |
|---|---|---|
| `config_version` | 7 | configuration format; newer versions are refused |
| `exploration.mode` | `spawn_points` | or `active` (no spawn points; Phase 7) |
| `exploration.strategy` | `memory` | the trained memory's choice per map family, or a fixed strategy |
| `exploration.memory_path` | `training/exploration_memory_v1.json` | persistent exploration memory (read-only in evaluation) |
| `exploration.approach_confidence`, `approach_hits`, `inspect_confidence` | 0.55, 2, 0.35 | uncertainty thresholds for discovered chests |
| `exploration.benchmark_max_ticks` | 6000 | Phase 7 benchmark episode length (200 simulated seconds) |
| `exploration.benchmark_observation_width`, `_height` | 320, 180 | Phase 7 benchmark resolution |

### 18.4 Results

All measured on the Python reference arena, not Unreal.

Details are in `docs/phase7_generalization_and_deployment.md`, section 15.

* **Generalisation** (1,600 episodes, held-out seeds and maps):
  * No significant gap for new layouts, obstacle arrangements, start poses or extra
    elevations.
  * Significant gaps for lighting (G = +0.11) and for chest appearance (+0.28), the latter a
    perception failure (F15).
  * This benchmark ran before the Phase 8 fixes. F21 ended 18 of its 800 Phase 7 episodes
    early, so indoor results are understated.
* **Without spawn points**, the Phase 7 agent opens about 1.6 chests per simulated minute,
  against 3.3 for the Phase 6 agent given spawn points.
* **Robustness** (200 episodes, ten fault kinds, paired twins): 85 of 94 applied faults
  recovered, 100 % clean terminations, 0 failed episodes, 0 over-reported openings. No fault
  kind has a measurable throughput cost.
* **Acceptance** (held-out map, 600 s episodes):
  * clean: 1.16 chests/min, 57 % of the reachable chests opened, interaction success 95 %,
    0 failures;
  * coverage 74.5 %, short of the 85 % target;
  * exact replay not guaranteed (F18).
* **Exploration strategies:** the trained memory/bandit is not better than the best fixed
  strategy.


## 19. Phase 8: full-system audit and readiness assessment

Phase 8 is an independent audit of Phases 1–7. It does not assume they work: it checks what
actually runs, measures it, fixes confirmed defects, and states readiness on the evidence. The
full report is `docs/phase8_full_system_audit.md`; the scorecard is
`docs/phase8_readiness_scorecard.md`.

**Verdict.** The verdict is UNVERIFIED for the intended Unreal Engine research environment.
That environment has never been built or run: there are no binaries and no map asset.
Every end-to-end result comes from the Python reference arena, a protocol-identical mock of
the Unreal server. Within that mock, the autonomous loop works end to end. The figures are in
the report.

### 19.1 Commands

```bash
# The whole audit: inventory, Unreal static audit, traced end-to-end episode (with video),
# ground-truth isolation, bridge security, vision, navigation, interaction, policy and planner
# comparisons, endurance, generalisation, resources, database integrity
python -m chest_hunter audit-system --episodes 40 --held-out-maps --profile-resources \
    --verify-database --workers 4 --output benchmark_results/phase8

# Export layouts for the Unreal project (any map; Phase 7 maps could not be exported before)
python -m chest_hunter export-arena-3d --map p7_complex_test_000_unfamiliar
```

`audit-system` writes the thirteen artifacts listed in the brief (`component_inventory.json` …
`audit_summary.json`) to `benchmark_results/phase8`. An evaluation it cannot run is recorded
as BLOCKED, never as zeros. It reuses the long Phase 7 results
(`benchmark_results/phase7`, `benchmark_results/robustness`) and the Phase 6 endurance runs,
and names their source.

### 19.2 Defects found and fixed

| # | Defect | Fix |
|---|---|---|
| F2 | Phase 7 maps could not be exported for Unreal at all | `export-arena-3d --map` |
| F7 | `profile-agent`, `benchmark-phase6`, `endurance-test` and `calibrate-policy` refused to run without an initialised database they never use | removed the requirement |
| F8 | experience partitions mixed supervised and unsupervised configurations | supervisor and monitors added to the partition key |
| F9 | every endurance resume after a crash in the session-finished window failed (integrity trigger) | lost-result sessions marked superseded |
| F11 | opened chests judged "closed" by the visual check were never credited (5.7 % of real openings); the agent re-aligned on them | bounded look-back, then credit on the agent's own opening notification (counted separately) |
| F12 | openings found by active exploration were not stored in SQLite | schema 7, `discovered_openings` |
| F14 | an opening could be credited to the wrong spawn point after a track mis-association | credit the spawn point actually reached |
| F1 | two controller handlers caught every exception silently | `BridgeError` only, counted |
| F17 | the exploring agent's normal frontier → scan cycle was read as a transition loop (489 false detections in 100 fault-free episodes; one 600 s episode aborted) | new coverage clears the loop window |
| F19 | after an F14 reattribution the original spawn point was credited a second time (1–4 phantom openings per 40 episodes) | both ids leave the unverified set |
| F20 | a failed interaction was credited on mixed visual evidence without the opening notification | vision-only credit needs unanimous looks |
| F21 | exploration was declared complete after the first scan when no ray crossed the start room's doorway (18 of 800 Phase 7 episodes ended at 3.9 s) | one denser ray fan before giving up; a mid-episode case remains (1 map in 100) |
| F22 | stuck off the NavMesh with no walked trail, the agent repeated its unstick manoeuvre in the same blocked direction until the supervisor aborted the episode (the one failed episode of the Phase 6 benchmark) | each further trail-less manoeuvre turns by a quarter turn |
| F23 | an `audit-system` re-run resumed the comparison row journals of an earlier run and reported rows played by older code | resume only when the recorded source tree matches the current, unmodified source |

Open: **F18**. Decisions depend on two wall-clock limits, so exact replay of an episode is not
guaranteed under CPU load. **F15**. Chest colours outside the trained range defeat detection.
**F3**. The Unreal builder ignores `surface_value`.

Every fixed defect has a regression test in `tests/test_phase8_audit.py`,
`tests/test_endurance.py` or `tests/test_phase6_workflow.py`. Each test replays the exact
episode that reproduced the defect where that was possible.

## 20. Running the tests

```bash
python -m unittest discover -s tests -v                # every phase (standard library only)
xvfb-run -a python -m unittest discover -s tests       # including the GUI tests on a headless Linux box
python -m pip install onnx onnxruntime                 # optional: enables the ONNX export tests
```

* **Coverage.** 553 tests:
  * Phase 1: 127; Phase 2: 109; Phase 3: 80.
  * Phase 4: 143. That is 141 test methods holding all 194 original cases (parametrised
    cases became sub-tests), plus 2 database migration tests.
  * Phase 5: 94. They cover:
    * the recorder and exact replay, and the dataset pipeline;
    * a numerical gradient check, training, and ONNX FP32/INT8 parity;
    * the policy selector's safety layer, and opening bookkeeping across handovers
      (stale phases dropped; each opening counted exactly once with a disabled model
      or frequent stuck handovers);
    * benchmark summaries with ground truth as the primary metric;
    * the Bayesian estimator, route memory, planner and recovery classes;
    * procedural maps and episodes on unseen maps;
    * CLI, configuration and migration 4;
    * teleoperation and the dashboard under Xvfb;
    * the end-to-end acceptance workflow.

  All tests are
  `unittest` test cases. The Phase 4 tests were first written for pytest and converted
  in Phase 5 without dropping any case; pytest can still run the whole suite.
  Earlier tests still pass unchanged, except two config tests whose example
  "unknown section" was literally named after a section that later became
  real (`navigation`, then `vision`).
* **Phase 2 tests cover:**
  * A*: straight lines, walls, blocked and enclosed goals,
    start = goal, out of bounds, diagonals, no corner-cutting,
    determinism, optimality against Dijkstra, cache invalidation.
  * Collision and line of sight.
  * Map generation, spawning statistics and episode resets.
  * Planner strategies, movement (arrival, waypoint progress, oscillation,
    stuck detection, acceleration limits).
  * Every FSM path, including failed interactions, path-blocked recovery,
    sealed targets, unreachable targets and tick limits.
  * Database integration (session, run row, statistics equal metrics,
    map association, no duplicates, repeatability).
  * No SQL and no A* during navigation ticks, and headless runs not
    importing Tkinter.
  * Benchmark determinism, the CLI and the GUI.
* **Phase 3 tests cover:**
  * Observations: validation, timestamps and frame IDs, failed captures,
    buffer reuse, all four resolutions, deterministic rendering without a
    display, camera round trip and the follow camera's dead zone.
  * Chest detection: present and absent, several chests, every scale,
    chests cut by the frame edge, partial occlusion by the agent, rejection
    of similar gold props, opened vs unopened under noise, NMS, and input
    validation.
  * Occupancy: observed cells match ground truth, unseen cells stay
    unknown, the map grows with motion, new walls are found, and map edges.
  * Tracking: stable identities and confirmation, short misses, expiry only
    while in view, fast expiry of tentative tracks, bounded memory, no
    duplicates, and one track kept through opening.
  * Localisation: world position under a moving camera, heading, a missing
    marker, a non-identity transform, noise reduction with motion
    prediction, and confidence decay and loss.
  * Navigation and behaviour:
    * Planning grid tracks the estimate; unknown cells cost more.
    * Frontier and prior scoring; unreachable frontiers are blacklisted.
    * Full episodes in which every reachable chest is opened and visually
      verified, with consistent SQLite records and deterministic repeats.
    * Failed and out-of-range interactions; replanning after a new wall.
    * Stuck recovery, lost targets, frame dropout and a missing marker.
    * Low resolution, late verification and the configuration switches.
    * Headless runs without Tkinter.
  * Ground-truth isolation:
    * An import audit of the policy modules.
    * No simulator objects held by the controller.
    * Chests that exist but are not drawn are judged empty.
  * Benchmark exports, pairing, determinism and argument validation; the
    visual CLI commands; vision config validation; and the visual debug GUI.
* **Isolation.** Tests use temporary or in-memory databases only.
* **GUI tests** are skipped automatically when Tkinter or a display is
  missing. Modules that open Tk windows free their Tk objects on the main thread when
  they finish, and tests that run Tk next to the mock server's threads keep automatic
  garbage collection off until then (`tests/tk_support.py`). Otherwise a later
  mock-server thread could free a leftover Tk object, which aborts the test process
  ("Tcl_AsyncDelete: async handler deleted by the wrong thread"); this happened in some
  Python 3.12 runs under Xvfb.

Verified with `python -m unittest discover -s tests` on Python 3.11 and 3.13 (553 tests
pass, 6 GUI tests skipped without a display) and with `xvfb-run -a` on Python 3.12 (553
pass, none skipped). `mypy --strict` passes on all 99 source modules, and ruff reports
nothing for the rule set E, F, W, B, UP, SIM, I (SIM117 excluded, line length 100).
The Unreal C++ code is not built by these
tests (see section 15); `tests/test_unreal_sources.py` compiles only the shared
SplitMix64/UnitRoll functions with the host C++ compiler.

## 21. Example output

```text
$ python -m chest_hunter simulate --map research_arena_01 --strategy weighted --seed 42 --headless
Simulating map research_arena_01 headless: strategy weighted, seed 42, episode 0, max 100000 ticks ...
Autonomous run finished: session 1, strategy weighted, seed 42, episode 0
  Outcome:               completed (all reachable chest locations visited)
  Simulated time:        137.333 s (4120 ticks at 30 Hz)
  Chest locations:       35 candidates, 35 visited, 0 unreachable, 0 abandoned
  Chests:                25 spawned, 25 found, 25 opened, 0 failed interaction(s)
  Chests opened/minute:  10.92 (simulated time)
  Travel distance:       368.4 cells (3684 world units)
  Recoveries:            0
  Tokens (fictional):    0
  A* searches:           33 (0 cache hits) for 33 route(s)
  Observations stored:   35 in 4 transaction(s)
  CPU time:              0.706 s (5,840 ticks per CPU second)

Session 1 on map research_arena_01 [completed]
  Duration:              137.333 s
  Locations visited:     35
  Chests found:          25
  Chests opened:         25
  Failed interactions:   0
  Chests opened/minute:  10.92
Autonomous agent run
  Strategy:                 weighted
  Seed / episode:           42 / 0
  Outcome:                  completed
```

That run's CPU time includes building the 34 distance fields for the map;
the episode itself takes about 0.05 s. Running
`simulate --render --strategy weighted --seed 42` gives exactly the same
result in the GUI: 4,120 ticks, 25 opened, 10.92 chests/min.

## 22. Programmatic use

```python
from chest_hunter.agent import SimulationRunner
from chest_hunter.config import load_config
from chest_hunter.database import DatabaseManager

settings = load_config()
with DatabaseManager(settings.database.path) as db:
    db.initialize()
    result = SimulationRunner(db, settings, strategy="nearest", seed=7).run()
    print(result.outcome, result.metrics.chests_per_minute, result.session_statistics)
```

For the vision-based agent, pass `agent="visual"` (or `"chest_vision"`) and
optionally `evaluate=True` to measure perception against ground truth:

```python
    runner = SimulationRunner(db, settings, map_id="visual_arena_01", seed=42,
                              agent="visual", evaluate=True)
    result = runner.run()
    print(runner.vision_evaluator.summary()["precision"], runner.visual_controller.analyses)
```

`SimulationRunner.step(n)` advances n fixed ticks, which is useful for custom
loops. `runner.env.add_obstacle(...)` changes the map at runtime; cached
paths are invalidated automatically.

## 23. Troubleshooting

| Symptom | Fix |
|---|---|
| `error: The GUI needs Tkinter` | Install Tkinter (`sudo apt install python3-tk` on Debian/Ubuntu; included in python.org installers) or use `--headless` |
| `error: Could not open the GUI window (no display ...)` | Run on a desktop session, use `xvfb-run -a python -m chest_hunter simulate --render`, or use `--headless` |
| `error: Map '...' has no enabled chest locations` | Run `python -m chest_hunter seed-demo` or add chests with `add-chest` |
| `error: None of the chest locations ... fit inside the grid` | Increase `map_width`/`map_height` or `world_units_per_cell` |
| `error: Database not found ... Run 'init-db' first` | Run `python -m chest_hunter init-db` |
| `No module named chest_hunter` | Activate the virtual environment and `pip install -e .`, or set `PYTHONPATH=src` |
| `error: Invalid configuration in ...` | The message names the field; see section 7 for ranges |
| GUI looks slow | Use the speed selector; simulation results do not depend on speed |
| Need more detail | Add `--log-level DEBUG` (every state transition is logged) |
| `No module named 'cv2'` or `'numpy'` with `--agent visual` | `pip install -e .` (or `pip install numpy opencv-python-headless`). The oracle agent does not need them |
| `error: Map 'visual_arena_01' has no enabled chest locations` | `python -m chest_hunter seed-demo --scenario visual_arena` |
| `error: ... needs vision.enabled = true` | Set `vision.enabled` to `true` in the config, or use `--agent oracle` |
| Visual agent slower than expected | Use 480x270 (`vision.default_width/height`); perception cost scales with pixels (section 3) |
| `simulate-3d`: `no research server on 127.0.0.1:47820` | Start the Unreal research project (`docs/phase4_3d_architecture.md`) or add `--backend reference` |
| `simulate-3d`: `... must be a loopback address` | The bridge only connects to this machine; use `127.0.0.1` |
| `simulate-3d`: `observation size ... is not supported` | Use 320x180, 480x270, 640x360 or 800x450 |
| `forbidden` from the research server | Expected for the visual agent: privileged state is only served to `--agent oracle` / `--evaluate` runs on a server started for them |
| `no trained policy in ... (policy.json missing)` | Train one (`train-policy`) or pass `--model training/checkpoints/bc_synthetic_v1` |
| `ONNX export needs the 'onnx' package` | `pip install onnx onnxruntime` (optional; the NumPy backend runs without them) |
| `argument --config: ...` with `train-policy` | The training configuration option is `--train-config` (`--config` selects the settings file) |
| Human demonstrations: `need Tkinter and a display` | Run on a desktop session (or `xvfb-run` for testing) |

## 24. Known limitations

* **2D only.** `z` is ignored, so stacked chests share a grid position. They
  are visited one after the other without moving.
* **Fixed spawns.** Spawn presence is fixed for an episode. Chests do not
  respawn and there are no other players or moving obstacles (runtime
  obstacles can be added through the API).
* **Weak default on demo data.** On the demo data the default `weighted`
  strategy is about 5% slower than `nearest` (section 13). `lookahead` uses a
  rate objective with a small beam, not an exact tour optimiser.
* **No path to the target.** When A* finds no path, the location is reported
  as unreachable but not stored as an observation, because the Phase 1
  schema has no "not observed" outcome. It is recorded in `simulation_runs`
  instead.
* **Batched writes.** At most one batch (`observation_batch_size`) can be
  lost if the process is killed. Normal completion, failure, cancel and
  window close all flush.
* **Session timestamps** use the simulated timeline (section 11), so a
  session that started now may show an `ended_at` a few simulated minutes
  later.
* **Untested platforms.** The GUI was verified on Linux/Xvfb only; macOS and
  Windows are untested.
* **Map setup.** Building the distance fields costs about 0.6-0.8 s per
  distinct map layout. This matters for `--vary-obstacles` benchmarks, where
  setup dominates wall time.

Phase 3:

* **Known camera geometry.** Each frame comes with the camera's origin and
  scale (section 14.7). The agent measures everything else from pixels, but
  it does not estimate the camera pose itself.
* **Top-down, no occlusion.** The camera sees the whole view from above, so
  walls never hide chests and the sealed-vault chest is visible. This is
  why the agent can mark it unreachable rather than search for it.
* **Stacked chests.** Two spawn points 0.28 cells apart (demo arena,
  Blue House attic and basement) look like one object from above, so the
  agent cannot tell which one holds the chest. With seed 42 the basement
  chest was opened but credited to the attic (which was empty), and the
  basement was then abandoned. Counts stay correct (25 of 25 opened), but
  the per-location record for that pair is wrong. Fixing this needs height
  information, which the 2D simulator does not have.
* **Low resolution.** At 320x180, chests are about 6 px wide. Recall drops
  (0.92 in the seed-42 episode against 0.96 at 640x360), and opened chests
  can need a late confirmation (section 14.9). The colour-only detector is
  unusable there (recall 0.39).
* **Priors are required for recording.** Observations are stored per
  database location, so a chest found away from every known spawn point is
  seen but not recorded.
* **Simple world.** Fixed lighting model, one camera style and synthetic
  sprites. There are no textures, shadows, perspective or motion blur, and
  the templates match the rendering exactly.

## 25. Recommended next steps

1. **Tour optimisation for full sweeps.** For example, nearest-neighbour
   plus 2-opt over the distance oracle, since full-sweep throughput is
   tour-bound.
2. **Learn spawn probabilities online** from `chest_observations` (for
   example Beta-Bernoulli updates) and feed them back as `P_i`.
3. **Perception radius.** Let the agent detect empty spawn points from a
   distance with line of sight and skip them.
4. **Richer dynamics.** Dynamic obstacles, chest respawns, a moving "storm"
   boundary, multiple agents.
5. **A stable environment interface** for later RL experiments (observation
   and action spaces wrapping `SimulationRunner.step`).
6. **Build and verify the prepared Unreal project** (Phase 4): compile the
   C++ module, run its automation tests, connect the agent, then recalibrate
   the detector and templates on Unreal frames and repeat the benchmark there.
7. **Camera pose from images** (visual odometry against the occupancy map),
   removing the last piece of geometric metadata from observations.
8. **Harder rendering:** perspective, partial occlusion by walls, textures
   and lighting changes. Then a learned detector could be compared with the
   classical ones on the same benchmark.
9. **Active perception:** choose viewpoints that resolve the most spawn
   points per second, rather than driving to frontier centres.
10. **Phase 4: confirm more openings.** About 7 % of real hybrid openings went
    unrecorded because the opened chest was never seen clearly. Use the
    reward/notification event together with a short look-back manoeuvre.
11. **Phase 4: stairs for NavMesh-free navigation.** Detect stairs from
    raycast height profiles so that visual-nav mode can reach upper floors.
12. **Phase 4: learned detector on Unreal frames.** Once Unreal runs, collect
    labelled frames through `-ResearchEvaluation` and compare a small learned
    detector with the HSV/template pipeline.
13. **Phase 5: real human demonstrations.** Record them with the teleoperation
    window and compare a policy trained on human data with the expert-trained one.
14. **Phase 5: DAgger-style training.** Run the learned policy, let the deterministic
    expert label the states it visits, and retrain. This attacks the compounding errors of
    single-frame behaviour cloning.
15. **Phase 5: memory in the policy.** Frame stacking or a small recurrent layer, so the
    policy can remember targets that leave the view.
