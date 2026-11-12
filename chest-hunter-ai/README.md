# chest-hunter-ai

A lightweight research project: an **autonomous agent that navigates a
controlled 2D research environment, finds simulated chests, opens them, and
tries to maximise chests opened per minute.**

| Phase | Status | What it provides |
|---|---|---|
| 1. Data foundation | Done | SQLite database of chest spawn points, simulation sessions and observations; CLI; tests |
| 2. Autonomous navigation | Done | Grid-world simulator, A* pathfinding, route planning, continuous movement, 10-state agent, Tkinter GUI, headless runs, benchmarking |

> The simulator is entirely self-contained. The project never talks to a game
> client, game service or network. It never reads or writes game memory or
> injects into processes, never sends keyboard or mouse input to other
> applications, and contains no anti-cheat evasion. The demo map is a
> **fictional** research arena, and its coordinates are not real game data.

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

**Out of scope for Phase 2:** machine learning (including reinforcement
learning), computer vision, large language models, GPU use and external APIs.

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
* **Standard library only.** NumPy was not needed.

### Source layout

```text
chest-hunter-ai/
├── pyproject.toml, README.md, .gitignore
├── config/settings.json          # all settings (Phase 1 + Phase 2)
├── data/                         # SQLite DB + benchmark CSVs (git-ignored)
├── docs/gui_screenshot.png
├── scripts/seed_demo_chests.py
├── src/chest_hunter/
│   ├── cli.py, config.py, database.py, models.py, exceptions.py, logging_config.py
│   ├── chest_repository.py, session_repository.py, run_repository.py, demo_data.py
│   ├── simulation/   environment.py, map_generator.py, entities.py, collision.py, renderer.py
│   ├── navigation/   astar.py, route_planner.py, movement.py
│   ├── agent/        controller.py, states.py, objectives.py, recorder.py, runner.py
│   └── analytics/    metrics.py, benchmark.py
└── tests/  (13 modules, 236 tests)
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

## 4. Supported operating systems

* **Linux:** tested with Python 3.11, 3.12 and 3.13. The GUI was tested with
  Python 3.12 + Tk 8.6 under Xvfb.
* **macOS, Windows 10/11:** expected to work but not tested yet. The code is
  pure Python using `pathlib`, with ASCII-only console output.

The GUI needs Tkinter. It ships with the python.org installers for Windows
and macOS; on Debian/Ubuntu install `python3-tk`. Headless mode, the
benchmark and every other command work without Tkinter.

## 5. Installation

```bash
python3.11 -m venv .venv          # any Python >= 3.11
source .venv/bin/activate         # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .        # editable install; no third-party dependencies
```

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

Overrides: `--config PATH` or `CHEST_HUNTER_CONFIG`, `--db PATH`,
`--log-level LEVEL`, `CHEST_HUNTER_HOME`. Relative paths resolve against the
project root, never the current directory.

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

## 11. Database structure (schema version 2)

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
| `seed-demo [--map MAP]` | Insert the 35 fictional demo locations (idempotent) |
| `list-chests`, `show-chest`, `add-chest`, `update-chest`, `delete-chest` | Manage spawn points |
| `start-session`, `record-observation`, `complete-session`, `abort-session`, `list-sessions`, `unvisited-chests` | Manual sessions (Phase 1) |
| `session-stats --session ID` | Session statistics. Agent-driven sessions also show strategy, seed, outcome, travel and recoveries |
| `map-stats [--map MAP] [--min-samples N]` | Map statistics, including observed spawn frequency |
| `simulate ...` | Run the autonomous agent (section 6) |
| `benchmark ...` | Paired strategy comparison (section 13) |

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

## 14. Running the tests

```bash
python -m unittest discover -s tests -v              # 236 tests
xvfb-run -a python -m unittest tests.test_renderer -v  # GUI tests on a headless Linux box
```

* **Coverage.** 127 Phase 1 tests (still passing unchanged, except one config
  test whose example "unknown section" was literally named `navigation`) and
  109 Phase 2 tests.
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
* **Isolation.** Tests use temporary or in-memory databases only.
* **GUI tests** are skipped automatically when Tkinter or a display is
  missing.

Verified on Python 3.11 and 3.13 (236 passed, the 2 GUI tests skipped
because those builds lack Tkinter) and Python 3.12 + Tk under Xvfb (236
passed, none skipped).

## 15. Example output

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

## 16. Programmatic use

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

`SimulationRunner.step(n)` advances n fixed ticks, which is useful for custom
loops. `runner.env.add_obstacle(...)` changes the map at runtime; cached
paths are invalidated automatically.

## 17. Troubleshooting

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

## 18. Known limitations

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

## 19. Recommended next steps

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
6. **A custom Unreal Engine research environment** reusing the same
   database, runner and metrics.
