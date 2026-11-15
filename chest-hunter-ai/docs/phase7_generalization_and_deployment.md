# Phase 7: generalisation, self-improving exploration and deployment readiness

Phase 7 makes the research agent work on 3D environments it has never seen. It does this
without being told where chests can appear, recovers from injected failures, and is packaged
as a runtime that can be checked and reproduced.

Everything here runs in the **controlled research simulator only**. The agent talks to the
research environment through the localhost bridge protocol of Phase 4. It has no interface to
any commercial game, game service or game process. Nothing in the project automates or
imitates a human player on a live service, reads protected memory, or evades anti-cheat
measures.

**Verification status.** Every measurement in this document was made on the Python reference
arena. That arena implements the same bridge protocol as the Unreal research project: same
requests, observations and navigation services. Unreal Engine is not installed on the
development machine. The components that still need Unreal validation are listed in section
14.

## 1. Architecture

```
generator7 (maps) ──► reference arena / Unreal research server ◄── bridge (localhost TCP)
                                                                         │
Runner3D ── FaultInjector (robustness tests only) ───────────────────────┤
   │                                                                     │
   ├── VisualAgent3D ── perception ── WorldModel3D (discovery mode) ─────┘
   │        │                              │
   │        ├── ActiveExplorer (frontiers, coverage map, camera sectors)
   │        ├── ConfidenceThresholds (approach / inspect / ignore)
   │        └── recovery (bounded) ◄── Supervisor (Phase 6 + odometry jumps, blank frames)
   └── PolicySelector (Phase 5 hybrid policy, unchanged)

ExplorationMemory (category statistics + strategy bandit) ── read at episode start,
updated between training episodes only
```

New modules:

| Module | Purpose |
|---|---|
| `simulation3d/generator7.py` | procedural environment families, held-out splits, shift variants, validation, versioning |
| `exploration/active.py` | coverage map, frontier candidates and scoring, camera sectors, uncertainty thresholds |
| `exploration/self_improving.py` | exploration memory (category statistics) and the contextual strategy bandit |
| `runtime/faults.py` | controlled failure injection with detection and recovery assessment |
| `runtime/startup.py` | deployment startup checks |
| `analytics/phase7_benchmark.py` | generalisation benchmark, strategy comparison, memory training |
| `analytics/robustness.py` | robustness benchmark |
| `cli_phase7.py` | the Phase 7 commands and `simulate-3d` options |

Changed modules:

* the controller (`agent/controller_3d.py`): exploration states, localisation monitor,
  discovery-mode recovery rules;
* the world model (`localization/world_model_3d.py`): discovery mode;
* `runner_3d.py`: exploration, thresholds, faults, coverage scoring;
* the supervisor: odometry jumps, blank-band frames;
* the reference environment: dynamic obstacles, appearance, generator dispatch;
* configuration: the `exploration` section and `config_version`;
* logging: JSON output;
* `cli.py`: SIGTERM handling;
* the dashboard: coverage map and Phase 7 panel;
* `profile-agent --report`.

## 2. Environment generation

`p7_<category>_<split>_<NNN>[_<variant>]`, generated deterministically from the map id.
Layout, obstacles, appearance and start pose each use their own seeded random stream.

| Category | Size | Complexity | Content |
|---|---|---|---|
| `open` | 90 m | 0.3 | a few shelters, containers, crates, rocks, tree groves; broad visibility |
| `indoor` | 64 m | 0.6 | one building complex of rooms, corridors and doors; the start is inside |
| `mixed` | 90 m | 0.5 | 3–5 buildings with interior rooms, an optional plateau, outdoor obstacles |
| `complex` | 90 m | 0.8 | 2–4 buildings with upper floors, 2–3 raised levels (1.0, 1.5, 2.25 m) by stairs, narrow alleys |

**Splits.** `train` and `test` use disjoint seed ranges (20 000+ and 70 000+, plus 1 000 per
category).

**Held-out maps.** A map is held out if it is a test layout or has any variant suffix:
variants of training layouts are evaluation-only. `assert_training_map()` refuses held-out
maps wherever training data is written: the exploration memory, demonstrations, and
persistent statistics.

**Variants** (distribution shifts, applied on top of a base layout):

| Variant | Change | Training range |
|---|---|---|
| `obstacles` | another obstacle stream, same buildings | - |
| `light` | lighting 0.62–0.72 or 1.22–1.32 | 0.85–1.12 |
| `appearance` | chest brightness 0.78–0.84 or 1.14–1.20; hue ±6–9; surface brightness 0.75–0.82 or 1.18–1.25 | 0.90–1.08; ±3; 0.92–1.08 |
| `start` | another start position and camera direction | - |
| `elevation` | two extra raised levels (1.5 and 2.25 m) with chests | - |
| `combined` | all of the above | - |

**Validation.** Up to 30 attempts per map id. A layout is accepted only if:

* the start is on the navigation graph;
* the region reachable from the start covers at least 15 % of the map;
* there are at least 8 spawn points;
* at least 80 % of the spawn points meant to be reachable have a path.

Unreachable spawn points are *kept and flagged* (`reachable=False`, tag `unreachable`). An
example is the sealed room some layouts contain (indoor complexes leave no space for one).
The generator version (`GENERATOR_VERSION = "7.0"`), settings and settings hash are stored in
`Arena3D.meta` and exported with the layout for Unreal (`export-arena-3d --map MAP_ID`; the `--map` option was added in Phase 8 - before it, only `research_arena_3d` could be exported, audit finding F2).

## 3. Active exploration (`exploration/active.py`)

The agent gets **no spawn points** (`--exploration active`). It builds its own picture of
the environment from three sources it is allowed to use:

* its camera;
* its odometry;
* the navigation service's path queries and walkability raycasts (`NAV_RAYCAST`, capped at
  15 m by the protocol).

**Coverage map.** A sparse grid of 2 m cells per floor level (z quantised to 0.25 m),
bounded at 30 000 cells (cells beyond the bound are counted as `dropped`).

* A 16-ray walkability fan is cast every 2.5 m of travel, or on a level change. Ray cells
  become walkable. The end of a full-length ray is marked as an open edge (space continues).
  The cell beyond a short ray is marked blocked.
* After each analysed frame, cells inside the camera's horizontal view wedge count as seen.
  A cell must be closer than both the 11 m view range and the walkable distance in that
  direction, so cells behind walls do not count.

**Frontier candidates.** Unseen walkable cells and open edges are grouped in 3×3-cell
blocks. Blocks already reached and cells found unreachable are skipped. Each block gets a
representative cell and is scored

```
S_i = (λg·G_i + λc·C_i) / (T_i + scan + ε) − λr·R_i
```

* `G` – expected information gain: unseen cells in the block, plus 4 per open edge.
* `C` – chest-discovery value. It is the learned discovery rate of the block's region type
  (enclosed / open / elevated) times the block's cell count, plus 1 per unconfirmed chest
  detection within 6 m.
* `T` – travel time. For the best few candidates it uses the NavMesh path length divided by
  the agent's speed; for the others, the straight distance × 1.3 detour. `scan` is 4 s and
  `ε` is 1 s.
* `R` – risk: failed approaches and unreachable marks within 4 m.

| Strategy | λg | λc | λr | Cluster bonus | Behaviour |
|---|---|---|---|---|---|
| `nearest_frontier` | – | – | 0.01 | – | score = 1/T: nearest worthwhile block |
| `max_information` | 1.0 | 0 | 0.01 | – | most new cells per second |
| `chest_prior` | 0.5 | 40 | 0.01 | – | one expected chest is worth 40 cells |
| `cluster` | 0.5 | 10 | 0.01 | 8 · Σ exp(−d/10 m) over discoveries | stays near found chests |
| `risk_adjusted` | 1.0 | 10 | 0.3 | – | avoids areas where approaches failed |

**Bounded computation.**

* Candidates are rebuilt only when needed: after 12 newly seen cells, when a target is
  needed, or after a target was reached or found unreachable.
* At most 36 candidates are pre-ranked and 12 scored. At most 4 path queries are made per
  selection.
* A 20 ms budget returns the best candidate found so far. Overruns are counted.
* With no candidate left, `done` is set and the episode can complete.

**Multi-scale search.** Each scale runs at its own rate, and the costs are measured
separately:

| Scale | When it runs | Measured as |
|---|---|---|
| global frontier selection | event-driven (the rules above) | `frontier_selection` |
| local navigation | path following every decision tick, replanning every 1.5 s or when the target area has been seen on the way | `route_selection`, `navigation_query` |
| visual inspection | at the vision-profile rate | `chest_detection` |
| exploration bookkeeping | fans and view integration, per analysed frame | `exploration_update` |

## 4. Camera search

`CameraSectors` splits the horizon into 12 sectors of 30°. Each sector tracks:

* the time it was last observed;
* its visibility (the fan's free distance / 11 m);
* detections and their best confidence;
* whether a detection there still needs inspection.

The gain of a sector is the number of unseen walkable cells in it within view range, times
its visibility. An uncertain detection adds 3. A sector observed in the last 20 s counts at
a quarter, unless it needs inspection.

**On arrival at a frontier target**, the scan plan takes the sectors with gain ≥ 1.5 in *one
sweep direction* from the current yaw. The direction is chosen by gain discounted by the
rotation needed. At most 5 sectors are visited. The scan stops when no sector promises
information. This avoids oscillating back and forth and scanning sectors that were just seen.

**While travelling**, the camera looks towards the best sector within 75° of the heading
and holds it for 1.5 s ("look while travelling"). This adds coverage without stopping.

## 5. Uncertainty

**Per track.** The detector's score, the number of hits and the position sigma. The raw
template-matching score is **not** treated as a probability. It is only compared with
thresholds, and those thresholds are validated by the benchmark outcomes (section 9), not
interpreted as calibrated confidence.

**Decision thresholds** (`ConfidenceThresholds`, configurable in `exploration`):

| Class | Rule | Effect |
|---|---|---|
| approach | confidence ≥ 0.55, ≥ 2 hits, sigma ≤ 1.2 m | the chest becomes a target |
| inspect | confidence ≥ 0.35 or ≥ 2 hits | the sector is flagged for a closer look; the track can become a target after more evidence |
| ignore | anything else, and every `chest_opened` track | – |

**Discovery mode** (`WorldModel3D(discover=True)`):

* A track that is not near an existing spawn point gets a *dynamic* spawn point with a
  negative id. Dynamic spawn points are never written to the database's location table.
* A track near *any* existing spawn point, including an opened one, is linked to it and not
  duplicated.
* When a dynamic spawn point is opened, it moves to where the chest was actually opened, so
  later sightings link to it.
* A discovered chest whose prompt never appears from two sides is given up.

**Localisation confidence.** If odometry moves further in one step than the agent can
(sprint speed × dt × 1.5 + 0.3 m), it is an odometry jump:

* the agent's localisation confidence is multiplied by 0.3;
* the supervisor raises `localisation_jump`, and its ladder relocalises: stale tracks are
  dropped and the agent does a full look-around.

**World-model uncertainty.** This is the mean sigma of the confirmed chest tracks. The
dashboard shows it.

## 6. Memory

| Memory | Lifetime | Bound | Content |
|---|---|---|---|
| coverage map | one episode | 30 000 cells | walkable / seen / blocked / open edges |
| candidates, blacklist, searched blocks | one episode | rebuilt on events; a blacklisted cell is never queried again | frontier bookkeeping |
| failures, discoveries | one episode | 200 entries each | risk and cluster terms |
| tracks | one episode | `performance.max_tracking_objects` (100); stale tracks expire | perception |
| exploration memory | persistent, versioned JSON | per map *category* only | discovery rates, travel speed, interaction / recovery / scan statistics, bandit |

The exploration memory never stores a map id, a position or a spawn outcome. It cannot carry
a held-out layout's answers.

* **Loading.** It loads only for the same agent version and generator version; any other
  version starts empty.
* **Writing.** It is written atomically, and only *between* training episodes on training
  maps.
* **Evaluation.** In evaluation it is read-only: `update(..., training=False)` raises.

## 7. Learning updates

`train-exploration` plays training layouts with **training seeds**. Benchmark seeds are moved
into the range from 900 000 up, and `assert_training_seed` refuses that range. For each
episode:

1. The bandit (Thompson sampling, Normal posterior, one context per map category) samples one
   strategy. The strategy is fixed for the whole episode; it is never changed during an
   episode.
2. After the episode, the memory adds:
   * the episode's cells seen and chests discovered per region type;
   * travel metres and seconds;
   * interaction attempts and openings;
   * recovery attempts and resolutions per failure class;
   * scans and the cells they revealed.
3. The bandit adds the episode's reward: chests opened per simulated minute (ground truth from
   the evaluation channel, used only to score the training episode).
4. With `--workers N`, N episodes run in parallel from the same memory snapshot. The memory is
   updated after the batch and saved after every update.

In evaluation the agent **exploits only**: the strategy with the best posterior mean for the
map's category. If the category has no training episodes it uses `max_information`. The
discovery rates feed `C`.

No new neural network was added. Heavyweight reinforcement learning is not justified by
these results (section 11).

**Training run** (`train-exploration --episodes 160 --seed 1 --workers 4`).

* Layouts `p7_<family>_train_000..039`, with the 4 families in rotation: 40 episodes per
  family.
* 6000 ticks (200 simulated seconds) per episode, 320×180, Phase 6 hybrid policy and
  supervisor.
* 857 s of wall clock. No errors.
* Mean reward 1.54 chests per simulated minute (95 % CI 1.43–1.65).

Posterior mean reward per arm (chests per simulated minute, ± posterior sd; n = training
episodes):

| Family | nearest_frontier | max_information | chest_prior | cluster | risk_adjusted | Bandit choice |
|---|---|---|---|---|---|---|
| open | 1.00 ± 0.40 (3) | **1.89 ± 0.24 (10)** | 1.68 ± 0.27 (8) | 1.81 ± 0.21 (13) | 1.77 ± 0.30 (6) | max_information |
| indoor | 0.74 ± 0.36 (4) | **1.06 ± 0.21 (13)** | 0.68 ± 0.36 (4) | 1.01 ± 0.27 (8) | 1.01 ± 0.23 (11) | max_information |
| mixed | 1.51 ± 0.27 (8) | **1.61 ± 0.27 (8)** | 1.51 ± 0.28 (7) | 1.58 ± 0.23 (11) | 1.43 ± 0.30 (6) | max_information |
| complex | 1.79 ± 0.21 (14) | 1.51 ± 0.30 (6) | 0.95 ± 0.57 (1) | 0.95 ± 0.57 (1) | **1.87 ± 0.18 (18)** | risk_adjusted |

How to read this:

* **The arms are close.** Apart from the clearly weak ones (nearest_frontier on open and
  indoor maps, chest_prior on indoor maps), the posterior intervals overlap. The bandit's
  choice is a *best current estimate*, not a demonstrated difference. Section 15 tests it
  against the fixed strategies on held-out maps.
* **Learned category statistics.**
  * Chests per newly seen cell in *elevated* regions: 0.24 (open), 0.04 (indoor),
    0.09 (mixed), 0.06 (complex).
  * In enclosed and open regions: 0.004–0.016.
  * Raised platforms hold far more chests per cell. This feeds `C` for the strategies that
    use it.
  * Travel speed 4.8–5.1 m/s; interaction success 0.92–0.96; scan gain 7–19 cells per scan.
  * Recovery success varies by failure class. It is 0.53–0.94 for blocked paths, 0.6–0.89 for
    failed interactions, 0.17–0.24 for alignment failures (mostly misdetections that are
    rightly given up), and 0 for unreachable targets (given up by design).

## 8. Generalisation test methodology

`benchmark-generalization --held-out-maps` runs 8 evaluation categories. Episode `e` of each
category uses map family `CATEGORIES[e % 4]` and layout index `e // 4`:

| Category | Map | Compared with `familiar` |
|---|---|---|
| familiar layouts, new spawn seeds | `p7_<cat>_train_NNN`, evaluation seed | – |
| new obstacle arrangements | `…_train_NNN_obstacles` | paired (same layout and seed) |
| unseen buildings and room layouts | `p7_<cat>_test_NNN` | unpaired |
| different lighting | `…_train_NNN_light` | paired |
| different chest appearances | `…_train_NNN_appearance` | paired |
| different camera start | `…_train_NNN_start` | paired |
| more complex elevations | `…_train_NNN_elevation` | paired |
| combined shifts | `p7_<cat>_test_NNN_combined` | unpaired |

**Configurations.**

* `phase7_active` – no spawn points; strategy from the exploration memory.
* `phase6_priors` – the Phase 6 hybrid agent with the supervisor. It is given the map's
  spawn points (not their outcomes) and plays the same episodes.

**Metrics.** Ground truth is used for these, through the evaluation channel, after the fact:

* throughput: chests actually opened per *simulated* minute;
* reachable chests opened;
* exploration coverage: the share of the reachable walkable area the camera covered, on the
  2 m grid;
* detection precision and recall: per-frame evaluation labels, pooled over episodes;
* chest-distance error and the agent's pose error;
* navigation success: routes started / (routes started + failed plans);
* recovery success: resolved / attempted recoveries, pooled.

The agent never receives ground truth.

**Gap.** `G = (E_familiar − E_unseen) / E_familiar` with `E` the mean throughput. It is
`None`, with the reason, when the familiar mean is 0. The 95 % interval is a percentile
bootstrap (2 000 resamples; pairs are resampled together for paired categories).

**Means and paired differences.** These have t-based 95 % intervals. The number of completed
episodes is reported per cell.

**Interruptions.** Every benchmark writes each finished episode row to
`rows.partial.jsonl` in its output directory as soon as the episode ends. This covers the
Phase 6 benchmark too. `--resume` plays only the missing jobs and refuses a journal written
with other parameters. The Phase 6 benchmark had to be re-run for this reason: its first run
was stopped by the session's 2-hour limit for background jobs after 540 of 700 jobs, and at
that time the rows were held in memory only.

## 9. Robustness testing

`benchmark-robustness --failure-injection` (the flag is required). Episode `e` injects
`FAULT_KINDS[e % 10]` on held-out layout `e` at a start drawn from the fault's window. Each
faulted episode is paired with a fault-free baseline on the same map, seed and episode.

| Fault | Injected at | Duration / trigger | Detection signal (agent or supervisor) |
|---|---|---|---|
| frame loss | bridge reply | 3 s of requested frames dropped | supervisor `invalid_observation` |
| corrupted frame | bridge reply | 3 s of frames with a zeroed band (30 % of rows) | supervisor `invalid_observation` (blank-band check) |
| false detection | detector output | 2 s of an extra chest box | not applicable: indistinguishable from a real chest at first |
| missed detection | detector output | 8 s without detections | not applicable: indistinguishable from an empty view |
| localisation drift | odometry | a permanent 2.5 m offset | odometry jump / supervisor `localisation_jump` |
| unexpected obstacle | environment | a 2 m crate 3 m ahead on the current path | navigation failure, recovery, `stationary` / `no_progress` |
| failed interaction | environment | the next 2 interactions fail | failed interactions |
| bridge disconnect | connection | the socket is closed | client re-send |
| slow inference | learned policy | the next 10 inferences take +35 ms (over the 20 ms budget) | selector timeout fallback / supervisor `inference_latency` |
| unreachable target | navigation service | the next goal area asked for has no path | unreachable / navigation failure |

**Scoring rules.**

* **Applied.** A fault counts as tested only if it was actually applied. A fault planned after
  the episode ended, or whose trigger never came, is reported separately.
* **Recovered.** The episode did not fail, and within 30 simulated seconds after the fault it
  opened a chest, resolved a target, or saw at least 10 new cells.
* **Terminated cleanly.** Recovered, or the episode ended with an explicit reason.

**Recovery stays bounded** by the Phase 6 mechanisms:

* supervisor ladders, per-incident attempts and the per-episode intervention cap;
* the controller's per-target attempt limit;
* the episode recovery budget, which in discovery mode grows with the chests found.

There is no unbounded retry anywhere. A blacklisted frontier is never queried again, and a
failed path is never served from the cache again.

## 10. CPU and memory profiling

`profile-agent --report PATH` plays episodes in one process, sequentially, and writes a JSON
report. It reports the **Python decision pipeline** and the **simulator** separately:

* **pipeline CPU** – the agent thread's CPU time in the top-level profiled sections:
  preprocessing, detection, tracking, interaction cues, exploration update, action generation
  (which includes route and frontier selection, navigation queries and learned inference),
  supervision and persistence. Nested sections are not added twice.
* **simulator CPU** – process CPU minus the above. This is the Python reference arena's
  physics and software renderer. It is **not** a measurement of the Unreal graphics workload,
  which needs a GPU (none was available; the report says so).
* memory – the process RSS after each episode;
* per-component call counts, mean and p95 latency, and CPU.

Optimisations in place:

| Optimisation | Status |
|---|---|
| cached routes | path cache (Phase 6), dropped after a blocked recovery |
| event-driven planning | frontier candidates rebuilt on events, bounded per selection |
| reused inference sessions | one ONNX session per process |
| bounded tracking history | at most 100 tracks; stale ones discarded |
| minimal frame retention | only the latest observation is kept |
| reduced debug rendering | headless by default; the dashboard is optional |
| batched database writes | route statistics flushed once per episode |
| adaptive perception rate | vision profiles and unchanged-frame skipping (Phase 6) |
| adaptive image resolution | **not implemented** as per-state switching; the resolution is a deployment setting (benchmarks use 320×180), measured below |
| region-of-interest processing | **not implemented**: the full-frame segmentation also feeds the empty-spawn votes; see section 13 |

**Results** (`benchmark_results/phase8/resource_profile.json`, produced by the Phase 8
audit's `--profile-resources` on the final code):

* 3 held-out maps per configuration (`p7_open_test_000`, `p7_indoor_test_000`,
  `p7_mixed_test_000`), 200 simulated seconds each.
* Active exploration, deterministic policy, one process, 4 vCPU, no GPU.

| configuration | episodes | sim s | wall s | pipeline CPU s / sim s (share of one core in real time) | pipeline CPU ms / analysed frame | frames analysed | bridge client CPU s | simulator CPU s | RSS first / last / max (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 320x180_5fps | 3 | 600.0 | 36.7 | 0.0097 | 1.938 | 3000 | 1.95 | 27.67 | 247.0 / 260.0 / 260.0 |
| 480x270_5fps | 3 | 600.0 | 45.2 | 0.0168 | 3.370 | 3000 | 2.32 | 31.38 | 265.8 / 269.9 / 269.9 |
| 640x360_10fps | 3 | 600.0 | 86.1 | 0.0348 | 4.636 | 4500 | 3.42 | 59.79 | 288.8 / 307.9 / 307.9 |
| 320x180_adaptive | 3 | 600.0 | 32.0 | 0.0092 | 2.147 | 2580 | 2.01 | 22.99 | 313.8 / 314.1 / 314.1 |

* **The Python decision pipeline is cheap.** The agent's own thread uses about 1 % of one core per simulated second at 320×180 and 5 analysed frames per second (1.9 ms per analysed frame), 1.7 % at 480×270 and 3.5 % at 640×360 and 10 fps (4.6 ms per frame).
* **Simulator CPU** is the Python reference arena (physics and software renderer). It is
  the larger share of process CPU and is **not** a measurement of Unreal's cost.
* **Adaptive profile:** it analysed 2,580 frames instead of 3,000 and used 5 % less pipeline CPU per simulated second (0.0092 vs 0.0097). Its CPU per analysed frame is higher (2.15 ms) only because the work done on every tick is divided by fewer frames. In these three episodes it opened fewer chests (1.1 vs 1.5 per minute on average). Three episodes cannot measure that cost, so the saving should not be assumed to be free; the Phase 6 benchmark measured it at −17 % throughput (Phase 6 document, section 12.1).
* **Resolution.** On the same three maps, 480×270 and 640×360 opened about as many chests
  as 320×180 (within 0.2 chests per minute; three episodes cannot separate them). The
  labelled-frame recall gain at higher resolution (Phase 8 audit, section F) does not show
  up as throughput here, because the agent re-observes and approaches what it has seen.
* **Memory:** the RSS values are those of the audit process after its earlier sections,
  not an agent's own footprint. An endurance agent stays at 97–110 MB, and an acceptance
  worker peaked at 146.5 MB.
* **Not measured:** GPU load, Unreal frame time and Unreal memory (no GPU, no Unreal).

## 11. Runtime packaging

* **Python** 3.11–3.13, tested on 3.11.18, 3.12.3 and 3.13.18.
* **Pinned dependencies** in `requirements-lock.txt`: numpy 2.4.6 (Python 3.11) / 2.5.4
  (3.12+), opencv-python-headless 5.0.0.93, onnxruntime 1.32.0; onnx 1.23.2 and pytest 9.2.0
  for export and tests.
* **Versioned configuration.** `config_version: 7`. Older files are read with defaults for the
  new settings. A newer version is refused.
* **Model version tracking.** The model id (`bc-…`) and backend are reported at startup and in
  every benchmark row's policy fields. The agent version is `AGENT_VERSION`. The exploration
  memory records the agent and generator versions.
* **Structured logging.** `--log-format json` writes one JSON object per line to stderr.
* **Graceful shutdown.** SIGTERM is handled like Ctrl+C. The episode's `finally` blocks close
  the bridge connection, stop the reference server, restore any fault-injection patches and
  flush the recorder. Exit code 130.

**Clean environment:**

```bash
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m chest_hunter init-db
python -m chest_hunter check-runtime
```

**Verified.**

* A fresh clone was installed into new Python 3.12.3 and 3.11.18 virtual environments with
  exactly these commands. The pinned install took 12 s; Python 3.11 resolves numpy 2.4.6.
* In the 3.12 environment, `check-runtime` reported 16 checks, 0 failed and 1 warning (no
  exploration memory in that clone). A `simulate-3d --exploration active --map procedural`
  episode then ran.
* The **full test suite passed in both clean environments** at commit `fc5877d`: 772 tests on
  3.12 under Xvfb (none skipped), and 772 on 3.11 (7 GUI tests skipped without a display).

A packaged Unreal build (not the Editor) is enough for Unreal runs. Start it with the
research server enabled, then run `check-runtime --backend unreal` and the usual commands
with `--backend unreal`.

## 12. Startup procedure

`python -m chest_hunter check-runtime [--backend unreal] [--json]` checks:

* the Python version and dependency versions against the lock file;
* the configuration version;
* the SQLite schema (opened read-only);
* the model format, input size and backend;
* the observation sizes;
* the exploration-memory version;
* the bridge protocol and environment version (the reference arena is started for the check;
  `--backend unreal` connects with a 3 s timeout);
* that the output directories are writable.

Every failure or warning comes with a remedy. The exit code is non-zero if a check fails.

## 13. Known limitations

* **Unreal.** Not run (section 14).
* **Appearance shift.** Chest colours outside the trained range defeat the colour
  segmentation. Detection precision and recall fall sharply on the `appearance` and
  `combined` categories (section 15), for the Phase 6 agent as well.
* **ROI processing and per-state resolution.** Not implemented. At 320×180 the whole
  perception pass is a few milliseconds, so the expected gain is small.
* **Coverage within the benchmark budget.** Coverage within 200 simulated seconds is far
  below the 85 % engineering target on 90 m maps. The acceptance run (section 16) uses longer
  episodes.
* **Bandit evidence.** The bandit has 40 training episodes per category (5 arms). Its choice
  is only as reliable as that evidence (section 7).

## 14. Unreal integration status

* **Unchanged, verified against the reference server.** The bridge protocol is unchanged
  from Phase 4 (version 1). Every Phase 7 feature uses existing requests: `STEP`,
  `NAV_QUERY`, `NAV_RAYCAST`, evaluation labels and the oracle state for scoring.
* **New, not in Unreal.** The reference server now reports `environment_version` in `HELLO`.
  The Unreal plug-in does not yet; `check-runtime` reports that as a warning.
* **Not run in Unreal:**
  * Phase 7 maps can be exported with `export-arena-3d --map MAP_ID` (same layout format as
    Phase 5). This was **not possible before Phase 8**: the command exported only
    `research_arena_3d`, contrary to an earlier version of this document (audit finding F2).
    The acceptance map and one held-out map per family are now in
    `Content/ResearchData`. Their loading in the Unreal project has **not** been run.
  * Lighting and appearance variants are applied by the reference renderer. The Unreal arena
    builder reads `lighting`, `chest_value` and `chest_hue_shift` but **not**
    `surface_value`, so the `appearance` variant is only partly reproducible in Unreal
    (audit finding F3). None of these parameters has been verified in Unreal.
  * Dynamic obstacles (fault injection) are a reference-server feature. The Unreal research
    server has no `ADD_OBSTACLE` request, so that fault is reference-only.
  * Frame timing, GPU load and the Unreal graphics workload were **not measured** (no Unreal,
    no GPU).

## 15. Benchmark results

All of these were measured on the Python reference arena, not Unreal.

### 15.1 Generalisation (`benchmark_results/phase7/`)

**Run.** `benchmark-generalization --episodes 100 --held-out-maps --workers 4`.

* 2 configurations × 8 categories × 100 episodes = **1,600 episodes. 0 errors, 0 failed
  episodes**, 0 over-reported openings.
* 6000 ticks (200 simulated s) per episode, 320×180, evaluation seed 900042.
* 7,280 s of wall clock on 4 workers.
* **Code version.** The agent code was that of commit `e1124bd`, i.e. *before* the Phase 8
  fixes F11/F14: openings whose lid was not seen were not credited and cost a few seconds of
  re-alignment. Throughput is scored on ground truth, so that defect affects time spent, not
  what is counted.

Throughput is ground-truth chests opened per simulated minute (mean, 95 % CI).

**Generalisation gap.** G = (E_familiar − E_category) / E_familiar, with a bootstrap 95 %
interval (pairs resampled together where the category is paired):

| Category | Phase 7 throughput | Phase 7 G | Phase 6 (spawn points given) | Phase 6 G |
|---|---|---|---|---|
| familiar layouts, new seeds | 1.58 [1.43, 1.73] | – | 3.35 [3.19, 3.50] | – |
| new obstacle arrangements (paired) | 1.59 | −0.01 [−0.10, +0.07] | 3.26 | +0.03 [−0.03, +0.08] |
| unseen buildings / rooms | 1.50 | +0.05 [−0.08, +0.17] | 3.36 | −0.01 [−0.08, +0.06] |
| different lighting (paired) | 1.40 | **+0.11 [+0.04, +0.18]** | 2.91 | **+0.13 [+0.08, +0.18]** |
| different chest appearance (paired) | 1.13 | **+0.28 [+0.21, +0.36]** | 2.20 | **+0.34 [+0.27, +0.42]** |
| different camera start (paired) | 1.55 | +0.02 [−0.06, +0.09] | 3.39 | −0.01 [−0.06, +0.04] |
| more complex elevations (paired) | 1.65 | −0.05 [−0.12, +0.03] | 3.32 | +0.01 [−0.04, +0.06] |
| combined shifts | 0.99 | **+0.37 [+0.25, +0.49]** | 1.70 | **+0.49 [+0.41, +0.57]** |

Detection and navigation (Phase 7 agent; per-frame evaluation labels, pooled):

| Category | Precision | Recall | Chest-distance error | Pose error | Nav. success | Coverage |
|---|---|---|---|---|---|---|
| familiar | 88 % | 89 % | 0.74 m | 0.03 m | 0.98 | 0.51 |
| unseen layout | 94 % | 88 % | 0.77 m | 0.03 m | 0.98 | 0.50 |
| lighting | 94 % | 80 % | 0.80 m | 0.03 m | 0.97 | 0.51 |
| appearance | **58 %** | **57 %** | 0.86 m | 0.04 m | 0.98 | 0.49 |
| combined | **60 %** | **46 %** | 0.93 m | 0.03 m | 0.96 | 0.45 |

What the numbers say:

* **The agent generalises to new layouts, obstacle arrangements, start poses and extra
  elevations.** No significant gap appears for either agent.
* **Lighting outside the trained range costs about 11–13 %**, through recall.
* **Chest colours outside the trained range cost about 30 %** for both agents, because the
  colour segmentation no longer recognises the chest. This is the main generalisation
  weakness, and it is in perception, not exploration.
* **Phase 7 throughput is about 1.7 chests/min below Phase 6** on paired episodes,
  significant in every category. The Phase 6 agent is told where chests can spawn; the
  Phase 7 agent must find them, and it covers about half of the reachable area in 200 s.
* **Indoor complexes are the hardest family for exploration**: 0.79 (familiar) and 1.04
  (unseen) against 3.1 with spawn points. The best families are complex (1.93 / 1.81) and
  open (1.78 / 1.76).
* **Exploration memory.** The trained memory chose `max_information` for open, indoor and
  mixed maps and `risk_adjusted` for complex maps (600 / 200 episodes).
* **Cost.** Frontier selection took 17.5 ms on average. The agent's own CPU was 1.6 % of one
  core per simulated second, against 1.5 % for the Phase 6 agent.

### 15.2 Robustness under injected failures (`benchmark_results/robustness/`)

**Run.** `benchmark-robustness --episodes 100 --failure-injection --workers 4`.

* Configuration `phase7_active` on held-out layouts, 4500 ticks (150 simulated s).
* Ten fault kinds in rotation, 10 episodes each. Every faulted episode has a fault-free
  twin on the same map and seed (paired baseline).
* 200 episodes, 0 errors, 973 s of wall clock (shared with another benchmark).
* **Code version:** commit `3ad12f3`, with the Phase 8 fixes F11, F12, F14, F17 and
  F19–F21. F22 and F23 came later. F23 concerns only the audit tool. F22 can change an episode only
  from a second off-mesh manoeuvre in a row without a walked trail; no episode of this run
  failed, so none was stuck that way.
* **Earlier runs:**
  * the run before F17 is kept in `benchmark_results/robustness_preF17/`;
  * the run between F17 and F19/F20 (commit `581a527`) had one over-reported opening, now
    fixed (F20);
  * the runs before F21 lost three map pairs to episodes that ended after the first scan.

**Definitions.**

* **Applied**: the fault actually happened. Some need a trigger: for example,
  `interaction_failure` needs an interaction attempt within the episode.
* **Detected**: the fault's own signal fired within 15 s, such as an invalid observation
  for a corrupted frame or a localisation jump for drift. `n/a` means the fault has no
  agent-side signal by design: a phantom or missed detection looks like an ordinary
  observation.
* **Recovered**: within 30 s the agent was making progress again: moving, opening, or
  exploring new area.

| Fault | Applied | Detected | Recovered | Clean end | Failed | Throughput | Paired diff vs twin |
|---|---|---|---|---|---|---|---|
| frame loss | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 0 | 1.80 | +0.24 [−0.10, +0.58] |
| corrupted frame | 10 / 10 | 10 / 10 | 9 / 10 | 10 / 10 | 0 | 1.40 | +0.20 [−0.11, +0.51] |
| false detection | 10 / 10 | n/a | 10 / 10 | 10 / 10 | 0 | 1.92 | +0.04 [−0.21, +0.29] |
| missed detection | 10 / 10 | n/a | 7 / 10 | 10 / 10 | 0 | 1.88 | −0.16 [−0.52, +0.20] |
| localisation drift | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 0 | 1.88 | −0.20 [−0.61, +0.21] |
| unexpected obstacle | 10 / 10 | 7 / 10 | 9 / 10 | 10 / 10 | 0 | 1.56 | +0.04 [−0.17, +0.25] |
| interaction failure | 6 / 10 | 6 / 6 | 6 / 6 | 6 / 6 | 0 | 2.47 | 0.00 [−0.27, +0.27] |
| bridge disconnect | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 | 0 | 1.76 | −0.12 [−0.39, +0.15] |
| slow inference | 8 / 10 | 8 / 8 | 4 / 8 | 8 / 8 | 0 | 1.90 | +0.10 [−0.06, +0.26] |
| unreachable target | 10 / 10 | 7 / 10 | 10 / 10 | 10 / 10 | 0 | 1.72 | −0.12 [−0.31, +0.07] |

**Overall.**

* 94 faults applied: 85 recovered (90 %), 94 clean terminations (100 %), 0 failed episodes.
* 0 over-reported openings in 200 episodes.
* Fault-free twins: 1.79 [1.61, 1.96] chests per simulated minute, 0 failed.
* **No fault kind costs measurable throughput.** Every paired interval includes 0, though
  with 6–10 pairs per kind only large effects could be detected.
* **The weak spots** (from the per-episode fault records):
  * **Slow inference:** 4 of 8 not recovered within 30 s. In all four the slow inferences
    were detected: the selector timed out, the supervisor reported a model failure and
    switched policy. The agent then abandoned one to three targets and showed no measured
    progress inside the window. The cause beyond that was not investigated.
  * **Unexpected obstacles:** 3 of 10 not detected; no navigation-failure signal fired. Two
    of those episodes recovered anyway.
  * **Unreachable targets:** 3 of 10 not detected. The blocked path queries happened (3–6
    per episode) without producing the unreachable signal; all three episodes recovered.
  * **Missed detections:** 3 of 10 not recovered within the window.
  * **Corrupted frames:** 1 of 10 not recovered. All 46 corrupted frames were detected and
    dropped, but the agent made no measured progress within the window.
* **Residual early ending (F21, mid-episode).** One map pair (`p7_indoor_test_018`, episode
  73) still ends as "exploration complete" at 77 s with 12.5 % coverage. The agent had
  explored a region whose exits no ray crossed, and the denser fan at its current position
  found none either.

**The F17 fix.** Same 100 twins before and after:

| | Before F17 | After F17 |
|---|---|---|
| Loop detections (fault-free twins) | 489 in 92 episodes | 152 in 61 episodes |
| Supervisor interventions (fault-free twins) | 380 | 143 |
| Loop detections (faulted episodes) | 489 | 159 |
| Throughput of the twins | 1.78 [1.61, 1.96] | 1.75 [1.57, 1.92] |
| Faults recovered | 85 / 91 | 84 / 91 |

Both columns were measured before F19, F20 and F21. In the final run (`3ad12f3`) the twins
show 159 loop detections in 62 episodes and 149 interventions.

### 15.3 Exploration strategies (`benchmark_results/phase7/strategies/`)

**Run.** `benchmark-exploration --episodes 20 --workers 4`: the five fixed strategies and the
memory-selected one, on the same 20 held-out layouts (paired). Commit `fc5877d`, before F17.

| Strategy | Throughput | Coverage | Travel (m) | Paired diff vs memory-selected |
|---|---|---|---|---|
| nearest frontier | 1.58 [1.19, 1.96] | 0.38 | 363 | +0.02 [−0.17, +0.20] |
| maximum information | 1.58 [1.26, 1.89] | 0.47 | 480 | +0.02 [−0.06, +0.09] |
| chest prior | 1.49 [1.23, 1.74] | 0.45 | 483 | −0.08 [−0.32, +0.17] |
| cluster | 1.29 [0.98, 1.60] | 0.35 | 440 | **−0.27 [−0.52, −0.02]** |
| risk adjusted | 1.49 [1.14, 1.83] | 0.47 | 484 | −0.08 [−0.25, +0.10] |
| memory-selected (bandit) | 1.56 [1.24, 1.88] | 0.47 | 479 | – |

The memory chose maximum information on 15 maps and risk adjusted on 5 (complex family). It
is **not better than the best fixed strategy.** Only `cluster` is distinguishably worse.
Nearest frontier reaches the same throughput with a quarter less travel and lower coverage.
In this arena the choice of frontier strategy hardly matters for throughput. The bandit adds
complexity without a measured gain (Phase 8 audit, section M).

## 16. End-to-end acceptance test

**Run.** `acceptance-phase7 --episodes 10 --workers 4`.

* **Map.** `p7_complex_test_000_unfamiliar`, never used for training. It is held out on
  every axis: unseen layout with rooms, outdoor areas and several elevations, plus
  unfamiliar lighting and start pose.
* **Episodes.** 10 episodes of 600 simulated seconds (18,000 ticks) for each group, clean
  and with one injected fault per episode (the ten fault kinds in turn).
* **Comparison.** The Phase 6 agent, given the spawn points, on the same episodes.
* **Code version.** Commit `3ad12f3`, with the Phase 8 fixes F11–F21 (F18 is open). F22 and F23 came later. F23 concerns only the audit tool. F22 can change an episode only
  from a second off-mesh manoeuvre in a row without a walked trail; no episode of this run
  failed, so none was stuck that way.
  The run before F17 is kept in `benchmark_results/phase7/acceptance_preF17/`.
* 41 episodes in 813 s of wall clock (shared with another benchmark), 0 errors, 0
  over-reported openings.

| Group | Throughput | Reachable opened | Interaction success | Coverage | Outcomes | Loops detected |
|---|---|---|---|---|---|---|
| Phase 7 active, clean | 1.16 [1.01, 1.31] | 57.4 % | 95.1 % | 0.75 [0.71, 0.79] | 10 tick limit | 158 |
| Phase 7 active, faults | 1.09 [0.89, 1.29] | 54.0 % | 91.6 % | 0.71 [0.65, 0.77] | 10 tick limit | 224 |
| Phase 6 with spawn points, clean | 3.14 [2.65, 3.63] | 60.9 % | 96.1 % | n/a | 10 completed | 5 |
| Phase 6 with spawn points, faults | 2.91 [2.15, 3.66] | 56.9 % | 95.0 % | n/a | 10 completed | 23 |

Engineering targets (Phase 7 agent, fault-free episodes):

| Target | Measured | Met |
|---|---|---|
| Interaction success ≥ 90 % | 95.1 % | yes |
| Coverage ≥ 85 % | 74.5 % | **no** |
| No failed or aborted episode | 0 / 20 (before F17: 1 aborted on the intervention budget) | yes |
| Exact replay (episode 0 played twice) | 10 vs 11 chests in this run and in `ea9216d`; identical in `c549dce` | **not guaranteed**: finding F18 |
| Peak memory of a worker | 146.5 MB | – |

Other measurements:

* **Detection on this held-out map:** precision 0.99, recall 0.83. Chest-distance error is
  1.7 m, pose error 0.07 m.
* **Navigation success:** 0.96.
* **Throughput against the Phase 6 agent (paired):**
  * −1.98 [−2.44, −1.52] chests per minute clean (0 better, 10 worse);
  * −1.82 [−2.52, −1.12] with faults.
* **Interpretation.** The agent finds and opens about 57 % of the reachable chests in
  10 minutes on a map it has never seen, without spawn-point knowledge, and survives every
  injected fault kind. It is about a third as productive as an agent that is told where
  chests can be.

**Exact replay (F18).** Episode 0 was played twice with the same seed, map and code:

| Run | Plays of episode 0 | CPU load |
|---|---|---|
| `ea9216d` | 10 and 11 chests | normal |
| `c549dce` | identical | normal |
| final run | 10 and 11 chests | heavy (shared with the Phase 6 benchmark) |

Two decisions depend on wall-clock time:

* the frontier-selection budget (20 ms; a selection takes 15 ms on average, and 6–10
  overruns occur per 600 s episode);
* the hybrid selector's 20 ms inference timeout.

The simulation is lock-step, so latency does not move simulated time. It can still change
which frontier or which policy is used.

What re-running showed:

* Re-running the episode in fresh processes under load gave six identical pairs.
* With a 3 ms frontier budget the episode gives 12 chests instead of 11, so the budget does
  change outcomes.
* Which of the two limits caused the single mismatch was not isolated.

Before F17 the same check passed in the acceptance run. The fix belongs in the next phase: work-based limits for
lock-step research runs, with the wall-clock limits kept for real-time use. It changes every
benchmark's semantics, so it was not made under the running Phase 8 queue.

## 17. Recommendations for the next phase

In priority order. The Phase 8 audit gives the evidence (`docs/phase8_full_system_audit.md`,
section O).

1. **Run the Unreal project.** Nothing in this document has been measured in Unreal.
   Building the module, creating the map and running the acceptance map over the C++ bridge
   is the first step. The results above may not transfer.
2. **Perception robustness to chest appearance.** This is the largest measured
   generalisation gap (G ≈ +0.3). Add a hue-independent cue before anything heavier, and
   measure its latency at 320×180.
3. **Reproducible research runs (F18).** Replace the wall-clock frontier budget and
   inference timeout with work-based limits when the simulation is lock-step. Keep the
   wall-clock limits for real-time use.
4. **Exploration completeness (F21 residual).** When the frontier search runs dry
   mid-episode, also cast denser fans from earlier fan origins before declaring the map
   explored.
5. **Coverage.** 74.5 % against the 85 % target in 600 s. The explorer's frontier choice
   hardly matters (section 15.3); travel time does. Shorter scans and fewer revisits are
   the likely levers.
6. **Simplify.** The exploration memory and bandit add no measured gain over a fixed
   strategy. Keep one or two strategies.
