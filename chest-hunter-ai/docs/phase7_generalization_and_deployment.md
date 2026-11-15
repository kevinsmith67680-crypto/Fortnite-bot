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
| adaptive image resolution | **not implemented** as per-state switching; the resolution is a deployment setting (benchmarks use 320×180), measured in section 12 |
| region-of-interest processing | **not implemented**: the full-frame segmentation also feeds the empty-spawn votes; see section 13 |

RESULTS_PROFILE

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

RESULTS_CLEAN_ENV

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

RESULTS_BENCHMARK

## 16. End-to-end acceptance test

RESULTS_ACCEPTANCE

## 17. Recommendations for the next phase

RESULTS_RECOMMENDATIONS
