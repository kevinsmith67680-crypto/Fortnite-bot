# Phase 7 baseline audit

This audit records where the project stood when Phase 7 started (commit `054c8ab`, the end of
the Phase 6 implementation). It covers what the agent depended on, what it could not yet do on
unfamiliar environments, and what it measured on the Phase 7 maps before any Phase 7 change
to the agent. Every number here comes from a run on this machine: a 4-vCPU Xeon at 2.1 GHz,
16 GB RAM, no GPU. All runs used the Python reference arena, which mocks the Unreal research
server's protocol. **Unreal Engine is not installed, so nothing in this audit was measured in
Unreal.**

## 1. Method

* I read the code paths a Phase 6 episode goes through: `Runner3D`, `VisualAgent3D`,
  `WorldModel3D`, the planners, the supervisor, the bridge client and server, the
  configuration, and the CLI.
* I ran development episodes on the new procedural maps (section 10).
* For the formal baseline, the Phase 6 configuration (`phase6_priors` = the Phase 6 hybrid
  agent with the supervisor) also runs as a configuration of the Phase 7 generalisation
  benchmark. It plays the same episodes as the Phase 7 agent, so the two are paired.

## 2. Dependence on spawn-point coordinates

* `Runner3D._priors` builds the agent's targets from the database's chest table. It raises
  `ValidationError` ("no spawn points") for a map without them. **On an unseen map with no
  spawn-point table, the Phase 6 agent cannot start.**
* `WorldModel3D` turns tracks into targets only by associating them with a spawn point within
  1.6 m. A chest away from every spawn point is tracked but never becomes a goal.
* The Phase 5 adaptive planner and the Phase 6 greedy and short-horizon planners all score
  *spawn points* (probability, travel time, cluster). None of them reasons about unexplored
  space.
* The scan planner turns a full circle at fixed steps. It has no notion of which directions
  were already seen.

Result: "exploration" before Phase 7 meant visiting known spawn points. Phase 7 needs
exploration without them (`exploration/active.py`, discovery mode in `WorldModel3D`).

## 3. Environments

* `research_arena_3d` is one hand-made layout.
* The Phase 5 procedural maps (`proc_<train|test>_NNN`) are one family. They vary buildings,
  plateaus, obstacles, start pose, lighting and chest appearance, but only *inside* one range
  that training also uses. They are validated with the navigation graph and split into
  training and test by disjoint seed ranges.
* Missing for Phase 7:
  * distinct environment families (open, indoor, mixed, complex);
  * distribution shifts *outside* the training ranges (lighting, chest appearance, start,
    obstacle arrangement, extra elevations, all combined);
  * a generator version stored with each map;
  * a rule that keeps shifted variants of training layouts out of training.

## 4. Uncertainty

* Tracks carry a detector confidence, a hit count and a position sigma. Phase 6 calibrated
  the *learned policy's* reliability (ECE 0.134 raw, 0.060 calibrated on the synthetic test
  split).
* The detector's template-matching score is a similarity, not a probability. It was used as
  a confidence without calibration, which Phase 7 keeps as-is. Phase 7 adds explicit
  thresholds (approach / inspect / ignore) on confidence, hits and sigma, and validates them
  in the benchmark. It does not reinterpret the score as a probability.
* There was no localisation confidence: an odometry jump was not detected.

## 5. Memory

* Episode memory: the tracker (bounded by `performance.max_tracking_objects`) and the
  world model's spawn-point statuses.
* Persistent memory: route statistics and experience partitions (Phase 6, migration 5/6).
  These are keyed by layout hash, so by design nothing transfers to an unseen layout.
  Nothing was learned *per environment family*.

## 6. Robustness

* The supervisor (Phase 6) detects invalid observations, state timeouts, loops, lack of
  progress, stationarity and repeated interaction failures. Recovery is bounded by ladders,
  per-incident attempt limits and a per-episode intervention cap.
* The bridge client re-sends a request after a dropped connection.
* The policy selector falls back to the deterministic controller on inference time-outs,
  errors and low confidence.
* Missing:
  * a way to *inject* failures on purpose and measure detection and recovery per failure
    kind;
  * odometry-jump detection;
  * detection of corrupted frames (frames with blank bands passed as valid).

## 7. Resource use (Phase 6 measurements)

* At 640×360 the agent's own thread used about 25 % of the process CPU. The rest was the
  reference simulator's renderer in the same process.
* Server-side A\* median was 18–28 ms, p95 about 90 ms.
* Over 40 repeated episodes, RSS rose from 83.7 to 95.2 MB (4.4 MB per 100 episodes after
  warm-up). The gc object count stayed stable.

Phase 7 keeps these numbers as the reference and measures the Phase 7 pipeline the same way
(`profile-agent --report`).

## 8. Deployment readiness

* Dependencies were lower bounds only (`numpy>=1.26`, `opencv-python-headless>=4.8`). There
  was no lock file.
* The configuration file had no version.
* There were no startup checks: problems with the schema, the model, the bridge or the
  output directories surfaced mid-run.
* Logging was plain text only.
* Ctrl+C was handled, but SIGTERM (container stop, `kill`) ended the process without the
  runner's `finally` blocks.

## 9. Defects found during Phase 7 development

Each of these was found by a run, not by reading. Each is fixed in the commits listed in
`docs/phase7_generalization_and_deployment.md`.

| Defect | Effect | Fix |
|---|---|---|
| A track of an already opened chest, seen again from another side, became a new discovered spawn point | the agent tried to open opened chests (10 of 20 recoveries in a 150 s episode) | discovery links a track to any existing spawn point within the association radius; a discovered spawn point moves to where the chest was opened |
| Recovery budget `(max_attempts + 1) * (spawn points + 5)` with zero spawn points | in exploration mode the budget was 20 for the whole episode; one extra fault ended episodes as `failed` | the budget grows with the discovered spawn points |
| The navigator cached a path the agent had just failed to follow | after a new obstacle the agent re-used the blocked path until the budget ran out | a blocked recovery clears the path cache |
| Recovery resolutions matched by track id | track ids change when a chest is re-acquired, so the recovery success rate read close to zero | resolutions are matched by spawn location |
| `nearest_frontier` ranked like `max_information`, and the chest term was about 1 % of the score | the strategy bandit compared near-identical arms | the score weights were rescaled (section 4 of the design document) |
| The robustness injector placed obstacles on the agent itself, or never applied some faults | invalid robustness results | the obstacle is placed 3 m ahead on the path, clear of chests; the interaction and unreachable faults wait for the next interaction / path query |

## 10. Baseline on the Phase 7 maps

**Development runs.** These are single episodes on `p7_mixed_train_000`, 9000 ticks
(300 simulated seconds), seed 5. They were taken while the Phase 7 explorer was being built,
so they show the starting point, not a benchmark:

| Agent | Chests opened | True coverage of the reachable area |
|---|---|---|
| Phase 6 agent, given the map's spawn points | 10 (in 168 s) | n/a |
| first active explorer | 5 | 0.445 |
| active explorer after the look-while-travelling and approach-threshold changes | 8 | 0.62 |

**Paired baseline.** The paired baseline is the `phase6_priors` configuration of the
generalisation benchmark (`benchmark_results/phase7/`). It is the Phase 6 agent, given the
map's spawn points. It is reported in the benchmark section of
`docs/phase7_generalization_and_deployment.md`, next to the Phase 7 agent on the same
episodes.

The Phase 6 agent is given information the Phase 7 agent does not get: where every chest can
appear. A higher Phase 6 throughput therefore does not mean the Phase 7 agent is worse at the
same task.
