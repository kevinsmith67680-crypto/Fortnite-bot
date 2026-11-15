# Phase 6: autonomous reliability, planning and computational optimisation

Phase 6 adds an episode supervisor, formal state specifications, short-horizon planning,
adaptive computation, learned-policy monitoring, partitioned experience, long-duration
(endurance) runs with checkpoints, resource monitoring, a component profiler, an operations
panel and a paired benchmark. The baseline it starts from is recorded in
`docs/phase6_baseline_audit.md`.

## 0. What is verified and what is not

| Part | Status |
|---|---|
| Supervisor, state specifications, loop detection, bounded escalation | **Verified** by unit tests (fake agent) and real episodes (`tests/test_supervisor.py`) |
| Short-horizon and greedy planning | **Verified** (`tests/test_advanced_planner.py`); benchmarked (section 12) |
| Adaptive computation (vision profiles, unchanged-frame skipping, caches, bounded tracking) | **Verified** (`tests/test_adaptive_compute.py`); benchmarked |
| Learned-policy monitors and reliability calibration | **Verified** (`tests/test_reliability.py`, `tests/test_phase6_workflow.py`); calibration measured on held-out synthetic demonstrations |
| Experience partitions and per-episode experience records | **Verified** (schema migrations 5 and 6, filters, summaries) |
| Endurance runs, checkpoints, resume, interruption, reset failures | **Verified** (`tests/test_endurance.py`); long runs in section 12.2 |
| Resource monitor, profiler, operations panel | **Verified** (the panel under Xvfb) |
| All of the above against **Unreal Engine** | **Not verified.** Unreal is not installed (audit, section 5). Everything ran on the Python reference arena, which speaks the same bridge protocol. |
| GPU utilisation, Unreal frame time | **Not measured** (no GPU, no Unreal). The resource monitor reports them as unavailable. |

## 1. Architecture modifications

```text
REFERENCE ARENA / UNREAL RESEARCH SERVER  (one per run in endurance mode, reset per episode)
        |  bridge protocol (localhost only)
        v
Runner3D --------------------------------------------------------------+
  | observation  -> Supervisor.filter_step (frame/telemetry validation)  |
  | agent / PolicySelector.tick  -> Supervisor.validate_action           |
  |   VisualAgent3D (Phase 4 FSM, transition table enforced)             |
  |     perception (shared segmentation, vision profile, frame skip)     |
  |     AdvancedPlanner (greedy | short-horizon) or Phase 4/5 rule       |
  |   PolicySelector (deterministic | learned | hybrid + monitors)       |
  | Supervisor.after_step: detections -> bounded escalation              |
  | Profiler sections, ResourceMonitor samples, EventLog                 |
  +-> SQLite: sessions, observations, runs (+partition), route memory    |
      (+partition key), episode_experience                             |
EnduranceRunner: many episodes, checkpoints, resume, signals, OperationsControl/panel
```

New modules:

* `runtime/`: `supervisor`, `state_spec`, `events`, `profiler`, `checkpoint`,
  `configurations`, `operations`;
* `exploration/advanced_planner.py`, `exploration/experience.py`;
* `learning/reliability.py`;
* `analytics/resource_monitor.py`, `analytics/endurance.py`, `analytics/phase6_benchmark.py`;
* `agent/operations_panel.py`, `cli_phase6.py`.

The learning pipeline stays offline. No network weights change during an episode, and the
supervisor, planner and monitors never read simulator ground truth: an import audit test checks this.

## 2. Agent supervisor

`runtime/supervisor.py` watches one episode from the runner's side of the loop. The runner
calls it three times per decision step: after the command is chosen, on the simulator's
reply, and after the agent has processed that reply.

| Condition | Detection (simulated time unless noted) | Default |
|---|---|---|
| `state_timeout` | time in a state beyond its specification (counted only while the state machine drives) | per state, section 3 |
| `stationary` | movement requested in ≥ 80 % of the steps of a window, displacement below a threshold | 6 s, 0.3 m |
| `transition_loop` | the same cycle of 2–4 states repeated without progress | window 12, 3 repetitions |
| `repeated_target` | one spawn point planned for again and again without being reached | 8 plans |
| `interaction_failures` | failed interactions at one spawn point (alignment failures excluded: walking round is the normal strategy) | 3 |
| `no_progress` | nothing opened, resolved, reached or newly detected | 90 s |
| `invalid_transition`, `entry_violation` | a transition the table refuses; a state entered without its entry conditions | - |
| `invalid_action`, `action_not_permitted` | non-finite or out-of-range commands; commands the producing state may not issue (fixed in place) | - |
| `invalid_observation` | non-finite or backward telemetry, a missing requested frame, a wrong frame size, a blank frame (dropped) | streak of 5 |
| `model_failure`, `inference_latency` | invalid outputs, errors and time-outs of the learned model; p95 latency over the last 50 inferences (wall clock) | 5 failures; 20 ms |
| `communication_failure` | the bridge failed after the client's own bounded re-send | - |

Every detection becomes a structured event (`runtime/events.py`). The log is bounded (the
newest 500 events, with exact counts per kind) and can be written as JSON lines. Endurance
runs write `events.jsonl`.

**Interventions** and their escalation ladders:

| Incident | Ladder (each rung at most once per incident) |
|---|---|
| state_timeout | replan → abandon target → unstick → relocalise → redirect |
| stationary | replan → unstick → abandon target → relocalise → deterministic fallback |
| transition_loop | abandon target → unstick → relocalise → deterministic fallback |
| repeated_target | abandon target |
| interaction_failures | mark the target failed |
| no_progress | replan → abandon target → unstick → relocalise → deterministic fallback → **abort** |
| invalid_transition, entry_violation | redirect → relocalise → **abort** |
| invalid_observations | relocalise → **abort** |
| model_failure, inference_latency | deterministic fallback |
| communication_failure | **abort** (the episode ends cleanly; the runner does not raise) |

The interventions do the following:

* **replan:** recalculate the route, with the navigation cache cleared.
* **abandon target:** the spawn point is not selected again in this episode.
* **unstick:** a one-second move in a direction the agent's own back-off does not use.
* **relocalise:** drop stale tracks and look around.
* **redirect:** leave the state for EXPLORING.
* **deterministic fallback:** switch the policy selector to the deterministic controller.
* **abort:** end the episode as `failed`, with the reason recorded.

Limits:

* A rung that does not apply in the current situation is skipped.
* An incident never gets the same intervention twice.
* At most `max_recovery_attempts` (3) interventions per incident. After that the incident is
  exhausted, unless its ladder ends in abort.
* At most `max_interventions` (30) per episode; then the episode is aborted.
* After an intervention, the same incident is ignored for a 3 s cooldown.
* The no-progress ladder is the backstop for everything else. That keeps the supervisor from
  ending episodes the agent would finish on its own (an early version did). In the benchmark
  the supervised configuration aborted no episode (section 12.1).

**Monitor mode** (`supervisor.mode = "monitor"`) detects and logs, with the action it would
have taken, but changes nothing. With the supervisor in monitor mode, Phase 5 benchmark
episodes reproduce their published rows exactly (tested). The baseline audit's false-alarm
analysis used this mode.

## 3. State-transition validation

The transition table (`controller_3d.TRANSITIONS_3D`) was already enforced. Phase 6 adds:

* `StateMachine.reject_invalid`: a transition the table does not allow is refused (state
  unchanged), counted and reported to the supervisor, which redirects. Without a supervisor
  the old behaviour (an exception, i.e. a programming error) is kept.
* `StateMachine.on_transition`: every accepted transition, including instant ones, reaches
  the supervisor. A bounded window of recent states feeds the loop detector.
* `runtime/state_spec.py`: for every state, its Phase 6 category, entry conditions (checked
  after entry), permitted commands, exit conditions, maximum duration, recovery transition
  and terminal failure conditions.

The agent keeps its Phase 4 state names, which are stored in session records. They map to
the prompt's categories:

| Agent state | Category | Entry check | May move | May interact | Max s |
|---|---|---|---|---|---|
| INITIALIZING | INITIALIZING | - | no | no | - |
| OBSERVING | OBSERVING | - | yes | no | 10 |
| SCANNING | OBSERVING | - | yes | no | 40 |
| EXPLORING | EXPLORING | - | yes | no | instant |
| TARGET_DETECTED | SELECTING_TARGET | - | yes | no | instant |
| PLANNING_ROUTE | PLANNING_ROUTE | goal set | yes | no | instant |
| NAVIGATING | NAVIGATING | goal and route | yes | no | 45 |
| APPROACHING | NAVIGATING | chest goal | yes | no | 20 |
| ALIGNING | ALIGNING | chest goal | yes | no | 8 |
| INTERACTING | INTERACTING | chest goal | **no** | yes | 6 |
| VERIFYING_OPEN | VERIFYING | goal set | yes | no | 10 |
| RECOVERING | RECOVERING | - | yes | no | instant |
| COMPLETED, FAILED | terminal | - | no | no | - |

The maximum durations are at least twice the longest visit measured in 18 baseline episodes
(audit, section 9). "May move" is permissive where the agent's own back-off manoeuvres can
overlap a state entry.

A command is checked against both states involved: the one it was produced in and the one
current after the tick. The last step of an interaction hold is produced in INTERACTING and
sent after the switch to VERIFYING_OPEN; removing its interact flag would shorten the hold.

## 4. Recovery policies

Recovery has three layers, each bounded:

1. **The agent's own handlers** (Phase 4/5): approach time-outs, side switches, look-backs,
   the per-target attempt limit, the episode recovery budget, and the classified strategies
   with per-class budgets (Phase 5, `smart_recovery`).
2. **The supervisor's ladders** (section 2), with per-incident and per-episode limits.
3. **The endurance runner**, between episodes. The environment must answer a ping and
   accept a reset; the reference server is restarted if its thread died. After
   `max_reset_attempts` (3) failed resets the run stops with that reason. It never retries
   without limit.

Recovery scenarios and what handles them:

| Scenario | Handled by | Test |
|---|---|---|
| Blocked route | follower stuck → RECOVERING (Phase 4); supervisor replan/unstick | `AgentInterventions.test_replan_from_navigation` |
| Lost target | Phase 5 bounded rescan; supervisor relocalise | `test_relocalise_from_navigation` |
| Invalid observation | frame dropped; streak → relocalise → abort | `test_a_streak_of_invalid_frames_escalates` |
| Model failure | selector fallback and disable; supervisor deterministic fallback | `ModelMonitoring` |
| Interaction failure | Phase 4 retry/abandon; supervisor marks the target failed | `test_interaction_failures_mark_the_target_failed` |
| Lost connection | episode ends as failed, no exception; endurance reset | `test_lost_connection_ends_the_episode_without_an_exception` |

## 5. Advanced planning

`exploration/advanced_planner.py` scores each candidate *i* (an unresolved spawn point, or a
chest the agent sees):

    S_i = (p_i q_i + λ I_i) / (T_i + ε) − μ R_i

The terms:

* **p** is the Bayesian spawn estimate; 1 for a visible chest.
* **q** is the route-memory reliability; 0.9 with no history.
* **I** is the information gain.
* **T** is the estimated time: path length / speed × the learned travel factor, plus 4 s for
  the approach and 2 s × p for the interaction.
* **R** is 1 − q, plus this episode's failed plans, unsuccessful inspections and failed
  attempts at that spawn point.

Defaults: λ = 0.15, μ = 0.005, ε = 1 s.

* **greedy:** the best single score.
* **short_horizon:** beam search (width 5) over sequences of up to 3 targets, drawn from at
  most 20 candidates. A sequence is worth (Σ p q + λ Σ I) / (Σ T + ε) − μ Σ R, so a spawn
  point at the edge of a cluster beats an isolated one at the same distance (tested).
  * Only the first leg uses a real path length: the NavMesh query the agent already makes for
    its 4 best candidates. Later legs use straight distance × 1.25, so planning adds no path
    queries.
  * Work is bounded: per decision at most `candidates + beam × candidates` sequence
    evaluations per extra level.
  * If a level takes longer than 15 ms (wall clock), the best sequence so far is used and the
    overrun counted. A test injects a clock to cover this.
* **Decision points only:** planning runs when the agent needs a target (no goal, a new chest
  seen, a route finished or failed), never per frame.
* **Ties** go to the lower spawn-point id (tested over all orderings).
* Opened, empty, unreachable and abandoned spawn points are never candidates.

Measured planner latency is in section 12.

## 6. Adaptive visual processing

Profiles of perception rates (analyses per simulated second):

| Profile | Travelling far from the goal | Searching / approaching | Aligning / interacting / verifying | Otherwise |
|---|---|---|---|---|
| phase5 (default) | 3 | 8 | 10 | 2 |
| efficient | 2 | 6 | 10 | 1 |
| dense | 5 | 10 | 10 | 5 |

* **Completed:** no frames are requested once the episode is terminal.
* **Unchanged-frame skipping** (`skip_unchanged_frames`): no frame is requested when the
  agent and its camera are still, it commands no movement or look, and the last analysed frame
  showed the same pose. This never happens while aligning, interacting or verifying, and at
  least one frame is requested every second.
* **Shared segmentation:** the detector reuses the agent's colour segmentation. This applies
  to every configuration and changes no result: the Phase 5 rows reproduce, and a test
  compares detections both ways on rendered frames.
* **Not implemented: cropping to image regions.** The detector's full-frame output also feeds
  the world model's empty-spawn-point votes, and profiling did not show detection to be the
  bottleneck (section 11).

## 7. Inference optimisation

* **One model per process:** the ONNX Runtime session is created once and reused for every
  prediction (tested). Benchmark workers cache one `LearnedPolicy` per process.
* **Path queries:** NavMesh plans are cached (bounded at 256 entries); a supervisor replan
  clears the cache.
* **Bounded memory:** the 3D tracker deletes expired tracks and keeps at most
  `max_tracking_objects` (100) alive. Supervisor events, loop windows, profiler samples and
  selector switch logs are all bounded.
* **Database writes** are batched by the observation recorder, not done per frame. The profiler
  measured them as negligible (section 11).

## 8. Experience updates

Persisted per episode:

* chest observations (presence, openings), as before;
* route records: travel times, success, failure kind, recoveries, opening;
* new: one `episode_experience` row. It holds coverage, openings, interaction attempts and
  failures, routes started, navigation failures, recoveries by class with outcomes, and the
  supervisor summary.

Updated from them:

* Bayesian spawn estimates;
* travel-time factors and navigation reliability (route memory);
* the planner's per-target risk (route reliability plus this episode's failures).

**Partitions** (`runtime/configurations.partition`) separate experience by:

* **map**;
* **environment version:** layout plus environment settings;
* **agent version:** `AGENT_VERSION`, the control policy and the model version;
* **simulation configuration:** observation size, field of view, decision interval, planner,
  compute and recovery settings.

The partition is stored in each run's parameters, in every route record
(`route_memory.partition_key`, migration 5) and in every experience row (migration 6). The
filters work as follows:

* Route memory reads one partition. Pre-Phase 6 records (empty key) are used only on request;
  the Phase 6 planner accepts them, since the arena history predates partitions.
* Spawn estimates can be restricted to one environment version. Presence does not depend on
  the agent, so the agent version does not filter them.
* Persistent experience from held-out test maps (`proc_test_*`) is refused unless allowed
  explicitly.

## 9. Checkpoint format

`runtime/checkpoint.py`, one JSON document (`checkpoint.json`):

```json
{"format": "chest-hunter-checkpoint", "version": 1, "kind": "endurance",
 "created": "...", "updated": "...", "config": {"...": "..."}, "config_hash": "16 hex",
 "agent_version": "6.0.0", "policy_version": "bc-... | deterministic", "seed": 42,
 "seed_sequence": "episode i uses (seed, i) on map maps[i % len(maps)]",
 "last_completed_episode": 57, "completed": 58, "results_file": "episodes.jsonl",
 "aggregate": {"...": "..."}, "checksum": "sha256 of the document without this field"}
```

* **Atomic write:** temporary file, `fsync`, `os.replace`, directory `fsync`. The previous
  valid checkpoint is kept as `checkpoint.json.bak`.
* **Load:** checksum, format and version are verified. A corrupted file falls back to the
  backup; if neither is usable, `CheckpointError` is raised and nothing is deleted.
* **Resume requires the same configuration hash and agent version.** The differing fields are
  named. The episode count and time budget may change.
* **Results** are an append-only `episodes.jsonl`, one fsync-ed line per finished episode. On
  resume:
  * a torn last line is dropped (reported);
  * duplicates are removed (first record wins);
  * results must be a contiguous run of episodes.
* **Database consistency** is checked on resume:
  * duplicate successful records are refused;
  * sessions left `running` by a crash, and completed sessions without a result line, are
    marked `aborted`;
  * the episode is played again from its start. The research environment has no deterministic
    mid-episode restore, so nothing partial is restored.

## 10. Endurance-test methodology

`python -m chest_hunter endurance-test` (`analytics/endurance.py`):

* **One environment for the whole run:** one research server and one bridge connection. Each
  episode resets the environment and builds a new agent. A weak reference to the previous
  agent must be dead after garbage collection: no stale targets, tracks or interactions.
* **Maps** rotate (`--map`, repeatable). Episode *i* uses seed `(seed, i)`.
* **Estimates** carry over only with `--preserve-estimates`, in the run's own database and
  partition.
* **Per episode:**
  * outcome, ground-truth throughput (scoring only), interventions and recovery results;
  * CPU time, current resident memory, live Python objects, threads and open file
    descriptors;
  * a resource-monitor sample (`resources.jsonl`).
* **Failed episodes** (exception, supervisor abort, lost connection) are logged and the run
  continues, but only after the environment answers a ping and accepts a reset.
* **Stopping:** SIGINT/SIGTERM or the operations panel stop the run gracefully. The current
  episode ends as `cancelled` and is replayed on resume. `--max-total-seconds` sets a
  wall-clock budget.
* **Memory growth** is the least-squares slope of resident memory after a warm-up of 10 % of
  the episodes (at least 5). The criterion is under 2 MB per 100 episodes; no verdict is given
  below 20 fitted episodes.

Results are in section 12.

## 11. Performance-profiling methodology

`python -m chest_hunter profile-agent` (`runtime/profiler.py`) wraps named sections of the
agent and runner:

* simulator step (frame capture and action, including the simulator's rendering);
* preprocessing (blur, HSV, segmentation);
* chest detection;
* tracking and localisation (world-model integration);
* interaction cues;
* route selection, NavMesh navigation queries, learned inference;
* action generation (the state-machine tick);
* supervision;
* database persistence.

Each section records:

* **latency** (`perf_counter`, wall clock);
* **CPU time of the calling thread** (`thread_time`). The reference simulator runs in another
  thread of the same process, so a section that waits for it has a long latency but little
  CPU time of its own.

Sections nest: action generation contains route selection, which contains navigation queries.
Samples are bounded (20,000 per component; seeded reservoir sampling beyond that). The report
ranks components by total time and gives mean, median, p95, maximum and CPU totals.

Results are in section 12.

## 12. Results

All results come from the **Python reference arena**, not Unreal Engine. CPU and wall
seconds are real time; throughput is per simulated minute.

### 12.1 Benchmark: five configurations × 100 paired episodes (`benchmark_results/phase6/benchmark/`)

**Run.**

```bash
python -m chest_hunter benchmark-phase6 --episodes 100 --seed 42 --workers 4 \
    --model training/checkpoints/bc_synthetic_v1 --experience-db training/experience.db \
    --output-dir benchmark_results/phase6/benchmark
```

* `research_arena_3d` at 640×360, spawn points given. An episode ends when every spawn point
  is resolved, or at 18,000 ticks (600 simulated seconds).
* Five configurations on the same 100 seeds (paired), 500 episodes, 0 errors. 2 h 13 min of
  wall clock with 4 workers, on a machine shared with other benchmarks.
* **Code version.** The run started at commit `d0612c3`, with the Phase 8 fixes F11–F20. The
  F21 fix was committed while it ran. F21 changes only active exploration, and these
  configurations never create an explorer (the runner builds one only when exploration is
  requested). The results therefore hold for the final code.

| Configuration | Chests / min [95 % CI] | Completed | Failed | Reachable opened | Interaction success | Navigation failure rate | Agent CPU s / ep | Process CPU s / ep | Peak RSS (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| phase4_deterministic (baseline) | **4.568 [4.474, 4.662]** | 100 | 0 | 85.6 % | 0.999 | 0.005 | 7.3 | 41.4 | 197 |
| phase5_hybrid | 3.442 [3.334, 3.549] | 99 | 0 | 79.8 % | 0.989 | 0.072 | 11.4 | 54.2 | 197 |
| phase6_hybrid_supervised | 3.447 [3.351, 3.544] | 100 | 0 | 79.7 % | 0.989 | 0.073 | 11.7 | 54.3 | 198 |
| phase6_hybrid_planning | 3.368 [3.268, 3.468] | 100 | 0 | 80.1 % | 0.989 | 0.071 | 11.8 | 55.0 | 198 |
| phase6_hybrid_optimised | 2.797 [2.711, 2.882] | 99 | **1** | 72.6 % | 0.985 | 0.095 | 11.1 | 49.6 | 198 |

The configurations add one component at a time:

* **supervised:** the hybrid plus the reliability supervisor and selector monitors;
* **planning:** plus short-horizon planning and classified recovery;
* **optimised:** plus adaptive computation (the `efficient` vision profile and
  unchanged-frame skipping).

The one `phase5_hybrid` episode not completed reached the 600 s limit; it is not a failure.

**Paired differences, each step against the previous one:**

| Step | Throughput difference [95 % CI] | Better / worse / tied | Agent CPU s per episode |
|---|---|---|---|
| hybrid vs deterministic | **−1.127 [−1.241, −1.012]** (−24.7 %) | 1 / 99 / 0 | +4.13 [+3.69, +4.57] |
| + supervisor | +0.006 [−0.041, +0.053] | 11 / 16 / 73 | +0.28 [−0.09, +0.64] |
| + short-horizon planning and classified recovery | −0.079 [−0.188, +0.029] | 39 / 61 / 0 | +0.12 [−0.21, +0.45] |
| + adaptive computation | **−0.572 [−0.703, −0.440]** (−17 %) | 18 / 82 / 0 | **−0.64 [−1.07, −0.20]** |

Other measurements:

| | deterministic | hybrid | supervised | planning | optimised |
|---|---|---|---|---|---|
| Over-reported openings (500 episodes in total) | 0 | 0 | 0 | 0 | 0 |
| Analysed frames per episode | 1,492 | 1,978 | 1,953 | 1,971 | 1,744 (+57 skipped) |
| Perception ms per analysed frame | 6.07 | 6.23 | 6.12 | 6.16 | 6.28 |
| Learned share of decisions | – | 22.7 % | 22.9 % | 22.0 % | (same model) |
| Inference ms, mean / p95 | – | 0.95 / 1.31 | 0.95 / 1.37 | 0.94 / 1.37 | 0.98 / – |
| Planning ms per decision (mean) | – | – | – | 0.47 | 0.47 |
| Supervisor detections / interventions / aborts | – | – | 17 / 32 / 0 | 28 / 39 / 0 | 30 / 55 / 1 |
| Travel per episode (m) | 828 | 884 | 888 | 942 | 957 |

**What the benchmark shows.**

* **No Phase 6 component improves throughput in fault-free episodes.**
  * The supervisor is neutral (73 of 100 pairs identical). Its value lies in faulted
    episodes, measured in the Phase 7 robustness benchmark and the Phase 8 audit.
  * Short-horizon planning changes no measurable outcome and adds 54 m of travel per
    episode. It is cheap: 0.47 ms per decision.
  * Adaptive computation saves 5 % of agent CPU and 10 % of process CPU (−5.4 s
    [−6.9, −3.9] per episode). It costs 17 % of throughput and produced the benchmark's only
    failed episode. **It should not be a default.** Use it only where the CPU budget is the
    binding limit.
* **The Phase 4 deterministic controller remains the best configuration:** 4.57 chests per
  minute, 85.6 % of reachable chests, 100 of 100 episodes completed. This agrees with
  Phase 5 (−23.9 % for the hybrid there, −24.7 % here).
* **Crediting is exact in all 500 episodes:** 0 over-reported openings after the Phase 8
  fixes F19 and F20.

**The failed episode (finding F22, fixed after this run).**

* `phase6_hybrid_optimised`, episode 57, was aborted by the supervisor at 558.6 s after
  90 s without progress. It replays exactly.
* **Cause.** The agent stood off the NavMesh on a raised spot (4 m up) and had not walked
  anywhere it could retrace. Each unstick manoeuvre therefore went straight behind the
  camera, into the same obstacle: 19 manoeuvres, all at the same bearing, and the agent
  never moved.
* **Fix.** Without a walked trail, each further manoeuvre now turns by a quarter turn. The
  first attempt and every trail-based manoeuvre are unchanged.
* **After the fix** the episode completes at 387.8 s with 17 chests (the row above: 15
  chests, aborted at 558.6 s).
* **Effect on the table.** The table was measured before the fix. This was the only failed
  episode in 500, and the fix changes behaviour only from a second trail-less manoeuvre in a
  row. Replacing the one row would raise the `optimised` mean by 0.01 chests per minute; no
  conclusion changes.
* Regression tests: `test_phase8_audit.test_f22_*`.

### 12.2 Endurance runs (`benchmark_results/phase6/endurance_{100,500,1000}`)

**Runs.**

```bash
python -m chest_hunter endurance-test --episodes N --max-ticks-per-episode 1500 \
    --output benchmark_results/phase6/endurance_N        # N = 100, 500, 1000
```

* Default endurance configuration: deterministic policy, short-horizon planner, supervisor
  on, adaptive compute on, 320×180, `research_arena_3d`, seed 42.
* 1500 ticks (50 simulated seconds) per episode, one reference server and one bridge
  connection for the whole run.
* All three runs started together from commit `e1124bd`, before the Phase 8 fixes. They are
  separate runs, not one run read at three points.
* Each finished in one session: no interruption, no resume.

| Metric | 100 episodes | 500 episodes | 1000 episodes |
|---|---|---|---|
| Episodes completed | 100 / 100 | 500 / 500 | 1000 / 1000 |
| Outcomes | 100 tick limit | 500 tick limit | 1000 tick limit |
| Failed episodes / communication failures / failed resets | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Supervisor interventions / aborts | 1 / 0 | 8 / 0 | 16 / 0 |
| Recovery success rate | 0.886 | 0.853 | 0.868 |
| Openings (ground truth) / over-reported | 432 / 0 | 2214 / 0 | 4472 / 0 |
| Chests per simulated minute [95 % CI] | 5.18 [4.96, 5.41] | 5.31 [5.22, 5.41] | 5.37 [5.30, 5.44] |
| CPU seconds per episode (mean) | 4.63 | 4.51 | 4.38 |
| RSS first / after warm-up / last / max (MB) | 97.0 / 101.7 / 102.1 / 107.0 | 97.3 / 102.3 / 105.3 / 109.1 | 97.0 / 103.0 / 110.3 / 110.6 |
| RSS growth after warm-up (MB per 100 episodes) | 1.36 | 0.54 | 0.47 |
| Live Python objects growth (per 100 episodes) | 325 | 198 | 206 |
| Open file descriptors / threads (min–max) | 9–9 / 3–3 | 9–9 / 3–3 | 9–9 / 3–3 |
| Previous agent released | 99 / 99 | 499 / 499 | 999 / 999 |
| Memory bounded (< 2 MB per 100 episodes) | yes | yes | yes |
| Wall clock | 34 min | 2 h 35 min | 3 h 13 min (3 runs in parallel) |

What the runs show:

* **1,000 consecutive episodes on one server connection: no failure, no communication
  error, no failed reset, no supervisor abort.**
* **Memory is bounded.** RSS grows by about 0.5 MB per 100 episodes after warm-up, and
  13 MB in total over 1,000 episodes. That is under the 2 MB-per-100 criterion, but not
  zero: about 200 live Python objects per 100 episodes accumulate. Their source was not
  identified. At this rate, adding 100 MB would take about 20,000 episodes. No episode's
  agent survived garbage collection.
* **File descriptors and threads are constant**, so no handle or thread leaks.
* **Throughput is stable across the run** (the three runs share their first 100 episodes;
  the means agree within their intervals).
* **What this does not show:**
  * Unreal: these are Python reference arena runs.
  * Long episodes: 50 simulated seconds each; long episodes appear in the Phase 7
    acceptance run, up to 600 s.
  * Active exploration: the endurance configuration uses spawn points.

### 12.3 Profiling and resource use

**From the benchmark (640×360, 100 episodes per configuration):**

* **Agent CPU** (the decision pipeline's own thread) is 0.027 s per simulated second for the
  deterministic controller and 0.034–0.035 s for the hybrid configurations. That is about
  3 % of one core in real time.
  * The learned policy is the difference: 0.95 ms per inference (p95 1.3–1.4 ms) on about
    a fifth of the decisions, plus its feature extraction.
  * Short-horizon planning costs 0.47 ms per decision on average and is negligible.
* **Perception** costs 6.1–6.3 ms per analysed frame at 640×360 (preprocessing, detection,
  tracking and interaction cues).
* **Process CPU** is 41–55 s per episode (267–373 simulated seconds). Most of it is the
  Python reference arena's software renderer and physics, which stand in for Unreal and
  say nothing about Unreal's cost.
* **Peak memory** is 197–198 MiB per benchmark worker, including ONNX Runtime and the
  reference arena.

**Component profile** (`benchmark_results/phase8/resource_profile.json`: Phase 8 audit,
final code, 3 held-out maps, active exploration, deterministic policy, one process):

| configuration | episodes | sim s | wall s | pipeline CPU s / sim s (share of one core in real time) | pipeline CPU ms / analysed frame | frames analysed | bridge client CPU s | simulator CPU s | RSS first / last / max (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 320x180_5fps | 3 | 600.0 | 36.7 | 0.0097 | 1.938 | 3000 | 1.95 | 27.67 | 247.0 / 260.0 / 260.0 |
| 480x270_5fps | 3 | 600.0 | 45.2 | 0.0168 | 3.370 | 3000 | 2.32 | 31.38 | 265.8 / 269.9 / 269.9 |
| 640x360_10fps | 3 | 600.0 | 86.1 | 0.0348 | 4.636 | 4500 | 3.42 | 59.79 | 288.8 / 307.9 / 307.9 |
| 320x180_adaptive | 3 | 600.0 | 32.0 | 0.0092 | 2.147 | 2580 | 2.01 | 22.99 | 313.8 / 314.1 / 314.1 |

Per component at 320×180 and 5 analysed frames per second (mean / p95 ms):

| Component | Calls (3 episodes, 600 simulated s) | Mean ms | p95 ms |
|---|---|---|---|
| preprocessing | 3,000 | 0.55 | 0.76 |
| chest detection | 3,000 | 0.59 | 1.16 |
| tracking and localisation | 3,000 | 0.37 | 0.79 |
| interaction cues | 3,000 | 0.04 | 0.23 |
| exploration update | 3,000 | 0.40 | 1.64 |
| action generation (state-machine tick) | 9,000 | 0.33 | 0.14 |
| route selection | 133 | 18.95 | 46.15 |
| frontier selection | 94 | 20.90 | 37.62 |
| navigation query | 385 | 6.25 | 16.85 |
| database persistence | 15 | 0.11 | 0.23 |
| simulator step (reference arena, wall clock) | 9,000 | 2.58 | 7.96 |

* **Route and frontier selection** are the most expensive single calls (about 20 ms
  each). They run only at decision points: route selection about every 4.5 simulated seconds,
  frontier selection about every 6.4.
* **Action generation** has a p95 below its mean: most ticks cost almost nothing, and the
  mean is raised by the rare ticks that include a route selection.
* **Database writes** are negligible: 0.1 ms, a few per episode.
* **Not measured:** GPU load, Unreal frame time and Unreal memory. There is no GPU and no
  Unreal installation. The resource monitor reports them as unavailable rather than
  estimating them.

### 12.4 Limitations

* **Reference arena only.** Every result in this section comes from the Python mock of the
  Unreal research server. None of it is evidence about Unreal.
* **One map** for the five-configuration benchmark (`research_arena_3d`). Held-out maps are
  covered by Phases 7 and 8.
* **The supervisor's value is in faults,** not clean episodes. It is neutral here; the
  Phase 7 robustness benchmark and the Phase 8 audit measure it under injected faults.
* **Adaptive computation costs throughput** (−17 %) for 5 % less agent CPU. Use it only
  where CPU is the binding constraint.
* **Wall-clock budgets.** The 15 ms planning budget and the 20 ms inference budget are wall
  clock. Exact replay of a lock-step research episode is therefore not guaranteed under CPU
  load (Phase 8 finding F18).
* **Endurance episodes are short** (50 simulated seconds) and use spawn points. About 200
  live Python objects per 100 episodes accumulate; their source was not identified.
* **Defects found after these runs.** The benchmark ran with the Phase 8 fixes up to F20.
  F22 (above) was found in this benchmark and fixed afterwards. The endurance runs predate
  all Phase 8 fixes; none of those fixes concerns resource use.
