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
| Endurance runs, checkpoints, resume, interruption, reset failures | **Verified** (`tests/test_endurance.py`); long runs in section 10 |
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
  ending episodes the agent would finish on its own; an early version did, see section 12.

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
