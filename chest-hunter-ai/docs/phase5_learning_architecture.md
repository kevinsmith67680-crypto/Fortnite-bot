# Phase 5: imitation learning, adaptive exploration and generalisation

Phase 5 adds a learning layer to the Phase 4 visual agent. The agent can record
demonstrations, train a compact imitation policy and run it on a CPU. It can also choose
between learned and deterministic control, learn spawn probabilities and travel times from
earlier episodes, recover from failures by failure class, and be evaluated on maps it has
never seen.

Everything runs in the project's own research environment. Nothing here touches a
commercial game, its matchmaking, its client or its anti-cheat. The agent acts only
through the environment's own control interface (the bridge `STEP` command), and the
policy sees only what the visual agent sees.

## 0. What is verified and what is not

| Part | Status |
|---|---|
| Demonstration recorder (start/pause/resume/stop, alignment checks, PNG frames, exact replay) | **Verified** by tests (`tests/test_demonstration_recorder.py`), against the reference server |
| Human teleoperation window | **Verified with simulated key events** under Xvfb. **No human demonstration has been recorded**: none was available. |
| Demonstration dataset used for training | **Synthetic**: 30 episodes recorded from the deterministic Phase 4 controller (`source = synthetic-scripted-expert`). It is labelled as synthetic everywhere, and no claim of human-like behaviour is made. |
| Dataset pipeline, NumPy behaviour cloning, ONNX FP32/INT8 export, CPU inference | **Verified**: tests include a numerical gradient check and export-parity checks; trained model and measurements in sections 5–8 |
| Policy selector and safety layer | **Verified** (`tests/test_policy_selector.py`) |
| Spawn estimator, route memory, adaptive planner, classified recovery | **Verified** (`tests/test_spawn_estimator.py`, `tests/test_adaptive_planner.py`) |
| Procedural maps and unseen-map execution | **Verified** on the Python reference arena (`tests/test_generalization.py`) |
| Benchmarks | Measured on the **Python reference arena** (section 13). They are not Unreal Engine results. |
| Unreal Engine | **Not available.** The C++ additions (procedural layouts, lighting and chest appearance, the map list in `HELLO`) are prepared but **not compiled or run**. Human teleoperation against Unreal works over the same bridge in principle, but is **untested**. |

## 1. Architecture and modifications

```text
   research environment (reference arena; Unreal prepared)        experience database
            | camera frames, own camera rig, drifting odometry          (SQLite)
            v                                                              ^
   perception (Phase 4: detector, distance, tracker, cues)                 |
            v                                                              |
   estimated world state (WorldModel3D) ------------+                      |
            |                                       |                      |
   deterministic controller (VisualAgent3D)   learned policy (ONNX/NumPy) |
     + AdaptivePlanner (spawn estimates,            |                      |
       route memory, scoring)                       |                      |
     + classified recovery                          |                      |
            +---------------- PolicySelector -------+                      |
                     (deterministic | learned | hybrid + safety layer)    |
                                     v                                     |
                     bridge STEP (the environment's control interface)    |
                                     v                                     |
              experience: sessions, observations, routes ------------------+
              demonstrations: frames + actions (DemonstrationRecorder)
                                     v
              OFFLINE: dataset -> behaviour cloning -> ONNX export (never in the loop)
```

New packages: `learning/` (recorder, teleoperation, demo generation, dataset, behaviour
cloning, export, learned policy, policy selector) and `exploration/` (spawn estimator,
frontier scoring, route memory, adaptive planner). New modules: `agent/recovery_3d.py`,
`simulation3d/procedural.py`, `analytics/` (stats, policy comparison, generalisation,
learning metrics) and `cli_learning.py`.

Changes to Phase 4 code are opt-in hooks: an optional `planner`, `smart_recovery`
(default off), `credit_opening()`, `release_interaction()` and extra metrics. Three
corrections were made:

* `INTERACTING → EXPLORING` was missing from the transition table. When a goal disappeared
  in the middle of an interaction (for example, credited by late verification), the episode
  crashed. A Phase 5 test found this; it never occurred in the Phase 4 benchmark.
* `reward_received` was recorded as "a reward happened earlier in this episode" instead of
  this opening's own notification. Only the database flag changes, not behaviour.
* A spawn point that is already marked opened is never counted again (one chest per spawn
  point and episode). The Phase 4 controller never did this; the guard matters when two
  controllers share the bookkeeping (section 9).

**The deterministic Phase 4 controller is unchanged.** Re-running Phase 4 benchmark
episodes on the final code gives identical results in every non-timing field.

## 2. Demonstration recording

* **Human:** `python -m chest_hunter record-demo --controller human --map research_arena_3d
  --output training/datasets/demo_01`. A small Tk window shows the agent's camera. Keys
  pressed in that window are turned into bridge actions:
  * W/S/A/D to move, the arrow keys to look, Space to jump, E to interact (hold), Shift to sprint;
  * P to pause or resume, Esc to stop and save.
  While paused the environment is not stepped, so recordings stay exactly replayable. Use
  `--backend unreal` to drive a running Unreal research server instead (untested).
* **Synthetic (scripted expert):** `record-demo --controller scripted --map ... --episodes N`.
  The deterministic controller plays and is recorded, labelled `synthetic-scripted-expert`.
  Its sessions and observations are written to the experience database
  (`learning.experience_db`), which the spawn estimator learns from.
* **Privacy:** only environment frames, commands and telemetry are stored. There is no desktop
  capture, no audio, and no user or host names.

## 3. Dataset format

```text
<dataset>/episode_00042/episode.json   metadata (ids, map, seed, source, synthetic, settings,
                                       counts, replayable, outcome, recorder version)
<dataset>/episode_00042/steps.npz      one row per simulator step: tick_start, ticks,
                                       action_time, action (float64; move_forward, move_right,
                                       look_yaw, look_pitch, jump, interact), observed,
                                       frame_ref, frame_id, frame_time, proprio, odometry_after,
                                       phase, outcome, reward
<dataset>/episode_00042/frames/*.png   128x72 BGR frames (lossless; identical frames stored once)
<dataset>/splits.json                  persistent train/validation/test assignment
```

`learning.demonstration_recorder.iter_samples(dir)` gives the human-readable sample view
the spec shows (`episode_id`, `frame_id`, `observation_path`, actions). Every simulator
step is stored, so `replay_episode` can reset the same map and seed, apply each action
and compare. It reproduces the recorded frames and positions bit for bit (tested).

## 4. Data validation rules

A sample is rejected, with a counted reason, when:

* the image is unreadable or has the wrong size;
* an action is not finite, a continuous action lies outside [-1, 1], or a binary action is
  not 0 or 1;
* the frame reference is missing;
* the frame time is not the action time (misalignment);
* the proprioceptive values are not finite.

Whole episodes are rejected when files are unreadable or the frame size or action layout
differs.

Splits are made by complete episode (default) or complete map. They are never made by
frame, because neighbouring frames are near-duplicates. The assignment is saved to
`splits.json` and reused, so the test split stays untouched while models are tuned on the
validation split. Normalisation statistics come from the training split only. Images are
read one batch at a time.

## 5. Model architecture

A compact CNN of 28,493 parameters, implemented in NumPy (`learning/behavior_cloning.py`):

```text
image 3x72x128 (normalised BGR) -> conv5x5/2 16 -> conv3x3/2 32 -> conv3x3/2 32 (ReLU)
-> average pooling into 4 vertical strips (keeps left/right position) -> 128
-> concat [camera pitch/60, speed/7] -> FC 64 -> FC 64 -> 13 outputs:
   tanh x4 (move_forward, move_right, look_yaw, look_pitch), sigmoid x2 (jump, interact),
   softmax x7 (controller phase; auxiliary, used as the hybrid confidence)
```

Loss: `L = λm·Huber(move) + λc·Huber(camera) + λi·BCE(interact) + λj·BCE(jump) +
λp·CE(phase)`. The interact BCE uses a positive-class weight computed from the training
split. The backward pass is checked against numerical gradients (relative error ~1e-7).

**Why not PyTorch:** its CPU wheel index (`download.pytorch.org`) is blocked from the
development machine, and the default PyPI Linux build pulls about 3 GB of CUDA libraries.
A 28k-parameter model does not justify that, and NumPy training on a CPU is enough at
this size.

## 6. Training configuration

`training/configs/bc_default.json`: up to 30 epochs, batch 32, Adam (lr 1e-3, weight
decay 1e-4), early stopping (patience 5) on the validation loss, seed 0, split by episode
(70/15/15), brightness jitter ±10 %. `train-policy` writes `policy.npz` (best validation
epoch), `policy.json` (architecture, normalisation, training config, dataset statistics,
validation and test metrics against a "predict the mean action" baseline) and
`history.json`.

**Trained model `training/checkpoints/bc_synthetic_v1`:**

* **Dataset:** the synthetic dataset `synthetic_expert_v1`:
  * 30 episodes on 13 maps (`research_arena_3d` with 6 episodes, `proc_train_000` to `proc_train_011` with 2 each);
  * 29,109 valid samples, 0 rejected;
  * 371 MB of 128×72 PNG frames.
* **Split by episode:** 22 training / 4 validation / 4 test episodes, with 21,715 / 4,068 / 3,326 samples.
* **Training:** 21 epochs, stopped early. The best validation loss was 0.0983 at epoch 16. It took 1,739 s on the 4-vCPU machine (about 55 s per epoch without competing load).

Held-out **test** split (episodes never used for training or for choosing the epoch):

| Output | Model | Baseline (predict the training mean) |
|---|---|---|
| move_forward MAE | **0.098** | 0.448 |
| move_right MAE | **0.091** | 0.130 |
| look_yaw MAE | 0.166 | **0.156** (the model is worse than the baseline here) |
| look_pitch MAE | **0.053** | 0.109 |
| interact accuracy / precision / recall | 1.000 / 1.000 / 1.000 | positive rate 0.171 |
| controller-phase accuracy (7 classes) | 0.942 | - |

The movement and pitch commands are learned well. The yaw rate is not: the expert's turning
depends on its route plan and target memory, which a single frame does not show. Interacting
is easy to learn because the expert holds the button exactly while the prompt or progress bar
is on screen. These are imitation metrics on synthetic expert data; they do not show
human-like behaviour.

## 7. Model export

`export-policy` (run automatically by `train-policy`) builds the ONNX graph directly from
the weights, including the output activations (opset 17). INT8 uses ONNX Runtime's static
QDQ quantisation with per-channel weights, calibrated on 200 training frames. Both models
are compared with the NumPy reference on held-out test frames before they are accepted,
and the export fails if FP32 differs by more than 1e-3. The results go to `export.json`.

Measured (`training/checkpoints/bc_synthetic_v1/export.json`, 256 held-out test frames):

| Model | Size | Agreement with the NumPy model |
|---|---|---|
| FP32 ONNX | 115.6 KB | max abs diff 1e-6; identical interact decisions |
| INT8 ONNX (QDQ, per-channel) | 42.3 KB | mean abs action diff 0.037 (max 0.44); interact decisions 100 % identical |

INT8 changes the continuous outputs noticeably, and it is also slower than FP32 for this
tiny model (below). FP32 therefore stays the default; `--quantized` selects INT8.

## 8. CPU inference

* **Backends:** ONNX Runtime (FP32 or INT8, single thread by default) or the NumPy
  reference network, which needs no optional dependency.
* **Input:** the agent's own camera frame, downscaled exactly like the demonstrations,
  plus its camera pitch and speed. No simulator state.
* **Frequency:** inference runs at most once per new frame. In learned mode the agent
  requests 8 frames per simulated second (`learning.learned_observation_rate`).

Single observation, batch 1, one thread, on the 4-vCPU Xeon 2.1 GHz development machine
(`export.json`, 300 runs after 20 warm-up runs):

| Backend | Mean | Median | p95 |
|---|---|---|---|
| ONNX Runtime FP32 | **0.14 ms** | 0.14 ms | 0.16 ms |
| ONNX Runtime INT8 | 0.21 ms | 0.21 ms | 0.25 ms |
| NumPy FP32 (no optional dependency) | 1.33 ms | 1.37 ms | 1.63 ms |

All three are far below the proposed 20 ms target. A complete `LearnedPolicy.predict` call
(downscaling 640×360 to 128×72, normalisation and ONNX Runtime FP32) measured 0.62 ms
median, of which preprocessing was 0.43 ms. That measurement ran while four benchmark
workers were loading the machine (load average 7). Per-episode inference latencies under
benchmark load are in section 13.

## 9. Hybrid control logic and safety layer

`PolicySelector` modes:

* **deterministic:** Phase 4.
* **learned:** the model drives every decision. The deterministic agent still runs
  perception and bookkeeping. Openings are credited when the agent's own notification
  arrives within 2.5 m of an unresolved spawn point.
* **hybrid:** the deterministic agent drives. In APPROACHING and ALIGNING its movement and
  camera commands are replaced by the learned ones, but only if:
  * the model passed validation (its validation MAE beats the mean-action baseline), and
  * P(approach or align) ≥ 0.6 for the current frame.

Safety layer, applied to every learned command:

* finite values;
* movement norm ≤ 1 and look rates ≤ 0.8;
* no pitching beyond the camera limits;
* outputs reused for at most 0.5 s;
* interaction only when the interaction prompt (or progress bar) is visible in the agent's
  own frame;
* jump only when grounded;
* outputs slower than 20 ms are discarded.

After 10 consecutive failures (error, timeout or invalid output) the learned policy is
disabled for the episode. In learned mode, a learned policy that keeps the agent in place
for 4 s gets up to 25 bounded handovers to the deterministic controller. Model failures
never propagate into the simulation loop (tested). The selector can be switched at run
time from the dashboard.

**Handing control back.** While the learned policy drives, the deterministic state machine
is paused, not reset. When it takes over again (a handover, a fallback or a switch away
from *learned*), an INTERACTING or VERIFYING_OPEN phase it was left in is dropped: its hold
timer, aim point and verification counters belong to an earlier situation, and its reward
counter may hold notifications for chests the learned policy opened. The first full
benchmark run showed why: before this rule, 26 of 100 learned episodes credited one spawn
point that was never opened, after a resumed verification saw an opened-looking chest
along its stale aim line (`stale_interactions_released` counts the dropped phases).

## 10. Adaptive exploration

`AdaptivePlanner` replaces two Phase 4 decisions:

* **Which unresolved spawn point to inspect next.** Score:
  `S_i = (p̂_i·q_i + λ·I_i + μ·C_i) / (T_i + ε)` (`exploration/frontier_scoring.py`), where:
  * `p̂_i` is the Bayesian spawn estimate;
  * `q_i` is navigation reliability from route memory;
  * `I_i` is information gain: the entropy of this episode's presence plus weighted
    posterior-variance reduction;
  * `C_i` is the value of nearby candidates (cluster);
  * `T_i` is path length / speed × the learned travel-time factor, plus interaction and
    approach overhead.
* **Which visible chest to approach first**, using the same score with p = 1.

Within an episode, known-empty, opened and unreachable spawn points are never candidates
again. Without data, every term falls back to the Phase 4 rule (`baseline_score`).

**Route memory** (`route_memory` table) stores, for each navigation attempt: start and
destination regions (15 m cells, separate per floor), planned length and time, actual
time, success, failure kind, recoveries and chest outcome. It gives:

* **Travel-time factors:** the median of actual / estimated.
* **Reliability:** Beta(4.5 + successes, 0.5 + failures), with older sessions decayed by
  0.7 per session. A single failure fades; repeated failures persist.

Records are tagged with the layout version: other versions are ignored, and `invalidate()`
marks records stale. Chest presence is never treated as a navigation outcome.

**Classified recovery** (`agent/recovery_3d.py`, `smart_recovery`) maps every recovery to
one of seven classes, each with a bounded strategy:

* `path_blocked`;
* `target_lost` (rescan near the last estimate);
* `localisation_uncertainty` (look again);
* `interaction_failure` (recheck distance; back off and re-verify an opening the agent was
  notified about, then credit it on the notification, counted separately);
* `camera_alignment_failure`;
* `target_unreachable`;
* `model_inference_failure`.

Each class has a budget, and recovery outcomes are recorded per class.

## 11. Bayesian spawn estimation

`p_i ~ Beta(α_i, β_i)`. The prior is `Beta(s·p0, s·(1−p0))`, with `p0` the configured
probability and `s` = 4 pseudo-observations, or `Beta(1, 1)`. Each valid observation adds
to α (chest present) or β (absent). The estimate is the posterior mean with a 95 %
credible interval (`analytics/stats.py`, no SciPy).

What counts as an observation:

* one sample per (session, location);
* completed sessions only;
* the same layout version only (unversioned manual sessions optional);
* unvisited spots keep their prior;
* rewards are never used.

Estimates go to `spawn_estimates`. `chest_locations.spawn_probability` is never changed.
Procedural maps store an uninformative prior of 0.5 in the database, while the generator
draws hidden true probabilities between 0.15 and 0.95, so the estimates have something
real to learn.

## 12. Generalisation testing

`simulation3d/procedural.py` generates `proc_train_NNN` and `proc_test_NNN` maps from
disjoint seed ranges (1000+ and 9000+). Each map varies:

* building count, size, colour and doors;
* raised plateaus with stairs (elevation);
* containers, crates, rocks, hay, gold decoys and tree groups;
* chest locations and true probabilities;
* lighting (×0.85–1.12) and chest appearance (brightness ×0.9–1.08, hue ±3°);
* start position and view direction.

Every map is checked with the navigation graph: the start must be on the graph and at
least 85 % of spawn points reachable. Training uses only `proc_train_*` maps and
`research_arena_3d` with seed 1000. Benchmarks use seed 42 (arena) and seed 5000
(procedural). `generate-maps --export-unreal` writes the layouts for the Unreal arena
builder.

## 13. Performance benchmarks

All numbers in this section come from the **Python reference arena**, not Unreal Engine.
Throughput is per **simulated** minute; CPU and wall times are real time. Machine: the
4-vCPU Intel Xeon 2.1 GHz development machine, no GPU, Python 3.13, 4 worker processes
running in parallel. The benchmarks ran on commit `f544bf6`.

### 13.1 Policy benchmark (research arena)

```bash
python -m chest_hunter benchmark-policies --episodes 100 --workers 4 \
    --strategies deterministic,learned,hybrid,hybrid_adaptive,deterministic_adaptive \
    --model training/checkpoints/bc_synthetic_v1 --experience-db training/experience.db \
    --output-dir benchmark_results/phase5/policies
```

* **Episodes:** `research_arena_3d`, base seed 42, 100 episodes per strategy. Every
  strategy plays the same 100 episodes (same chest draws and start pose), so all
  comparisons are paired by episode.
* **Limits:** 18,000 ticks (600 simulated s) per episode, 640×360 camera.
* **History:** the adaptive strategies read the training-time history (6 episodes, seed
  1000) and do not update it during this benchmark.
* **Outcome:** all 500 episodes completed, with no errors, in 8,259 s wall time.

**Primary metric: chests actually opened per simulated minute.** Openings are taken from
the simulator's ground truth, which is used only for scoring. The agent's own count is
reported as a secondary metric, because it can miss real openings (under-counting) or,
in principle, count openings that did not happen (over-counting). The 95 % intervals use
the t distribution. Paired differences are per episode.

| Strategy | Mean /min | Median | SD | 95 % CI | Paired Δ vs deterministic [95 % CI] | % change | Better / worse / tied |
|---|---|---|---|---|---|---|---|
| deterministic | **4.403** | 4.385 | 0.559 | [4.292, 4.514] | baseline | - | - |
| learned | 0.844 | 0.800 | 0.137 | [0.817, 0.871] | −3.559 [−3.666, −3.452] | −80.8 % | 0 / 100 / 0 |
| hybrid | 3.350 | 3.359 | 0.532 | [3.245, 3.456] | −1.053 [−1.170, −0.935] | −23.9 % | 3 / 97 / 0 |
| hybrid_adaptive | 3.296 | 3.322 | 0.570 | [3.183, 3.409] | −1.107 [−1.229, −0.984] | −25.1 % | 3 / 97 / 0 |
| deterministic_adaptive | 4.397 | 4.383 | 0.488 | [4.301, 4.494] | −0.005 [−0.100, +0.089] (not significant) | −0.1 % | 53 / 47 / 0 |

Secondary metric, chests counted by the agent per simulated minute:

| Strategy | Mean /min | 95 % CI | Paired Δ vs deterministic [95 % CI] | % change | Under- / over-counted (total) | Credited by notification |
|---|---|---|---|---|---|---|
| deterministic | 4.133 | [4.015, 4.250] | baseline | - | 126 / 0 | 0 |
| learned | 0.844 | [0.817, 0.871] | −3.289 [−3.403, −3.174] | −79.6 % | 0 / 0 | 0 |
| hybrid | 3.183 | [3.072, 3.293] | −0.950 [−1.076, −0.824] | −23.0 % | 97 / 0 | 0 |
| hybrid_adaptive | 3.296 | [3.183, 3.409] | −0.836 [−0.963, −0.710] | −20.2 % | 0 / 0 | 165 |
| deterministic_adaptive | 4.397 | [4.301, 4.494] | **+0.265 [+0.163, +0.367]** | **+6.4 %** | 0 / 0 | 155 |

The +6.4 % for deterministic_adaptive is **bookkeeping, not more chests**. The classified
recovery confirms openings the agent was notified about but did not see clearly, which
the plain controller leaves uncounted. Measured against ground truth, the two
strategies are indistinguishable. In all 500 episodes no strategy recorded an opening
that did not happen.

Other measurements (means per episode unless noted):

| Strategy | % of available chests opened | Chests found | Navigation success | Interaction success | Spawn points resolved | Travel / chest | RECOVERING entries | Simulated s |
|---|---|---|---|---|---|---|---|---|
| deterministic | 85.9 % | 94.3 % | 0.996 | 0.937 | 100 % | 43.7 m | 11.4 | 279.0 |
| learned | 35.7 % | 55.9 % | 0.775 | 0.661 | 43.5 % | 90.6 m | 10.3 | 600.0 |
| hybrid | 79.2 % | 94.1 % | 1.000 | 0.936 | 99.4 % | 49.4 m | 25.2 | 342.9 |
| hybrid_adaptive | 80.0 % | 93.6 % | 0.999 | 0.990 | 99.5 % | 50.3 m | 16.6 | 352.9 |
| deterministic_adaptive | 86.8 % | 93.6 % | 0.940 | 0.998 | 100 % | 44.6 m | 2.9 | 281.8 |

How the columns are defined:

* *Chests found*: available chests the agent saw, from its own world model.
* *Interaction success*: openings the agent counted, divided by its interaction attempts.
  Under-counting therefore lowers it.
* *RECOVERING entries*: how often the Phase 4 recovery state was entered.

Recovery attempts by class (totals over 100 episodes):

| Class | deterministic | learned | hybrid | hybrid_adaptive | deterministic_adaptive |
|---|---|---|---|---|---|
| camera_alignment_failure | 802 | 29 | 1,556 | 1,013 | 169 |
| interaction_failure | 339 | 11 | 559 | 835 | 602 |
| path_blocked | 0 | 988 | 405 | 388 | 0 |
| target_lost | 0 | 0 | 0 | 312 | 178 |
| target_unreachable | 29 | 29 | 32 | 47 | 54 |
| localisation_uncertainty | 0 | 0 | 0 | 10 | 6 |
| model_inference_failure | 0 | 0 | 0 | 0 | 0 |

Selector fallbacks (totals over 100 episodes):

* **learned:**
  * `no_observation` 100: the first decision of each episode, before the first frame.
  * `stuck_recovery` 2,500: exactly the cap of 25 handovers in every episode.
  * `timeout` 2: inferences over 20 ms, discarded.
* **hybrid:** `low_confidence` 22,061. These are approach or align decisions where
  P < 0.6, which the controller handled itself.
* **hybrid_adaptive:** `low_confidence` 11,580.

**Why the learned and hybrid policies underperform:**

* **learned:**
  * Every episode reached the 600 s limit.
  * The policy found only 56 % of the available chests. Its travel per chest was twice
    the controller's, and it used up all 25 stuck handovers in every episode.
  * Causes:
    * The policy sees one frame and has no memory and no route plan. Once no chest is
      in view, it has no information about where the unexplored spawn points are.
    * Its yaw output is worse than a constant on held-out data (section 6).
    * Behaviour cloning compounds small errors into states the expert never visited.
      The 988 path-blocked recoveries are consistent with this.
  * After the 25 handovers are used, a stuck learned policy stays stuck until the limit.
* **hybrid:**
  * The model steers only the approach and alignment, but its steering is less precise
    than the analytic controller it imitates.
  * Camera-alignment recoveries doubled (1,556 against 802), and path-blocked recoveries
    appeared (405).
  * Episodes took 64 s longer (+22.9 %, paired 95 % CI +52.7 to +75.1 s), and opened
    1.6 fewer chests (−7.8 %, CI −2.0 to −1.2).

Because of these results, the deterministic controller stays the default policy (the
`learning.policy` setting).

**Adaptive planning:**

* Neither ablation shows a measurable gain in chests opened: deterministic_adaptive
  against deterministic is above; hybrid_adaptive against hybrid is −0.054 /min
  [−0.193, +0.085], −1.6 %.
* deterministic_adaptive travelled 10.7 % further (+88 m [+76, +100] per episode) in the
  same simulated time (+2.9 s [−0.8, +6.5]).
* It entered the RECOVERING state 74 % less often, because look-backs and rescans
  resolve those situations first.
* On this hand-made arena the configured spawn probabilities are already correct, so the
  spawn estimates have nothing to correct (section 15).

**Resources (real time, 4 workers sharing 4 vCPUs):**

| Strategy | CPU s / episode | CPU s per simulated s | Wall s / episode | Perception ms / analysed frame | Inference ms mean / p95 | Peak RSS per worker |
|---|---|---|---|---|---|---|
| deterministic | 45.3 | 0.162 | 45.7 | 6.2 | - | 140 MiB |
| learned | 117.4 | 0.196 | 118.4 | 5.9 | 0.89 / 1.15 | 140 MiB |
| hybrid | 58.9 | 0.172 | 59.5 | 6.5 | 0.86 / 1.11 | 140 MiB |
| hybrid_adaptive | 59.9 | 0.170 | 60.4 | 6.4 | 0.86 / 1.11 | 140 MiB |
| deterministic_adaptive | 44.4 | 0.158 | 44.9 | 6.2 | - | 140 MiB |

So one worker simulates about 5–6× faster than real time on one core of this machine.
That is simulator throughput, not the speed of any real game. Inference latency is the
per-episode mean and p95 of `LearnedPolicy.predict` (preprocessing plus ONNX Runtime
FP32), averaged over episodes, measured while all four cores were busy.

### 13.2 Reproducibility, and what the first run found

The benchmark was run twice with identical settings. The first run exposed three
bookkeeping errors, all fixed (commit `31f34d4`) before the second run:

1. **Discovery reported 0 for the selector strategies.** The ground-truth summary looked
   for the agent's world model on the selector wrapper instead of the wrapped agent.
2. **Over-counting in learned mode.** 26 of 100 learned episodes credited one spawn point
   that was never opened. A stale verification phase was resumed after a handover
   (section 9).
3. **A misleading primary metric.** The first run ranked strategies by the agent's own
   count. deterministic_adaptive then appeared 6.4 % better, but ground truth showed
   −0.1 %. Ground truth is now the primary metric.

The deterministic and deterministic_adaptive strategies do not use the selector. Their
100 episodes each were **identical in both runs** (every non-timing column of every row),
so the fixes did not change the baseline.

### 13.3 Generalisation: familiar vs unseen maps

```bash
python -m chest_hunter benchmark-generalization --familiar 4 --unseen 4 --episodes 6 --workers 4 \
    --model training/checkpoints/bc_synthetic_v1 --experience-db training/experience.db \
    --output-dir benchmark_results/phase5/generalization
```

**Familiar maps:** `proc_train_000`–`003`.

* The demonstrations include 2 episodes on each (seed 1000), so their frames were in the
  training data, some in the validation and test episodes.
* The experience database holds 2 completed sessions per map.

**Unseen maps:** `proc_test_000`–`003`, generated from the test seed range.

* No trained component saw their layouts, obstacles, chest locations, lighting, chest
  appearance or start poses.
* There is no history for them.

**Episodes:**

* Seed 5000, never used for training, with 6 episodes per map.
* Each (strategy, map) pair plays its episodes in sequence, in one in-memory copy of the
  experience database, so hybrid_adaptive learns from its own earlier episodes on that map.
* Episodes are paired across strategies by (map, episode).
* All 192 episodes completed with no errors: 1,154 s wall for the familiar group and
  1,084 s for the unseen group.

| Strategy | Familiar: mean [95 % CI] | Unseen: mean [95 % CI] | Unseen − familiar, Welch [95 % CI] | Paired Δ vs deterministic, familiar [95 % CI] | Paired Δ vs deterministic, unseen [95 % CI] |
|---|---|---|---|---|---|
| deterministic | **3.673** [3.292, 4.054] | **3.422** [2.964, 3.880] | −0.251 [−0.832, +0.329] (−6.8 %) | baseline | baseline |
| learned | 0.646 [0.566, 0.726] | 0.671 [0.578, 0.764] | +0.025 [−0.094, +0.144] (+3.9 %) | −3.027 [−3.386, −2.668] (−82.4 %), 0 / 24 better | −2.751 [−3.135, −2.367] (−80.4 %), 0 / 24 better |
| hybrid | 3.063 [2.781, 3.345] | 2.686 [2.409, 2.963] | −0.377 [−0.762, +0.008] (−12.3 %) | −0.610 [−0.884, −0.335] (−16.6 %), 5 / 24 better | −0.736 [−1.025, −0.446] (−21.5 %), 4 / 24 better |
| hybrid_adaptive | 3.102 [2.821, 3.384] | 2.626 [2.342, 2.910] | −0.476 [−0.866, −0.087] (−15.4 %) | −0.570 [−0.835, −0.306] (−15.5 %), 4 / 24 better | −0.796 [−1.111, −0.480] (−23.2 %), 4 / 24 better |

Chests actually opened per simulated minute. The familiar and unseen groups use different
maps, so their difference is unpaired.

Further measurements (means per episode):

| Group | Strategy | Agent-counted /min | Under- / over-counted | % of available opened | Chests found | Navigation success | Travel / chest | Simulated s | CPU s | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| familiar | deterministic | 3.607 | 4 / 0 | 78.3 % | 92.4 % | 0.937 | 57.1 m | 144.3 | 17.0 | 170 MiB |
| familiar | learned | 0.646 | 0 / 0 | 59.0 % | 83.8 % | 0.780 | 174.7 m | 600.0 | 85.7 | 170 MiB |
| familiar | hybrid | 3.047 | 1 / 0 | 74.5 % | 93.5 % | 0.956 | 61.7 m | 162.0 | 20.0 | 178 MiB |
| familiar | hybrid_adaptive | 3.102 | 0 / 0 | 82.2 % | 93.2 % | 0.965 | 60.9 m | 177.5 | 21.5 | 171 MiB |
| unseen | deterministic | 3.241 | 11 / 0 | 75.8 % | 83.4 % | 0.972 | 61.7 m | 131.1 | 15.2 | 192 MiB |
| unseen | learned | 0.671 | 0 / 0 | 68.8 % | 84.1 % | 0.829 | 164.3 m | 600.0 | 83.4 | 168 MiB |
| unseen | hybrid | 2.565 | 7 / 0 | 67.1 % | 82.5 % | 0.992 | 75.3 m | 145.9 | 17.6 | 202 MiB |
| unseen | hybrid_adaptive | 2.626 | 0 / 0 | 72.0 % | 88.3 % | 0.937 | 73.4 m | 160.9 | 18.7 | 195 MiB |

Under- and over-counted values are totals over 24 episodes. The learned policy reached
the 600 s limit in all 48 of its episodes, using 591 and 598 stuck handovers of the 600
allowed.

Throughput by position in the episode sequence (mean over the 4 maps), with the number
of chests present:

| Group | Episode | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| familiar | chests present | 10.5 | 9.0 | 8.5 | 13.25 | 11.5 | 14.0 |
| familiar | deterministic | 3.511 | 3.185 | 2.900 | 3.774 | 4.118 | 4.549 |
| familiar | hybrid | 3.164 | 2.939 | 2.233 | 3.040 | 3.248 | 3.754 |
| familiar | hybrid_adaptive | 3.214 | 2.805 | 2.484 | 3.065 | 3.172 | 3.875 |
| familiar | learned | 0.600 | 0.625 | 0.450 | 0.675 | 0.775 | 0.750 |
| unseen | chests present | 8.5 | 8.25 | 8.0 | 11.75 | 9.75 | 13.0 |
| unseen | deterministic | 2.712 | 3.083 | 2.728 | 4.109 | 3.338 | 4.560 |
| unseen | hybrid | 2.111 | 2.147 | 2.452 | 3.146 | 2.815 | 3.445 |
| unseen | hybrid_adaptive | 2.119 | 2.192 | 2.111 | 3.120 | 2.728 | 3.486 |
| unseen | learned | 0.500 | 0.600 | 0.525 | 0.850 | 0.675 | 0.875 |

**Interpretation:**

* **The rise over the sequence is not learning.** It follows the number of chests drawn
  for each episode, and it is just as strong for deterministic, which keeps no history.
* **The adaptation effect is the paired difference hybrid_adaptive − hybrid**, which
  differ only in adaptive planning and recovery:
  * familiar: +0.040 /min [−0.140, +0.219] (+1.3 %);
  * unseen: −0.060 /min [−0.294, +0.174] (−2.2 %).
  * Its least-squares slope over the sequence is +0.009 /min per episode
    [−0.099, +0.117] on familiar maps and +0.002 [−0.138, +0.143] on unseen ones.
  * So there is no measurable improvement within six episodes.
* **The generalisation gap:**
  * deterministic itself loses 6.8 % on the unseen maps, which hold fewer chests.
  * The learned components' deficit against deterministic grows from −16.6 % to −21.5 %
    (hybrid) and from −15.5 % to −23.2 % (hybrid_adaptive).
  * Neither change is significant. The Welch difference-in-differences is −0.126 /min
    [−0.514, +0.262] and −0.225 [−0.626, +0.176].
  * The learned policy's deficit is about the same on both groups (−82.4 %, −80.4 %).
  * A gap smaller than about 0.4–0.5 /min cannot be resolved with 24 episodes per group.

@@ABLATION_RESULTS@@

## 14. Hardware requirements

* **Inference:** a CPU only. The 28k-parameter model takes well under a millisecond per
  observation with ONNX Runtime (section 8), and the NumPy fallback needs no optional
  package.
* **Training:** a CPU; about 55 s per epoch on 21.7k samples on the 4-vCPU development
  machine.
* **Agent:** the reference arena's software renderer dominates its cost (Phase 4: 16.5 ms
  per 640×360 frame). Peak memory per worker process was about 116 MiB in Phase 4.

The Unreal runtime was not measured. A full UE5 editor is not expected to run reliably on
8 GB of RAM or integrated graphics, and adding the learned policy does not change that.
The Python agent and the Unreal runtime should be benchmarked separately.

## 15. Known limitations

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
  * In the arena benchmark it accounts for all of deterministic_adaptive's +6.4 % gain in
    agent-counted throughput. Against ground truth the change is −0.1 % (not
    significant).
  * Such openings are counted separately (`credited_by_notification`). The benchmark
    checks every recorded opening against ground truth, and none of the 500 episodes
    counted an opening that did not happen.
* **Procedural maps** have single-storey buildings, with elevation coming from plateaus. True
  spawn probabilities are independent uniform draws, so there is no spatial correlation
  for the planner to exploit.
* **Unreal** was not available. The C++ additions are not compiled, teleoperation against
  Unreal is untested, and exact replay there would need deterministic engine stepping.
* **Training in NumPy** is practical only for small models like this one. A larger model
  would need a framework (PyTorch or similar) and probably a GPU.
@@P5_RESULT_LIMITATIONS@@

## 16. Future improvements

* Record real human demonstrations (the teleoperation window is ready) and compare
  human-trained against expert-trained policies.
* Use DAgger-style data aggregation: run the learned policy, let the deterministic expert
  label the visited states, and retrain. This addresses the compounding-error problem of
  pure behaviour cloning.
* Add temporal context (frame stacking or a small recurrent layer), so the policy can
  remember targets that leave the view.
* Build and run the Unreal project, calibrate perception on its frames, and repeat the
  benchmarks there.
* Online re-estimation inside an episode, for example: a cleared cluster lowers its
  neighbours' probability when presence is correlated.
