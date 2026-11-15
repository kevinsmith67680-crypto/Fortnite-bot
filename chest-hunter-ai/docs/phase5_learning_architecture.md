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

@@BENCHMARK_RESULTS@@

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
  about but could not see is credited after two look-backs. Such openings are counted
  separately (`credited_by_notification`), and the benchmark checks every recorded opening
  against ground truth.
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
