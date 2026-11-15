# Phase 6 baseline audit

This audit records the state of the project at the start of Phase 6 (commit `61c08f9`,
end of Phase 5), before any Phase 6 change, plus the measurements taken to decide what to
change. Every number below was measured in this environment; nothing is estimated. All
episodes ran on the **Python reference arena** (a protocol-identical stand-in for the Unreal
research server), not on Unreal Engine.

## 1. Environment

| Item | Value |
|---|---|
| Machine | 4 vCPU Intel Xeon @ 2.10 GHz, 16 GB RAM, no GPU (`nvidia-smi` not present) |
| Python | 3.11.18, 3.12.3 (with Xvfb for GUI tests), 3.13.18 |
| Packages | numpy 2.4/2.5, OpenCV 5.0.0, onnxruntime 1.32.0, onnx 1.23.2; `psutil` not installed (not needed) |
| Unreal Engine | **not installed** (`UnrealEditor`, `UnrealEditor-Cmd`, `RunUAT.sh` absent) |
| Unreal project | `unreal/ChestHunterResearch` present (C++ prepared in Phase 4/5, never compiled) |

## 2. Existing test results

At commit `61c08f9`: **553 tests pass** with `python -m unittest discover -s tests` on Python
3.11 and 3.13 (6 GUI tests skipped without a display) and with `xvfb-run -a` on Python 3.12
(none skipped). `mypy --strict` passed on 99 modules; ruff reported nothing.

## 3. Working components (verified)

| Component | Evidence |
|---|---|
| Phases 1–3 (database, 2D simulation, vision) | their tests pass |
| Phase 4 3D reference arena, bridge, perception, visual agent | tests pass; deterministic episodes reproduce exactly |
| Phase 5 learning pipeline, policy selector, adaptive planner, procedural maps | tests pass |
| Reproducibility | arena episodes 0–1 of the Phase 5 benchmark (`deterministic`, `hybrid`, seed 42, 640×360) re-run at `61c08f9` matched the published rows in every non-timing field |
| Repeated episodes in one process | 40 consecutive deterministic episodes (`proc_train_000`, 320×180) all completed; threads and open file descriptors were the same after every episode (1 and 4) |

## 4. Problems found

| # | Problem | Where | Consequence | Phase 6 action |
|---|---|---|---|---|
| 1 | The same image work done twice per analysed frame: the agent blurs, converts to HSV and segments the frame, and the detector repeats exactly the same steps | `controller_3d.perceive`, `chest_detector_3d.detect` | `segment_classes` was called 3,014 times for 1,507 frames in one profiled episode | the detector reuses the agent's segmentation (bit-identical detections) |
| 2 | An invalid state transition raises `RuntimeError`, ending the episode with an exception | `agent/fsm.py` | a controller bug crashes a long run | refused, logged and counted when a supervisor watches; supervisor redirects |
| 3 | No episode-level supervision: time limits exist only inside some handlers (approach 15 s, align 1.6 s, verify 2 s, bounded scans); no detection of loops, no-progress, invalid frames or a lost connection | — | a stuck pattern is only stopped by the tick limit or the recovery budget | reliability supervisor |
| 4 | A repeated EXPLORING ↔ OBSERVING cycle (waiting to land back on the navigation mesh) of 27 transitions | hybrid arena episode 0 | time lost; bounded only by the agent's own 12-wait rule | loop detector + bounded interventions |
| 5 | The NavMesh path cache key ignores the goal tolerance | `NavMeshNavigator.plan` | a cached plan for one tolerance can answer a query with another (latent; not changed, to keep results reproducible) | documented |
| 6 | Each episode starts a new reference server and connection | `Runner3D.run` | no way to test a persistent environment with resets | `Runner3D(client=...)`; endurance runner keeps one server |
| 7 | No runtime checkpoints for long experiments | — | an interrupted run loses its place | atomic checkpoints, resume |
| 8 | Experience is partitioned by map and layout only | spawn estimator, route memory | route statistics of different agent versions and settings are mixed | partitions (environment, agent version, configuration) |
| 9 | The hybrid gate treats the softmax phase output as confidence | `PolicySelector` | not shown to be calibrated | measured (section 9); calibrator available, opt-in |
| 10 | The deterministic controller under-counts its own openings (visual confirmation fails) | Phase 4 controller | 6 of 40 repeated episodes counted one opening fewer than really happened | unchanged (known Phase 4 limitation); ground truth is the primary metric |

**Unbounded collections checked:**

* `Tracker3D._tracks` is bounded: expired tracks are deleted, at most `max_tracks` (100) are
  alive.
* `StateMachine.history` is a `deque(maxlen=1000)`; the NavMesh cache is cleared at 256
  entries.
* Per-episode lists (`SelectorStats.latencies_ms`, the recovery log's pending targets) grow
  within one episode only. A new agent is built for every episode.

## 5. Missing dependencies

* **Unreal Engine** (editor, build tools): everything Unreal-specific is unverified.
* **GPU**: none; no GPU utilisation can be measured.
* `psutil` is not installed. Phase 6 reads `/proc` and `resource` from the standard library instead.

## 6. Baseline throughput

Published Phase 5 benchmark (100 paired arena episodes, seed 42, 640×360, 600 simulated s
limit; chests really opened per **simulated** minute):

| Strategy | Mean | 95 % CI |
|---|---|---|
| deterministic (Phase 4 controller) | 4.403 | [4.292, 4.514] |
| hybrid (Phase 5) | 3.350 | [3.245, 3.456] |

The Phase 6 benchmark re-runs both as its baseline configurations (see the Phase 6 document).

## 7. Baseline latency

**Simulator round trip vs agent work.** In two profiled arena episodes, process CPU time was 75 s
(deterministic) and 84 s (hybrid). CPU time of the agent's own thread was 17 s and 22 s. The
rest is the reference simulator, which runs in another thread of the same process: rendering
640×360 frames, physics, and NavMesh queries. On a real Unreal setup that work runs in the
engine (GPU rendering, native NavMesh), not in Python.

**Path queries.** Server-side A* in the Python reference NavMesh, 60 random spawn-point pairs,
single thread:

| Map | Median | p95 | Max |
|---|---|---|---|
| research_arena_3d | 18.2 ms | 92.5 ms | 197.7 ms |
| proc_train_000 | 28.4 ms | 86.3 ms | 144.5 ms |

The agent asks for up to four path lengths each time it chooses an inspection target. In one
profiled episode that was 152 path-length calls in 45 target selections.

**Perception.** The Phase 5 benchmark recorded 5.9–6.5 ms per analysed frame (4 workers sharing
4 vCPUs). `cProfile` timings of the same code are inflated by contention for the interpreter
lock with the simulator thread, so they are not quoted. Component timings measured with the
Phase 6 profiler (thread CPU time and wall-clock latency) are in section 10.

## 8. Baseline memory and long runs

40 consecutive deterministic episodes in one process (`proc_train_000`, 320×180, new reference
server per episode, one in-memory database):

| Measure | Value |
|---|---|
| Outcomes | 40 / 40 completed (all spawn points resolved) |
| Resident memory | 83.7 MB after episode 0, 91.2 MB after episode 1, 94.2 MB after episode 9, 95.2 MB after episode 39 |
| Growth after warm-up | 4.4 MB per 100 episodes (least squares over episodes 5–39; the database grows by its records) |
| Live Python objects | 34,328–34,925 (no trend) |
| Threads / open descriptors after each episode | 1 / 4 (constant) |
| CPU per episode | 10.8 s for 144 simulated s on average |
| Openings counted vs really made | 6 episodes counted one opening fewer than really happened (problem 10) |

## 9. State durations and loop patterns

Longest single visit to each timed state (simulated seconds) in 12 deterministic and 6
hybrid arena episodes (seed 42, episodes 0–11 and 0–5, 640×360). These set the supervisor's
per-state limits. Each limit is at least twice the longest visit measured here.

| State | Visits (det.) | Median | p99 | Max (det.) | Max (hybrid) | Phase 6 limit |
|---|---|---|---|---|---|---|
| NAVIGATING | 543 | 2.87 | 16.80 | 18.73 | 16.80 | 45 |
| APPROACHING | 447 | 0.87 | 1.27 | 1.80 | 9.27 | 20 |
| ALIGNING | 463 | 0.13 | 1.67 | 1.67 | 1.67 | 8 |
| INTERACTING | 240 | 2.07 | 2.07 | 2.07 | 2.07 | 6 |
| VERIFYING_OPEN | 238 | 0.07 | 2.07 | 2.07 | 2.07 | 10 |
| SCANNING | 305 | 0.40 | 1.07 | 1.13 | 1.53 | 40 |
| OBSERVING | 14 | 0.07 | 0.13 | 0.13 | 2.40 | 10 |

Longest alternating two-state cycles:

* deterministic: 4 transitions (PLANNING_ROUTE ↔ NAVIGATING when a target estimate moved,
  APPROACHING ↔ ALIGNING when stepping closer);
* hybrid: 27 transitions (EXPLORING ↔ OBSERVING, problem 4).

**Policy confidence** (problem 9). This was measured on the held-out test episodes of the
synthetic demonstration set: 3,326 samples, 893 of them approaching or aligning. Results are
in `benchmark_results/phase6/reliability`.

* As a phase probability, the phase output P(approach or align) is well calibrated: expected
  calibration error 0.016.
* As a predictor of whether the learned command is within 0.25 of the expert's, it is weak:
  AUC 0.56, and an expected calibration error of 0.134 if read as that probability.
* Above the 0.6 gate, 90.4 % of the commands are within 0.25, against 79.3 % below it.
* Histogram binning, fitted on the validation episodes, lowers the calibration error to 0.060
  but cannot add discrimination.

## 10. Component timings

See section 10 of `docs/phase6_reliability_and_optimization.md`. The Phase 6 profiler was run
on the baseline configuration with the machine otherwise idle, after the benchmark.

## 11. Features that need Unreal Engine (not verified here)

* rendering, frame time and GPU use of the real environment;
* the Unreal NavMesh, collision and physics behind the bridge;
* teleoperation and the endurance acceptance run against Unreal;
* compiling the C++ plugin and the Phase 4/5 C++ additions.

The Python reference arena speaks the same bridge protocol, so the agent, supervisor,
planner and endurance runner are exercised end to end. Performance and behaviour in Unreal
can differ.
