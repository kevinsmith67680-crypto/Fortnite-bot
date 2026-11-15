# Phase 8 readiness scorecard

**Classification: UNVERIFIED.** The intended environment, the Unreal Engine research
simulator, has never been built or run. Every end-to-end result comes from the Python
reference arena, a protocol-identical mock of it. Within that mock, every gate that can be
evaluated passes.

* **Weighted score:** **50.3 / 100**, under a rubric that caps mock-only evidence at 70. The
  score summarises the evidence; the gates decide readiness.
* **Not a deployment verdict.** It says nothing about live Fortnite matches. The project is
  confined to an authorised research simulator, and readiness for anything else is not
  assessed or implied.
* **Sources:** `benchmark_results/phase8/` (`audit-system`, final code) and
  `docs/phase8_full_system_audit.md`.

## How the scores are assigned

Each category is scored from 0 to 100 against fixed criteria, written before the final audit
numbers were read:

* **Evidence ceiling.** A category is scored on the environment it was measured in.
  * Results from the Python reference arena, a mock of the intended Unreal environment,
    are capped at **70**.
  * The 2D phases (1–3), whose intended environment is the 2D simulator itself, are not
    capped.
  * Unreal integration has no executed evidence. It can score only for prepared source and
    static consistency, capped at **15**.
* **Within the ceiling**, points are deducted for each measured weakness, with the reason
  stated. A deduction is never offset by a strength in another category.
* **No measurement, no points.** A category with no measurement scores 0, not an estimate.

The weighted score summarises the evidence. **It does not decide readiness.** The gates do
that.

## Gates

Any failed gate prevents READY, whatever the score.

| Gate | Rule |
|---|---|
| Actual Unreal integration verified | the Unreal module built, the map loaded, and an episode ran over the C++ bridge |
| Autonomous loop completes | all twelve stages exercised in a traced episode that ends normally |
| Chest interactions verified | the simulator's interaction rules all hold and there are no false successes in clean episodes |
| Visual verification reliable | false failures ≤ 5 % in clean episodes |
| No critical data-integrity defects | no duplicate openings, no phantom openings, integrity check and foreign keys pass |
| No severe resource leaks | memory growth under 2 MB per 100 episodes in every endurance run |
| Not entirely dependent on mocks | at least one end-to-end result from the intended environment |

Classification:

* **READY FOR CONTROLLED RESEARCH USE**: all gates pass.
* **CONDITIONALLY READY**: all gates pass except non-critical ones, with stated conditions.
* **NOT READY**: a gate fails on evidence.
* **UNVERIFIED**: the intended environment was never run, so the gates that depend on it
  cannot be evaluated.

## Category scores

| Category | Weight | Ceiling | Score | Evidence (section of the audit report) and deductions |
|---|---|---|---|---|
| Core autonomous functionality | 20 % | 70 | **60** | The full loop runs end to end, with all 12 stages traced (B), and no failed episode in any benchmark after the fixes. −5: coverage target missed in the acceptance run ({{ACC_COV}} vs 85 %). −5: exact replay not guaranteed (F18) |
| Unreal integration | 15 % | 15 | **10** | The project and C++ are prepared, and protocol, units and argument ranges match the Python side statically (E). −5: no map asset, never compiled, and `surface_value` is ignored (F3) |
| Computer vision | 10 % | 70 | **45** | Precision 0.93–0.97 and latency 0.9–3.0 ms (F). −15: chest-appearance shift defeats detection (F15, G = +0.28). −10: recall 0.48 at 320×180 per frame, and opened/closed confusion (F16) |
| Navigation | 10 % | 70 | **65** | 32/32 reachable goals; 5/5 unreachable goals detected after one query; detour about 1.0 (G). −5: odometry-only localisation, with no visual correction of a persistent offset |
| Chest interaction | 10 % | 70 | **65** | 7/7 simulator rules. 0 false successes and 0 false failures in 20 cross-checked episodes; 0 over-reports in every benchmark after F19/F20 (H). −5: about 5 % of credits rest on the opening notification after a look-back, not a visual confirmation |
| Reliability | 10 % | 70 | **60** | {{ROB_RECOVERED}} of applied faults recovered, 100 % clean terminations, endurance to 1000 episodes (K). −5: slow-inference recovery {{ROB_SLOW}}. −5: residual loop alarms ({{TWIN_LOOPS_SHORT}}) |
| Performance efficiency | 10 % | 70 | **50** | Python pipeline about 1 % of a core at 320×180 (L). −20: the Unreal graphics workload, frame time and GPU are unmeasured (BLOCKED), and that is most of the real cost |
| Generalisation | 5 % | 70 | **45** | No gap for layout, obstacles, start or elevation (K). −15: appearance gap. −10: lighting gap (G = +0.11) |
| Data integrity | 5 % | 70 | **65** | 13/14 checks; the failing one is an opening still being verified at episode end (L). −5: openings in the last ~6 s of an episode are never recorded |
| Testing and documentation | 5 % | 70 | **55** | {{TESTS}} (N). −15: the suite missed four P1 behaviour defects found only by full benchmarks (F17, F19, F20, F21), and the Phase 6 tests had never been run as a suite (F7–F9) |

Weighted score: 0.20×60 + 0.15×10 + 0.10×(45 + 65 + 65 + 60 + 50) + 0.05×(45 + 65 + 55)
= 12.0 + 1.5 + 28.5 + 8.25 = **50.3**.

## Gate results (`audit_summary.json → readiness`)

| Gate | Result |
|---|---|
| Actual Unreal integration verified | **not met**: never built or run |
| Autonomous loop completes | met (reference arena) |
| Chest interactions verified | met (reference arena) |
| Visual verification reliable (≤ 5 % false failures) | met: 0 % in clean episodes |
| No critical data-integrity defects | met |
| No severe resource leaks | met: ≤ 1.4 MB per 100 episodes |
| Not entirely dependent on mocked components | **not met** |

## What would change the classification

* **To READY FOR CONTROLLED RESEARCH USE:**
  * Build and run the Unreal project (audit report, section O, step 1).
  * Run `audit-system` against `--backend unreal`.
  * Pass the same gates there.
  * Re-measure perception on Unreal frames.
* **What would keep it NOT READY even then:** any false success or more than 5 % false
  failures in Unreal episodes, or a failed episode in the Unreal endurance run.
