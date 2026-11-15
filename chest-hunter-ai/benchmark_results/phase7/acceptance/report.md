# Phase 7 end-to-end acceptance

Map p7_complex_test_000_unfamiliar (held out: unseen layout, rooms, outdoor areas, several elevations, unfamiliar lighting and start). Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. 10 episodes per group, 18000 ticks.

| group | n | throughput | reachable opened | interaction success | coverage | outcomes | failed | loops detected |
|---|---|---|---|---|---|---|---|---|
| phase7_active/clean | 10 | 1.160 [1.012, 1.308] (n=10) | 57.4 % | 95.1 % | 0.745 [0.705, 0.786] (n=10) | {'tick_limit': 10} | 0 | 158 |
| phase7_active/fault | 10 | 1.090 [0.886, 1.294] (n=10) | 54.0 % | 91.6 % | 0.709 [0.650, 0.767] (n=10) | {'tick_limit': 10} | 0 | 224 |
| phase6_priors/clean | 10 | 3.138 [2.648, 3.628] (n=10) | 60.9 % | 96.1 % | n/a | {'completed': 10} | 0 | 5 |
| phase6_priors/fault | 10 | 2.908 [2.152, 3.664] (n=10) | 56.9 % | 95.0 % | n/a | {'completed': 10} | 0 | 23 |

Engineering targets (Phase 7 agent, fault-free episodes):

* interaction_success: measured 0.9508, target 0.9, met: True
* reachable_opened: measured 0.5743, target None, met: None
* coverage: measured 0.7454, target 0.85, met: False
* reproducible (episode 0 played twice, identical outcome, chests_opened_truth, travel_distance_m, exploration_coverage, interaction_attempts, recoveries, sim_seconds): False
* peak RSS of any worker: 146.5 MB

Phase 7 minus Phase 6 (paired by episode):

* clean: throughput -1.978 [-2.439, -1.517] (n=10), better/worse/tied 0/10/0
* fault: throughput -1.818 [-2.521, -1.115] (n=10), better/worse/tied 1/9/0
