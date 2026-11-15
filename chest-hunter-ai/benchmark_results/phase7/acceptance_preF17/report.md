# Phase 7 end-to-end acceptance

Map p7_complex_test_000_unfamiliar (held out: unseen layout, rooms, outdoor areas, several elevations, unfamiliar lighting and start). Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. 10 episodes per group, 18000 ticks.

| group | n | throughput | reachable opened | interaction success | coverage | outcomes | failed | loops detected |
|---|---|---|---|---|---|---|---|---|
| phase7_active/clean | 10 | 1.120 [0.952, 1.288] (n=10) | 55.5 % | 94.9 % | 0.744 [0.698, 0.791] (n=10) | {'tick_limit': 10} | 0 | 281 |
| phase7_active/fault | 10 | 0.991 [0.819, 1.164] (n=10) | 49.0 % | 91.7 % | 0.701 [0.653, 0.749] (n=10) | {'tick_limit': 9, 'failed': 1} | 1 | 283 |
| phase6_priors/clean | 10 | 3.138 [2.648, 3.628] (n=10) | 60.9 % | 96.1 % | n/a | {'completed': 10} | 0 | 5 |
| phase6_priors/fault | 10 | 2.908 [2.152, 3.664] (n=10) | 56.9 % | 95.0 % | n/a | {'completed': 10} | 0 | 23 |

Engineering targets (Phase 7 agent, fault-free episodes):

* interaction_success: measured 0.9492, target 0.9, met: True
* reachable_opened: measured 0.5545, target None, met: None
* coverage: measured 0.7444, target 0.85, met: False
* reproducible (episode 0 played twice, identical outcome, chests_opened_truth, travel_distance_m, exploration_coverage, interaction_attempts, recoveries, sim_seconds): True
* peak RSS of any worker: 137.7 MB

Phase 7 minus Phase 6 (paired by episode):

* clean: throughput -2.018 [-2.418, -1.618] (n=10), better/worse/tied 0/10/0
* fault: throughput -1.917 [-2.623, -1.210] (n=10), better/worse/tied 1/9/0
