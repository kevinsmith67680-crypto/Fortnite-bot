# Phase 7 end-to-end acceptance

Map p7_complex_test_000_unfamiliar (held out: unseen layout, rooms, outdoor areas, several elevations, unfamiliar lighting and start). Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. 10 episodes per group, 18000 ticks.

| group | n | throughput | reachable opened | interaction success | coverage | outcomes | failed | loops detected |
|---|---|---|---|---|---|---|---|---|
| phase7_active/clean | 10 | 1.120 [0.942, 1.298] (n=10) | 55.5 % | 94.9 % | 0.733 [0.692, 0.774] (n=10) | {'tick_limit': 10} | 0 | 158 |
| phase7_active/fault | 10 | 1.090 [0.875, 1.304] (n=10) | 54.0 % | 92.4 % | 0.726 [0.673, 0.779] (n=10) | {'tick_limit': 10} | 0 | 152 |
| phase6_priors/clean | 10 | 3.138 [2.648, 3.628] (n=10) | 60.9 % | 96.1 % | n/a | {'completed': 10} | 0 | 5 |
| phase6_priors/fault | 10 | 2.908 [2.152, 3.664] (n=10) | 56.9 % | 95.0 % | n/a | {'completed': 10} | 0 | 23 |

Engineering targets (Phase 7 agent, fault-free episodes):

* interaction_success: measured 0.9492, target 0.9, met: True
* reachable_opened: measured 0.5545, target None, met: None
* coverage: measured 0.7331, target 0.85, met: False
* reproducible (episode 0 played twice, identical outcome, chests_opened_truth, travel_distance_m, exploration_coverage, interaction_attempts, recoveries, sim_seconds): False
* peak RSS of any worker: 143.6 MB

Phase 7 minus Phase 6 (paired by episode):

* clean: throughput -2.018 [-2.510, -1.526] (n=10), better/worse/tied 0/10/0
* fault: throughput -1.818 [-2.531, -1.105] (n=10), better/worse/tied 1/9/0
