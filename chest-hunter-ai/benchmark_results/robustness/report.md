# Phase 7 robustness benchmark (controlled failure injection)

Environment: Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. Configuration phase7_active; 100 faulted episodes on held-out layouts, each with a paired fault-free baseline; 4500 ticks.

| fault | applied / episodes | detected | recovered | clean end | failed | throughput | paired diff vs baseline |
|---|---|---|---|---|---|---|---|
| frame_loss | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.800 [1.256, 2.344] (n=10) | 0.240 [-0.096, 0.576] (n=10) |
| corrupted_frame | 10 / 10 | 100 % (10/10) | 90 % (9/10) | 100 % (10/10) | 0 | 1.400 [0.948, 1.852] (n=10) | 0.200 [-0.109, 0.509] (n=10) |
| false_detection | 10 / 10 | n/a | 100 % (10/10) | 100 % (10/10) | 0 | 1.920 [1.335, 2.505] (n=10) | 0.040 [-0.210, 0.290] (n=10) |
| missed_detection | 10 / 10 | n/a | 70 % (7/10) | 100 % (10/10) | 0 | 1.878 [0.992, 2.764] (n=10) | -0.160 [-0.522, 0.202] (n=10) |
| localisation_drift | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.880 [1.497, 2.263] (n=10) | -0.200 [-0.610, 0.210] (n=10) |
| unexpected_obstacle | 10 / 10 | 70 % (7/10) | 90 % (9/10) | 100 % (10/10) | 0 | 1.560 [0.935, 2.185] (n=10) | 0.040 [-0.171, 0.251] (n=10) |
| interaction_failure | 6 / 10 | 100 % (6/6) | 100 % (6/6) | 100 % (6/6) | 0 | 2.467 [1.494, 3.439] (n=6) | 0.000 [-0.266, 0.266] (n=6) |
| bridge_disconnect | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.760 [1.124, 2.396] (n=10) | -0.120 [-0.392, 0.151] (n=10) |
| slow_inference | 8 / 10 | 100 % (8/8) | 50 % (4/8) | 100 % (8/8) | 0 | 1.900 [1.147, 2.653] (n=8) | 0.100 [-0.055, 0.255] (n=8) |
| unreachable_target | 10 / 10 | 70 % (7/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.720 [1.101, 2.339] (n=10) | -0.120 [-0.313, 0.073] (n=10) |

Overall: 94 faults applied; recovered 90 % (85/94); clean termination 100 % (94/94); failed episodes 0.
Baseline (no fault): 100 episodes, throughput 1.788 [1.614, 1.962] (n=100), failed 0.
