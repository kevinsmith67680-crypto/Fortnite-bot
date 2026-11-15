# Phase 7 robustness benchmark (controlled failure injection)

Environment: Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. Configuration phase7_active; 100 faulted episodes on held-out layouts, each with a paired fault-free baseline; 4500 ticks.

| fault | applied / episodes | detected | recovered | clean end | failed | throughput | paired diff vs baseline |
|---|---|---|---|---|---|---|---|
| frame_loss | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.720 [1.314, 2.126] (n=10) | 0.160 [-0.148, 0.468] (n=10) |
| corrupted_frame | 10 / 10 | 100 % (10/10) | 90 % (9/10) | 100 % (10/10) | 0 | 1.400 [0.948, 1.852] (n=10) | 0.120 [-0.116, 0.356] (n=10) |
| false_detection | 10 / 10 | n/a | 100 % (10/10) | 100 % (10/10) | 0 | 1.920 [1.335, 2.505] (n=10) | 0.040 [-0.122, 0.202] (n=10) |
| missed_detection | 8 / 10 | n/a | 88 % (7/8) | 100 % (8/8) | 0 | 2.300 [1.411, 3.189] (n=8) | -0.050 [-0.467, 0.367] (n=8) |
| localisation_drift | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.800 [1.437, 2.163] (n=10) | -0.320 [-0.873, 0.233] (n=10) |
| unexpected_obstacle | 9 / 10 | 56 % (5/9) | 89 % (8/9) | 100 % (9/9) | 0 | 1.511 [0.777, 2.245] (n=9) | -0.044 [-0.229, 0.140] (n=9) |
| interaction_failure | 6 / 10 | 100 % (6/6) | 100 % (6/6) | 100 % (6/6) | 0 | 2.600 [1.730, 3.470] (n=6) | 0.133 [-0.083, 0.350] (n=6) |
| bridge_disconnect | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.720 [1.087, 2.353] (n=10) | -0.040 [-0.131, 0.051] (n=10) |
| slow_inference | 8 / 10 | 100 % (8/8) | 50 % (4/8) | 100 % (8/8) | 0 | 1.900 [1.147, 2.653] (n=8) | 0.100 [-0.055, 0.255] (n=8) |
| unreachable_target | 10 / 10 | 70 % (7/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.720 [1.101, 2.339] (n=10) | -0.120 [-0.313, 0.073] (n=10) |

Overall: 91 faults applied; recovered 92 % (84/91); clean termination 100 % (91/91); failed episodes 0.
Baseline (no fault): 100 episodes, throughput 1.768 [1.589, 1.948] (n=100), failed 0.
