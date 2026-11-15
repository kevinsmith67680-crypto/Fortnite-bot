# Phase 7 robustness benchmark (controlled failure injection)

Environment: Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. Configuration phase7_active; 100 faulted episodes on held-out layouts, each with a paired fault-free baseline; 4500 ticks.

| fault | applied / episodes | detected | recovered | clean end | failed | throughput | paired diff vs baseline |
|---|---|---|---|---|---|---|---|
| frame_loss | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.680 [1.258, 2.102] (n=10) | 0.120 [-0.212, 0.452] (n=10) |
| corrupted_frame | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.400 [0.948, 1.852] (n=10) | 0.120 [-0.116, 0.356] (n=10) |
| false_detection | 10 / 10 | n/a | 100 % (10/10) | 100 % (10/10) | 0 | 1.920 [1.335, 2.505] (n=10) | 0.040 [-0.122, 0.202] (n=10) |
| missed_detection | 8 / 10 | n/a | 75 % (6/8) | 100 % (8/8) | 0 | 2.300 [1.358, 3.242] (n=8) | 0.050 [-0.469, 0.569] (n=8) |
| localisation_drift | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.840 [1.454, 2.226] (n=10) | -0.160 [-0.496, 0.176] (n=10) |
| unexpected_obstacle | 9 / 10 | 67 % (6/9) | 89 % (8/9) | 100 % (9/9) | 0 | 1.511 [0.777, 2.245] (n=9) | -0.089 [-0.294, 0.116] (n=9) |
| interaction_failure | 6 / 10 | 100 % (6/6) | 100 % (6/6) | 100 % (6/6) | 0 | 2.400 [1.520, 3.280] (n=6) | -0.067 [-0.383, 0.249] (n=6) |
| bridge_disconnect | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.760 [1.124, 2.396] (n=10) | 0.040 [-0.051, 0.131] (n=10) |
| slow_inference | 8 / 10 | 100 % (8/8) | 50 % (4/8) | 100 % (8/8) | 0 | 1.900 [1.147, 2.653] (n=8) | 0.100 [-0.055, 0.255] (n=8) |
| unreachable_target | 10 / 10 | 70 % (7/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.720 [1.101, 2.339] (n=10) | -0.120 [-0.313, 0.073] (n=10) |

Overall: 91 faults applied; recovered 92 % (84/91); clean termination 100 % (91/91); failed episodes 0.
Baseline (no fault): 100 episodes, throughput 1.748 [1.572, 1.925] (n=100), failed 0.
