# Phase 7 robustness benchmark (controlled failure injection)

Environment: Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. Configuration phase7_active; 100 faulted episodes on held-out layouts, each with a paired fault-free baseline; 4500 ticks.

| fault | applied / episodes | detected | recovered | clean end | failed | throughput | paired diff vs baseline |
|---|---|---|---|---|---|---|---|
| frame_loss | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.520 [1.037, 2.003] (n=10) | 0.000 [-0.135, 0.135] (n=10) |
| corrupted_frame | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.520 [0.835, 2.205] (n=10) | -0.200 [-0.652, 0.252] (n=10) |
| false_detection | 10 / 10 | n/a | 100 % (10/10) | 100 % (10/10) | 0 | 1.920 [1.335, 2.505] (n=10) | -0.000 [-0.234, 0.234] (n=10) |
| missed_detection | 8 / 10 | n/a | 100 % (8/8) | 100 % (8/8) | 0 | 2.050 [1.371, 2.729] (n=8) | 0.100 [-0.055, 0.255] (n=8) |
| localisation_drift | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.840 [1.478, 2.202] (n=10) | -0.200 [-0.478, 0.078] (n=10) |
| unexpected_obstacle | 9 / 10 | 78 % (7/9) | 100 % (9/9) | 100 % (9/9) | 0 | 1.644 [0.984, 2.305] (n=9) | -0.178 [-0.525, 0.170] (n=9) |
| interaction_failure | 6 / 10 | 100 % (6/6) | 100 % (6/6) | 100 % (6/6) | 0 | 2.533 [1.627, 3.440] (n=6) | -0.067 [-0.238, 0.105] (n=6) |
| bridge_disconnect | 10 / 10 | 100 % (10/10) | 100 % (10/10) | 100 % (10/10) | 0 | 1.560 [0.935, 2.185] (n=10) | -0.000 [-0.135, 0.135] (n=10) |
| slow_inference | 8 / 10 | 100 % (8/8) | 38 % (3/8) | 100 % (8/8) | 0 | 1.850 [1.286, 2.413] (n=8) | 0.000 [0.000, 0.000] (n=8) |
| unreachable_target | 10 / 10 | 60 % (6/10) | 90 % (9/10) | 100 % (10/10) | 0 | 1.560 [0.935, 2.185] (n=10) | -0.120 [-0.526, 0.286] (n=10) |

Overall: 91 faults applied; recovered 93 % (85/91); clean termination 100 % (91/91); failed episodes 0.
Baseline (no fault): 100 episodes, throughput 1.783 [1.610, 1.956] (n=100), failed 0.
