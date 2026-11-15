# Exploration strategies: fixed vs bandit-selected

Environment: Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. Categories ['unseen_layout'], 40 episodes, seed 42.

| configuration | n | throughput | coverage | travel (m) | strategies used |
|---|---|---|---|---|---|
| audit_planner_nearest | 40 | 3.844 [3.512, 4.177] (n=40) | n/a | 364.3 [322.5, 406.1] (n=40) | {'None': 40} |
| audit_planner_adaptive | 40 | 3.651 [3.347, 3.954] (n=40) | n/a | 386.2 [341.2, 431.2] (n=40) | {'None': 40} |
| audit_planner_greedy | 40 | 3.828 [3.490, 4.166] (n=40) | n/a | 364.6 [323.9, 405.2] (n=40) | {'None': 40} |
| audit_planner_short_horizon | 40 | 3.554 [3.205, 3.902] (n=40) | n/a | 397.2 [352.5, 441.9] (n=40) | {'None': 40} |
| audit_planner_frontier | 40 | 1.665 [1.442, 1.888] (n=40) | 0.505 [0.461, 0.550] (n=40) | 506.0 [477.3, 534.8] (n=40) | {'max_information': 40} |

Paired difference to audit_planner_nearest (throughput):

| configuration | difference | better / worse / tied | significant |
|---|---|---|---|
| audit_planner_adaptive | -0.194 [-0.431, 0.044] (n=40) | 15 / 19 / 6 | False |
| audit_planner_greedy | -0.016 [-0.158, 0.125] (n=40) | 8 / 13 / 19 | False |
| audit_planner_short_horizon | -0.291 [-0.585, 0.003] (n=40) | 14 / 24 / 2 | False |
| audit_planner_frontier | -2.179 [-2.510, -1.849] (n=40) | 0 / 40 / 0 | True |

Bandit choice per map family: {'open': ('max_information', 'no exploration memory'), 'indoor': ('max_information', 'no exploration memory'), 'mixed': ('max_information', 'no exploration memory'), 'complex': ('max_information', 'no exploration memory')}
