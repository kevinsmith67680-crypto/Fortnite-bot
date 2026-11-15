# Exploration strategies: fixed vs bandit-selected

Environment: Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. Categories ['unseen_layout'], 20 episodes, seed 42.

| configuration | n | throughput | coverage | travel (m) | strategies used |
|---|---|---|---|---|---|
| phase7_nearest_frontier | 20 | 1.575 [1.189, 1.961] (n=20) | 0.383 [0.321, 0.446] (n=20) | 363.3 [314.3, 412.3] (n=20) | {'nearest_frontier': 20} |
| phase7_max_information | 20 | 1.575 [1.256, 1.893] (n=20) | 0.470 [0.387, 0.554] (n=20) | 479.8 [422.9, 536.6] (n=20) | {'max_information': 20} |
| phase7_chest_prior | 20 | 1.485 [1.229, 1.741] (n=20) | 0.448 [0.370, 0.524] (n=20) | 482.8 [426.3, 539.3] (n=20) | {'chest_prior': 20} |
| phase7_cluster | 20 | 1.290 [0.981, 1.599] (n=20) | 0.352 [0.297, 0.406] (n=20) | 439.7 [386.7, 492.8] (n=20) | {'cluster': 20} |
| phase7_risk_adjusted | 20 | 1.485 [1.143, 1.827] (n=20) | 0.465 [0.386, 0.544] (n=20) | 484.3 [426.8, 541.8] (n=20) | {'risk_adjusted': 20} |
| phase7_active | 20 | 1.560 [1.236, 1.884] (n=20) | 0.467 [0.387, 0.547] (n=20) | 479.1 [422.5, 535.6] (n=20) | {'max_information': 15, 'risk_adjusted': 5} |

Paired difference to phase7_active (throughput):

| configuration | difference | better / worse / tied | significant |
|---|---|---|---|
| phase7_nearest_frontier | 0.015 [-0.170, 0.200] (n=20) | 6 / 7 / 7 | False |
| phase7_max_information | 0.015 [-0.057, 0.087] (n=20) | 1 / 1 / 18 | False |
| phase7_chest_prior | -0.075 [-0.324, 0.174] (n=20) | 7 / 5 / 8 | False |
| phase7_cluster | -0.270 [-0.519, -0.021] (n=20) | 2 / 9 / 9 | True |
| phase7_risk_adjusted | -0.075 [-0.245, 0.095] (n=20) | 1 / 2 / 17 | False |

Bandit choice per map family: {'open': ('max_information', 'best posterior mean (exploit only)'), 'indoor': ('max_information', 'best posterior mean (exploit only)'), 'mixed': ('max_information', 'best posterior mean (exploit only)'), 'complex': ('risk_adjusted', 'best posterior mean (exploit only)')}
