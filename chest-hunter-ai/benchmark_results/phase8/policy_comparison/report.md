# Exploration strategies: fixed vs bandit-selected

Environment: Python reference arena (protocol-identical mock of the Unreal research server); NOT Unreal Engine. Categories ['unseen_layout'], 40 episodes, seed 42.

| configuration | n | throughput | coverage | travel (m) | strategies used |
|---|---|---|---|---|---|
| audit_policy_deterministic | 40 | 3.844 [3.512, 4.177] (n=40) | n/a | 364.3 [322.5, 406.1] (n=40) | {'None': 40} |
| audit_policy_learned | 40 | 1.241 [1.083, 1.399] (n=40) | n/a | 387.1 [358.0, 416.3] (n=40) | {'None': 40} |
| audit_policy_hybrid | 40 | 3.389 [3.097, 3.681] (n=40) | n/a | 366.8 [328.1, 405.5] (n=40) | {'None': 40} |

Paired difference to audit_policy_deterministic (throughput):

| configuration | difference | better / worse / tied | significant |
|---|---|---|---|
| audit_policy_learned | -2.603 [-2.897, -2.309] (n=40) | 0 / 40 / 0 | True |
| audit_policy_hybrid | -0.456 [-0.700, -0.211] (n=40) | 9 / 30 / 1 | True |

Bandit choice per map family: {'open': ('max_information', 'no exploration memory'), 'indoor': ('max_information', 'no exploration memory'), 'mixed': ('max_information', 'no exploration memory'), 'complex': ('max_information', 'no exploration memory')}
