| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 100 | 4.403 (4.385, 0.559) [4.292, 4.514] | 4.133 | 85.9 | 0.996 | 0.937 | 1.000 | 43.7 | 11.4 | - | 0.000 |
| deterministic_adaptive | 100 | 4.388 (4.347, 0.485) [4.292, 4.485] | 4.388 | 86.7 | 0.939 | 0.998 | 1.000 | 44.7 | 2.9 | - | 0.000 |
| hybrid_adaptive | 100 | 3.281 (3.257, 0.611) [3.160, 3.403] | 3.281 | 79.5 | 0.998 | 0.988 | 0.989 | 50.5 | 16.3 | 0.91 | 0.216 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- deterministic_adaptive: mean -0.015 /min [95 % CI -0.107, +0.078], -0.3 %, better 53, worse 47, tied 0
- hybrid_adaptive: mean -1.122 /min [95 % CI -1.259, -0.984], -25.5 %, better 3, worse 97, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- deterministic_adaptive: mean +0.256 /min [95 % CI +0.155, +0.356], +6.2 %, better 71, worse 29, tied 0 (significant)
- hybrid_adaptive: mean -0.851 /min [95 % CI -0.989, -0.713], -20.6 %, better 8, worse 92, tied 0 (significant)
