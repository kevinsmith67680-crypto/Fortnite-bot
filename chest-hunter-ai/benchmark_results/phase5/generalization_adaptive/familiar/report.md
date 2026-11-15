| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 48 | 3.680 (3.513, 0.942) [3.407, 3.954] | 3.501 | 79.8 | 0.966 | 0.948 | 1.000 | 57.1 | 5.0 | - | 0.000 |
| deterministic_adaptive | 48 | 3.500 (3.381, 0.855) [3.251, 3.748] | 3.500 | 81.7 | 0.943 | 1.000 | 1.000 | 59.1 | 3.2 | - | 0.000 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- deterministic_adaptive: mean -0.181 /min [95 % CI -0.315, -0.047], -4.9 %, better 16, worse 32, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- deterministic_adaptive: mean -0.002 /min [95 % CI -0.156, +0.152], -0.1 %, better 20, worse 28, tied 0
