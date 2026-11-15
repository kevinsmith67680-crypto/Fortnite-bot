| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 48 | 3.457 (3.386, 1.014) [3.163, 3.752] | 3.247 | 76.2 | 0.984 | 0.941 | 1.000 | 59.2 | 6.0 | - | 0.000 |
| deterministic_adaptive | 48 | 3.383 (3.118, 1.054) [3.077, 3.689] | 3.383 | 75.5 | 0.923 | 1.000 | 1.000 | 64.4 | 1.5 | - | 0.000 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- deterministic_adaptive: mean -0.074 /min [95 % CI -0.244, +0.096], -2.1 %, better 22, worse 26, tied 0

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- deterministic_adaptive: mean +0.136 /min [95 % CI -0.095, +0.367], +4.2 %, better 25, worse 23, tied 0
