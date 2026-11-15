| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 24 | 3.422 (3.431, 1.085) [2.964, 3.880] | 3.241 | 75.8 | 0.972 | 0.944 | 1.000 | 61.7 | 5.2 | - | 0.000 |
| hybrid | 24 | 2.686 (2.642, 0.657) [2.409, 2.963] | 2.565 | 67.1 | 0.992 | 0.920 | 1.000 | 75.3 | 10.3 | 0.89 | 0.192 |
| hybrid_adaptive | 24 | 2.710 (2.815, 0.671) [2.427, 2.993] | 2.710 | 72.9 | 0.989 | 1.000 | 1.000 | 71.8 | 5.6 | 0.91 | 0.150 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- hybrid: mean -0.736 /min [95 % CI -1.025, -0.446], -21.5 %, better 4, worse 20, tied 0 (significant)
- hybrid_adaptive: mean -0.712 /min [95 % CI -1.011, -0.412], -20.8 %, better 5, worse 19, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- hybrid: mean -0.676 /min [95 % CI -1.015, -0.337], -20.9 %, better 7, worse 17, tied 0 (significant)
- hybrid_adaptive: mean -0.531 /min [95 % CI -0.913, -0.148], -16.4 %, better 7, worse 17, tied 0 (significant)
