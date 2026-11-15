| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 24 | 3.673 (3.513, 0.903) [3.292, 4.054] | 3.607 | 78.3 | 0.937 | 0.980 | 1.000 | 57.1 | 3.3 | - | 0.000 |
| hybrid | 24 | 3.063 (2.931, 0.668) [2.781, 3.345] | 3.047 | 74.5 | 0.956 | 0.987 | 1.000 | 61.7 | 7.4 | 0.89 | 0.173 |
| hybrid_adaptive | 24 | 3.011 (2.895, 0.658) [2.733, 3.289] | 3.011 | 80.8 | 0.938 | 0.981 | 1.000 | 62.3 | 6.3 | 0.89 | 0.160 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- hybrid: mean -0.610 /min [95 % CI -0.884, -0.335], -16.6 %, better 5, worse 19, tied 0 (significant)
- hybrid_adaptive: mean -0.662 /min [95 % CI -0.918, -0.405], -18.0 %, better 2, worse 22, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- hybrid: mean -0.560 /min [95 % CI -0.839, -0.280], -15.5 %, better 6, worse 18, tied 0 (significant)
- hybrid_adaptive: mean -0.596 /min [95 % CI -0.863, -0.329], -16.5 %, better 4, worse 20, tied 0 (significant)
