| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 24 | 3.422 (3.431, 1.085) [2.964, 3.880] | 3.241 | 75.8 | 0.972 | 0.944 | 1.000 | 61.7 | 5.2 | - | 0.000 |
| learned | 24 | 0.671 (0.600, 0.220) [0.578, 0.764] | 0.671 | 68.8 | 0.829 | 0.681 | 0.771 | 164.3 | 11.2 | 0.87 | 0.878 |
| hybrid | 24 | 2.686 (2.642, 0.657) [2.409, 2.963] | 2.565 | 67.1 | 0.992 | 0.920 | 1.000 | 75.3 | 10.3 | 0.83 | 0.192 |
| hybrid_adaptive | 24 | 2.626 (2.813, 0.673) [2.342, 2.910] | 2.626 | 72.0 | 0.937 | 0.997 | 1.000 | 73.4 | 7.4 | 0.84 | 0.147 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- learned: mean -2.751 /min [95 % CI -3.135, -2.367], -80.4 %, better 0, worse 24, tied 0 (significant)
- hybrid: mean -0.736 /min [95 % CI -1.025, -0.446], -21.5 %, better 4, worse 20, tied 0 (significant)
- hybrid_adaptive: mean -0.796 /min [95 % CI -1.111, -0.480], -23.2 %, better 4, worse 20, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- learned: mean -2.570 /min [95 % CI -2.997, -2.142], -79.3 %, better 0, worse 24, tied 0 (significant)
- hybrid: mean -0.676 /min [95 % CI -1.015, -0.337], -20.9 %, better 7, worse 17, tied 0 (significant)
- hybrid_adaptive: mean -0.615 /min [95 % CI -1.011, -0.218], -19.0 %, better 7, worse 17, tied 0 (significant)
