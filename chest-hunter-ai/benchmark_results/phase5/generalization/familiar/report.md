| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 24 | 3.673 (3.513, 0.903) [3.292, 4.054] | 3.607 | 78.3 | 0.937 | 0.980 | 1.000 | 57.1 | 3.3 | - | 0.000 |
| learned | 24 | 0.646 (0.600, 0.189) [0.566, 0.726] | 0.646 | 59.0 | 0.780 | 0.742 | 0.669 | 174.7 | 12.9 | 0.88 | 0.879 |
| hybrid | 24 | 3.063 (2.931, 0.668) [2.781, 3.345] | 3.047 | 74.5 | 0.956 | 0.987 | 1.000 | 61.7 | 7.4 | 0.86 | 0.173 |
| hybrid_adaptive | 24 | 3.102 (2.958, 0.667) [2.821, 3.384] | 3.102 | 82.2 | 0.965 | 0.984 | 1.000 | 60.9 | 6.4 | 0.85 | 0.150 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- learned: mean -3.027 /min [95 % CI -3.386, -2.668], -82.4 %, better 0, worse 24, tied 0 (significant)
- hybrid: mean -0.610 /min [95 % CI -0.884, -0.335], -16.6 %, better 5, worse 19, tied 0 (significant)
- hybrid_adaptive: mean -0.570 /min [95 % CI -0.835, -0.306], -15.5 %, better 4, worse 20, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- learned: mean -2.961 /min [95 % CI -3.319, -2.602], -82.1 %, better 0, worse 24, tied 0 (significant)
- hybrid: mean -0.560 /min [95 % CI -0.839, -0.280], -15.5 %, better 6, worse 18, tied 0 (significant)
- hybrid_adaptive: mean -0.504 /min [95 % CI -0.775, -0.233], -14.0 %, better 6, worse 18, tied 0 (significant)
