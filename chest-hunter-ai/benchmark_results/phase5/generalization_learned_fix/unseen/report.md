| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 24 | 3.422 (3.431, 1.085) [2.964, 3.880] | 3.241 | 75.8 | 0.972 | 0.944 | 1.000 | 61.7 | 5.2 | - | 0.000 |
| learned | 24 | 0.650 (0.600, 0.206) [0.563, 0.737] | 0.650 | 66.8 | 0.800 | 0.716 | 0.758 | 169.2 | 12.3 | 1.00 | 0.878 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- learned: mean -2.772 /min [95 % CI -3.168, -2.375], -81.0 %, better 0, worse 24, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- learned: mean -2.591 /min [95 % CI -3.030, -2.151], -79.9 %, better 0, worse 24, tied 0 (significant)
