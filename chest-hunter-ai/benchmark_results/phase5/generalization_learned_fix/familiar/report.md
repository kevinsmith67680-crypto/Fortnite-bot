| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 24 | 3.673 (3.513, 0.903) [3.292, 4.054] | 3.607 | 78.3 | 0.937 | 0.980 | 1.000 | 57.1 | 3.3 | - | 0.000 |
| learned | 24 | 0.683 (0.650, 0.193) [0.602, 0.765] | 0.683 | 62.5 | 0.730 | 0.742 | 0.703 | 174.4 | 13.0 | 0.95 | 0.878 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- learned: mean -2.989 /min [95 % CI -3.339, -2.639], -81.4 %, better 0, worse 24, tied 0 (significant)

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- learned: mean -2.923 /min [95 % CI -3.273, -2.573], -81.0 %, better 0, worse 24, tied 0 (significant)
