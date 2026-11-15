| Strategy | Episodes | Opened (truth) /min mean (median, sd) [95 % CI] | Reported /min | % avail. opened | Nav. success | Interact. success | Coverage | Travel/chest m | Recoveries | Inference ms | Learned share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic | 100 | 4.403 (4.385, 0.559) [4.292, 4.514] | 4.133 | 85.9 | 0.996 | 0.937 | 1.000 | 43.7 | 11.4 | - | 0.000 |
| learned | 100 | 0.844 (0.800, 0.137) [0.817, 0.871] | 0.844 | 35.7 | 0.775 | 0.661 | 0.435 | 90.6 | 10.3 | 0.89 | 0.878 |
| hybrid | 100 | 3.350 (3.359, 0.532) [3.245, 3.456] | 3.183 | 79.2 | 1.000 | 0.936 | 0.994 | 49.4 | 25.2 | 0.86 | 0.248 |
| hybrid_adaptive | 100 | 3.296 (3.322, 0.570) [3.183, 3.409] | 3.296 | 80.0 | 0.999 | 0.990 | 0.995 | 50.3 | 16.6 | 0.86 | 0.216 |
| deterministic_adaptive | 100 | 4.397 (4.383, 0.488) [4.301, 4.494] | 4.397 | 86.8 | 0.940 | 0.998 | 1.000 | 44.6 | 2.9 | - | 0.000 |

Paired differences per simulated minute, chests opened (ground truth) (deterministic baseline):

- learned: mean -3.559 /min [95 % CI -3.666, -3.452], -80.8 %, better 0, worse 100, tied 0 (significant)
- hybrid: mean -1.053 /min [95 % CI -1.170, -0.935], -23.9 %, better 3, worse 97, tied 0 (significant)
- hybrid_adaptive: mean -1.107 /min [95 % CI -1.229, -0.984], -25.1 %, better 3, worse 97, tied 0 (significant)
- deterministic_adaptive: mean -0.005 /min [95 % CI -0.100, +0.089], -0.1 %, better 53, worse 47, tied 0

Paired differences per simulated minute, chests counted by the agent (deterministic baseline):

- learned: mean -3.289 /min [95 % CI -3.403, -3.174], -79.6 %, better 0, worse 100, tied 0 (significant)
- hybrid: mean -0.950 /min [95 % CI -1.076, -0.824], -23.0 %, better 6, worse 94, tied 0 (significant)
- hybrid_adaptive: mean -0.836 /min [95 % CI -0.963, -0.710], -20.2 %, better 7, worse 93, tied 0 (significant)
- deterministic_adaptive: mean +0.265 /min [95 % CI +0.163, +0.367], +6.4 %, better 72, worse 28, tied 0 (significant)
