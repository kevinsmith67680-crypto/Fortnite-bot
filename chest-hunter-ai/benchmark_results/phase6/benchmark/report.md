| Configuration | Episodes | Opened /min mean (median, sd) [95 % CI] | Completed | Failed | % reachable opened | Interaction success | Navigation failure rate | Recovery success | Sim s between failures | Agent CPU s | Process CPU s | Peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phase4_deterministic | 100 | 4.568 (4.517, 0.474) [4.474, 4.662] | 1.0 | 0 | 85.6 | 0.999 | 0.005 | 0.7333 | 119.29 | 7.3 | 41.4 | 197 |
| phase5_hybrid | 100 | 3.442 (3.450, 0.540) [3.334, 3.549] | 0.99 | 0 | 79.8 | 0.989 | 0.072 | 0.6654 | 23.96 | 11.4 | 54.2 | 197 |
| phase6_hybrid_supervised | 100 | 3.447 (3.448, 0.487) [3.351, 3.544] | 1.0 | 0 | 79.7 | 0.989 | 0.073 | 0.6555 | 22.71 | 11.7 | 54.3 | 198 |
| phase6_hybrid_planning | 100 | 3.368 (3.370, 0.504) [3.268, 3.468] | 1.0 | 0 | 80.1 | 0.989 | 0.071 | 0.6568 | 20.83 | 11.8 | 55.0 | 198 |
| phase6_hybrid_optimised | 100 | 2.797 (2.829, 0.430) [2.711, 2.882] | 0.99 | 1 | 72.6 | 0.985 | 0.095 | 0.5912 | 14.76 | 11.1 | 49.6 | 198 |

Paired differences (phase4_deterministic baseline), mean [95 % CI]:

- phase5_hybrid: opened/min -1.127 [-1.241, -1.012] (-24.7 %, significant); agent CPU s +4.128 [+3.686, +4.570] (+56.9 %, significant); process CPU s +12.819 [+10.693, +14.946] (+31.0 %, significant)
- phase6_hybrid_supervised: opened/min -1.121 [-1.227, -1.015] (-24.5 %, significant); agent CPU s +4.406 [+4.121, +4.691] (+60.8 %, significant); process CPU s +12.901 [+11.857, +13.944] (+31.2 %, significant)
- phase6_hybrid_planning: opened/min -1.200 [-1.315, -1.086] (-26.3 %, significant); agent CPU s +4.525 [+4.220, +4.830] (+62.4 %, significant); process CPU s +13.574 [+12.445, +14.703] (+32.8 %, significant)
- phase6_hybrid_optimised: opened/min -1.772 [-1.885, -1.659] (-38.8 %, significant); agent CPU s +3.890 [+3.546, +4.233] (+53.6 %, significant); process CPU s +8.164 [+6.940, +9.387] (+19.7 %, significant)
