# APPO Basic versus Jev

Primary run, fixed before APPO evaluation: **1111_greedy_eps0.1**.

| Controller | Train | Dev | Calibration | Test | OOD |
|---|---:|---:|---:|---:|---:|
| Jev | 10/24 (41.7%) | 3/6 (50.0%) | 4/6 (66.7%) | 3/6 (50.0%) | 3/6 (50.0%) |
| 1111_greedy_eps0.1 | 8/24 (33.3%) | 3/6 (50.0%) | 2/6 (33.3%) | 3/6 (50.0%) | 1/6 (16.7%) |
| 1111_greedy_eps0 | 8/24 (33.3%) | 3/6 (50.0%) | 3/6 (50.0%) | 3/6 (50.0%) | 0/6 (0.0%) |
| 1111_sample_eps0 | 9/24 (37.5%) | 3/6 (50.0%) | 2/6 (33.3%) | 2/6 (33.3%) | 1/6 (16.7%) |
| 2222_greedy_eps0.1 | 12/24 (50.0%) | 4/6 (66.7%) | 4/6 (66.7%) | 4/6 (66.7%) | 3/6 (50.0%) |
| 2222_greedy_eps0 | 12/24 (50.0%) | 4/6 (66.7%) | 4/6 (66.7%) | 4/6 (66.7%) | 3/6 (50.0%) |
| 2222_sample_eps0 | 11/24 (45.8%) | 4/6 (66.7%) | 4/6 (66.7%) | 4/6 (66.7%) | 3/6 (50.0%) |
| 3333_greedy_eps0.1 | 22/24 (91.7%) | 4/6 (66.7%) | 5/6 (83.3%) | 5/6 (83.3%) | 3/6 (50.0%) |
| 3333_greedy_eps0 | 20/24 (83.3%) | 3/6 (50.0%) | 3/6 (50.0%) | 5/6 (83.3%) | 3/6 (50.0%) |
| 3333_sample_eps0 | 21/24 (87.5%) | 4/6 (66.7%) | 5/6 (83.3%) | 5/6 (83.3%) | 5/6 (83.3%) |

All rows cover the same 48 Basic cases. No controller was chosen using test outcomes.

| Controller | Test mean ticks | OOD mean ticks | Test mean ammo consumed | OOD mean ammo consumed |
|---|---:|---:|---:|---:|
| Jev | 167.33 | 235.67 | 12.00 | 16.67 |
| 1111_greedy_eps0.1 | 155.17 | 284.50 | 4.50 | 2.33 |
| 1111_greedy_eps0 | 155.17 | 286.00 | 2.33 | 1.83 |
| 1111_sample_eps0 | 198.33 | 243.17 | 3.50 | 1.50 |
| 2222_greedy_eps0.1 | 112.00 | 172.17 | 7.17 | 10.83 |
| 2222_greedy_eps0 | 112.00 | 163.50 | 7.67 | 10.83 |
| 2222_sample_eps0 | 111.17 | 196.00 | 7.67 | 13.00 |
| 3333_greedy_eps0.1 | 128.00 | 165.17 | 8.50 | 10.50 |
| 3333_greedy_eps0 | 106.33 | 169.00 | 7.50 | 11.67 |
| 3333_sample_eps0 | 107.17 | 148.17 | 7.17 | 9.00 |

- Success is a positive KILLCOUNT delta, not positive native reward.
- Test and OOD have six independent environment cases each; checkpoints reuse these cases.
- APPO receives RGB and recurrent image history; Jev receives structured visible-object text.
- This measures complete game controllers with different observation representations.
- Only the primary and epsilon=0.1 controls match the existing Jev exploration rate.
- No API calls, optimization, or checkpoint uploads were performed.
