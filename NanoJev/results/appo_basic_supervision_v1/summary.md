# APPO action-supervision evaluation

Fresh matched episodes; synchronized case definitions and fixed action controllers.

| Run | Split / scenario | Success | Wilson 95% | Physical ticks | Ammo used |
|---|---|---:|---|---:|---:|
| frozen_sft | ood/basic | 59/128 (46.1%) | 37.7%–54.7% | 192.68 | 13.70 |
| frozen_sft | test/basic | 56/128 (43.8%) | 35.5%–52.4% | 187.89 | 13.45 |
| hard_s17 | ood/basic | 126/128 (98.4%) | 94.5%–99.6% | 39.38 | 1.95 |
| hard_s17 | test/basic | 128/128 (100.0%) | 97.1%–100.0% | 22.31 | 1.12 |
| soft_s17 | ood/basic | 127/128 (99.2%) | 95.7%–99.9% | 64.80 | 3.15 |
| soft_s17 | test/basic | 128/128 (100.0%) | 97.1%–100.0% | 32.92 | 1.95 |
| hard_s29 | ood/basic | 128/128 (100.0%) | 97.1%–100.0% | 50.97 | 2.61 |
| hard_s29 | test/basic | 128/128 (100.0%) | 97.1%–100.0% | 30.24 | 1.83 |
| appo_1111 | ood/basic | 128/128 (100.0%) | 97.1%–100.0% | 34.19 | 1.31 |
| appo_1111 | test/basic | 128/128 (100.0%) | 97.1%–100.0% | 20.78 | 1.11 |
| random | ood/basic | 61/128 (47.7%) | 39.2%–56.3% | 204.22 | 5.98 |
| random | test/basic | 76/128 (59.4%) | 50.7%–67.5% | 171.52 | 6.95 |

Paired comparisons against **frozen_sft**:

| Run | Split / scenario | Success delta | Improved / regressed | Exact McNemar p |
|---|---|---:|---:|---:|
| hard_s17 | ood/basic | +52.3% | 67 / 0 | 1.3553e-20 |
| hard_s17 | test/basic | +56.2% | 72 / 0 | 4.2352e-22 |
| soft_s17 | ood/basic | +53.1% | 68 / 0 | 6.7763e-21 |
| soft_s17 | test/basic | +56.2% | 72 / 0 | 4.2352e-22 |
| hard_s29 | ood/basic | +53.9% | 69 / 0 | 3.3881e-21 |
| hard_s29 | test/basic | +56.2% | 72 / 0 | 4.2352e-22 |
| appo_1111 | ood/basic | +53.9% | 69 / 0 | 3.3881e-21 |
| appo_1111 | test/basic | +56.2% | 72 / 0 | 4.2352e-22 |
| random | ood/basic | +1.6% | 24 / 22 | 0.883 |
| random | test/basic | +15.6% | 31 / 11 | 0.0028872 |

- Only registered new cases are included; no historical Jev cohort is pooled into this report.
- Source hashes identify original complete files. Explicit split selection is a view, not a replacement source.
- APPO consumes native-rendered recurrent pixels; NanoJev consumes structured visible text. This compares complete policies with different observations.
- Success means observed kill before the native deadline, not positive native reward. Physical ticks and action decisions differ.
- Wilson intervals use episodes as units. Exact McNemar tests use paired discordant cases; p-values are unadjusted across comparisons.
- Paired deltas are other minus reference. Ticks and ammo include failures and early successes and do not independently measure skill.
- Hard seed 17, hard seed 29 and soft seed 17 remain distinct runs. This is not an estimate over many independent training seeds.
- Hashes, case definitions, seeded actions and transition counters are verified here; no simulator replay or model inference is performed.
- Action probabilities describe action choice, not winning probability. Offline expert-action NLL is not evaluated by this rollout summary.
