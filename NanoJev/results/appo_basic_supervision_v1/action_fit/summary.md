# APPO action-policy fit

Every model is evaluated on the same frozen expert states and targets.

| Model | Split | Decisions | Argmax agreement | Hard NLL | Soft CE | Soft KL | Brier hard | Brier soft | ECE 15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_sft | calibration | 377 | 25.73% | 2.664719 | 2.674847 | 2.491878 | 1.336035 | 1.242481 | 0.669315 |
| frozen_sft | test | 756 | 26.98% | 2.665005 | 2.683597 | 2.498204 | 1.315928 | 1.227258 | 0.656736 |
| frozen_sft | ood | 589 | 30.05% | 2.110850 | 2.106603 | 2.062299 | 1.147885 | 1.121404 | 0.558956 |
| hard_s17 | calibration | 377 | 80.37% | 0.592024 | 0.686140 | 0.503171 | 0.332656 | 0.258890 | 0.106110 |
| hard_s17 | test | 756 | 81.61% | 0.611951 | 0.691217 | 0.505825 | 0.342321 | 0.262525 | 0.129563 |
| hard_s17 | ood | 589 | 86.08% | 0.658419 | 0.671668 | 0.627364 | 0.362304 | 0.345564 | 0.249065 |
| soft_s17 | calibration | 377 | 75.86% | 0.525126 | 0.560752 | 0.377783 | 0.295213 | 0.210353 | 0.082771 |
| soft_s17 | test | 756 | 77.12% | 0.527418 | 0.558430 | 0.373038 | 0.292201 | 0.204982 | 0.062052 |
| soft_s17 | ood | 589 | 79.29% | 0.529215 | 0.535655 | 0.491351 | 0.293808 | 0.273069 | 0.081563 |
| hard_s29 | calibration | 377 | 81.43% | 0.621820 | 0.683545 | 0.500577 | 0.304507 | 0.222913 | 0.120134 |
| hard_s29 | test | 756 | 82.14% | 0.601444 | 0.665057 | 0.479664 | 0.294473 | 0.213187 | 0.108305 |
| hard_s29 | ood | 589 | 79.97% | 0.721067 | 0.729515 | 0.685211 | 0.334389 | 0.315175 | 0.113815 |

ECE measures agreement with expert action labels; it is not a calibration metric for winning the game.
CE, KL and NLL use natural logarithms. Vector Brier sums all action components. Decisions from one episode are correlated.
No run-specific training targets, temperature fitting or test-based checkpoint selection are used.
The JSON report includes per-bin ECE counts, ignored non-Basic counts, input IDs and source/checkpoint hashes.
