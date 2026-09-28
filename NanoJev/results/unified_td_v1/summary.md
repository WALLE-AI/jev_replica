# Unified MC versus TD comparison

Complete: **True**. Frozen cases: 228.

All probability metrics use the fixed observed-outcome dataset. Game scores use new Q-controller trajectories. Across-seed summaries describe mean and observed range; no significance test is inferred.

Predeclared primary comparison: **mix_n3 versus mc**.

## Fixed-policy probability scores by arm

| Arm | Split | Questions | NLL | Vector Brier |
| --- | --- | --- | --- | --- |
| mc_seed17 | dev | 2721 | 0.245182 | 0.135466 |
| mc_seed17 | calibration | 2606 | 0.576265 | 0.320908 |
| mc_seed17 | test | 2716 | 0.514960 | 0.271787 |
| mc_seed17 | ood | 3107 | 0.456229 | 0.259882 |
| mix_n3_seed17 | dev | 2721 | 0.301561 | 0.162165 |
| mix_n3_seed17 | calibration | 2606 | 0.462913 | 0.264062 |
| mix_n3_seed17 | test | 2716 | 0.473800 | 0.277081 |
| mix_n3_seed17 | ood | 3107 | 0.425947 | 0.233893 |
| mix_n8_seed17 | dev | 2721 | 0.257646 | 0.141643 |
| mix_n8_seed17 | calibration | 2606 | 0.445760 | 0.256079 |
| mix_n8_seed17 | test | 2716 | 0.438687 | 0.267467 |
| mix_n8_seed17 | ood | 3107 | 0.403533 | 0.213297 |
| td_n8_seed17 | dev | 2721 | 0.294361 | 0.180029 |
| td_n8_seed17 | calibration | 2606 | 0.395792 | 0.214763 |
| td_n8_seed17 | test | 2716 | 0.434137 | 0.238936 |
| td_n8_seed17 | ood | 3107 | 0.371665 | 0.232915 |
| mc_seed29 | dev | 2721 | 0.190113 | 0.093896 |
| mc_seed29 | calibration | 2606 | 0.691361 | 0.361537 |
| mc_seed29 | test | 2716 | 0.498031 | 0.235541 |
| mc_seed29 | ood | 3107 | 0.481292 | 0.239313 |
| mix_n3_seed29 | dev | 2721 | 0.361356 | 0.219079 |
| mix_n3_seed29 | calibration | 2606 | 0.512402 | 0.267832 |
| mix_n3_seed29 | test | 2716 | 0.477057 | 0.266878 |
| mix_n3_seed29 | ood | 3107 | 0.376437 | 0.194814 |
| mix_n8_seed29 | dev | 2721 | 0.209189 | 0.095884 |
| mix_n8_seed29 | calibration | 2606 | 0.600517 | 0.321614 |
| mix_n8_seed29 | test | 2716 | 0.474046 | 0.240893 |
| mix_n8_seed29 | ood | 3107 | 0.486874 | 0.237718 |
| td_n8_seed29 | dev | 2721 | 0.243507 | 0.157014 |
| td_n8_seed29 | calibration | 2606 | 0.609222 | 0.308797 |
| td_n8_seed29 | test | 2716 | 0.429004 | 0.233537 |
| td_n8_seed29 | ood | 3107 | 0.335584 | 0.178860 |

## Probability scores by task

| Arm | Split | Task | Questions | NLL | Vector Brier |
| --- | --- | --- | --- | --- | --- |
| mc_seed17 | dev | maze | 1956 | 0.255963 | 0.172312 |
| mc_seed17 | dev | snake | 147 | 0.416837 | 0.220953 |
| mc_seed17 | dev | shooting | 618 | 0.062746 | 0.013132 |
| mc_seed17 | calibration | maze | 1804 | 0.220243 | 0.143039 |
| mc_seed17 | calibration | snake | 132 | 1.062825 | 0.617759 |
| mc_seed17 | calibration | shooting | 670 | 0.445726 | 0.201925 |
| mc_seed17 | test | maze | 1957 | 0.073834 | 0.046731 |
| mc_seed17 | test | snake | 142 | 1.188867 | 0.644957 |
| mc_seed17 | test | shooting | 617 | 0.282180 | 0.123674 |
| mc_seed17 | ood | maze | 2618 | 0.503383 | 0.258425 |
| mc_seed17 | ood | snake | 151 | 0.744412 | 0.479639 |
| mc_seed17 | ood | shooting | 338 | 0.120891 | 0.041582 |
| mix_n3_seed17 | dev | maze | 1956 | 0.251091 | 0.166442 |
| mix_n3_seed17 | dev | snake | 147 | 0.606925 | 0.315028 |
| mix_n3_seed17 | dev | shooting | 618 | 0.046667 | 0.005024 |
| mix_n3_seed17 | calibration | maze | 1804 | 0.327594 | 0.193191 |
| mix_n3_seed17 | calibration | snake | 132 | 0.675525 | 0.412158 |
| mix_n3_seed17 | calibration | shooting | 670 | 0.385621 | 0.186838 |
| mix_n3_seed17 | test | maze | 1957 | 0.045197 | 0.031108 |
| mix_n3_seed17 | test | snake | 142 | 1.139612 | 0.690267 |
| mix_n3_seed17 | test | shooting | 617 | 0.236591 | 0.109867 |
| mix_n3_seed17 | ood | maze | 2618 | 0.583284 | 0.286370 |
| mix_n3_seed17 | ood | snake | 151 | 0.610352 | 0.389220 |
| mix_n3_seed17 | ood | shooting | 338 | 0.084204 | 0.026091 |
| mix_n8_seed17 | dev | maze | 1956 | 0.222888 | 0.145508 |
| mix_n8_seed17 | dev | snake | 147 | 0.506811 | 0.275173 |
| mix_n8_seed17 | dev | shooting | 618 | 0.043240 | 0.004247 |
| mix_n8_seed17 | calibration | maze | 1804 | 0.365881 | 0.209260 |
| mix_n8_seed17 | calibration | snake | 132 | 0.589309 | 0.372770 |
| mix_n8_seed17 | calibration | shooting | 670 | 0.382088 | 0.186209 |
| mix_n8_seed17 | test | maze | 1957 | 0.039509 | 0.024557 |
| mix_n8_seed17 | test | snake | 142 | 1.044673 | 0.668782 |
| mix_n8_seed17 | test | shooting | 617 | 0.231879 | 0.109062 |
| mix_n8_seed17 | ood | maze | 2618 | 0.698732 | 0.334439 |
| mix_n8_seed17 | ood | snake | 151 | 0.433430 | 0.280397 |
| mix_n8_seed17 | ood | shooting | 338 | 0.078436 | 0.025054 |
| td_n8_seed17 | dev | maze | 1956 | 0.270508 | 0.163912 |
| td_n8_seed17 | dev | snake | 147 | 0.579800 | 0.373250 |
| td_n8_seed17 | dev | shooting | 618 | 0.032776 | 0.002927 |
| td_n8_seed17 | calibration | maze | 1804 | 0.065242 | 0.024268 |
| td_n8_seed17 | calibration | snake | 132 | 0.724259 | 0.431756 |
| td_n8_seed17 | calibration | shooting | 670 | 0.397876 | 0.188266 |
| td_n8_seed17 | test | maze | 1957 | 0.097417 | 0.053183 |
| td_n8_seed17 | test | snake | 142 | 0.966614 | 0.553904 |
| td_n8_seed17 | test | shooting | 617 | 0.238380 | 0.109720 |
| td_n8_seed17 | ood | maze | 2618 | 0.442501 | 0.284836 |
| td_n8_seed17 | ood | snake | 151 | 0.604342 | 0.390220 |
| td_n8_seed17 | ood | shooting | 338 | 0.068153 | 0.023689 |
| mc_seed29 | dev | maze | 1956 | 0.331440 | 0.189272 |
| mc_seed29 | dev | snake | 147 | 0.224165 | 0.091871 |
| mc_seed29 | dev | shooting | 618 | 0.014734 | 0.000545 |
| mc_seed29 | calibration | maze | 1804 | 0.376646 | 0.196096 |
| mc_seed29 | calibration | snake | 132 | 1.221934 | 0.697239 |
| mc_seed29 | calibration | shooting | 670 | 0.475504 | 0.191275 |
| mc_seed29 | test | maze | 1957 | 0.051588 | 0.037454 |
| mc_seed29 | test | snake | 142 | 1.162697 | 0.557263 |
| mc_seed29 | test | shooting | 617 | 0.279807 | 0.111907 |
| mc_seed29 | ood | maze | 2618 | 0.704707 | 0.316922 |
| mc_seed29 | ood | snake | 151 | 0.669164 | 0.377174 |
| mc_seed29 | ood | shooting | 338 | 0.070004 | 0.023844 |
| mix_n3_seed29 | dev | maze | 1956 | 0.331580 | 0.179183 |
| mix_n3_seed29 | dev | snake | 147 | 0.737247 | 0.477540 |
| mix_n3_seed29 | dev | shooting | 618 | 0.015241 | 0.000513 |
| mix_n3_seed29 | calibration | maze | 1804 | 0.470218 | 0.236066 |
| mix_n3_seed29 | calibration | snake | 132 | 0.620664 | 0.377392 |
| mix_n3_seed29 | calibration | shooting | 670 | 0.446323 | 0.190039 |
| mix_n3_seed29 | test | maze | 1957 | 0.030684 | 0.017446 |
| mix_n3_seed29 | test | snake | 142 | 1.135135 | 0.671947 |
| mix_n3_seed29 | test | shooting | 617 | 0.265350 | 0.111241 |
| mix_n3_seed29 | ood | maze | 2618 | 0.645414 | 0.300315 |
| mix_n3_seed29 | ood | snake | 151 | 0.416540 | 0.260514 |
| mix_n3_seed29 | ood | shooting | 338 | 0.067356 | 0.023612 |
| mix_n8_seed29 | dev | maze | 1956 | 0.377769 | 0.183758 |
| mix_n8_seed29 | dev | snake | 147 | 0.229488 | 0.102981 |
| mix_n8_seed29 | dev | shooting | 618 | 0.020310 | 0.000912 |
| mix_n8_seed29 | calibration | maze | 1804 | 0.292323 | 0.165652 |
| mix_n8_seed29 | calibration | snake | 132 | 1.087334 | 0.610301 |
| mix_n8_seed29 | calibration | shooting | 670 | 0.421894 | 0.188888 |
| mix_n8_seed29 | test | maze | 1957 | 0.060616 | 0.040650 |
| mix_n8_seed29 | test | snake | 142 | 1.106462 | 0.571213 |
| mix_n8_seed29 | test | shooting | 617 | 0.255060 | 0.110817 |
| mix_n8_seed29 | ood | maze | 2618 | 0.809247 | 0.338186 |
| mix_n8_seed29 | ood | snake | 151 | 0.578345 | 0.350723 |
| mix_n8_seed29 | ood | shooting | 338 | 0.073029 | 0.024245 |
| td_n8_seed29 | dev | maze | 1956 | 0.204038 | 0.145640 |
| td_n8_seed29 | dev | snake | 147 | 0.518917 | 0.325280 |
| td_n8_seed29 | dev | shooting | 618 | 0.007567 | 0.000121 |
| td_n8_seed29 | calibration | maze | 1804 | 0.550613 | 0.244459 |
| td_n8_seed29 | calibration | snake | 132 | 0.789826 | 0.490561 |
| td_n8_seed29 | calibration | shooting | 670 | 0.487228 | 0.191372 |
| td_n8_seed29 | test | maze | 1957 | 0.073058 | 0.047774 |
| td_n8_seed29 | test | snake | 142 | 0.927429 | 0.540937 |
| td_n8_seed29 | test | shooting | 617 | 0.286526 | 0.111900 |
| td_n8_seed29 | ood | maze | 2618 | 0.470883 | 0.228302 |
| td_n8_seed29 | ood | snake | 151 | 0.473102 | 0.284947 |
| td_n8_seed29 | ood | shooting | 338 | 0.062766 | 0.023330 |

## Game success by variant

| Arm | Split/task/variant | Successes / episodes |
| --- | --- | --- |
| mc_seed17 | calibration/maze/maze16 | 1/2 |
| mc_seed17 | calibration/maze/maze50 | 0/2 |
| mc_seed17 | calibration/maze/maze8 | 5/6 |
| mc_seed17 | calibration/shooting/doom_basic | 4/6 |
| mc_seed17 | calibration/shooting/doom_predict_position | 0/6 |
| mc_seed17 | calibration/snake/snake12 | 1/2 |
| mc_seed17 | calibration/snake/snake8 | 6/6 |
| mc_seed17 | dev/maze/maze16 | 0/2 |
| mc_seed17 | dev/maze/maze50 | 0/2 |
| mc_seed17 | dev/maze/maze8 | 6/6 |
| mc_seed17 | dev/shooting/doom_basic | 1/6 |
| mc_seed17 | dev/shooting/doom_predict_position | 1/6 |
| mc_seed17 | dev/snake/snake12 | 0/2 |
| mc_seed17 | dev/snake/snake8 | 4/6 |
| mc_seed17 | ood/maze/maze16 | 0/2 |
| mc_seed17 | ood/maze/maze50 | 0/2 |
| mc_seed17 | ood/maze/maze8 | 0/6 |
| mc_seed17 | ood/shooting/doom_basic | 0/6 |
| mc_seed17 | ood/shooting/doom_predict_position | 1/6 |
| mc_seed17 | ood/snake/snake12 | 0/2 |
| mc_seed17 | ood/snake/snake8 | 3/6 |
| mc_seed17 | test/maze/maze16 | 0/2 |
| mc_seed17 | test/maze/maze50 | 0/2 |
| mc_seed17 | test/maze/maze8 | 4/6 |
| mc_seed17 | test/shooting/doom_basic | 4/6 |
| mc_seed17 | test/shooting/doom_predict_position | 0/6 |
| mc_seed17 | test/snake/snake12 | 0/2 |
| mc_seed17 | test/snake/snake8 | 5/6 |
| mc_seed17 | train/maze/maze16 | 0/4 |
| mc_seed17 | train/maze/maze50 | 0/4 |
| mc_seed17 | train/maze/maze8 | 18/24 |
| mc_seed17 | train/shooting/doom_basic | 9/24 |
| mc_seed17 | train/shooting/doom_predict_position | 1/24 |
| mc_seed17 | train/snake/snake12 | 1/4 |
| mc_seed17 | train/snake/snake8 | 20/24 |
| mix_n3_seed17 | calibration/maze/maze16 | 1/2 |
| mix_n3_seed17 | calibration/maze/maze50 | 0/2 |
| mix_n3_seed17 | calibration/maze/maze8 | 4/6 |
| mix_n3_seed17 | calibration/shooting/doom_basic | 1/6 |
| mix_n3_seed17 | calibration/shooting/doom_predict_position | 0/6 |
| mix_n3_seed17 | calibration/snake/snake12 | 1/2 |
| mix_n3_seed17 | calibration/snake/snake8 | 5/6 |
| mix_n3_seed17 | dev/maze/maze16 | 0/2 |
| mix_n3_seed17 | dev/maze/maze50 | 0/2 |
| mix_n3_seed17 | dev/maze/maze8 | 5/6 |
| mix_n3_seed17 | dev/shooting/doom_basic | 1/6 |
| mix_n3_seed17 | dev/shooting/doom_predict_position | 0/6 |
| mix_n3_seed17 | dev/snake/snake12 | 1/2 |
| mix_n3_seed17 | dev/snake/snake8 | 6/6 |
| mix_n3_seed17 | ood/maze/maze16 | 0/2 |
| mix_n3_seed17 | ood/maze/maze50 | 0/2 |
| mix_n3_seed17 | ood/maze/maze8 | 1/6 |
| mix_n3_seed17 | ood/shooting/doom_basic | 0/6 |
| mix_n3_seed17 | ood/shooting/doom_predict_position | 0/6 |
| mix_n3_seed17 | ood/snake/snake12 | 0/2 |
| mix_n3_seed17 | ood/snake/snake8 | 2/6 |
| mix_n3_seed17 | test/maze/maze16 | 0/2 |
| mix_n3_seed17 | test/maze/maze50 | 0/2 |
| mix_n3_seed17 | test/maze/maze8 | 4/6 |
| mix_n3_seed17 | test/shooting/doom_basic | 0/6 |
| mix_n3_seed17 | test/shooting/doom_predict_position | 1/6 |
| mix_n3_seed17 | test/snake/snake12 | 1/2 |
| mix_n3_seed17 | test/snake/snake8 | 6/6 |
| mix_n3_seed17 | train/maze/maze16 | 0/4 |
| mix_n3_seed17 | train/maze/maze50 | 0/4 |
| mix_n3_seed17 | train/maze/maze8 | 18/24 |
| mix_n3_seed17 | train/shooting/doom_basic | 1/24 |
| mix_n3_seed17 | train/shooting/doom_predict_position | 1/24 |
| mix_n3_seed17 | train/snake/snake12 | 1/4 |
| mix_n3_seed17 | train/snake/snake8 | 21/24 |
| mix_n8_seed17 | calibration/maze/maze16 | 1/2 |
| mix_n8_seed17 | calibration/maze/maze50 | 0/2 |
| mix_n8_seed17 | calibration/maze/maze8 | 4/6 |
| mix_n8_seed17 | calibration/shooting/doom_basic | 1/6 |
| mix_n8_seed17 | calibration/shooting/doom_predict_position | 0/6 |
| mix_n8_seed17 | calibration/snake/snake12 | 2/2 |
| mix_n8_seed17 | calibration/snake/snake8 | 6/6 |
| mix_n8_seed17 | dev/maze/maze16 | 2/2 |
| mix_n8_seed17 | dev/maze/maze50 | 0/2 |
| mix_n8_seed17 | dev/maze/maze8 | 5/6 |
| mix_n8_seed17 | dev/shooting/doom_basic | 1/6 |
| mix_n8_seed17 | dev/shooting/doom_predict_position | 0/6 |
| mix_n8_seed17 | dev/snake/snake12 | 1/2 |
| mix_n8_seed17 | dev/snake/snake8 | 5/6 |
| mix_n8_seed17 | ood/maze/maze16 | 0/2 |
| mix_n8_seed17 | ood/maze/maze50 | 0/2 |
| mix_n8_seed17 | ood/maze/maze8 | 0/6 |
| mix_n8_seed17 | ood/shooting/doom_basic | 0/6 |
| mix_n8_seed17 | ood/shooting/doom_predict_position | 0/6 |
| mix_n8_seed17 | ood/snake/snake12 | 0/2 |
| mix_n8_seed17 | ood/snake/snake8 | 5/6 |
| mix_n8_seed17 | test/maze/maze16 | 0/2 |
| mix_n8_seed17 | test/maze/maze50 | 0/2 |
| mix_n8_seed17 | test/maze/maze8 | 5/6 |
| mix_n8_seed17 | test/shooting/doom_basic | 2/6 |
| mix_n8_seed17 | test/shooting/doom_predict_position | 0/6 |
| mix_n8_seed17 | test/snake/snake12 | 1/2 |
| mix_n8_seed17 | test/snake/snake8 | 6/6 |
| mix_n8_seed17 | train/maze/maze16 | 0/4 |
| mix_n8_seed17 | train/maze/maze50 | 0/4 |
| mix_n8_seed17 | train/maze/maze8 | 18/24 |
| mix_n8_seed17 | train/shooting/doom_basic | 4/24 |
| mix_n8_seed17 | train/shooting/doom_predict_position | 1/24 |
| mix_n8_seed17 | train/snake/snake12 | 1/4 |
| mix_n8_seed17 | train/snake/snake8 | 18/24 |
| td_n8_seed17 | calibration/maze/maze16 | 1/2 |
| td_n8_seed17 | calibration/maze/maze50 | 0/2 |
| td_n8_seed17 | calibration/maze/maze8 | 3/6 |
| td_n8_seed17 | calibration/shooting/doom_basic | 0/6 |
| td_n8_seed17 | calibration/shooting/doom_predict_position | 0/6 |
| td_n8_seed17 | calibration/snake/snake12 | 0/2 |
| td_n8_seed17 | calibration/snake/snake8 | 3/6 |
| td_n8_seed17 | dev/maze/maze16 | 2/2 |
| td_n8_seed17 | dev/maze/maze50 | 0/2 |
| td_n8_seed17 | dev/maze/maze8 | 5/6 |
| td_n8_seed17 | dev/shooting/doom_basic | 1/6 |
| td_n8_seed17 | dev/shooting/doom_predict_position | 0/6 |
| td_n8_seed17 | dev/snake/snake12 | 0/2 |
| td_n8_seed17 | dev/snake/snake8 | 5/6 |
| td_n8_seed17 | ood/maze/maze16 | 0/2 |
| td_n8_seed17 | ood/maze/maze50 | 0/2 |
| td_n8_seed17 | ood/maze/maze8 | 0/6 |
| td_n8_seed17 | ood/shooting/doom_basic | 0/6 |
| td_n8_seed17 | ood/shooting/doom_predict_position | 0/6 |
| td_n8_seed17 | ood/snake/snake12 | 0/2 |
| td_n8_seed17 | ood/snake/snake8 | 1/6 |
| td_n8_seed17 | test/maze/maze16 | 0/2 |
| td_n8_seed17 | test/maze/maze50 | 0/2 |
| td_n8_seed17 | test/maze/maze8 | 5/6 |
| td_n8_seed17 | test/shooting/doom_basic | 0/6 |
| td_n8_seed17 | test/shooting/doom_predict_position | 0/6 |
| td_n8_seed17 | test/snake/snake12 | 0/2 |
| td_n8_seed17 | test/snake/snake8 | 4/6 |
| td_n8_seed17 | train/maze/maze16 | 0/4 |
| td_n8_seed17 | train/maze/maze50 | 0/4 |
| td_n8_seed17 | train/maze/maze8 | 19/24 |
| td_n8_seed17 | train/shooting/doom_basic | 1/24 |
| td_n8_seed17 | train/shooting/doom_predict_position | 1/24 |
| td_n8_seed17 | train/snake/snake12 | 1/4 |
| td_n8_seed17 | train/snake/snake8 | 17/24 |
| mc_seed29 | calibration/maze/maze16 | 1/2 |
| mc_seed29 | calibration/maze/maze50 | 0/2 |
| mc_seed29 | calibration/maze/maze8 | 5/6 |
| mc_seed29 | calibration/shooting/doom_basic | 0/6 |
| mc_seed29 | calibration/shooting/doom_predict_position | 0/6 |
| mc_seed29 | calibration/snake/snake12 | 0/2 |
| mc_seed29 | calibration/snake/snake8 | 3/6 |
| mc_seed29 | dev/maze/maze16 | 2/2 |
| mc_seed29 | dev/maze/maze50 | 0/2 |
| mc_seed29 | dev/maze/maze8 | 4/6 |
| mc_seed29 | dev/shooting/doom_basic | 1/6 |
| mc_seed29 | dev/shooting/doom_predict_position | 0/6 |
| mc_seed29 | dev/snake/snake12 | 1/2 |
| mc_seed29 | dev/snake/snake8 | 4/6 |
| mc_seed29 | ood/maze/maze16 | 0/2 |
| mc_seed29 | ood/maze/maze50 | 0/2 |
| mc_seed29 | ood/maze/maze8 | 2/6 |
| mc_seed29 | ood/shooting/doom_basic | 0/6 |
| mc_seed29 | ood/shooting/doom_predict_position | 0/6 |
| mc_seed29 | ood/snake/snake12 | 2/2 |
| mc_seed29 | ood/snake/snake8 | 1/6 |
| mc_seed29 | test/maze/maze16 | 0/2 |
| mc_seed29 | test/maze/maze50 | 0/2 |
| mc_seed29 | test/maze/maze8 | 4/6 |
| mc_seed29 | test/shooting/doom_basic | 0/6 |
| mc_seed29 | test/shooting/doom_predict_position | 0/6 |
| mc_seed29 | test/snake/snake12 | 2/2 |
| mc_seed29 | test/snake/snake8 | 6/6 |
| mc_seed29 | train/maze/maze16 | 1/4 |
| mc_seed29 | train/maze/maze50 | 0/4 |
| mc_seed29 | train/maze/maze8 | 18/24 |
| mc_seed29 | train/shooting/doom_basic | 1/24 |
| mc_seed29 | train/shooting/doom_predict_position | 0/24 |
| mc_seed29 | train/snake/snake12 | 0/4 |
| mc_seed29 | train/snake/snake8 | 19/24 |
| mix_n3_seed29 | calibration/maze/maze16 | 2/2 |
| mix_n3_seed29 | calibration/maze/maze50 | 0/2 |
| mix_n3_seed29 | calibration/maze/maze8 | 6/6 |
| mix_n3_seed29 | calibration/shooting/doom_basic | 2/6 |
| mix_n3_seed29 | calibration/shooting/doom_predict_position | 0/6 |
| mix_n3_seed29 | calibration/snake/snake12 | 2/2 |
| mix_n3_seed29 | calibration/snake/snake8 | 6/6 |
| mix_n3_seed29 | dev/maze/maze16 | 1/2 |
| mix_n3_seed29 | dev/maze/maze50 | 0/2 |
| mix_n3_seed29 | dev/maze/maze8 | 6/6 |
| mix_n3_seed29 | dev/shooting/doom_basic | 1/6 |
| mix_n3_seed29 | dev/shooting/doom_predict_position | 0/6 |
| mix_n3_seed29 | dev/snake/snake12 | 0/2 |
| mix_n3_seed29 | dev/snake/snake8 | 5/6 |
| mix_n3_seed29 | ood/maze/maze16 | 0/2 |
| mix_n3_seed29 | ood/maze/maze50 | 0/2 |
| mix_n3_seed29 | ood/maze/maze8 | 0/6 |
| mix_n3_seed29 | ood/shooting/doom_basic | 0/6 |
| mix_n3_seed29 | ood/shooting/doom_predict_position | 0/6 |
| mix_n3_seed29 | ood/snake/snake12 | 0/2 |
| mix_n3_seed29 | ood/snake/snake8 | 3/6 |
| mix_n3_seed29 | test/maze/maze16 | 1/2 |
| mix_n3_seed29 | test/maze/maze50 | 0/2 |
| mix_n3_seed29 | test/maze/maze8 | 4/6 |
| mix_n3_seed29 | test/shooting/doom_basic | 2/6 |
| mix_n3_seed29 | test/shooting/doom_predict_position | 0/6 |
| mix_n3_seed29 | test/snake/snake12 | 0/2 |
| mix_n3_seed29 | test/snake/snake8 | 6/6 |
| mix_n3_seed29 | train/maze/maze16 | 1/4 |
| mix_n3_seed29 | train/maze/maze50 | 0/4 |
| mix_n3_seed29 | train/maze/maze8 | 22/24 |
| mix_n3_seed29 | train/shooting/doom_basic | 7/24 |
| mix_n3_seed29 | train/shooting/doom_predict_position | 4/24 |
| mix_n3_seed29 | train/snake/snake12 | 1/4 |
| mix_n3_seed29 | train/snake/snake8 | 20/24 |
| mix_n8_seed29 | calibration/maze/maze16 | 2/2 |
| mix_n8_seed29 | calibration/maze/maze50 | 0/2 |
| mix_n8_seed29 | calibration/maze/maze8 | 5/6 |
| mix_n8_seed29 | calibration/shooting/doom_basic | 3/6 |
| mix_n8_seed29 | calibration/shooting/doom_predict_position | 1/6 |
| mix_n8_seed29 | calibration/snake/snake12 | 0/2 |
| mix_n8_seed29 | calibration/snake/snake8 | 5/6 |
| mix_n8_seed29 | dev/maze/maze16 | 2/2 |
| mix_n8_seed29 | dev/maze/maze50 | 0/2 |
| mix_n8_seed29 | dev/maze/maze8 | 6/6 |
| mix_n8_seed29 | dev/shooting/doom_basic | 4/6 |
| mix_n8_seed29 | dev/shooting/doom_predict_position | 0/6 |
| mix_n8_seed29 | dev/snake/snake12 | 0/2 |
| mix_n8_seed29 | dev/snake/snake8 | 5/6 |
| mix_n8_seed29 | ood/maze/maze16 | 0/2 |
| mix_n8_seed29 | ood/maze/maze50 | 0/2 |
| mix_n8_seed29 | ood/maze/maze8 | 3/6 |
| mix_n8_seed29 | ood/shooting/doom_basic | 2/6 |
| mix_n8_seed29 | ood/shooting/doom_predict_position | 0/6 |
| mix_n8_seed29 | ood/snake/snake12 | 0/2 |
| mix_n8_seed29 | ood/snake/snake8 | 1/6 |
| mix_n8_seed29 | test/maze/maze16 | 0/2 |
| mix_n8_seed29 | test/maze/maze50 | 0/2 |
| mix_n8_seed29 | test/maze/maze8 | 6/6 |
| mix_n8_seed29 | test/shooting/doom_basic | 5/6 |
| mix_n8_seed29 | test/shooting/doom_predict_position | 0/6 |
| mix_n8_seed29 | test/snake/snake12 | 1/2 |
| mix_n8_seed29 | test/snake/snake8 | 6/6 |
| mix_n8_seed29 | train/maze/maze16 | 3/4 |
| mix_n8_seed29 | train/maze/maze50 | 0/4 |
| mix_n8_seed29 | train/maze/maze8 | 19/24 |
| mix_n8_seed29 | train/shooting/doom_basic | 11/24 |
| mix_n8_seed29 | train/shooting/doom_predict_position | 1/24 |
| mix_n8_seed29 | train/snake/snake12 | 2/4 |
| mix_n8_seed29 | train/snake/snake8 | 21/24 |
| td_n8_seed29 | calibration/maze/maze16 | 2/2 |
| td_n8_seed29 | calibration/maze/maze50 | 0/2 |
| td_n8_seed29 | calibration/maze/maze8 | 6/6 |
| td_n8_seed29 | calibration/shooting/doom_basic | 0/6 |
| td_n8_seed29 | calibration/shooting/doom_predict_position | 0/6 |
| td_n8_seed29 | calibration/snake/snake12 | 2/2 |
| td_n8_seed29 | calibration/snake/snake8 | 5/6 |
| td_n8_seed29 | dev/maze/maze16 | 1/2 |
| td_n8_seed29 | dev/maze/maze50 | 0/2 |
| td_n8_seed29 | dev/maze/maze8 | 6/6 |
| td_n8_seed29 | dev/shooting/doom_basic | 1/6 |
| td_n8_seed29 | dev/shooting/doom_predict_position | 0/6 |
| td_n8_seed29 | dev/snake/snake12 | 1/2 |
| td_n8_seed29 | dev/snake/snake8 | 4/6 |
| td_n8_seed29 | ood/maze/maze16 | 0/2 |
| td_n8_seed29 | ood/maze/maze50 | 0/2 |
| td_n8_seed29 | ood/maze/maze8 | 2/6 |
| td_n8_seed29 | ood/shooting/doom_basic | 0/6 |
| td_n8_seed29 | ood/shooting/doom_predict_position | 0/6 |
| td_n8_seed29 | ood/snake/snake12 | 0/2 |
| td_n8_seed29 | ood/snake/snake8 | 4/6 |
| td_n8_seed29 | test/maze/maze16 | 0/2 |
| td_n8_seed29 | test/maze/maze50 | 0/2 |
| td_n8_seed29 | test/maze/maze8 | 5/6 |
| td_n8_seed29 | test/shooting/doom_basic | 0/6 |
| td_n8_seed29 | test/shooting/doom_predict_position | 0/6 |
| td_n8_seed29 | test/snake/snake12 | 2/2 |
| td_n8_seed29 | test/snake/snake8 | 6/6 |
| td_n8_seed29 | train/maze/maze16 | 0/4 |
| td_n8_seed29 | train/maze/maze50 | 0/4 |
| td_n8_seed29 | train/maze/maze8 | 18/24 |
| td_n8_seed29 | train/shooting/doom_basic | 1/24 |
| td_n8_seed29 | train/shooting/doom_predict_position | 0/24 |
| td_n8_seed29 | train/snake/snake12 | 1/4 |
| td_n8_seed29 | train/snake/snake8 | 19/24 |
| Initial Q | calibration/maze/maze16 | 2/2 |
| Initial Q | calibration/maze/maze50 | 0/2 |
| Initial Q | calibration/maze/maze8 | 6/6 |
| Initial Q | calibration/shooting/doom_basic | 1/6 |
| Initial Q | calibration/shooting/doom_predict_position | 0/6 |
| Initial Q | calibration/snake/snake12 | 0/2 |
| Initial Q | calibration/snake/snake8 | 5/6 |
| Initial Q | dev/maze/maze16 | 1/2 |
| Initial Q | dev/maze/maze50 | 0/2 |
| Initial Q | dev/maze/maze8 | 5/6 |
| Initial Q | dev/shooting/doom_basic | 0/6 |
| Initial Q | dev/shooting/doom_predict_position | 0/6 |
| Initial Q | dev/snake/snake12 | 1/2 |
| Initial Q | dev/snake/snake8 | 5/6 |
| Initial Q | ood/maze/maze16 | 0/2 |
| Initial Q | ood/maze/maze50 | 0/2 |
| Initial Q | ood/maze/maze8 | 3/6 |
| Initial Q | ood/shooting/doom_basic | 1/6 |
| Initial Q | ood/shooting/doom_predict_position | 0/6 |
| Initial Q | ood/snake/snake12 | 0/2 |
| Initial Q | ood/snake/snake8 | 3/6 |
| Initial Q | test/maze/maze16 | 0/2 |
| Initial Q | test/maze/maze50 | 0/2 |
| Initial Q | test/maze/maze8 | 4/6 |
| Initial Q | test/shooting/doom_basic | 1/6 |
| Initial Q | test/shooting/doom_predict_position | 0/6 |
| Initial Q | test/snake/snake12 | 1/2 |
| Initial Q | test/snake/snake8 | 5/6 |
| Initial Q | train/maze/maze16 | 0/4 |
| Initial Q | train/maze/maze50 | 0/4 |
| Initial Q | train/maze/maze8 | 20/24 |
| Initial Q | train/shooting/doom_basic | 4/24 |
| Initial Q | train/shooting/doom_predict_position | 2/24 |
| Initial Q | train/snake/snake12 | 1/4 |
| Initial Q | train/snake/snake8 | 21/24 |

## Condition task-macro success: seed mean [min, max]

| Condition | Split | Success rate |
| --- | --- | --- |
| mc | train | 0.4690 [0.4311, 0.5069] |
| mc | dev | 0.4292 [0.4222, 0.4361] |
| mc | calibration | 0.4639 [0.3250, 0.6028] |
| mc | test | 0.4597 [0.4528, 0.4667] |
| mc | ood | 0.1722 [0.1528, 0.1917] |
| mix_n3 | train | 0.5146 [0.4633, 0.5660] |
| mix_n3 | dev | 0.4778 [0.4694, 0.4861] |
| mix_n3 | calibration | 0.5500 [0.4444, 0.6556] |
| mix_n3 | test | 0.4625 [0.4528, 0.4722] |
| mix_n3 | ood | 0.1208 [0.1167, 0.1250] |
| mix_n8 | train | 0.5174 [0.4484, 0.5863] |
| mix_n8 | dev | 0.5486 [0.5111, 0.5861] |
| mix_n8 | calibration | 0.5403 [0.5278, 0.5528] |
| mix_n8 | test | 0.5722 [0.5139, 0.6306] |
| mix_n8 | ood | 0.2028 [0.1972, 0.2083] |
| td_n8 | train | 0.4293 [0.4261, 0.4325] |
| td_n8 | dev | 0.4694 [0.4694, 0.4694] |
| td_n8 | calibration | 0.4083 [0.2583, 0.5583] |
| td_n8 | test | 0.4167 [0.3333, 0.5000] |
| td_n8 | ood | 0.1375 [0.0417, 0.2333] |

## Fixed-policy probability scores: seed mean [min, max]

| Condition | Split/metric | Score |
| --- | --- | --- |
| mc | dev/nll | 0.2176 [0.1901, 0.2452] |
| mc | dev/vector_brier | 0.1147 [0.0939, 0.1355] |
| mc | calibration/nll | 0.6338 [0.5763, 0.6914] |
| mc | calibration/vector_brier | 0.3412 [0.3209, 0.3615] |
| mc | test/nll | 0.5065 [0.4980, 0.5150] |
| mc | test/vector_brier | 0.2537 [0.2355, 0.2718] |
| mc | ood/nll | 0.4688 [0.4562, 0.4813] |
| mc | ood/vector_brier | 0.2496 [0.2393, 0.2599] |
| mix_n3 | dev/nll | 0.3315 [0.3016, 0.3614] |
| mix_n3 | dev/vector_brier | 0.1906 [0.1622, 0.2191] |
| mix_n3 | calibration/nll | 0.4877 [0.4629, 0.5124] |
| mix_n3 | calibration/vector_brier | 0.2659 [0.2641, 0.2678] |
| mix_n3 | test/nll | 0.4754 [0.4738, 0.4771] |
| mix_n3 | test/vector_brier | 0.2720 [0.2669, 0.2771] |
| mix_n3 | ood/nll | 0.4012 [0.3764, 0.4259] |
| mix_n3 | ood/vector_brier | 0.2144 [0.1948, 0.2339] |
| mix_n8 | dev/nll | 0.2334 [0.2092, 0.2576] |
| mix_n8 | dev/vector_brier | 0.1188 [0.0959, 0.1416] |
| mix_n8 | calibration/nll | 0.5231 [0.4458, 0.6005] |
| mix_n8 | calibration/vector_brier | 0.2888 [0.2561, 0.3216] |
| mix_n8 | test/nll | 0.4564 [0.4387, 0.4740] |
| mix_n8 | test/vector_brier | 0.2542 [0.2409, 0.2675] |
| mix_n8 | ood/nll | 0.4452 [0.4035, 0.4869] |
| mix_n8 | ood/vector_brier | 0.2255 [0.2133, 0.2377] |
| td_n8 | dev/nll | 0.2689 [0.2435, 0.2944] |
| td_n8 | dev/vector_brier | 0.1685 [0.1570, 0.1800] |
| td_n8 | calibration/nll | 0.5025 [0.3958, 0.6092] |
| td_n8 | calibration/vector_brier | 0.2618 [0.2148, 0.3088] |
| td_n8 | test/nll | 0.4316 [0.4290, 0.4341] |
| td_n8 | test/vector_brier | 0.2362 [0.2335, 0.2389] |
| td_n8 | ood/nll | 0.3536 [0.3356, 0.3717] |
| td_n8 | ood/vector_brier | 0.2059 [0.1789, 0.2329] |

## Paired difference from MC: seed mean [min, max]

Positive game deltas favor the named condition.

| Condition | Split | Task-macro success delta |
| --- | --- | --- |
| mix_n3 | train | 0.0456 [-0.0437, 0.1349] |
| mix_n3 | dev | 0.0486 [0.0333, 0.0639] |
| mix_n3 | calibration | 0.0861 [-0.1583, 0.3306] |
| mix_n3 | test | 0.0028 [0.0000, 0.0056] |
| mix_n3 | ood | -0.0514 [-0.0667, -0.0361] |
| mix_n8 | train | 0.0484 [-0.0585, 0.1553] |
| mix_n8 | dev | 0.1194 [0.0889, 0.1500] |
| mix_n8 | calibration | 0.0764 [-0.0750, 0.2278] |
| mix_n8 | test | 0.1125 [0.0611, 0.1639] |
| mix_n8 | ood | 0.0306 [0.0056, 0.0556] |
| td_n8 | train | -0.0397 [-0.0809, 0.0015] |
| td_n8 | dev | 0.0403 [0.0333, 0.0472] |
| td_n8 | calibration | -0.0556 [-0.3444, 0.2333] |
| td_n8 | test | -0.0431 [-0.1194, 0.0333] |
| td_n8 | ood | -0.0347 [-0.1111, 0.0417] |

## Training and target-computation costs

| Arm | Selected / completed updates | Training seconds | Peak GPU GB | Online padded tokens | Target forwards | Target padded tokens | Target forward seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mc_seed17 | 100/200 | 1299.926970 | 16.802317 | 13373928 | 0 | 0 | 0 |
| mix_n3_seed17 | 100/200 | 1513.683243 | 19.187846 | 13373928 | 2581 | 10953589 | 212.5880002072081 |
| mix_n8_seed17 | 100/200 | 1494.300502 | 19.187846 | 13373928 | 2165 | 9823890 | 192.2468266217038 |
| td_n8_seed17 | 200/200 | 1503.598898 | 19.187846 | 13373928 | 2165 | 9823890 | 191.74686690140516 |
| mc_seed29 | 100/200 | 1318.420816 | 16.821417 | 13614323 | 0 | 0 | 0 |
| mix_n3_seed29 | 100/200 | 1531.272303 | 19.206946 | 13614323 | 2595 | 11107257 | 214.9742742702365 |
| mix_n8_seed29 | 100/200 | 1504.511073 | 19.206946 | 13614323 | 2167 | 9772651 | 189.77555201482028 |
| td_n8_seed29 | 200/200 | 1512.942062 | 19.206946 | 13614323 | 2167 | 9772651 | 190.40170729719102 |

## Verification

The JSON report retains source hashes, checkpoint/config matches, every declared configuration difference, controller differences, paired case counts, and policy-retention metrics. Missing results remain explicit.

- Probability metrics are recomputed against fixed observed terminal outcomes under the dataset policy.
- NLL uses finite logits and stable log-sum-exp without probability clipping; Brier sums both Boolean classes.
- Game metrics come from new completed Q-controller episodes, with every success and failure retained.
- Deltas are other minus MC, paired by seed and exact case ID; task macros equally weight tasks.
- Across-seed summaries give mean and observed range only, without a significance claim.
- All declared seeds are required for a condition aggregate; incomplete cells remain null.
- Training time and recorded token/target counters are computational costs, not probability or game metrics.
