# Qwen3-0.6B vs Qwen3-1.7B — NanoJev unified SFT comparison

Winning 1.7B arm: **hard_lr1e5**

## Offline dev/calibration/test/ood cross-entropy

| Split | 0.6B (hard_lr1e5) | 1.7B (hard_lr1e5) |
|---|---:|---:|
| dev | 0.6170 | 0.6281 |
| calibration | 0.6306 | 0.6378 |
| test | 0.6594 | 0.6637 |
| ood | 0.7634 | 0.7150 |

## Closed-loop 548-case rollout (test split)

| Task | 0.6B test | 1.7B test |
|---|---:|---:|
| Maze | 3/10 | 5/10 |
| Snake | 1/8 | 7/8 |
| Basic | 128/128 | 107/128 |
| Predict Position | 37/128 | 21/128 |

## Closed-loop 548-case rollout (ood split)

| Task | 0.6B ood | 1.7B ood |
|---|---:|---:|
| Maze | 0/10 | 0/10 |
| Snake | 1/8 | 6/8 |
| Basic | 118/128 | 49/128 |
| Predict Position | 12/128 | 17/128 |
