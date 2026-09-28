# NanoJev Benchmark 评测报告:Qwen3-0.6B vs Qwen3-1.7B 复现对比

本报告基于本地复现训练(数据/评测流程与官方 `unified-games-v1` 发布完全一致,均未调用 Jev 付费 API 或 Sonic Doom 专家权重,详见 `复现说明.md`),对比两种 backbone 规模在同一套 SFT 配方下的表现。

## 一、训练配置

两组模型均使用相同的复现配方(`TRAINING_RECIPE.md` / `docs/SONIC_PREDICT_POSITION.md`):

- 四臂对比:hard/soft 目标 × backbone lr {1e-5, 2e-5}(head lr = backbone lr × 10)
- 600 步、24 题/更新(Maze 8 + Snake 8 + Basic 4 + Predict Position 4)、BF16 + gradient checkpointing、种子 17
- Loss 权重:Maze 1/3、Snake 1/3、Basic 1/6、Predict Position 1/6
- 选型规则:四臂 + 未训练初始化基线中取 dev CE 最低者

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Backbone 来源 | 官方 `training_initialization`(已做过 Basic-only SFT) | 本地 `/home/dataset0/images/Qwen3-1.7B` 原始预训练权重 + 随机初始化决策头(从零热启动) |
| GPU | 1 张 A100(microbatch-questions=8) | 1 张 A100(microbatch-questions=4,更小以避免 OOM) |
| 训练耗时(胜出 arm) | 3947 秒(~66 分钟) | 5384 秒(~90 分钟) |
| 显存峰值(胜出 arm) | 16.86 GB | 35.02 GB |

**踩坑记录**:1.7B 用官方默认 `microbatch-questions=8` 会在训练中途(非首几步)OOM(某些较长候选序列把 34~36GB 峰值推过 40GB 上限),降到 `microbatch-questions=4`、`max-microbatch-tokens=16384` 并开启 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后稳定跑完全部 4 个 arm。

## 二、四臂 SFT 选型结果

### Qwen3-0.6B

| Candidate | Step | Dev CE |
|---|---:|---:|
| **hard_lr1e5** | 600 | **0.6170**(胜出) |
| soft_lr1e5 | 600 | 0.6260 |
| hard_lr2e5 | 600 | 0.6328 |
| soft_lr2e5 | 600 | 0.6435 |
| 未训练初始化基线 | — | 1.0683 |

### Qwen3-1.7B

| Candidate | Step | Dev CE |
|---|---:|---:|
| **hard_lr1e5** | 600 | **0.6281**(胜出) |
| soft_lr1e5 | 600 | 0.6333 |
| hard_lr2e5 | 500 | 0.6615 |
| soft_lr2e5 | 600 | 0.6980 |
| 未训练初始化基线 | — | 1.1413 |

两种规模下都是同一组超参(`hard` 目标、backbone lr=1e-5)胜出,与官方发布版本的选型结果一致。

## 三、离线交叉熵(胜出 arm,全五分区)

| Split | 0.6B | 1.7B | 差值(1.7B − 0.6B) |
|---|---:|---:|---:|
| dev | 0.6170 | 0.6281 | +0.0111(略差) |
| calibration | 0.6306 | 0.6378 | +0.0072(略差) |
| test | 0.6594 | 0.6637 | +0.0043(略差) |
| ood | 0.7634 | **0.7150** | **-0.0484(更好)** |

1.7B 在分布内(dev/calibration/test)略差,但在分布外(ood)明显更好——说明更大模型的泛化能力更强,但同样的 600 步/24题预算不足以让它在训练分布内充分收敛到和 0.6B 一样低的 loss。

## 四、闭环游戏评测(548 局,test+ood,q_greedy 控制器,epsilon=0.1,种子 17)

### test 分区(与 README 274 案例表可比)

| Task | 0.6B | 1.7B | 结论 |
|---|---:|---:|---|
| Maze | 3/10 | **5/10** | 1.7B 更好 |
| Snake | 1/8 | **7/8** | 1.7B 大幅改善 |
| ViZDoom Basic | **128/128** | 107/128 | 0.6B 明显更好 |
| ViZDoom Predict Position | **37/128** | 21/128 | 0.6B 明显更好 |

### ood 分区

| Task | 0.6B | 1.7B | 结论 |
|---|---:|---:|---|
| Maze | 0/10 | 0/10 | 打平(均失败) |
| Snake | 1/8 | **6/8** | 1.7B 大幅改善 |
| ViZDoom Basic | **118/128** | 49/128 | 0.6B 明显更好 |
| ViZDoom Predict Position | 12/128 | **17/128** | 1.7B 略好 |

## 五、与官方发布 NanoJev(0.6B,test 274 例)的对照

| 系统 | Maze | Snake | Basic | Predict Position |
|---|---:|---:|---:|---:|
| 官方 NanoJev(0.6B) | 4/10 | 8/8 | 128/128 | 27/128 |
| **本次复现 0.6B** | 3/10 | 1/8 | **128/128** | **37/128** |
| **本次复现 1.7B** | **5/10** | **7/8** | 107/128 | 21/128 |
| Jev(参照基线) | 7/10 | 8/8 | 56/128 | 11/128 |
| 未微调 Qwen3-0.6B | 2/10 | 0/8 | 56/128 | 11/128 |

## 六、关键结论

1. **导航类任务(Maze/Snake)受益于更大 backbone**:1.7B 在 Snake 上从 1/8 跃升到 7/8(ood 上 1/8→6/8),印证了此前对 0.6B 版本 Snake 失败模式的诊断——模型对"继续同方向走会不会撞墙"缺乏空间推理能力,而更大模型的表征能力缓解了这一缺陷。Maze 也有小幅提升。

2. **射击类任务(Basic/Predict Position)反而随模型变大而下降**,尤其 Basic 在 ood 上从 118/128 跌到 49/128。可能原因:
   - 同样 600 步 / 24 题预算下,1.7B 参数量增加近 3 倍但没有增加训练步数,四类任务共享一套固定 loss 权重(1/3,1/3,1/6,1/6),对于收敛更慢的更大模型,射击任务的权重占比小(1/6+1/6=1/3),可能没被充分优化。
   - 离线 CE 也反映出 1.7B 在 dev/test 上并未低于 0.6B,说明模型尚未在训练预算内充分收敛,而 Basic/Predict Position 的闭环胜负对策略微小偏差非常敏感。

3. **没有"越大越好"的简单结论**:1.7B 相比 0.6B 是有得有失,不是全面碾压。若要让 1.7B 真正发挥参数量优势,大概率需要更多训练步数,或针对四类任务的收敛速度差异调整采样/loss 权重,而不是直接沿用为 0.6B 调好的超参数配方。

## 七、复现产物路径

| 内容 | 路径 |
|---|---|
| 0.6B 四臂训练输出 | `/tmp/nanojev_runs/{hard_lr1e5,hard_lr2e5,soft_lr1e5,soft_lr2e5}/` |
| 1.7B 四臂训练输出 | `/tmp/nanojev_runs/qwen3_1_7b/{hard_lr1e5,hard_lr2e5,soft_lr1e5,soft_lr2e5}/` |
| 1.7B 初始化 checkpoint(随机决策头+预训练 backbone) | `/tmp/nanojev_runs/init_qwen3_1_7b/` |
| 0.6B 闭环评测 548 局 | `/tmp/nanojev_runs/eval/hard_lr1e5_test_ood_episodes.jsonl` |
| 1.7B 闭环评测 548 局 | `/tmp/nanojev_runs/qwen3_1_7b/eval/hard_lr1e5_test_ood_episodes.jsonl` |
| 训练队列脚本 | `run_full_sft_reproduction.sh`、`run_qwen3_1_7b_queue.sh` |
| 初始化 checkpoint 构建脚本 | `scripts_local/build_fresh_init_checkpoint.py` |
| 原始对比数据(机器生成) | `qwen3_1_7b_vs_0_6b_comparison.md` |

⚠️ 注意:`/tmp/nanojev_runs/` 下所有训练与评测产物存放在 `/tmp`(非持久分区),机器重启会被清空,如需长期保留需另行拷贝到 `/home` 或其他持久存储。
