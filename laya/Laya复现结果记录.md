# Laya 复现结果记录

> 记录本次端到端复现（架构实现→数据构建→RLCD训练→校准→评测→路由验证）的实际执行结果，
> 对应 `Laya项目复现执行方案.md` 的阶段 1-5。所有代码在 `repro/` 目录下。

## 环境

- uv venv, Python 3.12.7, torch 2.14.0(+cu126), transformers 5.17.0, huggingface_hub 1.32.0
- GPU: 6、7 号 A100-40GB（DDP，`CUDA_VISIBLE_DEVICES=6,7`）
- 基础模型：`models/ModernBERT-large`（421M，英语）、`models/mmBERT-base`（322M，多语言）
- 训练数据：`LocalLLaMA/typed-decisions`（1,200 训练 case / 400 测试 case，4 个工作流，教师软标签）

## 架构复现

直接复用仓库自带的 `laya.common`（`DecisionModel`、`build_sequence`、`render_options`、
`collate_items`、`proper_reward`、`temp_bucket`），未重新实现——这部分是"用现成的模型代码
+ 自己的数据/训练脚本训出权重"，而不是从零重写架构。

**踩坑记录**：`build_model(cfg, encoder_dir=..., pretrained=True)` 有两条分支：
`encoder_dir` 指向一个**存在的目录**时会走"只建架构、跳过权重初始化，调用方随后自己
`load_state_dict` 一个完整 checkpoint"的分支（`AutoModel.from_config` + `_no_init_weights()`），
而不是"从这个目录加载预训练权重"。误传 `encoder_dir=` 导致编码器权重全是未初始化的垃圾值，
前向传播直接产出 NaN。正确用法：只把路径塞进 `cfg["encoder"]`，不传 `encoder_dir`，
才会走 `AutoModel.from_pretrained(cfg["encoder"])` 真正加载权重。

## 训练结果

RLCD 训练（GRPO 风格策略梯度 + 严格适当评分规则奖励 + 满权重软交叉熵），
4 epoch，batch 8×2GPU×grad_accum4，lr 2.5e-5(编码器)/1e-4(决策头)，cosine 调度。

| 模型 | ce_loss (epoch0→3) | 训练耗时 |
|---|---|---|
| english (ModernBERT-large) | 1.18 → 0.91 | ~280s |
| multilingual (mmBERT-base) | 1.16 → 0.89 | ~220s |

**踩坑记录**：校准阶段把不同 mini-batch（选项数不同、padding 宽度不同）的 logits/target/mask
直接 `torch.stack` 导致尺寸不匹配崩溃（`RuntimeError: stack expects each tensor to be equal
size`）。修复：先按每条数据真实选项数切片，同一温度分桶内再统一 pad 到桶内最大宽度后再 stack。
english 训练当次崩在这一步，用 `repro/calibrate_and_export.py` 从已保存的 epoch-3 权重
（`checkpoint_latest/`）直接续跑校准+导出，无需重新训练；multilingual 训练时该 bug
已修复，一次跑通。

**次要问题（未修复，不影响产出）**：训练全部完成、`final/` checkpoint 已完整写盘后，
DDP 的收尾 `dist.barrier()` + `destroy_process_group()` 在 multilingual 一次跑中挂死
（一个 rank 卡在 NCCL busy-wait 不退出）。核对过文件完整性无误后直接 kill 掉了挂起的进程。

## 效果验证（typed-decisions test 集，2000 条决策，argmax 准确率）

| | 零样本（随机决策头） | 复现训练后 | 原文对照* |
|---|---|---|---|
| english 整体 | 0.401 | **0.703** | 0.362 → 0.766 |
| english choice/score/noul | 0.235 / 0.359 / 0.623 | 0.692 / 0.664 / 0.767 | — |
| multilingual 整体 | 0.246 | **0.710** | 0.352 → (未单独公布) |
| multilingual choice/score/noul | 0.103 / 0.156 / 0.507 | 0.678 / 0.656 / 0.812 | — |

\* 原文数字来自 `Laya项目深度技术解读报告.md` 第5节，是官方多次迭代、超参搜索后的结果；
本次复现只跑了一轮、无超参搜索，量级吻合（"零样本接近随机→微调后大幅提升"），
绝对数值有差距在预期内。

## 校准（温度拟合）结果 —— 一个值得记录的反直觉发现

在 400 条**训练时从未见过**的校准切片上，按 `(问题类型, 选项数)` 分桶拟合温度：

| bucket | 拟合温度 |
|---|---|
| noul:2 | 1.35 (english) / 1.32 (multilingual) |
| score:3-5 | 1.09 / 1.11 |
| choice:3-5 | 1.02 / 1.00 |

在 english test 集上对比校准前后：

| | 准确率 | 平均置信度 | ECE |
|---|---|---|---|
| 未校准 (T=1.0) | 0.7030 | 0.5828 | **0.1202** |
| 校准后 | 0.7030 | 0.5633 | **0.1397** |

**校准反而把 ECE 从 0.12 拉高到 0.14**，与原文"温度拟合显著改善校准"的结论方向相反。
分析：原文基础 checkpoint 是"训练完很久、从未做过温度拟合"的过度自信模型（原始 ECE
高达 0.46），校准空间很大；而本次复现训练本身就是拿 soft cross-entropy 直接对齐教师概率
分布训练出来的，模型出训练环境时已经比较"诚实"（未校准 ECE 只有 0.12），再叠加一层只用
400 条小样本拟合的温度，反而引入了过拟合噪声。**结论：温度校准的收益取决于训练前模型
本身有多离谱；用 soft-label 交叉熵直接训练出来的模型，校准收益可能有限甚至为负，
不能默认"多做一步校准就一定更好"。**

## 真实推理 / 路由验证

用 `laya.load()` 加载复现出的 checkpoint（补全 `tokenizer/`、`encoder/` 子目录后，
是标准 Laya checkpoint 目录结构，SDK 无需任何改动即可直接用），跑通了：

1. **`Agent.predict()` 单模型推理**：english checkpoint 在一条训练集内场景上给出与
   金标签分布一致的判断。
2. **`Router` 多 checkpoint 路由**：`router.attach()` 挂载自训的 english/multilingual
   两个 Agent，测试英语/中文/印地语/德语四种输入：

   | 输入语言 | 路由到 | 路由依据 | department 判断 |
   |---|---|---|---|
   | 英语 | english | 拉丁文字+英语 | billing (0.476) |
   | 中文 | multilingual | 非拉丁文字（汉字100%） | billing (0.676) |
   | 印地语 | multilingual | 非拉丁文字（天城文100%） | billing (0.689) |
   | 德语 | multilingual | 拉丁文字但判定为德语 | billing (0.594) |

   四种语言全部正确路由且分类正确，其中**中文、印地语、德语完全没有出现在训练数据中**
   （training data 是英语的 agent-trace/客服/发票/安全事件场景）——模型能答对纯粹来自
   mmBERT-base 预训练自带的跨语言迁移能力，印证了深度解读报告"多语言能力来自基础编码器
   预训练而非训练数据翻译"的结论。

   （`churn_risk` 在四种语言上判断都偏低 0.35~0.39，尽管每句话都带明确的取消威胁语气——
   这是训练数据领域不匹配：typed-decisions 数据集里没有"账单+流失风险"这个具体任务组合，
   不是路由或架构层面的问题。）

## 跨任务零样本评测：MASSIVE Intent / XNLI

typed-decisions 之外，仓库还下载了 MASSIVE Intent 和 XNLI 两个评测数据集
（原文用于评测 **base** english/multilingual checkpoint 的同任务零样本能力）。
我们的两个 checkpoint 已经在 typed-decisions 上做过 RLCD 微调，因此这里的评测
是**跨任务零样本泛化**测试，不是原文口径的"同任务零样本"对比。评测脚本
`repro/eval_massive_xnli.py` 直接读本地下载好的数据（不联网），复用
`research/eval/laya_eval.py` 的采样方法（seed 13、20 选项、干扰项采样）。

**MASSIVE Intent**（8 语言各 100 条，20 选项，随机基线 0.05）：

| 语言 | english | multilingual |
|---|---|---|
| en/de/fr/es/zh-CN/hi/ar/sw | 0.04~0.09 | 0.07~0.29 |
| **宏平均** | **0.0625**（≈随机） | **0.1613**（3x 随机） |

**XNLI**（8 语言各 100 条，3 分类，随机基线 0.33）：两个 checkpoint 在全部 8 种
语言上准确率**精确都是 0.33**——抽查发现模型对每一条 premise/hypothesis 都固定
回答同一个选项（如 "entailment"），0.33 只是恰好等于该类别在数据集里的先验占比，
不是真的具备推理能力。

**结论**：mmBERT-base（multilingual）预训练带来的通用语义理解能力，即使在
从未训练过的 MASSIVE 任务上也明显优于纯英语 ModernBERT-large，方向上印证了
"多语言能力主要来自基础编码器预训练"的结论；但两个 checkpoint 对 XNLI 这类
训练时完全未涉及的语义关系判断任务**没有任何可用的迁移能力**，会直接坍缩成
常数预测——说明 RLCD 微调是"专精"而非"通用增强"，与原文"Laya 是待专精的
快速基座，不是零样本通用决策引擎"的定位一致。

原始结果（含每条 case 详情）：`repro/eval_massive_xnli_english.json`、
`repro/eval_massive_xnli_multilingual.json`。

## 产出文件清单

```
repro/
├── prepare_data.py            # parquet -> 训练item（build_sequence编码）
├── train_ddp.py               # RLCD训练主循环（含build_model调用bug修复、校准stack bug修复）
├── calibrate_and_export.py    # 断点续跑：从checkpoint_latest单独补做校准+导出
├── eval_checkpoint.py         # argmax准确率评测（typed-decisions）
├── eval_calibration.py        # ECE校准前后对比
├── eval_massive_xnli.py       # MASSIVE Intent / XNLI 跨任务零样本评测（本地离线）
├── prepared/ , prepared_ml/   # english/multilingual 各自的 train/cal/test items
├── checkpoints/english/final_deploy/       # 可直接 laya.load() 的英语checkpoint
├── checkpoints/multilingual/final_deploy/  # 可直接 laya.load() 的多语言checkpoint
├── train_english.log, train_multilingual.log
```

## 已知遗留问题（后续可选修复）

1. DDP 收尾 `dist.barrier()` 偶发死锁，需要排查是否与 `find_unused_parameters=True`
   或某个 rank 提前退出 autograd 图路径有关。
2. 校准温度拟合在小样本（400条）+ 已经较诚实的模型上可能是负收益，若要在生产中使用，
   应该先测 ECE 前后对比再决定是否应用校准，而不是默认应用。
3. 本次只训了一轮、无超参搜索/多种子平均，如需对齐原文更高的绝对准确率数字，
   需要按 `Laya项目复现执行方案.md` 阶段2的"自建教师软标签"路线扩充数据规模，
   或对当前数据做多种子重复训练取平均。
