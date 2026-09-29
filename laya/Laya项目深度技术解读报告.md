# Laya 项目深度技术解读报告

> 调研对象：`/home/dataset1/gaojing/jev_replica/laya`（GitHub: NandhaKishorM/laya，PyPI 包 `laya`）
> 报告日期：2026-09-28

## 0. 项目定位一句话总结

Laya 是一个**多语言、非自回归的"系统一（System 1）"决策引擎**：给定任意状态（文本/邮件/工单/JSON）与一组"类型化问题"（`choice`/`score`/`noul`），在**单次前向传播**内并行给出所有问题的答案与经过校准的置信度，不生成文本、无需解析、不会"幻觉"。核心卖点：33ms 单问题延迟、100+ 语言路由、用"严格适当评分规则（strictly proper scoring rules）"训练的强化学习方法（自称 RLCD），并对标 TypeSafe 的 Jev 产品（HTTP 协议兼容）。

三个发布的 checkpoint：

| 名称 | 编码器 | 参数量 | 上下文 | 定位 |
|---|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512（可扩展） | 英语 |
| `laya-multilingual` | mmBERT-base | 322M | 1024（可到 8192） | 100+ 语言，速度快 2x |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | 针对 typed-decisions 基准微调 |

---

## 1. 技术架构总览

代码库分层清晰：
- `laya/` — 核心 SDK：`agent.py`（单 checkpoint 推理）、`router.py`（多 checkpoint 路由）、`common.py`（模型定义与共享工具，torch 依赖）、`lang.py`（无依赖的脚本/语言检测）、`confidence.py`（纯 Python 置信度/弃权逻辑）、`onnx_agent.py`（ONNX 运行时镜像实现）、`tl_kernels.py`/`fast.py`（TileLang GPU 加速路径）、`structured.py`（JSON Schema/pydantic → 问题编译）、`presets.py`（预置问题模板）、`shortlist.py`（高基数选项预筛）、`evals.py`/`evals_cli.py`（评测引擎与 CLI 门禁）、`hooks.py`（预测生命周期钩子）、`integrations/`（LangChain、LlamaIndex、CrewAI、MCP）。
- `research/` — 独立于 `laya` 包之外的评测/基准脚本与结果存档（不被生产代码 import）。
- `notebooks/` — 标准微调 pipeline（Kaggle 2×T4）。
- `laya-ts/` — TypeScript/浏览器移植。

三条推理后端并行存在，共享同一套编解码函数（`build_sequence`/`render_options`/`温度分桶`/`置信度计算`），只是张量执行方式不同：
1. **PyTorch eager**（默认）
2. **TileLang 融合 GPU 内核 + CUDA Graph**（`fast=True`，可选）
3. **ONNX Runtime**（`ONNXAgent`，可选，支持导出/量化）

这种"一套编解码词汇表 + 三种执行后端"的设计是保证跨后端数值一致性（parity）的架构核心。

---

## 2. 网络结构设计

### 2.1 单一共享决策头，而非"每类型一个头"

Laya **没有**为 `choice`/`score`/`noul` 分别设计独立的分类头，也不会为每个问题动态构建头。整个模型只有一套共享架构（`laya/common.py:278-349`，类 `DecisionModel`）：

```python
class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1, no_init=False):
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4*d, dropout, batch_first=True, norm_first=True)
        layer.self_attn = _DynamicMultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)      # 0=choice, 1=score, 2=noul
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
```

- `encoder`：预训练基础模型（ModernBERT-large 或 mmBERT-base），`attn_implementation="sdpa"`。
- `head`：额外的 2 层 `TransformerEncoder`（norm-first），自注意力用自研的 `_DynamicMultiheadAttention` 替换（为了 ONNX/追踪导出时形状安全）。
- `type_emb`：仅 3 行的 Embedding，加到序列**每个 token** 的隐状态上，把"这一行是 choice/score/noul 问题"这个信号广播进整个序列——这是模型区分问题类型的**唯一**结构化机制。
- `scorer`：一个共享 MLP（LayerNorm→Linear→GELU→Linear(→1)），对每个"标记位置（marker）"输出一个标量 logit，供所有类型问题复用。
- `act_head`：独立的小 MLP，输出"是否应该采取行动/弃权"的辅助信号（读取 `[CLS]` 池化状态 + 4 个置信度特征）。

**结论**：所谓"类型化决策"完全靠 (a) `type_emb` 注入的问题类型信息，和 (b) 解码时对同一份 K 元分类输出的**不同解释方式**实现，网络本身并无 choice-head / score-head 的区分。

### 2.2 一次前向传播如何同时回答多个问题——"每选项一个 [MASK]"机制

序列构造格式（`build_sequence`，`laya/common.py:135-207`）：

```
[CLS] <type> instructions [SEP] [MASK] opt0_tokens [MASK] opt1_tokens ... [SEP] state [SEP]
```

- `render_options` 把每个问题的 `criteria` 渲染成选项文本：choice → `"label: description"`；score → `"level i: description"`；noul → 固定的 `[false_label, true_label]`（语义顺序恒为 `[false, true]`）。
- 每个选项文本前插入一个字面 `[MASK]` token，其位置记录为 `marker`。
- `head_max_len`（默认 192）限制每个选项描述的 token 预算；预算不足时会在 `usage["options"]` 中标记 `collapsed_options`。

**"一次前向回答多个问题"的真实含义**：对同一个 state，每个问题各生成一行序列（各自带自己的 `[MASK]` 标记），这些行在 batch 维度上拼接、共享同一次编码器前向计算——并行来自 **batch 维度**，而不是"一行多头"。

前向计算核心（`DecisionModel.forward`，`laya/common.py:313-349`）：

```python
h = self.encoder(input_ids, attention_mask).last_hidden_state
h = h + self.type_emb(qtype)[:, None, :]                     # 注入问题类型
for layer in self.head.layers: h = layer(h, src_key_padding_mask=pad)
idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
m = torch.gather(h, 1, idx)                                   # 取出每个 [MASK] 处的隐状态
logits = self.scorer(m).squeeze(-1)                            # 同一个 scorer 输出每选项一个 logit
logits = logits.masked_fill(~marker_mask, -1e4)
```

解码（`_decode_answers`，`laya/agent.py:869-924`）按 `type` 分派：
- **choice**：`argmax(p)` 映射回标签。
- **score**：期望值 `Σ i·p_i`（对有序等级做期望，而非简单 argmax）。
- **noul**：直接取 `p[1]`（"true"槽位）作为 `P(true)`。

即：类型只改变对同一份 K 类分布输出的**解读方式**，网络结构本身是统一的。

### 2.3 路由架构（Router）

- **脚本/语言检测**（`laya/lang.py`，纯 Python、无第三方依赖）：Unicode 区块分类脚本 → 非拉丁脚本超阈值直接判定非英语 → 拉丁文本则用停用词表 + 变音符号频率启发式猜语言（覆盖 en/fr/de/es/pt/it/nl/ro/罗马化孟加拉语/az）→ 还会扫描"混合内容"（如英语客服工单里嵌一段德语客户留言）。核心问题只有一个：**"英语 checkpoint 能不能读懂这段文本"**。
- **Router 选择逻辑**（`Router._route`，`laya/router.py:618-709`）：优先级为 显式 `model=` > 显式 `task=` > typed-decisions 工作流自动检测（需 `auto_task_detection=True`）> 显式 `lang=` > `lang_guess` 回调/语言码提示 > 内置 `analyse()` 检测 > `default`（默认英语）。
- **Checkpoint 生命周期**：懒加载 + LRU（默认 `max_loaded=2`，因为自动路由只在 english/multilingual 间选择），`preload()` 可全部预载；`predict_batch` 会按"解析出的 checkpoint + 相同问题 schema"分组，让异构批次也能共享前向传播。

### 2.4 GPU 快速路径（TileLang 融合内核）

`agent.accelerate()` / `laya[fast]`，仅 CUDA，直接替换 `model.forward`，绕过 PyTorch eager 和 `torch.compile`：

- **融合内核**（`laya/tl_kernels.py`，均以行数 `M` 为运行时动态符号，一次 JIT 编译服务任意 batch/长度）：
  - `gemm_kernel`：通用 GEMM + 可选偏置/激活融合（GELU/ReLU）。
  - `gemm_geglu_kernel`：ModernBERT 门控 MLP 的 GEGLU 融合（一次内核完成 `gelu(A@Wi[:F]) * (A@Wi[F:])`）。
  - `add_ln_kernel`：残差相加（fp32 累加，因为 ModernBERT-large 的残差激活值可达 ~3e4，bf16 精度不够）+ LayerNorm 融合。
  - `rope_kernel`：直接在打包好的 QKV buffer 上原地做 rotate-half RoPE。
  - `attn_kernel`：flash-attention 风格的在线 softmax，支持 padding mask 与双向滑窗注意力。
- **CUDA Graph**：按 `(batch 桶, 长度桶)` 而非精确形状捕获图（batch 向上取 2 的幂，长度按粒度取整），首次遇到新桶时 warm-up 两次后捕获，之后仅 `copy_` 输入并 `replay()`，消除 Python 端每次 ~200 次内核启动的开销。
- 数值精度：与 fp32 前向相比最大 |Δp| ≤ 0.05（bf16）/ 0.009（fp16），argmax 一致率 ≥ 47/48。

### 2.5 置信度与温度校准

- `answer_confidence = max(p)`：**经过校准的**量，是 ECE 计算目标、也是 `min_confidence` 弃权阈值所读取的量。
- `confidence`：归一化香农熵 `1 - H(p)/log(k)`，描述分布集中程度，**未校准**。
- 温度缩放：checkpoint 自带按问题类型（choice/score/noul）的 3 元温度向量，外加更细粒度的按选项数分桶（`temp_bucket`：`2`/`3-5`/`6-10`/`11+`）的覆盖值，解码时 `logits/T` 再 softmax。温度被限制在 `[0.5, 5.0]`（防止过度"锐化"导致虚假高置信度）。

### 2.6 ONNX 导出路径

`ONNXAgent` 与 PyTorch `Agent` 几乎逐行镜像，仅推理执行方式换成 `onnxruntime.InferenceSession`；`_DynamicMultiheadAttention` 的存在正是为了让注意力可安全追踪导出（避免把序列长度硬编码进导出图）。输入/输出张量契约（`input_ids`/`attention_mask`/`marker_pos`/`marker_mask`/`qtype` → `logits`/`act_logits`）与 `DecisionModel.forward` 完全一致，编解码逻辑复用同一套 `laya/common.py`/`laya/confidence.py` 函数，从而保证 PyTorch/TileLang/ONNX 三后端结果一致。

---

## 3. 使用的基础模型

- **英语 checkpoint（`laya`）**：ModernBERT-large（421M），50k 英语 BPE 词表，仅擅长英语/拉丁文本；在非拉丁脚本上会"自信地失败"（如高棉语 Khmer：0.000 准确率 @ 0.952 平均置信度）。
- **多语言 checkpoint（`laya-multilingual`）**：mmBERT-base（322M），支持 RoPE 长上下文外推（1024 默认，最高 8192），是 100+ 语言覆盖的**主要来源**——多语言能力几乎完全来自基础编码器自身的多语言预训练，而非微调数据的多语言化。
- **加载方式**：`build_model(cfg, encoder_dir, pretrained=False)` 在 meta device 上以 `_no_init_weights()` 构建架构（不消耗初始化开销），再用 `safetensors.torch.load_file` 加载权重并做兼容性校验（`_verify_compatibility`：检查 config 字段、权重前缀 `encoder.`/`type_emb.`/`scorer.`/`act_head.`、形状严格匹配）。
- **代码库中没有证据表明团队自行从零预训练或对基础编码器做过持续预训练/领域适配预训练**——ModernBERT-large 与 mmBERT-base 都是直接下载的公开预训练权重，Laya 的贡献在于其上新增的 `type_emb`/`head`/`scorer`/`act_head` 结构，与编码器一起通过下游 RLCD 微调联合训练。README 自陈："Laya is a fast base to specialise, not a zero-shot decision engine"——基础 checkpoint 在 typed-decisions 基准上零样本准确率（0.362/0.352）甚至低于多数类基线（0.461），印证了这一点。

---

## 4. 数据如何构建

### 4.1 统一数据 schema

所有训练/评测样本都由三部分构成：**state**（任意文本/dict/JSON）、**questions**（类型化问题字典：`type`+`instructions`+`criteria`）、**gold**（金标签或分布）。

评测样本示例（`research/evals/fixture.jsonl`）：
```json
{"state": "I was charged twice for the same order, please refund me.",
 "questions": {"intent": {"type": "choice", "instructions": "What is the user asking for?",
               "criteria": {"billing": "refunds or charges", "technical": "bugs or errors",
                             "sales": "pricing or purchase"}}},
 "expected": {"intent": "billing"}, "tags": ["billing"], "language": "en"}
```

训练样本使用**软标签（教师概率分布）**而非硬标签（`docs/finetune.md`）：`gold[qid] = {"label":, "probabilities": {...}}`，通过 `build_training_item` 转成 `{ids, markers, qtype, target, label}`。

### 4.2 微调基准数据集来源

微调所用的 `typed-decisions` 数据集**并非在本仓库内生成**，而是直接从 Hugging Face 拉取现成数据集 `LocalLLaMA/typed-decisions`（config `"all"`）。仓库只负责消费（预处理→RLCD 训练→校准→评测→发布）。

- 规模：训练集 1,200 个 case → 展开为 6,000 个类型化决策；测试集 400 个 case / 2,000 个决策。
- **四个工作流**：Agent Trace Observability（智能体轨迹可观测性）、Customer Service（客服）、Invoice Processing（发票处理）、Security Incidents（安全事件）。
- **标注机制**：标签由**教师模型生成的软概率分布**而非人工硬标注——"Teacher Self-Agreement"（0.735 准确率）这一行本身就是教师模型自一致性的度量基线，微调后的 Laya（0.766）甚至超过了这个教师自一致性天花板。
- 已确认与 MASSIVE/XNLI 无训练重叠。

一个更具体展示"如何构造合成数据"的案例是文档记录的**浏览器代理决策头**微调（`docs/finetune_browser_agent.md`）：爬取 421 个真实网页 → 先选定答案元素、再用本地 Qwen3-8B **反向生成**用户目标（保证标签干净，无需教师"解题"）→ 真实浏览器执行轨迹的 DONE 状态 → 同页面新目标构造负样本（防止"有历史记录"成为捷径特征）→ 引入公开数据集 Mind2Web 重新格式化 → On-policy DAgger 修正（本地 LLM 裁判现行模型的每一步）。文档记录的负面经验（模板化目标措辞泄露捷径信号、历史/DONE 相关性教出虚假规则、类别不平衡）对理解数据构建陷阱很有参考价值。

### 4.3 公开数据集在评测中的复用

- **MASSIVE Intent**（`mteb/amazon_massive_intent`）：state = `{"utterance": text}`；单个 `choice` 问题，固定指令 `"What is the user asking for in \`utterance\`?"`，`criteria` = 金标签 + `random.Random(seed=13)` 每语言重新采样的 distractor 标签池（默认 20 个选项）。覆盖全部 51 种语言。
- **XNLI**：state = `{"premise":, "hypothesis":}`；单个 `choice` 问题 `relation`，三分类（entailment/neutral/contradiction）语义化描述为 criteria；覆盖 15 种语言。
- 六个"应用工作流"基准（`bench_apps.py`）：客服工单分类（Tobi-Bueck/customer-support-tickets + Banking77）、邮件+钓鱼检测（Enron spam/phishing）、LLM 护栏（lmsys/toxic-chat jailbreak）、RAG 段落相关性（MS MARCO）、内容审核（lmsys/toxic-chat toxicity）、模型路由（gsm8k/mbpp/writing/factual 领域分类）；均按同一模式把原始文本→state，数据集标签集→choice/noul criteria。AG News/Banking77/DAIR Emotion/SST-5 仅用作与第三方公开 Jev 数据对比的英语零样本参照点，明确排除在训练之外。

### 4.4 多语言数据的本质来源

**没有证据表明用了机器翻译扩充多语言训练数据**。100+ 语言覆盖主要靠：(a) `laya-multilingual` 选用 mmBERT-base 这个本身就多语言预训练的编码器；(b) 一个纯启发式的**路由层**（`laya/lang.py`）把非英语/非拉丁文本分流到多语言 checkpoint，而不是靠某种翻译后的微调语料。`laya-multilingual` 甚至"出厂不带任何拟合好的温度值"，进一步印证其多语言特定的微调投入相对英语要轻得多。

### 4.5 数据/问题构造的规范化与校验

- `criteria` 归一化（`Agent._to_internal`）：布尔键 noul criteria 归一化为字符串 `"true"/"false"`；bare list 形式的 choice criteria 展开为 `{label: None}`；拒绝空标签、重复/哈希冲突标签、不可哈希标签，均以明确 `ValueError` 报错。
- JSON Schema/pydantic → 问题编译（`laya/structured.py`）有明确容量上限（`MAX_PROPERTIES=32`、`MAX_OPTIONS=32`、`MAX_SCORE_LEVELS=10`），对不支持的 schema 形状（自由字符串、嵌套对象、`$ref`、多分支联合等）在构造时直接拒绝。
- 社区贡献的诊断数据集（中文场景，`research/benchmarks/feishu_zh`、`research/benchmarks/zh_short_commands`）为 AI 辅助合成的小规模固定用例集，专门用于揭示"criteria 构造方式（是否附加场景/JSON state）如何影响 noul 问题的标签分布坍缩"这类问题，属于评测诊断而非训练语料。

---

## 5. 训练策略（RLCD）

### 5.1 "RLCD" 是不是真的强化学习？——是，但是一种轻量、单步的策略梯度

核心奖励函数（`laya/common.py:410-436`，`proper_reward`）是**对数评分（log score）+ 球面评分（spherical score）+（仅对 score 类问题）排序概率评分（RPS）的凸组合**——三者都是严格适当评分规则，当且仅当模型输出分布等于真实分布时期望奖励最大化。

```python
def proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q, cdf_t = torch.cumsum(q, -1), torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r
```

真正的 RL 步骤（微调 notebook 中的 `train_ddp.py`）：对当前（detach 的）logits 施加 `GROUP_SIZE=4` 个零均值高斯噪声扰动，得到 4 组"采样"的 soft 分布，用 `proper_reward` 对教师金标签打分，做**组内均值基线相减 + 白化**得到 advantage（GRPO 风格，无独立价值网络），再用标准 REINFORCE 形式的策略梯度损失更新，与**满权重的软交叉熵**损失相加。

**结论**：RLCD 确实是货真价实的策略梯度方法（非纯监督学习的营销包装），但属于窄范围、低方差、单步、无环境 rollout、无回放缓冲区的简化版本——本质是"GRPO 风格的策略梯度 + 严格适当评分规则奖励，外加一个权重更大的软交叉熵锚定项"，不应理解为大规模 RLHF。

### 5.2 端到端微调 Pipeline（`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`）

1. **数据构建**：加载 `LocalLLaMA/typed-decisions`，用 `build_training_item` 把每个 `(state, question, gold)` 转成 tokenize 好的训练 item（含 `[MASK]` marker 位置、类型 id、目标分布）。
2. **模型/损失**：`DecisionModel` 无按类型区分的损失函数——损失是类型无关的（软 CE + 策略梯度），`qtype` 唯一起作用之处是 (a) 送入 head 的 type embedding，(b) `proper_reward` 中仅对 score 类问题激活的 RPS 惩罚项。
3. **训练循环**：`torchrun --nproc_per_node=2 train_ddp.py`，4 epoch；每个 micro-batch：fp16 autocast 前向 → RL 损失 + 软 CE 损失 → 梯度缩放反传 → 4 步累积 → 梯度裁剪（1.0）→ 优化器步进 → cosine 学习率调度。每个 epoch 后保存滚动 checkpoint 以应对崩溃/超时。
4. **校准拟合**：训练结束后（仅 rank 0），在**训练前预留、从未参与训练**的校准切片上，对每个问题类型分别用 LBFGS 在 log 空间拟合一个温度标量（clamp 到 `[0.1, 10]`），并显式删除继承的按选项数分桶的温度覆盖值，防止其掩盖新拟合结果。
5. **评测**：在官方测试集（400 case / 2,000 决策，4 个工作流）上计算准确率、软准确率、Brier、ECE、MAE 等指标，并与 TypeSafe Jev（0.727）、ModernBERT-base 专用模型（0.646）、教师自一致性天花板（0.735）对比——微调后 Laya 得 **0.766**。
6. 可选：推送到 Hugging Face Hub。

### 5.3 硬件与超参数

| 项目 | 取值 |
|---|---|
| 硬件 | Kaggle 免费 2×T4（DDP） |
| epoch | 4 |
| 微批次 | 8 序列/GPU |
| 梯度累积 | 4 步（有效 batch = 64） |
| 优化器 | AdamW，weight_decay 0.01 |
| 学习率 | 编码器 2.5e-5，决策头 1.0e-4 |
| 调度 | cosine → eta_min 1e-6 |
| GRPO 组大小 | 4 |
| 探索噪声 σ | 0.4 → 0.1 线性退火 |
| 精度 | fp16 autocast + GradScaler |
| 显存优化 | 编码器 + 决策头均启用梯度检查点 |
| 序列预算 | max_len 1024, head_max_len 256 |

单卡（RTX 4070 Ti SUPER）的浏览器代理微调变体验证：去掉梯度检查点可获得约 1.25x 的"免费"加速。

### 5.4 校准机制细节

- 训练后拟合：每类型一个温度（3 个标量），LBFGS 最小化 held-out 校准集上的 NLL。
- 推理时（基础 checkpoint 的更细粒度方案）：按 `(问题类型, 选项数)` 分桶拟合温度，把 mean ECE 从 0.466 降到 0.081（`laya`）、从 0.314 降到 0.106（`laya-multilingual`，出厂无任何拟合温度）。
- 加载时数值 clamp 到 `[0.5, 5.0]`（比拟合时的 `[0.1, 10]` 更紧），防止过度锐化产生虚假高置信度（文档记录过一次 0.1 温度把 0.24 的真实概率放大成虚假 0.99 的事故）。
- 温度缩放**不改变 argmax/准确率**，只改变置信度的校准程度。

---

## 6. 评测方法与结果

### 6.1 `laya-evals` CLI 门禁系统

`laya/evals.py` + `laya/evals_cli.py`，import 时刻不依赖 torch，纯 Python/numpy 实现，可在 CI 中零权重下载运行。核心能力：解析带 `state`/`questions`/`expected` 的 JSONL 数据集 → 用 `RouterRunner`（torch）或 `OnnxRunner`（ONNX）跑推理 → 计算 `ChoiceAccuracy`/`NoulAccuracy`/`ScoreMAE`/`ScoreWithin(tol)`/`MeanConfidence`/ECE（15 分箱）/延迟分位数 → 按 language/model/qid/tag 切片聚合 → 与 baseline JSON 做容差比较（`--tolerance METRIC=VALUE`）。

CI 双轨：
1. 无权重单测（`tests/test_evals*.py`）在每个 PR 上跑，测指标数学、数据解析、CLI；
2. `.github/workflows/evals.yml` 每周一 + 发版时跑真实 `research/eval/laya_eval.py` 全量评测并与已提交基线比对，**不阻塞 PR**——失败只是发布数字回归的信号。

### 6.2 独立的 MASSIVE/XNLI 评测哈希（`research/eval/`）

为响应社区诉求（issue #35）建立的独立、可复现的第三方评测脚本，故意放在 `laya` 包之外（避免拉入 `datasets` 依赖）。方法：固定 prompt 格式、固定 seed=13、每语言重新采样 20 个选项、15 分箱 ECE，覆盖全部 51 种 MASSIVE 语言。产出四段式 JSON（config/report/summary/cases），使第三方可零网络复现每一条汇总数字。

值得注意的**自曝的不一致**：英语 checkpoint 在 51 种语言上 100% 精确复现已提交结果；多语言 checkpoint 仅 6/51 语言精确复现（宏平均从 0.3661 变为 0.4008）——团队排查了选项集摘要、权重字节一致性、数据集版本、`build_sequence`、`head_max_len` 等因素但未能定位原因，选择"标记问题而非隐藏问题"公开记录。

### 6.3 元变换（metamorphic）与呈现方式（presentation）检查

- **Presentation checks**：针对 score 问题"位置偏置"bug（`laya-multilingual` 很少选中排在第一位的等级）设计的无标签回归检测，用 10 个固定英语状态测量"槽位 0 logit 偏差"和"选项顺序打乱后首位选中率"。结果：英语 checkpoint 通过，`laya-multilingual` 失败（首位选中率仅 0.017，远低于均匀期望的 0.333）。
- **Metamorphic option-order robustness**：对每个 MASSIVE case 做选项顺序打乱和标签重命名两种变换，度量语义一致率、概率漂移、JS 散度、置信度漂移，用于把"位置敏感性"和"词面标签敏感性"区分开——没有统一阈值，"可接受的波动取决于使用场景"。

### 6.4 关键量化结论汇总

- **路由把可用语言数从 23/51 提升到 45/51**（MASSIVE intent，3 倍随机基线为通过标准）。
- **英语 checkpoint 在非拉丁脚本上"自信地失败"**：高棉语 0.000 准确率 @ 0.952 平均置信度；51 语言宏平均 ECE 高达 0.733——这正是"路由必须发生在前向传播之前，置信度门控无法救场"的论据来源。
- **两个 checkpoint 出厂都过度自信**：温度拟合后 mean ECE 从 0.466→0.081（英语）、0.314→0.106（多语言，出厂无任何拟合温度）。
- **typed-decisions 零样本接近随机**：0.362/0.352，甚至低于多数类基线 0.461；微调后达到 0.766，超过教师自一致性天花板 0.735，也超过 Jev 公开数据 0.727。
- **速度**：单问题 32.8ms，批量 10 问题时 7.2ms/问题（T4 GPU），批量吞吐 103–332 问题/秒。

### 6.5 与 Jev 的对比方法论——"诚实的局限性"

所有文档均明确声明：**Jev 从未被直接运行**，引用的 Jev 数字均来自第三方公开评测（`AbdelStark/jev-benchmarks`、`nibzard/decision-model-benchmark`），样本量和 prompt 均不同，"仅供参考，不是受控的正面对比"。README 单辟"Where Jev leads"和"Honest limits"章节，自曝的局限包括：
- 高基数选项（>20 个）场景明显弱于 Jev（Banking77：Jev 0.870 vs Laya 0.425，77 个标签下每标签仅 3-4 个 token 预算）；
- 应避免布尔词形式的 choice 标签（`true`/`false`），模型可能只认字面标签而非状态内容；
- 语义标签也不能保证否定句安全（issue #377 记录了误判案例）；
- 有序 `score` 是最弱的原语（SST-5 仅 0.372）；
- `noul` 存在跟随选项标签字面而非状态内容的风险（issue #156）；
- `act_probability` 辅助信号"尚不可用"（AUROC 仅 0.30，远低于 `confidence` 的 0.77），明确指示应该用 `confidence` 而非它做门控；
- 内容审核任务几乎处于随机水平（0.530 准确率，宏 F1 仅 0.400），Demo 中的 Moderation 标签页被标注为"不可靠"。

### 6.6 六大应用工作流基准（`bench_apps.py`）

支持工单分类、邮件+钓鱼检测、LLM 护栏（越狱检测）、RAG 段落相关性、内容审核、模型路由（领域分类）——每项 400 个测试用例，三个 checkpoint 对比，同时保留三个可与 Jev 公开数字对标的英语基准套件（AG News/Banking77/DAIR Emotion），并在代码中硬编码注明数据来源和版本，避免误导为直接对比。

---

## 7. 综合评价

**架构层面**的核心创新是把"任意类型化问题"统一编码为"选项前插 [MASK] + 类型嵌入广播"的单一 K 类分类问题，用同一个共享 scorer 头处理，靠 batch 维度并行而非多头分支实现"一次前向答多题"，工程上干净且易于跨 PyTorch/TileLang/ONNX 三后端保持一致。

**训练层面**的 RLCD 是名副其实但克制的策略梯度方法（GRPO 风格 + 严格适当评分规则奖励），本质上是让模型学会输出经过良好校准的软概率分布，而非单纯拟合硬标签，这与其"置信度具有统计意义"的产品主张自洽。

**数据层面**呈现出"自有代码库并不生产原始训练语料"的特点——typed-decisions 微调数据来自外部 HF 数据集（教师软标签），多语言能力主要靠选用多语言预训练编码器 + 纯启发式路由，而非翻译扩增语料；这使得 Laya 的"数据护城河"其实相对薄弱，更依赖用户自行微调（README 反复强调"零样本基础 checkpoint 只是用于专精的起点，而非零样本决策引擎"）。

**评测层面**的态度值得称道：独立、可复现、torch-free 的评测哈希，明确记录未解决的复现差异（多语言 checkpoint 45/51 vs 6/51），系统列出"诚实的局限性"，并明确声明未直接运行竞品 Jev、所有对比数字的来源与局限，是本项目区别于一般"营销型跑分"的关键特征。

---

## 附：关键文件索引

| 主题 | 关键文件 |
|---|---|
| 模型/头架构 | `laya/common.py:278-349`(DecisionModel), `:135-207`(build_sequence), `:106-133`(render_options) |
| 推理解码 | `laya/agent.py:869-924`(_decode_answers) |
| 路由 | `laya/router.py:618-709`(_route), `laya/lang.py:585-637`(analyse) |
| GPU 快速路径 | `laya/tl_kernels.py`, `laya/fast.py:106-209` |
| 置信度/校准 | `laya/common.py:457-521`, `laya/confidence.py` |
| ONNX | `laya/onnx_agent.py:520-695` |
| 训练/RLCD | `laya/common.py:410-436`(proper_reward), notebook `train_ddp.py` cell, `docs/finetune.md` |
| 数据构建 | `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`(cell 6), `docs/finetune_browser_agent.md`, `laya/evals.py:31-61` |
| 评测系统 | `laya/evals.py`, `laya/evals_cli.py`, `research/eval/README.md`, `research/evals/README.md`, `BENCHMARKS.md` |
