# NanoJev 项目深度解读报告

## 0. 项目定位

NanoJev 是对 TypeSafe 公司 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)（"System One 决策模型"）理念的开源小规模复刻：用一个 **0.6B 参数**的 Qwen3-0.6B 模型，接收 `state`（状态）+ `question`（问题）+ `candidates`（候选项），**直接输出完整概率分布**，全程不做任何自回归 token 生成（"zero output-token decoding"）。当前已在 Maze（迷宫）、Snake（贪吃蛇）、ViZDoom Basic（瞄准射击）、ViZDoom Predict Position（预判移动靶）四个任务上用**同一个 checkpoint** 跑通，发布版本为 `unified-games-v1`（对应训练实验 `hard_lr1e5` 的 step-400）。

项目最突出的气质是**极端的工程审计文化**：几乎每个数字都能通过脚本+SHA256 溯源，README/docs 反复自我纠偏（"这不代表校准概率""这不是官方复现"），未完成项在 Roadmap 里如实标注，而非包装成已完成结果——这在当前 AI 项目里相当罕见。

---

## 1. 技术架构设计

### 1.1 核心抽象：State–Question–Candidates 契约

所有任务被统一抽象为三元组请求（`docs/TYPESAFE_CONTRACT.md`），三种问题类型：
- **Boolean**：单一命题，输出 `p_true`；
- **Choice**：2–255 个动态候选，组内 softmax 多分类；
- **Score**：2–10 档有序等级，输出分布与概率加权期望。

关键契约约束：**问题 ID 从不进入模型输入**（防止 ID 泄露答案）、**问题间互不感知**（前向无跨问题注意力）、**候选顺序不影响单条路径编码**、**Score 每档只看自身描述、不注入序号**。这些都用单元测试（`test_question_contract.py`）强制校验。

### 1.2 模型结构（`DecisionModel`，`scripts/train_toy_decisions.py`）

- **Backbone**：`Qwen/Qwen3-0.6B`，**全量微调**（非 LoRA），FP32 存储 + BF16 autocast 前向。
- **打分机制**：每个候选被拼成完整序列 `state + question + candidate + "Decision:"`，backbone 前向后取**最后一个非 padding token 的 hidden state**，经 `LayerNorm → Linear(hidden,1)` 得到该候选的标量 logit——没有 vocab 投影、没有采样循环。
- **并行化（"parallel decisions"）**：一个 batch 内所有 state×question 的候选路径**展平成一个大矩阵**统一 padding，一次 backbone 前向后再按候选数切回二维打分张量。文档坦承这是"重复前缀、不做共享"的朴素实现，**共享前缀/树状注意力**被明确列为 Roadmap 未来优化项（`research/tree_attention_check.json` 是对这条路径的独立预研审计）。
- **三种头的具体实现**：
  - Boolean：两个隐含 logit `[0, z]` 做 softmax，等价于 sigmoid；
  - Choice：候选标量 logits 可选叠加 **set attention**（`MultiheadAttention`，4 头，`key_padding_mask` 处理 2–255 变长候选），让模型感知候选集合整体，再统一 softmax；
  - Score：各档独立打分后 softmax，`score = Σ i·p_i`（概率加权期望），`level = argmax`。

三种头共享同一 backbone 与 `scalar` 投影，仅在"如何组织候选 logits"上分叉，这是"一个骨干网络覆盖所有任务"的关键设计。

### 1.3 推理服务

`scripts/serve_decisions.py` 是一个纯标准库 `HTTPServer`，权重**只加载一次**，`POST /api/evaluate` 接收 state/question 批次返回完整分布。工程细节：拒绝重复 JSON key、拒绝 NaN/Inf、限制请求体/候选规模、全程 `HF_HUB_OFFLINE=1` 离线加载本地 `best.safetensors`。环境接口统一为 `reset/step/close`：Maze 用 5×5 局部窗口+已探索边图，Snake 用完整蛇身，ViZDoom 用结构化可见物体状态（非像素输入）。

---

## 2. 模型训练策略

### 2.1 渐进式监督蒸馏路线（非一次性端到端）

```
Toy 阶段（井字棋/BFS网格，纯程序化真值，验证架构）
  → 统一 SFT 初版（Maze+Snake+Basic+PP，Jev 引导 episode 蒸馏）
  → APPO 专家监督替换 Basic 数据（Basic: 56/128 → 128/128）
  → Sonic Doom 专家监督替换 Predict Position 数据（PP: 11/128 → 27/128）
  → Unified TD / RLCD 探索性后训练（未采纳进发布模型）
```

每一步遵循"只替换目标任务数据、其余任务 byte-for-byte 保留"的纪律，并用训练审计 JSON、checkpoint SHA256 保证可复现。

### 2.2 关键超参数

- **损失**：逐题完整 categorical cross-entropy（hard/soft 标签统一走同一 loss 形式）。
- **优化器**：AdamW，**backbone 与决策头分组学习率**——`hard_lr1e5` 实验组为 backbone lr=1e-5、head lr=1e-4（脚本默认 2e-5/2e-4），另有头部 warmup 阶段（head-only 训练，先冻结 backbone）。
- **任务混合权重**：Maze : Snake : Basic : PredictPosition = **1/3 : 1/3 : 1/6 : 1/6**，用分层重要性采样按精确权重构造 batch，避免小样本任务被淹没。
- **四臂消融**：`hard_lr2e5`、`soft_lr2e5`、`hard_lr1e5`、`soft_lr1e5` 四组同步训练、同一初始化种子，仅变量是 hard-target(one-hot) vs soft-target(专家概率分布) 与学习率；按 dev 集加权成功率选出 `hard_lr1e5` 的 step-400（dev 成功率从初始化 69.64% 提升到 80.31%）。

### 2.3 RLCD/TD 探索（未采纳）

作者自研了一个"paired proper-scoring 策略梯度"估计器 `paired_brier_pg`（受 RLCD 启发但非官方复现），从当前分布采样 M=32 个预测标签构造无偏梯度估计器，数学证明与直接 CE/Brier 优化效果相当（"支持实现正确性，不支持优越性主张"）。训练代码里保留了完整的 TD(n-step) 支持路径，但发布版本 `td_weight=0`，即**未启用**，仅作对照实验。README Roadmap 中"RLCD post-training"仍标记为未完成项。

---

## 3. 数据构建

### 3.1 规模与切分

每个 target 变体共 **18,760 道决策题**（ViZDoom Predict Position 11,173 + Basic 5,160 + Maze 1,469 + Snake 958），五个 split（train/dev/calibration/test/OOD）。训练集 10,898 题，经 target-validity 过滤后 **10,893** 题可用。此外含 **896 条 Predict Position 专家 episode、17,498 条决策记录**。

### 3.2 三类数据来源

| 任务 | 状态/候选构造 | Gold 来源 |
|---|---|---|
| Maze/Snake | 程序生成网格世界，局部原子几何判断（`clear_N/E/S/W`）+ 全图规划问题 | BFS 精确最短路 / 避碰+曼哈顿距离启发式，**纯程序化真值，不依赖任何模型** |
| ViZDoom Basic | 结构化可见物体状态 | 公开预训练 **APPO**（Sample Factory `edbeeching/doom_basic_1111`）专家策略 |
| ViZDoom Predict Position | 320×240 结构化状态 | 第三方开源 **Sonic Doom**（CNN+GRU）视觉专家策略，与学生环境同 seed 双实例同步逐 tick 采集 |

**重要澄清**：训练数据的"教师"全部来自程序化真值或公开第三方预训练策略，**不是**蒸馏自外部 Jev API——`research/nanojev_comparison_*` 与 Jev API 探测脚本只用于**评测对比**（拿 Jev 的动作分布做基线打分），与训练数据构建是两条独立管线。

### 3.3 Hard vs Soft 双变体 & 防泄漏设计

- **Hard target**：专家 argmax 动作的 one-hot；**Soft target**：专家网络输出的完整归一化概率分布，两者共用同一批 state/question/candidates，仅监督目标不同，用于四臂消融。
- **Split 隔离**：按环境分组键防泄漏——Maze 用地图 D4 对称标准形去重、Snake 用初始模拟器种子、ViZDoom 用 scenario+环境 seed 分组，保证同一等价类/同一地图不跨 split。
- **数据质量控制**：拒绝重复 JSON key/非法归一化的无效标注（target-validity filter）；所有专家轨迹都在独立标准环境里逐 tick 重放校验，杜绝录制误差。

---

## 4. 评测结果

在 548 个冻结测试+OOD case（同一观测接口、候选动作、seeded epsilon-greedy 控制器）上：

| 模型 | Maze | Snake | Basic | Predict Position |
|---|---:|---:|---:|---:|
| **NanoJev** | 4/10 | **8/8** | **128/128** | **27/128** |
| Jev（外部API） | **7/10** | 8/8 | 56/128 | 11/128 |
| 未调优 Qwen3-0.6B | 2/10 | 0/8 | 56/128 | 11/128 |

所有轨迹均经独立环境重放验证。**Maze 长程规划是明确短板**（弱于 Jev），说明"局部原子判断准"不等于"全局规划强"——这是当前方法论的核心瓶颈，文档对此如实呈现而非回避。

---

## 5. 总体评价

**亮点**
- 架构干净：State-Question-Candidates + 打分头 + softmax/sigmoid/期望的抽象，用极小的 0.6B 模型在多个任务上超过号称更强的 Jev；
- 工程审计文化极强：SHA256 溯源、独立重放验证、防泄漏 split 设计、Roadmap 如实标注未完成项；
- 数据构建路径清晰且诚实：纯程序化真值 + 公开第三方专家蒸馏，未夸大声称蒸馏自 Jev。

**局限（文档自陈）**
- 当前实现"重复前缀、无前缀/KV 共享"，候选越多计算浪费越大，属已知待优化项；
- 长程规划（Maze）能力仍弱于外部对照；
- RLCD/TD 后训练目前只是与直接 CE/Brier 打平，尚未证明优势，仍是探索性附加项。
