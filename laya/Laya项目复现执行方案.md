# Laya 项目复现执行方案

> 目标：从零构建一个功能等价的"多语言、非自回归 System 1 决策引擎"（类 Laya），覆盖模型架构、数据构建、训练（RLCD）、校准、评测、部署全链路。
> 前置：本方案基于 `Laya项目深度技术解读报告.md` 的调研结论制定，关键代码位置以该报告的"附：关键文件索引"为准。

---

## 0. 总体路线图

```
阶段0 环境与基线   → 阶段1 架构实现 → 阶段2 数据构建 → 阶段3 RLCD训练
   → 阶段4 校准拟合 → 阶段5 评测门禁 → 阶段6 加速与导出 → 阶段7 服务化与集成
```

建议按此顺序推进，每阶段结束都有可验证的产出物（checkpoint / 报告 / 通过的测试），不要跳阶段并行开工，因为阶段2-4强耦合（数据格式决定训练格式，训练格式决定校准脚本）。

---

## 阶段 0：环境与基线准备

**目标**：跑通"下载基础编码器 + 空架构前向"的最小闭环，验证工具链。

- [ ] 环境：Python ≥3.10；`torch`、`transformers`、`safetensors`、`huggingface_hub`；GPU 建议至少 1 张 T4 级别（16GB）用于验证，2×T4 或以上用于训练。
- [ ] 拉取基础编码器权重：
  - 英语路线：`answerdotai/ModernBERT-large`（或同等开源 ModernBERT-large 权重）
  - 多语言路线：`jhu-clsp/mmBERT-base`（或同等 mmBERT-base 权重）
- [ ] 验证两个编码器可用 `AutoModel.from_pretrained` 正常加载，`AutoTokenizer` 正常分词，确认 mmBERT 的 RoPE 长上下文外推参数（theta）可读。
- [ ] 确定复现范围（**关键决策点，建议先用 AskUser 澄清或自行拍板**）：
  1. 只复现**英语单 checkpoint**（成本最低，验证架构/训练流程）
  2. 英语 + 多语言双 checkpoint + Router（完整功能对等）
  3. 完整三 checkpoint（含 typed-decisions 微调）+ 全部评测套件（完全对等）
  - 建议：先做 (1)，验证架构与 RLCD 训练闭环无误后，再扩展到 (2)(3)。

**产出物**：可复现的 conda/venv 环境说明、编码器权重本地缓存、一份"环境自检脚本"输出。

---

## 阶段 1：架构实现（对应原报告"网络结构设计"章节）

### 1.1 核心决策模型 `DecisionModel`

按报告中的结构逐一实现（不新增无关抽象）：

```python
class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1):
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4*d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers)
        self.type_emb = nn.Embedding(3, d)          # 0=choice 1=score 2=noul
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
```

要点提醒：
- 若需要 ONNX 导出，`nn.MultiheadAttention` 需自行替换为形状安全的实现（避免追踪时把序列长度硬编码进图）；不追求导出可先用标准 `nn.MultiheadAttention` 简化开发。
- `type_emb` 广播到**每个 token**（不是只加到 CLS），这是问题类型注入的唯一路径。
- `scorer` 是全局共享的，不要为不同问题类型建多个 scorer。

### 1.2 序列构造与选项渲染

实现 `render_options(question)`：
- choice → `"{label}: {description}"`
- score → `"level {i}: {description}"`
- noul → 固定顺序 `[false_label, true_label]`

实现 `build_sequence(tokenizer, state, question, max_len, head_max_len)`：
- 格式：`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`
- 记录每个 `[MASK]` 的 token 位置为 `markers`
- 用 `head_max_len` 截断选项区，用 `max_len` 截断整体（state 超长时按需要截左/截右）

实现批处理整理 `collate_items`：输出 `input_ids/attention_mask/marker_pos/marker_mask/qtype`。

### 1.3 前向与解码

- forward：`encoder → +type_emb → head → gather(marker_pos) → scorer → mask → logits`
- decode：
  - choice: `argmax(softmax(logits/T))`
  - score: `Σ i·p_i`（期望值，非 argmax）
  - noul: `p[1]`

### 1.4 置信度与温度缩放

- `answer_confidence = max(p)`（校准目标量）
- `confidence = 1 - H(p)/log(k)`（未校准的熵度量，仅供参考）
- 温度：`temperature[qtype]` 三元向量 + 可选按 `(qtype, 选项数分桶)` 的覆盖表，`temp_bucket` 分桶建议：`2 / 3-5 / 6-10 / 11+`
- 加载时 clamp 到 `[0.5, 5.0]`，防止过度锐化产生虚假高置信度

### 1.5 路由层（若做多 checkpoint）

- 纯 Python 无依赖的脚本/语言检测：Unicode 区块分类 + 停用词/变音符号启发式（覆盖目标语言列表自定）
- Router 选择优先级：显式 model > 显式 task > 语言提示 > 检测结果 > 默认值
- LRU 缓存已加载 checkpoint，支持 `preload()`

**验收标准**：给定固定随机种子的编码器初始化，跑一个 fixture 问题（如 3 选项 choice），确认前向输出 shape、mask 正确、`[MASK]` gather 位置与手工构造序列一致。建议先写单元测试（对齐原报告 `tests/test_training.py` 中的"reward strictly proper" "permutation invariance" "masking correctness"等验证思路，自建等价测试）。

**产出物**：可跑通的 `common.py`/`agent.py` 等价实现 + 单元测试全绿。

---

## 阶段 2：数据构建（对应原报告"数据如何构建"章节）

### 2.1 统一数据 schema（务必先定死，后续训练/评测代码都依赖它）

```json
{"state": "...(文本/dict/JSON)...",
 "questions": {"qid": {"type": "choice|score|noul", "instructions": "...",
                        "criteria": {...} }},
 "gold": {"qid": {"label": "...", "probabilities": {"optA": 0.7, "optB": 0.3}}}}
```

### 2.2 训练数据来源（三条可选路径，按复现深度选择）

1. **直接复用/等价替代 `LocalLLaMA/typed-decisions`**：若该数据集公开可下载，直接拉取，跳过自建；若不可得，进入路径 2。
2. **自建"教师软标签"合成数据**（推荐，工程上最贴近原方案）：
   - 确定 3-4 个目标工作流（可参照原方案：Agent 轨迹可观测性 / 客服 / 发票处理 / 安全事件，或替换为自己业务的等价工作流）。
   - 为每个工作流准备/爬取/合成状态文本（真实工单、公开邮件数据集、合成对话等）。
   - 用一个更强的教师模型（本地开源 LLM 或 API 大模型）对每条 `(state, question)` 输出**校准过的概率分布**而非硬标签——具体做法：多次采样/多角度提示后统计频率，或直接让教师模型输出置信度分布并做温度校正。
   - 记录"教师自一致性"（同一输入多次采样教师答案的一致率）作为后续训练效果的天花板参照。
   - 负样本/难例构造参考原方案"浏览器代理"案例的经验：
     - 用"先选答案、再反向生成问题/目标"的方式保证标签干净；
     - 避免模板化措辞泄漏捷径特征；
     - 对不平衡类别做 3-4x 重加权，防止稀有类别被吞掉；
     - 可选：on-policy 修正——用初版模型产出候选，教师模型/规则做裁判，回收进训练集（DAgger 风格）。
3. **公开数据集重新格式化**（用于零样本能力/多语言能力评测，也可少量混入训练）：
   - 意图分类类：MASSIVE Intent（`mteb/amazon_massive_intent`）→ 单 choice 问题，正确标签 + 随机采样 distractor 构成选项池（建议 20 个选项，固定 seed）。
   - 自然语言推理类：XNLI → 单 choice 问题，三分类语义化描述作为 criteria。
   - 应用场景类：客服工单（如 Tobi-Bueck/customer-support-tickets）、垃圾邮件/钓鱼（Enron spam）、护栏/越狱检测（toxic-chat）、段落相关性（MS MARCO）、内容审核（toxic-chat toxicity）、模型路由领域分类（gsm8k/mbpp/writing/factual 等）。

### 2.3 多语言数据策略

- 若要复现"100+语言"能力：**优先选用本身多语言预训练的编码器（如 mmBERT）**，把主要精力放在路由检测层，而不是费力去做多语言训练数据翻译扩增——原项目的实践表明多语言能力主要来自基础模型预训练而非微调语料翻译。
- 若时间/预算允许，可以在少量高频语言（如中/日/韩/法/德/西/阿）上补充原生（非翻译）标注数据，进一步提升该语言下游任务表现。

### 2.4 数据校验规则（构造问题时必须强校验，防止训练时静默出错）

- `noul` criteria 键归一化为字符串 `"true"/"false"`
- `choice` criteria 可接受 list 或 dict，内部统一展开为 `{label: description}`
- 拒绝：空标签、重复/哈希冲突标签（如 `1` 与 `1.0` 与 `True`）、不可哈希标签 —— 直接报错，不静默丢弃
- 选项 token 预算耗尽导致选项区分不清时，需要有告警机制（对应原方案的 `collapsed_options`）

**验收标准**：至少产出（a）N 个工作流 × M 条状态的训练集（软标签），（b）MASSIVE/XNLI 格式化好的评测集，（c）一份数据统计报告（各类型问题数量、平均选项数、语言分布）。

**产出物**：`data/train.jsonl`、`data/eval_*.jsonl`、数据构建脚本、数据统计报告。

---

## 阶段 3：RLCD 训练（对应原报告"训练策略"章节）

### 3.1 损失函数：严格适当评分规则奖励 + 策略梯度 + 软交叉熵

核心奖励（对数评分 + 球面评分 + 有序问题的 RPS 惩罚）：

```python
def proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21):
    q = q * mask
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

GRPO 风格策略梯度步骤：
1. 对当前（detach 的）logits 加 `GROUP_SIZE=4` 组零均值高斯噪声（σ 建议从 0.4 线性退火到 0.1）
2. 每组 softmax 得到采样分布 `q`，用 `proper_reward` 对教师软标签打分
3. 组内均值基线相减 + 标准差白化，得到 advantage
4. 策略梯度损失 `loss_rl = -(adv * logp).mean()`（`logp` 是采样噪声在高斯策略下的对数似然）
5. 与满权重软交叉熵损失 `loss_ce` 相加：`loss = loss_rl + 1.0 * loss_ce`

**先验证正确性**：写单元测试验证 `proper_reward` 的"严格适当性"——网格搜索报告概率，确认奖励期望在报告概率等于目标概率时最大化；验证选项排列不变性；验证 mask 正确性；验证 RPS 只对 score 类型生效。

### 3.2 训练超参数建议（参照原方案，视自有硬件调整）

| 项目 | 建议起点 |
|---|---|
| 硬件 | ≥2×16GB GPU（DDP），单卡也可（去掉分布式） |
| epoch | 3-4 |
| 微批次 | 8/GPU |
| 梯度累积 | 4（有效 batch 64） |
| 优化器 | AdamW, weight_decay 0.01 |
| 学习率 | 编码器 2.5e-5，决策头 1e-4（分组学习率） |
| 调度 | cosine → eta_min 1e-6 |
| GRPO 组大小 | 4 |
| 探索噪声 σ | 0.4→0.1 线性退火 |
| 精度 | fp16/bf16 autocast + GradScaler |
| 显存优化 | 编码器 + 决策头均可选梯度检查点（若显存紧张再开，原方案实测关闭可省 ~1.25x 训练时间） |
| 序列预算 | max_len 1024，head_max_len 192-256 |

### 3.3 工程细节

- 训练前**预留校准切片**（如 400 条或 10%），绝不参与训练权重更新，专供阶段4温度拟合使用。
- 每 epoch 保存滚动 checkpoint，防止长训练中断丢失进度。
- 分布式训练如遇到"未使用参数"报错（`act_head` 若某些 batch 未被用于损失），加一个 `0.0 * act_head_output.sum()` 之类的 no-op 项保持梯度图完整，或直接设置 `find_unused_parameters=True`。

**验收标准**：训练 loss 收敛、校准切片上的准确率/软准确率达到预期区间（可参照原项目 typed-decisions 基准从零样本 ~0.36 提升到微调后 ~0.7+ 作为量级参考，具体数值取决于自建数据质量）。

**产出物**：训练好的 `model.safetensors` + 训练日志 + loss/reward 曲线。

---

## 阶段 4：校准拟合

- 用阶段3预留的校准切片（**从未训练过**），对每个问题类型分别用 LBFGS 在 log 空间拟合温度标量：
  ```python
  loss = -(target * log_softmax(logits / T, -1)).sum(-1).mean()  # 对 log(T) 做梯度下降
  ```
  clamp 到 `[0.1, 10]`，样本不足（如 <10 条）时回退到 1.0（或 1.2 出错时兜底）。
- 更精细版本：按 `(问题类型, 选项数分桶)` 拟合温度覆盖表（分桶如 `2 / 3-5 / 6-10 / 11+`），并在加载模型时把该覆盖表数值 clamp 到更紧的 `[0.5, 5.0]`（区别于拟合时的宽松范围，用于生产安全）。
- 拟合完成后写入模型配置文件（如 `rl_agent_config.json`），并清空旧的分桶覆盖表，防止其掩盖新拟合结果。
- 计算拟合前后的 ECE（15 分箱）对比，作为校准效果的量化证据。

**验收标准**：温度拟合后 ECE 明显下降（原项目标杆：mean ECE 从 0.4+ 降到 0.1 以内），且 argmax 准确率不变（验证温度缩放只影响置信度不影响决策）。

**产出物**：拟合好的温度值 + 拟合前后 ECE 对比报告。

---

## 阶段 5：评测体系搭建（对应原报告"评测"章节）

### 5.1 无权重单测（纯 Python/numpy，CI 友好）

- 数据集解析校验（state/questions/expected 字段完整性、类型匹配）
- 指标计算：`ChoiceAccuracy`/`NoulAccuracy`/`ScoreMAE`/`ScoreWithin(tol)`/`MeanConfidence`/ECE（15 分箱）
- 阈值/基线比较 CLI：支持 `--min-accuracy`、`--max-ece`、`--baseline` + `--tolerance`

### 5.2 独立的语言/任务评测哈希

- 固定 seed、固定 prompt 格式，跑全量 MASSIVE 支持语言 + XNLI，产出四段式 JSON（config/report/summary/cases），确保第三方可零网络离线复现汇总数字。
- 建议同时跑通"英语 checkpoint 在非拉丁文字上应大幅下降"这一健全性检查，作为路由必要性的证据。

### 5.3 鲁棒性诊断（元变换 + 呈现方式检查）

- **选项顺序打乱**测试：同一 case 打乱选项顺序，测语义一致率、概率漂移、JS 散度。
- **标签重命名**测试：把选项标签换成无语义的 `A/B/C`，隔离"位置敏感" vs "词面标签敏感"两种失效模式。
- **score 位置偏置检测**：固定状态、构造选项文本完全相同的多等级 score 问题，检查模型是否偏爱某个固定槽位。

### 5.4 CI 门禁建议

- PR 级别：跑无权重单测，秒级完成，必须通过。
- 定期（如每周）：跑真实 checkpoint 全量评测，与已提交基线比较，容差告警但不阻塞 PR（发布数字变化需要人工复核，而非自动拦截）。

### 5.5 对外声明规范（诚实的局限性文档）

- 明确写清楚：哪些评测直接跑了自己的模型，哪些对比数字是引用第三方公开结果（并注明来源、样本量、prompt 差异），避免混淆为"受控对比"。
- 列出已知局限（如高基数选项场景弱、有序 score 较弱、某类问题存在标签文字捷径等），随着自建过程中发现新问题持续更新。

**产出物**：`evals` CLI 工具、评测报告 JSON、鲁棒性诊断报告、公开的局限性说明文档。

---

## 阶段 6：加速与导出（可选，视性能需求决定是否投入）

- **ONNX 导出**：优先保证注意力实现在追踪时不硬编码序列长度（自定义 attention 或使用支持动态轴导出的实现），验证导出后与 PyTorch 结果的数值一致性（argmax 一致率、最大概率漂移）。
- **INT8 量化**（CPU 部署）：导出后做逐通道量化，评测精度损失是否可接受。
- **GPU 融合内核加速**（可选，工程量较大，仅在有明确延迟指标要求时投入）：可考虑用 TensorRT / TorchInductor / 自定义 Triton 内核替代原项目的 TileLang 方案，核心融合点：QKV GEMM+bias、GEGLU MLP、残差+LayerNorm、原地 RoPE、滑窗注意力；配合按 `(batch桶, 长度桶)` 捕获 CUDA Graph 消除内核启动开销。

**验收标准**：加速路径与基线路径在固定测试集上数值差异在可接受范围内（如最大概率差 ≤0.05，argmax 一致率 ≥95%），并记录实测加速比。

---

## 阶段 7：服务化与生态集成（按需）

- HTTP 服务：暴露 `/predict`、`/predict/batch` 等接口，环境变量控制设备/预加载/线程数。
- 批处理接口：`predict_batch`（按 checkpoint + 问题 schema 分组共享前向）、`predict_long`（滑窗扫描长文档）。
- 生态集成（按需，优先级最低）：LangChain/LangGraph 节点、结构化 JSON Schema → 问题编译、MCP server。

---

## 风险与关键决策点清单

| 决策点 | 影响 | 建议 |
|---|---|---|
| 是否需要多语言/Router | 决定是否需要 mmBERT + 语言检测层，工作量差异大 | 先做单英语 checkpoint 验证闭环，再决定是否扩展 |
| 训练数据来源（外部数据集 vs 自建教师标注） | 决定阶段2工作量与最终效果上限 | 优先尝试复用/等价替代公开数据集，自建仅在必要时投入 |
| 教师模型选择（自建数据时） | 直接决定"教师自一致性天花板"，即模型效果上限 | 选用尽量强的教师模型，并测量自一致性作为预期上限 |
| 是否需要 GPU 融合内核加速 | 工程量大，收益仅体现在极致延迟场景 | 除非有明确 SLA 要求，建议后置或跳过 |
| 是否需要与某竞品做正面对比 | 涉及测评公正性和法律/公关风险 | 优先做"自身能力边界"披露，第三方对比数字务必注明来源与局限 |

---

## 附：与深度解读报告的对应关系

| 本方案阶段 | 对应深度解读报告章节 |
|---|---|
| 阶段1 架构实现 | 第2节 网络结构设计 |
| 阶段2 数据构建 | 第4节 数据如何构建 |
| 阶段3 RLCD训练 | 第5节 训练策略 |
| 阶段4 校准拟合 | 第5.4节 校准机制细节 |
| 阶段5 评测体系 | 第6节 评测方法与结果 |
| 阶段6 加速与导出 | 第2.4/2.6节 TileLang / ONNX |
| 阶段0/7 环境/服务化 | 第3节 使用的基础模型 / 第1节 技术架构总览 |
