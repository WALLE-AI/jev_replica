# Laya 复现 — 训练环境说明 与 模型/数据下载清单

## 一、训练环境：是否可以参照 NanoJev 的 uv 环境

**结论：可以作为基础环境直接复用（核心版本一致），但需要按 Laya 的需求增删依赖，不能原样照搬。**

### 1. 已核实的 NanoJev `.venv`（uv 管理）实际版本

路径：`/home/dataset1/gaojing/jev_replica/NanoJev/.venv`

```
Python        3.12.7  (uv 0.8.15 创建)
torch         2.14.0+cu126
transformers  5.17.0
safetensors   0.8.0
numpy         2.5.3
huggingface_hub 1.32.0
```

对照 Laya 自身 `pyproject.toml` 的依赖下限（`torch>=2.0.0`、`transformers>=4.48.0`、`safetensors>=0.4.0`、`huggingface_hub>=0.20.0`、`numpy>=1.20.0`）与 README 中"依赖下限"说明（"huggingface_hub 1.x、transformers 5.x、torch 2.14 都要求 Python ≥3.10"），**NanoJev 这套 uv 环境的核心四件套版本正好落在 Laya 要求的区间内**，可以直接作为基座。

### 2. 需要在 NanoJev 环境基础上做的调整

| 操作 | 包 | 原因 |
|---|---|---|
| **移除**（可选，节省体积） | `vizdoom`, `gymnasium`, `pygame*` | NanoJev 游戏相关依赖，Laya 复现用不到 |
| **新增** | `datasets` | 加载 MASSIVE / XNLI / typed-decisions 等 HF Hub 数据集，NanoJev venv 里**未安装**此包（其数据采集走自定义脚本，不用 `datasets` 库） |
| **新增（按需）** | `onnx`, `onnxruntime`, `onnxscript` | 若要复现 ONNX 导出/量化路径 |
| **新增（按需）** | `fastapi`, `uvicorn`, `python-multipart` | 若要复现 HTTP 服务化 |
| **新增（按需）** | `tilelang>=0.1.14` | 若要复现 GPU 融合内核加速路径（仅 CUDA） |
| **新增（按需）** | `pydantic>=2.0.0` | 若要复现 JSON Schema → 问题编译功能 |
| **确认** | `pip` 本身 | NanoJev venv 里 `pip` 模块缺失（uv 纯管理），日常用 `uv pip install` / `uv add` 即可，不必额外装 pip |

### 3. 建议的搭建方式

直接新建一个独立 uv 环境（不要复用/污染 NanoJev 的 venv，两个项目应该隔离），但**版本锁定与 NanoJev 一致**，可最大化复用已验证过的 CUDA/驱动兼容组合：

```bash
cd /path/to/laya-repro
uv venv --python 3.12
source .venv/bin/activate

# 核心四件套，版本对齐 NanoJev 已验证组合
uv pip install torch==2.14.0 transformers==5.17.0 safetensors==0.8.0 numpy==2.5.3 huggingface_hub==1.32.0

# Laya 复现新增
uv pip install datasets accelerate

# 按需（评测/导出/服务化）
uv pip install onnx onnxruntime onnxscript
uv pip install fastapi uvicorn python-multipart
uv pip install pydantic

# 仅当有 CUDA GPU 且要做融合内核加速时
uv pip install tilelang
```

若目标机器的 CUDA 版本与 NanoJev 所用的 `cu126` 不一致，改用 `--torch-backend=auto`（uv 会按本机驱动自动选包）：

```bash
uv pip install torch --torch-backend=auto
```

---

## 二、模型与数据手动下载清单

以下按"必需 / 按需"分类，全部给出 Hugging Face 仓库 ID，便于你手动 `huggingface-cli download` 或 `snapshot_download`。

### A. 基础编码器（必需，训练起点）

| 用途 | 建议仓库 | 说明 |
|---|---|---|
| 英语 checkpoint 编码器 | `answerdotai/ModernBERT-large` | Laya `english` 使用的编码器（421M） |
| 多语言 checkpoint 编码器 | `jhu-clsp/mmBERT-base` | Laya `multilingual` 使用的编码器（322M），若做多语言复现才需要 |

```bash
huggingface-cli download answerdotai/ModernBERT-large --local-dir ./models/ModernBERT-large
huggingface-cli download jhu-clsp/mmBERT-base --local-dir ./models/mmBERT-base
```

### B. Laya 官方发布的 checkpoint（按需，用作对比基线 / 不重新训练时直接用）

若只是想跑通 Router/推理链路做对比，不打算自己训练，可以直接下载官方权重：

| checkpoint | 仓库/子目录 | 说明 |
|---|---|---|
| `laya`（英语） | `convaiinnovations/laya`（root） | 421M |
| `laya-multilingual` | `convaiinnovations/laya`，`subfolder="multilingual"` | 322M |
| `laya-typed-decisions` | `convaiinnovations/laya`，`subfolder="typed-decisions"` | 421M，微调后 |

```bash
huggingface-cli download convaiinnovations/laya --local-dir ./models/laya
# multilingual / typed-decisions 是同一仓库的子目录，上面这条命令会一并拉取
```

也可单独下多语言变体仓库（README 中同时列出的独立入口）：
```bash
huggingface-cli download convaiinnovations/laya-multilingual --local-dir ./models/laya-multilingual
```

### C. 微调训练数据（必需，若要复现 RLCD 微调）

| 用途 | 仓库 | 说明 |
|---|---|---|
| typed-decisions 微调数据集 | `LocalLLaMA/typed-decisions`（config `"all"`） | 1,200 case / 6,000 决策训练集 + 400 case / 2,000 决策测试集，四个工作流（Agent Trace Observability / Customer Service / Invoice Processing / Security Incidents），标签为教师软概率分布 |

```bash
huggingface-cli download LocalLLaMA/typed-decisions --repo-type dataset --local-dir ./data/typed-decisions
```

> 若该数据集后续在 HF 上被下架/改名，需按"阶段2数据构建"方案里的"自建教师软标签"路径自行合成替代数据。

### D. 评测数据集（按需，用于零样本/多语言/鲁棒性评测）

| 数据集 | 仓库 | 用途 |
|---|---|---|
| MASSIVE Intent | `mteb/amazon_massive_intent` | 51 语言意图分类，choice 问题格式化 |
| XNLI | `facebook/xnli` | 15 语言自然语言推理，choice 问题格式化 |
| Banking77 | `banking77`（或 `PolyAI/banking77`） | 高基数选项场景对比（77 类） |
| Customer Support Tickets | `Tobi-Bueck/customer-support-tickets` | 客服工单分类工作流 |
| Enron Spam / Phishing | 视具体镜像而定（如 `SetFit/enron_spam`，钓鱼邮件集需另找，如 `ealvaradob/phishing-dataset` 系列） | 邮件+钓鱼检测工作流 |
| Toxic-Chat | `lmsys/toxic-chat` | 护栏（越狱检测）+ 内容审核（毒性检测）两个工作流 |
| MS MARCO | `microsoft/ms_marco` | RAG 段落相关性工作流 |
| GSM8K / MBPP | `openai/gsm8k`, `google-research-datasets/mbpp` | 模型路由（领域分类）工作流的正样本域 |
| AG News / DAIR Emotion / SST-5 | `ag_news`, `dair-ai/emotion`, `SetFit/sst5` | 仅用于与第三方公开 Jev 数字对比的英语零样本参照，非必需 |

```bash
huggingface-cli download mteb/amazon_massive_intent --repo-type dataset --local-dir ./data/massive_intent
huggingface-cli download facebook/xnli --repo-type dataset --local-dir ./data/xnli
huggingface-cli download lmsys/toxic-chat --repo-type dataset --local-dir ./data/toxic-chat
# 其余按需下载，注意部分数据集在 HF 上有多个同名镜像，下载前先确认字段结构（label 列名、split 划分）
```

### E. 教师模型（按需，仅"自建合成训练数据"路径需要）

若走"阶段2数据构建"里的"自建教师软标签"路线（不复用 `LocalLLaMA/typed-decisions`），需要一个较强的本地/API 教师模型来生成软概率分布标签。原项目"浏览器代理"案例使用的是本地 Qwen3-8B：

| 用途 | 建议仓库 |
|---|---|
| 本地教师模型（生成软标签/反向生成目标） | `Qwen/Qwen3-8B`（或更强的本地模型，视显存而定；也可换成任何你有 API 权限的更强模型） |

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

### F. 磁盘空间预估

| 项 | 大致体积 |
|---|---|
| ModernBERT-large | ~1.7 GB (fp32) |
| mmBERT-base | ~1.3 GB (fp32) |
| Laya 官方三 checkpoint（合计） | ~4-5 GB |
| typed-decisions 数据集 | 数十 MB（文本级数据集，很小） |
| MASSIVE + XNLI + 各评测数据集 | 合计约 1-3 GB |
| Qwen3-8B（若用作教师） | ~16 GB (bf16) |

建议预留 **≥30GB** 磁盘空间用于模型+数据全量下载。

---

## 三、下载命令统一入口（可直接复制执行）

```bash
export HF_HOME=$PWD/.hf_cache   # 可选：统一缓存目录，避免污染 ~/.cache

mkdir -p models data

# 基础编码器（必需）
huggingface-cli download answerdotai/ModernBERT-large --local-dir models/ModernBERT-large
huggingface-cli download jhu-clsp/mmBERT-base          --local-dir models/mmBERT-base

# 官方 checkpoint（对比基线，可选）
huggingface-cli download convaiinnovations/laya --local-dir models/laya

# 微调数据集（必需，若要复现 RLCD 微调）
huggingface-cli download LocalLLaMA/typed-decisions --repo-type dataset --local-dir data/typed-decisions

# 评测数据集（按需）
huggingface-cli download mteb/amazon_massive_intent --repo-type dataset --local-dir data/massive_intent
huggingface-cli download facebook/xnli              --repo-type dataset --local-dir data/xnli
huggingface-cli download lmsys/toxic-chat            --repo-type dataset --local-dir data/toxic-chat

# 教师模型（仅自建数据路径需要）
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

> 部分仓库可能需要先 `huggingface-cli login` 或设置 `HF_TOKEN`（尤其是需要接受协议的数据集）。执行前建议先用 `huggingface-cli download --dry-run`（或浏览器打开仓库页面）确认许可协议与字段结构，再批量下载。
