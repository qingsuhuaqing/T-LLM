# CLAUDE.md - Project Guide / 项目指南

> **Purpose / 目的**: This file provides comprehensive guidance to Claude Code for understanding and working with this Time-LLM Enhanced v4 research project.
> 本文件为 Claude Code 提供全面的项目理解和工作指南。

---

## Table of Contents / 目录

1. [Project Overview / 项目概述](#1-project-overview--项目概述)
2. [Project Status / 项目状态](#2-project-status--项目状态)
3. [v4 Architecture / v4 架构](#3-v4-architecture--v4-架构)
4. [Trainable Parameters / 可训练参数](#4-trainable-parameters--可训练参数)
5. [Code Structure / 代码结构](#5-code-structure--代码结构)
6. [Key Code Locations / 关键代码位置](#6-key-code-locations--关键代码位置)
7. [Common Commands / 常用命令](#7-common-commands--常用命令)
8. [CLI Parameters / 命令行参数](#8-cli-parameters--命令行参数)
9. [Checkpoint & Resume / 断点续训](#9-checkpoint--resume--断点续训)
10. [Troubleshooting / 故障排除](#10-troubleshooting--故障排除)

---

## 1. Project Overview / 项目概述

Time-LLM is a framework that reprograms frozen LLMs for time series forecasting (ICLR 2024). This project extends it with two innovation modules:

- **TAPR** (Trend-Aware Patch Router): Multi-scale prediction + cross-scale fusion
- **GRAM** (Global Retrieval-Augmented Memory): Historical pattern retrieval + adaptive gating

The project has evolved through 4 versions. **v4 is the current active version**.

### Version History / 版本历史

| Version | DM (下采样) | C2F (融合) | PVDR (检测) | AKG (门控) |
|---------|------------|-----------|------------|-----------|
| v3 | avg_pool(k不同) | softmax(W)直接加权 | 3-sigma规则 | 固定sigmoid(C) |
| **v4** | **多相x[:,p::k,:] (统一k=4)** | **MAD投票 + DWT趋势约束** | **逻辑回归 + 贝叶斯变点** | **PVDR信号阈值自适应** |

---

## 2. Project Status / 项目状态

### Completed / 已完成

| Task / 任务 | Status / 状态 |
|-------------|---------------|
| Paper reproduction (ETTh1, ETTm1) / 论文复现 | Done |
| Environment setup (6GB VRAM) / 环境搭建 | Done |
| GPT-2 + Qwen 2.5 3B (4-bit) adaptation / 模型适配 | Done |
| v3 implementation / v3 实现 | Done |
| **v4 implementation / v4 实现** | **Done** |
| v4 verification (shape/gradient/syntax) / v4 验证 | Done |

### Pending / 待完成

| Task / 任务 | Priority / 优先级 |
|-------------|-------------------|
| v4 training + results / v4 训练出结果 | High |
| Ablation experiments / 消融实验 | Medium |
| Thesis writing / 论文撰写 | Medium |

---

## 3. v4 Architecture / v4 架构

### Data Flow / 数据流

```
Input [B, T, N]
    │
    ▼
[Baseline Time-LLM] ──→ pred_s1 [B, pred_len, N]
    │                        │
    │                        ▼
    │              ┌─── DM v4 (PolyphaseMultiScale) ───┐
    │              │  S2(k=2,diff): 2 branches          │
    │              │  S3(k=4,identity): 4 branches      │
    │              │  S4(k=4,smooth): 4 branches        │
    │              │  S5(k=4,trend): 4 branches         │
    │              │  Total: 14 polyphase branches      │
    │              └────────────┬───────────────────────┘
    │                           ▼
    │              ┌─── C2F v4 (ExpertVotingFusion) ────┐
    │              │  Stage1: MAD intra-scale voting     │
    │              │    → consolidated [S1..S5]          │
    │              │  Stage2: DWT cross-scale trend      │
    │              │    → weighted_pred                  │
    │              └────────────┬───────────────────────┘
    │                           ▼
    ├──→ x_normed ──→ PVDR v4 (EnhancedDualRetriever)
    │              │  LogisticExtremeClassifier (6 params)
    │              │  BayesianChangePointDetector (3 params)
    │              │  4-mode retrieval: direct/extreme/
    │              │    turning_point/normal
    │              │  → hist_refs + pvdr_signals
    │              └────────────┬───────────────────────┘
    │                           ▼
    │              ┌─── AKG v4 (ThresholdAdaptiveGating) ┐
    │              │  effective_gate = sigmoid(C)         │
    │              │    × (1-extreme_adj)                 │
    │              │    × (1-cp_adj) × conf_adj           │
    │              │  3 learnable thresholds              │
    │              │  Fusion Matrix F [2S, pred_len]      │
    │              └────────────┬───────────────────────┘
    │                           ▼
    └──────────────────→ denormalize → output
```

### Loss Function / 损失函数

```
L_total = L_main + warmup × (
    λ_consist × L_consist   (DWT wavelet trend consistency)
  + λ_gate    × L_gate      (gate decisiveness regularization)
  + λ_expert  × L_expert    (expert correction rate, v4 NEW)
)
```

### Five Trainable Matrix Types / 五类可训练矩阵

| Matrix / 矩阵 | Shape | Description / 描述 |
|-------|-------|---------|
| Parameter Matrix Theta_s | 4 ScaleEncoders, ~100K | Per-scale Conv + FlattenHead |
| Weight Matrix W | [5, pred_len] | C2F cross-scale fusion weights |
| Connection Matrix C | [5, pred_len] | AKG base gate (history vs prediction) |
| Fusion Matrix F | [10, pred_len] | AKG final combination |
| Threshold params tau | 12 total | AKG(3) + LogisticRegression(6) + BayesianCP(3) |

---

## 4. Trainable Parameters / 可训练参数

```
┌──────────────────────────────────────────────────────────────┐
│  FROZEN (LLM Backbone)                                       │
│  ├── GPT-2: 124M params                                     │
│  └── Qwen 2.5 3B: ~1.5GB (4-bit)                            │
├──────────────────────────────────────────────────────────────┤
│  TRAINABLE — Baseline Time-LLM (~56M for GPT-2)             │
│  ├── PatchEmbedding: ~800 params                             │
│  ├── Mapping Layer: ~50M params                              │
│  ├── Reprogramming Layer: ~6M params                         │
│  └── FlattenHead: ~37K params                                │
├──────────────────────────────────────────────────────────────┤
│  TRAINABLE — v4 Innovation Modules (~103K)                   │
│  ├── DM (PolyphaseMultiScale):                               │
│  │   ├── 4 PatchEmbeddings (Conv1d)                          │
│  │   └── 4 ScaleEncoders (Conv1d×2 + FlattenHead)           │
│  ├── C2F (ExpertVotingFusion):                               │
│  │   └── weight_matrix [5, 96] = 480 params                 │
│  ├── PVDR (EnhancedDualRetriever):                           │
│  │   ├── LightweightPatternEncoder (Linear(5, 64))          │
│  │   ├── LogisticExtremeClassifier (Linear(5, 1)) = 6       │
│  │   ├── BayesianChangePointDetector = 3                    │
│  │   └── Memory bank thresholds [5] = 5                     │
│  └── AKG (ThresholdAdaptiveGating):                          │
│      ├── connection_matrix [5, 96] = 480                    │
│      ├── fusion_matrix [10, 96] = 960                       │
│      └── 3 threshold params = 3                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Code Structure / 代码结构

```
Time-LLM/
├── CLAUDE.md                           # This guide / 本指南
├── README.md                           # Technical documentation / 技术文档
├── claude_v4_readme.md                 # v4 architecture design spec / v4 架构设计文档
│
├── models/
│   ├── TimeLLM.py                      # Original baseline model / 原始基线模型
│   ├── TimeLLM_Enhanced_v3.py          # v3 model (deprecated) / v3 模型
│   └── TimeLLM_Enhanced_v4.py          # v4 model (ACTIVE) / v4 模型 (当前)
│
├── layers/
│   ├── Embed.py                        # PatchEmbedding
│   ├── StandardNorm.py                 # Instance normalization
│   ├── TAPR_v3.py                      # v3 TAPR (deprecated)
│   ├── TAPR_v4.py                      # v4: PolyphaseMultiScale + ExpertVotingFusion
│   ├── GRAM_v3.py                      # v3 GRAM (deprecated)
│   └── GRAM_v4.py                      # v4: EnhancedDualRetriever + ThresholdAdaptiveGating
│
├── run_main.py                         # Original training entry / 原始训练入口
├── run_main_enhanced_v3.py             # v3 training entry (deprecated)
├── run_main_enhanced_v4.py             # v4 training entry (ACTIVE) / v4 训练入口
│
├── scripts/
│   ├── TimeLLM_ETTh1_enhanced_v4.sh    # v4 training script (ACTIVE) / v4 训练脚本
│   ├── TimeLLM_ETTh1_enhanced_v3_2.sh  # v3 training script (reference)
│   └── ...
│
├── data_provider/
│   ├── data_factory.py                 # Dataset router
│   └── data_loader.py                  # Dataset classes
│
├── dataset/
│   ├── ETT-small/                      # ETTh1, ETTh2, ETTm1, ETTm2
│   └── prompt_bank/                    # Domain prompt texts
│
├── base_models/
│   ├── gpt2/                           # GPT-2 weights
│   └── openai-community/gpt2/         # GPT-2 (alternative path)
│
├── utils/
│   ├── tools.py                        # EarlyStopping, adjust_learning_rate, etc.
│   └── metrics.py                      # MSE, MAE, RMSE
│
├── md/                                 # Documentation directory / 文档目录
│   ├── chuangxin*.md                   # Innovation schemes / 创新方案
│   ├── work*.md                        # Technical analysis / 技术分析
│   └── wenti.md                        # Troubleshooting / 故障排除
│
└── Task_Requirements/                  # Thesis requirements / 毕设要求
    ├── task.md
    └── Reference-writing-style.md
```

---

## 6. Key Code Locations / 关键代码位置

### v4 Module Mapping / v4 模块对应

| Module / 模块 | File / 文件 | Class / 类 | Key Lines |
|-------|------|-------|-----------|
| DM v4 | `layers/TAPR_v4.py` | `PolyphaseMultiScale` | 185-328 |
| C2F v4 | `layers/TAPR_v4.py` | `ExpertVotingFusion` | 335-581 |
| DWT Helper | `layers/TAPR_v4.py` | `dwt_haar_1d()` | 165-178 |
| PVDR v4 | `layers/GRAM_v4.py` | `EnhancedDualRetriever` | 347-557 |
| AKG v4 | `layers/GRAM_v4.py` | `ThresholdAdaptiveGating` | 564-691 |
| LogisticExtreme | `layers/GRAM_v4.py` | `LogisticExtremeClassifier` | 68-99 |
| BayesianCP | `layers/GRAM_v4.py` | `BayesianChangePointDetector` | 106-225 |
| MemoryBank | `layers/GRAM_v4.py` | `EnhancedMultiScaleMemoryBank` | 232-341 |
| Main Model | `models/TimeLLM_Enhanced_v4.py` | `Model` | 96-579 |
| Baseline forward | `models/TimeLLM_Enhanced_v4.py` | `_baseline_forecast()` | 376-434 |
| v4 forecast flow | `models/TimeLLM_Enhanced_v4.py` | `forecast()` | 455-514 |
| Auxiliary loss | `models/TimeLLM_Enhanced_v4.py` | `compute_auxiliary_loss()` | 520-564 |

### Reused from v3 (verbatim) / 从 v3 复用

| Component | Source |
|-----------|--------|
| `SignalTransform` + `TRANSFORM_REGISTRY` | `TAPR_v3.py:24-94` → `TAPR_v4.py:31-101` |
| `ScaleEncoder` | `TAPR_v3.py:101-132` → `TAPR_v4.py:108-139` |
| `_create_patches()` | `TAPR_v3.py:139-151` → `TAPR_v4.py:146-158` |
| `LightweightPatternEncoder` | `GRAM_v3.py:25-55` → `GRAM_v4.py:31-61` |
| `FlattenHead` | `TimeLLM.py` → `TimeLLM_Enhanced_v4.py:41-53` |
| `ReprogrammingLayer` | `TimeLLM.py` → `TimeLLM_Enhanced_v4.py:60-89` |

---

## 7. Common Commands / 常用命令

### v4 Training (GPT-2, Full modules) / v4 训练

```bash
# Full v4 (TAPR + GRAM)
bash scripts/TimeLLM_ETTh1_enhanced_v4.sh

# Ablation: TAPR only
USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh

# Ablation: GRAM only
USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh

# Ablation: Baseline only
USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
```

### Direct Python Execution / 直接 Python 运行

```bash
python run_main_enhanced_v4.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model_comment v4_gpt2_TAPR_GRAM \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 512 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --batch_size 32 --d_model 64 --n_heads 8 --d_ff 128 \
  --llm_model GPT2 --llm_dim 768 --llm_layers 12 \
  --train_epochs 50 --patience 10 --learning_rate 0.0001 \
  --use_tapr --n_scales 5 --downsample_rates 1,2,4,4,4 \
  --use_gram --build_memory --top_k 5 --d_repr 64 \
  --warmup_steps 500 --seed 2021 --prompt_domain 1
```

### Environment Check / 环境验证

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from layers.TAPR_v4 import PolyphaseMultiScale, ExpertVotingFusion; print('TAPR_v4 OK')"
python -c "from layers.GRAM_v4 import EnhancedDualRetriever, ThresholdAdaptiveGating; print('GRAM_v4 OK')"
python -c "from models.TimeLLM_Enhanced_v4 import Model; print('Model OK')"
```

---

## 8. CLI Parameters / 命令行参数

### v4 New Parameters (added on top of v3) / v4 新增参数

| Parameter | Type | Default | Description / 描述 |
|-----------|------|---------|---------|
| `--mad_threshold` | float | 3.0 | MAD outlier detection multiplier for intra-scale voting |
| `--reliability_threshold` | float | 0.3 | Correction rate threshold for unreliable scale flag |
| `--reliability_penalty` | float | 0.3 | Weight multiplier for unreliable scales |
| `--lambda_expert` | float | 0.05 | Expert correction loss weight |
| `--cp_min_segment` | int | 16 | Bayesian CP detector minimum segment length |
| `--cp_max_depth` | int | 4 | Bayesian CP detector maximum recursion depth |

### TAPR Parameters (from v3) / TAPR 参数

| Parameter | Type | Default | Description / 描述 |
|-----------|------|---------|---------|
| `--use_tapr` | flag | False | Enable DM + C2F modules |
| `--n_scales` | int | 5 | Total scales including baseline |
| `--downsample_rates` | str | `1,2,4,4,4` | Per-scale rates (must match DM polyphase k) |
| `--lambda_consist` | float | 0.1 | Consistency loss weight |
| `--decay_factor` | float | 0.5 | Inference weight decay for inconsistent scales |

### GRAM Parameters (from v3) / GRAM 参数

| Parameter | Type | Default | Description / 描述 |
|-----------|------|---------|---------|
| `--use_gram` | flag | False | Enable PVDR + AKG modules |
| `--lambda_gate` | float | 0.01 | Gate regularization loss weight |
| `--similarity_threshold` | float | 0.8 | PVDR base retrieval threshold |
| `--extreme_threshold_reduction` | float | 0.2 | Threshold reduction for extreme data |
| `--top_k` | int | 5 | PVDR top-K retrieval count |
| `--d_repr` | int | 64 | Pattern representation dimension |
| `--build_memory` | flag | False | Build PVDR memory bank (required for first run) |

### Key Base Parameters / 关键基础参数

| Parameter | Default | Note |
|-----------|---------|------|
| `--llm_model` | LLAMA | Use `GPT2` for v4 training |
| `--llm_dim` | 4096 | Use `768` for GPT-2 |
| `--llm_layers` | 6 | GPT-2 has 12 total layers |
| `--batch_size` | 32 | Reduce if OOM |
| `--seq_len` | 96 | v4 shell script uses 512 |
| `--pred_len` | 96 | Prediction horizon |

---

## 9. Checkpoint & Resume / 断点续训

### Can Change (no shape conflict) / 可以修改

```
BATCH, NUM_WORKERS, LEARNING_RATE, TRAIN_EPOCHS, PATIENCE
LAMBDA_CONSIST, LAMBDA_GATE, LAMBDA_EXPERT (loss weights)
DECAY_FACTOR, MAD_THRESHOLD, RELIABILITY_THRESHOLD, RELIABILITY_PENALTY
WARMUP_STEPS, TEST_EPOCHS, SAVE_STEPS, SAVE_TOTAL_LIMIT
```

### Cannot Change (shape mismatch) / 不可修改

```
LLM_PATH, LLM_DIM, LLM_LAYERS, --llm_model
SEQ_LEN, LABEL_LEN, PRED_LEN
D_MODEL, D_FF, N_HEADS, ENC_IN, DEC_IN, C_OUT
USE_TAPR, USE_GRAM, N_SCALES, D_REPR, DOWNSAMPLE_RATES
CP_MIN_SEGMENT, CP_MAX_DEPTH (affects BayesianCP structure)
```

### Resume Notes / 续训注意事项

- `BUILD_MEMORY` must be `0` when resuming (memory bank saved in checkpoint)
- v3 and v4 checkpoints are **incompatible** (different module interfaces)
- Set `RESUME_FROM` to the checkpoint path, `RESUME_COUNTER` to override EarlyStopping count

### Inference from Checkpoint / 从 checkpoint 推理

```python
ckpt = torch.load('checkpoints/.../checkpoint')
model.load_state_dict(ckpt['model'])
model.eval()
```

Checkpoint fields are saved/loaded in `run_main_enhanced_v4.py` training loop via `safe_torch_save()`.

---

## 10. Troubleshooting / 故障排除

### OOM Priority / 显存不足处理优先级

1. Reduce `BATCH` (32 -> 16 -> 8 -> 4)
2. Reduce `SEQ_LEN` (512 -> 384 -> 256)
3. Reduce `D_FF` (128 -> 64 -> 32)
4. Reduce `D_MODEL` (64 -> 32)
5. Reduce `N_SCALES` (5 -> 3)
6. Reduce `TOP_K` (5 -> 3)
7. Disable GRAM (`USE_GRAM=0`)
8. Reduce `LLM_LAYERS` (12 -> 6 -> 4)

### Common Issues / 常见问题

| Issue / 问题 | Solution / 解决方案 |
|--------------|---------------------|
| `CUDA_HOME does not exist` | `pip uninstall deepspeed -y` |
| dtype mismatch (bfloat16 vs float32) | Fixed in model code (`.float()` conversion) |
| OOM with GPT-2 | GPT-2 is ~0.5GB; issue is batch/seq_len, not model size |
| v3 checkpoint in v4 | Incompatible; must retrain from scratch |
| CRLF line endings in shell scripts | `sed -i 's/\r$//' scripts/*.sh` |

### VRAM Estimate (GPT-2 + v4, batch=32) / 显存估算

```
GPT-2 (FP32, 12 layers):          ~0.5 GB
Trainable params:                  ~0.3 GB
DM v4 (4 ScaleEncoders):          ~2 MB
C2F v4 (weight_matrix + DWT):     < 1 KB
PVDR v4 (memory + detectors):     ~5.3 MB
AKG v4 (matrices + thresholds):   < 1 KB
Intermediate tensors:              ~0.8 GB
System overhead:                   ~1.0 GB
Total:                             ~2.6 GB
```

---

**Last Updated / 最后更新**: 2026-02-24
**Active Version / 当前版本**: v4
**Entry Point / 入口文件**: `run_main_enhanced_v4.py`
**Training Script / 训练脚本**: `scripts/TimeLLM_ETTh1_enhanced_v4.sh`
