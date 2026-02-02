# CLAUDE.md - Project Guide / 项目指南

> **Purpose / 目的**: This file provides comprehensive guidance to Claude Code for understanding and working with this Time-LLM research project.
> 本文件为 Claude Code 提供全面的项目理解和工作指南。

---

## Table of Contents / 目录

1. [Project Overview / 项目概述](#1-project-overview--项目概述)
2. [Project Status / 项目状态](#2-project-status--项目状态)
3. [Documentation Index / 文档索引](#3-documentation-index--文档索引)
4. [Innovation Points / 创新点](#4-innovation-points--创新点)
5. [Architecture & Trainable Parameters / 架构与可训练参数](#5-architecture--trainable-parameters--架构与可训练参数)
6. [Hardware Constraints / 硬件限制](#6-hardware-constraints--硬件限制)
7. [Common Commands / 常用命令](#7-common-commands--常用命令)
8. [Code Structure / 代码结构](#8-code-structure--代码结构)
9. [Troubleshooting / 故障排除](#9-troubleshooting--故障排除)

---

## 1. Project Overview / 项目概述

### What is Time-LLM? / Time-LLM 是什么？

Time-LLM is a framework that reprograms frozen Large Language Models (LLMs) for time series forecasting. Instead of training from scratch, it leverages the pattern recognition capabilities of pre-trained LLMs.

Time-LLM 是一个将冻结的大语言模型（LLM）重编程用于时间序列预测的框架。它不从零开始训练，而是利用预训练 LLM 的模式识别能力。

### Core Mechanisms / 核心机制

```
Time Series Data / 时序数据
    │
    ▼
[1. Patching] - Split into patches / 切分为块
    │
    ▼
[2. Patch Embedding] - Project to d_model dimension / 投影到 d_model 维度
    │
    ▼
[3. Reprogramming Layer] - Cross-attention alignment / 交叉注意力对齐
    │         Query: Patch embeddings (time series domain / 时序域)
    │         Key/Value: LLM word embeddings (text domain / 文本域)
    │
    ▼
[4. Prompt-as-Prefix] - Prepend statistical prompts / 前缀统计提示
    │         min, max, median, trend, top-5 lags (FFT)
    │
    ▼
[5. Frozen LLM Forward] - GPT-2/Qwen/LLAMA processes embeddings / 冻结LLM处理嵌入
    │
    ▼
[6. Output Projection] - FlattenHead maps to pred_len / 输出投影到预测长度
    │
    ▼
Prediction / 预测结果
```

### Key Insight / 核心洞察

**Input Reprogramming / 输入重编程**: Maps time series to LLM's word embedding space via learned projection, making LLM treat time series as "special text features".

通过可学习投影将时序映射到 LLM 词向量空间，使 LLM 将时序视为"特殊文本特征"。

**Prompt Reprogramming / 提示重编程**: Encodes statistical features (mean, variance, trend) as natural language prompts, activating LLM's inherent trend recognition capabilities.

将统计特征编码为自然语言提示，激活 LLM 内在的趋势识别能力。

---

## 2. Project Status / 项目状态

### Completed Work / 已完成工作

| Task / 任务 | Status / 状态 | Details / 详情 |
|-------------|---------------|----------------|
| Paper reproduction / 论文复现 | ✅ Done / 完成 | ETTh1, ETTm1 datasets verified / 已验证 |
| Environment setup / 环境搭建 | ✅ Done / 完成 | 6GB VRAM adaptation / 6GB显存适配 |
| Model replacement / 模型替换 | ✅ Done / 完成 | GPT-2 → Qwen 2.5 3B (4-bit) |
| Code fixes / 代码修复 | ✅ Done / 完成 | 4 critical bugs fixed / 4个关键bug已修复 |
| Innovation design / 创新设计 | ✅ Done / 完成 | 7 schemes documented / 7个方案已文档化 |
| Thesis proposal / 开题报告 | ✅ Done / 完成 | `report.md`, `baogao.md` |

### Pending Work / 待完成工作

| Task / 任务 | Priority / 优先级 | Document / 文档 |
|-------------|-------------------|-----------------|
| Implement Scheme 1 (Hybrid Model) / 实现方案一（混合模型） | 🥇 High / 高 | `md/chuangxin-jieshi1.md` |
| Implement Inter-Variate Attention / 实现变量间注意力 | 🥈 High / 高 | `md/chuangxin-shijian.md` |
| Ablation experiments / 消融实验 | 🥉 Medium / 中 | - |
| Thesis writing / 论文撰写 | Medium / 中 | - |

---

## 3. Documentation Index / 文档索引

### Root Directory Files / 根目录文件

| File / 文件 | Purpose / 用途 |
|-------------|----------------|
| `CLAUDE.md` | **This file** - Project guide for AI assistants / 本文件 - AI助手项目指南 |
| `README.md` | Original Time-LLM documentation / 原始Time-LLM文档 |
| `report.md` | **Thesis proposal (Chinese)** / 开题报告（中文） |
| `baogao.md` | **Thesis proposal (Formal)** / 开题报告（正式版） |

### `md/` Directory - Core Documentation / md目录 - 核心文档

#### Technical Analysis / 技术分析
| File / 文件 | Content / 内容 |
|-------------|----------------|
| `md/work.md` | Initial project notes / 初始项目笔记 |
| `md/work1.md` | Environment setup guide / 环境搭建指南 |
| `md/work2.md` | **Deep technical analysis** - Data flow, architecture, metrics / 深度技术分析 - 数据流、架构、指标 |
| `md/work3.md` | Complete technical documentation / 完整技术文档 |
| `md/Trainable-part.md` | **Trainable parameters analysis** (~56M params) / 可训练参数分析 |
| `md/tuidao.md` | Mathematical derivations / 数学推导 |

#### Innovation Schemes / 创新方案
| File / 文件 | Content / 内容 |
|-------------|----------------|
| `md/chuangxin.md` | **Innovation overview** - 7 schemes summary / 创新概述 - 7个方案总结 |
| `md/chuangxin-lilun.md` | **Theoretical analysis** - Each scheme's theory / 理论分析 - 每个方案的理论基础 |
| `md/chuangxin-shijian.md` | **Implementation guide** - Code for all 7 schemes / 实践指南 - 7个方案的代码 |
| `md/chuangxin-jieshi1.md` | **Scheme 1 deep dive** - Hybrid model with traditional methods / 方案一深度解析 - 传统模型混合 |

#### Operational Guides / 操作指南
| File / 文件 | Content / 内容 |
|-------------|----------------|
| `md/mingling.md` | Command parameters guide / 命令参数指南 |
| `md/mingling_2.md` | Extended command guide / 扩展命令指南 |
| `md/wenti.md` | **Troubleshooting** - All issues and solutions / 故障排除 - 所有问题和解决方案 |
| `md/yunxing-fuxian.md` | Reproduction guide / 复现指南 |
| `md/yunxing-wenda.md` | Q&A during running / 运行问答 |
| `md/yunfuwuqi.md` | Cloud server deployment / 云服务器部署 |

### `Task_Requirements/` Directory / 任务要求目录

| File / 文件 | Content / 内容 |
|-------------|----------------|
| `task.md` | Thesis task requirements / 毕设任务书 |
| `Reference-writing-style.md` | User's writing style reference / 用户写作风格参考 |

---

## 4. Innovation Points / 创新点

### Overview of 7 Schemes / 7个方案概述

| # | Scheme / 方案 | Key Idea / 核心思想 | Expected Gain / 预期收益 | Priority / 优先级 |
|---|---------------|---------------------|--------------------------|-------------------|
| 1 | **Hybrid Model** / 混合模型 | Traditional + LLM residual learning / 传统模型+LLM残差学习 | MSE ↓15-20%, interpretability / 可解释性 | 🥇 Highest / 最高 |
| 2 | **Multi-Scale Decomposition** / 多尺度分解 | Different scales for different patterns / 不同尺度捕获不同模式 | MSE ↓5-10% | Medium / 中 |
| 3 | **Frequency Enhancement** / 频域增强 | FFT decompose trend/seasonal / FFT分解趋势/季节性 | MSE ↓10-15% | 🥈 High / 高 |
| 4 | **Inter-Variate Attention** / 变量间注意力 | Attention on variable dimension / 在变量维度做注意力 | MSE ↓8-15% | 🥈 High / 高 |
| 5 | **Dynamic Prompt** / 动态提示 | Learnable prompt encoder / 可学习提示编码器 | MSE ↓10-20% | Medium / 中 |
| 6 | **MoE (Sparse Experts)** / 稀疏专家混合 | Multiple experts for different patterns / 多专家处理不同模式 | Capacity ↑4x | Low / 低 |
| 7 | **Vocabulary Initialization** / 词表初始化 | Task-specific initialization / 任务相关初始化 | Convergence ↑20-30% | Low / 低 |

### Scheme 1 Deep Dive (Recommended) / 方案一详解（推荐）

**Document / 文档**: `md/chuangxin-jieshi1.md`

#### Three Sub-approaches / 三个子方案

**A. Residual Learning Architecture / 残差学习架构** ⭐⭐⭐⭐⭐
```
Original Series → [ARIMA/ES] → Linear Prediction + Residual
                                        ↓
                               [Time-LLM learns residual]
                                        ↓
              Final = Linear Prediction + Nonlinear Prediction
```
- Theory: Hybrid ARIMA-LSTM paper proves effectiveness / 理论：混合ARIMA-LSTM论文证明有效性
- Expected: MSE ↓10-15% / 预期：MSE下降10-15%

**B. Segment-wise Adaptive Fusion / 分段自适应融合** ⭐⭐⭐⭐
```
Input Sequence → [Segment Analyzer] → Different weights per segment
                     ↓
    Segment 1 (periodic) → Traditional weight: 0.8
    Segment 2 (complex)  → Time-LLM weight: 0.9
    Segment 3 (trending) → Traditional weight: 0.7
```
- Theory: AMD Framework (Mixture-of-Experts) / 理论：AMD框架
- Expected: Additional MSE ↓3-5% / 预期：额外MSE下降3-5%

**C. Knowledge Distillation / 知识蒸馏** ⭐⭐⭐
- Soft label distillation: Learn teacher's distribution / 软标签蒸馏
- Decomposition distillation: Learn trend/seasonal decomposition / 分解蒸馏
- Behavior distillation: Learn direction/magnitude consistency / 行为蒸馏
- Theory: DE-TSMCL achieves MSE ↓24.2% on ETTm1 / 理论：DE-TSMCL在ETTm1上MSE下降24.2%

### Applicable Scenarios / 适用场景

| Scheme / 方案 | Best For / 最适合 | Dataset Examples / 数据集示例 |
|---------------|-------------------|------------------------------|
| Hybrid Model / 混合模型 | Periodic + interpretability needed / 周期性+需要可解释性 | ETT, Traffic |
| Frequency Enhancement / 频域增强 | Strong periodicity / 强周期性 | ETTh (24h/168h cycles) |
| Inter-Variate Attention / 变量间注意力 | Multi-variate with correlations / 多变量且有相关性 | Electricity, Weather |
| Multi-Scale / 多尺度 | Long sequences (720+) / 长序列 | ETTm (15-min data) |

---

## 5. Architecture & Trainable Parameters / 架构与可训练参数

### Frozen vs Trainable / 冻结与可训练

```
┌─────────────────────────────────────────────────────────────┐
│                    Time-LLM Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  FROZEN (LLM Backbone) / 冻结部分                            │
│  ├── GPT-2: 124M params / 参数                               │
│  ├── Qwen 2.5 3B: ~1.5GB (4-bit) / 4-bit量化后               │
│  └── Word Embeddings: Used as Key/Value / 用作K/V            │
├─────────────────────────────────────────────────────────────┤
│  TRAINABLE (~56M params for GPT-2) / 可训练部分              │
│  ├── PatchEmbedding: ~800 params                             │
│  │   └── Conv1d + Positional Embedding                       │
│  ├── Mapping Layer: ~50M params                              │
│  │   └── Linear(vocab_size=50257, num_tokens=1000)           │
│  ├── Reprogramming Layer: ~6M params                         │
│  │   └── Cross-Attention (Q: patches, K/V: mapped words)     │
│  └── FlattenHead: ~37K params                                │
│      └── Linear(head_nf, pred_len)                           │
└─────────────────────────────────────────────────────────────┘
```

### Data Shape Flow (ETTh1 Example) / 数据形状流（ETTh1示例）

| Stage / 阶段 | Shape / 形状 | Description / 描述 |
|--------------|--------------|---------------------|
| Input / 输入 | `[32, 96, 7]` | Batch=32, SeqLen=96, N_vars=7 |
| After Patching / 分块后 | `[224, 12, 16]` | B*N=224, num_patches=12, d_model=16 |
| After Reprogramming / 重编程后 | `[224, 12, 768]` | Mapped to llm_dim=768 |
| With Prompt / 加提示后 | `[224, 140, 768]` | Prompt(128) + Patches(12) |
| LLM Output / LLM输出 | `[224, 140, 768]` | GPT-2 forward |
| Final Output / 最终输出 | `[32, 96, 7]` | After FlattenHead + denormalize |

### Key Code Locations / 关键代码位置

| Component / 组件 | File / 文件 | Lines / 行号 |
|------------------|-------------|--------------|
| Main Model / 主模型 | `models/TimeLLM.py` | Full file / 全文件 |
| LLM Loading / LLM加载 | `models/TimeLLM.py` | 43-96 (Qwen), 83-117 (GPT-2) |
| LLM Freezing / LLM冻结 | `models/TimeLLM.py` | 163-164 |
| Prompt Construction / 提示构建 | `models/TimeLLM.py` | 207-230 |
| PatchEmbedding | `layers/Embed.py` | 160-186 |
| ReprogrammingLayer | `models/TimeLLM.py` | 267-305 |
| Instance Normalization / 实例归一化 | `layers/StandardNorm.py` | Full file / 全文件 |

---

## 6. Hardware Constraints / 硬件限制

### Current Setup / 当前配置

- **GPU**: NVIDIA GTX 1660 Ti (6GB VRAM)
- **OS**: Windows 11 + WSL2 (Ubuntu)
- **LLM**: Qwen 2.5 3B with 4-bit quantization / 4-bit量化

### VRAM Budget / 显存预算

| Component / 组件 | VRAM / 显存 |
|------------------|-------------|
| Qwen 2.5 3B (4-bit) | ~1.5 GB |
| Trainable params / 可训练参数 | ~0.5 GB |
| Intermediate tensors / 中间变量 | ~2.0 GB |
| System overhead / 系统占用 | ~1.0 GB |
| **Total / 总计** | **~5.0 GB** ✅ |

### Recommended Parameters / 推荐参数

| Parameter / 参数 | Value / 值 | Reason / 原因 |
|------------------|------------|---------------|
| `--batch_size` | 4-8 | VRAM limit / 显存限制 |
| `--llm_layers` | 6 | **Critical** - Prevents OOM / 防止溢出 |
| `--seq_len` | 96-512 | Balance accuracy/memory / 平衡精度和内存 |
| `--d_model` | 16-32 | Patch embedding dimension / 分块嵌入维度 |
| `--d_ff` | 32-128 | FFN dimension / FFN维度 |
| `--load_in_4bit` | True | Enable quantization / 启用量化 |

---

## 7. Common Commands / 常用命令

### Training with Qwen 2.5 3B (Recommended) / 使用Qwen 2.5 3B训练（推荐）

```bash
# WSL/Linux
bash ./scripts/TimeLLM_ETTm1_2.sh

# Or direct command / 或直接命令
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model TimeLLM \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size 4 \
  --llm_model QWEN \
  --llm_model_path "./base_models/Qwen2.5-3B" \
  --llm_dim 2048 \
  --llm_layers 6 \
  --load_in_4bit \
  --prompt_domain 1 \
  --train_epochs 10
```

### Training with GPT-2 (Fallback) / 使用GPT-2训练（备选）

```bash
accelerate launch --num_processes 1 --mixed_precision fp16 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 4 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 6 \
  --train_epochs 10
```

### Environment Verification / 环境验证

```bash
# Check CUDA / 检查CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Monitor GPU / 监控GPU
watch -n 1 nvidia-smi
```

---

## 8. Code Structure / 代码结构

```
Time-LLM/
├── CLAUDE.md                 # This guide / 本指南
├── README.md                 # Original docs / 原始文档
├── report.md                 # Thesis proposal / 开题报告
├── baogao.md                 # Thesis proposal (formal) / 开题报告（正式）
│
├── md/                       # Documentation / 文档目录
│   ├── chuangxin*.md        # Innovation schemes / 创新方案
│   ├── work*.md             # Technical analysis / 技术分析
│   ├── Trainable-part.md    # Trainable params / 可训练参数
│   ├── mingling*.md         # Command guides / 命令指南
│   └── wenti.md             # Troubleshooting / 故障排除
│
├── Task_Requirements/        # Thesis requirements / 毕设要求
│   ├── task.md              # Task description / 任务描述
│   └── Reference-writing-style.md
│
├── models/
│   └── TimeLLM.py           # Core model / 核心模型 ⭐
│
├── layers/
│   ├── Embed.py             # PatchEmbedding / 分块嵌入
│   └── StandardNorm.py      # Instance normalization / 实例归一化
│
├── data_provider/
│   ├── data_factory.py      # Dataset router / 数据集路由
│   └── data_loader.py       # Dataset classes / 数据集类
│
├── dataset/
│   ├── ETT-small/           # ETTh1, ETTh2, ETTm1, ETTm2
│   └── prompt_bank/         # Domain prompts / 领域提示
│
├── base_models/
│   ├── gpt2/                # GPT-2 weights / GPT-2权重
│   └── Qwen2.5-3B/          # Qwen 2.5 3B weights / Qwen权重
│
├── scripts/
│   ├── TimeLLM_ETTh1.sh     # ETTh1 training script / 训练脚本
│   ├── TimeLLM_ETTm1_2.sh   # WSL optimized script / WSL优化脚本
│   └── ...
│
├── run_main.py              # Main entry point / 主入口 ⭐
├── run_m4.py                # M4 benchmark entry / M4基准入口
└── utils/
    ├── tools.py             # Utilities / 工具函数
    └── metrics.py           # Evaluation metrics / 评估指标
```

---

## 9. Troubleshooting / 故障排除

### Common Issues / 常见问题

**Full troubleshooting guide / 完整故障排除指南**: `md/wenti.md`

| Issue / 问题 | Solution / 解决方案 |
|--------------|---------------------|
| `KeyError: 'qwen2'` | `pip install "transformers>=4.40.0" --upgrade` |
| `CUDA_HOME does not exist` | `pip uninstall deepspeed -y` |
| `Input type mismatch (bfloat16 vs float32)` | Fixed in `models/TimeLLM.py:297-298` |
| `AttributeError: 'content'` | Fixed in `run_main.py:133-134` |
| OOM (Out of Memory) / 显存溢出 | Reduce `--batch_size` to 2, `--llm_layers` to 4 |
| Slow convergence / 收敛慢 | Increase `--learning_rate` to 0.01 |

### Critical Code Fixes Applied / 已应用的关键修复

1. **`run_main.py` Line 107**: Removed `deepspeed_plugin` for single GPU / 移除deepspeed_plugin支持单GPU
2. **`run_main.py` Lines 133-134**: Moved `load_content()` before model creation / 将load_content移至模型创建前
3. **`models/TimeLLM.py` Lines 297-298**: Fixed dtype mismatch for 4-bit quantization / 修复4-bit量化数据类型不匹配

---

## Quick Reference Card / 快速参考卡

```
┌─────────────────────────────────────────────────────────────┐
│                    Time-LLM Quick Reference                  │
├─────────────────────────────────────────────────────────────┤
│ Innovation Docs / 创新文档:                                  │
│   md/chuangxin.md          - Overview / 概述                 │
│   md/chuangxin-lilun.md    - Theory / 理论                   │
│   md/chuangxin-shijian.md  - Implementation / 实现           │
│   md/chuangxin-jieshi1.md  - Scheme 1 Deep Dive / 方案一详解 │
├─────────────────────────────────────────────────────────────┤
│ Technical Docs / 技术文档:                                   │
│   md/work2.md              - Deep analysis / 深度分析        │
│   md/Trainable-part.md     - Trainable params / 可训练参数   │
│   md/wenti.md              - Troubleshooting / 故障排除      │
├─────────────────────────────────────────────────────────────┤
│ Key Parameters / 关键参数:                                   │
│   --llm_layers 6           ⭐ CRITICAL for 6GB VRAM          │
│   --batch_size 4           ⭐ CRITICAL for 6GB VRAM          │
│   --load_in_4bit           ⭐ Enable 4-bit quantization      │
│   --prompt_domain 1        Load domain prompts               │
├─────────────────────────────────────────────────────────────┤
│ Priority Innovation / 优先创新:                              │
│   1. Hybrid Model (Scheme 1) - md/chuangxin-jieshi1.md      │
│   2. Inter-Variate Attention - md/chuangxin-shijian.md §4   │
│   3. Frequency Enhancement - md/chuangxin-shijian.md §3     │
└─────────────────────────────────────────────────────────────┘
```

---

**Last Updated / 最后更新**: 2026-01-11
**Project Status / 项目状态**: Innovation implementation phase / 创新实现阶段
**Hardware / 硬件**: NVIDIA GTX 1660 Ti (6GB) + Qwen 2.5 3B (4-bit)
- 3. 训练完成后推理
# 加载 checkpoint (不是 checkpoint_step_N)
ckpt = torch.load('checkpoints/.../checkpoint')
model.load_state_dict(ckpt['model'])
model.eval()这个训练完成后推理,ckpt相关参数,在哪里体现的?