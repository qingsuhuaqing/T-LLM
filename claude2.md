# claude2.md - 创新模块实现文档

> **Version**: 1.0
> **Date**: 2026-02-07
> **Author**: Claude Code (Opus 4.5)

---

## 目录

1. [概述](#1-概述)
2. [创新模块架构](#2-创新模块架构)
3. [TAPR 模块详解](#3-tapr-模块详解)
4. [GRA-M 模块详解](#4-gra-m-模块详解)
5. [集成架构](#5-集成架构)
6. [新增文件清单](#6-新增文件清单)
7. [命令行参数](#7-命令行参数)
8. [使用指南](#8-使用指南)
9. [消融实验设计](#9-消融实验设计)
10. [Checkpoint 机制](#10-checkpoint-机制)

---

## 1. 概述

### 1.1 问题背景

原始 Time-LLM 存在两个核心问题：

**A. 多尺度感知缺失（Scale Blindness）**
- Time-LLM 将时间序列切分为固定长度的 Patch 进行处理
- 缺乏对数据多分辨率特性的显式建模
- 直接将原始 Patch 输入 LLM，容易被高频噪声干扰，忽视长期宏观趋势

**B. 局部上下文局限（Local Context Limitation）**
- 主要依赖输入窗口（Look-back Window）内的信息进行推理
- 缺乏对全局历史数据的回溯能力
- 在面对突发事件或稀疏数据时表现不佳

### 1.2 解决方案

本项目提出 **"内结构化 + 外检索化"** 的双重增强框架：

| 模块 | 解决问题 | 核心思想 |
|------|----------|----------|
| **TAPR** | 多尺度感知缺失 | 多尺度分解 + 先粗后细预测 |
| **GRA-M** | 局部上下文局限 | 历史模式检索 + 自适应门控 |

### 1.3 创新点总结

1. **多尺度可分解混合 (DM)**: 通过多分辨率下采样捕获不同时间尺度的模式
2. **先粗后细预测头 (C2F)**: 层级决策机制，先判断趋势方向再细化预测
3. **模式-数值双重检索 (PVDR)**: 结合形状和数值相似度检索历史模式
4. **自适应知识门控 (AKG)**: 动态决定依赖历史经验还是当前推理

---

## 2. 创新模块架构

### 2.1 整体数据流

```
原始时序输入 [B, seq_len, n_vars]
        │
        ▼
┌───────────────────────────────────────┐
│  Instance Normalization (归一化)       │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Patch Embedding (分块嵌入)            │
│  [B*N, num_patches, d_model]          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  ★ TAPR (趋势感知尺度路由)             │ ◄── 新增模块1
│  ├─ MultiScaleDecomposition           │
│  │  └─ 多分辨率下采样 + 跨尺度注意力    │
│  └─ CoarseToFineHead                  │
│     └─ 方向→幅度→数值 层级预测         │
│                                       │
│  输出: enhanced_x, trend_embed        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  ★ GRA-M (全局检索增强记忆)            │ ◄── 新增模块2
│  ├─ PatternValueDualRetriever         │
│  │  └─ 形状+数值 双重相似度检索        │
│  └─ AdaptiveKnowledgeGating           │
│     └─ 门控融合: 历史经验 vs 当前推理   │
│                                       │
│  输出: enhanced_x, context_embed      │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Reprogramming Layer (重编程)          │
│  Cross-Attention: Q=patches, KV=words │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Prompt Construction (提示构建)        │
│  [trend_embed, retrieval_embed,       │
│   text_prompt, reprogrammed_patches]  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Frozen LLM Forward (冻结LLM前向)      │
│  GPT-2 / Qwen / LLAMA                 │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Output Projection (输出投影)          │
│  FlattenHead → [B, pred_len, n_vars]  │
└───────────────────────────────────────┘
        │
        ▼
预测结果 [B, pred_len, n_vars]
```

### 2.2 模块依赖关系

```
layers/
├── TAPR.py                    # TAPR 主模块
│   ├── MultiScaleDecomposition
│   ├── SeriesDecomposition
│   └── CoarseToFineHead
│
├── GRAM.py                    # GRA-M 主模块
│   ├── PatternEncoder
│   ├── PatternValueDualRetriever
│   ├── AdaptiveKnowledgeGating
│   └── RetrievalAugmentedPrompt
│
└── Embed.py                   # 原有模块 (PatchEmbedding)

models/
├── TimeLLM.py                 # 原始模型
└── TimeLLM_Enhanced.py        # 增强模型 (集成 TAPR + GRA-M)

run_main.py                    # 原始训练脚本
run_main_enhanced.py           # 增强训练脚本
```

---

## 3. TAPR 模块详解

### 3.1 模块定位

**TAPR (Trend-Aware Patch Router)** - 趋势感知尺度路由

解决 Time-LLM 的"尺度混淆"问题，通过多尺度分解和层级预测增强模型对不同时间分辨率模式的感知能力。

### 3.2 子模块 A: MultiScaleDecomposition

**功能**: 可分解多尺度混合

**核心思想**:
- 微观尺度: 高频波动、噪声、短期突变
- 中观尺度: 日内周期、工作日模式
- 宏观尺度: 长期趋势、季节性变化

**实现流程**:

```
输入: x [B*N, num_patches, d_model]
        │
        ▼
┌─────────────────────────────────────────────┐
│  Scale 0 (1x): 原始分辨率                    │
│  ├─ Conv1d 特征提取                          │
│  └─ 趋势-季节性分解                          │
│                                             │
│  Scale 1 (4x): 4倍下采样                     │
│  ├─ AvgPool1d(kernel=4)                     │
│  ├─ Conv1d 特征提取                          │
│  ├─ 趋势-季节性分解                          │
│  └─ Interpolate 上采样回原长度               │
│                                             │
│  Scale 2 (16x): 16倍下采样                   │
│  ├─ AvgPool1d(kernel=16)                    │
│  ├─ Conv1d 特征提取                          │
│  ├─ 趋势-季节性分解                          │
│  └─ Interpolate 上采样回原长度               │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  跨尺度注意力融合 (从粗到细)                  │
│  for i in [1, 0]:                           │
│      attn_out = CrossAttention(             │
│          query=scale_features[i],           │
│          key=scale_features[i+1],           │
│          value=scale_features[i+1]          │
│      )                                      │
│      scale_features[i] += attn_out          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  自适应权重融合                              │
│  weights = Softmax(learnable_scale_weights) │
│  multi_scale_out = Concat(all_scales)       │
│  output = Linear(multi_scale_out) + x       │
└─────────────────────────────────────────────┘
        │
        ▼
输出: enhanced_x, weighted_trend, all_trends
```

**关键代码位置**: `layers/TAPR.py:26-154`

### 3.3 子模块 B: CoarseToFineHead

**功能**: 先粗后细预测头

**核心思想**: 层级决策机制，"先判涨跌再细分比例"

**层级结构**:

```
Level 1 (方向分类):
┌─────────────────────────────────────────┐
│  输入: pooled_features [B, d_model]     │
│  输出: direction_logits [B, 3]          │
│        (上涨 / 平稳 / 下跌)              │
└─────────────────────────────────────────┘
        │
        ▼
Level 2 (幅度分类):
┌─────────────────────────────────────────┐
│  输入: [features, direction_probs]      │
│  输出: magnitude_logits [B, 4]          │
│        (强涨 / 弱涨 / 弱跌 / 强跌)       │
└─────────────────────────────────────────┘
        │
        ▼
Level 3 (条件回归):
┌─────────────────────────────────────────┐
│  输入: [features, trend_embedding]      │
│  输出: predictions [B, pred_len]        │
│        (具体数值预测)                    │
└─────────────────────────────────────────┘
```

**层次化一致性损失 (HCL)**:

```python
# 约束细粒度预测符合粗粒度趋势判断
expected_mag_from_dir = build_expected_distribution(direction_probs)
loss_hcl = KL_Divergence(magnitude_probs, expected_mag_from_dir)
```

**关键代码位置**: `layers/TAPR.py:194-341`

### 3.4 TAPR 辅助损失

```python
total_loss = (
    lambda_dir * CrossEntropy(direction_logits, direction_labels) +
    lambda_mag * CrossEntropy(magnitude_logits, magnitude_labels) +
    lambda_hcl * KL_Divergence(magnitude_probs, expected_from_direction)
)
```

**默认权重**: `lambda_trend = 0.1`

---

## 4. GRA-M 模块详解

### 4.1 模块定位

**GRA-M (Global Retrieval-Augmented Memory)** - 全局检索增强记忆

解决 Time-LLM 的"局部上下文局限"问题，通过检索历史相似模式并自适应融合，突破 LLM 上下文窗口限制。

### 4.2 子模块 A: PatternValueDualRetriever

**功能**: 模式-数值双重检索器

**核心机制**:
1. 检索与当前输入形状相似的历史模式
2. 同时获取历史模式后续的"未来值"作为参考
3. 结合形状(Shape)和数值(Value)进行相似度度量

**实现流程**:

```
离线阶段 (训练前):
┌─────────────────────────────────────────┐
│  build_memory(train_data, train_labels) │
│  ├─ PatternEncoder 编码所有训练样本      │
│  ├─ ValueEncoder 编码数值特征            │
│  ├─ 融合: w*shape + (1-w)*value         │
│  └─ 存储: memory_keys, memory_values    │
└─────────────────────────────────────────┘

在线阶段 (推理时):
┌─────────────────────────────────────────┐
│  retrieve(query)                        │
│  ├─ 编码当前查询                         │
│  ├─ 计算与记忆库的欧几里得距离            │
│  ├─ TopK 选择最相似的 K 个历史模式        │
│  ├─ Softmax 归一化相似度权重              │
│  └─ 返回: retrieved_refs, weights       │
└─────────────────────────────────────────┘
```

**相似度计算**:
```python
# 欧几里得距离 (时序数据的形状相似性)
distances = ||query_key - memory_keys||^2
similarity = -distances / temperature
```

**关键代码位置**: `layers/GRAM.py:94-237`

### 4.3 子模块 B: AdaptiveKnowledgeGating

**功能**: 自适应知识门控

**核心机制**: 动态决定"听从历史经验"还是"依赖当前推理"

**实现流程**:

```
输入: current_feat, retrieved_refs, similarity_weights
        │
        ▼
┌─────────────────────────────────────────┐
│  1. 编码当前特征                         │
│     current_encoded = Encoder(current)  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  2. 计算匹配度                           │
│     match_input = [current, retrieved]  │
│     match_scores = MatchScorer(input)   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  3. 加权聚合检索结果                     │
│     combined = similarity * match_scores│
│     weighted_ref = Sum(combined * refs) │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  4. 门控决策                             │
│     gate = Sigmoid(MLP([current, ref])) │
│     fused = gate*current + (1-gate)*ref │
└─────────────────────────────────────────┘
        │
        ▼
输出: fused_output, gate_weights, match_scores
```

**关键代码位置**: `layers/GRAM.py:240-368`

### 4.4 GRA-M 辅助损失

```python
# 门控熵损失 (鼓励多样性)
gate_entropy = -(g*log(g) + (1-g)*log(1-g)).mean()

# 门控一致性损失 (高置信度时门控稳定)
gate_consistency = |gate - 0.5| * confidence

total_loss = -0.01 * gate_entropy + lambda_ref * gate_consistency
```

**默认权重**: `lambda_retrieval = 0.1`

---

## 5. 集成架构

### 5.1 TimeLLM_Enhanced 模型

**文件位置**: `models/TimeLLM_Enhanced.py`

**主要修改点**:

```python
class Model(nn.Module):
    def __init__(self, configs):
        # ... 原有初始化 ...

        # 新增: 模块开关
        self.use_tapr = getattr(configs, 'use_tapr', True)
        self.use_gram = getattr(configs, 'use_gram', True)

        # 新增: TAPR 模块
        if self.use_tapr:
            self.tapr = TAPR(configs)
            self.trend_to_llm = nn.Linear(d_model, llm_dim)

        # 新增: GRA-M 模块
        if self.use_gram:
            self.gram = GRAM(configs)
            self.retrieval_prompt = RetrievalAugmentedPrompt(d_model, llm_dim)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Step 1-6: 原有流程 (归一化、Patch嵌入等)

        # 新增 Step 6.5: TAPR 处理
        if self.tapr is not None:
            enc_out, trend_embed, trend_info = self.tapr(enc_out)
            trend_embed_llm = self.trend_to_llm(trend_embed)

        # 新增 Step 6.7: GRA-M 处理
        if self.gram is not None:
            enc_out, context_embed, retrieval_info = self.gram(enc_out)
            retrieval_prompt_embed = self.retrieval_prompt(context_embed)

        # Step 7-11: 原有流程 (重编程、LLM、输出投影等)

        # 拼接增强嵌入
        llm_input = cat([trend_embed, retrieval_embed, prompt, enc_out])
```

### 5.2 LLM 输入构成

```
原始 Time-LLM:
[prompt_embeddings, reprogrammed_patches]

增强后:
[trend_embed, retrieval_embed, prompt_embeddings, reprogrammed_patches]
    ↑              ↑
    │              └── GRA-M 检索上下文 (可选)
    └── TAPR 趋势嵌入 (可选)
```

---

## 6. 新增文件清单

| 文件 | 类型 | 描述 |
|------|------|------|
| `layers/TAPR.py` | 新增 | TAPR 模块 (486行) |
| `layers/GRAM.py` | 新增 | GRA-M 模块 (627行) |
| `models/TimeLLM_Enhanced.py` | 新增 | 增强版模型 (521行) |
| `run_main_enhanced.py` | 新增 | 增强版训练脚本 (551行) |
| `scripts/TimeLLM_ETTh1_enhanced.sh` | 新增 | 训练脚本 (详细注释) |
| `claude2.md` | 新增 | 本文档 |

### 6.1 layers/__init__.py 更新

需要在 `layers/__init__.py` 中添加:

```python
from .TAPR import TAPR
from .GRAM import GRAM, RetrievalAugmentedPrompt
```

### 6.2 models/__init__.py 更新

需要在 `models/__init__.py` 中添加:

```python
from . import TimeLLM_Enhanced
```

---

## 7. 命令行参数

### 7.1 新增参数列表

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--use_tapr` | flag | False | 启用 TAPR 模块 |
| `--n_scales` | int | 3 | 多尺度分解的尺度数量 |
| `--lambda_trend` | float | 0.1 | 趋势辅助损失权重 |
| `--use_gram` | flag | False | 启用 GRA-M 模块 |
| `--top_k` | int | 5 | 检索的相似模式数量 |
| `--lambda_retrieval` | float | 0.1 | 检索辅助损失权重 |
| `--build_memory` | flag | False | 构建检索记忆库 |
| `--d_repr` | int | 128 | 模式表示向量维度 |

### 7.2 使用示例

```bash
# 启用完整增强模型
python run_main_enhanced.py \
  --use_tapr \
  --use_gram \
  --build_memory \
  --n_scales 3 \
  --top_k 5 \
  --lambda_trend 0.1 \
  --lambda_retrieval 0.1 \
  ... # 其他原有参数
```

---

## 8. 使用指南

### 8.1 首次训练

```bash
# 1. 确保已安装依赖
pip install torch transformers accelerate

# 2. 准备数据集
# dataset/ETT-small/ETTh1.csv

# 3. 运行训练脚本
bash scripts/TimeLLM_ETTh1_enhanced.sh
```

### 8.2 断点续训

```bash
# 修改脚本中的 RESUME_FROM 变量
RESUME_FROM="/path/to/checkpoint_step_1000/checkpoint.pt"

# 断点续训时应禁用记忆库重建
BUILD_MEMORY=0

# 运行脚本
bash scripts/TimeLLM_ETTh1_enhanced.sh
```

### 8.3 训练完成后推理

```python
import torch
from models.TimeLLM_Enhanced import Model

# 加载 checkpoint
ckpt = torch.load('checkpoints/.../checkpoint.pt')

# 初始化模型 (需要相同配置)
model = Model(args).float()
model.load_state_dict(ckpt['model'])
model.eval()

# 推理
with torch.no_grad():
    predictions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

**Checkpoint 包含的状态**:
- `model`: 模型权重
- `optimizer`: 优化器状态
- `scheduler`: 学习率调度器状态
- `epoch`: 当前 epoch
- `global_step`: 全局步数
- `best_score`: EarlyStopping 最佳分数
- `val_loss_min`: 最小验证损失
- `counter`: EarlyStopping 计数器
- `rng_state`: 随机数生成器状态
- `scaler`: AMP scaler 状态 (如果使用)
- `sampler_state`: 采样器状态

---

## 9. 消融实验设计

### 9.1 实验矩阵

| 实验 | TAPR | GRA-M | 预期效果 |
|------|------|-------|----------|
| Baseline | ✗ | ✗ | 原始 Time-LLM |
| Exp-1 | ✓ | ✗ | 验证多尺度分解效果 |
| Exp-2 | ✗ | ✓ | 验证检索增强效果 |
| Exp-3 | ✓ | ✓ | 完整增强模型 |

### 9.2 参数敏感性实验

**TAPR 参数**:
- `n_scales`: [2, 3, 4]
- `lambda_trend`: [0.05, 0.1, 0.2]

**GRA-M 参数**:
- `top_k`: [3, 5, 10]
- `lambda_retrieval`: [0.05, 0.1, 0.2]

### 9.3 评估指标

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- 趋势分类准确率 (Direction Accuracy)
- 门控权重分布分析

---

## 10. Checkpoint 机制

### 10.1 保存时机

1. **每 N 步保存** (通过 `--save_steps` 控制)
2. **每个 Epoch 结束时** (EarlyStopping 触发)

### 10.2 保存内容

```python
checkpoint_payload = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': current_epoch,
    'global_step': global_step,
    'best_score': early_stopping.best_score,
    'val_loss_min': early_stopping.val_loss_min,
    'counter': early_stopping.counter,
    'rng_state': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all()
    },
    'scaler': scaler.state_dict() if scaler else None,
    'sampler_state': sampler.state_dict() if sampler else None
}
```

### 10.3 恢复流程

```python
# 1. 加载 checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# 2. 恢复模型
model.load_state_dict(ckpt['model'])

# 3. 恢复优化器和调度器
optimizer.load_state_dict(ckpt['optimizer'])
scheduler.load_state_dict(ckpt['scheduler'])

# 4. 恢复 EarlyStopping 状态
early_stopping.best_score = ckpt['best_score']
early_stopping.val_loss_min = ckpt['val_loss_min']
early_stopping.counter = ckpt['counter']

# 5. 恢复随机状态
restore_rng_state(ckpt['rng_state'])

# 6. 恢复训练位置
start_epoch = ckpt['epoch']
global_step = ckpt['global_step']
```

---

## 附录 A: 代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| MultiScaleDecomposition | layers/TAPR.py | 26-154 |
| SeriesDecomposition | layers/TAPR.py | 157-191 |
| CoarseToFineHead | layers/TAPR.py | 194-341 |
| TAPR (主模块) | layers/TAPR.py | 344-485 |
| PatternEncoder | layers/GRAM.py | 27-91 |
| PatternValueDualRetriever | layers/GRAM.py | 94-237 |
| AdaptiveKnowledgeGating | layers/GRAM.py | 240-368 |
| GRAM (主模块) | layers/GRAM.py | 371-564 |
| RetrievalAugmentedPrompt | layers/GRAM.py | 567-626 |
| TimeLLM_Enhanced.Model | models/TimeLLM_Enhanced.py | 59-479 |
| 辅助损失计算 | models/TimeLLM_Enhanced.py | 426-465 |
| 记忆库构建 | models/TimeLLM_Enhanced.py | 467-478 |

---

## 附录 B: 显存优化建议

| 显存 | 推荐配置 |
|------|----------|
| 6GB | batch=8, seq_len=256, llm_layers=4, d_model=32 |
| 8GB | batch=12, seq_len=384, llm_layers=6, d_model=48 |
| 12GB | batch=24, seq_len=512, llm_layers=8, d_model=64 |
| 15GB+ | batch=32, seq_len=512, llm_layers=12, d_model=64 |

---

**文档结束**

*Generated by Claude Code (Opus 4.5) - 2026-02-07*
