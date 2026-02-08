# claude3.md - Time-LLM Enhanced 技术文档

> **版本**: 2026-02-08
> **状态**: 已修复所有已知问题，待消融实验验证
> **目的**: 记录增强版 Time-LLM 的完整架构、数据流、修改记录和消融实验指南

---

## 目录

1. [架构总览](#1-架构总览)
2. [模块详解](#2-模块详解)
3. [数据流详解](#3-数据流详解)
4. [修改记录](#4-修改记录)
5. [命令参数说明](#5-命令参数说明)
6. [消融实验指南](#6-消融实验指南)
7. [Checkpoint 与推理](#7-checkpoint-与推理)
8. [故障排除](#8-故障排除)

---

## 1. 架构总览

### 1.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Time-LLM Enhanced 架构                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: [B, seq_len, n_vars]                                                │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ Instance Norm   │  消除样本间尺度差异                                     │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ Channel Indep.  │  [B, T, N] → [B*N, T, 1]                               │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ├──────────────────────────────────────────────────┐               │
│           │                                                  │               │
│           ▼                                                  ▼               │
│  ┌─────────────────┐                              ┌─────────────────┐       │
│  │ Patch Embedding │                              │ 统计特征提取     │       │
│  │ [B*N, P, d_model]                              │ min/max/median  │       │
│  └────────┬────────┘                              │ trend/lags      │       │
│           │                                        └────────┬────────┘       │
│           │                                                 │               │
│           ▼                                                 ▼               │
│  ┌─────────────────────────────────────┐          ┌─────────────────┐       │
│  │ ★ TAPR (Trend-Aware Patch Router)  │          │ Text Prompt     │       │
│  │   ├── DM: 多尺度分解                │          │ 构建            │       │
│  │   └── C2F: 趋势分类                 │          └────────┬────────┘       │
│  └────────┬────────────────────────────┘                   │               │
│           │ trend_embed_llm [B*N, 1, d_llm]                │               │
│           │                                                 │               │
│           ▼                                                 │               │
│  ┌─────────────────────────────────────┐                   │               │
│  │ ★ GRA-M (Retrieval-Augmented Memory)│                   │               │
│  │   ├── PVDR: 检索相似模式            │                   │               │
│  │   └── AKG: 自适应门控               │                   │               │
│  └────────┬────────────────────────────┘                   │               │
│           │ retrieval_embed [B*N, 2, d_llm]                │               │
│           │                                                 │               │
│           ▼                                                 │               │
│  ┌─────────────────┐                                        │               │
│  │ Reprogramming   │  跨模态注意力对齐                       │               │
│  │ Layer           │  [B*N, P, d_llm]                       │               │
│  └────────┬────────┘                                        │               │
│           │                                                 │               │
│           ▼                                                 ▼               │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │              LLM Input 拼接                               │              │
│  │  [trend_embed, retrieval_embed, prompt_embed, patch_embed]│              │
│  └────────────────────────┬──────────────────────────────────┘              │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────┐                                                        │
│  │   Frozen LLM    │  GPT-2 / Qwen / LLAMA (冻结)                           │
│  │   Forward       │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ FlattenHead     │  输出投影                                              │
│  │ [B, pred_len, N]│                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ ★ C2F Fusion   │  趋势引导融合 (仅训练时)                                │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                        │
│  │ Instance Denorm │  反归一化                                              │
│  └────────┬────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  Output: [B, pred_len, n_vars]                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 模块层级结构

```
TimeLLM_Enhanced (models/TimeLLM_Enhanced.py)
│
├── 原有模块 (Time-LLM Baseline)
│   ├── normalize_layers (Instance Normalization)
│   ├── patch_embedding (PatchEmbedding)
│   ├── mapping_layer (词表映射)
│   ├── reprogramming_layer (跨模态注意力)
│   ├── output_projection (FlattenHead)
│   └── llm_model (GPT-2/Qwen/LLAMA, 冻结)
│
├── ★ 创新模块一: TAPR (layers/TAPR.py)
│   │
│   ├── TAPR-DM: MultiScaleDecomposition (可分解多尺度)
│   │   ├── scale_encoders (深度可分离卷积)
│   │   ├── decomp_layers (SeriesDecomposition)
│   │   ├── cross_scale_attn (跨尺度注意力)
│   │   └── scale_weights (可学习尺度权重)
│   │
│   └── TAPR-C2F: CoarseToFineHead (先粗后细预测头)
│       ├── direction_head (Level 1: 方向分类)
│       ├── magnitude_head (Level 2: 幅度分类)
│       ├── regression_head (Level 3: 条件回归)
│       ├── trend_embeddings (趋势类别嵌入)
│       ├── trend_thresholds (可学习阈值)
│       └── gate (趋势引导门控)
│
├── ★ 创新模块二: GRA-M (layers/GRAM.py)
│   │
│   ├── GRA-M-PVDR: PatternValueDualRetriever (模式-数值双重检索)
│   │   ├── pattern_encoder (PatternEncoder, Transformer)
│   │   ├── value_encoder (数值统计编码)
│   │   ├── shape_weight (形状-数值权重)
│   │   └── memory_keys/values (检索记忆库)
│   │
│   └── GRA-M-AKG: AdaptiveKnowledgeGating (自适应知识门控)
│       ├── current_encoder (当前特征编码)
│       ├── retrieved_encoder (检索特征编码)
│       ├── match_scorer (匹配度评估)
│       ├── gate_network (门控网络)
│       └── cross_attention (交叉注意力融合)
│
├── ★ C2F趋势融合 (TimeLLM_Enhanced._apply_trend_fusion)
│   ├── trend_fusion (融合网络)
│   └── trend_gate (融合门控)
│
└── RetrievalAugmentedPrompt (检索增强Prompt构建)
    ├── context_to_llm (上下文投影)
    └── confidence_embed (置信度编码)
```

---

## 2. 模块详解

### 2.1 TAPR (Trend-Aware Patch Router) - 趋势感知尺度路由

**所属文件**: `layers/TAPR.py`

**设计理念**: 时间序列包含多种频率成分，单一尺度难以同时捕获短期波动和长期趋势。

#### 2.1.1 TAPR-DM: MultiScaleDecomposition

```python
# 位置: layers/TAPR.py:32-184

class MultiScaleDecomposition(nn.Module):
    """
    多尺度分解模块

    核心思想:
    - 微观尺度 (1x): 高频波动、噪声、短期突变
    - 中观尺度 (4x): 日内周期、工作日模式
    - 宏观尺度 (16x): 长期趋势、季节性变化

    处理流程:
    1. 平均池化下采样到各尺度
    2. 深度可分离卷积提取特征
    3. 趋势-季节性分解 (SeriesDecomposition)
    4. 上采样回原始长度
    5. 跨尺度注意力融合 (粗→细)
    6. 可学习权重加权融合
    """
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_scales` | 3 | 尺度数量 (1x, 4x, 16x) |
| `kernel_sizes` | [1, 4, 16] | 各尺度下采样核大小 |

**输出**:
- `multi_scale_out`: 融合后特征 `[B*N, L, d_model]`
- `weighted_trend`: 加权趋势分量
- `trend_features`: 各尺度趋势列表

#### 2.1.2 TAPR-C2F: CoarseToFineHead

```python
# 位置: layers/TAPR.py:231-402

class CoarseToFineHead(nn.Module):
    """
    先粗后细预测头

    层级决策机制:
    - Level 1 (3分类): 上涨 / 下跌 / 平稳
    - Level 2 (4分类): 强涨 / 弱涨 / 弱跌 / 强跌
    - Level 3 (回归): 基于趋势类别的条件回归

    创新点:
    1. 人类预测思路: 先判大方向，再细化数值
    2. 层次化一致性损失 (HCL): 强制细粒度符合粗粒度
    3. 可学习阈值: 自适应不同数据集分布
    """
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_trend_classes` | 4 | 趋势分类数 |
| `trend_thresholds` | [-0.02, -0.005, 0.005, 0.02] | 可学习分类阈值 |

### 2.2 GRA-M (Global Retrieval-Augmented Memory) - 全局检索增强记忆

**所属文件**: `layers/GRAM.py`

**设计理念**: 利用历史相似模式作为预测参考，突破LLM上下文窗口限制。

#### 2.2.1 GRA-M-PVDR: PatternValueDualRetriever

```python
# 位置: layers/GRAM.py:110-262

class PatternValueDualRetriever(nn.Module):
    """
    模式-数值双重检索器

    核心机制:
    1. 形状相似性: Transformer编码 + CLS token聚合
    2. 数值相似性: 统计特征编码
    3. 双重融合: 可学习权重 shape_weight
    4. Top-K检索: 欧几里得距离 → softmax权重
    """
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_repr` | 128 | 表示向量维度 |
| `top_k` | 5 | 检索数量 |
| `temperature` | 0.1 | softmax温度 |
| `max_memory_size` | 10000 | 最大记忆容量 |

#### 2.2.2 GRA-M-AKG: AdaptiveKnowledgeGating

```python
# 位置: layers/GRAM.py:265-403

class AdaptiveKnowledgeGating(nn.Module):
    """
    自适应知识门控

    动态决策:
    - gate_weight → 1: 依赖当前推理
    - gate_weight → 0: 依赖历史经验

    处理流程:
    1. 编码当前特征和检索特征
    2. 计算每个检索结果的匹配度
    3. 门控网络决定融合权重
    4. 交叉注意力融合检索结果
    5. 残差连接保留原始信息
    """
```

#### 2.2.3 _resize_last_dim 工具函数

```python
# 位置: layers/GRAM.py:464-494

@staticmethod
def _resize_last_dim(tensor, target_dim):
    """
    将任意形状张量的最后一维统一对齐到 target_dim

    设计目的:
    - 解决 memory_values 维度不一致问题
    - 避免 "...x1 and 64x64" 这类错误
    - 兼容旧 checkpoint 加载

    实现:
    - 使用 F.interpolate 进行线性插值
    - 保持前面维度不变
    """
```

### 2.3 C2F趋势融合

**所属文件**: `models/TimeLLM_Enhanced.py:491-553`

```python
def _apply_trend_fusion(self, dec_out, trend_info, n_vars):
    """
    让趋势分类结果真正影响最终预测

    设计理念:
    - 趋势分类提供未来走势判断
    - 门控机制控制融合强度
    - 高置信度趋势应有更大影响

    处理流程:
    1. 获取趋势分类概率 (4类)
    2. 计算预期变化因子: trend_factors = [-0.02, -0.005, 0.005, 0.02]
    3. 门控权重控制融合强度 (最大50%)
    4. 时间渐进权重 (开始小，结束大)
    5. 乘法调整: dec_out * (1 + adjustment)
    """
```

---

## 3. 数据流详解

### 3.1 形状变化流程 (ETTh1 示例)

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 | `[32, 512, 7]` | B=32, seq_len=512, n_vars=7 |
| Instance Norm | `[32, 512, 7]` | 消除尺度差异 |
| Channel Independence | `[224, 512, 1]` | B*N=224 |
| Patch Embedding | `[224, 63, 64]` | P=63 patches, d_model=64 |
| TAPR-DM | `[224, 63, 64]` | 多尺度融合 |
| TAPR-C2F | `[224, 4]` (logits) | 趋势分类 |
| GRA-M检索 | `[224, 5, 96, 64]` | top_k=5, pred_len=96 |
| GRA-M门控 | `[224, 63, 64]` | 融合后特征 |
| Reprogramming | `[224, 63, 768]` | d_llm=768 |
| LLM Input | `[224, 194, 768]` | trend(1) + retrieval(2) + prompt(128) + patch(63) |
| LLM Output | `[224, 194, 768]` | GPT-2 forward |
| FlattenHead | `[32, 96, 7]` | 输出预测 |
| C2F Fusion | `[32, 96, 7]` | 趋势调整 (训练时) |
| Denorm | `[32, 96, 7]` | 反归一化 |

### 3.2 辅助损失计算流程

```
训练时:
├── 主损失: MSE(outputs, batch_y_target)
│
├── TAPR辅助损失 (lambda_trend=0.5):
│   ├── loss_direction: CrossEntropy (方向分类)
│   ├── loss_magnitude: CrossEntropy (幅度分类)
│   └── loss_hcl: KL-Divergence (层次一致性)
│
└── GRA-M辅助损失 (lambda_retrieval=0.3):
    ├── gate_entropy: 鼓励门控多样性
    └── gate_consistency: 高置信度时鼓励决断性

总损失 = MSE + warmup_factor * (TAPR损失 + GRA-M损失)

Warmup机制:
- warmup_steps = 500
- warmup_factor = min(1.0, current_step / 500)
- 训练初期降低辅助损失权重，先学主任务
```

---

## 4. 修改记录

### 4.1 问题修复清单

| 序号 | 问题 | 文件:行号 | 修复方案 |
|------|------|-----------|----------|
| 1 | `torch.tensor(0.0)` device不匹配 | TAPR.py:525, GRAM.py:657 | 添加 `device=y_true.device` |
| 2 | y_true维度不匹配 | TAPR.py:534-547 | `y_true.permute(0,2,1).reshape(B*N, P)` |
| 3 | 记忆库特征不合理 | GRAM.py:524-562 | 使用统计特征替代原始输入mean |
| 4 | 辅助损失权重过小 | TimeLLM_Enhanced.py:107-108 | 0.1→0.5 (trend), 0.1→0.3 (retrieval) |
| 5 | C2F预测未使用 | TimeLLM_Enhanced.py:491-553 | 新增 `_apply_trend_fusion()` |
| 6 | memory_values维度错误 | GRAM.py:464-494 | 新增 `_resize_last_dim()` |
| 7 | warmup未传递 | run_main_enhanced.py:458-461 | 传递 `current_step=global_step` |

### 4.2 超参数调整

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `learning_rate` | 0.001 | 0.0005 | 防止过拟合 |
| `dropout` | 0.1 | 0.15 | 增强正则化 |
| `patience` | 10 | 15 | 更多收敛时间 |
| `lambda_trend` | 0.1 | 0.5 | 让趋势分类起作用 |
| `lambda_retrieval` | 0.1 | 0.3 | 让检索模块起作用 |
| `lambda_dir` | 0.1 | 0.3 | 内部方向损失权重 |
| `lambda_mag` | 0.1 | 0.3 | 内部幅度损失权重 |
| `lambda_hcl` | 0.05 | 0.1 | 内部一致性损失权重 |
| `label_smoothing` | 0 | 0.1 | 防止过拟合 |

---

## 5. 命令参数说明

### 5.1 新增模块参数

#### TAPR 相关

| 参数 | 类型 | 默认值 | 说明 | 是否发挥作用 |
|------|------|--------|------|-------------|
| `--use_tapr` | flag | False | 启用TAPR模块 | ✅ 控制模块开关 |
| `--n_scales` | int | 3 | 多尺度分解尺度数量 | ⚠️ 需验证 |
| `--lambda_trend` | float | 0.5 | 趋势辅助损失权重 | ✅ 已增大 |

**`N_SCALES` 参数详解**:
- 默认值 `3` 表示: 1x (原始) + 4x (中尺度) + 16x (粗尺度)
- 更多尺度可捕获更丰富模式，但增加计算量
- 消融建议: 测试 2, 3, 4 对比效果

**`LAMBDA_TREND` 参数详解**:
- 控制趋势分类损失对总损失的贡献
- 原值 0.1 过小，辅助损失几乎不起作用 (0.5 * 0.1 * 0.1 ≈ 0.005)
- 新值 0.5，有效损失约 0.5 * 0.5 * (0.3+0.3+0.1) ≈ 0.175

#### GRA-M 相关

| 参数 | 类型 | 默认值 | 说明 | 是否发挥作用 |
|------|------|--------|------|-------------|
| `--use_gram` | flag | False | 启用GRA-M模块 | ✅ 控制模块开关 |
| `--top_k` | int | 5 | 检索相似模式数量 | ⚠️ 需验证 |
| `--lambda_retrieval` | float | 0.3 | 检索辅助损失权重 | ✅ 已增大 |
| `--build_memory` | flag | False | 是否构建记忆库 | ✅ 首次必须启用 |
| `--d_repr` | int | 128 | 模式表示向量维度 | ⚠️ 需验证 |

**`TOP_K` 参数详解**:
- 控制每次检索返回的相似历史模式数量
- 更多检索结果提供更丰富参考，但可能引入噪声
- 消融建议: 测试 3, 5, 10 对比效果

**`BUILD_MEMORY` 参数详解**:
- 首次训练必须设为 1 (True)
- 从训练数据构建历史模式库 (最多5000样本)
- **断点续训时应设为 0**，避免重建覆盖

**`D_REPR` 参数详解**:
- 模式表示向量的维度
- 较大维度可表示更复杂模式，但增加内存占用
- 建议: 与 d_ff 保持一致或更小

### 5.2 训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--learning_rate` | 0.0005 | 降低防止过拟合 |
| `--dropout` | 0.15 | 增强正则化 |
| `--patience` | 15 | 更多收敛时间 |
| `--batch_size` | 32 (15GB) / 8 (6GB) | 根据显存调整 |
| `--llm_layers` | 12 (15GB) / 6 (6GB) | 使用的LLM层数 |

### 5.3 Checkpoint 参数

| 参数 | 说明 |
|------|------|
| `--save_steps` | 每N步保存 (0=关闭) |
| `--save_total_limit` | 只保留最近N个checkpoint |
| `--resume_from_checkpoint` | 断点续训路径 |
| `--resume_counter` | 覆盖EarlyStopping计数 |

---

## 6. 消融实验指南

### 6.1 实验配置

```bash
# 修改 scripts/TimeLLM_ETTh1_enhanced.sh 中的以下变量

# 【实验1】仅TAPR模块 (验证多尺度分解效果)
USE_TAPR=1
USE_GRAM=0

# 【实验2】仅GRA-M模块 (验证检索增强效果)
USE_TAPR=0
USE_GRAM=1

# 【实验3】完整模型 (TAPR + GRA-M)
USE_TAPR=1
USE_GRAM=1

# 【实验4】基线模型 (原始Time-LLM)
USE_TAPR=0
USE_GRAM=0
# 或直接使用 run_main.py
```

### 6.2 参数敏感性实验

```bash
# N_SCALES 敏感性
N_SCALES=2  # 实验A
N_SCALES=3  # 实验B (默认)
N_SCALES=4  # 实验C

# TOP_K 敏感性
TOP_K=3   # 实验A
TOP_K=5   # 实验B (默认)
TOP_K=10  # 实验C

# LAMBDA_TREND 敏感性
LAMBDA_TREND=0.3  # 实验A
LAMBDA_TREND=0.5  # 实验B (默认)
LAMBDA_TREND=0.7  # 实验C

# LAMBDA_RETRIEVAL 敏感性
LAMBDA_RETRIEVAL=0.2  # 实验A
LAMBDA_RETRIEVAL=0.3  # 实验B (默认)
LAMBDA_RETRIEVAL=0.5  # 实验C
```

### 6.3 判断参数是否发挥作用

**方法1: 观察辅助损失**
```
训练日志中观察:
- aux_loss 应该显著 > 0 (如 0.1~0.5)
- 如果 aux_loss ≈ 0.02，说明权重仍然过小

正常情况:
Epoch: 1 | Train Loss: 0.XXXX | Aux Loss: 0.15~0.30
```

**方法2: 对比消融实验结果**
```
| 配置 | MSE | MAE | 说明 |
|------|-----|-----|------|
| Baseline | 0.40 | 0.42 | 原始Time-LLM |
| +TAPR | 0.38 | 0.40 | 若下降则有效 |
| +GRA-M | 0.37 | 0.39 | 若下降则有效 |
| +Both | 0.35 | 0.38 | 若最低则组合有效 |
```

**方法3: 检查趋势分类准确率**
```python
# 在训练日志中添加 (可选):
direction_acc = (pred_dir == true_dir).float().mean()
magnitude_acc = (pred_mag == true_mag).float().mean()
print(f"Dir Acc: {direction_acc:.2%}, Mag Acc: {magnitude_acc:.2%}")
```

### 6.4 调整建议

**如果辅助损失仍然过小**:
```python
# 在 TimeLLM_Enhanced.py 中进一步增大:
self.lambda_trend = 0.8   # 从0.5增大
self.lambda_retrieval = 0.5  # 从0.3增大
```

**如果过拟合加剧**:
```bash
# 降低学习率
LEARNING_RATE=0.0002

# 增加dropout
DROPOUT=0.2

# 增加warmup步数
# 在代码中修改 warmup_steps=1000
```

**如果检索模块不起作用**:
```bash
# 增加检索数量
TOP_K=10

# 增加表示维度
D_REPR=256

# 确保记忆库已构建
BUILD_MEMORY=1
```

---

## 7. Checkpoint 与推理

### 7.1 Checkpoint 结构

训练过程中会保存两种类型的 checkpoint:

```
checkpoints/
├── long_term_forecast_ETTh1_512_96_...-GPT2_Enhanced/
│   ├── checkpoint                # EarlyStopping最佳模型 (推理用)
│   ├── checkpoint_step_1757/     # Step checkpoint
│   │   └── checkpoint.pt
│   ├── checkpoint_step_3514/
│   │   └── checkpoint.pt
│   └── ...
```

### 7.2 Checkpoint 内容

```python
ckpt_payload = {
    'model': model.state_dict(),          # 模型权重
    'optimizer': optimizer.state_dict(),   # 优化器状态
    'scheduler': scheduler.state_dict(),   # 学习率调度器
    'epoch': epoch,                         # 当前epoch
    'global_step': global_step,             # 全局步数
    'best_score': -vali_loss,               # 最佳验证分数
    'val_loss_min': vali_loss,              # 最小验证损失
    'counter': early_stopping.counter,      # EarlyStopping计数
    'rng_state': {...},                     # 随机状态
    'sampler_state': {...},                 # 采样器状态
    'scaler': scaler.state_dict(),          # AMP状态 (可选)
}
```

### 7.3 推理加载

```python
import torch
from models import TimeLLM_Enhanced as TimeLLM

# 1. 准备配置 (需与训练时一致)
class Args:
    task_name = 'long_term_forecast'
    seq_len = 512
    pred_len = 96
    # ... 其他必要参数
    use_tapr = True
    use_gram = True
    n_scales = 3
    top_k = 5
    d_repr = 128
    # ...

args = Args()

# 2. 创建模型
model = TimeLLM.Model(args).float()

# 3. 加载 checkpoint (使用 EarlyStopping 保存的最佳模型)
ckpt_path = 'checkpoints/long_term_forecast_ETTh1_512_96_.../checkpoint'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# 4. 加载权重
if isinstance(ckpt, dict) and 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
else:
    model.load_state_dict(ckpt)

# 5. 设置评估模式
model.eval()
model.cuda()  # 如需GPU

# 6. 推理
with torch.no_grad():
    # x_enc: [B, seq_len, n_vars]
    # x_mark_enc, x_dec, x_mark_dec: 时间特征
    outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    # outputs: [B, pred_len, n_vars]
```

### 7.4 断点续训

```bash
# 从 step checkpoint 恢复
RESUME_FROM="${CHECKPOINTS}/.../checkpoint_step_1757/checkpoint.pt"

# 从 EarlyStopping checkpoint 恢复
RESUME_FROM="${CHECKPOINTS}/.../checkpoint"

# 可选: 重置 EarlyStopping 计数
RESUME_COUNTER=0

# 断点续训时注意:
BUILD_MEMORY=0  # 避免重建记忆库
```

---

## 8. 故障排除

### 8.1 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `mat1 and mat2 shapes cannot be multiplied (Nx1 and 64x64)` | memory_values 维度不匹配 | 已修复 `_resize_last_dim()` |
| `Expected all tensors to be on the same device` | device 不一致 | 已修复 `device=y_true.device` |
| `size mismatch for direction_logits` | y_true 维度不匹配 | 已修复 permute+reshape |
| OOM (CUDA out of memory) | 显存不足 | 见下方显存优化 |
| 验证损失一直上升 | 过拟合 | 降低学习率、增加dropout |

### 8.2 显存优化 (优先级从高到低)

```bash
1) 降 BATCH (32→16→8→4)
2) 降 SEQ_LEN (512→384→256→128)
3) 降 LLM_LAYERS (12→6→4→2)
4) 降 D_FF (128→64→32)
5) 降 D_MODEL (64→32→16)
6) 降 N_SCALES (3→2)
7) 降 TOP_K (5→3)
8) 禁用 GRA-M (USE_GRAM=0)
9) 启用 --load_in_4bit (如用Qwen)

# 6GB 显存推荐:
BATCH=8, SEQ_LEN=256, LLM_LAYERS=4, D_MODEL=32, D_FF=64

# 15GB 显存推荐:
BATCH=32, SEQ_LEN=512, LLM_LAYERS=12, D_MODEL=64, D_FF=128
```

### 8.3 调试建议

```python
# 在模型forward中添加形状检查:
print(f"x_enc: {x_enc.shape}")
print(f"enc_out after patch: {enc_out.shape}")
print(f"enc_out after TAPR: {enc_out.shape}")
print(f"trend_info keys: {trend_info.keys() if trend_info else None}")
print(f"magnitude_logits: {trend_info.get('magnitude_logits', 'N/A').shape}")
```

---

## 附录: 文件索引

| 文件 | 说明 |
|------|------|
| `models/TimeLLM_Enhanced.py` | 增强版主模型 |
| `layers/TAPR.py` | TAPR模块 (DM + C2F) |
| `layers/GRAM.py` | GRA-M模块 (PVDR + AKG) |
| `run_main_enhanced.py` | 训练入口 |
| `scripts/TimeLLM_ETTh1_enhanced.sh` | 训练脚本 |
| `claude3.md` | 本文档 |

---

**最后更新**: 2026-02-08
**状态**: 所有已知问题已修复，待消融实验验证
 我已经生成了完整的 claude3.md 文档。文档包含以下内容：

  文档概要

  1. 架构总览

  - 完整的数据流图 (ASCII 图示)
  - 模块层级结构图，明确标注了 TAPR-DM、TAPR-C2F、GRA-M-PVDR、GRA-M-AKG 各子模块所属的大模块

  2. 模块详解

  - TAPR-DM (MultiScaleDecomposition): 多尺度分解，包含代码位置 layers/TAPR.py:32-184
  - TAPR-C2F (CoarseToFineHead): 先粗后细预测头，包含代码位置 layers/TAPR.py:231-402
  - GRA-M-PVDR (PatternValueDualRetriever): 模式-数值双重检索器
  - GRA-M-AKG (AdaptiveKnowledgeGating): 自适应知识门控
  - C2F趋势融合: _apply_trend_fusion() 方法
  - _resize_last_dim: 之前修复的维度对齐工具函数

  3. 数据流详解

  - 完整的形状变化表 (ETTh1 示例)
  - 辅助损失计算流程和 warmup 机制说明

  4. 修改记录

  - 7个问题修复清单 (含文件:行号)
  - 超参数调整表 (原值→新值→原因)

  5. 命令参数说明

  重点参数作用说明:
  | 参数           | 是否发挥作用    | 消融建议             |
  |--------------|-----------|------------------|
  | N_SCALES     | ⚠️ 需验证    | 测试 2, 3, 4       |
  | LAMBDA_TREND | ✅ 已增大到0.5 | 测试 0.3, 0.5, 0.7 |
  | TOP_K        | ⚠️ 需验证    | 测试 3, 5, 10      |
  | BUILD_MEMORY | ✅ 首次必须启用  | 断点续训设为0          |
  | D_REPR       | ⚠️ 需验证    | 与d_ff保持一致        |

  6. 消融实验指南

  - 4种实验配置 (仅TAPR / 仅GRA-M / 完整模型 / 基线)
  - 参数敏感性实验设置
  - 判断参数是否发挥作用的3种方法
  - 调整建议

  7. Checkpoint 与推理

  - Checkpoint 结构和内容说明
  - 推理加载代码示例
  - 断点续训注意事项

  8. 故障排除

  - 常见错误及解决方案
  - 显存优化优先级
  - 调试建议