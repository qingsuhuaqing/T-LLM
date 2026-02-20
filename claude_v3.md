# claude4.md - Time-LLM Enhanced v3 技术文档

> **版本**: 2026-02-20
> **状态**: 已完成全部模块开发，待消融实验验证
> **目的**: 记录增强版 v3 的完整架构、四类可训练矩阵、数据流、修改记录和消融实验指南
> **核心改变**: v2 "观察者模式" → v3 "主动多尺度预测 + 检索增强融合"

---

## 目录

1. [架构总览](#1-架构总览)
2. [四类可训练矩阵定义](#2-四类可训练矩阵定义)
3. [模块详解](#3-模块详解)
4. [数据流详解](#4-数据流详解)
5. [版本演进对比](#5-版本演进对比)
6. [命令参数说明](#6-命令参数说明)
7. [消融实验指南](#7-消融实验指南)
8. [Checkpoint 与推理](#8-checkpoint-与推理)
9. [故障排除](#9-故障排除)

---

## 1. 架构总览

### 1.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Time-LLM Enhanced v3 架构                                │
│                  "主动多尺度预测 + 检索增强融合"                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: [B, seq_len, n_vars]                                               │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                       │
│  │ Instance Norm   │  x_normed: [B, T, N]                                  │
│  └────────┬────────┘                                                       │
│           │                                                                 │
│           ├──────────────────────────────┐                                  │
│           │                              │                                  │
│           ▼                              ▼                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐              │
│  │ ① Baseline Time-LLM │    │ ② DM (Decomposable Multi-    │              │
│  │   (完全不修改)        │    │    Scale) 多尺度独立预测     │              │
│  │                      │    │                              │              │
│  │ Patch → Reprogram    │    │  S2: ds=2,  diff变换         │              │
│  │ → Prompt → LLM       │    │  S3: ds=4,  identity         │              │
│  │ → FlattenHead        │    │  S4: ds=8,  smooth变换       │              │
│  │                      │    │  S5: ds=16, trend变换        │              │
│  │ "参数矩阵" Θ_1       │    │  "参数矩阵" Θ_2..Θ_5        │              │
│  └────────┬─────────────┘    └────────┬─────────────────────┘              │
│           │                           │                                     │
│     pred_s1 [B,P,N]         pred_s2..s5 [B,P,N] × 4                       │
│           │                           │                                     │
│           └───────────┬───────────────┘                                     │
│                       │                                                     │
│                       ▼                                                     │
│           ┌──────────────────────────────┐                                  │
│           │ ③ C2F (Coarse-to-Fine        │                                  │
│           │    Fusion) 跨尺度融合         │                                  │
│           │                              │                                  │
│           │ "权重矩阵" W [5, pred_len]   │                                  │
│           │ softmax → 加权求和            │                                  │
│           │ 训练: KL一致性约束            │                                  │
│           │ 推理: 硬一致性降权            │                                  │
│           └────────┬─────────────────────┘                                  │
│                    │                                                        │
│            weighted_pred [B,P,N]                                            │
│                    │                                                        │
│           ┌────────┴──────────┐                                             │
│           │                   │                                             │
│           ▼                   ▼                                             │
│  ┌──────────────────┐  ┌──────────────────────────────┐                    │
│  │ scale_preds      │  │ ④ PVDR (Pattern-Value Dual   │                    │
│  │ (5个尺度预测)     │  │    Retriever) 多尺度检索     │                    │
│  │                  │  │                              │                    │
│  │                  │  │ 每个尺度独立记忆库            │                    │
│  │                  │  │ 极端数据检测 (3σ)            │                    │
│  │                  │  │ 可学习阈值 per scale          │                    │
│  │                  │  │ Top-K=5 加权检索              │                    │
│  └────────┬─────────┘  └────────┬─────────────────────┘                    │
│           │                     │                                           │
│           │               hist_refs [B,P,N] × 5                            │
│           │                     │                                           │
│           └─────────┬───────────┘                                           │
│                     │                                                       │
│                     ▼                                                       │
│           ┌──────────────────────────────┐                                  │
│           │ ⑤ AKG (Adaptive Knowledge   │                                  │
│           │    Gating) 自适应知识门控    │                                  │
│           │                              │                                  │
│           │ "联系矩阵" C [5, pred_len]   │                                  │
│           │ sigmoid → 历史vs预测门控      │                                  │
│           │                              │                                  │
│           │ "融合矩阵" F [10, pred_len]  │                                  │
│           │ softmax → 最终融合            │                                  │
│           └────────┬─────────────────────┘                                  │
│                    │                                                        │
│            final_pred [B,P,N]                                               │
│                    │                                                        │
│                    ▼                                                        │
│           ┌─────────────────┐                                               │
│           │ Instance Denorm │  反归一化                                      │
│           └────────┬────────┘                                               │
│                    │                                                        │
│                    ▼                                                        │
│           Output: [B, pred_len, n_vars]                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 模块层级结构

```
TimeLLM_Enhanced_v3 (models/TimeLLM_Enhanced_v3.py)
│
├── 原有模块 (Time-LLM Baseline, 完全不修改)
│   ├── normalize_layers (Instance Normalization)
│   ├── patch_embedding (PatchEmbedding: Conv1d + Positional)
│   ├── mapping_layer (词表映射: Linear(vocab, 1000))
│   ├── reprogramming_layer (跨模态注意力: Q=patch, KV=mapped_words)
│   ├── output_projection (FlattenHead: Flatten + Linear)
│   └── llm_model (Qwen 2.5 3B / GPT-2 / LLAMA, 冻结)
│
├── ★ 创新模块一: TAPR_v3 (layers/TAPR_v3.py)
│   │
│   ├── DM: DecomposableMultiScale (可分解多尺度独立预测)
│   │   ├── SignalTransform (信号变换: diff/smooth/trend)
│   │   ├── _downsample (平均池化下采样)
│   │   ├── _create_patches (分块)
│   │   ├── patch_embeddings: ModuleList (4个Conv1d嵌入)
│   │   └── scale_encoders: ModuleList (4个ScaleEncoder)
│   │       ├── conv1, conv2 (Conv1d 特征提取)
│   │       └── linear (FlattenHead → pred_len)
│   │
│   └── C2F: CoarseToFineFusion (粗到细融合)
│       ├── weight_matrix: Parameter [n_scales, pred_len]
│       ├── consistency_loss (对称KL散度)
│       └── _apply_hard_consistency (推理时降权)
│
├── ★ 创新模块二: GRAM_v3 (layers/GRAM_v3.py)
│   │
│   ├── PVDR: PatternValueDualRetriever (模式-数值双重检索器)
│   │   ├── LightweightPatternEncoder (5统计特征 → d_repr)
│   │   ├── ExtremeDetector (z-score 3σ检测)
│   │   └── MultiScaleMemoryBank (per-scale key/value存储)
│   │       ├── keys_{0..4}: Buffer [M, d_repr]
│   │       ├── values_{0..4}: Buffer [M, pred_len]
│   │       └── thresholds: Parameter [n_scales]
│   │
│   └── AKG: AdaptiveKnowledgeGating (自适应知识门控)
│       ├── connection_matrix: Parameter [n_scales, pred_len]
│       ├── fusion_matrix: Parameter [2*n_scales, pred_len]
│       └── gate_loss (门控决断性正则化)
│
└── _aux_cache: dict (辅助损失缓存)
```

### 1.3 关键设计思想

| 设计原则 | 说明 |
|----------|------|
| **Baseline不修改** | 原始Time-LLM的forward完全保持不变，v3模块只在baseline输出后处理 |
| **独立预测+融合** | 每个尺度产生独立的 [B, pred_len, N] 预测，通过矩阵加权融合 |
| **per-timestep粒度** | 权重/联系/融合矩阵均为 [scales, pred_len]，允许不同时间步不同权重 |
| **模块可开关** | `--use_tapr` 和 `--use_gram` 独立控制，支持消融实验 |

---

## 2. 四类可训练矩阵定义

v3 的核心创新是引入四类可训练矩阵，每类矩阵负责不同层面的信息处理：

### 2.1 参数矩阵 Θ_s（尺度内部）

**位置**: `layers/TAPR_v3.py` — `ScaleEncoder` 类内部

**含义**: 每个辅助尺度 (S2-S5) 拥有独立的编码器，其 Conv1d + Linear 层的参数集合构成该尺度的"参数矩阵"。

**结构**:
```python
# 每个 ScaleEncoder 的参数:
self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)  # d_model² × 3 + d_model
self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)  # d_model² × 3 + d_model
self.linear = nn.Linear(head_nf, pred_len)                          # head_nf × pred_len + pred_len
```

**参数量** (d_model=32, head_nf≈32×33=1056, pred_len=96):
- conv1: 32×32×3 + 32 = 3,104
- conv2: 32×32×3 + 32 = 3,104
- linear: 1,056×96 + 96 = 101,472
- patch_embed (Conv1d): 16×32×3 = 1,536
- **每个尺度约 ~25K，4个尺度共 ~100K**

**学习方式**: 标准反向传播，通过主MSE损失优化

### 2.2 权重矩阵 W（C2F跨尺度融合）

**位置**: `layers/TAPR_v3.py:349` — `CoarseToFineFusion.weight_matrix`

**形状**: `[n_scales, pred_len]` = `[5, 96]`

**含义**: 对每个预测时间步，为5个尺度的预测分配权重。经过 softmax(dim=0) 归一化后，各尺度权重之和为1。

```python
self.weight_matrix = nn.Parameter(torch.ones(n_scales, pred_len) / n_scales)
# 初始化: 均匀分配 → 1/5 = 0.2

# 前向传播:
weights = F.softmax(self.weight_matrix, dim=0)  # [5, 96]，每列和=1
w = weights.unsqueeze(1).unsqueeze(-1)           # [5, 1, 96, 1]
weighted_pred = (w * stacked).sum(dim=0)          # [B, 96, N]
```

**参数量**: 5 × 96 = 480

**设计依据**: TimeMixer论文指出不同尺度在不同预测步长上贡献不同（粗尺度在远期更重要，细尺度在近期更重要），因此用 per-timestep 粒度。

### 2.3 联系矩阵 C（AKG历史-预测关系）

**位置**: `layers/GRAM_v3.py:331` — `AdaptiveKnowledgeGating.connection_matrix`

**形状**: `[n_scales, pred_len]` = `[5, 96]`

**含义**: 每个尺度每个时间步的"预测vs历史"门控。sigmoid(C[s,t]) → 1 表示信任当前预测，→ 0 表示信任历史检索。

```python
self.connection_matrix = nn.Parameter(torch.zeros(n_scales, pred_len))
# 初始化: 0 → sigmoid(0) = 0.5（平衡态）

# 前向传播:
gates = torch.sigmoid(self.connection_matrix)  # [5, 96]
for s in range(n_scales):
    g = gates[s].unsqueeze(0).unsqueeze(-1)    # [1, 96, 1]
    connected_s = g * scale_preds[s] + (1 - g) * hist_refs[s]
```

**参数量**: 5 × 96 = 480

**正则化**: L_gate = -mean(|sigmoid(C) - 0.5|)，鼓励门控远离0.5做出明确决策

### 2.4 融合矩阵 F（AKG最终融合）

**位置**: `layers/GRAM_v3.py:336` — `AdaptiveKnowledgeGating.fusion_matrix`

**形状**: `[2*n_scales, pred_len]` = `[10, 96]`

**含义**: 将所有来源（5个原始尺度预测 + 5个经联系矩阵处理的结果）融合为最终预测。经 softmax(dim=0) 归一化。

```python
self.fusion_matrix = nn.Parameter(torch.zeros(2 * n_scales, pred_len))
# 初始化: 0 → softmax后均匀 = 1/10 = 0.1

# 前向传播:
all_sources = scale_preds + connected  # 10 个来源
fusion_weights = F.softmax(self.fusion_matrix, dim=0)  # [10, 96]
stacked = torch.stack(all_sources, dim=0)               # [10, B, 96, N]
w = fusion_weights.unsqueeze(1).unsqueeze(-1)            # [10, 1, 96, 1]
final_pred = (w * stacked).sum(dim=0)                    # [B, 96, N]
```

**参数量**: 10 × 96 = 960

### 2.5 矩阵总览

| 矩阵 | 符号 | 形状 | 参数量 | 位置 | 归一化 |
|------|------|------|--------|------|--------|
| 参数矩阵 | Θ_s | 每个ScaleEncoder | ~25K×4 | TAPR_v3.py | N/A |
| 权重矩阵 | W | [5, 96] | 480 | TAPR_v3.py:349 | softmax(dim=0) |
| 联系矩阵 | C | [5, 96] | 480 | GRAM_v3.py:331 | sigmoid |
| 融合矩阵 | F | [10, 96] | 960 | GRAM_v3.py:336 | softmax(dim=0) |
| **总新增** | | | **~103K** | | |

---

## 3. 模块详解

### 3.1 DM (Decomposable Multi-scale) — 可分解多尺度独立预测

**文件**: `layers/TAPR_v3.py:158-318`

**设计理念**: 时间序列包含多种频率成分。不同下采样率 + 不同信号变换可以捕获不同粒度的模式。每个尺度产生独立的预测，而非仅提取特征。

#### 3.1.1 信号变换

**文件**: `layers/TAPR_v3.py:24-94`

| 变换 | 函数 | 公式 | 输出长度 | 目的 |
|------|------|------|---------|------|
| identity | `SignalTransform.identity` | y = x | T | 保留原始 |
| diff | `SignalTransform.diff` | y[t] = x[t+1] - x[t] | T-1 | 变化率 |
| smooth | `SignalTransform.smooth` | y = MA(x, kernel=25) | T | 去噪 |
| trend | `SignalTransform.trend` | y = MA(x, kernel=T//4) | T | 宏观趋势 |

```python
# TRANSFORM_REGISTRY: (forward_fn, inverse_fn)
'identity': (SignalTransform.identity, SignalTransform.identity_inv),
'diff':     (SignalTransform.diff,     SignalTransform.diff_inv),
'smooth':   (SignalTransform.smooth,   SignalTransform.smooth_inv),
'trend':    (SignalTransform.trend,    SignalTransform.trend_inv),
```

#### 3.1.2 尺度配置

| 尺度 | 下采样率 | 信号变换 | 有效长度 (T=512) | num_patches |
|------|---------|---------|------------------|-------------|
| S1 | 1 | 无 (baseline) | 512 | 63 |
| S2 | 2 | diff | 256→255 | 32 |
| S3 | 4 | identity | 128 | 16 |
| S4 | 8 | smooth | 64 | 8 |
| S5 | 16 | trend | 32 | 4 |

#### 3.1.3 ScaleEncoder 结构

```python
class ScaleEncoder(nn.Module):
    """单尺度轻量编码器 — "参数矩阵" Θ_s 的载体"""

    def __init__(self, d_model, patch_len, stride, head_nf, pred_len, dropout=0.1):
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_nf, pred_len)

    def forward(self, x):
        # x: [B*N, num_patches, d_model]
        h = x.permute(0, 2, 1)          # [B*N, d_model, num_patches]
        h = self.dropout(self.act(self.conv1(h)))
        h = self.dropout(self.act(self.conv2(h)))
        h = self.flatten(h)              # [B*N, d_model * num_patches]
        return self.linear(h)            # [B*N, pred_len]
```

#### 3.1.4 DM forward 完整流程

```python
def forward(self, x_normed, pred_s1):
    # x_normed: [B, T, N]
    # pred_s1:  [B, pred_len, N] — baseline已完成

    scale_preds = [pred_s1]  # S1 = baseline

    for idx, (ds_rate, transform_name) in enumerate(self.scale_cfgs):
        # 1. 下采样: [B, T, N] → [B, T//ds, N]
        x_ds = self._downsample(x_normed, ds_rate)

        # 2. 信号变换: [B, T_ds, N] → [B, T_trans, N]
        fwd_fn, inv_fn = TRANSFORM_REGISTRY[transform_name]
        x_transformed, aux = fwd_fn(x_ds)

        # 3. Channel-independent: [B, T_trans, N] → [B*N, T_trans, 1]
        x_ci = x_transformed.permute(0,2,1).reshape(B*N, T_trans, 1)

        # 4. 创建patches: [B*N, num_patches, patch_len]
        patches, num_patches = _create_patches(x_ci, patch_len, stride)

        # 5. Patch嵌入: Conv1d(patch_len → d_model)
        embedded = patch_embed(patches.permute(0,2,1)).permute(0,2,1)
        # embedded: [B*N, num_patches, d_model]

        # 6. 必要时插值对齐patch数量
        if cur_patches != expected_patches:
            embedded = F.interpolate(...)

        # 7. ScaleEncoder → pred: [B*N, pred_len]
        pred_scale = encoder(embedded)

        # 8. Reshape: [B*N, pred_len] → [B, pred_len, N]
        pred_scale = pred_scale.reshape(B, N, pred_len).permute(0,2,1)

        scale_preds.append(pred_scale)

    return scale_preds  # list of 5 × [B, pred_len, N]
```

### 3.2 C2F (Coarse-to-Fine Fusion) — 粗到细融合

**文件**: `layers/TAPR_v3.py:325-448`

#### 3.2.1 前向传播（权重矩阵 W）

```python
class CoarseToFineFusion(nn.Module):
    def __init__(self, n_scales, pred_len, decay_factor=0.5, consistency_mode='hybrid'):
        self.weight_matrix = nn.Parameter(torch.ones(n_scales, pred_len) / n_scales)

    def forward(self, scale_preds, training=True):
        stacked = torch.stack(scale_preds, dim=0)  # [5, B, pred_len, N]
        weights = F.softmax(self.weight_matrix, dim=0)  # [5, pred_len]

        if not training and self.consistency_mode in ('hard', 'hybrid'):
            weights = self._apply_hard_consistency(scale_preds, weights)

        w = weights.unsqueeze(1).unsqueeze(-1)  # [5, 1, pred_len, 1]
        weighted_pred = (w * stacked).sum(dim=0)  # [B, pred_len, N]
        return weighted_pred
```

#### 3.2.2 一致性机制

**训练时 — 软约束 (KL散度)**:
```python
def consistency_loss(self, scale_preds):
    """鼓励各尺度在趋势方向上达成一致"""
    for i in range(n):
        for j in range(i + 1, n):
            trend_i = self._trend_distribution(scale_preds[i])  # [B, N, 3]
            trend_j = self._trend_distribution(scale_preds[j])  # [B, N, 3]
            # 对称KL散度
            loss += (KL(trend_i || trend_j) + KL(trend_j || trend_i)) / 2
    return loss / n_pairs
```

趋势方向分布: 将预测转为 (下降, 平稳, 上涨) 的软概率分布
```python
def _trend_distribution(self, pred):
    diff = pred[:, -1, :] - pred[:, 0, :]       # 整体趋势
    normalized_diff = diff / pred.std(dim=1)     # 归一化
    down_prob = sigmoid(-normalized_diff - 0.5)
    up_prob = sigmoid(normalized_diff - 0.5)
    flat_prob = 1 - down_prob - up_prob
    return stack([down, flat, up])  # [B, N, 3]
```

**推理时 — 硬约束 (降权)**:
```python
def _apply_hard_consistency(self, scale_preds, weights):
    # 以最粗尺度 (S5) 为基准
    coarse_sign = (scale_preds[-1][:,-1,:] - scale_preds[-1][:,0,:]) > 0

    for s in range(n_scales - 1):
        fine_sign = ...
        if disagree_fraction > 0.5:
            weights[s] *= 0.5  # decay_factor
    # 重新归一化
    return weights / weights.sum(dim=0)
```

### 3.3 PVDR (Pattern-Value Dual Retriever) — 模式-数值双重检索器

**文件**: `layers/GRAM_v3.py:131-307`

#### 3.3.1 LightweightPatternEncoder

```python
class LightweightPatternEncoder(nn.Module):
    """5个统计特征 → d_repr 维表示"""
    # 特征: mean, std, max, min, trend (首尾差)
    self.proj = nn.Sequential(
        nn.Linear(5, d_repr),
        nn.LayerNorm(d_repr),
    )
```

#### 3.3.2 ExtremeDetector

```python
class ExtremeDetector(nn.Module):
    """z-score 3σ 规则检测极端数据"""
    def forward(self, x):  # [B, T, N]
        z_scores = (x - mean) / std
        has_extreme = (z_scores.abs() > 3.0).any(dim=1).any(dim=1)  # [B]
        return has_extreme, z_scores
```

#### 3.3.3 MultiScaleMemoryBank

```python
class MultiScaleMemoryBank(nn.Module):
    """每个尺度独立的 (key, value) 存储"""
    # keys_{s}: Buffer [M, d_repr]   — 模式表示
    # values_{s}: Buffer [M, pred_len] — 对应未来值
    # thresholds: Parameter [n_scales] — 可学习相似度阈值

    # 初始化:
    self.thresholds = nn.Parameter(torch.full((n_scales,), 0.8))
    # sigmoid(0.8) ≈ 0.69 → 基础阈值
```

#### 3.3.4 检索流程

```python
def retrieve(self, x_normed, scale_idx, has_extreme=None):
    # 1. 编码查询: [B, T_ds, N] → mean → [B, T_ds] → encoder → [B, d_repr]
    query_key = F.normalize(self.encoder(x_normed.mean(dim=-1)), dim=-1)

    # 2. 余弦相似度: [B, d_repr] × [M, d_repr]^T → [B, M]
    similarity = torch.mm(query_key, mem_keys.t())

    # 3. 阈值判断 (极端数据时降低阈值)
    threshold = sigmoid(self.memory_bank.thresholds[scale_idx])
    if has_extreme:
        threshold -= 0.2  # extreme_threshold_reduction

    # 4. Top-K 检索
    top_sim, top_idx = similarity.topk(k=5, dim=-1)
    retrieved = mem_values[top_idx]  # [B, K, pred_len]

    # 5. 加权平均
    sim_weights = softmax(top_sim, dim=-1)
    hist_ref = (retrieved * sim_weights.unsqueeze(-1)).sum(dim=1)  # [B, pred_len]
    return hist_ref, valid_mask
```

#### 3.3.5 记忆库构建

```python
def build_memory(self, train_loader, device, downsample_rates, max_samples=5000):
    for each (batch_x, batch_y) in train_loader:
        all_inputs.append(batch_x)                        # [B, T, N]
        all_targets.append(batch_y[:, -pred_len:, :])     # [B, pred_len, N]

    y_mean = all_y.mean(dim=-1)  # [M, pred_len] — 跨变量压缩

    for s, ds_rate in enumerate(downsample_rates):
        x_ds = downsample(all_x, ds_rate)
        x_flat = x_ds.mean(dim=-1)           # [M, T_ds]
        keys = F.normalize(encoder(x_flat))  # [M, d_repr]
        memory_bank.set_bank(s, keys, y_mean)
```

### 3.4 AKG (Adaptive Knowledge Gating) — 自适应知识门控

**文件**: `layers/GRAM_v3.py:314-377`

#### 3.4.1 联系矩阵 C — 历史vs预测门控

```python
# 初始化: sigmoid(0) = 0.5 → 平衡态
self.connection_matrix = nn.Parameter(torch.zeros(n_scales, pred_len))

gates = torch.sigmoid(self.connection_matrix)  # [5, 96]
for s in range(n_scales):
    g = gates[s].unsqueeze(0).unsqueeze(-1)  # [1, 96, 1]
    # gate → 1: 信任当前预测
    # gate → 0: 信任历史检索
    connected_s = g * scale_preds[s] + (1 - g) * hist_refs[s]
```

#### 3.4.2 融合矩阵 F — 最终融合

```python
# 10个来源: 5个原始预测 + 5个门控后结果
all_sources = scale_preds + connected  # list of 10

self.fusion_matrix = nn.Parameter(torch.zeros(2 * n_scales, pred_len))
fusion_weights = F.softmax(self.fusion_matrix, dim=0)  # [10, 96]

stacked = torch.stack(all_sources, dim=0)      # [10, B, 96, N]
w = fusion_weights.unsqueeze(1).unsqueeze(-1)  # [10, 1, 96, 1]
final_pred = (w * stacked).sum(dim=0)          # [B, 96, N]
```

#### 3.4.3 门控正则化损失

```python
def gate_loss(self):
    """鼓励门控做出明确决策 (偏向0或1)"""
    gates = torch.sigmoid(self.connection_matrix)
    return -(gates - 0.5).abs().mean()
    # 负号: 最大化距离0.5的偏差
```

---

## 4. 数据流详解

### 4.1 形状变化流程 (ETTh1 示例: B=4, T=512, N=7)

| 阶段 | 形状 | 说明 | 文件:行号 |
|------|------|------|----------|
| **输入** | `[4, 512, 7]` | B=4, seq_len=512, n_vars=7 | |
| **Instance Norm** | `[4, 512, 7]` | 消除尺度差异 | TimeLLM_Enhanced_v3.py:356 |
| **Channel Indep.** | `[28, 512, 1]` | B*N=28 | TimeLLM_Enhanced_v3.py:359 |
| **Patch Embedding** | `[28, 63, 32]` | P=63, d_model=32 | TimeLLM_Enhanced_v3.py:396 |
| **Reprogramming** | `[28, 63, 32]` | 映射到d_ff维度 | TimeLLM_Enhanced_v3.py:398 |
| **Prompt + LLM** | `[28, ?, 2048]` | prompt + patches输入 | TimeLLM_Enhanced_v3.py:400-401 |
| **FlattenHead** | `[4, 96, 7]` | pred_s1 (normalized) | TimeLLM_Enhanced_v3.py:408-409 |
| **DM S2 (ds=2)** | `[4, 96, 7]` | diff变换, pred_s2 | TAPR_v3.py:265-316 |
| **DM S3 (ds=4)** | `[4, 96, 7]` | identity, pred_s3 | |
| **DM S4 (ds=8)** | `[4, 96, 7]` | smooth, pred_s4 | |
| **DM S5 (ds=16)** | `[4, 96, 7]` | trend, pred_s5 | |
| **C2F融合** | `[4, 96, 7]` | weighted_pred | TAPR_v3.py:353-376 |
| **PVDR检索 ×5** | 5×`[4, 96, 7]` | hist_refs | GRAM_v3.py:272-307 |
| **AKG门控** | `[4, 96, 7]` | final_pred (normalized) | GRAM_v3.py:338-367 |
| **Denorm** | `[4, 96, 7]` | 反归一化 | TimeLLM_Enhanced_v3.py:485 |

### 4.2 DM内部详细形状变化 (以S2为例)

| 步骤 | 操作 | 形状 |
|------|------|------|
| 输入 | x_normed | [4, 512, 7] |
| 下采样 (ds=2) | _downsample | [4, 256, 7] |
| diff变换 | SignalTransform.diff | [4, 255, 7] |
| Channel Indep. | permute+reshape | [28, 255, 1] |
| 创建patches | _create_patches | [28, 31, 16] |
| Patch嵌入 | Conv1d(16→32) | [28, 31, 32] |
| 插值对齐 | F.interpolate | [28, 33, 32] |
| ScaleEncoder | Conv×2 + Linear | [28, 96] |
| Reshape | → [B, pred_len, N] | [4, 96, 7] |

### 4.3 辅助损失计算流程

```
训练时:
├── 主损失: MSE(final_pred, batch_y_target)
│
├── TAPR辅助损失 (lambda_consist=0.1):
│   └── L_consistency: Σ symmetric_KL(trend_i, trend_j) / n_pairs
│       ├── 10对尺度组合: C(5,2) = 10
│       ├── 每对: 计算趋势方向分布 [B, N, 3] (下/平/上)
│       └── 对称KL: (KL(i||j) + KL(j||i)) / 2
│
└── GRAM辅助损失 (lambda_gate=0.01):
    └── L_gate: -mean(|sigmoid(connection_matrix) - 0.5|)
        └── 鼓励门控做出明确决策

总损失 = MSE + warmup_factor × (0.1 × L_consistency + 0.01 × L_gate)

Warmup机制:
- warmup_steps = 500
- warmup_factor = min(1.0, current_step / 500)
- 训练初期降低辅助损失权重，先学主任务
```

### 4.4 梯度流向

确保四类矩阵都能收到梯度的关键设计:

```
final_pred
    │
    ├── AKG.fusion_matrix ← softmax → 加权求和 → ✅ 梯度OK
    │
    ├── AKG.connection_matrix ← sigmoid → 门控 → ✅ 梯度OK
    │
    ├── C2F.weight_matrix ← softmax → 加权求和
    │   └── 关键: akg_preds[0] = weighted_pred
    │       (将C2F输出注入AKG输入, 保证梯度传播)
    │       → ✅ 梯度OK
    │
    └── ScaleEncoder参数 ← Conv + Linear → 预测
        └── 通过scale_preds传递 → ✅ 梯度OK
```

---

## 5. 版本演进对比

### 5.1 v1 → v2 → v3 对比

| 特性 | v1 (直接修改) | v2 (观察者模式) | v3 (主动预测融合) |
|------|-------------|----------------|-----------------|
| TAPR角色 | 修改enc_out | 仅观察，不修改 | 产生独立预测 |
| GRAM角色 | 强制检索融合 | 极高阈值(0.95) | 多尺度记忆+门控 |
| 辅助损失权重 | 0.5/0.3 | 0.01/0.001 | 0.1/0.01 |
| Baseline影响 | ❌ 破坏100% | ✅ 完全不影响 | ✅ 完全不影响 |
| 预期MSE改善 | 恶化 | ±0.5% | ↓10-20% |
| 检索触发 | 全量 | ~1-3% | 全量(有阈值) |
| 融合方式 | 无 | 冲突时微调 | 四类矩阵 |
| 新增参数 | ~2K | ~5K | ~103K |

### 5.2 关键设计差异

```
v2: enc_out ──────────> Reprogramming → LLM → FlattenHead → pred
        │
        └→ [观察] → trend_embed (额外prompt token)
                 → context_embed (极少触发)

v3: x_enc ─────────> [Baseline完整运行] → pred_s1
        │                                      │
        └→ [DM] → pred_s2..s5 ────────────────┤
                                               ▼
                                        [C2F: W矩阵融合]
                                               │
                  [PVDR: 多尺度检索] ──────────┤
                                               ▼
                                        [AKG: C矩阵+F矩阵]
                                               │
                                        final_pred
```

---

## 6. 命令参数说明

### 6.1 v3 TAPR 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_tapr` | flag | False | 启用DM+C2F模块 |
| `--n_scales` | int | 5 | 总尺度数(含baseline) |
| `--downsample_rates` | str | "1,2,4,8,16" | 各尺度下采样率 |
| `--lambda_consist` | float | 0.1 | 一致性损失权重 |
| `--consistency_mode` | str | "hybrid" | soft/hard/hybrid |
| `--decay_factor` | float | 0.5 | 推理时不一致尺度降权因子 |

**`n_scales` 详解**:
- 5 表示: S1(baseline) + S2(ds=2) + S3(ds=4) + S4(ds=8) + S5(ds=16)
- 减少到 3: 仅使用 S1 + S2 + S3，参数量和计算量减半
- 增加需修改 `DecomposableMultiScale.SCALE_CFG`

**`lambda_consist` 详解**:
- 控制KL一致性损失的权重
- 过大: 强制各尺度预测相同，失去多尺度意义
- 过小: 各尺度预测可能严重矛盾
- 建议范围: 0.05 ~ 0.3

### 6.2 v3 GRAM 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_gram` | flag | False | 启用PVDR+AKG模块 |
| `--lambda_gate` | float | 0.01 | 门控正则化权重 |
| `--similarity_threshold` | float | 0.8 | 基础检索阈值 (sigmoid前) |
| `--extreme_sigma` | float | 3.0 | 极端数据检测σ阈值 |
| `--extreme_threshold_reduction` | float | 0.2 | 极端时阈值降低量 |
| `--top_k` | int | 5 | Top-K检索数量 |
| `--d_repr` | int | 64 | 模式表示维度 |
| `--build_memory` | flag | False | 是否构建记忆库 |

**`build_memory` 重要说明**:
- 首次训练: **必须启用** `--build_memory`
- 断点续训: **必须关闭** (记忆库已在checkpoint中)
- 记忆库存储在 `MultiScaleMemoryBank` 的buffer中，会随checkpoint保存

**`similarity_threshold` 详解**:
- 存储为原始值 (0.8)，使用时经 sigmoid: sigmoid(0.8) ≈ 0.69
- 是 `nn.Parameter`，训练过程中会自动调整
- 极端数据时降低: 0.69 - 0.2 = 0.49 → 更容易匹配

### 6.3 训练参数

| 参数 | 推荐值 (6GB) | 推荐值 (16GB+) | 说明 |
|------|-------------|---------------|------|
| `--batch_size` | 4 | 16-32 | 批次大小 |
| `--llm_layers` | 6 | 12-24 | LLM使用层数 |
| `--d_model` | 32 | 64 | Patch嵌入维度 |
| `--d_ff` | 32 | 128 | FFN维度 |
| `--seq_len` | 512 | 512 | 输入序列长度 |
| `--pred_len` | 96 | 96 | 预测长度 |
| `--warmup_steps` | 500 | 500 | 辅助损失warmup |
| `--load_in_4bit` | ✅ | 可选 | 4-bit量化 |

---

## 7. 消融实验指南

### 7.1 四组主消融

```bash
# E1: Baseline only (原始Time-LLM)
USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh

# E2: +TAPR only (DM + C2F)
USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh

# E3: +GRAM only (PVDR + AKG)
USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh

# E4: Full (DM + C2F + PVDR + AKG)
USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh
```

### 7.2 参数敏感性实验

```bash
# n_scales 敏感性 (需修改代码中的 SCALE_CFG)
# 3尺度: S1 + S2 + S3
# 5尺度: S1 + S2 + S3 + S4 + S5 (默认)

# lambda_consist 敏感性
--lambda_consist 0.05   # 实验A
--lambda_consist 0.1    # 实验B (默认)
--lambda_consist 0.3    # 实验C

# top_k 敏感性
--top_k 3    # 实验A
--top_k 5    # 实验B (默认)
--top_k 10   # 实验C

# lambda_gate 敏感性
--lambda_gate 0.005   # 实验A
--lambda_gate 0.01    # 实验B (默认)
--lambda_gate 0.05    # 实验C
```

### 7.3 判断参数是否发挥作用

**方法1: 观察辅助损失**
```
正常情况 (辅助损失有效):
Epoch: 1 | Train: 0.XXXX | Aux: 0.01~0.10

异常情况 (辅助损失过小):
Epoch: 1 | Train: 0.XXXX | Aux: 0.0001
→ 尝试增大 lambda_consist 或 lambda_gate
```

**方法2: 检查权重矩阵分布**
```python
# 训练后检查C2F权重矩阵
model_unwrapped = accelerator.unwrap_model(model)
if model_unwrapped.c2f is not None:
    w = F.softmax(model_unwrapped.c2f.weight_matrix, dim=0)
    print(f"C2F weights per scale:\n{w.mean(dim=1)}")
    # 期望: 各尺度权重不完全相同，说明学到了有意义的差异
```

**方法3: 检查门控分布**
```python
if model_unwrapped.akg is not None:
    gates = torch.sigmoid(model_unwrapped.akg.connection_matrix)
    print(f"AKG gates: mean={gates.mean():.3f}, std={gates.std():.3f}")
    # 期望: std > 0.1，说明门控做出了分化决策
    # 若 std ≈ 0: 门控未学到有意义的差异
```

### 7.4 预期结果

| 实验 | 预期MSE变化 | 说明 |
|------|-------------|------|
| Baseline (E1) | 基准 | ~0.38 (ETTh1) |
| +TAPR (E2) | ↓5-10% | 多尺度独立预测提供互补信息 |
| +GRAM (E3) | ↓3-5% | 历史检索提供参考 |
| Full (E4) | ↓10-20% | 综合效果 |

---

## 8. Checkpoint 与推理

### 8.1 Checkpoint 结构

```
checkpoints/
├── long_term_forecast_ETTh1_512_96_..._v3_TAPR_GRAM/
│   ├── checkpoint                    # EarlyStopping最佳模型 (推理用)
│   ├── checkpoint_step_500/
│   │   └── checkpoint.pt             # Step checkpoint
│   ├── checkpoint_step_1000/
│   │   └── checkpoint.pt
│   └── ...
```

### 8.2 Checkpoint 内容

```python
ckpt_payload = {
    'model': model.state_dict(),          # 模型权重 (含memory_bank buffers)
    'optimizer': optimizer.state_dict(),   # 优化器状态
    'scheduler': scheduler.state_dict(),   # 学习率调度器
    'epoch': epoch,                         # 当前epoch
    'global_step': global_step,             # 全局步数
    'best_score': -vali_loss,               # 最佳验证分数
    'val_loss_min': vali_loss,              # 最小验证损失
    'counter': early_stopping.counter,      # EarlyStopping计数
    'rng_state': {...},                     # 随机状态
    'scaler': scaler.state_dict(),          # AMP状态 (可选)
}
```

**注意**: `model.state_dict()` 包含 `pvdr.memory_bank.keys_0`, `pvdr.memory_bank.values_0` 等 buffer，断点续训时会自动恢复记忆库。

### 8.3 推理加载

```python
import torch
from models import TimeLLM_Enhanced_v3 as TimeLLM

# 1. 准备配置 (需与训练时一致)
class Args:
    task_name = 'long_term_forecast'
    seq_len = 512
    pred_len = 96
    enc_in = 7
    d_model = 32
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 32
    dropout = 0.1
    patch_len = 16
    stride = 8
    llm_model = 'QWEN'
    llm_dim = 2048
    llm_layers = 6
    llm_model_path = './base_models/Qwen2.5-3B'
    load_in_4bit = True
    prompt_domain = 1
    content = "..."  # 需调用 load_content(args)
    # v3 参数
    use_tapr = True
    use_gram = True
    n_scales = 5
    downsample_rates = '1,2,4,8,16'
    d_repr = 64
    top_k = 5
    similarity_threshold = 0.8
    extreme_sigma = 3.0
    extreme_threshold_reduction = 0.2
    lambda_consist = 0.1
    lambda_gate = 0.01
    decay_factor = 0.5
    consistency_mode = 'hybrid'

args = Args()

# 2. 创建模型
model = TimeLLM.Model(args).float()

# 3. 加载 checkpoint
ckpt_path = 'checkpoints/long_term_forecast_ETTh1_512_96_.../checkpoint'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# 4. 加载权重 (处理memory_bank buffers)
if isinstance(ckpt, dict) and 'model' in ckpt:
    state_dict = ckpt['model']
    # 分离memory_bank buffers
    memory_buffers = {}
    keys_to_remove = []
    for key in list(state_dict.keys()):
        if 'memory_bank' in key and ('keys_' in key or 'values_' in key):
            memory_buffers[key] = state_dict[key]
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)

    # 恢复memory_bank buffers
    for key, value in memory_buffers.items():
        parts = key.split('.')
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
else:
    model.load_state_dict(ckpt)

# 5. 推理
model.eval()
model.cuda()
with torch.no_grad():
    outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    # outputs: [B, pred_len, n_vars]
```

### 8.4 断点续训

```bash
# 从 step checkpoint 恢复
python run_main_enhanced_v3.py \
  ... \
  --resume_from_checkpoint checkpoints/.../checkpoint_step_500/checkpoint.pt \
  --build_memory 0  # 重要: 不重建记忆库

# 从 EarlyStopping checkpoint 恢复
python run_main_enhanced_v3.py \
  ... \
  --resume_from_checkpoint checkpoints/.../checkpoint \
  --build_memory 0
```

### 8.5 参数分类 (断点续训)

**✅ 自动恢复（无需修改）**

| 状态 | checkpoint key |
|------|----------------|
| 模型权重 + memory buffers | `model` |
| 优化器状态 | `optimizer` |
| 调度器状态 | `scheduler` |
| epoch / global_step | `epoch`, `global_step` |
| EarlyStopping | `best_score`, `val_loss_min`, `counter` |
| 随机状态 | `rng_state` |

**⚠️ 可安全修改**

| 参数 | 说明 |
|------|------|
| `--train_epochs` | 可增加总训练轮数 |
| `--patience` | 可增加早停耐心值 |
| `--lambda_consist` | 辅助损失权重可微调 |
| `--lambda_gate` | 辅助损失权重可微调 |

**❌ 不可修改**

| 参数 | 原因 |
|------|------|
| `--n_scales` | 影响DM、C2F、AKG矩阵维度 |
| `--d_repr` | 影响PVDR编码器和记忆库维度 |
| `--top_k` | 不影响模型结构，但影响检索逻辑 |
| `--seq_len`, `--pred_len` | 影响所有模块维度 |
| `--d_model`, `--d_ff` | 影响Patch Embedding和ScaleEncoder |
| `--use_tapr`, `--use_gram` | 模块开关变化导致权重缺失 |

---

## 9. 故障排除

### 9.1 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Conv1d channel mismatch` | patches维度排列错误 | 已修复: 添加permute(0,2,1) |
| `C2F weight_matrix grad=None` | 梯度被AKG截断 | 已修复: akg_preds[0]=weighted_pred |
| `memory_bank buffer size mismatch` | 记忆库未构建 | 首次训练添加 `--build_memory` |
| OOM | 显存不足 | 见下方显存优化 |
| `NaN in consistency_loss` | 趋势分布为0 | 已修复: clamp(min=1e-6) |

### 9.2 显存优化 (优先级从高到低)

```bash
1) 降 BATCH (4 → 2)
2) 降 SEQ_LEN (512 → 256)
3) 降 LLM_LAYERS (6 → 4 → 2)
4) 降 D_MODEL (32 → 16)
5) 降 N_SCALES (5 → 3)
6) 禁用 GRAM (USE_GRAM=0)
7) 启用 --load_in_4bit

# 6GB 显存推荐:
batch_size=4, seq_len=512, llm_layers=6, d_model=32, load_in_4bit

# 16GB 显存推荐:
batch_size=16, seq_len=512, llm_layers=12, d_model=64

# 24GB+ 显存推荐:
batch_size=32, seq_len=512, llm_layers=24, d_model=64
```

### 9.3 显存估算

| 组件 | 显存 |
|------|------|
| Qwen 2.5 3B (4-bit, 6层) | ~1.5 GB |
| Time-LLM 原始可训练参数 | ~0.5 GB |
| DM: 4个ScaleEncoder (~25K × 4) | ~2 MB |
| C2F: 权重矩阵 [5, 96] | < 1 KB |
| PVDR: 5个记忆库 (5000 × 5 scales) | ~5 MB |
| AKG: 联系矩阵 + 融合矩阵 | < 1 KB |
| 中间张量 (5尺度并行) | ~0.3 GB |
| 系统开销 | ~1.0 GB |
| **总计** | **~5.3 GB** ✅ |

### 9.4 调试建议

```python
# 在 forecast() 中添加形状检查:
print(f"pred_s1: {pred_s1.shape}")
print(f"scale_preds: {[p.shape for p in scale_preds]}")
print(f"weighted_pred: {weighted_pred.shape}")
if hist_refs:
    print(f"hist_refs: {[h.shape for h in hist_refs]}")
print(f"final_pred: {final_pred.shape}")

# 检查梯度:
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

---

## 附录: 文件索引

| 文件 | 说明 | 行数 |
|------|------|------|
| `layers/TAPR_v3.py` | DM + C2F 模块 | ~449 |
| `layers/GRAM_v3.py` | PVDR + AKG 模块 | ~377 |
| `models/TimeLLM_Enhanced_v3.py` | 集成模型 | ~542 |
| `run_main_enhanced_v3.py` | 训练入口 | ~615 |
| `scripts/TimeLLM_ETTh1_enhanced_v3.sh` | 训练脚本 | ~143 |
| `claude-enhanced-v3.md` | 简要技术文档 | ~192 |
| `claude4.md` | 本文档 (详细技术文档) | ~本文 |

---

**最后更新**: 2026-02-20
**状态**: 所有模块开发完成，四类矩阵梯度已验证，待消融实验
**生成工具**: Claude Code (Opus 4.6)
