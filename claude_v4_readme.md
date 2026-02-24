# claude_v4_readme.md - Time-LLM Enhanced v4 技术文档

> **版本**: 2026-02-24
> **状态**: 架构设计完成，待代码实现
> **目的**: 记录增强版 v4 的完整架构升级、模块内部改进、数据流、参数分析和模块接口兼容性
> **核心改变**: v3 模块内部升级 — DM多相采样 / C2F专家投票 / PVDR增强检测 / AKG阈值门控

---

## 目录

1. [架构总览](#1-架构总览)
2. [v3 → v4 变更摘要](#2-v3--v4-变更摘要)
3. [DM v4：多相交错多尺度预测](#3-dm-v4多相交错多尺度预测)
4. [C2F v4：专家投票 + 层级趋势约束](#4-c2f-v4专家投票--层级趋势约束)
5. [PVDR v4：增强检测 + 全局修正](#5-pvdr-v4增强检测--全局修正)
6. [AKG v4：阈值驱动的自适应门控](#6-akg-v4阈值驱动的自适应门控)
7. [完整数据流](#7-完整数据流)
8. [可训练参数分析](#8-可训练参数分析)
9. [损失函数设计](#9-损失函数设计)
10. [模块接口兼容性 (v3 → v4)](#10-模块接口兼容性-v3--v4)

---

## 1. 架构总览

### 1.1 设计哲学

v4 **不改变** v3 的整体数据流和架构图（figure保持不变），仅升级每个模块的**内部实现**：

| 模块 | v3 实现 | v4 升级 | 升级目的 |
|------|---------|---------|---------|
| DM | 平均池化下采样 | 统一k=4多相采样 + 变换区分尺度 | 数据完整保留，变换提供频率分离 |
| C2F | 简单加权求和 | 专家投票(尺度内) + 趋势约束(跨尺度) | 剔除异常分支，保证趋势一致性 |
| PVDR | 3-sigma极端检测 | 逻辑回归 + 贝叶斯变点检测 | 更精准的极端/转折点识别 |
| AKG | 可学习sigmoid门控 | 阈值驱动的门控调节 | 根据PVDR信号动态调整门控 |

### 1.2 整体数据流（与v3完全相同）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Time-LLM Enhanced v4 架构                                │
│              "多相多尺度预测 + 增强检索自适应融合"                            │
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
│  │ ① Baseline Time-LLM │    │ ② DM v4 (Polyphase Multi-   │              │
│  │   (完全不修改)        │    │    Scale) 多相多尺度预测     │              │
│  │                      │    │                              │              │
│  │ Patch → Reprogram    │    │  S2: k=2, diff, 2分支        │              │
│  │ → Prompt → LLM       │    │  S3: k=4, identity, 4分支    │              │
│  │ → FlattenHead        │    │  S4: k=4, smooth, 4分支      │              │
│  │                      │    │  S5: k=4, trend, 4分支       │              │
│  │ "参数矩阵" Θ_1       │    │  "参数矩阵" Θ_2..Θ_5 (共享) │              │
│  └────────┬─────────────┘    └────────┬─────────────────────┘              │
│           │                           │                                     │
│     pred_s1 [B,P,N]       branch_preds: 每尺度k个分支预测                   │
│           │                           │                                     │
│           │                           ▼                                     │
│           │               ┌──────────────────────────────┐                 │
│           │               │ ③-Stage1: 尺度内专家投票      │                 │
│           │               │   MAD异常检测 → 剔除异常分支   │                 │
│           │               │   → consolidated_preds ×4     │                 │
│           │               └────────┬─────────────────────┘                 │
│           │                        │                                        │
│           └────────────┬───────────┘                                        │
│                        │ scale_preds = [s1, voted_s2..s5]                   │
│                        ▼                                                    │
│           ┌──────────────────────────────┐                                  │
│           │ ③-Stage2: C2F v4 跨尺度      │                                  │
│           │   趋势约束融合               │                                  │
│           │                              │                                  │
│           │ "权重矩阵" W [5, pred_len]   │                                  │
│           │ × reliability_factor         │                                  │
│           │ × consistency_factor         │                                  │
│           │ softmax → 加权求和            │                                  │
│           └────────┬─────────────────────┘                                  │
│                    │                                                        │
│            weighted_pred [B,P,N]                                            │
│                    │                                                        │
│           ┌────────┴──────────┐                                             │
│           │                   │                                             │
│           ▼                   ▼                                             │
│  ┌──────────────────┐  ┌──────────────────────────────┐                    │
│  │ scale_preds      │  │ ④ PVDR v4 (Enhanced Dual     │                    │
│  │ (5个尺度预测)     │  │    Retriever) 增强双重检索   │                    │
│  │                  │  │                              │                    │
│  │                  │  │ 逻辑回归极端分类器            │                    │
│  │                  │  │ 贝叶斯变点检测器              │                    │
│  │                  │  │ 增强检索逻辑                  │                    │
│  │                  │  │ 记忆库标签机制                │                    │
│  └────────┬─────────┘  └────────┬─────────────────────┘                    │
│           │                     │                                           │
│           │               hist_refs [B,P,N] × 5                            │
│           │               pvdr_signals                                      │
│           │                     │                                           │
│           └─────────┬───────────┘                                           │
│                     │                                                       │
│                     ▼                                                       │
│           ┌──────────────────────────────┐                                  │
│           │ ⑤ AKG v4 (Threshold-driven  │                                  │
│           │    Gating) 阈值驱动门控      │                                  │
│           │                              │                                  │
│           │ base_gate = sigmoid(C)       │                                  │
│           │ × (1 - extreme_adj)          │                                  │
│           │ × (1 - cp_adj)               │                                  │
│           │ × conf_adj                   │                                  │
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

### 1.3 模块层级结构

```
TimeLLM_Enhanced_v4 (models/TimeLLM_Enhanced_v4.py)
│
├── 原有模块 (Time-LLM Baseline, 完全不修改)
│   ├── normalize_layers (Instance Normalization)
│   ├── patch_embedding (PatchEmbedding)
│   ├── mapping_layer / reprogramming_layer
│   ├── output_projection (FlattenHead)
│   └── llm_model (冻结)
│
├── ★ 创新模块一: TAPR_v4
│   │
│   ├── DM v4: PolyphaseMultiScale (多相交错多尺度预测)
│   │   ├── SignalTransform (复用v3: diff/smooth/trend)
│   │   ├── _polyphase_extract (多相分支提取, 统一k=4, 替代avg_pool)
│   │   ├── _create_patches (分块, 复用v3)
│   │   ├── patch_embeddings: ModuleList (4个Conv1d嵌入)
│   │   └── scale_encoders: ModuleList (4个ScaleEncoder, 同k尺度分支间共享)
│   │
│   └── C2F v4: ExpertVotingFusion (专家投票 + 层级趋势约束)
│       ├── Stage1: IntraScaleExpertVoting
│       │   ├── MAD异常检测 (Median Absolute Deviation)
│       │   └── correction_tracker (修正率追踪器)
│       ├── Stage2: CrossScaleTrendConstraint
│       │   ├── DWT小波分解 (趋势提取)
│       │   └── cosine_similarity (趋势一致性)
│       └── weight_matrix: Parameter [n_scales, pred_len]
│
├── ★ 创新模块二: GRAM_v4
│   │
│   ├── PVDR v4: EnhancedDualRetriever (增强检测 + 双重检索)
│   │   ├── LightweightPatternEncoder (复用v3)
│   │   ├── LogisticExtremeClassifier (逻辑回归极端分类, 替代3-sigma)
│   │   │   └── Linear(5, 1) + sigmoid
│   │   ├── BayesianChangePointDetector (贝叶斯变点检测)
│   │   │   └── 递归二分 + 贝叶斯后验检验
│   │   └── MultiScaleMemoryBank (增强版, 含标签机制)
│   │       ├── keys_{0..4}: Buffer [M, d_repr]
│   │       ├── values_{0..4}: Buffer [M, pred_len]
│   │       ├── labels_{0..4}: Buffer [M] — 'normal'/'extreme'/'turning_point'
│   │       └── thresholds: Parameter [n_scales]
│   │
│   └── AKG v4: ThresholdAdaptiveGating (阈值驱动自适应门控)
│       ├── connection_matrix: Parameter [n_scales, pred_len]
│       ├── fusion_matrix: Parameter [2*n_scales, pred_len]
│       ├── extreme_threshold: Parameter [1] — τ_extreme
│       ├── cp_threshold: Parameter [1] — τ_cp
│       ├── conf_threshold: Parameter [1] — τ_conf
│       └── gate_loss (门控正则化)
│
└── _aux_cache: dict (辅助损失缓存)
```

### 1.4 关键设计思想

| 设计原则 | v3 | v4 升级 |
|----------|-----|---------|
| **Baseline不修改** | 保持不变 | 保持不变 |
| **DM数据完整性** | 平均池化丢弃信息 | 统一k=4多相采样，100%数据利用 |
| **DM多尺度实现** | 不同下采样率(2/4/8/16) | 统一k=4 + 不同信号变换区分频率 |
| **C2F鲁棒性** | 所有分支等权融合 | 先剔除异常分支，再趋势约束 |
| **PVDR检测精度** | 3-sigma固定规则 | 可学习逻辑回归 + 贝叶斯变点 |
| **AKG门控灵活性** | 固定sigmoid门控 | PVDR信号驱动的动态门控调节 |

---

## 2. v3 → v4 变更摘要

### 2.1 模块级变更

| 模块 | v3 方法 | v4 方法 | 变更类型 |
|------|---------|---------|---------|
| DM下采样 | `_downsample(x, rate)` → avg_pool(rate不同) | `_polyphase_extract(x, k=4)` → 统一k=4分支 | **算法替换** |
| DM尺度区分 | 不同下采样率(2/4/8/16) | 统一k=4 + 不同信号变换(diff/id/smooth/trend) | **策略变更** |
| DM编码器 | 每尺度1个前向传播 | 每尺度k个前向传播(共享编码器) | **循环扩展** |
| C2F融合 | softmax(W) → 直接加权 | Stage1(MAD投票) + Stage2(DWT约束) + 加权 | **两阶段扩展** |
| PVDR极端检测 | `ExtremeDetector` (3-sigma) | `LogisticExtremeClassifier` | **模型替换** |
| PVDR变点检测 | 无 | `BayesianChangePointDetector` | **新增组件** |
| PVDR检索逻辑 | 统一Top-K | 分类检索(极端/转折/正常) | **策略增强** |
| PVDR记忆库 | key + value | key + value + label | **字段扩展** |
| AKG门控 | `sigmoid(C)` → 固定 | `sigmoid(C) * adjustments(pvdr_signals)` | **动态调节** |
| AKG阈值 | 无 | 3个可学习阈值参数 | **新增参数** |

### 2.2 接口变更

v4的模块内部接口发生变更（DM输出分支预测，C2F负责投票+融合），但**TAPR整体的外部接口**不变（输入归一化序列+baseline预测，输出weighted_pred）。集成模型 `forecast()` 需要小幅调整。

| 接口 | v3签名 | v4签名 | 变更说明 |
|------|--------|--------|---------|
| DM输入 | `(x_normed, pred_s1)` | `(x_normed, pred_s1)` | 不变 |
| DM输出 | `scale_preds: list of 5×[B,P,N]` | `pred_s1, branch_preds_dict` | ⚠️ 返回分支预测字典 |
| C2F输入 | `(scale_preds)` | `(pred_s1, branch_preds_dict)` | ⚠️ 接收分支预测，内部完成投票 |
| C2F输出 | `weighted_pred: [B,P,N]` | `weighted_pred: [B,P,N]` | 不变 |
| PVDR输入 | `(x_normed, downsample_rates)` | `(x_normed, downsample_rates)` | 不变 |
| PVDR输出 | `hist_refs: list of 5×[B,P,N]` | `hist_refs, pvdr_signals` | ⚠️ 新增pvdr_signals |
| AKG输入 | `(scale_preds, hist_refs)` | `(scale_preds, hist_refs, pvdr_signals)` | ⚠️ 新增pvdr_signals (可选，向后兼容) |
| AKG输出 | `final_pred: [B,P,N]` | `final_pred: [B,P,N]` | 不变 |

---

## 3. DM v4：多相交错多尺度预测

### 3.1 核心改变：平均池化 → 统一k多相分解 + 变换区分

**v3问题**：平均池化下采样会**丢失信息**。对于下采样率 k，k个采样点被平均为1个点，高频细节被不可逆地抹除。

**v4方案**：多相分解（Polyphase Decomposition）+ 信号变换频率分离。

**关键设计决策——为什么统一 k=4？**

多相分解存在刚性约束：分支数 = k = 分支内采样间距。这意味着：
- k=8 → 8个分支，每分支64点（~8 patches）→ 勉强可用
- k=16 → 16个分支，每分支32点（~3 patches）→ **ScaleEncoder输入严重不足**

v4的解决方案：**解耦"粒度控制"和"下采样率"**。
- 所有辅助尺度统一使用 k=4 多相分解，保证每分支128点（~16 patches），编码器输入充足
- "多尺度"效果通过**信号变换**实现：diff提取变化率、smooth提取低频、trend提取宏观趋势
- 变换本身完成了频率分离，不依赖于不同的下采样率

**对比 v3 的"真多尺度"**：

| 维度 | v3 (不同k) | v4 (统一k=4) |
|------|-----------|-------------|
| 间距差异 | k=2/4/8/16 → 真不同粒度 | 统一k=4 → 仅变换不同 |
| ScaleEncoder输入质量 | S5仅~3 patches → 不可靠 | 全部≥16 patches → 充足 |
| 频率分离来源 | 下采样率 + 变换 | 仅变换 |
| 投票有效性 | S5的16分支统计不稳定 | 4分支投票稳定可靠 |
| 前向传播次数 | 30次 (2+4+8+16) | 14次 (2+4+4+4) |

核心判断：**一个粒度正确但预测质量差的S5（3 patches），不如一个粒度降级但预测质量好的S5（16 patches）**。而且在128点上做 kernel=32 的trend变换，已经等效于观察宏观趋势——变换本身提供了频率分离。

### 3.2 多相分解公式

对于粒度为 k 的尺度，原始序列 $x[n]$ 被分解为 k 个多相分支：

$$x_p[m] = x[k \cdot m + p], \quad p = 0, 1, \ldots, k-1$$

其中：
- $p$ 为相位索引（phase index）
- $m$ 为分支内采样索引
- 每个分支长度为 $\lfloor T/k \rfloor$

**关键性质**：$\bigcup_{p=0}^{k-1} \{x_p[m]\}_{m=0}^{T/k-1} = \{x[n]\}_{n=0}^{T-1}$，即所有分支的并集**恰好覆盖全部原始数据**，无冗余无遗漏。

### 3.3 尺度配置（统一k + 变换区分）

v4所有辅助尺度（S3/S4/S5）统一使用 k=4，S2使用 k=2。不同尺度通过**信号变换**区分频率聚焦：

| 尺度 | k | 信号变换 | 变换参数 | 分支数 | 每分支长度 (T=512) | 每分支Patches | 频率聚焦 |
|------|---|---------|---------|--------|-------------------|--------------|---------|
| S1 | 1 | 无 (baseline LLM) | - | 1 | 512 | 63 | 全频 |
| S2 | 2 | diff | - | 2 | 256→255(diff) | ~31 | 中高频变化率 |
| S3 | 4 | identity | - | 4 | 128 | ~16 | 中频原始模式 |
| S4 | 4 | smooth | kernel=25 | 4 | 128 | ~16 | 低频平滑趋势 |
| S5 | 4 | trend | kernel=32 | 4 | 128 | ~16 | 极低频宏观趋势 |
| **总计** | | | | **15** | | | |

**S3/S4/S5的多相分支完全相同**（都是 x[:, 0::4, :], x[:, 1::4, :], x[:, 2::4, :], x[:, 3::4, :]），差异全部来自变换：

```
S3 分支0: x[0], x[4], x[8], ...  → identity → 原样保留 → 中频模式
S4 分支0: x[0], x[4], x[8], ...  → smooth(k=25) → 低频平滑版
S5 分支0: x[0], x[4], x[8], ...  → trend(k=32) → 宏观趋势版
```

### 3.4 共享ScaleEncoder设计

**关键设计**：同一k值的尺度，其多相分支**共享同一个 ScaleEncoder**。

注意：S3/S4/S5虽然都使用k=4，但因为信号变换不同，它们各自拥有独立的ScaleEncoder：

```
尺度编码器配置:
├── S2 (k=2, diff):     2 个分支 → 共享 1 个 ScaleEncoder_2
├── S3 (k=4, identity): 4 个分支 → 共享 1 个 ScaleEncoder_3
├── S4 (k=4, smooth):   4 个分支 → 共享 1 个 ScaleEncoder_4
└── S5 (k=4, trend):    4 个分支 → 共享 1 个 ScaleEncoder_5
总计: 4 个 ScaleEncoder（与v3相同）
```

共享理由：
1. 同一尺度的不同相位分支具有相同的统计特性（平稳性假设）
2. 参数共享保持参数量与v3相同（~100K）
3. k个分支相当于k倍数据增强，有助于编码器泛化

### 3.5 DM v4 前向传播完整流程

```python
def forward(self, x_normed, pred_s1):
    # x_normed: [B, T, N]
    # pred_s1:  [B, pred_len, N] — baseline已完成

    all_branch_preds = {s: [] for s in range(n_aux_scales)}  # 按尺度分组

    for idx, (k, transform_name) in enumerate(self.scale_cfgs):
        branch_preds_s = []

        for p in range(k):
            # 1. 多相提取: [B, T, N] → [B, T/k, N]
            x_p = x_normed[:, p::k, :]          # 步长k，偏移p

            # 2. 信号变换: [B, T/k, N] → [B, T_trans, N]
            fwd_fn, inv_fn = TRANSFORM_REGISTRY[transform_name]
            x_transformed, aux = fwd_fn(x_p)

            # 3. Channel-independent: [B, T_trans, N] → [B*N, T_trans, 1]
            x_ci = x_transformed.permute(0,2,1).reshape(B*N, T_trans, 1)

            # 4. 创建patches: [B*N, num_patches, patch_len]
            patches, num_patches = _create_patches(x_ci, patch_len, stride)

            # 5. Patch嵌入 (该尺度所有分支共享):
            embedded = patch_embed_s(patches.permute(0,2,1)).permute(0,2,1)
            # embedded: [B*N, num_patches, d_model]

            # 6. 必要时插值对齐patch数量
            if cur_patches != expected_patches:
                embedded = F.interpolate(...)

            # 7. ScaleEncoder (该尺度所有分支共享) → pred: [B*N, pred_len]
            pred_p = encoder_s(embedded)

            # 8. Reshape: [B*N, pred_len] → [B, pred_len, N]
            pred_p = pred_p.reshape(B, N, pred_len).permute(0,2,1)

            branch_preds_s.append(pred_p)

        all_branch_preds[idx] = branch_preds_s  # k个 [B, pred_len, N]

    # 输出: pred_s1 + 各尺度的分支预测列表
    # 注意: 尺度内合并由C2F Stage1完成
    return pred_s1, all_branch_preds
```

### 3.6 DM内部详细形状变化 (以S3为例, k=4)

| 步骤 | 操作 | 形状 |
|------|------|------|
| 输入 | x_normed | [4, 512, 7] |
| 多相提取 (p=0) | x[:, 0::4, :] | [4, 128, 7] |
| 多相提取 (p=1) | x[:, 1::4, :] | [4, 128, 7] |
| 多相提取 (p=2) | x[:, 2::4, :] | [4, 128, 7] |
| 多相提取 (p=3) | x[:, 3::4, :] | [4, 128, 7] |
| identity变换 | 无变化 | [4, 128, 7] |
| Channel Indep. | permute+reshape | [28, 128, 1] |
| 创建patches | _create_patches | [28, ~16, 16] |
| Patch嵌入 (共享) | Conv1d(16→32) | [28, ~16, 32] |
| 插值对齐 | F.interpolate | [28, 33, 32] |
| ScaleEncoder (共享) | Conv×2 + Linear | [28, 96] |
| Reshape | → [B, pred_len, N] | [4, 96, 7] |
| **4个分支输出** | branch_preds_s3 | **4 × [4, 96, 7]** |

---

## 4. C2F v4：专家投票 + 层级趋势约束

### 4.1 两阶段设计概述

v3的C2F直接对5个尺度预测做softmax加权融合。v4将融合过程分为两个阶段：

```
DM v4输出 (每尺度k个分支预测)
        │
        ▼
┌─────────────────────────────┐
│ Stage 1: 尺度内专家投票      │
│   S2: 2个分支 → MAD投票 → 1 │
│   S3: 4个分支 → MAD投票 → 1 │
│   S4: 4个分支 → MAD投票 → 1 │
│   S5: 4个分支 → MAD投票 → 1 │
│   + 修正率追踪               │
└────────────┬────────────────┘
             │ consolidated_preds: 5 × [B, P, N]
             ▼
┌─────────────────────────────┐
│ Stage 2: 跨尺度趋势约束      │
│   DWT小波分解各尺度趋势       │
│   S5趋势为基准方向            │
│   不一致尺度 → 降权惩罚       │
│   可靠性标记 → 额外降权       │
└────────────┬────────────────┘
             │ weighted_pred: [B, P, N]
             ▼
```

### 4.2 Stage 1 — 尺度内专家投票

对每个辅助尺度的 k 个分支预测进行异常检测和投票合并：

```python
def intra_scale_expert_voting(self, branch_preds):
    """
    branch_preds: list of k × [B, pred_len, N]
    return: consolidated_pred [B, pred_len, N], num_outliers int
    """
    stacked = torch.stack(branch_preds, dim=0)  # [k, B, P, N]

    # 1. 计算中位数预测作为参考
    median_pred = stacked.median(dim=0).values   # [B, P, N]

    # 2. 计算每个分支与中位数的偏差
    deviations = (stacked - median_pred.unsqueeze(0)).abs()  # [k, B, P, N]

    # 3. MAD (Median Absolute Deviation)
    MAD = deviations.median(dim=0).values        # [B, P, N]

    # 4. 异常判断: 偏差 > 3 × MAD 的分支为异常
    per_branch_deviation = deviations.mean(dim=(1,2,3))  # [k]
    MAD_scalar = MAD.mean()
    is_outlier = per_branch_deviation > 3 * MAD_scalar   # [k] bool

    # 5. 非异常分支等权平均
    valid_mask = ~is_outlier                              # [k]
    if valid_mask.sum() == 0:
        # 所有分支均异常 → 退化为全部平均
        consolidated = stacked.mean(dim=0)
    else:
        consolidated = stacked[valid_mask].mean(dim=0)    # [B, P, N]

    num_outliers = is_outlier.sum().item()
    return consolidated, num_outliers
```

**修正率追踪**：

```python
# 在训练过程中持续追踪每个尺度的修正率
self.correction_counts = {s: 0 for s in range(n_aux_scales)}
self.total_counts = {s: 0 for s in range(n_aux_scales)}

# 每次投票后更新
self.correction_counts[s] += num_outliers
self.total_counts[s] += k

# 修正率计算
correction_rate_s = correction_counts[s] / total_counts[s]

# 可靠性标记: 修正率 > 30% 的尺度标记为"不可靠"
RELIABILITY_THRESHOLD = 0.3
is_unreliable_s = correction_rate_s > RELIABILITY_THRESHOLD
```

### 4.3 Stage 2 — 跨尺度趋势约束

**小波分解 (DWT)**：

对每个尺度的合并预测进行一级小波分解，分离低频趋势和高频波动：

```python
def cross_scale_trend_constraint(self, consolidated_preds, reliability_flags):
    """
    consolidated_preds: list of 5 × [B, pred_len, N]
    reliability_flags: list of 5 × bool (是否不可靠)
    """
    # 1. DWT分解每个尺度预测
    approx_list = []  # 低频趋势
    detail_list = []  # 高频波动
    for s, pred in enumerate(consolidated_preds):
        approx_s, detail_s = dwt_1d(pred)  # Haar小波
        approx_list.append(approx_s)
        detail_list.append(detail_s)

    # 2. 以最粗尺度 (S5) 趋势为基准
    ref_trend = approx_list[-1]  # S5的趋势近似

    # 3. 趋势一致性检查
    consistency_scores = []
    for s in range(len(consolidated_preds)):
        score = F.cosine_similarity(
            approx_list[s].reshape(B, -1),
            ref_trend.reshape(B, -1),
            dim=-1
        ).mean()  # scalar
        consistency_scores.append(score)

    # 4. 计算调整后的权重
    weights = F.softmax(self.weight_matrix, dim=0)  # [5, pred_len]

    for s in range(len(consolidated_preds)):
        # 趋势一致性惩罚 (仅在明显不一致时)
        if consistency_scores[s] < -0.3:  # 明显趋势相反
            weights[s] *= self.decay_factor

        # 可靠性惩罚
        if reliability_flags[s]:
            weights[s] *= self.reliability_penalty  # 额外降权

    # 5. 重新归一化并融合
    weights = weights / weights.sum(dim=0, keepdim=True)
    stacked = torch.stack(consolidated_preds, dim=0)
    w = weights.unsqueeze(1).unsqueeze(-1)
    weighted_pred = (w * stacked).sum(dim=0)

    return weighted_pred
```

### 4.4 DWT实现 (Haar小波)

```python
def dwt_1d(x):
    """
    一级Haar小波分解
    x: [B, T, N]
    returns: approx [B, T//2, N], detail [B, T//2, N]
    """
    # 偶数长度处理
    if x.shape[1] % 2 != 0:
        x = x[:, :-1, :]

    x_even = x[:, 0::2, :]  # [B, T//2, N]
    x_odd = x[:, 1::2, :]   # [B, T//2, N]

    approx = (x_even + x_odd) / math.sqrt(2)   # 低频近似
    detail = (x_even - x_odd) / math.sqrt(2)    # 高频细节

    return approx, detail
```

### 4.5 C2F v4 完整接口

```python
class ExpertVotingFusion(nn.Module):
    """
    C2F v4: 专家投票 + 层级趋势约束

    Stage 1: 尺度内MAD投票 → 剔除异常分支 → consolidated_preds
    Stage 2: 跨尺度DWT趋势约束 → 权重调节 → weighted_pred

    输入接口兼容v3:
      scale_preds: list of 5 × [B, pred_len, N]  ← 由DM合并后传入
      或 branch_preds: dict of {scale_idx: list of k × [B, P, N]}  ← DM v4直传

    输出接口兼容v3:
      weighted_pred: [B, pred_len, N]
    """

    def __init__(self, n_scales, pred_len, decay_factor=0.5,
                 reliability_penalty=0.3, mad_threshold=3.0):
        self.weight_matrix = nn.Parameter(torch.ones(n_scales, pred_len) / n_scales)
        # ...

    def forward(self, pred_s1, branch_preds_dict, training=True):
        # Stage 1: 投票
        consolidated = [pred_s1]  # S1无需投票
        reliability_flags = [False]  # S1默认可靠
        for s_idx, branches in branch_preds_dict.items():
            c_pred, n_outliers = self.intra_scale_expert_voting(branches)
            consolidated.append(c_pred)
            reliability_flags.append(self._update_reliability(s_idx, n_outliers, len(branches)))

        # Stage 2: 趋势约束融合
        weighted_pred = self.cross_scale_trend_constraint(consolidated, reliability_flags)
        return weighted_pred
```

---

## 5. PVDR v4：增强检测 + 全局修正

### 5.1 检测模块升级

#### 5.1.1 逻辑回归极端分类器

**替代v3的3-sigma规则**，使用可学习的逻辑回归模型判断极端数据：

```python
class LogisticExtremeClassifier(nn.Module):
    """
    可学习的极端数据分类器
    输入: 5个统计特征 [mean, std, max, min, trend]
    输出: extreme_probability ∈ [0, 1]
    """
    def __init__(self):
        self.linear = nn.Linear(5, 1)  # 仅5+1=6个参数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, T, N]
        # 提取统计特征
        features = torch.stack([
            x.mean(dim=1).mean(dim=-1),   # mean: [B]
            x.std(dim=1).mean(dim=-1),    # std: [B]
            x.max(dim=1).values.mean(dim=-1),   # max: [B]
            x.min(dim=1).values.mean(dim=-1),   # min: [B]
            (x[:,-1,:] - x[:,0,:]).mean(dim=-1), # trend: [B]
        ], dim=-1)  # [B, 5]

        logits = self.linear(features)    # [B, 1]
        extreme_prob = self.sigmoid(logits).squeeze(-1)  # [B]
        return extreme_prob
```

**训练方式**：在记忆库构建阶段，使用 z-score 标签（|z| > 3σ 的样本标记为极端）作为监督信号，对逻辑回归进行少量梯度更新。

**优势**：
- v3的3-sigma是固定规则，对不同分布的数据表现不一致
- 逻辑回归可以学习到数据特有的极端判别边界
- 仅6个参数，计算量可忽略

#### 5.1.2 贝叶斯变点检测器

**新增组件**：检测输入序列中的结构性变化点（转折点）。

```python
class BayesianChangePointDetector(nn.Module):
    """
    基于贝叶斯后验的递归二分变点检测

    核心思想: 在每个候选分割点评估 "分割后两段分别建模" vs "整段统一建模" 的贝叶斯因子

    参数: 3个可学习参数 (先验超参: prior_mean, prior_var, noise_var)
    """
    def __init__(self, min_segment=16, max_depth=4):
        self.min_segment = min_segment  # 最小段长
        self.max_depth = max_depth      # 最大递归深度
        # 先验超参
        self.prior_mean = nn.Parameter(torch.zeros(1))
        self.prior_var = nn.Parameter(torch.ones(1))
        self.noise_var = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        x: [B, T, N] → 取均值 → [B, T]
        returns:
          change_points: [B, max_cp] — 变点位置 (-1表示无)
          confidence: [B, max_cp] — 变点置信度
          near_end_score: [B] — 序列末尾附近有变点的概率
        """
        x_flat = x.mean(dim=-1)  # [B, T]
        B, T = x_flat.shape

        change_points = []
        confidences = []

        for b in range(B):
            cps, confs = self._recursive_partition(x_flat[b], 0, T, depth=0)
            change_points.append(cps)
            confidences.append(confs)

        # 关键信号: 末尾附近是否有变点
        near_end_score = self._compute_near_end_score(change_points, confidences, T)

        return change_points, confidences, near_end_score

    def _recursive_partition(self, segment, start, end, depth):
        """递归二分: 在当前段内寻找最优分割点"""
        if end - start < 2 * self.min_segment or depth >= self.max_depth:
            return [], []

        best_score = -float('inf')
        best_pos = -1

        for pos in range(start + self.min_segment, end - self.min_segment):
            # 贝叶斯因子: log p(data | 分割) - log p(data | 不分割)
            left = segment[start:pos]
            right = segment[pos:end]
            whole = segment[start:end]

            log_bf = (self._log_marginal(left) + self._log_marginal(right)
                     - self._log_marginal(whole))

            if log_bf > best_score:
                best_score = log_bf
                best_pos = pos

        if best_score > 0:  # 分割优于不分割
            confidence = torch.sigmoid(torch.tensor(best_score))
            # 递归检测左右子段
            left_cps, left_confs = self._recursive_partition(segment, start, best_pos, depth+1)
            right_cps, right_confs = self._recursive_partition(segment, best_pos, end, depth+1)
            return left_cps + [best_pos] + right_cps, left_confs + [confidence] + right_confs
        else:
            return [], []

    def _log_marginal(self, segment):
        """计算段的对数边际似然 (正态-正态共轭)"""
        n = len(segment)
        s_mean = segment.mean()
        s_var = segment.var() + 1e-8

        prior_var = F.softplus(self.prior_var)
        noise_var = F.softplus(self.noise_var)

        # 正态-正态共轭的边际似然
        post_var = 1.0 / (1.0/prior_var + n/noise_var)
        post_mean = post_var * (self.prior_mean/prior_var + n*s_mean/noise_var)

        log_ml = (-n/2 * torch.log(2 * math.pi * noise_var)
                  + 0.5 * torch.log(post_var / prior_var)
                  - 0.5 * (n*s_var/noise_var + s_mean**2 * n/noise_var
                           - post_mean**2/post_var + self.prior_mean**2/prior_var))
        return log_ml

    def _compute_near_end_score(self, change_points, confidences, T):
        """计算末尾附近变点的概率"""
        near_end_threshold = T * 0.8  # 后20%区域
        scores = []
        for cps, confs in zip(change_points, confidences):
            near_end = [c for cp, c in zip(cps, confs) if cp > near_end_threshold]
            scores.append(max(near_end) if near_end else 0.0)
        return torch.tensor(scores)
```

### 5.2 增强检索逻辑

基于逻辑回归和变点检测的结果，PVDR v4的检索策略分为四种模式：

```python
def retrieve_enhanced(self, x_normed, scale_idx, extreme_prob, near_end_score, max_sim):
    """
    增强检索逻辑

    参数:
        extreme_prob: [B] — 极端数据概率
        near_end_score: [B] — 末尾变点概率
        max_sim: [B] — 最大相似度
    """
    # 基础阈值
    threshold = torch.sigmoid(self.memory_bank.thresholds[scale_idx])

    # 模式判断 (per-sample)
    for b in range(B):
        if max_sim[b] > 0.95:
            # 模式A: 极相似数据 → 直接检索修正
            # 不做阈值过滤，直接用Top-1结果
            mode = 'direct'
            k_retrieve = 1

        elif extreme_prob[b] > self.extreme_tau:
            # 模式B: 极端数据 → 降低阈值，增加检索量
            # 从"extreme"标签子集中优先检索
            mode = 'extreme'
            threshold_adj = threshold - 0.2
            k_retrieve = self.top_k * 2  # 检索更多候选

        elif near_end_score[b] > self.cp_tau:
            # 模式C: 转折点 → 从"turning_point"子集检索
            mode = 'turning_point'
            k_retrieve = self.top_k

        else:
            # 模式D: 正常数据 → 标准Top-K检索
            mode = 'normal'
            k_retrieve = self.top_k

    return hist_ref, valid_mask, retrieval_confidence
```

### 5.3 记忆库增强

v4的记忆库在v3基础上增加标签字段：

```python
class EnhancedMultiScaleMemoryBank(nn.Module):
    """
    增强版记忆库: key + value + label

    labels取值:
    - 0: normal (正常数据)
    - 1: extreme (极端数据)
    - 2: turning_point (转折点附近)
    """
    def __init__(self, n_scales, d_repr, pred_len, max_samples=5000):
        for s in range(n_scales):
            self.register_buffer(f'keys_{s}', torch.zeros(max_samples, d_repr))
            self.register_buffer(f'values_{s}', torch.zeros(max_samples, pred_len))
            self.register_buffer(f'labels_{s}', torch.zeros(max_samples, dtype=torch.long))  # NEW
        self.thresholds = nn.Parameter(torch.full((n_scales,), 0.8))

    def build_with_labels(self, all_x, all_y, extreme_classifier, cp_detector):
        """构建时自动标注标签"""
        extreme_probs = extreme_classifier(all_x)
        _, _, near_end_scores = cp_detector(all_x)

        labels = torch.zeros(len(all_x), dtype=torch.long)
        labels[extreme_probs > 0.5] = 1       # extreme
        labels[near_end_scores > 0.5] = 2     # turning_point
        # (极端 + 转折同时满足时优先标记为turning_point)

        # 存储 keys, values, labels
        # ...
```

### 5.4 PVDR v4输出信号

PVDR v4除了返回 hist_refs（与v3兼容），还返回丰富的信号供AKG使用：

```python
pvdr_signals = {
    'extreme_score': extreme_prob,          # [B] — 极端数据概率
    'change_point_score': near_end_score,   # [B] — 转折点概率
    'retrieval_confidence': conf_per_scale, # [B, 5] — 每尺度检索质量
    'hist_refs': hist_refs,                 # 5 × [B, pred_len, N]
}
```

---

## 6. AKG v4：阈值驱动的自适应门控

### 6.1 核心改变

v3的AKG使用固定的 `sigmoid(C)` 门控。v4引入PVDR信号驱动的动态调节：

```
v3: effective_gate = sigmoid(C)
v4: effective_gate = sigmoid(C) × (1 - extreme_adj) × (1 - cp_adj) × conf_adj
```

### 6.2 PVDR信号集成

```python
class ThresholdAdaptiveGating(nn.Module):
    def __init__(self, n_scales, pred_len):
        # 复用v3的核心矩阵
        self.connection_matrix = nn.Parameter(torch.zeros(n_scales, pred_len))
        self.fusion_matrix = nn.Parameter(torch.zeros(2 * n_scales, pred_len))

        # NEW: 可学习阈值参数
        self.extreme_threshold = nn.Parameter(torch.tensor(0.5))   # τ_extreme
        self.cp_threshold = nn.Parameter(torch.tensor(0.5))        # τ_cp
        self.conf_threshold = nn.Parameter(torch.tensor(0.5))      # τ_conf
```

### 6.3 阈值驱动的门控调节

```python
def compute_effective_gate(self, base_gate, pvdr_signals):
    """
    base_gate: [n_scales, pred_len] — sigmoid(C)
    pvdr_signals: dict with extreme_score, change_point_score, retrieval_confidence
    return: effective_gate [n_scales, pred_len]
    """
    extreme_score = pvdr_signals['extreme_score']          # [B]
    cp_score = pvdr_signals['change_point_score']          # [B]
    conf = pvdr_signals['retrieval_confidence']             # [B, n_scales]

    # 阈值函数: 软阈值 (可微分)
    τ_ext = torch.sigmoid(self.extreme_threshold)
    τ_cp = torch.sigmoid(self.cp_threshold)
    τ_conf = torch.sigmoid(self.conf_threshold)

    # 极端数据调节: 极端时降低gate → 更信任历史
    extreme_adj = torch.sigmoid(
        10 * (extreme_score - τ_ext)  # 陡峭sigmoid实现软阈值
    )  # [B] → [B, 1, 1] 广播

    # 变点调节: 变点附近降低gate → 更信任历史
    cp_adj = torch.sigmoid(
        10 * (cp_score - τ_cp)
    )  # [B] → [B, 1, 1] 广播

    # 检索置信度调节: 检索置信高时允许更多历史影响
    conf_adj = torch.sigmoid(
        10 * (conf - τ_conf)
    )  # [B, n_scales] → [B, n_scales, 1] 广播

    # 最终有效门控
    effective_gate = base_gate.unsqueeze(0)  # [1, n_scales, pred_len]
    effective_gate = effective_gate * (1 - extreme_adj.unsqueeze(1).unsqueeze(-1))
    effective_gate = effective_gate * (1 - cp_adj.unsqueeze(1).unsqueeze(-1))
    effective_gate = effective_gate * conf_adj.unsqueeze(-1)

    return effective_gate  # [B, n_scales, pred_len]
```

### 6.4 门控语义解读

| 场景 | extreme_adj | cp_adj | conf_adj | effective_gate | 含义 |
|------|-------------|--------|----------|----------------|------|
| 正常 + 低置信检索 | 低 | 低 | 低 | ≈ sigmoid(C) × 0 → 低 | 谨慎，少用检索 |
| 正常 + 高置信检索 | 低 | 低 | 高 | ≈ sigmoid(C) × 1 → 正常 | 标准门控 |
| 极端 + 高置信检索 | 高 | 低 | 高 | ≈ sigmoid(C) × 0.3 → 低 | 信任历史 |
| 变点 + 高置信检索 | 低 | 高 | 高 | ≈ sigmoid(C) × 0.3 → 低 | 信任历史 |
| 极端 + 变点 | 高 | 高 | 高 | ≈ sigmoid(C) × 0.1 → 很低 | 强信任历史 |

### 6.5 融合矩阵（与v3相同）

```python
def forward(self, scale_preds, hist_refs, pvdr_signals=None):
    # Stage 1: 门控（v4增强版）
    base_gate = torch.sigmoid(self.connection_matrix)  # [5, 96]

    if pvdr_signals is not None:
        effective_gate = self.compute_effective_gate(base_gate, pvdr_signals)
        # [B, 5, 96]
    else:
        effective_gate = base_gate.unsqueeze(0)  # 降级为v3行为

    connected = []
    for s in range(n_scales):
        g = effective_gate[:, s, :].unsqueeze(-1)  # [B, 96, 1]
        connected_s = g * scale_preds[s] + (1 - g) * hist_refs[s]
        connected.append(connected_s)

    # Stage 2: 融合矩阵（与v3完全相同）
    all_sources = scale_preds + connected  # 10个来源
    fusion_weights = F.softmax(self.fusion_matrix, dim=0)  # [10, 96]
    stacked = torch.stack(all_sources, dim=0)               # [10, B, 96, N]
    w = fusion_weights.unsqueeze(1).unsqueeze(-1)            # [10, 1, 96, 1]
    final_pred = (w * stacked).sum(dim=0)                    # [B, 96, N]

    return final_pred
```

---

## 7. 完整数据流

### 7.1 形状变化流程 (ETTh1 示例: B=4, T=512, N=7, pred_len=96)

| 阶段 | 形状 | 说明 |
|------|------|------|
| **输入** | `[4, 512, 7]` | B=4, seq_len=512, n_vars=7 |
| **Instance Norm** | `[4, 512, 7]` | 消除尺度差异 |
| **Baseline Forward** | `[4, 96, 7]` | pred_s1 (完全复用v3/原始) |
| --- **DM v4** --- | | |
| S2多相提取 (p=0) | `[4, 256, 7]` | x[:, 0::2, :] |
| S2多相提取 (p=1) | `[4, 256, 7]` | x[:, 1::2, :] |
| S2 diff变换 | 2× `[4, 255, 7]` | 每分支独立变换 |
| S2 ScaleEncoder(共享) | 2× `[4, 96, 7]` | 2个分支预测 |
| S3多相提取 (p=0..3) | 4× `[4, 128, 7]` | k=4, identity |
| S3 ScaleEncoder(共享) | 4× `[4, 96, 7]` | 4个分支预测 |
| S4多相提取 (p=0..3) | 4× `[4, 128, 7]` | k=4, 与S3相同原始分支 |
| S4 smooth变换 | 4× `[4, 128, 7]` | 移动平均(kernel=25) |
| S4 ScaleEncoder(共享) | 4× `[4, 96, 7]` | 4个分支预测 |
| S5多相提取 (p=0..3) | 4× `[4, 128, 7]` | k=4, 与S3相同原始分支 |
| S5 trend变换 | 4× `[4, 128, 7]` | 大窗口平均(kernel=32) |
| S5 ScaleEncoder(共享) | 4× `[4, 96, 7]` | 4个分支预测 |
| --- **C2F v4 Stage1** --- | | |
| S2投票 (2→1) | `[4, 96, 7]` | MAD异常剔除 |
| S3投票 (4→1) | `[4, 96, 7]` | MAD异常剔除 |
| S4投票 (4→1) | `[4, 96, 7]` | MAD异常剔除 |
| S5投票 (4→1) | `[4, 96, 7]` | MAD异常剔除 |
| consolidated_preds | 5× `[4, 96, 7]` | [s1, voted_s2..s5] |
| --- **C2F v4 Stage2** --- | | |
| DWT分解 | 5× `[4, 48, 7]` | 趋势近似 (半长) |
| 趋势一致性 | 5× scalar | cosine_similarity |
| 权重调节 | `[5, 96]` | W × reliability × consistency |
| **weighted_pred** | `[4, 96, 7]` | 最终加权融合 |
| --- **PVDR v4** --- | | |
| 极端分类 | `[4]` | extreme_prob |
| 变点检测 | `[4]` | near_end_score |
| 多尺度检索 ×5 | 5× `[4, 96, 7]` | hist_refs (分模式检索) |
| 检索置信度 | `[4, 5]` | per-scale confidence |
| --- **AKG v4** --- | | |
| base_gate | `[5, 96]` | sigmoid(C) |
| effective_gate | `[4, 5, 96]` | 阈值调节后 (per-batch) |
| connected | 5× `[4, 96, 7]` | 门控混合结果 |
| all_sources | 10× `[4, 96, 7]` | 5 pred + 5 connected |
| **final_pred** | `[4, 96, 7]` | F矩阵融合 |
| **Denorm** | `[4, 96, 7]` | 反归一化 → 输出 |

### 7.2 计算量分析

| 操作 | v3次数 | v4次数 | 数据量对比 |
|------|--------|--------|-----------|
| ScaleEncoder前向 | 4次 (各1次) | 14次 (2+4+4+4) | 每次处理T/k数据 |
| 总处理数据量 | Σ T/k_s = T×(1/2+1/4+1/8+1/16)≈0.94T | Σ 分支: 2×(T/2)+4×3×(T/4) = 4T | v4处理更多数据 |
| MAD投票 | 无 | 4次 | 轻量比较运算 |
| DWT小波 | 无 | 5次 | 轻量加减运算 |
| 逻辑回归 | 无 | 1次 | 5→1线性层 |
| 贝叶斯变点 | 无 | 1次 | 递归二分 (max_depth=4) |
| AKG门控计算 | sigmoid(C) | sigmoid(C) + 3个调节 | 额外3次逐元素运算 |

**关键结论**: 总处理数据量与v3相同，新增计算均为轻量操作。

---

## 8. 可训练参数分析

### 8.1 v3 → v4 参数对比

| 组件 | v3 参数量 | v4 参数量 | 变化 | 说明 |
|------|----------|----------|------|------|
| ScaleEncoders (×4, 共享) | ~100K | ~100K | 不变 | 分支间共享，参数量不增 |
| C2F weight_matrix W | 480 | 480 | 不变 | [5, pred_len] |
| ExtremeDetector | ~0 (无参数规则) | ~6 | 新增 | Linear(5,1) + bias |
| BayesianChangePointDetector | 0 | 3 | 新增 | prior_mean, prior_var, noise_var |
| AKG connection_matrix C | 480 | 480 | 不变 | [5, pred_len] |
| AKG fusion_matrix F | 960 | 960 | 不变 | [10, pred_len] |
| AKG threshold params | 0 | 3 | 新增 | τ_extreme, τ_cp, τ_conf |
| Memory labels buffer | 0 | ~5000×5 | 新增 | buffer不参与训练 |
| **总可训练新增** | **~103K** | **~103K + ~12** | **可忽略** | |

### 8.2 显存影响

| 新增项 | 显存占用 |
|--------|---------|
| 14个分支预测 (中间张量) | 14 × [B, pred_len, N] × 4byte ≈ 14 × 4×96×7×4 = ~150KB |
| DWT分解结果 | 5 × [B, pred_len/2, N] × 4byte ≈ ~40KB |
| 逻辑回归/变点检测 | < 1KB |
| 标签buffer | 5000 × 5 × 4byte = ~100KB |
| **总新增** | **~0.3MB** (可忽略) |

### 8.3 v4总显存估算

| 组件 | 显存 |
|------|------|
| Qwen 2.5 3B (4-bit, 6层) | ~1.5 GB |
| Time-LLM 原始可训练参数 | ~0.5 GB |
| DM v4: 4个ScaleEncoder (~25K × 4) | ~2 MB |
| DM v4: 14个分支中间张量 | ~0.15 MB |
| C2F v4: 权重矩阵 + DWT | < 1 KB |
| PVDR v4: 5个记忆库 + 标签 | ~5.1 MB |
| PVDR v4: 逻辑回归 + 变点检测 | < 1 KB |
| AKG v4: 联系矩阵 + 融合矩阵 + 阈值 | < 1 KB |
| 中间张量 (前向/反向) | ~0.3 GB |
| 系统开销 | ~1.0 GB |
| **总计** | **~3.3 GB** (具体显存占用可通过命令参数调节) |

---

## 9. 损失函数设计

### 9.1 总损失公式

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \alpha \cdot (\lambda_{consist} \cdot \mathcal{L}_{consist} + \lambda_{gate} \cdot \mathcal{L}_{gate} + \lambda_{expert} \cdot \mathcal{L}_{expert})$$

其中 $\alpha = \min(1.0, \text{step} / 500)$ 为warmup因子。

### 9.2 各损失分量

**主损失 (不变)**:
$$\mathcal{L}_{main} = \text{MSE}(\hat{Y}_{final}, Y_{true})$$

**一致性损失 (v4升级: KL → 小波趋势一致性)**:

v3使用KL散度比较趋势方向分布。v4使用小波趋势一致性：

$$\mathcal{L}_{consist} = \frac{1}{S} \sum_{s=1}^{S} \max(0, -\cos(\text{approx}_s, \text{approx}_{S5}))$$

- 仅当趋势方向明显相反时产生惩罚 (cosine < 0)
- 比KL散度更直接衡量趋势一致性

**门控正则化损失 (不变)**:
$$\mathcal{L}_{gate} = -\frac{1}{S \times T_{pred}} \sum_{s,t} |\sigma(C_{s,t}) - 0.5|$$

**专家修正损失 (v4新增)**:

$$\mathcal{L}_{expert} = \frac{1}{S_{aux}} \sum_{s=2}^{S} \text{correction\_rate}_s$$

- 惩罚修正率高的尺度，鼓励各分支预测趋于一致
- 修正率由C2F Stage1的投票过程实时计算

### 9.3 损失权重推荐值

| 超参数 | 符号 | v3默认值 | v4推荐值 | 说明 |
|--------|------|---------|---------|------|
| 一致性权重 | λ_consist | 0.1 | 0.1 | 不变 |
| 门控正则权重 | λ_gate | 0.01 | 0.01 | 不变 |
| 专家修正权重 | λ_expert | 无 | 0.05 | 新增，中等强度 |
| Warmup步数 | warmup_steps | 500 | 500 | 不变 |

### 9.4 辅助损失计算流程

```
训练时:
├── 主损失: MSE(final_pred, batch_y_target)
│
├── 一致性损失 (lambda_consist=0.1):
│   └── L_consist: 小波趋势一致性
│       ├── DWT分解5个consolidated_preds
│       ├── 各尺度趋势近似 vs S5趋势近似
│       └── max(0, -cosine_similarity) 取平均
│
├── 门控正则化 (lambda_gate=0.01):
│   └── L_gate: -mean(|sigmoid(C) - 0.5|)
│       └── 鼓励门控做出明确决策
│
└── 专家修正损失 (lambda_expert=0.05):
    └── L_expert: mean(correction_rate_s)
        └── 惩罚高修正率尺度

总损失 = MSE + warmup_factor × (0.1 × L_consist + 0.01 × L_gate + 0.05 × L_expert)
```

---

## 10. 模块接口兼容性 (v3 → v4)

### 10.1 接口变更清单

| 接口 | v3签名 | v4签名 | 兼容性 |
|------|--------|--------|--------|
| `DM.forward()` | `(x_normed, pred_s1) → scale_preds` | `(x_normed, pred_s1) → pred_s1, branch_preds_dict` | ⚠️ 输出格式变更 |
| `C2F.forward()` | `(scale_preds) → weighted_pred` | `(pred_s1, branch_preds_dict) → weighted_pred` | ⚠️ 输入格式变更 |
| `PVDR.retrieve_all_scales()` | `(x_normed, ds_rates) → hist_refs` | `(x_normed, ds_rates) → hist_refs, pvdr_signals` | ⚠️ 输出扩展 |
| `AKG.forward()` | `(scale_preds, hist_refs) → final_pred` | `(scale_preds, hist_refs, pvdr_signals) → final_pred` | ✅ 向后兼容 (pvdr_signals=None) |

### 10.2 集成模型 forecast() 变更

```python
# v3 forecast():
def forecast(self, x_enc, ...):
    pred_s1 = self._baseline_forecast(x_enc, ...)
    x_normed = self.normalize_layers(x_enc, 'norm')

    if self.use_tapr:
        scale_preds = self.dm(x_normed, pred_s1)
        weighted_pred = self.c2f(scale_preds)
    else:
        scale_preds = [pred_s1]
        weighted_pred = pred_s1

    if self.use_gram:
        hist_refs = self.pvdr.retrieve_all_scales(x_normed, self.downsample_rates)
        akg_preds = list(scale_preds)
        akg_preds[0] = weighted_pred
        final_pred = self.akg(akg_preds, hist_refs)
    else:
        final_pred = weighted_pred

    return self.normalize_layers(final_pred, 'denorm')

# v4 forecast():
def forecast(self, x_enc, ...):
    pred_s1 = self._baseline_forecast(x_enc, ...)
    x_normed = self.normalize_layers(x_enc, 'norm')

    if self.use_tapr:
        # v4: DM返回分支预测，C2F内部完成投票+融合
        pred_s1_out, branch_preds_dict = self.dm(x_normed, pred_s1)
        weighted_pred = self.c2f(pred_s1_out, branch_preds_dict)
        # C2F内部: Stage1投票 → consolidated_preds → Stage2趋势约束 → weighted_pred
        consolidated_preds = self.c2f.get_consolidated_preds()  # 缓存供AKG使用
    else:
        consolidated_preds = [pred_s1]
        weighted_pred = pred_s1

    if self.use_gram:
        # v4: PVDR返回增强信号
        hist_refs, pvdr_signals = self.pvdr.retrieve_all_scales(x_normed, self.downsample_rates)
        akg_preds = list(consolidated_preds)
        akg_preds[0] = weighted_pred
        # v4: AKG接收pvdr_signals进行阈值门控
        final_pred = self.akg(akg_preds, hist_refs, pvdr_signals=pvdr_signals)
    else:
        final_pred = weighted_pred

    return self.normalize_layers(final_pred, 'denorm')
```

### 10.3 梯度流向（v4）

```
final_pred
    │
    ├── AKG.fusion_matrix ← softmax → 加权求和 → ✅ 梯度OK
    │
    ├── AKG.connection_matrix ← sigmoid × adjustments → 门控 → ✅ 梯度OK
    │
    ├── AKG.threshold_params ← sigmoid → 软阈值调节 → ✅ 梯度OK (新增)
    │
    ├── C2F.weight_matrix ← softmax × reliability × consistency → ✅ 梯度OK
    │   └── akg_preds[0] = weighted_pred (保证梯度传播)
    │
    ├── LogisticExtremeClassifier ← extreme_adj → AKG调节 → ✅ 梯度OK (新增)
    │
    ├── BayesianCP.prior_params ← log_marginal → near_end_score → ✅ 梯度OK (新增)
    │
    └── ScaleEncoder参数 ← Conv + Linear → 分支预测 → 投票后传播
        └── 通过 consolidated_preds → ✅ 梯度OK
```

---

## 附录A: v3 → v4 迁移清单

| 步骤 | 操作 | 影响文件 |
|------|------|---------|
| 1 | 创建 `layers/TAPR_v4.py` | 新文件 |
| 2 | 创建 `layers/GRAM_v4.py` | 新文件 |
| 3 | 创建 `models/TimeLLM_Enhanced_v4.py` | 新文件 |
| 4 | 创建 `run_main_enhanced_v4.py` | 新文件 |
| 5 | 创建 `scripts/TimeLLM_ETTh1_enhanced_v4.sh` | 新文件 |
| 6 | 新增命令参数 | run_main_enhanced_v4.py |

### 新增命令参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lambda_expert` | float | 0.05 | 专家修正损失权重 |
| `--mad_threshold` | float | 3.0 | MAD异常检测倍数 |
| `--reliability_threshold` | float | 0.3 | 尺度不可靠修正率阈值 |
| `--reliability_penalty` | float | 0.3 | 不可靠尺度降权因子 |
| `--cp_min_segment` | int | 16 | 变点检测最小段长 |
| `--cp_max_depth` | int | 4 | 变点检测最大递归深度 |

**其余参数完全复用v3**（`--use_tapr`, `--use_gram`, `--n_scales`, 等）。

---

## 附录B: 消融实验指南 (v4扩展)

### B.1 v3沿用的四组主消融

```bash
# E1-E4: 与v3相同
USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh  # E1
USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh  # E2
USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh  # E3
USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh  # E4
```

### B.2 v4新增消融

| 实验 | 变量 | 说明 |
|------|------|------|
| E5 | DM v4 (polyphase) vs DM v3 (avg_pool) | 多相采样的增益 |
| E6 | C2F v4 (voting+trend) vs C2F v3 (直接加权) | 投票+约束的增益 |
| E7 | PVDR v4 (logistic+CP) vs PVDR v3 (3-sigma) | 增强检测的增益 |
| E8 | AKG v4 (threshold) vs AKG v3 (fixed sigmoid) | 阈值门控的增益 |
| E9 | L_expert enabled vs disabled | 专家修正损失的作用 |

---

**最后更新**: 2026-02-24
**状态**: 架构设计完成，待代码实现
**生成工具**: Claude Code (Opus 4.6)
