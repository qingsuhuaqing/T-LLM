# Time-LLM Enhanced v4 技术文档

> **Time Series Forecasting by Reprogramming Large Language Models**
>
> 通过重编程大语言模型实现时间序列预测 | ICLR 2024
>
> **Enhanced v4**: 多相多尺度预测 + 增强检索融合

---

## 目录

- [一、项目概述](#一项目概述)
- [二、v4 创新架构](#二v4-创新架构)
- [三、v4 模块详解](#三v4-模块详解)
- [四、数据流与维度变化](#四数据流与维度变化)
- [五、命令行参数详解](#五命令行参数详解)
- [六、训练脚本与消融实验](#六训练脚本与消融实验)
- [七、训练过程与指标](#七训练过程与指标)
- [八、断点续训](#八断点续训)
- [九、常见问题与解决方案](#九常见问题与解决方案)
- [十、参考资料](#十参考资料)

---

## 一、项目概述

### 1.1 项目目标

Time-LLM 是一个将**冻结的大语言模型 (LLM)** 应用于**时间序列预测**的框架。核心思想：

- **冻结 LLM 参数**：不修改 LLM 的预训练权重
- **重编程输入**：将时间序列数据"翻译"成 LLM 能理解的形式
- **利用 LLM 能力**：借助 LLM 的序列建模能力进行预测

### 1.2 v4 创新点

在原始 Time-LLM 基础上，v4 新增两大创新模块：

| 模块 | 全称 | 核心能力 |
|------|------|----------|
| **TAPR** | Trend-Aware Patch Router | 多相多尺度预测 + 专家投票融合 |
| **GRAM** | Global Retrieval-Augmented Memory | 增强检测检索 + 阈值自适应门控 |

### 1.3 版本演进

| 版本 | DM (下采样) | C2F (融合) | PVDR (检测) | AKG (门控) |
|------|-----------|-----------|------------|-----------|
| v3 | avg_pool(k不同) | softmax(W)加权 | 3-sigma规则 | 固定sigmoid(C) |
| **v4** | **多相x[:,p::k,:] (统一k=4)** | **MAD投票 + DWT趋势约束** | **逻辑回归 + 贝叶斯变点** | **PVDR信号阈值自适应** |

**v4 关键改进**:
- DM: 数据100%保留 (多相分解无信息损失，替代有损的平均池化)
- C2F: 两阶段融合 (尺度内异常分支剔除 + 跨尺度趋势一致性约束)
- PVDR: 可学习检测 (6参数逻辑回归替代固定3-sigma规则 + 3参数贝叶斯变点)
- AKG: 动态门控 (基于极端/变点/置信度信号自适应调整门控强度)

---

## 二、v4 创新架构

### 2.1 整体流程图

```
输入时序 [B, T, N]
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ [1] Baseline Time-LLM (冻结LLM前向)                         │
│     Normalize → Patching → Reprogramming → LLM → FlattenHead│
│     → pred_s1 [B, pred_len, N]                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────────┐
    ▼                       ▼                           ▼
┌────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ DM v4      │    │ PVDR v4          │    │                  │
│ 多相多尺度 │    │ 增强双重检索     │    │ x_normed         │
│            │    │                  │    │ (归一化输入)     │
│ S2: k=2    │    │ 逻辑极端分类器   │    │                  │
│   diff变换 │    │ 贝叶斯变点检测   │    │                  │
│   2分支    │    │ 4模式检索        │    │                  │
│ S3: k=4    │    │ → hist_refs      │    │                  │
│   恒等变换 │    │ → pvdr_signals   │    │                  │
│   4分支    │    └────────┬─────────┘    │                  │
│ S4: k=4    │             │              │                  │
│   平滑变换 │             │              │                  │
│   4分支    │             │              │                  │
│ S5: k=4    │             ▼              │                  │
│   趋势变换 │    ┌──────────────────┐    │                  │
│   4分支    │    │ AKG v4           │    │                  │
│            │    │ 阈值自适应门控   │    │                  │
│ 共14分支   │    │                  │    │                  │
└──────┬─────┘    │ effective_gate = │    │                  │
       │          │   sigmoid(C)     │    │                  │
       ▼          │   × (1-ext_adj)  │    │                  │
┌────────────┐    │   × (1-cp_adj)   │    │                  │
│ C2F v4     │    │   × conf_adj     │    │                  │
│ 专家投票   │    │                  │    │                  │
│            │    │ Fusion Matrix F  │    │                  │
│ Stage1:    │    │ → final_pred     │    │                  │
│  MAD投票   │    └────────┬─────────┘    │                  │
│  → 5个     │             │              │                  │
│    consolidated          │              │                  │
│            │             ▼              │                  │
│ Stage2:    │    ┌──────────────────┐    │                  │
│  DWT趋势  │    │ Denormalize      │    │                  │
│  → weighted│    │ 反归一化         │    │                  │
│    pred    │    │ → 最终输出       │    │                  │
└──────┬─────┘    └──────────────────┘    │                  │
       │                   ▲              │                  │
       └───────────────────┘──────────────┘                  │
```

### 2.2 五类可训练矩阵/参数

| 编号 | 名称 | 形状 | 参数量 | 所在模块 |
|------|------|------|--------|----------|
| 1 | 参数矩阵 Theta_s | 4个ScaleEncoder | ~100K | DM v4 |
| 2 | 权重矩阵 W | [5, 96] | 480 | C2F v4 |
| 3 | 联系矩阵 C | [5, 96] | 480 | AKG v4 |
| 4 | 融合矩阵 F | [10, 96] | 960 | AKG v4 |
| 5 | 阈值参数 tau | 12个 | 12 | AKG(3) + LogisticReg(6) + BayesianCP(3) |

### 2.3 损失函数

```
L_total = L_main + warmup × (λ_consist × L_consist + λ_gate × L_gate + λ_expert × L_expert)
```

| 损失项 | 公式 | 说明 |
|--------|------|------|
| L_main | MSE(pred, true) | 主预测损失 |
| L_consist | mean(max(0, -cos_sim(approx_s, approx_S5))) | DWT小波趋势一致性 |
| L_gate | -mean(\|sigmoid(C) - 0.5\|) | 门控决断性正则化 |
| L_expert | mean(correction_rate_s) | 专家修正率损失 (v4新增) |

---

## 三、v4 模块详解

### 3.1 DM v4: PolyphaseMultiScale (多相多尺度预测)

**代码位置**: `layers/TAPR_v4.py:185-328`

#### 核心思想

用多相分解 `x[:, p::k, :]` 替代平均池化下采样，**数据100%保留**，无信息损失。

#### 尺度配置

```python
SCALE_CFG = [
    (2, 'diff'),       # S2: k=2, 差分变换, 2个分支
    (4, 'identity'),   # S3: k=4, 恒等变换, 4个分支
    (4, 'smooth'),     # S4: k=4, 平滑变换, 4个分支
    (4, 'trend'),      # S5: k=4, 趋势变换, 4个分支
]
# S1 (baseline) 由 Time-LLM 主模型处理
# 总分支数: 2 + 4 + 4 + 4 = 14
```

#### 前向流程

```
输入 x_normed [B, T, N]
    │
    ├─→ S2 (k=2):
    │    ├─ phase 0: x[:,0::2,:] → diff → patches → encoder → pred [B, 96, N]
    │    └─ phase 1: x[:,1::2,:] → diff → patches → encoder → pred [B, 96, N]
    │
    ├─→ S3 (k=4):
    │    ├─ phase 0: x[:,0::4,:] → identity → patches → encoder → pred
    │    ├─ phase 1: x[:,1::4,:] → identity → patches → encoder → pred
    │    ├─ phase 2: x[:,2::4,:] → identity → patches → encoder → pred
    │    └─ phase 3: x[:,3::4,:] → identity → patches → encoder → pred
    │
    ├─→ S4 (k=4): smooth变换, 同S3结构
    └─→ S5 (k=4): trend变换, 同S3结构
```

**关键**: 同一尺度内的所有分支共享 ScaleEncoder (仅4个 ScaleEncoder，分支内参数共享)。

#### 输出格式

```python
# 返回
pred_s1,           # [B, pred_len, N] — 原封不动传回baseline预测
branch_preds_dict  # {0: [pred_p0, pred_p1],           # S2的2个分支预测
                   #  1: [pred_p0, ..., pred_p3],       # S3的4个分支预测
                   #  2: [pred_p0, ..., pred_p3],       # S4的4个分支预测
                   #  3: [pred_p0, ..., pred_p3]}       # S5的4个分支预测
```

---

### 3.2 C2F v4: ExpertVotingFusion (专家投票融合)

**代码位置**: `layers/TAPR_v4.py:335-581`

#### Stage 1: 尺度内 MAD 投票

```
每个尺度的 k 个分支预测:
    │
    ├─ 计算中位数预测 (参考基准)
    ├─ 计算每分支偏差 (MAD = Median Absolute Deviation)
    ├─ 偏差 > mad_threshold × MAD → 标记为异常分支
    ├─ 平均非异常分支 → consolidated_pred
    └─ 记录修正率 = 被剔除分支数 / 总分支数
```

**特殊处理**: S2 (k=2) 跳过 MAD (仅2个分支，MAD退化)，直接取均值。

#### Stage 2: 跨尺度 DWT 趋势约束

```
5个 consolidated_preds (S1..S5):
    │
    ├─ Haar DWT 分解每个预测 → approx (趋势) + detail (细节)
    ├─ S5 的 approx 作为参考趋势方向
    ├─ 计算每尺度趋势与 S5 的 cosine_similarity
    │
    ├─ 趋势惩罚: cos_sim < -0.3 → 权重 × decay_factor
    ├─ 可靠性惩罚: correction_rate > threshold → 权重 × reliability_penalty
    │
    └─ softmax(W) × 惩罚 → 重归一化 → 加权求和 → weighted_pred
```

#### Haar DWT 小波分解

```python
def dwt_haar_1d(x):  # x: [B, T, N]
    approx = (x[:, 0::2, :] + x[:, 1::2, :]) / sqrt(2)  # 低频近似
    detail = (x[:, 0::2, :] - x[:, 1::2, :]) / sqrt(2)  # 高频细节
    return approx, detail  # 各 [B, T//2, N]
```

---

### 3.3 PVDR v4: EnhancedDualRetriever (增强双重检索)

**代码位置**: `layers/GRAM_v4.py:347-557`

#### 三个检测/编码组件

| 组件 | 类名 | 参数量 | 功能 |
|------|------|--------|------|
| 极端分类器 | `LogisticExtremeClassifier` | 6 | 可学习极端数据检测 (Linear(5,1) + sigmoid) |
| 变点检测器 | `BayesianChangePointDetector` | 3 | 递归二分贝叶斯变点检测 (3个先验超参) |
| 模式编码器 | `LightweightPatternEncoder` | ~330 | 5统计特征 → d_repr维向量 (Linear(5,64) + LayerNorm) |

#### LogisticExtremeClassifier

```python
# 输入: x [B, T, N]
# 提取5个统计特征: [mean, std, max, min, trend]
# → Linear(5, 1) → sigmoid → extreme_prob [B]
```

替代 v3 的 3-sigma 固定规则，变为可学习的极端数据判别器。

#### BayesianChangePointDetector

```python
# 递归二分检测变点位置
# 对每个候选分割点计算 Bayesian Factor (两段模型 vs 单段模型的似然比)
# 使用 softmax 加权 (可微分) 替代硬 argmax
# 关注序列末端 (T × 0.8 之后) 的变点 → near_end_score [B]
```

3个可学习先验: `prior_mean`, `prior_var`, `noise_var` (后两者通过 softplus 保证正值)。

#### 4 模式检索

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| A: Direct | max_sim > 0.95 | Top-1, 无阈值过滤 (极相似直接用) |
| B: Extreme | extreme_prob > 0.5 | Top-K×2, 降低检索阈值 |
| C: Turning Point | near_end_score > 0.5 | 标准 Top-K, 优先转折点标签 |
| D: Normal | 默认 | 标准 Top-K, 标准阈值 |

#### 增强记忆库 (EnhancedMultiScaleMemoryBank)

- 每尺度存储 (key, value, label) 三元组
- label: 0=normal, 1=extreme, 2=turning_point
- `build_with_labels()`: 训练时自动标注所有样本
- `get_subset_by_label()`: 按标签筛选子集用于模式C检索

---

### 3.4 AKG v4: ThresholdAdaptiveGating (阈值自适应门控)

**代码位置**: `layers/GRAM_v4.py:564-691`

#### 门控公式

```
base_gate = sigmoid(C)                        # [n_scales, pred_len]

extreme_adj = sigmoid(10 × (extreme_score - τ_ext))     # [B]
cp_adj      = sigmoid(10 × (cp_score - τ_cp))           # [B]
conf_adj    = sigmoid(10 × (confidence - τ_conf))        # [B, n_scales]

effective_gate = base_gate × (1 - extreme_adj) × (1 - cp_adj) × conf_adj
                                                          # [B, n_scales, pred_len]
```

| 信号 | 含义 | 对门控的影响 |
|------|------|-------------|
| extreme_adj 高 | 当前数据为极端值 | 降低门控 → 更信任历史参考 |
| cp_adj 高 | 序列末端有变点 | 降低门控 → 更信任历史参考 |
| conf_adj 高 | 检索置信度高 | 允许更多历史参考 |

3个可学习阈值: `extreme_threshold`, `cp_threshold`, `conf_threshold` (通过 sigmoid 归一化)。

#### 两阶段融合

```
Stage 1: 逐尺度门控
    connected_s = gate × scale_pred + (1-gate) × hist_ref    # 每尺度

Stage 2: 融合矩阵
    all_sources = [scale_preds..., connected...]               # 2×n_scales 个来源
    → softmax(F) 加权求和 → final_pred
```

---

## 四、数据流与维度变化

### 4.1 Baseline Time-LLM 维度变化

**配置**: batch=32, seq_len=512, pred_len=96, N_vars=7, d_model=64, d_ff=128, llm_dim=768 (GPT-2)

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 | `[32, 512, 7]` | B, seq_len, N_vars |
| 归一化 | `[32, 512, 7]` | instance norm |
| 转置 | `[32, 7, 512]` | → PatchEmbedding |
| Patching | `[224, 64, 64]` | B*N=224, num_patches, d_model |
| Reprogramming | `[224, 64, 768]` | → llm_dim |
| + Prompt | `[224, ~192, 768]` | prompt_len + num_patches |
| LLM输出 | `[224, ~192, 768]` | GPT-2 forward |
| 截取d_ff | `[224, ~192, 128]` | 取前d_ff维 |
| Reshape | `[32, 7, 128, ~192]` | B, N, d_ff, total_len |
| FlattenHead | `[32, 7, 96]` | → pred_len |
| pred_s1 | `[32, 96, 7]` | B, pred_len, N |

### 4.2 DM v4 分支维度变化 (以 S3, k=4 为例)

| 步骤 | 形状 | 说明 |
|------|------|------|
| 多相提取 | `[32, 128, 7]` | x[:,0::4,:], T/k=128 |
| 信号变换 | `[32, 128, 7]` | identity (不变) |
| Channel-independent | `[224, 128, 1]` | B*N |
| Patching | `[224, 16, 16]` | num_patches, patch_len |
| PatchEmbedding | `[224, 16, 64]` | → d_model |
| Interpolate | `[224, 32, 64]` | 对齐到 expected_patches |
| ScaleEncoder | `[224, 96]` | → pred_len |
| Reshape | `[32, 96, 7]` | B, pred_len, N |

### 4.3 C2F v4 维度变化

| 步骤 | 形状 | 说明 |
|------|------|------|
| 输入: 14个分支 | 14 × `[32, 96, 7]` | branch_preds_dict |
| Stage1 投票后 | 5 × `[32, 96, 7]` | consolidated (S1..S5) |
| DWT分解 | 5 × `[32, 48, 7]` | approx (半长) |
| softmax(W) | `[5, 96]` | 权重矩阵 |
| 加权求和 | `[32, 96, 7]` | weighted_pred |

### 4.4 PVDR + AKG v4 维度变化

| 步骤 | 形状 | 说明 |
|------|------|------|
| extreme_prob | `[32]` | 极端概率 |
| near_end_score | `[32]` | 变点得分 |
| hist_refs | 5 × `[32, 96, 7]` | 历史参考 |
| retrieval_confidence | `[32, 5]` | 检索置信度 |
| effective_gate | `[32, 5, 96]` | 每样本门控 |
| connected | 5 × `[32, 96, 7]` | 门控混合结果 |
| all_sources | 10 × `[32, 96, 7]` | scale_preds + connected |
| fusion_matrix F | `[10, 96]` | 融合权重 |
| final_pred | `[32, 96, 7]` | 最终预测 |

---

## 五、命令行参数详解

### 5.1 基础配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task_name` | str | `long_term_forecast` | 任务类型 |
| `--is_training` | int | 1 | 1=训练, 0=测试 |
| `--model_id` | str | - | 实验标识符 |
| `--model_comment` | str | - | 模型备注 |
| `--model` | str | `TimeLLM` | 模型名称 |
| `--seed` | int | 2021 | 随机种子 |

### 5.2 数据配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | `ETTm1` | 数据集名称 |
| `--root_path` | str | `./dataset` | 数据根目录 |
| `--data_path` | str | `ETTh1.csv` | 数据文件名 |
| `--features` | str | `M` | M=多变量, S=单变量 |
| `--seq_len` | int | 96 | 输入序列长度 (v4推荐512) |
| `--label_len` | int | 48 | 解码器起始token长度 |
| `--pred_len` | int | 96 | 预测长度 |

### 5.3 模型结构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enc_in` | int | 7 | 变量数量 |
| `--d_model` | int | 16 | Patch嵌入维度 (v4推荐64) |
| `--n_heads` | int | 8 | 注意力头数 |
| `--d_ff` | int | 32 | FFN维度 (v4推荐128) |
| `--patch_len` | int | 16 | Patch长度 |
| `--stride` | int | 8 | Patch步长 |
| `--dropout` | float | 0.1 | Dropout比例 |

### 5.4 LLM 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--llm_model` | str | `LLAMA` | LLM类型 (GPT2/LLAMA/BERT) |
| `--llm_dim` | int | 4096 | LLM隐藏维度 (GPT-2=768) |
| `--llm_layers` | int | 6 | 使用的LLM层数 |
| `--llm_model_path` | str | `''` | 本地模型路径 |
| `--load_in_4bit` | flag | False | 4-bit量化 |

**LLM 维度对照表:**

| LLM 模型 | llm_dim | 参数量 | 显存 |
|----------|---------|--------|------|
| GPT-2 | 768 | 124M | ~500 MB |
| Qwen 2.5 3B (4-bit) | 2048 | 3B | ~1.5 GB |
| LLaMA-7B | 4096 | 7B | ~14 GB |

### 5.5 TAPR 参数 (DM + C2F)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_tapr` | flag | False | 启用 DM + C2F |
| `--n_scales` | int | 5 | 总尺度数 (含baseline) |
| `--downsample_rates` | str | `1,2,4,4,4` | 各尺度下采样率 (须与DM的k值对齐) |
| `--lambda_consist` | float | 0.1 | DWT一致性损失权重 |
| `--decay_factor` | float | 0.5 | 推理时不一致尺度降权因子 |

### 5.6 GRAM 参数 (PVDR + AKG)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_gram` | flag | False | 启用 PVDR + AKG |
| `--lambda_gate` | float | 0.01 | 门控正则化损失权重 |
| `--similarity_threshold` | float | 0.8 | 基础检索阈值 |
| `--extreme_threshold_reduction` | float | 0.2 | 极端数据阈值降低量 |
| `--top_k` | int | 5 | 检索相似模式数 |
| `--d_repr` | int | 64 | 模式表示向量维度 |
| `--build_memory` | flag | False | 构建检索记忆库 (首次须开) |

### 5.7 v4 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mad_threshold` | float | 3.0 | MAD异常检测倍数 |
| `--reliability_threshold` | float | 0.3 | 修正率不可靠判定阈值 |
| `--reliability_penalty` | float | 0.3 | 不可靠尺度降权因子 |
| `--lambda_expert` | float | 0.05 | 专家修正损失权重 |
| `--cp_min_segment` | int | 16 | 贝叶斯变点最小段长 |
| `--cp_max_depth` | int | 4 | 贝叶斯变点最大递归深度 |

### 5.8 训练配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_epochs` | int | 10 | 训练轮数 (v4推荐50) |
| `--batch_size` | int | 32 | 批大小 |
| `--patience` | int | 10 | Early Stopping耐心值 |
| `--learning_rate` | float | 0.0001 | 学习率 |
| `--warmup_steps` | int | 500 | 辅助损失warmup步数 |
| `--save_steps` | int | 0 | 每N步保存checkpoint (0=关闭) |
| `--save_total_limit` | int | 0 | 最多保留N个step checkpoint |

---

## 六、训练脚本与消融实验

### 6.1 v4 训练脚本

**文件**: `scripts/TimeLLM_ETTh1_enhanced_v4.sh`

```bash
# 完整v4 (TAPR + GRAM)
bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
```

### 6.2 消融实验指南

| 实验 | 命令 | 目的 |
|------|------|------|
| E1: Baseline | `USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh` | 原始Time-LLM |
| E2: +TAPR v4 | `USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh` | 仅多相+投票 |
| E3: +GRAM v4 | `USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh` | 仅增强检索 |
| E4: Full v4 | `USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh` | 完整v4 |
| E5: v4 vs v3 DM | 分别运行v4和v3脚本, 仅启用TAPR | 多相vs平均池化 |
| E7: v4 vs v3 PVDR | 分别运行v4和v3脚本, 仅启用GRAM | 逻辑回归vs3-sigma |
| E9: L_expert | `LAMBDA_EXPERT=0/0.05` 对比 | 专家损失增益 |

### 6.3 参数敏感性实验

| 参数 | 测试值 |
|------|--------|
| MAD_THRESHOLD | 2.0, 3.0, 4.0 |
| RELIABILITY_THRESHOLD | 0.2, 0.3, 0.5 |
| LAMBDA_EXPERT | 0.01, 0.05, 0.1 |
| CP_MIN_SEGMENT | 8, 16, 32 |
| CP_MAX_DEPTH | 2, 4, 6 |

---

## 七、训练过程与指标

### 7.1 训练输出示例

```
======================================================================
Time-LLM Enhanced v4 — Polyphase Multi-scale + Enhanced Retrieval
======================================================================
  TAPR v4 (DM Polyphase + C2F Voting): ENABLED
    n_scales: 5
    downsample_rates: 1,2,4,4,4
    mad_threshold: 3.0
    reliability_threshold: 0.3
    lambda_expert: 0.05
  GRAM v4 (PVDR Enhanced + AKG Threshold): ENABLED
    top_k: 5
    cp_min_segment: 16
    cp_max_depth: 4
======================================================================

[v4] Building PVDR memory banks (with labels) for all scales...
[v4] Memory banks built successfully

Epoch 1 | Train Loss: 0.4523 Vali Loss: 0.3912
  Aux: consist=0.0234 gate=-0.0912 expert=0.1250 warmup=0.25
```

### 7.2 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **MSE** | mean((pred - true)^2) | 均方误差 (主要指标) |
| **MAE** | mean(\|pred - true\|) | 平均绝对误差 |
| **RMSE** | sqrt(MSE) | 均方根误差 |

---

## 八、断点续训

### 8.1 可修改参数 (不影响模型形状)

```
BATCH, NUM_WORKERS, LEARNING_RATE, TRAIN_EPOCHS, PATIENCE
LAMBDA_CONSIST, LAMBDA_GATE, LAMBDA_EXPERT
DECAY_FACTOR, MAD_THRESHOLD, RELIABILITY_THRESHOLD, RELIABILITY_PENALTY
WARMUP_STEPS, TEST_EPOCHS, SAVE_STEPS, SAVE_TOTAL_LIMIT
```

### 8.2 不可修改参数 (会导致形状不匹配)

```
LLM_PATH, LLM_DIM, LLM_LAYERS, --llm_model
SEQ_LEN, LABEL_LEN, PRED_LEN
D_MODEL, D_FF, N_HEADS, ENC_IN, DEC_IN, C_OUT
USE_TAPR, USE_GRAM, N_SCALES, D_REPR, DOWNSAMPLE_RATES
CP_MIN_SEGMENT, CP_MAX_DEPTH
```

### 8.3 注意事项

- `BUILD_MEMORY` 续训时**必须设为 0** (记忆库已在checkpoint中)
- v3 和 v4 checkpoint **不兼容** (模块接口变更)
- 设置 `RESUME_FROM` 指向 checkpoint 路径
- 设置 `RESUME_COUNTER` 手动覆盖 EarlyStopping 计数 (-1=不覆盖)

### 8.4 推理加载

```python
ckpt = torch.load('checkpoints/.../checkpoint')
model.load_state_dict(ckpt['model'])
model.eval()
```

---

## 九、常见问题与解决方案

### 9.1 OOM 处理优先级

| 优先级 | 操作 | 范围 |
|--------|------|------|
| 1 | 降 BATCH | 32 → 16 → 8 → 4 |
| 2 | 降 SEQ_LEN | 512 → 384 → 256 |
| 3 | 降 D_FF | 128 → 64 → 32 |
| 4 | 降 D_MODEL | 64 → 32 |
| 5 | 降 N_SCALES | 5 → 3 |
| 6 | 降 TOP_K | 5 → 3 |
| 7 | 禁用 GRAM | USE_GRAM=0 |
| 8 | 降 LLM_LAYERS | 12 → 6 → 4 |

### 9.2 常见问题

| 问题 | 解决方案 |
|------|----------|
| `CUDA_HOME does not exist` | `pip uninstall deepspeed -y` |
| dtype不匹配 (bfloat16/float32) | 模型代码中已修复 (`.float()`) |
| v3 checkpoint加载v4 | 不兼容，须重新训练 |
| Shell脚本语法错误 | `sed -i 's/\r$//' scripts/*.sh` (CRLF修复) |
| DOWNSAMPLE_RATES错误 | 必须是 `1,2,4,4,4` (与DM多相k值对齐) |

### 9.3 显存估算 (GPT-2 + v4, batch=32)

```
GPT-2 (FP32, 12层):               ~0.5 GB
可训练参数:                        ~0.3 GB
DM v4 (4 ScaleEncoders):          ~2 MB
C2F v4 (weight_matrix + DWT):     < 1 KB
PVDR v4 (memory + detectors):     ~5.3 MB
AKG v4 (matrices + thresholds):   < 1 KB
中间张量:                          ~0.8 GB
系统开销:                          ~1.0 GB
总计:                              ~2.6 GB
```

---

## 十、参考资料

### 10.1 论文

- **Time-LLM**: Time Series Forecasting by Reprogramming Large Language Models (ICLR 2024)
- **TimeMixer**: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024)
- **N-HiTS**: Neural Hierarchical Interpolation for Time Series Forecasting (AAAI 2023)
- **PatchTST**: A Time Series is Worth 64 Words (ICLR 2023)
- **Autoformer**: Decomposition Transformers with Auto-Correlation (NeurIPS 2021)
- **TFT**: Temporal Fusion Transformers (IJF 2021)
- **BOCPD**: Bayesian Online Changepoint Detection (Adams & MacKay, 2007)

### 10.2 项目文档

| 文档 | 内容 |
|------|------|
| `CLAUDE.md` | Claude Code 快速参考指南 |
| `README.md` | 本文档 (完整技术文档) |
| `claude_v4_readme.md` | v4 架构设计文档 (权威规格说明) |
| `md/work2.md` | 深度技术解析 (baseline) |
| `md/wenti.md` | 故障排除汇总 |
| `md/chuangxin.md` | 创新方案概述 |

### 10.3 代码入口

| 文件 | 用途 |
|------|------|
| `run_main_enhanced_v4.py` | **v4 训练入口** (当前使用) |
| `scripts/TimeLLM_ETTh1_enhanced_v4.sh` | **v4 训练脚本** (当前使用) |
| `run_main.py` | 原始 Time-LLM 训练入口 |

---

**文档版本**: v4.0
**最后更新**: 2026-02-24
**当前版本**: Enhanced v4 (PolyphaseMultiScale + ExpertVotingFusion + EnhancedDualRetriever + ThresholdAdaptiveGating)
**适用环境**: GPT-2 (本地) / Qwen 2.5 3B (4-bit) + CUDA GPU
