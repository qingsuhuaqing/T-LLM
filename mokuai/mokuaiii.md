# Time-LLM 创新模块扩展方案

> **基于大模型语义融合的时间序列预测模型 —— 新增创新模块设计文档**
>
> **作者**: 王振达
> **日期**: 2026年2月
> **版本**: v1.0

---

## 目录

1. [研究背景与动机](#一研究背景与动机)
2. [模块一：渐进式粒度预测模块（Progressive Granularity Prediction Module, PGPM）](#二模块一渐进式粒度预测模块progressive-granularity-prediction-module-pgpm)
3. [模块二：检索增强的历史模式匹配模块（Retrieval-Augmented Pattern Matching Module, RAPM）](#三模块二检索增强的历史模式匹配模块retrieval-augmented-pattern-matching-module-rapm)
4. [模块集成与整体架构](#四模块集成与整体架构)
5. [实验设计与预期效果](#五实验设计与预期效果)
6. [参考文献](#六参考文献)

---

## 一、研究背景与动机

### 1.1 现有框架局限性分析

基于开题报告中对Time-LLM框架的技术路线分析，当前基线模型虽已实现了时序数据与自然语言模态的有效对齐，但仍存在以下可改进空间：

| 局限性 | 具体表现 | 影响 |
|--------|----------|------|
| **单一预测粒度** | 直接从Patch表示映射到最终预测值 | 缺乏从宏观趋势到微观细节的渐进式推理过程 |
| **历史模式利用不足** | 仅依赖当前输入窗口的统计特征 | 未能充分挖掘训练数据中相似历史模式的参考价值 |
| **预测过程可解释性有限** | 端到端直接输出 | 难以追溯预测决策的形成过程 |

### 1.2 创新动机

**核心思想来源**：结合用户提出的"先学习判断是否上涨还是下降，再进行细分，考虑上涨或下降的比例"以及"通过相似的历史数据逐步缩小范围"的创新思路，设计两个相互补充的模块：

1. **渐进式粒度预测模块（PGPM）**：实现从粗粒度趋势方向判断到细粒度幅度量化的层级式预测范式
2. **检索增强的历史模式匹配模块（RAPM）**：通过检索历史相似模式提供预测参考，增强模型的模式泛化能力

### 1.3 文献支撑

本方案设计基于以下前沿研究成果：

- **多尺度混合架构**：SST (CIKM 2025) 证明了长短程模式分离策略的有效性 [1]
- **检索增强预测**：RAFT (ICML 2025) 验证了历史模式检索对预测精度的显著提升 [2]
- **粗到细预测范式**：mr-Diff (ICLR 2024) 展示了多分辨率渐进预测的优越性 [3]
- **对比学习表征**：TimesURL (AAAI 2024) 提供了通用时序表示学习的框架 [4]
- **多尺度分解混合**：TimeMixer (ICLR 2024) 证实了多尺度信息解耦的有效性 [5]

---

## 二、模块一：渐进式粒度预测模块（Progressive Granularity Prediction Module, PGPM）

### 2.1 模块概述

#### 2.1.1 解决的问题

传统时序预测模型采用单步映射策略，直接从输入表示预测目标值，这种方式存在以下问题：

1. **预测跨度过大**：从高维特征空间一步跳跃到精确数值，中间缺乏过渡
2. **误差累积风险**：细节预测依赖于整体把握，缺乏层级约束
3. **可解释性不足**：无法追溯"先判断趋势、再量化幅度"的人类认知过程

#### 2.1.2 增加的目的

PGPM模块旨在模拟人类专家的预测思维过程：

```
人类专家预测思路：
"首先判断整体是上涨还是下降" → "再估计涨跌的大致幅度区间" → "最后精确量化具体数值"
```

通过引入层级式粒度预测机制，实现：
- **趋势方向判断**（粗粒度）：先确定预测的整体方向性
- **变化幅度估计**（中粒度）：量化变化的程度范围
- **精确数值预测**（细粒度）：在前两层约束下输出最终值

#### 2.1.3 模块原理

**理论基础**：

本模块借鉴了以下研究的核心思想：

1. **SST的长短程分解策略** [1]：将时序分解为长程模式（宏观趋势）和短程变化（局部细节），证明了Mamba擅长捕捉长期结构而Transformer更适合短期动态

2. **mr-Diff的多分辨率生成** [3]：从粗粒度到细粒度逐步生成预测，"先生成粗糙趋势，再逐步添加细节信息"

3. **C2FAR的分箱预测** [6]：首先分类预测值所在的粗粒度区间，然后在该区间内细化

**创新融合**：

PGPM将上述思想与Time-LLM的语义融合框架结合，设计三阶段渐进预测：

```
Stage 1: Direction Classification (方向分类)
         ↓ 约束传递
Stage 2: Magnitude Regression (幅度回归)
         ↓ 约束传递
Stage 3: Precise Prediction (精确预测)
```

### 2.2 架构设计

#### 2.2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Progressive Granularity Prediction Module         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LLM Output Embeddings                                              │
│  [B*N, L, d_ff]                                                     │
│         │                                                            │
│         ├─────────────────┬─────────────────┬────────────────┐      │
│         ▼                 ▼                 ▼                │      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │      │
│  │   Coarse     │  │   Medium     │  │    Fine      │       │      │
│  │   Encoder    │  │   Encoder    │  │   Encoder    │       │      │
│  │  (Trend)     │  │ (Magnitude)  │  │  (Precise)   │       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │      │
│         │                 │                 │                │      │
│         ▼                 ▼                 ▼                │      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │      │
│  │  Direction   │  │  Magnitude   │  │   Value      │       │      │
│  │  Classifier  │  │  Estimator   │  │  Regressor   │       │      │
│  │  (↑/↓/→)     │  │  (幅度区间)   │  │  (精确值)    │       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │      │
│         │                 │                 │                │      │
│         │    ┌────────────┘                 │                │      │
│         │    │                              │                │      │
│         ▼    ▼                              ▼                │      │
│  ┌────────────────┐                  ┌──────────────┐       │      │
│  │   Constraint   │─────────────────▶│   Gated      │       │      │
│  │   Propagation  │                  │   Fusion     │       │      │
│  └────────────────┘                  └──────┬───────┘       │      │
│                                             │                │      │
│                                             ▼                │      │
│                                     Final Prediction         │      │
│                                     [B, pred_len, N]         │      │
│                                                              │      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.2.2 各组件详细设计

**A. 粗粒度趋势编码器（Coarse Trend Encoder）**

```python
class CoarseTrendEncoder(nn.Module):
    """
    粗粒度趋势编码器

    功能：从LLM输出中提取全局趋势特征，用于方向判断
    设计理念：借鉴SST中Mamba擅长长程模式的发现
    """
    def __init__(self, d_input, d_hidden=256, n_directions=3):
        super().__init__()
        self.n_directions = n_directions  # 上涨、下跌、横盘

        # 全局池化 + MLP
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 趋势特征提取
        self.trend_extractor = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU()
        )

        # 方向分类器
        self.direction_classifier = nn.Linear(d_hidden // 2, n_directions)

        # 方向嵌入（用于约束传递）
        self.direction_embedding = nn.Embedding(n_directions, d_hidden // 2)

    def forward(self, x):
        """
        Args:
            x: [B*N, L, d_ff] - LLM输出表示

        Returns:
            direction_logits: [B*N, n_directions] - 方向分类logits
            direction_embed: [B*N, d_hidden//2] - 方向嵌入
            trend_features: [B*N, d_hidden//2] - 趋势特征
        """
        # 全局池化
        x_pooled = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)  # [B*N, d_ff]

        # 趋势特征
        trend_features = self.trend_extractor(x_pooled)  # [B*N, d_hidden//2]

        # 方向分类
        direction_logits = self.direction_classifier(trend_features)

        # 软方向嵌入（训练时使用soft label，推理时使用argmax）
        if self.training:
            direction_probs = F.softmax(direction_logits, dim=-1)
            direction_embed = torch.einsum('bd,de->be',
                                          direction_probs,
                                          self.direction_embedding.weight)
        else:
            direction_idx = direction_logits.argmax(dim=-1)
            direction_embed = self.direction_embedding(direction_idx)

        return direction_logits, direction_embed, trend_features
```

**B. 中粒度幅度估计器（Medium Magnitude Estimator）**

```python
class MediumMagnitudeEstimator(nn.Module):
    """
    中粒度幅度估计器

    功能：在趋势方向约束下，估计变化幅度的区间范围
    设计理念：借鉴C2FAR的分箱策略，先定位区间再细化
    """
    def __init__(self, d_input, d_direction, d_hidden=256, n_bins=10):
        super().__init__()
        self.n_bins = n_bins

        # 方向条件融合
        self.direction_fusion = nn.Sequential(
            nn.Linear(d_input + d_direction, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU()
        )

        # 局部模式提取（借鉴Transformer捕捉短期动态的优势）
        self.local_attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 幅度区间预测
        self.magnitude_predictor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, n_bins)
        )

        # 区间边界参数（可学习）
        self.bin_boundaries = nn.Parameter(
            torch.linspace(-0.5, 0.5, n_bins + 1)
        )

        # 幅度嵌入（用于约束传递）
        self.magnitude_embedding = nn.Embedding(n_bins, d_hidden // 2)

    def forward(self, x, direction_embed, trend_features):
        """
        Args:
            x: [B*N, L, d_ff] - LLM输出表示
            direction_embed: [B*N, d_direction] - 方向嵌入
            trend_features: [B*N, d_trend] - 趋势特征

        Returns:
            magnitude_logits: [B*N, n_bins] - 幅度区间logits
            magnitude_embed: [B*N, d_hidden//2] - 幅度嵌入
            magnitude_value: [B*N, 1] - 连续幅度估计值
        """
        B_N, L, D = x.shape

        # 方向条件融合
        direction_expanded = direction_embed.unsqueeze(1).expand(-1, L, -1)
        x_conditioned = torch.cat([x, direction_expanded], dim=-1)
        x_fused = self.direction_fusion(x_conditioned)  # [B*N, L, d_hidden]

        # 局部注意力
        attn_out, _ = self.local_attention(x_fused, x_fused, x_fused)

        # 池化得到幅度特征
        magnitude_features = attn_out.mean(dim=1)  # [B*N, d_hidden]

        # 幅度区间分类
        magnitude_logits = self.magnitude_predictor(magnitude_features)

        # 软幅度嵌入
        if self.training:
            magnitude_probs = F.softmax(magnitude_logits, dim=-1)
            magnitude_embed = torch.einsum('bb,be->be',
                                          magnitude_probs,
                                          self.magnitude_embedding.weight)
        else:
            magnitude_idx = magnitude_logits.argmax(dim=-1)
            magnitude_embed = self.magnitude_embedding(magnitude_idx)

        # 连续幅度值（区间中心的加权平均）
        bin_centers = (self.bin_boundaries[:-1] + self.bin_boundaries[1:]) / 2
        magnitude_probs = F.softmax(magnitude_logits, dim=-1)
        magnitude_value = torch.einsum('bb,b->b', magnitude_probs, bin_centers)

        return magnitude_logits, magnitude_embed, magnitude_value.unsqueeze(-1)
```

**C. 细粒度精确预测器（Fine Precise Predictor）**

```python
class FinePrecisePredictor(nn.Module):
    """
    细粒度精确预测器

    功能：在方向和幅度约束下，输出最终精确预测值
    设计理念：多约束条件下的残差学习
    """
    def __init__(self, d_input, d_direction, d_magnitude, d_hidden=256, pred_len=96):
        super().__init__()
        self.pred_len = pred_len

        # 多约束融合
        self.constraint_fusion = nn.Sequential(
            nn.Linear(d_direction + d_magnitude, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU()
        )

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU()
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Sigmoid()
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, pred_len)
        )

    def forward(self, x, direction_embed, magnitude_embed, magnitude_value):
        """
        Args:
            x: [B*N, L, d_ff] - LLM输出表示
            direction_embed: [B*N, d_direction] - 方向嵌入
            magnitude_embed: [B*N, d_magnitude] - 幅度嵌入
            magnitude_value: [B*N, 1] - 连续幅度估计值

        Returns:
            predictions: [B*N, pred_len] - 最终预测值
        """
        B_N, L, D = x.shape

        # 约束特征融合
        constraint_features = torch.cat([direction_embed, magnitude_embed], dim=-1)
        constraint_embed = self.constraint_fusion(constraint_features)  # [B*N, d_hidden]

        # 输入特征变换
        x_features = self.feature_transform(x.mean(dim=1))  # [B*N, d_hidden]

        # 门控融合：平衡约束信息与原始特征
        combined = torch.cat([constraint_embed, x_features], dim=-1)
        gate_value = self.gate(combined)

        fused_features = gate_value * constraint_embed + (1 - gate_value) * x_features

        # 基础预测
        base_prediction = self.predictor(fused_features)  # [B*N, pred_len]

        # 幅度缩放（使预测与估计幅度对齐）
        # 预测的标准差应大致等于估计的幅度
        predictions = base_prediction * (magnitude_value.abs() + 0.1)

        return predictions
```

**D. 完整PGPM模块**

```python
class ProgressiveGranularityPredictionModule(nn.Module):
    """
    渐进式粒度预测模块 (PGPM)

    完整实现：整合粗、中、细三个粒度的预测组件
    """
    def __init__(self, configs):
        super().__init__()

        self.d_ff = configs.d_ff
        self.pred_len = configs.pred_len
        self.n_directions = 3  # 上涨、下跌、横盘
        self.n_bins = 10  # 幅度区间数量
        d_hidden = 256

        # 三个粒度编码器
        self.coarse_encoder = CoarseTrendEncoder(
            d_input=self.d_ff,
            d_hidden=d_hidden,
            n_directions=self.n_directions
        )

        self.medium_estimator = MediumMagnitudeEstimator(
            d_input=self.d_ff,
            d_direction=d_hidden // 2,
            d_hidden=d_hidden,
            n_bins=self.n_bins
        )

        self.fine_predictor = FinePrecisePredictor(
            d_input=self.d_ff,
            d_direction=d_hidden // 2,
            d_magnitude=d_hidden // 2,
            d_hidden=d_hidden,
            pred_len=self.pred_len
        )

        # 辅助损失权重
        self.direction_loss_weight = 0.1
        self.magnitude_loss_weight = 0.1

    def forward(self, x, labels=None):
        """
        Args:
            x: [B*N, L, d_ff] - LLM输出表示
            labels: [B*N, pred_len] - 真实标签（用于计算辅助损失）

        Returns:
            predictions: [B*N, pred_len] - 最终预测
            aux_losses: dict - 辅助损失（训练时）
        """
        aux_losses = {}

        # Stage 1: 粗粒度方向判断
        direction_logits, direction_embed, trend_features = self.coarse_encoder(x)

        # Stage 2: 中粒度幅度估计
        magnitude_logits, magnitude_embed, magnitude_value = self.medium_estimator(
            x, direction_embed, trend_features
        )

        # Stage 3: 细粒度精确预测
        predictions = self.fine_predictor(
            x, direction_embed, magnitude_embed, magnitude_value
        )

        # 计算辅助损失（训练时）
        if self.training and labels is not None:
            # 方向标签：根据预测期整体变化方向
            direction_labels = self._compute_direction_labels(labels)
            direction_loss = F.cross_entropy(direction_logits, direction_labels)
            aux_losses['direction_loss'] = direction_loss * self.direction_loss_weight

            # 幅度标签：根据预测期变化幅度
            magnitude_labels = self._compute_magnitude_labels(labels)
            magnitude_loss = F.cross_entropy(magnitude_logits, magnitude_labels)
            aux_losses['magnitude_loss'] = magnitude_loss * self.magnitude_loss_weight

        return predictions, aux_losses

    def _compute_direction_labels(self, labels):
        """根据真实标签计算方向分类标签"""
        # labels: [B*N, pred_len]
        start_val = labels[:, 0]
        end_val = labels[:, -1]
        diff = end_val - start_val

        # 0: 下跌, 1: 横盘, 2: 上涨
        threshold = labels.std(dim=1) * 0.1
        direction = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
        direction[diff > threshold] = 2  # 上涨
        direction[diff < -threshold] = 0  # 下跌
        direction[(diff >= -threshold) & (diff <= threshold)] = 1  # 横盘

        return direction

    def _compute_magnitude_labels(self, labels):
        """根据真实标签计算幅度区间标签"""
        # 计算标准化后的变化幅度
        magnitude = (labels.max(dim=1).values - labels.min(dim=1).values) / (labels.std(dim=1) + 1e-8)

        # 映射到区间
        magnitude_labels = (magnitude * (self.n_bins - 1)).long().clamp(0, self.n_bins - 1)

        return magnitude_labels
```

### 2.3 与Time-LLM集成

#### 2.3.1 修改位置

**文件**: `models/TimeLLM.py`

```python
# 在 Model.__init__ 中添加 (约第 245 行之后)
# ========== 新增: 渐进式粒度预测模块 ==========
self.use_pgpm = getattr(configs, 'use_pgpm', False)
if self.use_pgpm:
    from layers.PGPM import ProgressiveGranularityPredictionModule
    self.pgpm = ProgressiveGranularityPredictionModule(configs)
# =============================================

# 修改 forecast 方法 (在 output_projection 之前)
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... 前面代码保持不变 ...

    dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
    dec_out = dec_out[:, :, :self.d_ff]

    # ========== 渐进式粒度预测 ==========
    if self.use_pgpm:
        dec_out_for_pgpm = dec_out[:, -self.patch_nums:, :]  # 取patch部分
        pgpm_out, aux_losses = self.pgpm(dec_out_for_pgpm)

        # 保存辅助损失用于训练
        self.pgpm_aux_losses = aux_losses

        # Reshape: [B*N, pred_len] -> [B, pred_len, N]
        B = dec_out.shape[0] // n_vars
        pgpm_out = pgpm_out.view(B, n_vars, self.pred_len)
        pgpm_out = pgpm_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(pgpm_out, 'denorm')
        return dec_out
    # ====================================

    # 原始输出路径保持不变
    dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
    # ... 后续代码 ...
```

#### 2.3.2 命令行参数

```python
# run_main.py 中添加
parser.add_argument('--use_pgpm', action='store_true',
                    help='Enable Progressive Granularity Prediction Module')
parser.add_argument('--pgpm_direction_weight', type=float, default=0.1,
                    help='Weight for direction classification loss')
parser.add_argument('--pgpm_magnitude_weight', type=float, default=0.1,
                    help='Weight for magnitude estimation loss')
```

### 2.4 预期效果与理论分析

| 指标 | 预期提升 | 理论依据 |
|------|----------|----------|
| MSE | ↓5-10% | 层级约束减少预测空间，降低大误差风险 |
| MAE | ↓3-8% | 方向先验指导有效减少反向预测 |
| 趋势准确率 | ↑15-20% | 显式方向分类增强趋势捕捉 |
| 可解释性 | 显著提升 | 可追溯三阶段预测形成过程 |

---

## 三、模块二：检索增强的历史模式匹配模块（Retrieval-Augmented Pattern Matching Module, RAPM）

### 3.1 模块概述

#### 3.1.1 解决的问题

当前Time-LLM框架仅利用当前输入窗口的信息进行预测，存在以下局限：

1. **历史知识利用不足**：训练数据中蕴含丰富的历史模式，但未在推理时显式利用
2. **稀有模式泛化困难**：低频出现的模式难以被模型有效记忆
3. **缺乏参考基准**：预测过程缺少历史相似案例的参照

#### 3.1.2 增加的目的

RAPM模块旨在实现"通过所有学习到的、相似的历史数据，逐步缩小范围，使用整个数据特征"的预测增强：

```
核心思想：
"当前输入与历史哪些数据最相似？" → "相似历史数据后续发展如何？" → "参考历史规律指导当前预测"
```

通过检索增强机制实现：
- **相似模式检索**：从历史数据库中检索与当前输入最相似的K个模式
- **历史参考融合**：将检索到的历史模式及其后续发展作为参考信息
- **预测增强**：结合模型预测与历史参考，输出更稳健的结果

#### 3.1.3 模块原理

**理论基础**：

1. **RAFT检索增强预测** [2]：ICML 2025发表的研究证明，通过检索与当前输入最相似的历史模式，并利用这些模式的未来值辅助预测，可实现86%的平均胜率

2. **TimesURL时序表示学习** [4]：AAAI 2024提出的对比学习框架，能够学习保持时序特性的通用表示，适合作为相似度计算基础

3. **KNN检索范式**：经典的基于相似性的预测方法在时序领域已被验证有效

**创新融合**：

RAPM将检索增强与Time-LLM的语义表示相结合：

```
Time-LLM Embeddings
        │
        ▼
┌───────────────────┐
│  Pattern Encoder  │ ──────────────┐
│  (学习可检索表示)  │               │
└─────────┬─────────┘               │
          │                         │
          ▼                         ▼
┌───────────────────┐    ┌───────────────────┐
│  Retrieval Index  │    │  Current Query    │
│  (历史模式库)      │◄───│  (当前输入表示)    │
└─────────┬─────────┘    └───────────────────┘
          │
          ▼
┌───────────────────┐
│  Top-K Similar    │
│  Patterns         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Reference Fusion │
│  (参考融合)        │
└─────────┬─────────┘
          │
          ▼
   Enhanced Prediction
```

### 3.2 架构设计

#### 3.2.1 整体架构图

```
┌────────────────────────────────────────────────────────────────────────┐
│           Retrieval-Augmented Pattern Matching Module (RAPM)           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    Pattern Memory Bank                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │  │
│  │  │ Pattern 1   │  │ Pattern 2   │  │ Pattern K   │    ...      │  │
│  │  │ [input|fut] │  │ [input|fut] │  │ [input|fut] │             │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              ▲                                         │
│                              │ 检索                                    │
│  Current Input               │                                         │
│  [B*N, L, d_ff]              │                                         │
│       │                      │                                         │
│       ▼                      │                                         │
│  ┌──────────────┐    ┌──────────────┐                                │  │
│  │   Pattern    │───▶│   KNN        │                                │  │
│  │   Encoder    │    │   Retriever  │                                │  │
│  └──────────────┘    └──────┬───────┘                                │  │
│                             │                                         │
│                             ▼ Top-K Patterns                          │
│                      ┌──────────────┐                                 │
│                      │  Retrieved   │                                 │
│                      │  Patterns    │                                 │
│                      │  [K, L+P, d] │                                 │
│                      └──────┬───────┘                                 │
│                             │                                         │
│       ┌─────────────────────┼─────────────────────┐                  │
│       │                     │                     │                  │
│       ▼                     ▼                     ▼                  │
│  ┌──────────┐        ┌──────────┐          ┌──────────┐             │
│  │ History  │        │ Future   │          │ Similar- │             │
│  │ Context  │        │ Reference│          │ ity      │             │
│  │ Encoder  │        │ Decoder  │          │ Weights  │             │
│  └────┬─────┘        └────┬─────┘          └────┬─────┘             │
│       │                   │                     │                    │
│       └───────────────────┼─────────────────────┘                    │
│                           │                                          │
│                           ▼                                          │
│                    ┌──────────────┐                                  │
│                    │   Weighted   │                                  │
│                    │   Reference  │                                  │
│                    │   Fusion     │                                  │
│                    └──────┬───────┘                                  │
│                           │                                          │
│       ┌───────────────────┼───────────────────┐                      │
│       │                   │                   │                      │
│       ▼                   ▼                   ▼                      │
│  ┌──────────┐      ┌──────────────┐    ┌──────────────┐             │
│  │  Model   │      │   Gated      │    │   Residual   │             │
│  │  Output  │─────▶│   Combine    │◄───│   Reference  │             │
│  └──────────┘      └──────┬───────┘    └──────────────┘             │
│                           │                                          │
│                           ▼                                          │
│                   Enhanced Prediction                                │
│                   [B, pred_len, N]                                   │
│                                                                      │
└────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 各组件详细设计

**A. 模式编码器（Pattern Encoder）**

```python
class PatternEncoder(nn.Module):
    """
    模式编码器

    功能：将时序特征编码为可检索的紧凑表示
    设计理念：借鉴TimesURL的对比学习思想，学习保持时序特性的表示
    """
    def __init__(self, d_input, d_pattern=128):
        super().__init__()
        self.d_pattern = d_pattern

        # 时序特征聚合
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.LayerNorm(d_input),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 全局-局部特征提取
        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.local_encoder = nn.Sequential(
            nn.Conv1d(d_input, d_input // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten()
        )

        # 模式投影
        # 全局特征 + 局部特征(4个位置 * d_input//2)
        proj_input_dim = d_input + (d_input // 2) * 4
        self.pattern_projector = nn.Sequential(
            nn.Linear(proj_input_dim, d_pattern * 2),
            nn.LayerNorm(d_pattern * 2),
            nn.GELU(),
            nn.Linear(d_pattern * 2, d_pattern),
            nn.LayerNorm(d_pattern)
        )

    def forward(self, x):
        """
        Args:
            x: [B, L, D] - 时序特征

        Returns:
            pattern: [B, d_pattern] - 模式编码
        """
        # 时序聚合
        x = self.temporal_aggregator(x)  # [B, L, D]

        # 全局特征
        x_t = x.permute(0, 2, 1)  # [B, D, L]
        global_feat = self.global_encoder(x_t)  # [B, D]

        # 局部特征
        local_feat = self.local_encoder(x_t)  # [B, D//2 * 4]

        # 拼接并投影
        combined = torch.cat([global_feat, local_feat], dim=-1)
        pattern = self.pattern_projector(combined)

        # L2归一化（便于余弦相似度计算）
        pattern = F.normalize(pattern, p=2, dim=-1)

        return pattern
```

**B. 模式记忆库（Pattern Memory Bank）**

```python
class PatternMemoryBank:
    """
    模式记忆库

    功能：存储历史模式及其后续发展，支持高效KNN检索
    设计理念：借鉴RAFT的检索增强策略
    """
    def __init__(self, capacity=10000, d_pattern=128, seq_len=96, pred_len=96):
        self.capacity = capacity
        self.d_pattern = d_pattern
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 存储结构
        self.patterns = None  # [N, d_pattern] - 模式编码
        self.inputs = None    # [N, seq_len] - 原始输入序列
        self.futures = None   # [N, pred_len] - 后续真实值
        self.count = 0

        # FAISS索引（用于高效KNN检索）
        self.index = None
        self.use_faiss = False

    def initialize(self, device='cpu'):
        """初始化存储"""
        self.patterns = torch.zeros(self.capacity, self.d_pattern, device=device)
        self.inputs = torch.zeros(self.capacity, self.seq_len, device=device)
        self.futures = torch.zeros(self.capacity, self.pred_len, device=device)

        # 尝试初始化FAISS
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.d_pattern)  # 内积（等价于归一化后的余弦相似度）
            self.use_faiss = True
        except ImportError:
            print("FAISS not available, using PyTorch for retrieval")
            self.use_faiss = False

    def add(self, patterns, inputs, futures):
        """
        添加新模式到记忆库

        Args:
            patterns: [B, d_pattern] - 模式编码
            inputs: [B, seq_len] - 输入序列
            futures: [B, pred_len] - 后续真实值
        """
        B = patterns.shape[0]

        # 循环替换策略
        indices = torch.arange(self.count, self.count + B) % self.capacity

        self.patterns[indices] = patterns.detach()
        self.inputs[indices] = inputs.detach()
        self.futures[indices] = futures.detach()

        self.count = min(self.count + B, self.capacity)

        # 更新FAISS索引
        if self.use_faiss and self.count > 0:
            self.index.reset()
            self.index.add(self.patterns[:self.count].cpu().numpy())

    def retrieve(self, query_patterns, k=5):
        """
        检索最相似的K个模式

        Args:
            query_patterns: [B, d_pattern] - 查询模式
            k: 检索数量

        Returns:
            retrieved_patterns: [B, K, d_pattern]
            retrieved_inputs: [B, K, seq_len]
            retrieved_futures: [B, K, pred_len]
            similarities: [B, K] - 相似度分数
        """
        B = query_patterns.shape[0]
        k = min(k, self.count)

        if self.use_faiss and self.count > k:
            # 使用FAISS进行快速检索
            query_np = query_patterns.detach().cpu().numpy()
            similarities, indices = self.index.search(query_np, k)

            indices = torch.from_numpy(indices).to(query_patterns.device)
            similarities = torch.from_numpy(similarities).to(query_patterns.device)
        else:
            # PyTorch实现
            # 计算余弦相似度
            valid_patterns = self.patterns[:self.count]  # [N, d_pattern]
            similarities = torch.mm(query_patterns, valid_patterns.T)  # [B, N]

            # Top-K
            similarities, indices = similarities.topk(k, dim=-1)

        # 收集检索结果
        retrieved_patterns = self.patterns[indices]  # [B, K, d_pattern]
        retrieved_inputs = self.inputs[indices]      # [B, K, seq_len]
        retrieved_futures = self.futures[indices]    # [B, K, pred_len]

        return retrieved_patterns, retrieved_inputs, retrieved_futures, similarities
```

**C. 参考融合模块（Reference Fusion Module）**

```python
class ReferenceFusionModule(nn.Module):
    """
    参考融合模块

    功能：将检索到的历史参考与模型预测融合
    设计理念：借鉴RAFT的weighted reference策略
    """
    def __init__(self, d_input, d_pattern, pred_len, n_retrieved=5):
        super().__init__()
        self.pred_len = pred_len
        self.n_retrieved = n_retrieved

        # 历史上下文编码
        self.history_encoder = nn.Sequential(
            nn.Linear(d_pattern, d_pattern),
            nn.LayerNorm(d_pattern),
            nn.GELU()
        )

        # 参考权重计算（基于相似度的注意力）
        self.weight_calculator = nn.Sequential(
            nn.Linear(d_pattern * 2, d_pattern),
            nn.GELU(),
            nn.Linear(d_pattern, 1)
        )

        # 参考值变换（适应当前上下文）
        self.reference_transformer = nn.Sequential(
            nn.Linear(pred_len, pred_len * 2),
            nn.LayerNorm(pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

        # 模型预测与参考融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(pred_len * 2 + d_pattern, pred_len),
            nn.Sigmoid()
        )

    def forward(self, model_output, query_pattern, retrieved_patterns,
                retrieved_futures, similarities):
        """
        Args:
            model_output: [B, pred_len] - 模型原始预测
            query_pattern: [B, d_pattern] - 当前输入的模式编码
            retrieved_patterns: [B, K, d_pattern] - 检索到的模式
            retrieved_futures: [B, K, pred_len] - 检索模式的后续值
            similarities: [B, K] - 相似度分数

        Returns:
            fused_output: [B, pred_len] - 融合后的预测
            reference_weight: [B, 1] - 参考权重（用于分析）
        """
        B, K, _ = retrieved_patterns.shape

        # 编码历史上下文
        history_context = self.history_encoder(retrieved_patterns)  # [B, K, d_pattern]

        # 计算自适应权重
        query_expanded = query_pattern.unsqueeze(1).expand(-1, K, -1)  # [B, K, d_pattern]
        combined = torch.cat([query_expanded, history_context], dim=-1)  # [B, K, d_pattern*2]
        attention_scores = self.weight_calculator(combined).squeeze(-1)  # [B, K]

        # 结合相似度的注意力
        attention_scores = attention_scores + similarities  # 相似度作为先验
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, K]

        # 加权参考
        weighted_reference = torch.einsum('bk,bkp->bp',
                                          attention_weights,
                                          retrieved_futures)  # [B, pred_len]

        # 变换参考值
        transformed_reference = self.reference_transformer(weighted_reference)

        # 门控融合
        fusion_input = torch.cat([model_output, transformed_reference, query_pattern], dim=-1)
        gate = self.fusion_gate(fusion_input)  # [B, pred_len]

        # 融合输出
        fused_output = gate * transformed_reference + (1 - gate) * model_output

        # 计算整体参考权重（用于分析）
        reference_weight = gate.mean(dim=-1, keepdim=True)

        return fused_output, reference_weight
```

**D. 完整RAPM模块**

```python
class RetrievalAugmentedPatternMatchingModule(nn.Module):
    """
    检索增强的历史模式匹配模块 (RAPM)

    完整实现：整合模式编码、检索和融合
    """
    def __init__(self, configs):
        super().__init__()

        self.d_ff = configs.d_ff
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_pattern = getattr(configs, 'rapm_d_pattern', 128)
        self.n_retrieved = getattr(configs, 'rapm_n_retrieved', 5)
        self.memory_capacity = getattr(configs, 'rapm_memory_capacity', 10000)

        # 模式编码器
        self.pattern_encoder = PatternEncoder(
            d_input=self.d_ff,
            d_pattern=self.d_pattern
        )

        # 模式记忆库
        self.memory_bank = PatternMemoryBank(
            capacity=self.memory_capacity,
            d_pattern=self.d_pattern,
            seq_len=self.seq_len,
            pred_len=self.pred_len
        )

        # 参考融合模块
        self.reference_fusion = ReferenceFusionModule(
            d_input=self.d_ff,
            d_pattern=self.d_pattern,
            pred_len=self.pred_len,
            n_retrieved=self.n_retrieved
        )

        # 是否初始化
        self.initialized = False

    def initialize(self, device):
        """延迟初始化记忆库"""
        if not self.initialized:
            self.memory_bank.initialize(device)
            self.initialized = True

    def update_memory(self, x_enc, y_true, llm_features):
        """
        更新记忆库（训练时调用）

        Args:
            x_enc: [B, seq_len, N] - 输入序列
            y_true: [B, pred_len, N] - 真实目标值
            llm_features: [B*N, L, d_ff] - LLM输出特征
        """
        B, T, N = x_enc.shape

        # 编码模式
        with torch.no_grad():
            patterns = self.pattern_encoder(llm_features)  # [B*N, d_pattern]

        # 重组输入和目标
        x_flat = x_enc.permute(0, 2, 1).reshape(B * N, T)  # [B*N, seq_len]
        y_flat = y_true.permute(0, 2, 1).reshape(B * N, self.pred_len)  # [B*N, pred_len]

        # 添加到记忆库
        self.memory_bank.add(patterns, x_flat, y_flat)

    def forward(self, llm_features, model_output):
        """
        Args:
            llm_features: [B*N, L, d_ff] - LLM输出特征
            model_output: [B*N, pred_len] - 模型原始预测

        Returns:
            enhanced_output: [B*N, pred_len] - 增强后的预测
            reference_info: dict - 检索信息（用于分析）
        """
        # 确保初始化
        self.initialize(llm_features.device)

        # 编码当前模式
        query_pattern = self.pattern_encoder(llm_features)  # [B*N, d_pattern]

        # 检索相似模式
        if self.memory_bank.count < self.n_retrieved:
            # 记忆库不足，直接返回原始输出
            return model_output, {'reference_weight': torch.zeros(1)}

        retrieved_patterns, retrieved_inputs, retrieved_futures, similarities = \
            self.memory_bank.retrieve(query_pattern, k=self.n_retrieved)

        # 融合参考
        enhanced_output, reference_weight = self.reference_fusion(
            model_output,
            query_pattern,
            retrieved_patterns,
            retrieved_futures,
            similarities
        )

        reference_info = {
            'reference_weight': reference_weight,
            'top_similarity': similarities[:, 0].mean(),
            'avg_similarity': similarities.mean()
        }

        return enhanced_output, reference_info
```

### 3.3 与Time-LLM集成

#### 3.3.1 修改位置

**文件**: `models/TimeLLM.py`

```python
# 在 Model.__init__ 中添加
# ========== 新增: 检索增强模块 ==========
self.use_rapm = getattr(configs, 'use_rapm', False)
if self.use_rapm:
    from layers.RAPM import RetrievalAugmentedPatternMatchingModule
    self.rapm = RetrievalAugmentedPatternMatchingModule(configs)
# ========================================

# 修改 forecast 方法
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... LLM处理后 ...

    dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
    dec_out = dec_out[:, :, :self.d_ff]

    # 原始预测路径
    dec_out_reshaped = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
    dec_out_reshaped = dec_out_reshaped.permute(0, 1, 3, 2).contiguous()
    original_output = self.output_projection(dec_out_reshaped[:, :, :, -self.patch_nums:])
    original_output = original_output.permute(0, 2, 1).contiguous()

    # ========== 检索增强 ==========
    if self.use_rapm:
        # Reshape for RAPM: [B, pred_len, N] -> [B*N, pred_len]
        B = original_output.shape[0]
        output_flat = original_output.permute(0, 2, 1).reshape(B * n_vars, self.pred_len)

        # 获取LLM特征
        llm_features = dec_out[:, -self.patch_nums:, :]  # [B*N, patch_nums, d_ff]

        # 检索增强
        enhanced_output, ref_info = self.rapm(llm_features, output_flat)

        # Reshape回来
        enhanced_output = enhanced_output.view(B, n_vars, self.pred_len)
        enhanced_output = enhanced_output.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(enhanced_output, 'denorm')
        return dec_out
    # ==============================

    dec_out = self.normalize_layers(original_output, 'denorm')
    return dec_out
```

#### 3.3.2 训练循环修改

```python
# 在 run_main.py 的训练循环中添加记忆库更新
if hasattr(model, 'rapm') and model.use_rapm:
    # 获取LLM特征（需要额外前向传播或缓存）
    with torch.no_grad():
        # 更新记忆库
        model.rapm.update_memory(batch_x, batch_y, cached_llm_features)
```

#### 3.3.3 命令行参数

```python
# run_main.py 中添加
parser.add_argument('--use_rapm', action='store_true',
                    help='Enable Retrieval-Augmented Pattern Matching Module')
parser.add_argument('--rapm_d_pattern', type=int, default=128,
                    help='Pattern embedding dimension')
parser.add_argument('--rapm_n_retrieved', type=int, default=5,
                    help='Number of patterns to retrieve')
parser.add_argument('--rapm_memory_capacity', type=int, default=10000,
                    help='Memory bank capacity')
```

### 3.4 预期效果与理论分析

| 指标 | 预期提升 | 理论依据 |
|------|----------|----------|
| MSE | ↓8-15% | RAFT论文报告86%胜率 [2] |
| MAE | ↓5-12% | 历史参考减少离群预测 |
| 稀有模式预测 | ↑20-30% | 检索机制弥补记忆不足 |
| 少样本场景 | ↑15-25% | 有效利用已有数据 |

---

## 四、模块集成与整体架构

### 4.1 两模块协同工作流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Time-LLM + PGPM + RAPM 整体架构                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入时序 x_enc                                                         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌──────────────────────────────────────────────────┐                  │
│  │                Time-LLM Backbone                  │                  │
│  │  [Patching] → [Reprogramming] → [Frozen LLM]     │                  │
│  └────────────────────────┬─────────────────────────┘                  │
│                           │                                             │
│                           ▼ LLM Features                                │
│       ┌───────────────────┴───────────────────┐                        │
│       │                                       │                        │
│       ▼                                       ▼                        │
│  ┌──────────────┐                      ┌──────────────┐               │
│  │     PGPM     │                      │     RAPM     │               │
│  │  渐进式粒度   │                      │  检索增强    │               │
│  │  预测模块     │                      │  模式匹配    │               │
│  └──────┬───────┘                      └──────┬───────┘               │
│         │                                     │                        │
│         │ Direction+Magnitude+Precise         │ Retrieved Reference     │
│         │                                     │                        │
│         └─────────────────┬───────────────────┘                        │
│                           │                                             │
│                           ▼                                             │
│                    ┌──────────────┐                                    │
│                    │   Final      │                                    │
│                    │   Ensemble   │                                    │
│                    │   Fusion     │                                    │
│                    └──────┬───────┘                                    │
│                           │                                             │
│                           ▼                                             │
│                    最终预测输出                                          │
│                    [B, pred_len, N]                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 模块组合策略

#### 4.2.1 独立使用

```bash
# 仅使用PGPM
python run_main.py --use_pgpm ...

# 仅使用RAPM
python run_main.py --use_rapm ...
```

#### 4.2.2 联合使用

```bash
# PGPM + RAPM 联合
python run_main.py --use_pgpm --use_rapm ...
```

#### 4.2.3 与已有创新方案组合

```bash
# PGPM + 变量间注意力
python run_main.py --use_pgpm --use_inter_variate ...

# RAPM + 频域增强
python run_main.py --use_rapm --use_frequency ...

# 全组合
python run_main.py --use_pgpm --use_rapm --use_inter_variate --use_frequency ...
```

### 4.3 新增文件清单

```
Time-LLM/
├── layers/
│   ├── PGPM.py                              # 渐进式粒度预测模块
│   │   ├── CoarseTrendEncoder               # 粗粒度趋势编码器
│   │   ├── MediumMagnitudeEstimator         # 中粒度幅度估计器
│   │   ├── FinePrecisePredictor             # 细粒度精确预测器
│   │   └── ProgressiveGranularityPredictionModule  # 完整模块
│   │
│   └── RAPM.py                              # 检索增强模式匹配模块
│       ├── PatternEncoder                   # 模式编码器
│       ├── PatternMemoryBank               # 模式记忆库
│       ├── ReferenceFusionModule           # 参考融合模块
│       └── RetrievalAugmentedPatternMatchingModule  # 完整模块
│
└── models/
    └── TimeLLM.py                           # 修改：集成新模块
```

---

## 五、实验设计与预期效果

### 5.1 消融实验设计

| 实验组 | 配置 | 验证目标 |
|--------|------|----------|
| Baseline | Time-LLM | 基准性能 |
| +PGPM | Time-LLM + PGPM | 渐进式预测的有效性 |
| +RAPM | Time-LLM + RAPM | 检索增强的有效性 |
| +Both | Time-LLM + PGPM + RAPM | 两模块协同效果 |
| +All | 全创新方案 | 整体提升 |

### 5.2 实验命令

```bash
# Baseline
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 4

# +PGPM
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 4 \
  --use_pgpm \
  --pgpm_direction_weight 0.1 \
  --pgpm_magnitude_weight 0.1

# +RAPM
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 4 \
  --use_rapm \
  --rapm_d_pattern 128 \
  --rapm_n_retrieved 5

# +Both
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 4 \
  --use_pgpm \
  --use_rapm
```

### 5.3 预期效果总结

| 数据集 | Baseline MSE | +PGPM | +RAPM | +Both | 总提升 |
|--------|--------------|-------|-------|-------|--------|
| ETTh1 | 0.375 | 0.356 (-5%) | 0.338 (-10%) | 0.319 (-15%) | **-15%** |
| ETTh2 | 0.288 | 0.274 (-5%) | 0.259 (-10%) | 0.245 (-15%) | **-15%** |
| ETTm1 | 0.302 | 0.287 (-5%) | 0.272 (-10%) | 0.257 (-15%) | **-15%** |
| ETTm2 | 0.175 | 0.166 (-5%) | 0.158 (-10%) | 0.149 (-15%) | **-15%** |

### 5.4 工作量说明

| 模块 | 代码量（估计） | 主要工作 |
|------|----------------|----------|
| PGPM | ~500行 | 三阶段预测器设计与实现、辅助损失计算 |
| RAPM | ~600行 | 模式编码器、记忆库、检索融合 |
| 集成 | ~200行 | 与Time-LLM框架集成、参数配置 |
| 实验 | ~100行 | 消融实验脚本、结果可视化 |
| **总计** | **~1400行** | 完整模块实现与验证 |

---

## 六、参考文献

[1] Xu, X., Chen, C., Liang, Y., et al. **SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series Forecasting**. *ACM CIKM 2025*. [arXiv:2404.14757](https://arxiv.org/abs/2404.14757)

[2] Han, S., Lee, S., Cha, M., Arik, S.O., Yoon, J. **RAFT: Retrieval Augmented Time Series Forecasting**. *ICML 2025*. [arXiv:2505.04163](https://arxiv.org/abs/2505.04163)

[3] Shen, L., et al. **Multi-Resolution Diffusion Models for Time Series Forecasting**. *ICLR 2024*. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/d64740dd69bcc90ba225a182984b81ba-Paper-Conference.pdf)

[4] Liu, J., Chen, S. **TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning**. *AAAI 2024*. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29299)

[5] Wang, S., et al. **TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting**. *ICLR 2024*. [GitHub](https://github.com/kwuking/TimeMixer)

[6] Kopp, M., et al. **C2FAR: Coarse-to-Fine Autoregressive Networks for Precise Probabilistic Forecasting**. *NeurIPS 2023*.

[7] Jin, M., Wang, S., Ma, L., et al. **Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**. *ICLR 2024*.

[8] Liu, Y., et al. **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**. *ICLR 2024*.

[9] Gu, A., Dao, T. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**. *ICLR 2024*.

[10] Zhou, T., et al. **FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting**. *ICML 2022*.

---

## 附录：代码实现检查清单

- [ ] `layers/PGPM.py` - 渐进式粒度预测模块完整实现
- [ ] `layers/RAPM.py` - 检索增强模式匹配模块完整实现
- [ ] `models/TimeLLM.py` - 模块集成代码
- [ ] `run_main.py` - 命令行参数添加
- [ ] 消融实验脚本
- [ ] 结果可视化代码

---

**文档版本**: v1.0
**生成日期**: 2026-02-02
**作者**: 王振达
**指导框架**: Time-LLM (ICLR 2024)
