# Time-LLM Enhanced v3 — 技术文档

## 1. 版本对比

| 特性 | v2 (观察者模式) | v3 (主动预测融合) |
|------|----------------|-----------------|
| 辅助模块角色 | 仅观察，不参与预测 | 产生独立预测，通过矩阵融合 |
| 辅助损失权重 | 0.01 (几乎无效) | 0.1 (适中，发挥作用) |
| 融合方式 | 无/冲突时微调 | 四类可训练矩阵 |
| 预期效果 | ≈ baseline | MSE ↓10-20% |
| Baseline修改 | 无 | 无 |

## 2. 四类可训练矩阵

### 2.1 参数矩阵 Θ_s (每个尺度内部)

每个辅助尺度 (S2-S5) 拥有独立的 `ScaleEncoder`，包含:
- Conv1d × 2 (特征提取)
- Linear (FlattenHead，映射到 pred_len)

这些参数通过反向传播学习，构成每个尺度的 "参数矩阵"。

### 2.2 权重矩阵 W (C2F跨尺度融合)

- 形状: `[n_scales, pred_len]`
- 作用: 对每个时间步，为各尺度预测分配权重
- 训练: 通过 softmax 归一化，KL散度一致性约束
- 推理: 检测趋势不一致的尺度并降权

```python
self.weight_matrix = nn.Parameter(torch.ones(5, 96) / 5)
weights = F.softmax(self.weight_matrix, dim=0)  # [5, 96]
```

### 2.3 联系矩阵 C (AKG历史-预测关系)

- 形状: `[n_scales, pred_len]`
- 作用: 每个尺度每个时间步的历史vs预测门控
- sigmoid(C[s,t]) = 1 → 信任当前预测
- sigmoid(C[s,t]) = 0 → 信任历史检索

```python
self.connection_matrix = nn.Parameter(torch.zeros(5, 96))
gates = torch.sigmoid(self.connection_matrix)
connected = gates * prediction + (1 - gates) * history
```

### 2.4 融合矩阵 F (AKG最终融合)

- 形状: `[2*n_scales, pred_len]`
- 前 n_scales 行: 各尺度原始预测的权重
- 后 n_scales 行: 各尺度经联系矩阵处理后结果的权重

```python
self.fusion_matrix = nn.Parameter(torch.zeros(10, 96))
fusion_weights = F.softmax(self.fusion_matrix, dim=0)
```

## 3. 模块详细设计

### 3.1 DM (Decomposable Multi-scale) — `layers/TAPR_v3.py`

**信号变换:**

| 尺度 | 下采样率 | 变换 | 目的 |
|------|---------|------|------|
| S1 | 1 | 无 (baseline) | 完整信息 |
| S2 | 2 | 一阶差分 | 变化率 |
| S3 | 4 | 保留原始 | 中尺度特征 |
| S4 | 8 | 移动平均 | 去噪趋势 |
| S5 | 16 | 大窗口平均 | 宏观趋势 |

**数据流:**
```
x_normed [B, T, N]
  → downsample (rate=ds)
  → signal_transform (diff/smooth/trend)
  → create_patches (patch_len=16, stride=8)
  → Conv1d embedding
  → ScaleEncoder (Conv×2 + Linear)
  → pred_s [B, pred_len, N]
```

### 3.2 C2F (Coarse-to-Fine Fusion) — `layers/TAPR_v3.py`

**训练时 (软约束):**
- 计算各尺度预测的趋势方向分布 (上/平/下)
- 对每对 (粗,细) 尺度计算对称 KL 散度
- L_consistency = Σ KL(fine ‖ coarse)

**推理时 (硬约束):**
- 以最粗尺度 (S5) 为基准趋势方向
- 检查每个细尺度是否与基准一致
- 不一致的尺度权重 × decay_factor (0.5)
- 重新归一化权重

### 3.3 PVDR (Pattern-Value Dual Retriever) — `layers/GRAM_v3.py`

**多尺度记忆库:**
- 每个尺度有独立的 (keys, values) 存储
- keys: `[M, d_repr]` — 统计特征编码 (mean, std, max, min, trend)
- values: `[M, pred_len]` — 对应的未来值 (均值压缩)

**极端数据检测:**
- z-score > 3σ 时标记为极端
- 极端样本: 检索阈值降低 0.2 (引入更多历史参考)

**检索条件:**
- 余弦相似度 > 可学习阈值 (per scale, 初始 sigmoid(0.8))
- Top-K=5 检索，按相似度加权平均

### 3.4 AKG (Adaptive Knowledge Gating) — `layers/GRAM_v3.py`

**两阶段融合:**

1. 联系矩阵 → 每个尺度内: 预测 vs 历史
2. 融合矩阵 → 跨尺度: 所有来源 → 最终预测

**门控正则化:**
- L_gate = -mean(|sigmoid(C) - 0.5|)
- 鼓励门控做出明确决策 (偏向0或1)

## 4. 损失函数

```
L_total = L_main + λ_consist × L_consistency + λ_gate × L_gate

L_main:        MSE(final_pred, y_true)
L_consistency: Σ symmetric_KL(trend_fine, trend_coarse) / n_pairs
L_gate:        -mean(|sigmoid(connection_matrix) - 0.5|)

λ_consist = 0.1
λ_gate = 0.01
warmup: 前500步辅助损失 × (step / 500)
```

## 5. 显存估算 (6GB VRAM, batch_size=4)

| 组件 | 显存 |
|------|------|
| Qwen 2.5 3B (4-bit, 6层) | ~1.5 GB |
| Time-LLM 原始可训练参数 | ~0.5 GB |
| DM: 4个 ScaleEncoder (~0.1M × 4) | ~2 MB |
| C2F: 权重矩阵 [5, 96] | < 1 KB |
| PVDR: 5个记忆库 (5000 × 5 scales) | ~5 MB |
| AKG: 联系矩阵 + 融合矩阵 | < 1 KB |
| 中间张量 (5尺度并行) | ~0.3 GB |
| 系统开销 | ~1.0 GB |
| **总计** | **~5.3 GB** |

## 6. 消融实验配置

```bash
# E1: Baseline only
USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh

# E2: +TAPR (DM + C2F)
USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh

# E3: +GRAM (PVDR + AKG)
USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh

# E4: Full (DM + C2F + PVDR + AKG)
USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh
```

## 7. 新增文件

| 文件 | 内容 |
|------|------|
| `layers/TAPR_v3.py` | SignalTransform + ScaleEncoder + DecomposableMultiScale + CoarseToFineFusion |
| `layers/GRAM_v3.py` | LightweightPatternEncoder + ExtremeDetector + MultiScaleMemoryBank + PatternValueDualRetriever + AdaptiveKnowledgeGating |
| `models/TimeLLM_Enhanced_v3.py` | 集成模型 (baseline + DM + C2F + PVDR + AKG) |
| `run_main_enhanced_v3.py` | 训练入口 (新参数 + 多损失训练) |
| `scripts/TimeLLM_ETTh1_enhanced_v3.sh` | 训练脚本 (支持消融实验) |
| `claude-enhanced-v3.md` | 本文档 |

## 8. 关键论文引用

| 技术 | 论文 |
|------|------|
| 多尺度独立预测+融合 | TimeMixer (ICLR 2024) |
| 插值对齐到统一长度 | N-HiTS (AAAI 2023) |
| 跨尺度归一化 | Scaleformer (ICLR 2023) |
| 粗到细条件分解 | C2FAR (NeurIPS 2022) |
| GLU门控融合 | TFT (IJF 2021) |
| 检索增强时序预测 | RAFT (ICML 2025) |
| 自适应检索混合器 | TS-RAG (NeurIPS 2025) |
| 信号分解 | Autoformer (NeurIPS 2021) |
| 多分辨率patch | Pathformer (ICLR 2024) |
| 自适应门控 | 检索增强门控框架 (SJTU 2025) |
