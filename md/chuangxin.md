# Time-LLM 创新改进方案 (Innovation Proposals)

> **目标**: 在 Time-LLM 框架基础上，参考 TimesNet、iTransformer、PatchTST、TimeMixer、Time-MoE 等最新模型，提出可行的创新改进方案
> **参考文献**: ICLR 2024-2025 顶会论文

---

## 一、当前 Time-LLM 架构分析

### 1.1 核心机制回顾

```
时序数据 → [Patching] → [Reprogramming Layer] → [Frozen LLM] → [Output Projection] → 预测结果
             ↑                    ↑
         局部模式提取         跨模态对齐
```

### 1.2 现有局限性

| 问题 | 描述 | 影响 |
|------|------|------|
| **单尺度 Patching** | 仅使用固定 patch_len=16, stride=8 | 无法捕获多尺度时序模式 |
| **通道独立处理** | 多变量被展平为独立样本处理 | 忽略变量间相关性 |
| **静态 Prompt** | 统计信息提示缺乏动态调整 | 难以适应非平稳时序 |
| **单一 LLM 层** | 仅使用 LLM 部分层 | 可能浪费深层语义能力 |
| **计算效率** | 大 LLM 推理开销大 | 实时场景受限 |

---

## 二、创新方案一: 多尺度分解混合 (Multi-Scale Decomposition Mixing)

### 2.1 创新原理

**灵感来源**: [TimeMixer (ICLR 2024)](https://github.com/kwuking/TimeMixer) 的多尺度分解思想

TimeMixer 证明了时序数据在不同尺度下展现不同模式:
- **细粒度**: 捕获局部波动、噪声
- **粗粒度**: 捕获趋势、周期性

**核心思想**: 将 Time-LLM 的单尺度 Patching 扩展为多尺度分解，然后在 LLM 内部进行尺度混合。

### 2.2 架构设计

```
原始时序
    │
    ├── [下采样 1x] → Patch Embedding (细粒度) → Reprogramming → LLM Layer 1-2
    │
    ├── [下采样 2x] → Patch Embedding (中粒度) → Reprogramming → LLM Layer 3-4
    │
    └── [下采样 4x] → Patch Embedding (粗粒度) → Reprogramming → LLM Layer 5-6
                                                                    │
                                                                    ▼
                                                        [Multi-Scale Fusion]
                                                                    │
                                                                    ▼
                                                              预测输出
```

### 2.3 代码修改位置

**文件**: `models/TimeLLM.py`

```python
# 新增: 多尺度分解模块 (在 class Model 中添加)
class MultiScaleDecomposition(nn.Module):
    """多尺度时序分解"""
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.downsamples = nn.ModuleList([
            nn.AvgPool1d(kernel_size=s, stride=s) if s > 1 else nn.Identity()
            for s in scales
        ])

    def forward(self, x):
        # x: [B, T, N]
        outputs = []
        for ds in self.downsamples:
            x_ds = ds(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T/s, N]
            outputs.append(x_ds)
        return outputs  # List of [B, T/s, N]

# 修改: forward 方法中添加多尺度处理
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    # 多尺度分解
    multi_scale_inputs = self.multi_scale_decomp(x_enc)

    # 每个尺度独立处理
    multi_scale_outputs = []
    for scale_idx, x_scale in enumerate(multi_scale_inputs):
        # Patching for this scale
        enc_out, _ = self.patch_embedding(x_scale)
        # Reprogramming
        enc_out = self.reprogramming_layer(enc_out, ...)
        # Use different LLM layers for different scales
        start_layer = scale_idx * 2
        end_layer = start_layer + 2
        for layer in self.llm_layers[start_layer:end_layer]:
            enc_out = layer(enc_out)
        multi_scale_outputs.append(enc_out)

    # Fusion
    output = self.multi_scale_fusion(multi_scale_outputs)
    return output
```

### 2.4 预期效果

- **MSE 降低**: 预计 5-10%，尤其在长周期数据集 (ETTh1/h2)
- **鲁棒性提升**: 对噪声和异常值更鲁棒
- **参数增加**: 约 20%，但推理时间增加有限

---

## 三、创新方案二: 变量间注意力增强 (Inter-Variate Attention)

### 3.1 创新原理

**灵感来源**: [iTransformer (ICLR 2024)](https://arxiv.org/abs/2310.06625)

iTransformer 发现:
- 原始 Transformer 在时间维度做 attention，忽略变量相关性
- **反转思路**: 在变量维度做 attention，每个变量作为一个 token

**核心思想**: 在 Time-LLM 的 Reprogramming 后添加 Inter-Variate Attention 模块。

### 3.2 架构设计

```
Patch Embeddings [B*N, num_patches, d_model]
            │
            ▼
    [Reprogramming Layer]
            │
            ▼
    [B*N, num_patches, llm_dim]
            │
            ▼
    [Reshape to [B, N, num_patches, llm_dim]]
            │
            ▼
┌───────────────────────────────────┐
│  Inter-Variate Attention          │
│  Query/Key/Value: 变量维度 N      │
│  捕获: 变量间相关性               │
└───────────────────────────────────┘
            │
            ▼
    [Reshape back to [B*N, num_patches, llm_dim]]
            │
            ▼
        [LLM Forward]
```

### 3.3 代码修改位置

**文件**: `models/TimeLLM.py`

```python
# 新增: 变量间注意力模块
class InterVariateAttention(nn.Module):
    """捕获多变量之间的相关性"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, n_vars):
        # x: [B*N, num_patches, d_model]
        B_N, num_patches, d_model = x.shape
        B = B_N // n_vars

        # Reshape: [B, N, num_patches, d_model] -> [B*num_patches, N, d_model]
        x = x.view(B, n_vars, num_patches, d_model)
        x = x.permute(0, 2, 1, 3).reshape(B * num_patches, n_vars, d_model)

        # Inter-variate attention (在变量维度 N 上做 attention)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # Reshape back: [B*num_patches, N, d_model] -> [B*N, num_patches, d_model]
        x = x.view(B, num_patches, n_vars, d_model).permute(0, 2, 1, 3)
        x = x.reshape(B * n_vars, num_patches, d_model)

        return x

# 在 forward 中调用
enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
enc_out = self.inter_variate_attn(enc_out, n_vars)  # 新增
```

### 3.4 预期效果

- **多变量预测提升**: MSE 降低 8-15%，特别是 `features=M` 场景
- **参数增加**: 约 5%
- **适用场景**: 变量间有强相关性的数据集 (如 Traffic, Electricity)

---

## 四、创新方案三: 动态 Prompt 生成 (Dynamic Prompt Generation)

### 4.1 创新原理

**灵感来源**: [AutoTimes (NeurIPS 2024)](https://arxiv.org/abs/2402.02370) 的自回归 prompt 思想

当前 Time-LLM 的 Prompt 是静态的统计信息:
- `min_value, max_value, median, trend, top-5 lags`

**局限性**:
- 无法捕获时序的非平稳性变化
- 对分布漂移 (Distribution Shift) 不敏感

**核心思想**: 使用可学习的 Prompt Encoder 动态生成 prompt embeddings。

### 4.2 架构设计

```
原始时序 x_enc
      │
      ├──────────────────────────────────┐
      │                                  │
      ▼                                  ▼
[统计特征提取]                    [Prompt Encoder (可学习)]
  min/max/median/...                    │
      │                                  │
      ▼                                  ▼
[Text Tokenizer]              [Learned Prompt Tokens]
      │                                  │
      ▼                                  ▼
[Static Prompt Emb]           [Dynamic Prompt Emb]
      │                                  │
      └──────────┬───────────────────────┘
                 │
                 ▼
        [Prompt Fusion (Attention)]
                 │
                 ▼
           Final Prompt Embeddings
```

### 4.3 代码修改位置

**文件**: `models/TimeLLM.py`

```python
# 新增: 动态 Prompt 编码器
class DynamicPromptEncoder(nn.Module):
    """从时序数据直接学习 prompt embeddings"""
    def __init__(self, seq_len, n_vars, llm_dim, num_prompt_tokens=32):
        super().__init__()
        self.num_tokens = num_prompt_tokens

        # 时序编码器
        self.temporal_encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.GELU(),
            nn.Linear(256, num_prompt_tokens * llm_dim)
        )

        # 可学习的基础 prompt tokens
        self.base_prompt = nn.Parameter(torch.randn(1, num_prompt_tokens, llm_dim))

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(llm_dim * 2, llm_dim),
            nn.Sigmoid()
        )

    def forward(self, x_enc):
        # x_enc: [B, T, N]
        B, T, N = x_enc.shape

        # 生成动态 prompt: [B, num_tokens, llm_dim]
        x_flat = x_enc.mean(dim=-1)  # [B, T]
        dynamic_prompt = self.temporal_encoder(x_flat)
        dynamic_prompt = dynamic_prompt.view(B, self.num_tokens, -1)

        # 与基础 prompt 融合
        base_prompt = self.base_prompt.expand(B, -1, -1)
        combined = torch.cat([dynamic_prompt, base_prompt], dim=-1)
        gate = self.fusion_gate(combined)

        output = gate * dynamic_prompt + (1 - gate) * base_prompt
        return output

# 在 forward 中替换静态 prompt
# 原始: prompt_embeddings = self.llm_model.get_input_embeddings()(prompt)
# 修改为:
static_prompt_emb = self.llm_model.get_input_embeddings()(prompt)
dynamic_prompt_emb = self.dynamic_prompt_encoder(x_enc)
prompt_embeddings = self.prompt_fusion(static_prompt_emb, dynamic_prompt_emb)
```

### 4.4 预期效果

- **非平稳时序性能提升**: MSE 降低 10-20%
- **迁移能力增强**: 跨数据集泛化更好
- **参数增加**: 约 15%

---

## 五、创新方案四: 稀疏专家混合 (Mixture of Experts for Time-LLM)

### 5.1 创新原理

**灵感来源**: [Time-MoE (ICLR 2025 Spotlight)](https://github.com/Time-MoE/Time-MoE)

Time-MoE 证明了:
- MoE 架构可以在保持计算效率的同时大幅扩展模型容量
- 2.4B 参数模型仅激活 1B 参数，显存需求 < 8GB

**核心思想**: 在 Time-LLM 的 Reprogramming Layer 后添加 MoE 层。

### 5.2 架构设计

```
Reprogrammed Embeddings [B*N, num_patches, llm_dim]
                │
                ▼
┌────────────────────────────────────────┐
│       Mixture of Experts Layer         │
│  ┌─────────────────────────────────┐   │
│  │  Router: 计算每个 patch 的      │   │
│  │  expert 分配概率                │   │
│  └─────────────────────────────────┘   │
│           │                             │
│    ┌──────┼──────┬──────┐              │
│    ▼      ▼      ▼      ▼              │
│  Expert1 Expert2 Expert3 Expert4       │
│  (Trend) (Season)(Short) (Long)        │
│    │      │      │      │              │
│    └──────┴──────┴──────┘              │
│           │                             │
│           ▼                             │
│    [Top-K Sparse Selection]            │
└────────────────────────────────────────┘
                │
                ▼
            LLM Forward
```

### 5.3 代码修改位置

**文件**: `models/TimeLLM.py`

```python
# 新增: 稀疏专家混合层
class TimeSeriesMoE(nn.Module):
    """时序专用的 MoE 层"""
    def __init__(self, d_model, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: 决定使用哪些专家
        self.router = nn.Linear(d_model, num_experts)

        # Experts: 每个专家专注于不同的时序模式
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        # 计算 router 概率
        router_logits = self.router(x)  # [B, L, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 稀疏激活专家
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]  # [B, L]
            prob = top_k_probs[:, :, i:i+1]  # [B, L, 1]

            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)  # [B, L]
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += prob[mask].squeeze(-1).unsqueeze(-1) * expert_output

        return self.norm(x + output)

# 在模型中使用
self.moe_layer = TimeSeriesMoE(d_model=llm_dim, num_experts=4, top_k=2)
# forward 中:
enc_out = self.reprogramming_layer(enc_out, ...)
enc_out = self.moe_layer(enc_out)  # 新增
```

### 5.4 预期效果

- **模型容量大幅提升**: 4x 专家 = 4x 容量，但只激活 2x
- **计算效率**: 推理时间仅增加约 30%
- **长序列性能**: 特别适合 `pred_len=720` 的长期预测

---

## 六、创新方案五: 频域增强 (Frequency Domain Enhancement)

### 6.1 创新原理

**灵感来源**: [TimesNet (ICLR 2023)](https://arxiv.org/abs/2210.02186) 的 2D 变换思想

TimesNet 将时序转换为 2D 表示:
- 行: 周期内的位置
- 列: 周期数

**核心思想**: 在 Patching 前添加频域分解，分离趋势和周期成分。

### 6.2 架构设计

```
原始时序 x_enc [B, T, N]
        │
        ▼
┌─────────────────────────────────┐
│     FFT 频域分解                │
│  ┌────────────┬────────────┐    │
│  │ 低频成分   │ 高频成分   │    │
│  │ (趋势)     │ (季节性)   │    │
│  └────────────┴────────────┘    │
└─────────────────────────────────┘
        │              │
        ▼              ▼
  [Trend Branch]  [Seasonal Branch]
        │              │
        ▼              ▼
  [Patch + Repr]  [Patch + Repr]
        │              │
        ▼              ▼
    [LLM Trend]   [LLM Seasonal]
        │              │
        └──────┬───────┘
               │
               ▼
        [Feature Fusion]
               │
               ▼
          最终预测
```

### 6.3 代码修改位置

**文件**: `models/TimeLLM.py`

```python
# 新增: 频域分解模块
class FrequencyDecomposition(nn.Module):
    """基于 FFT 的趋势-季节性分解"""
    def __init__(self, top_k_freqs=5):
        super().__init__()
        self.top_k = top_k_freqs

    def forward(self, x):
        # x: [B, T, N]
        B, T, N = x.shape

        # FFT 变换
        x_fft = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(T, device=x.device)

        # 分离低频 (趋势) 和高频 (季节性)
        amplitude = torch.abs(x_fft)

        # Top-K 主要频率
        _, top_indices = torch.topk(amplitude.mean(dim=(0, 2)), self.top_k)

        # 低频成分 (趋势)
        mask_low = torch.zeros_like(x_fft)
        mask_low[:, :3, :] = 1  # 保留前3个低频
        trend = torch.fft.irfft(x_fft * mask_low, n=T, dim=1)

        # 高频成分 (季节性)
        seasonal = x - trend

        return trend, seasonal

# 在 forward 中使用
trend, seasonal = self.freq_decomp(x_enc)

# 分别处理
trend_out = self.forward_branch(trend, "trend")
seasonal_out = self.forward_branch(seasonal, "seasonal")

# 融合
output = self.fusion_layer(trend_out, seasonal_out)
```

### 6.4 预期效果

- **周期性数据集提升显著**: ETTh1/h2 MSE 降低 10-15%
- **可解释性增强**: 可视化趋势和季节性成分
- **计算开销**: FFT 非常高效，额外开销 < 5%

---

## 七、实现优先级建议

| 方案 | 难度 | 预期收益 | 优先级 | 适用场景 |
|------|------|---------|--------|---------|
| **方案二: 变量间注意力** | ⭐⭐ | 8-15% MSE↓ | 🥇 最高 | 多变量预测 |
| **方案五: 频域增强** | ⭐⭐ | 10-15% MSE↓ | 🥈 高 | 周期性数据 |
| **方案一: 多尺度分解** | ⭐⭐⭐ | 5-10% MSE↓ | 🥉 中 | 长序列预测 |
| **方案三: 动态 Prompt** | ⭐⭐⭐ | 10-20% MSE↓ | 中 | 非平稳时序 |
| **方案四: MoE** | ⭐⭐⭐⭐ | 模型容量提升 | 低 | 大规模部署 |

---

## 八、实验设计建议

### 8.1 消融实验

```bash
# 基线
python run_main.py --model TimeLLM --data ETTh1 --pred_len 96

# + 变量间注意力
python run_main.py --model TimeLLM --data ETTh1 --pred_len 96 --use_inter_variate_attn

# + 频域增强
python run_main.py --model TimeLLM --data ETTh1 --pred_len 96 --use_freq_decomp

# + 全部改进
python run_main.py --model TimeLLM --data ETTh1 --pred_len 96 --use_all_improvements
```

### 8.2 对比实验

| 模型 | ETTh1 MSE | ETTh2 MSE | ETTm1 MSE | ETTm2 MSE |
|------|-----------|-----------|-----------|-----------|
| Time-LLM (基线) | 0.375 | 0.288 | 0.302 | 0.175 |
| + Inter-Variate | ? | ? | ? | ? |
| + Frequency | ? | ? | ? | ? |
| + All | ? | ? | ? | ? |
| iTransformer | 0.386 | 0.297 | 0.334 | 0.180 |
| TimeMixer | 0.370 | 0.281 | 0.299 | 0.170 |

---

## 九、参考资源

### 9.1 论文

- [Time-LLM (ICLR 2024)](https://arxiv.org/abs/2310.01728)
- [iTransformer (ICLR 2024)](https://arxiv.org/abs/2310.06625)
- [TimeMixer (ICLR 2024)](https://arxiv.org/abs/2405.14616)
- [Time-MoE (ICLR 2025)](https://arxiv.org/abs/2409.16040)
- [TimesNet (ICLR 2023)](https://arxiv.org/abs/2210.02186)
- [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730)

### 9.2 代码库

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) - 包含 iTransformer, TimesNet, PatchTST 等
- [TimeMixer](https://github.com/kwuking/TimeMixer)
- [Time-MoE](https://github.com/Time-MoE/Time-MoE)

---

## 十、总结

本文档提出了五个基于最新研究的 Time-LLM 改进方案:

1. **多尺度分解混合**: 借鉴 TimeMixer，捕获不同尺度的时序模式
2. **变量间注意力增强**: 借鉴 iTransformer，建模多变量相关性
3. **动态 Prompt 生成**: 提升对非平稳时序的适应能力
4. **稀疏专家混合**: 借鉴 Time-MoE，高效扩展模型容量
5. **频域增强**: 借鉴 TimesNet，分离趋势和季节性成分

**推荐实施路径**:
1. 先实现方案二 (变量间注意力) - 改动小，收益高
2. 再实现方案五 (频域增强) - 对周期性数据效果好
3. 最后尝试方案一 (多尺度分解) - 全面提升

---

**文档更新时间**: 2026-01-02
**参考文献**: ICLR 2023-2025, NeurIPS 2024
**作者**: Zhenda Wang

Sources:
- [Time-Series-Library (GitHub)](https://github.com/thuml/Time-Series-Library)
- [TimeMixer (GitHub)](https://github.com/kwuking/TimeMixer)
- [Time-MoE (GitHub)](https://github.com/Time-MoE/Time-MoE)
- [iTransformer Article](https://www.datasciencewithmarco.com/blog/itransformer-the-latest-breakthrough-in-time-series-forecasting)
- [TimeMixer Article](https://medium.com/the-forecaster/timemixer-exploring-the-latest-model-in-time-series-forecasting-056d9c883f46)
