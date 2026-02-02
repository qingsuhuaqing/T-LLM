# Time-LLM 完整技术文档

> **Time Series Forecasting by Reprogramming Large Language Models**
>
> 通过重编程大语言模型实现时间序列预测 | ICLR 2024

---

## 目录

- [一、项目概述](#一项目概述)
- [二、项目架构](#二项目架构)
- [三、核心模块详解](#三核心模块详解)
- [四、数据流与维度变化](#四数据流与维度变化)
- [五、命令行参数详解](#五命令行参数详解)
- [六、训练脚本示例](#六训练脚本示例)
- [七、训练过程与指标](#七训练过程与指标)
- [八、常见问题与解决方案](#八常见问题与解决方案)
- [九、创新改进方向](#九创新改进方向)
- [十、参考资料](#十参考资料)

---

## 一、项目概述

### 1.1 项目目标

Time-LLM 是一个将**大语言模型 (LLM)** 应用于**时间序列预测**的创新框架。其核心思想是：

- **冻结 LLM 参数**：不修改 LLM 的预训练权重
- **重编程输入**：将时间序列数据"翻译"成 LLM 能理解的形式
- **利用 LLM 能力**：借助 LLM 强大的序列建模能力进行预测

### 1.2 核心创新点

| 创新点 | 描述 | 优势 |
|--------|------|------|
| **Patching** | 将时序切分为固定长度的块 | 降低计算复杂度，类比文本 Token |
| **Reprogramming** | 通过 Cross-Attention 对齐时序与文本 | 实现跨模态迁移 |
| **Prompt-as-Prefix** | 动态生成包含统计信息的提示词 | 增强可解释性 |
| **冻结 LLM** | 仅训练适配层，LLM 参数不变 | 大幅降低训练成本 |

### 1.3 适用场景

- **长期预测 (Long-term Forecast)**：预测未来 96/192/336/720 个时间步
- **短期预测 (Short-term Forecast)**：M4 竞赛数据集
- **多变量预测 (Multivariate)**：同时预测多个相关变量

---

## 二、项目架构

### 2.1 目录结构

```
Time-LLM/
├── models/                          # 核心模型定义
│   ├── TimeLLM.py                  # ★★★ Time-LLM 主模型
│   ├── Autoformer.py               # 基线模型
│   └── DLinear.py                  # 基线模型
│
├── layers/                          # 神经网络层组件
│   ├── Embed.py                    # ★★★ PatchEmbedding 实现
│   ├── StandardNorm.py             # ★★★ 实例归一化层
│   ├── Transformer_EncDec.py       # Transformer 组件
│   └── SelfAttention_Family.py     # 注意力机制
│
├── data_provider/                   # 数据加载管道
│   ├── data_factory.py             # ★ 数据集路由器
│   ├── data_loader.py              # ★★ 数据加载器
│   └── m4.py                       # M4 数据集加载
│
├── utils/                           # 工具函数
│   ├── tools.py                    # ★ 训练工具
│   ├── metrics.py                  # ★ 评估指标
│   └── timefeatures.py             # 时间特征编码
│
├── dataset/                         # 数据集目录
│   ├── ETT-small/                  # ETT 数据集
│   └── prompt_bank/                # ★★ 领域描述提示词
│
├── scripts/                         # 训练脚本
│   ├── TimeLLM_ETTh1_2.sh          # ETTh1 训练脚本
│   └── TimeLLM_ETTm1_2.sh          # ETTm1 训练脚本
│
├── md/                              # 技术文档
│   ├── work2.md                    # 深度技术解析
│   ├── wenti.md                    # 问题汇总
│   ├── chuangxin.md                # 创新方案
│   └── Trainable-part.md           # 可训练部分解析
│
├── run_main.py                      # ★★★ 主训练入口
├── run_m4.py                        # M4 短期预测入口
└── CLAUDE.md                        # Claude Code 指南
```

### 2.2 模块职责速查

| 模块 | 文件 | 核心职责 |
|------|------|----------|
| **主模型** | `models/TimeLLM.py` | 定义 Time-LLM 完整架构 |
| **Patch 嵌入** | `layers/Embed.py` | 时序切块与嵌入 |
| **归一化** | `layers/StandardNorm.py` | 实例归一化/反归一化 |
| **数据加载** | `data_provider/data_loader.py` | 读取 CSV、切片、标准化 |
| **训练循环** | `run_main.py` | 训练/验证/测试流程 |
| **评估指标** | `utils/metrics.py` | MSE/MAE/RMSE 计算 |

---

## 三、核心模块详解

### 3.1 整体数据流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Time-LLM 完整数据流程                                │
└─────────────────────────────────────────────────────────────────────────┘

原始时序数据 (CSV 文件)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ [1] 数据加载与预处理 (data_provider/data_loader.py)              │
│     - StandardScaler 标准化 (使用训练集均值/方差)                │
│     - 时间特征编码 (月/日/星期/小时)                             │
│     - 滑动窗口切片                                               │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
输入张量: x_enc [B, seq_len, N_vars] = [4, 512, 7]
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ [2] 实例归一化 (layers/StandardNorm.py: Normalize)               │
│     - 计算每个样本的均值和标准差                                  │
│     - Z-score 标准化: (x - mean) / std                          │
│     - 保存统计量用于反归一化                                      │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
归一化数据: x_enc [B, seq_len, N_vars] = [4, 512, 7]
       │
       ├────────────────────────────────────────────┐
       ▼                                            ▼
┌────────────────────────────┐    ┌─────────────────────────────────┐
│ [3] Prompt 构建             │    │ [4] Patching                    │
│ (TimeLLM.py: 276-287)       │    │ (layers/Embed.py: PatchEmbedding)│
│                             │    │                                 │
│ 提取统计特征:               │    │ 1. Padding: 边界填充            │
│ - min, max, median          │    │ 2. Unfold: 滑动窗口切分         │
│ - trend (上升/下降)         │    │ 3. Reshape: 维度重组            │
│ - top-5 lags (FFT)          │    │ 4. Conv1D: Token 嵌入           │
│                             │    │                                 │
│ 拼接领域描述生成 Prompt     │    │ [4, 512, 7] → [28, 64, 32]     │
└────────────────────────────┘    └─────────────────────────────────┘
       │                                            │
       ▼                                            ▼
prompt_embeddings                           enc_out (Patch 嵌入)
[28, ~128, 2048]                           [28, 64, 32]
       │                                            │
       │                                            ▼
       │                          ┌─────────────────────────────────┐
       │                          │ [5] Reprogramming Layer          │
       │                          │ (TimeLLM.py: ReprogrammingLayer) │
       │                          │                                  │
       │                          │ Cross-Attention:                 │
       │                          │ - Q: Patch Embeddings (时序)     │
       │                          │ - K/V: 压缩词表 (文本)           │
       │                          │                                  │
       │                          │ [28, 64, 32] → [28, 64, 2048]   │
       │                          └─────────────────────────────────┘
       │                                            │
       ▼                                            ▼
┌──────────────────────────────────────────────────────────────────┐
│ [6] 拼接 Prompt + Patch Embeddings                               │
│     llama_enc_out = Concat([prompt_embeddings, enc_out], dim=1)  │
│     Shape: [28, ~128+64, 2048] = [28, ~192, 2048]                │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ [7] LLM Forward (冻结的 LLM Backbone)                            │
│     - GPT-2 / LLAMA / Qwen 等                                    │
│     - 参数完全冻结，仅做特征提取                                  │
│     Output: [28, ~192, 2048]                                     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ [8] 截取 + 重塑                                                  │
│     - 截取前 d_ff 维: [28, ~192, 32]                             │
│     - Reshape: [4, 7, 32, ~192]                                  │
│     - 提取 Patch 部分: [4, 7, 32, 64]                            │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ [9] FlattenHead (Output Projection)                              │
│ (TimeLLM.py: FlattenHead)                                        │
│     - Flatten: [4, 7, 32*64] = [4, 7, 2048]                      │
│     - Linear: [4, 7, 2048] → [4, 7, 96]                          │
│     - Permute: [4, 96, 7]                                        │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
预测结果 (归一化空间): [4, 96, 7]
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ [10] 反归一化 (Denormalize)                                      │
│      dec_out = dec_out * stdev + mean                            │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
最终预测: [B, pred_len, N_vars] = [4, 96, 7]
```

---

### 3.2 PatchEmbedding 详解

**代码位置**: `layers/Embed.py` 第 160-186 行

**作用**: 将连续的时间序列切分为固定长度的"块"(Patch)，然后嵌入到低维向量空间。

#### 3.2.1 类定义与参数

```python
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        """
        参数说明:
        - d_model: Patch 嵌入后的维度 (如 32)
        - patch_len: 每个 Patch 的长度 (如 16)
        - stride: 滑动窗口步长 (如 8，表示 50% 重叠)
        - dropout: Dropout 比例
        """
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

        # 边界填充层: 复制最后一个时间步填充 stride 长度
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Token 嵌入: 1D 卷积将 Patch 映射到 d_model 维
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        self.dropout = nn.Dropout(dropout)
```

#### 3.2.2 前向传播流程

```python
def forward(self, x):
    """
    输入: x [B, N_vars, seq_len] = [4, 7, 512]
    输出: (x_embedded, n_vars)
    """
    # 步骤 1: 记录变量数
    n_vars = x.shape[1]  # n_vars = 7

    # 步骤 2: 边界填充
    # [4, 7, 512] → [4, 7, 512 + 8] = [4, 7, 520]
    x = self.padding_patch_layer(x)

    # 步骤 3: 滑动窗口切分
    # unfold(dimension=-1, size=patch_len, step=stride)
    # [4, 7, 520] → [4, 7, num_patches, patch_len]
    # num_patches = (520 - 16) / 8 + 1 = 64
    # [4, 7, 520] → [4, 7, 64, 16]
    x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    # 步骤 4: 维度重组
    # [4, 7, 64, 16] → [4*7, 64, 16] = [28, 64, 16]
    x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

    # 步骤 5: Token 嵌入 (1D 卷积)
    # [28, 64, 16] → [28, 64, d_model] = [28, 64, 32]
    x = self.value_embedding(x)

    return self.dropout(x), n_vars
```

#### 3.2.3 维度变化详解 (以 ETTh1 + seq_len=512 为例)

| 步骤 | 操作 | 输入形状 | 输出形状 | 说明 |
|------|------|----------|----------|------|
| 输入 | - | - | `[4, 7, 512]` | B=4, N_vars=7, seq_len=512 |
| 转置 | permute(0,2,1) | `[4, 7, 512]` | `[4, 512, 7]` | 在 forward 外部执行 |
| 再转置 | permute(0,2,1) | `[4, 512, 7]` | `[4, 7, 512]` | 进入 PatchEmbedding |
| 填充 | ReplicationPad1d | `[4, 7, 512]` | `[4, 7, 520]` | 填充 stride=8 |
| 切分 | unfold | `[4, 7, 520]` | `[4, 7, 64, 16]` | 64 个 Patch，每个长度 16 |
| 重组 | reshape | `[4, 7, 64, 16]` | `[28, 64, 16]` | 合并 B 和 N_vars |
| 嵌入 | Conv1d | `[28, 64, 16]` | `[28, 64, 32]` | 映射到 d_model=32 |

#### 3.2.4 关键参数计算公式

```python
# Patch 数量计算
num_patches = (seq_len + stride - patch_len) // stride + 1
# 例: (512 + 8 - 16) // 8 + 1 = 504 // 8 + 1 = 63 + 1 = 64

# 或使用原始公式 (含 padding)
num_patches = (seq_len - patch_len) // stride + 2
# 例: (512 - 16) // 8 + 2 = 62 + 2 = 64
```

---

### 3.3 Mapping Layer 详解

**代码位置**: `models/TimeLLM.py` 第 233-236 行

**作用**: 将 LLM 的大词表压缩为小型可学习的"虚拟词表"。

#### 3.3.1 初始化

```python
# 获取 LLM 词嵌入矩阵
self.word_embeddings = self.llm_model.get_input_embeddings().weight
# GPT-2: [50257, 768]
# Qwen 2.5 3B: [151936, 2048]

self.vocab_size = self.word_embeddings.shape[0]  # 50257 或 151936
self.num_tokens = 1000  # 压缩后的虚拟词表大小 (硬编码)

# 线性层: 词表压缩
self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
# GPT-2: Linear(50257, 1000)
# Qwen: Linear(151936, 1000)
```

#### 3.3.2 压缩过程

```python
# TimeLLM.py 第 294 行
source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

# 详细过程:
# 1. 原始词嵌入: [vocab_size, llm_dim] = [50257, 768]
# 2. 转置: [768, 50257]
# 3. Linear 映射: [768, 50257] @ [50257, 1000] = [768, 1000]
# 4. 再转置: [1000, 768]
# 输出: source_embeddings [1000, llm_dim]
```

#### 3.3.3 为什么需要压缩？

| 原因 | 说明 |
|------|------|
| **计算效率** | Cross-Attention 复杂度 O(L×S)，S 从 50257 降为 1000 |
| **参数效率** | 学习 1000 个与任务相关的语义原型即可 |
| **领域专用** | 时序预测不需要完整的自然语言词汇 |

---

### 3.4 ReprogrammingLayer 详解

**代码位置**: `models/TimeLLM.py` 第 325-363 行

**作用**: 通过 Cross-Attention 将时序 Patch 嵌入"翻译"到 LLM 的嵌入空间。

#### 3.4.1 类定义

```python
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        """
        参数说明:
        - d_model: Patch 嵌入维度 (如 32)
        - n_heads: 注意力头数 (如 8)
        - d_keys: 每个头的维度 (默认 d_model // n_heads = 4)
        - d_llm: LLM 隐藏层维度 (如 2048)
        """
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)  # d_keys = 32 // 8 = 4

        # Query: 来自时序 Patch
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # Linear(32, 4*8) = Linear(32, 32)

        # Key: 来自压缩词表
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        # Linear(2048, 32)

        # Value: 来自压缩词表
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        # Linear(2048, 32)

        # 输出投影: 映射回 LLM 维度
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        # Linear(32, 2048)

        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)
```

#### 3.4.2 前向传播

```python
def forward(self, target_embedding, source_embedding, value_embedding):
    """
    参数:
    - target_embedding: Patch 嵌入 [B*N, num_patches, d_model] = [28, 64, 32]
    - source_embedding: 压缩词表 [num_tokens, d_llm] = [1000, 2048]
    - value_embedding: 同 source_embedding
    """
    B, L, _ = target_embedding.shape  # B=28, L=64
    S, _ = source_embedding.shape      # S=1000
    H = self.n_heads                   # H=8

    # Query 投影: [28, 64, 32] → [28, 64, 32] → [28, 64, 8, 4]
    target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)

    # Key 投影: [1000, 2048] → [1000, 32] → [1000, 8, 4]
    source_embedding = self.key_projection(source_embedding).view(S, H, -1)

    # Value 投影: [1000, 2048] → [1000, 32] → [1000, 8, 4]
    value_embedding = self.value_projection(value_embedding).view(S, H, -1)

    # Cross-Attention
    out = self.reprogramming(target_embedding, source_embedding, value_embedding)
    # out: [28, 64, 8, 4]

    out = out.reshape(B, L, -1)  # [28, 64, 32]

    return self.out_projection(out)  # [28, 64, 2048]
```

#### 3.4.3 注意力计算

```python
def reprogramming(self, target_embedding, source_embedding, value_embedding):
    B, L, H, E = target_embedding.shape  # [28, 64, 8, 4]

    scale = 1. / sqrt(E)  # 1 / sqrt(4) = 0.5

    # 计算注意力分数
    # Q: [B, L, H, E] = [28, 64, 8, 4]
    # K: [S, H, E] = [1000, 8, 4]
    # einsum("blhe,she->bhls"): [28, 8, 64, 1000]
    scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

    # Softmax 归一化
    A = self.dropout(torch.softmax(scale * scores, dim=-1))
    # A: [28, 8, 64, 1000]

    # 加权聚合 Value
    # V: [S, H, E] = [1000, 8, 4]
    # einsum("bhls,she->blhe"): [28, 64, 8, 4]
    reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

    return reprogramming_embedding
```

#### 3.4.4 维度变化汇总

| 步骤 | 张量 | 形状 | 说明 |
|------|------|------|------|
| 输入 Q | target_embedding | `[28, 64, 32]` | Patch 嵌入 |
| 输入 K/V | source_embedding | `[1000, 2048]` | 压缩词表 |
| Q 投影 | - | `[28, 64, 8, 4]` | 分头 |
| K 投影 | - | `[1000, 8, 4]` | 分头 |
| V 投影 | - | `[1000, 8, 4]` | 分头 |
| 注意力 | scores | `[28, 8, 64, 1000]` | Q×K^T |
| 聚合 | out | `[28, 64, 8, 4]` | Attention×V |
| 合并 | out | `[28, 64, 32]` | 合并头 |
| 输出投影 | out | `[28, 64, 2048]` | 映射到 LLM 维度 |

---

### 3.5 FlattenHead 详解

**代码位置**: `models/TimeLLM.py` 第 15-27 行

**作用**: 将 LLM 的高维输出映射为时间序列预测值。

#### 3.5.1 类定义

```python
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        参数说明:
        - n_vars: 变量数量 (如 7)
        - nf: 输入特征维度 = d_ff * num_patches (如 32 * 64 = 2048)
        - target_window: 预测长度 = pred_len (如 96)
        - head_dropout: Dropout 比例
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # 展平最后两个维度
        self.linear = nn.Linear(nf, target_window)  # Linear(2048, 96)
        self.dropout = nn.Dropout(head_dropout)
```

#### 3.5.2 前向传播

```python
def forward(self, x):
    """
    输入: x [B, N_vars, d_ff, num_patches] = [4, 7, 32, 64]
    输出: [B, N_vars, pred_len] = [4, 7, 96]
    """
    # 步骤 1: Flatten
    # [4, 7, 32, 64] → [4, 7, 32*64] = [4, 7, 2048]
    x = self.flatten(x)

    # 步骤 2: Linear
    # [4, 7, 2048] → [4, 7, 96]
    x = self.linear(x)

    # 步骤 3: Dropout
    x = self.dropout(x)

    return x  # [4, 7, 96]
```

#### 3.5.3 参数计算

```python
# 输入维度
nf = d_ff * num_patches
# 例: 32 * 64 = 2048

# 参数量
weight_params = nf * pred_len = 2048 * 96 = 196,608
bias_params = pred_len = 96
total_params = 196,704 ≈ 200K
```

---

### 3.6 Normalize 层详解

**代码位置**: `layers/StandardNorm.py`

**作用**: 实例级归一化与反归一化。

#### 3.6.1 类定义

```python
class Normalize(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        """
        参数说明:
        - num_features: 特征数量 (即变量数 N_vars)
        - eps: 数值稳定性常数
        - affine: 是否使用可学习的缩放和偏移
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
```

#### 3.6.2 归一化过程

```python
def forward(self, x, mode):
    """
    x: [B, seq_len, N_vars] = [4, 512, 7]
    mode: 'norm' 或 'denorm'
    """
    if mode == 'norm':
        # 计算统计量 (在时间维度上)
        self._get_statistics(x)
        x = self._normalize(x)
    elif mode == 'denorm':
        x = self._denormalize(x)
    return x

def _get_statistics(self, x):
    """计算均值和标准差"""
    dim2reduce = tuple(range(1, x.ndim - 1))  # dim=(1,) 即时间维度
    self.mean = torch.mean(x, dim=dim2reduce, keepdim=True)
    # mean: [4, 1, 7]
    self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True) + self.eps)
    # stdev: [4, 1, 7]

def _normalize(self, x):
    """Z-score 归一化"""
    x = x - self.mean  # 减均值
    x = x / self.stdev  # 除标准差
    return x

def _denormalize(self, x):
    """反归一化"""
    x = x * self.stdev  # 乘标准差
    x = x + self.mean   # 加均值
    return x
```

---

### 3.7 Prompt 构建详解

**代码位置**: `models/TimeLLM.py` 第 270-291 行

#### 3.7.1 统计特征提取

```python
# 第 264-268 行: 计算统计量
min_values = torch.min(x_enc, dim=1)[0]      # [B*N, 1]
max_values = torch.max(x_enc, dim=1)[0]      # [B*N, 1]
medians = torch.median(x_enc, dim=1).values  # [B*N, 1]
lags = self.calcute_lags(x_enc)              # [B*N, 5] Top-5 自相关 lags
trends = x_enc.diff(dim=1).sum(dim=1)        # [B*N, 1] 趋势 (差分求和)
```

#### 3.7.2 Prompt 构建

```python
# 第 276-287 行
prompt_ = (
    f"<|start_prompt|>Dataset description: {self.description}"
    # 领域描述，从 prompt_bank/{dataset}.txt 加载

    f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
    # 任务描述

    "Input statistics: "
    f"min value {min_values_str}, "
    f"max value {max_values_str}, "
    f"median value {median_values_str}, "
    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
    f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
    # 统计信息
)
```

#### 3.7.3 示例 Prompt

```
<|start_prompt|>Dataset description: The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. This dataset consists of 2 years data from two separated counties in China...

Task description: forecast the next 96 steps given the previous 512 steps information;

Input statistics: min value -1.234, max value 2.567, median value 0.345, the trend of input is upward, top 5 lags are : [24, 48, 72, 96, 168]<|<end_prompt>|>
```

#### 3.7.4 FFT 自相关计算 (calcute_lags)

```python
def calcute_lags(self, x_enc):
    """
    计算 Top-5 自相关 lags
    x_enc: [B*N, seq_len, 1]
    """
    # FFT 变换
    q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)

    # 自相关 = FFT(x) * conj(FFT(x))
    res = q_fft * torch.conj(k_fft)

    # 逆 FFT 得到自相关序列
    corr = torch.fft.irfft(res, dim=-1)

    # 取均值后找 Top-5 位置
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, self.top_k, dim=-1)

    return lags  # [B*N, 5]
```

---

## 四、数据流与维度变化

### 4.1 完整维度变化表 (以 ETTh1 + Qwen 2.5 3B 为例)

**配置参数:**
- `batch_size = 4`
- `seq_len = 512`
- `pred_len = 96`
- `N_vars = 7` (enc_in)
- `patch_len = 16`
- `stride = 8`
- `d_model = 32`
- `d_ff = 32`
- `llm_dim = 2048`
- `n_heads = 8`
- `num_tokens = 1000`

| 阶段 | 变量名 | 形状 | 维度说明 | 代码位置 |
|------|--------|------|----------|----------|
| 输入 | `batch_x` | `[4, 512, 7]` | B, seq_len, N_vars | run_main.py:189 |
| 归一化后 | `x_enc` | `[4, 512, 7]` | 同上 | TimeLLM.py:259 |
| 转置 | `x_enc` | `[4, 7, 512]` | B, N_vars, seq_len | TimeLLM.py:262 |
| Patching 后 | `enc_out` | `[28, 64, 32]` | B*N, num_patches, d_model | Embed.py:185 |
| Reprogramming 后 | `enc_out` | `[28, 64, 2048]` | B*N, num_patches, llm_dim | TimeLLM.py:299 |
| Prompt 嵌入 | `prompt_embeddings` | `[28, ~128, 2048]` | B*N, prompt_len, llm_dim | TimeLLM.py:292 |
| 拼接后 | `llama_enc_out` | `[28, ~192, 2048]` | B*N, total_len, llm_dim | TimeLLM.py:300 |
| LLM 输出 | `dec_out` | `[28, ~192, 2048]` | 同上 | TimeLLM.py:301 |
| 截取 d_ff | `dec_out` | `[28, ~192, 32]` | B*N, total_len, d_ff | TimeLLM.py:302 |
| Reshape | `dec_out` | `[4, 7, 32, ~192]` | B, N, d_ff, total_len | TimeLLM.py:305 |
| 提取 Patch | `dec_out` | `[4, 7, 32, 64]` | B, N, d_ff, num_patches | TimeLLM.py:308 |
| Flatten | - | `[4, 7, 2048]` | B, N, d_ff*num_patches | TimeLLM.py:24 |
| Linear | - | `[4, 7, 96]` | B, N, pred_len | TimeLLM.py:25 |
| Permute | `dec_out` | `[4, 96, 7]` | B, pred_len, N | TimeLLM.py:309 |
| 反归一化 | `dec_out` | `[4, 96, 7]` | 最终预测 | TimeLLM.py:311 |

### 4.2 关键变量计算公式

```python
# Patch 数量
num_patches = (seq_len - patch_len) // stride + 2
            = (512 - 16) // 8 + 2
            = 62 + 2
            = 64

# FlattenHead 输入维度
head_nf = d_ff * num_patches
        = 32 * 64
        = 2048

# 展平后的 Batch 大小
B_N = batch_size * N_vars
    = 4 * 7
    = 28
```

### 4.3 变量名含义速查

| 变量名 | 全称 | 含义 |
|--------|------|------|
| `B` | Batch size | 批大小，每次处理的样本数 |
| `N` / `N_vars` | Number of variables | 变量数量，时序的特征维度 |
| `T` / `seq_len` | Sequence length | 输入序列长度 |
| `pred_len` | Prediction length | 预测长度/预测步数 |
| `d_model` | Model dimension | Patch 嵌入维度 |
| `d_ff` | Feed-forward dimension | 前馈网络维度 |
| `llm_dim` / `d_llm` | LLM dimension | LLM 隐藏层维度 |
| `n_heads` / `H` | Number of heads | 注意力头数 |
| `num_patches` / `L` | Number of patches | Patch 数量 |
| `patch_len` | Patch length | 每个 Patch 的长度 |
| `stride` | Stride | 滑动窗口步长 |
| `vocab_size` | Vocabulary size | LLM 原始词表大小 |
| `num_tokens` / `S` | Number of tokens | 压缩后虚拟词表大小 |

---

## 五、命令行参数详解

### 5.1 基础配置参数

| 参数 | 类型 | 默认值 | 说明 | 代码对应 |
|------|------|--------|------|----------|
| `--task_name` | str | `long_term_forecast` | 任务类型 | run_main.py:30 |
| `--is_training` | int | 1 | 是否训练 (1=训练, 0=测试) | run_main.py:32 |
| `--model_id` | str | - | 实验标识符 | run_main.py:33 |
| `--model_comment` | str | - | 模型备注，用于保存路径 | run_main.py:34 |
| `--model` | str | `TimeLLM` | 模型名称 | run_main.py:35 |
| `--seed` | int | 2021 | 随机种子 | run_main.py:37 |

### 5.2 数据配置参数

| 参数 | 类型 | 默认值 | 说明 | 代码对应 |
|------|------|--------|------|----------|
| `--data` | str | `ETTm1` | 数据集名称 | data_factory.py:4-13 |
| `--root_path` | str | `./dataset` | 数据根目录 | run_main.py:41 |
| `--data_path` | str | `ETTh1.csv` | 数据文件名 | run_main.py:42 |
| `--features` | str | `M` | M=多变量, S=单变量, MS=多变量预测单变量 | data_loader.py:60-64 |
| `--target` | str | `OT` | 目标变量名 (用于 S/MS) | data_loader.py:64 |
| `--freq` | str | `h` | 时间频率 (h=小时, t=分钟, d=天) | data_loader.py:82-83 |
| `--checkpoints` | str | `./checkpoints/` | 模型保存路径 | run_main.py:53 |

### 5.3 序列长度参数

| 参数 | 类型 | 默认值 | 说明 | 计算示例 |
|------|------|--------|------|----------|
| `--seq_len` | int | 96 | 输入序列长度 | 历史 512 个时间步 |
| `--label_len` | int | 48 | 解码器起始 token 长度 | 用于 Teacher Forcing |
| `--pred_len` | int | 96 | 预测长度 | 预测未来 96 个时间步 |

**序列关系图:**
```
|<------ seq_len (512) ------>|
|__________________|__________|____________|
   历史输入        label_len   pred_len
                  (48)        (96)
```

### 5.4 模型结构参数

| 参数 | 类型 | 默认值 | 说明 | 代码对应 |
|------|------|--------|------|----------|
| `--enc_in` | int | 7 | 编码器输入维度 (变量数) | TimeLLM.py:249 |
| `--dec_in` | int | 7 | 解码器输入维度 | - |
| `--c_out` | int | 7 | 输出维度 | - |
| `--d_model` | int | 16 | Patch 嵌入维度 | TimeLLM.py:231 |
| `--n_heads` | int | 8 | 注意力头数 | TimeLLM.py:238 |
| `--d_ff` | int | 32 | 前馈网络维度 | TimeLLM.py:37 |
| `--e_layers` | int | 2 | 编码器层数 (未使用) | - |
| `--d_layers` | int | 1 | 解码器层数 (未使用) | - |
| `--dropout` | float | 0.1 | Dropout 比例 | TimeLLM.py:228 |
| `--patch_len` | int | 16 | Patch 长度 | TimeLLM.py:40 |
| `--stride` | int | 8 | Patch 步长 | TimeLLM.py:41 |

### 5.5 LLM 配置参数

| 参数 | 类型 | 默认值 | 说明 | 代码对应 |
|------|------|--------|------|----------|
| `--llm_model` | str | `LLAMA` | LLM 类型 (LLAMA/GPT2/BERT/QWEN) | TimeLLM.py:43-211 |
| `--llm_dim` | int | 4096 | LLM 隐藏层维度 | TimeLLM.py:39 |
| `--llm_layers` | int | 6 | 使用的 LLM 层数 | TimeLLM.py:60 |
| `--llm_model_path` | str | `` | 本地模型路径 | TimeLLM.py:43-96 |
| `--load_in_4bit` | flag | False | 启用 4-bit 量化 | TimeLLM.py:50-57 |
| `--prompt_domain` | int | 0 | 是否使用领域提示词 (1=是) | TimeLLM.py:223-226 |

**LLM 维度对照表:**

| LLM 模型 | llm_dim | 参数量 | 显存需求 (FP16) |
|----------|---------|--------|-----------------|
| GPT-2 | 768 | 124M | ~500 MB |
| BERT-base | 768 | 110M | ~500 MB |
| LLaMA-7B | 4096 | 7B | ~14 GB |
| Qwen 2.5 3B | 2048 | 3B | ~6 GB |
| Qwen 2.5 3B (4-bit) | 2048 | 3B | ~1.5 GB |

### 5.6 训练配置参数

| 参数 | 类型 | 默认值 | 说明 | 代码对应 |
|------|------|--------|------|----------|
| `--num_workers` | int | 10 | DataLoader 工作进程数 | data_factory.py:62 |
| `--itr` | int | 1 | 实验重复次数 | run_main.py:89 |
| `--train_epochs` | int | 10 | 训练轮数 | run_main.py:90 |
| `--batch_size` | int | 32 | 训练批大小 | run_main.py:92 |
| `--patience` | int | 10 | Early Stopping 耐心值 | run_main.py:94 |
| `--learning_rate` | float | 0.0001 | 学习率 | run_main.py:95 |
| `--loss` | str | `MSE` | 损失函数 | run_main.py:97 |
| `--lradj` | str | `type1` | 学习率调整策略 | tools.py:11-35 |
| `--use_amp` | flag | False | 混合精度训练 | run_main.py:100 |

### 5.7 参数在代码中的使用示例

```python
# run_main.py 中的参数解析
parser = argparse.ArgumentParser(description='Time-LLM')
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=96)
args = parser.parse_args()

# TimeLLM.py 中的使用
class Model(nn.Module):
    def __init__(self, configs):
        self.pred_len = configs.pred_len      # 使用 args.pred_len
        self.seq_len = configs.seq_len        # 使用 args.seq_len
        self.d_ff = configs.d_ff              # 使用 args.d_ff

        # 计算 Patch 数量
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)

        # 计算 FlattenHead 输入维度
        self.head_nf = self.d_ff * self.patch_nums

        # 初始化 FlattenHead
        self.output_projection = FlattenHead(
            configs.enc_in,      # n_vars
            self.head_nf,        # nf = d_ff * num_patches
            self.pred_len        # target_window
        )
```

---

## 六、训练脚本示例

### 6.1 完整训练命令 (Qwen 2.5 3B + 4-bit)

```bash
#!/bin/bash
# 文件: scripts/TimeLLM_ETTh1_2.sh

cd /mnt/e/timellm/Time-LLM

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model_comment Qwen3B_fast \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 4 \
  --d_model 32 \
  --n_heads 8 \
  --d_ff 32 \
  --llm_dim 2048 \
  --llm_layers 6 \
  --num_workers 2 \
  --prompt_domain 1 \
  --train_epochs 3 \
  --itr 1 \
  --dropout 0.1 \
  --llm_model QWEN \
  --llm_model_path /mnt/e/timellm/Time-LLM/base_models/Qwen2.5-3B \
  --load_in_4bit
```

### 6.2 参数分组说明

```bash
# ========== 任务配置 ==========
--task_name long_term_forecast    # 长期预测任务
--is_training 1                   # 训练模式
--model TimeLLM                   # 使用 Time-LLM 模型

# ========== 数据配置 ==========
--root_path ./dataset/ETT-small/  # 数据目录
--data_path ETTh1.csv             # 数据文件
--data ETTh1                      # 数据集类型
--features M                      # 多变量预测

# ========== 序列配置 ==========
--seq_len 512                     # 输入 512 个时间步
--label_len 48                    # 解码器起始 48 步
--pred_len 96                     # 预测 96 个时间步

# ========== 模型配置 ==========
--enc_in 7                        # 7 个输入变量
--d_model 32                      # Patch 嵌入维度
--d_ff 32                         # 前馈网络维度
--n_heads 8                       # 注意力头数

# ========== LLM 配置 ==========
--llm_model QWEN                  # 使用 Qwen 模型
--llm_dim 2048                    # Qwen 隐藏层维度
--llm_layers 6                    # 使用 6 层 LLM
--llm_model_path .../Qwen2.5-3B   # 本地模型路径
--load_in_4bit                    # 启用 4-bit 量化

# ========== 训练配置 ==========
--batch_size 4                    # 批大小 4
--train_epochs 3                  # 训练 3 轮
--prompt_domain 1                 # 使用领域提示词
--num_workers 2                   # 2 个数据加载进程
```

### 6.3 6GB 显存配置建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--batch_size` | 2-4 | 必须保持较小 |
| `--seq_len` | 96-512 | 512 为上限 |
| `--llm_layers` | 4-6 | 不宜过多 |
| `--d_model` | 16-32 | 保持较小 |
| `--d_ff` | 16-32 | 保持较小 |
| `--load_in_4bit` | 启用 | 必须使用 4-bit 量化 |

---

## 七、训练过程与指标

### 7.1 训练流程

```python
# run_main.py 训练循环 (第 179-267 行)
for epoch in range(args.train_epochs):
    model.train()

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        # 1. 前向传播
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # 2. 计算损失
        loss = criterion(outputs, batch_y)  # MSE Loss

        # 3. 反向传播
        accelerator.backward(loss)
        model_optim.step()

        # 4. 每 100 次迭代打印
        if (i + 1) % 100 == 0:
            print(f"iters: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

    # 5. 验证与测试
    vali_loss, vali_mae = vali(model, vali_loader)
    test_loss, test_mae = vali(model, test_loader)

    # 6. Early Stopping
    early_stopping(vali_loss, model, path)
```

### 7.2 训练输出示例

```
Loading checkpoint shards: 100%|████████████| 2/2 [00:10<00:00, 5.35s/it]

iters: 100, epoch: 1 | loss: 0.2009771
        speed: 1.1923s/iter; left time: 708333.2633s
iters: 200, epoch: 1 | loss: 1.1028175
        speed: 1.1227s/iter; left time: 666836.8862s
...

Epoch: 1 cost time: 3600.45
Epoch: 1 | Train Loss: 0.4523 Vali Loss: 0.3912 Test Loss: 0.4012 MAE Loss: 0.4567

EarlyStopping counter: 0 out of 10
Validation loss decreased (inf --> 0.391200). Saving model ...

Epoch: 2 cost time: 3542.12
Epoch: 2 | Train Loss: 0.3821 Vali Loss: 0.3654 Test Loss: 0.3701 MAE Loss: 0.4321
...

Epoch: 10 | Train Loss: 0.1987 Vali Loss: 0.2012 Test Loss: 0.2034 MAE Loss: 0.3678
```

### 7.3 评估指标说明

| 指标 | 公式 | 含义 | 代码位置 |
|------|------|------|----------|
| **MSE** | $\frac{1}{N}\sum(y_{pred} - y_{true})^2$ | 均方误差 | metrics.py:18-19 |
| **MAE** | $\frac{1}{N}\sum|y_{pred} - y_{true}|$ | 平均绝对误差 | metrics.py:14-15 |
| **RMSE** | $\sqrt{MSE}$ | 均方根误差 | metrics.py:22-23 |
| **MAPE** | $\frac{1}{N}\sum|\frac{y_{pred} - y_{true}}{y_{true}}|$ | 平均绝对百分比误差 | metrics.py:26-27 |

### 7.4 Checkpoint 保存

**保存位置:**
```
checkpoints/
└── long_term_forecast_ETTh1_512_96_TimeLLM_ETTh1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-Qwen3B_fast/
    └── checkpoint
```

**保存内容 (仅可训练参数):**
- `patch_embedding.*`: PatchEmbedding 参数 (~800)
- `mapping_layer.*`: Mapping Layer 参数 (~150M for Qwen)
- `reprogramming_layer.*`: Reprogramming Layer 参数 (~6M)
- `output_projection.*`: FlattenHead 参数 (~200K)

**文件大小:** 约 200-600 MB (取决于 LLM 词表大小)

---

## 八、常见问题与解决方案

### 8.1 环境问题

| 问题 | 解决方案 |
|------|----------|
| `KeyError: 'qwen2'` | `pip install "transformers>=4.40.0" --upgrade` |
| `CUDA_HOME does not exist` | `pip uninstall deepspeed -y` |
| `DeepSpeed is not installed` | 修改 run_main.py 第 107 行，移除 `deepspeed_plugin` |

### 8.2 数据类型问题

| 问题 | 解决方案 |
|------|----------|
| `Input type mismatch (BFloat16 vs Float32)` | 修改 TimeLLM.py 第 297-298 行，使用 `.float()` 转换 |

### 8.3 显存不足 (OOM)

**调整优先级:**
1. 降低 `--batch_size` (4 → 2)
2. 减少 `--llm_layers` (6 → 4)
3. 缩短 `--seq_len` (512 → 256)
4. 降低 `--d_ff` (32 → 16)

### 8.4 详细问题文档

请参考 `md/wenti.md` 获取完整的问题汇总和解决方案。

---

## 九、创新改进方向

### 9.1 已提出的创新方案

| 方案 | 灵感来源 | 预期收益 | 难度 |
|------|----------|----------|------|
| **多尺度分解** | TimeMixer | 5-10% MSE↓ | ⭐⭐⭐ |
| **变量间注意力** | iTransformer | 8-15% MSE↓ | ⭐⭐ |
| **动态 Prompt** | AutoTimes | 10-20% MSE↓ | ⭐⭐⭐ |
| **稀疏专家混合** | Time-MoE | 模型容量提升 | ⭐⭐⭐⭐ |
| **频域增强** | TimesNet | 10-15% MSE↓ | ⭐⭐ |

### 9.2 推荐实施路径

1. **首先**: 实现变量间注意力 (改动小，收益高)
2. **其次**: 实现频域增强 (对周期性数据效果好)
3. **最后**: 实现多尺度分解 (全面提升)

### 9.3 详细创新文档

请参考 `md/chuangxin.md` 获取完整的创新方案设计和代码示例。

---

## 十、参考资料

### 10.1 论文

- **Time-LLM**: [Time Series Forecasting by Reprogramming Large Language Models](https://arxiv.org/abs/2310.01728) (ICLR 2024)
- **iTransformer**: [Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625) (ICLR 2024)
- **TimeMixer**: [TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://arxiv.org/abs/2405.14616) (ICLR 2024)
- **PatchTST**: [A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730) (ICLR 2023)

### 10.2 代码库

- **Time-LLM 原始仓库**: https://github.com/KimMeen/Time-LLM
- **Time-Series-Library**: https://github.com/thuml/Time-Series-Library
- **数据集下载**: [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

### 10.3 项目文档

| 文档 | 内容 |
|------|------|
| `README.md` | 本文档，完整项目指南 |
| `CLAUDE.md` | Claude Code 快速参考 |
| `md/work2.md` | 深度技术解析 |
| `md/wenti.md` | 问题汇总 |
| `md/chuangxin.md` | 创新方案 |
| `md/Trainable-part.md` | 可训练部分详解 |
| `md/mingling.md` | 命令参数详解 |

---

## 附录: 可训练参数汇总

### A.1 参数量统计

| 组件 | 形状 | 参数量 | 占比 |
|------|------|--------|------|
| **PatchEmbedding** | Conv1d(16, 32, 3) | ~1,500 | <0.01% |
| **Mapping Layer** | Linear(151936, 1000) | ~152M | 96% |
| **Reprogramming Layer** | 4 个 Linear | ~6M | 4% |
| **FlattenHead** | Linear(2048, 96) | ~200K | 0.1% |
| **总计 (Qwen 2.5 3B)** | - | **~158M** | 100% |

### A.2 冻结参数

| 组件 | 参数量 | 说明 |
|------|--------|------|
| **Qwen 2.5 3B (4-bit)** | 3B | 完全冻结，仅做特征提取 |

---

**文档版本**: v2.0
**最后更新**: 2026-01-09
**适用环境**: WSL + NVIDIA 6GB GPU + Qwen 2.5 3B (4-bit)
**作者**: Zhenda Wang 
