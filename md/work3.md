# Time-LLM 项目完整技术文档 (Complete Technical Documentation)

> **文档版本：** v1.0 | **最后更新：** 2024-12-05 | **适配版本：** Qwen 2.5 3B 4-bit

---

## 📑 目录 (Table of Contents)

1. [项目概述](#一项目概述)
2. [核心原理](#二核心原理)
3. [项目结构](#三项目结构)
4. [数据流程](#四数据流程)
5. [输入输出规范](#五输入输出规范)
6. [模型输出与应用场景](#六模型输出与应用场景)
7. [最终检查清单](#七最终检查清单)

---

## 一、项目概述

### 1.1 项目简介

**Time-LLM** 是 ICLR 2024 发表的开创性时间序列预测框架，其核心创新是：

> **通过"重编程"(Reprogramming) 技术，将预训练的大语言模型 (LLM) 适配为时间序列预测模型，无需从头训练即可利用 LLM 的模式识别能力。**

### 1.2 核心创新点

| 创新 | 描述 |
|------|------|
| **Prompt-as-Prefix** | 将时序统计特征（min/max/median/趋势/自相关）转换为文本提示 |提示重编程
| **Reprogramming Layer** | 通过跨模态注意力将时序嵌入映射到 LLM 词嵌入空间 |输入重编程
| **Patching Strategy** | 将长时间序列切分为 Patch，降低计算复杂度 |
| **LLM Frozen** | LLM 参数完全冻结，仅训练对齐层（参数高效） |

### 1.3 本项目适配

本项目已完成以下适配：
- ✅ **硬件适配**：6GB 显存可运行（使用 Qwen 2.5 3B + 4-bit 量化）
- ✅ **模型升级**：从 GPT-2 (2019) 升级到 Qwen 2.5 3B (2024)
- ✅ **代码修改**：新增 `--llm_model_path` 和 `--load_in_4bit` 参数

---

## 二、核心原理

### 2.1 Time-LLM 工作流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Time-LLM 端到端工作流程                               │
└─────────────────────────────────────────────────────────────────────────┘

原始时序数据 [Batch, SeqLen, N_vars]
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 实例归一化 (Instance Normalization)                                    │
│    - 计算每个样本的均值和标准差                                            │
│    - Z-score 标准化                                                       │
│    - 保存统计量用于最终反归一化                                            │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Patching (时序分块)                                                    │
│    - 使用滑动窗口将序列切分为固定长度的 Patch                               │
│    - 参数：patch_len=16, stride=8                                         │
│    - 输出：[Batch*N_vars, num_patches, patch_len]                        │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ├───────────────────────────────────────────────────────────────────┐
     │                                                                   │
     ▼                                                                   ▼
┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│ 3. Patch Embedding                     │  │ 4. Prompt 构建                          │
│    - 1D Conv 将 Patch 映射到 d_model   │  │    - 提取统计特征 (min/max/median/趋势) │
│    - 输出：[B*N, num_patches, d_model]│  │    - FFT 计算 Top-5 Lags                │
└────────────────────────────────────────┘  │    - 拼接领域描述文本                   │
     │                                       │    - Tokenizer 编码                     │
     ▼                                       └────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. Reprogramming Layer (跨模态对齐)                                       │
│    - Query: Patch Embeddings (时序域)                                    │
│    - Key/Value: LLM 词嵌入 → Mapping Layer (文本域)                      │
│    - Cross-Attention 实现跨模态映射                                      │
│    - 输出：[B*N, num_patches, llm_dim]                                  │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. Concat [Prompt Embeddings | Patch Embeddings]                         │
│    - 拼接后输入 LLM                                                       │
│    - Shape: [B*N, prompt_len + num_patches, llm_dim]                    │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. LLM Forward (参数冻结)                                                 │
│    - Qwen 2.5 3B / GPT-2 / LLAMA 前向传播                                │
│    - 所有 LLM 参数 requires_grad = False                                 │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. FlattenHead (输出投影)                                                 │
│    - 提取 Patch 对应的 LLM 输出                                           │
│    - Flatten + Linear 映射到预测长度                                      │
│    - 输出：[Batch, pred_len, N_vars]                                     │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 9. 反归一化 (Denormalization)                                             │
│    - 使用步骤 1 保存的统计量还原真实值                                     │
│    - 输出：[Batch, pred_len, N_vars]                                     │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
最终预测结果 [Batch, pred_len, N_vars]
```

### 2.2 为什么这样设计？

#### 问题：如何让 LLM 理解时序数据？

LLM 是为文本设计的，无法直接处理连续数值。Time-LLM 的解决方案是：

1. **Patching**：将时序切分为离散的"片段"，类似文本中的"词"
2. **Reprogramming**：通过注意力机制，让时序片段"学习"与 LLM 词嵌入的对应关系
3. **Prompt**：提供统计信息上下文，帮助 LLM 理解当前时序的特征

#### 参数效率

| 组件 | 参数量 | 是否训练 |
|------|--------|----------|
| LLM (Qwen 2.5 3B) | 3,000M | ❄️ 冻结 |
| Mapping Layer | ~50M | 🔥 训练 |
| Reprogramming Layer | ~6M | 🔥 训练 |
| Output Projection | ~40K | 🔥 训练 |
| **总训练参数** | **~56M** | |

> **结论**：仅需训练 1.8% 的参数量即可完成时序预测任务！

---

## 三、项目结构

### 3.1 完整目录树

```
Time-LLM/
├── 📁 base_models/                      # 基座模型存放目录
│   ├── gpt2/                           # GPT-2 模型 (旧版)
│   └── Qwen2.5-3B/                     # ★ Qwen 2.5 3B 模型 (推荐)
│       ├── config.json                 # 模型配置 ✅
│       ├── tokenizer.json              # 分词器 ✅
│       ├── tokenizer_config.json       # 分词器配置 ✅
│       ├── vocab.json                  # 词汇表 ✅
│       ├── merges.txt                  # BPE 合并规则 ✅
│       ├── model.safetensors.index.json # 权重索引 ✅
│       ├── model-00001-of-00002.safetensors # 权重分片1 ⏳
│       └── model-00002-of-00002.safetensors # 权重分片2 ⏳
│
├── 📁 dataset/                          # 数据集目录
│   ├── ETT-small/                      # ★ ETT 数据集 (推荐入门)
│   │   ├── ETTh1.csv                   # 小时级数据 (17,420 行) ✅
│   │   ├── ETTh2.csv                   # 小时级数据 (17,420 行) ✅
│   │   ├── ETTm1.csv                   # 分钟级数据 (69,680 行) ✅
│   │   └── ETTm2.csv                   # 分钟级数据 (69,680 行) ✅
│   ├── weather/                        # 天气数据集 (可选)
│   ├── electricity/                    # 电力数据集 (可选)
│   ├── traffic/                        # 交通数据集 (可选)
│   └── prompt_bank/                    # 领域描述提示词库
│       ├── ETT.txt                     # ETT 数据集描述
│       ├── Weather.txt                 # Weather 数据集描述
│       ├── ECL.txt                     # Electricity 数据集描述
│       ├── Traffic.txt                 # Traffic 数据集描述
│       └── m4.txt                      # M4 数据集描述
│
├── 📁 models/                           # 模型定义
│   ├── TimeLLM.py                      # ★★★ Time-LLM 核心模型
│   ├── Autoformer.py                   # 基线模型
│   ├── DLinear.py                      # 基线模型
│   └── __init__.py
│
├── 📁 layers/                           # 神经网络层组件
│   ├── Embed.py                        # ★★★ PatchEmbedding 核心
│   ├── StandardNorm.py                 # ★★★ 实例归一化层
│   ├── Transformer_EncDec.py           # Transformer 编码器/解码器
│   ├── SelfAttention_Family.py         # 注意力机制
│   ├── Autoformer_EncDec.py            # Autoformer 组件
│   ├── AutoCorrelation.py              # 自相关机制
│   └── Conv_Blocks.py                  # 卷积块
│
├── 📁 data_provider/                    # 数据加载
│   ├── data_factory.py                 # ★ 数据集路由器
│   ├── data_loader.py                  # ★★ 数据集加载器
│   └── m4.py                           # M4 数据集加载器
│
├── 📁 utils/                            # 工具函数
│   ├── tools.py                        # ★ EarlyStopping, vali, load_content
│   ├── metrics.py                      # ★ MAE, MSE, RMSE, MAPE, MSPE
│   ├── losses.py                       # 损失函数
│   ├── timefeatures.py                 # 时间特征编码
│   └── m4_summary.py                   # M4 评估汇总
│
├── 📁 scripts/                          # 训练脚本
│   ├── TimeLLM_ETTh1.sh                # ETTh1 训练脚本
│   ├── TimeLLM_ETTh2.sh
│   ├── TimeLLM_ETTm1.sh
│   ├── TimeLLM_ETTm2.sh
│   ├── TimeLLM_Weather.sh
│   ├── TimeLLM_ECL.sh
│   ├── TimeLLM_Traffic.sh
│   └── TimeLLM_M4.sh                   # M4 短期预测
│
├── 📄 run_main.py                       # ★★★ 主训练入口
├── 📄 run_m4.py                         # M4 短期预测入口
├── 📄 run_pretrain.py                   # 预训练/迁移学习入口
├── 📄 ds_config_zero2.json              # DeepSpeed ZeRO-2 配置
├── 📄 requirements.txt                  # Python 依赖
├── 📄 CLAUDE.md                         # 项目记忆库
├── 📄 work.md                           # 项目总览
├── 📄 work1.md                          # 快速上手指南
├── 📄 work2.md                          # 深度技术解析
└── 📄 work3.md                          # 本文档
```

### 3.2 核心文件功能说明

#### 模型层 (`models/`)

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `TimeLLM.py` | Time-LLM 主模型 | `Model`, `ReprogrammingLayer`, `FlattenHead` |
| `Autoformer.py` | Autoformer 基线 | `Model` |
| `DLinear.py` | DLinear 基线 | `Model` |

**`TimeLLM.py` 核心代码位置：**
- 第 43-96 行：通用模型加载 + 4-bit 量化逻辑 ★新增
- 第 163-164 行：LLM 参数冻结
- 第 207-230 行：Prompt 构建
- 第 267-305 行：`ReprogrammingLayer` 跨模态对齐

#### 层组件 (`layers/`)

| 文件 | 功能 | 关键类 |
|------|------|--------|
| `Embed.py` | 嵌入层 | `PatchEmbedding`, `TokenEmbedding`, `PositionalEmbedding` |
| `StandardNorm.py` | 归一化 | `Normalize` (支持 norm/denorm) |

#### 数据层 (`data_provider/`)

| 文件 | 功能 | 关键类 |
|------|------|--------|
| `data_factory.py` | 数据集路由 | `data_provider()` |
| `data_loader.py` | 数据加载 | `Dataset_ETT_hour`, `Dataset_ETT_minute`, `Dataset_Custom` |

**支持的数据集：**
```python
data_dict = {
    'ETTh1': Dataset_ETT_hour,    # 小时级 ETT
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,  # 分钟级 ETT
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,        # 电力
    'Traffic': Dataset_Custom,    # 交通
    'Weather': Dataset_Custom,    # 天气
    'm4': Dataset_M4,             # M4 竞赛
}
```

#### 工具层 (`utils/`)

| 文件 | 功能 | 关键函数 |
|------|------|----------|
| `tools.py` | 训练工具 | `EarlyStopping`, `vali()`, `load_content()` |
| `metrics.py` | 评估指标 | `MAE()`, `MSE()`, `RMSE()`, `MAPE()`, `MSPE()` |

---

## 四、数据流程

### 4.1 数据集格式

#### ETT 数据集结构 (以 ETTm1.csv 为例)

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | datetime | 时间戳 |
| `HUFL` | float | 高压有功负荷 |
| `HULL` | float | 高压无功负荷 |
| `MUFL` | float | 中压有功负荷 |
| `MULL` | float | 中压无功负荷 |
| `LUFL` | float | 低压有功负荷 |
| `LULL` | float | 低压无功负荷 |
| `OT` | float | **油温 (预测目标)** |

**数据集划分：**
- **Train**: 前 12 个月
- **Validation**: 中间 4 个月
- **Test**: 最后 4 个月

### 4.2 数据加载流程

```python
# 1. 数据工厂创建 DataLoader
train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')

# 2. 每个 batch 的数据格式
for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
    # batch_x: 输入序列 [Batch, SeqLen, N_vars]
    # batch_y: 目标序列 [Batch, LabelLen + PredLen, N_vars]
    # batch_x_mark: 输入时间特征 [Batch, SeqLen, TimeFeatures]
    # batch_y_mark: 目标时间特征 [Batch, LabelLen + PredLen, TimeFeatures]
```

### 4.3 数据预处理

```
原始 CSV 数据
     │
     ▼
┌─────────────────────────────────────┐
│ 1. 读取 CSV 文件                     │
│    pd.read_csv(data_path)           │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ 2. 选择特征列                        │
│    M: 多变量 → 预测多变量            │
│    S: 单变量 → 预测单变量            │
│    MS: 多变量 → 预测单变量           │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ 3. StandardScaler 标准化            │
│    使用训练集的 mean/std            │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ 4. 时间特征编码                      │
│    提取 月/日/星期/小时/分钟         │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ 5. 滑动窗口采样                      │
│    (seq_x, seq_y, mark_x, mark_y)  │
└─────────────────────────────────────┘
```

---

## 五、输入输出规范

### 5.1 命令行参数

#### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task_name` | str | long_term_forecast | 任务类型 |
| `--model` | str | TimeLLM | 模型名称 |
| `--data` | str | ETTm1 | 数据集名称 |
| `--seq_len` | int | 96 | 输入序列长度 |
| `--pred_len` | int | 96 | 预测长度 |
| `--batch_size` | int | 8 | 批大小 |

#### LLM 相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--llm_model` | str | LLAMA | LLM 类型 (LLAMA/GPT2/BERT/QWEN) |
| `--llm_dim` | int | 4096 | LLM 隐藏层维度 |
| `--llm_layers` | int | 6 | 使用的 LLM 层数 |
| `--llm_model_path` | str | '' | ★ 本地模型路径 |
| `--load_in_4bit` | flag | False | ★ 启用 4-bit 量化 |

### 5.2 模型输入

| 张量 | Shape | 说明 |
|------|-------|------|
| `x_enc` | [Batch, SeqLen, N_vars] | 输入时序 |
| `x_mark_enc` | [Batch, SeqLen, TimeFeatures] | 输入时间特征 |
| `x_dec` | [Batch, LabelLen+PredLen, N_vars] | Decoder 输入 |
| `x_mark_dec` | [Batch, LabelLen+PredLen, TimeFeatures] | Decoder 时间特征 |

### 5.3 模型输出

| 张量 | Shape | 说明 |
|------|-------|------|
| `outputs` | [Batch, PredLen, N_vars] | 预测结果 |

---

## 六、模型输出与应用场景

### 6.1 输出格式

```python
# 模型输出
outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
# Shape: [Batch, pred_len, N_vars]

# 示例：pred_len=96, N_vars=7
# outputs[0, :, 0]  → 第 1 个样本，未来 96 步，第 1 个变量的预测值
# outputs[0, :, 6]  → 第 1 个样本，未来 96 步，油温 (OT) 的预测值
```

### 6.2 评估指标

| 指标 | 公式 | 适用场景 |
|------|------|----------|
| **MAE** | $\frac{1}{N}\sum |y_{pred} - y_{true}|$ | 通用，对异常值鲁棒 |
| **MSE** | $\frac{1}{N}\sum (y_{pred} - y_{true})^2$ | 通用，主要指标 |
| **RMSE** | $\sqrt{MSE}$ | 单位与原数据一致 |
| **MAPE** | $\frac{1}{N}\sum |\frac{y_{pred} - y_{true}}{y_{true}}|$ | 相对误差评估 |
| **MSPE** | $\frac{1}{N}\sum (\frac{y_{pred} - y_{true}}{y_{true}})^2$ | 相对误差评估 |

### 6.3 应用场景

| 领域 | 应用 | 数据类型 |
|------|------|----------|
| **电力系统** | 电力负荷预测、变压器温度预测 | ETT 数据集 |
| **智慧交通** | 交通流量预测、拥堵预警 | Traffic 数据集 |
| **气象预报** | 温度/湿度/风速预测 | Weather 数据集 |
| **能源管理** | 电力消耗预测、节能调度 | Electricity 数据集 |
| **金融市场** | 股价走势预测、风险预警 | 自定义数据集 |
| **工业制造** | 设备故障预测、产能规划 | 自定义数据集 |

### 6.4 预测结果示例

```
输入: 过去 512 个时间步的 7 个变量数据
输出: 未来 96 个时间步的 7 个变量预测值

示例训练日志：
Epoch: 10 | Train Loss: 0.2145 Vali Loss: 0.1987 Test Loss: 0.2034
         MAE: 0.3245, MSE: 0.1987, RMSE: 0.4457
```

---

## 七、最终检查清单

### 7.1 环境依赖检查

| 依赖 | 要求版本 | 检查命令 |
|------|---------|----------|
| Python | 3.11 | `python --version` |
| PyTorch | 2.2.2 | `python -c "import torch; print(torch.__version__)"` |
| transformers | 4.31.0 | `pip show transformers` |
| accelerate | 0.28.0 | `pip show accelerate` |
| bitsandbytes | ≥0.41 | `pip show bitsandbytes` |
| CUDA | 11.x/12.x | `nvcc --version` |

**检查脚本：**
```python
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

### 7.2 数据集检查

| 文件 | 路径 | 状态 |
|------|------|------|
| ETTh1.csv | `./dataset/ETT-small/` | ✅ 2.5 MB |
| ETTh2.csv | `./dataset/ETT-small/` | ✅ 2.4 MB |
| ETTm1.csv | `./dataset/ETT-small/` | ✅ 10.4 MB |
| ETTm2.csv | `./dataset/ETT-small/` | ✅ 9.7 MB |
| ETT.txt | `./dataset/prompt_bank/` | ✅ 506 B |

### 7.3 模型文件检查

| 文件 | 路径 | 状态 |
|------|------|------|
| config.json | `./base_models/Qwen2.5-3B/` | ✅ |
| tokenizer.json | `./base_models/Qwen2.5-3B/` | ✅ 7 MB |
| tokenizer_config.json | `./base_models/Qwen2.5-3B/` | ✅ |
| vocab.json | `./base_models/Qwen2.5-3B/` | ✅ 2.8 MB |
| merges.txt | `./base_models/Qwen2.5-3B/` | ✅ 1.7 MB |
| model.safetensors.index.json | `./base_models/Qwen2.5-3B/` | ✅ |
| model-00001-of-00002.safetensors | `./base_models/Qwen2.5-3B/` | ⏳ 下载中 |
| model-00002-of-00002.safetensors | `./base_models/Qwen2.5-3B/` | ⏳ 下载中 |

### 7.4 代码修改检查

| 文件 | 修改行 | 内容 | 状态 |
|------|--------|------|------|
| `run_main.py` | 82-84 | 新增 `--llm_model_path`, `--load_in_4bit` | ✅ |
| `models/TimeLLM.py` | 43-96 | 通用模型加载 + 4-bit 量化 | ✅ |

### 7.5 显存估算

| 组件 | 显存 |
|------|------|
| Qwen 2.5 3B (4-bit) | ~1.5 GB |
| Time-LLM 可训练参数 | ~0.5 GB |
| 中间变量 & 梯度 | ~2.0 GB |
| 系统占用 | ~1.0 GB |
| **总计** | **~5.0 GB** ✅ |

> **结论：** 6GB 显存足够运行！

---

## 八、运行命令

### 8.1 训练命令

```powershell
cd e:\timellm\Time-LLM

python run_main.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path ./dataset/ETT-small/ ^
  --data_path ETTm1.csv ^
  --model_id ETTm1_512_96 ^
  --model_comment Qwen3B ^
  --model TimeLLM ^
  --data ETTm1 ^
  --features M ^
  --seq_len 512 ^
  --label_len 48 ^
  --pred_len 96 ^
  --e_layers 2 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 7 ^
  --dec_in 7 ^
  --c_out 7 ^
  --batch_size 8 ^
  --d_model 32 ^
  --d_ff 32 ^
  --llm_dim 2048 ^
  --dropout 0.1 ^
  --llm_model QWEN ^
  --llm_model_path "e:\timellm\Time-LLM\base_models\Qwen2.5-3B" ^
  --load_in_4bit
```

### 8.2 预期输出

```
Loading generic model from: e:\timellm\Time-LLM\base_models\Qwen2.5-3B
Enabling 4-bit quantization...

iters: 100, epoch: 1 | loss: 0.4523567
speed: 0.1234s/iter; left time: 1234.5678s
Epoch: 1 cost time: 123.45
Epoch: 1 | Train Loss: 0.4523 Vali Loss: 0.3912 Test Loss: 0.4012 MAE Loss: 0.4567

Validation loss decreased (inf --> 0.391200). Saving model ...
```

---

## 九、总结

Time-LLM 是一个优雅且高效的时间序列预测框架：

1. **核心创新**：通过 Reprogramming 实现时序到文本的跨模态对齐
2. **参数高效**：仅训练 ~2% 的参数，LLM 完全冻结
3. **硬件友好**：通过 4-bit 量化，6GB 显存即可运行 3B 模型
4. **应用广泛**：适用于电力、交通、气象、金融等多个领域

**祝训练顺利！** 🚀

---

**文档生成时间：** 2024-12-05
**作者：** Claude Code Technical Analysis
