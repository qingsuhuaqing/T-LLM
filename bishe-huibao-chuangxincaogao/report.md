# 研究生毕业设计开题报告

## 课题名称：基于大模型语义融合的时间序列预测模型设计与实现

---

## 一、对课题任务的理解

### 1.1 任务书核心要求

根据导师下发的任务书，本课题的主要任务包括：

1. **学习掌握大模型在时间序列预测中的应用**，研究大模型适配时序预测任务的关键技术，复现 Time-LLM 论文
2. **在 Time-LLM 论文的基础上，设计优化的模块**，分析优化后的模型在长序列、多变量数据集下的优势与瓶颈
3. 完成上述内容并撰写毕设论文

### 1.2 我对课题的理解

在深入阅读 Time-LLM 论文和相关文献后，我对这个课题有了较为清晰的认识。

**核心问题**：传统的时序预测模型从零开始训练，难以学习到足够丰富的序列模式；而大语言模型（LLM）通过海量文本预训练获得了强大的模式识别能力。Time-LLM 的创新之处在于通过"重编程"（Reprogramming）技术，在保持 LLM 权重冻结的前提下，将时序数据"翻译"成 LLM 能理解的表示，从而借用 LLM 的能力进行预测。

**我的理解**：这其实是一种跨模态知识迁移的思路。Time-LLM 提出了两种核心机制：

1. **输入重编程（Input Reprogramming）**：它不直接把数字扔给 LLM，而是使用 Patching（分块）和 Projection（投影）层，把时间序列切片后通过线性层映射到 LLM 的词向量空间。这样 LLM 会认为这些时序数据是某种"特殊的文本特征"。

2. **提示重编程（Prompt Reprogramming）**：在输入前加上特定的自然语言提示，比如数据的统计特征（均值、方差、趋势等）被转化为文本描述。这样做的目的是激活 LLM 内部原本就有的"趋势识别"和"模式匹配"能力。

我认为这个思路很有启发性。一方面，输入重编程利用了 LLM 对 NLP 的优势；另一方面，提示重编程让模型的预测有迹可循，增强了可解释性。而且我认为这也是多模态应用的一种可能——用文字信息详细描述其他模态的内容，通过模态间的映射实现融合。

---

## 二、已完成的工作

### 2.1 环境搭建与论文复现

在项目初期，我完成了以下基础工作：

**（1）实验环境配置**

由于我的 GPU 显存只有 6GB（GTX 1660 Ti），无法直接运行原论文使用的 Llama-7B 模型。经过调研，我采用了以下解决方案：

- 使用 **Qwen 2.5 3B** 替代 Llama-7B，并通过 **4-bit 量化**（BitsAndBytes NF4）将显存占用压缩至约 1.5GB
- 配置 Hugging Face Accelerate 进行混合精度训练
- 调整 batch_size 为 4-8，seq_len 为 512

**（2）代码修改与调试**

在复现过程中，我遇到并解决了多个技术问题：

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `KeyError: 'qwen2'` | transformers 版本过低 | 升级至 ≥4.40.0 |
| 数据类型不匹配 | 4-bit 量化导致 bfloat16 与 float32 冲突 | 修改 TimeLLM.py 第 297-298 行，统一数据类型 |
| `AttributeError: 'content'` | load_content() 调用顺序错误 | 将其移至模型初始化之前 |
| 显存溢出 OOM | llm_layers 默认 32 层过多 | 减少至 6 层 |

**（3）基线实验结果**

在 ETTm1 数据集上，我成功复现了 Time-LLM 的基本性能：

```
训练输出示例：
iters: 100, epoch: 1 | loss: 0.2009
speed: 1.12s/iter
```

### 2.2 对 Time-LLM 架构的深入理解

通过阅读源码，我梳理了 Time-LLM 的完整数据流：

```
输入: [Batch=32, SeqLen=96, N_vars=7]
    ↓ Instance Normalization
    ↓ Patching (patch_len=16, stride=8)
中间: [B*N=224, num_patches=12, d_model=16]
    ↓ Reprogramming Layer (Cross-Attention)
    ↓ 映射到 LLM 维度
中间: [224, 12, llm_dim=768/2048]
    ↓ 与 Prompt Embeddings 拼接
    ↓ Frozen LLM Forward
    ↓ FlattenHead 输出投影
输出: [32, pred_len=96, 7]
```

**可训练参数分析**（以 GPT-2 为例）：

| 组件 | 参数量 | 职责 |
|------|--------|------|
| PatchEmbedding | ~800 | 将 Patch 嵌入到 d_model 维度 |
| Mapping Layer | ~50M | 词表压缩映射（50257→1000） |
| Reprogramming Layer | ~6M | Cross-Attention 实现跨模态对齐 |
| FlattenHead | ~37K | 输出投影到预测长度 |
| **总计** | **~56M** | LLM 参数完全冻结 |

这种"冻结主干+轻量微调"的设计非常巧妙，既利用了 LLM 的预训练知识，又避免了大规模参数更新带来的计算开销。

---

## 三、文献调研综述

### 3.1 基于 Transformer 的时序预测模型

**[1] PatchTST (ICLR 2023)**

借鉴视觉 Transformer 的 Patch 思想，将时间序列分割为子序列作为 Token。采用通道独立策略降低计算复杂度。这篇论文对 Time-LLM 的 Patching 设计有直接影响。

**[2] iTransformer (ICLR 2024)**

颠覆性地提出在变量维度而非时间维度做注意力。每个变量作为一个 Token，通过变量间注意力捕获多变量相关性。这一思路启发了我的创新点之一——变量间注意力增强。

**[3] TimesNet (ICLR 2023)**

发现时序变化可分解为周期内变化和周期间变化，通过 FFT 检测周期后重塑为 2D 张量处理。其频域分解思想可用于增强 Time-LLM。

### 3.2 多尺度与混合模型

**[4] TimeMixer (ICLR 2024)**

提出多尺度分解混合架构，细粒度捕获局部波动，粗粒度捕获长期趋势。其 PDM 和 FMM 模块的设计对多尺度融合有参考价值。

**[5] N-BEATS (ICLR 2020)**

通过约束网络输出为多项式基（趋势）和傅里叶基（季节性）的组合，实现可解释的预测分解。这种可解释性设计正是 Time-LLM 所缺乏的。

**[6] ES-RNN (M4 Competition Winner)**

M4 竞赛冠军，将指数平滑与 RNN 结合。ES 负责分解趋势/季节性，RNN 学习跨序列模式。证明了传统模型与深度学习融合的有效性。

### 3.3 LLM 用于时序预测

**[7] Time-LLM (ICLR 2024)**

本课题的基础论文。核心贡献是 Reprogramming Layer 实现时序到文本的跨模态对齐，以及 Prompt-as-Prefix 的统计信息编码。

**[8] AutoTimes (NeurIPS 2024)**

将时序预测表述为上下文预测，动态生成 Prompt。其自适应 Prompt 思想可用于增强 Time-LLM 对非平稳时序的适应能力。

**[9] Time-MoE (ICLR 2025)**

首次将稀疏专家混合引入时序基础模型，2.4B 参数仅激活 1B。其 MoE 架构可考虑引入 Time-LLM。

### 3.4 可解释性与知识蒸馏

**[10] Temporal Fusion Transformer (2021)**

提出变量选择网络量化特征重要性，可解释的多头注意力衡量时间步贡献。其可解释性设计是本课题的重要参考。

### 3.5 文献综述总结

| 研究方向 | 代表工作 | 对本课题的启发 |
|---------|---------|---------------|
| Patch 思想 | PatchTST | 基础架构设计 |
| 变量间关系 | iTransformer | 创新点：变量间注意力 |
| 频域分解 | TimesNet | 创新点：频域增强 |
| 多尺度混合 | TimeMixer | 创新点：多尺度分解 |
| 可解释性 | N-BEATS, TFT | 创新点：传统模型融合 |
| 混合架构 | ES-RNN | 理论支撑：残差学习 |

---

## 四、创新点设计与分析

基于文献调研和对 Time-LLM 的深入理解，我拟设计以下创新模块：

### 4.1 创新点一：可解释性增强与传统模型集成

**问题分析**：Time-LLM 虽然利用了 LLM 的能力，但预测过程仍是黑盒，缺乏可解释性。而传统统计模型（ARIMA、指数平滑）具有透明的数学结构。

**设计思路**：采用残差学习架构，传统模型先捕获线性成分，Time-LLM 学习残差中的非线性模式。

```
原始序列 → [ARIMA/ES] → 线性预测 + 残差
                              ↓
                      [Time-LLM 学习残差]
                              ↓
            最终预测 = 线性预测 + 非线性预测
```

**预期作用**：
- MSE 降低 10-15%（线性+非线性分工明确）
- 输出包含趋势/季节性分解（可解释）
- 传统模型提供置信度参考

**适用场景**：周期性明显的数据（ETT 电力负荷、Traffic 交通流量）

### 4.2 创新点二：分段自适应融合

**问题分析**：输入序列的不同区间可能呈现不同模式（如前半段平稳、后半段剧烈波动），固定权重的融合无法适应这种变化。

**设计思路**：设计分段模式检测器，根据每段的特征（周期性、趋势性、噪声水平）动态分配传统模型与 Time-LLM 的权重。

```python
# 模式检测指标
if FFT能量集中度 > 0.7:  # 强周期性
    传统模型权重 = 0.8
elif 线性趋势强度 > 0.5:  # 强趋势
    传统模型权重 = 0.7
else:  # 复杂模式
    Time-LLM权重 = 0.9
```

**预期作用**：
- 额外降低 MSE 3-5%
- 输出各段的模式判断与权重分配
- 增强模型对复杂时序的适应性

**适用场景**：非平稳时序、模式变化剧烈的数据

### 4.3 创新点三：变量间注意力增强

**问题分析**：Time-LLM 将多变量展平为独立样本处理（B*N），完全忽略了变量间的相关性。而实际数据中，变量间往往存在强关联（如电力负荷的有功/无功功率）。

**设计思路**：在 Reprogramming Layer 后添加 Inter-Variate Attention 模块，在变量维度进行注意力计算。

```
Reprogramming 输出 [B*N, L, D]
    ↓ Reshape → [B, N, L, D]
    ↓ 转置 → [B, L, N, D]
    ↓ 在 N 维度做 Multi-Head Attention
    ↓ 捕获变量间相关性
    ↓ Reshape back → [B*N, L, D]
```

**预期作用**：
- 多变量预测 MSE 降低 8-15%
- 可输出变量重要性权重（可解释）
- 参数增加仅约 5%

**适用场景**：多变量预测（features=M）、变量间有强相关性的数据

### 4.4 创新点四：频域增强

**问题分析**：时序数据通常包含趋势和周期成分，在时域直接处理可能混淆这两种模式。

**设计思路**：在 Patching 前通过 FFT 分解为低频（趋势）和高频（季节性）成分，分别用不同参数的分支处理后融合。

```
原始序列 → [FFT 分解]
              ↓
    低频成分（趋势）→ 粗 Patch → Time-LLM 分支 1
    高频成分（季节）→ 细 Patch → Time-LLM 分支 2
              ↓
         [特征融合] → 最终预测
```

**预期作用**：
- 周期性数据 MSE 降低 10-15%
- 可分别可视化趋势/季节预测
- FFT 计算高效，额外开销 < 5%

**适用场景**：ETTh（小时数据，24h/168h 周期明显）、Weather（季节性）

### 4.5 创新点优先级与组合策略

| 创新点 | 实现难度 | 预期收益 | 优先级 |
|--------|---------|---------|--------|
| 传统模型集成 | ⭐⭐⭐ | 10-15% MSE↓ + 可解释性 | 🥇 最高 |
| 变量间注意力 | ⭐⭐ | 8-15% MSE↓ | 🥈 高 |
| 频域增强 | ⭐⭐ | 10-15% MSE↓ | 🥉 高 |
| 分段自适应 | ⭐⭐⭐ | 3-5% MSE↓ | 中 |

**推荐组合**：传统模型集成 + 变量间注意力，兼顾可解释性与性能提升。

---

## 五、执行方案与进度计划

### 5.1 总体技术路线

```
阶段一：基础复现 → 阶段二：创新实现 → 阶段三：实验验证 → 阶段四：论文撰写
```

### 5.2 详细进度安排

| 周次 | 任务内容 | 产出物 | 验收标准 |
|------|---------|--------|---------|
| **第1-2周** | 精读 Time-LLM 论文，理解架构细节 | 论文笔记 | 能复述核心机制 |
| **第3周** | 精读 iTransformer、TimeMixer 等论文 | 文献综述初稿 | 10篇文献总结 |
| **第4周** | 环境搭建，配置 Qwen 2.5 3B 量化 | 可运行环境 | 模型能加载 |
| **第5周** | 运行 Time-LLM 官方代码，理解数据流 | 代码注释文档 | 能解释每个模块 |
| **第6周** | 完成 ETTh1/ETTm1 基线复现 | 基线结果 | MSE 与论文误差<5% |
| **第7-8周** | 实现创新点一：残差学习+传统模型集成 | 代码+实验结果 | 能正常训练 |
| **第9周** | 实现创新点二：分段自适应融合 | 代码+实验结果 | 权重可视化 |
| **第10周** | 实现创新点三：变量间注意力 | 代码+实验结果 | 注意力热力图 |
| **第11周** | 实现创新点四：频域增强 | 代码+实验结果 | 分解可视化 |
| **第12周** | 消融实验：验证各创新点有效性 | 消融实验表格 | 明确各模块贡献 |
| **第13周** | 对比实验：与 PatchTST、iTransformer 比较 | 对比实验表格 | 性能优于基线 |
| **第14周** | 可解释性展示：趋势分解、注意力可视化 | 可视化图表 | 论文配图完成 |
| **第15周** | 撰写毕业论文：方法与实验章节 | 论文初稿 | 核心章节完成 |
| **第16周** | 撰写论文：引言、相关工作、结论 | 论文终稿 | 全文完成 |
| **第17周** | 论文修改、答辩 PPT 准备 | 答辩材料 | 准备充分 |

### 5.3 实验设计

**实验一：基线复现**
```bash
python run_main.py --model TimeLLM --data ETTh1 --pred_len 96
# 预期：MSE ≈ 0.375
```

**实验二：消融实验**
```bash
# 基线
python run_main.py --model TimeLLM --data ETTh1

# + 传统模型集成
python run_main.py --model TimeLLM --data ETTh1 --use_hybrid

# + 变量间注意力
python run_main.py --model TimeLLM --data ETTh1 --use_inter_variate

# + 全部创新点
python run_main.py --model TimeLLM --data ETTh1 --use_all
```

**实验三：多数据集验证**

| 数据集 | 变量数 | 特点 | 验证目标 |
|--------|-------|------|---------|
| ETTh1/h2 | 7 | 强周期性 | 频域增强效果 |
| ETTm1/m2 | 7 | 高频采样 | 长序列性能 |
| Weather | 21 | 多变量 | 变量间注意力效果 |

### 5.4 预期结果

| 模型配置 | ETTh1 MSE | ETTm1 MSE | 可解释性 |
|---------|-----------|-----------|---------|
| Time-LLM (基线) | 0.375 | 0.302 | 低 |
| + 残差学习 | 0.330 | 0.265 | 高 |
| + 变量间注意力 | 0.320 | 0.255 | 中 |
| + 频域增强 | 0.315 | 0.250 | 高 |
| 全部组合 | **0.300** | **0.235** | **高** |

### 5.5 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 显存不足 | 中 | 高 | 减小 batch_size、使用梯度累积 |
| 创新点效果不显著 | 中 | 中 | 准备备选方案（动态Prompt、MoE） |
| 实验周期超期 | 低 | 高 | 优先在 ETTh1 验证，确认有效后扩展 |

---

## 六、学习体会与展望

### 6.1 学习体会

通过这段时间的学习和实践，我对大模型在时序预测中的应用有了深刻理解：

1. **跨模态迁移的可行性**：Time-LLM 证明了 LLM 的序列建模能力可以迁移至时序领域，这为多模态学习提供了新思路。

2. **轻量化适配的重要性**：冻结 LLM 主干、只训练轻量适配层的策略，既保留了预训练知识，又大幅降低了计算成本。这种思路在资源受限场景下尤为重要。

3. **可解释性的价值**：深度模型的黑盒特性限制了其在关键场景的应用。将传统统计模型的可解释性与深度学习的表达能力结合，是一个有价值的研究方向。

4. **工程实践的挑战**：从论文到可运行代码，涉及环境配置、数据类型、显存优化等大量工程问题。这些实践经验对科研能力的提升非常重要。

### 6.2 展望

本课题的研究成果有望在以下方面产生价值：

1. **学术价值**：提出可解释的混合预测框架，为 LLM 时序预测领域贡献新方法
2. **应用价值**：在电力负荷预测、交通流量预测等场景提供可解释的预测依据
3. **延伸价值**：为多模态融合、通用人工智能（AGI）在垂直领域的应用提供技术参考

---

## 七、参考文献

[1] Nie Y, et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR, 2023.

[2] Wu H, et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR, 2023.

[3] Liu Y, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR, 2024.

[4] Jin M, et al. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. ICLR, 2024.

[5] Wang S, et al. TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. ICLR, 2024.

[6] Oreshkin B N, et al. N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting. ICLR, 2020.

[7] Liu Y, et al. AutoTimes: Autoregressive Time Series Forecasters via Large Language Models. NeurIPS, 2024.

[8] Shi X, et al. Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. ICLR, 2025.

[9] Smyl S. A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Time Series Forecasting. International Journal of Forecasting, 2020.

[10] Lim B, et al. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. International Journal of Forecasting, 2021.

---

**报告日期**：2026年1月

**字数统计**：约 4200 字
