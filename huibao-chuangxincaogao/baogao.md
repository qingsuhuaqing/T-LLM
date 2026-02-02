# 研究生毕业设计开题报告

## 课题名称：基于大模型语义融合的时间序列预测模型设计与实现

---

## 一、研究背景与意义

### 1.1 时间序列预测的重要性

时间序列预测是数据科学领域的核心任务之一，广泛应用于能源负荷预测、金融市场分析、交通流量预测、气象预报等关键场景。准确的时间序列预测能够为决策者提供科学依据，降低运营成本，提升资源配置效率。然而，传统时序预测方法在面对复杂的非线性模式、长程依赖关系以及多变量交互时往往表现不足。

### 1.2 深度学习在时序预测中的应用与局限

近年来，深度学习技术在时序预测领域取得了显著进展。循环神经网络（RNN）、长短期记忆网络（LSTM）以及 Transformer 架构相继被引入时序建模。特别是 PatchTST[1]、TimesNet[2]、iTransformer[3] 等模型在多个基准数据集上刷新了性能记录。然而，这些专用模型存在以下局限：

1. **模式识别能力受限**：从零训练的模型难以学习到足够丰富的序列模式表示
2. **泛化能力不足**：针对特定数据集训练的模型难以迁移至其他领域
3. **可解释性缺失**：深度模型的预测过程缺乏透明度，难以解释预测依据

### 1.3 大语言模型赋能时序预测的新范式

大语言模型（Large Language Models, LLM）通过海量文本预训练，获得了强大的模式识别、序列建模和推理能力。这一能力是否可以迁移至时间序列领域？Time-LLM[4] 的提出为这一问题提供了肯定的答案。

Time-LLM 通过"重编程"（Reprogramming）技术，在保持 LLM 权重冻结的前提下，将时序数据映射至 LLM 的语义空间，从而利用 LLM 预训练获得的模式识别能力进行时序预测。这一范式具有以下优势：

1. **输入重编程**：将时间序列切片（Patching），通过线性投影层映射至 LLM 的词向量空间，使 LLM 将时序特征视为"特殊的文本特征"进行处理
2. **提示重编程**：在输入前添加包含时序统计特征（均值、方差、趋势、主要周期等）的自然语言提示，激活 LLM 内部的趋势识别与模式匹配能力

这一方法不仅展示了跨模态知识迁移的可能性，也为通用人工智能（AGI）在垂直领域的应用提供了技术路径。将时序信息映射为文本表示并借助 LLM 能力进行预测，可以极大扩展模型的适配性与预测准确性，同时为具身智能在特定场景下的应用提供了可能。

### 1.4 研究意义

本课题旨在深入研究 Time-LLM 框架，针对其在可解释性、长序列建模、多变量关联捕获等方面的不足，设计创新性改进模块。研究成果将：

1. 推动大语言模型在时序预测领域的应用深化
2. 提升模型的可解释性，增强预测结果的可信度
3. 探索传统统计模型与深度学习的融合范式
4. 为多模态学习与跨领域知识迁移提供技术参考

---

## 二、国内外研究现状

### 2.1 基于 Transformer 的时序预测模型

**PatchTST (ICLR 2023)**[1]：借鉴视觉 Transformer 中 Patch 的思想，将时间序列分割为固定长度的子序列（Patch），作为 Transformer 的输入 Token。该方法通过通道独立策略（Channel Independence）有效降低了计算复杂度，在长期预测任务上取得了优异性能。

**iTransformer (ICLR 2024)**[3]：颠覆了传统 Transformer 在时间维度做注意力的范式，提出在变量维度进行注意力计算。每个变量被视为一个 Token，通过变量间注意力机制捕获多变量相关性，在多变量预测任务上显著优于同期模型。

**TimesNet (ICLR 2023)**[2]：发现时间序列的变化可以分解为周期内变化（Intraperiod）与周期间变化（Interperiod），通过 FFT 检测主要周期，将 1D 时序重塑为 2D 张量，利用 2D 卷积捕获复杂模式。

### 2.2 多尺度与频域分解方法

**TimeMixer (ICLR 2024)**[5]：提出多尺度分解混合架构，将时序数据在不同采样尺度下分别处理。细粒度尺度捕获局部波动，粗粒度尺度捕获长期趋势，通过 Past-Decomposable-Mixing (PDM) 和 Future-Multipredictor-Mixing (FMM) 模块实现尺度间信息融合。

**N-BEATS (ICLR 2020)**[6]：提出可解释的神经基扩展分析，通过约束网络输出为多项式基（趋势）和傅里叶基（季节性）的线性组合，实现预测结果的可解释分解。

### 2.3 大语言模型用于时序预测

**Time-LLM (ICLR 2024)**[4]：首次系统性地提出将预训练 LLM 重编程用于时序预测。核心创新包括：(1) Patch Embedding 将时序映射至 LLM 输入空间；(2) Reprogramming Layer 通过交叉注意力实现时序-文本跨模态对齐；(3) Prompt-as-Prefix 将统计信息编码为文本提示。

**AutoTimes (NeurIPS 2024)**[7]：将时序预测表述为上下文预测（In-context Forecasting），通过自回归方式动态生成 Prompt，增强模型对非平稳时序的适应能力。

**Time-MoE (ICLR 2025)**[8]：首次将稀疏专家混合（Mixture of Experts）引入时序基础模型，在 24 亿参数规模下通过稀疏激活机制，实现高容量与低计算的平衡。

### 2.4 混合模型与可解释性增强

**ES-RNN (M4 Competition Winner)**[9]：M4 预测竞赛冠军方法，将指数平滑（Exponential Smoothing）与 RNN 结合。ES 负责分解时序的水平、趋势和季节性成分，RNN 学习跨序列的共享局部趋势，证明了混合方法的有效性。

**Temporal Fusion Transformer (TFT)**[10]：提出变量选择网络（Variable Selection Network）量化每个输入特征的重要性，通过可解释的多头注意力机制衡量过去时间步的贡献，实现了预测过程的透明化。

### 2.5 研究现状总结

| 方法类型 | 代表工作 | 优势 | 局限 |
|---------|---------|------|------|
| 专用 Transformer | PatchTST, iTransformer | 长序列建模能力强 | 缺乏预训练知识 |
| 多尺度分解 | TimeMixer, N-BEATS | 多尺度模式捕获 | 计算开销较大 |
| LLM 重编程 | Time-LLM, AutoTimes | 利用预训练知识 | 可解释性不足 |
| 混合模型 | ES-RNN, TFT | 可解释性强 | 架构复杂 |

现有研究在各自方向取得了突破，但尚缺乏系统性地将传统统计模型优势与 LLM 重编程范式深度融合的工作。本课题将针对这一空白展开研究。

---

## 三、研究内容与创新点

### 3.1 研究目标

本课题以 Time-LLM 为基础框架，设计并实现可解释性增强的混合预测模型。具体目标包括：

1. 深入理解 Time-LLM 的技术原理与实现细节
2. 完成 Time-LLM 论文的完整复现与验证
3. 设计传统模型与 LLM 深度融合的创新模块
4. 在长序列、多变量数据集上验证改进效果
5. 完成模型性能分析与可解释性展示

### 3.2 核心创新点

#### 创新点一：残差学习混合架构（Residual Learning Architecture）

**核心思想**：传统统计模型先捕获线性成分，Time-LLM 学习残差中的非线性成分。

时间序列可分解为：
$$y_t = L_t + N_t + \epsilon_t$$

其中 $L_t$ 为线性成分（趋势、季节性），$N_t$ 为非线性成分（复杂模式、突变），$\epsilon_t$ 为随机噪声。

**架构设计**：
```
原始序列 → [传统模型(ARIMA/ES)] → 线性预测 + 残差
                                        ↓
                               [Time-LLM 学习残差]
                                        ↓
              最终预测 = 线性预测 + 非线性预测
```

**理论支撑**：Hybrid ARIMA-LSTM 研究[11]证明，残差修正框架将预测任务分解为线性趋势分析和非线性残差学习，可有效发挥两类模型的互补优势。

**预期效果**：MSE 降低 10-15%，同时提供趋势/季节性分解的可解释输出。

#### 创新点二：分段自适应融合（Segment-wise Adaptive Fusion）

**核心思想**：输入序列的不同区间可能呈现不同模式，应动态调整传统模型与 Time-LLM 的融合权重。

**架构设计**：
```
输入序列 [t_0, t_1, ..., t_T]
    │
    ▼
[分段模式检测器]
    │
    ├── Segment 1: 周期性强 → 传统模型权重 0.8
    ├── Segment 2: 复杂模式 → Time-LLM 权重 0.9
    └── Segment 3: 趋势明显 → 传统模型权重 0.7
              │
              ▼
       [分段加权融合] → 最终预测
```

**模式检测指标**：
| 检测方法 | 判断标准 | 对应策略 |
|---------|---------|---------|
| ADF 平稳性检验 | p-value < 0.05 → 平稳 | ARIMA 权重高 |
| FFT 能量集中度 | > 0.7 → 强周期性 | 传统模型权重高 |
| 线性回归斜率 | 趋势强度 > 0.5 | Holt-Winters 适用 |
| 噪声水平 | > 0.6 → 高噪声 | 移动平均预处理 |

**理论支撑**：AMD Framework[12] 使用 Mixture-of-Experts 的自适应特性，为不同时间模式设计不同预测器，通过时间模式选择器动态分配权重。

**预期效果**：MSE 额外降低 3-5%，输出包含各段的模式判断与权重分配报告。

#### 创新点三：知识蒸馏增强（Knowledge Distillation Enhancement）

**核心思想**：让 Time-LLM 学习传统模型的优势，将统计模型的先验知识迁移至深度模型。

**三种蒸馏策略**：

1. **软标签蒸馏**：Time-LLM 学习传统模型的预测分布
   $$L_{distill} = \alpha \cdot L_{MSE}(y_{pred}, y_{true}) + (1-\alpha) \cdot L_{KL}(y_{pred}, y_{teacher})$$

2. **分解特征蒸馏**：学习传统模型的趋势/季节性分解能力
   - 在 Time-LLM 输出层添加趋势头和季节头
   - 与传统模型的 STL 分解结果对齐

3. **行为模仿蒸馏**：学习预测方向一致性、幅度一致性、频谱相似度
   - 方向一致性：$sign(y'_{pred}) = sign(y'_{teacher})$
   - 幅度一致性：归一化后的变化幅度对齐
   - 频谱一致性：FFT 振幅谱的相似度

**理论支撑**：DE-TSMCL[13] 在 ETTm1 数据集上通过知识蒸馏实现 MSE 提升 24.2%，MAE 提升 14.7%。

**预期效果**：收敛速度提升 20-30%，模型泛化能力增强。

### 3.3 组合方案设计

将上述三个创新点整合为完整的混合预测框架：

```
原始序列 y_t
    │
    ▼
[传统模型 (ARIMA/ES)]
    │
    ├── 线性预测 ŷ_linear
    │
    └── 残差 r_t = y_t - ŷ_linear
              │
              ▼
         [分段分析器]
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  Seg1      Seg2      Seg3
 (周期强)  (复杂)   (趋势强)
    │         │         │
    ▼         ▼         ▼
 权重0.3   权重0.9   权重0.4
    │         │         │
    └─────────┼─────────┘
              │
              ▼
         [Time-LLM]
              │
              ▼
         非线性预测 ŷ_nonlinear
              │
              ▼
最终预测 = ŷ_linear + Σ(w_i × ŷ_nonlinear_i)
```

### 3.4 可解释性输出设计

模型将输出完整的可解释性报告：

```
═══════════════════════════════════════════════════════
              Time-LLM 可解释性预测报告
═══════════════════════════════════════════════════════

1. 输入数据特征分析
   - 数据平稳性: [平稳/非平稳] (ADF p-value: 0.023)
   - 主要周期: [24, 168] 小时 (FFT 分析)
   - 整体趋势: [上升/下降/平稳] (斜率: +0.0012)

2. 模型选择与权重分配
   - 传统模型: ARIMA(2,1,1) + 季节性分解
   - 混合权重: Traditional=0.3, Time-LLM=0.7

3. 预测分解
   - 趋势成分: [由传统模型主导]
   - 季节成分: [由 Holt-Winters 贡献]
   - 残差成分: [由 Time-LLM 捕获]

═══════════════════════════════════════════════════════
```

---

## 四、实验方案与评价指标

### 4.1 实验环境

| 配置项 | 规格 |
|-------|------|
| 操作系统 | Windows 11 + WSL2 (Ubuntu) |
| GPU | NVIDIA GTX 1660 Ti (6GB VRAM) |
| 深度学习框架 | PyTorch 2.0+ |
| LLM 后端 | Qwen 2.5 3B (4-bit 量化) |
| 加速库 | Hugging Face Accelerate, BitsAndBytes |

### 4.2 数据集

| 数据集 | 类型 | 变量数 | 序列长度 | 采样频率 |
|-------|------|-------|---------|---------|
| ETTh1/h2 | 电力负荷 | 7 | 17420 | 小时 |
| ETTm1/m2 | 电力负荷 | 7 | 69680 | 15分钟 |
| Weather | 气象 | 21 | 52696 | 10分钟 |
| Electricity | 用电量 | 321 | 26304 | 小时 |

### 4.3 评价指标

| 指标 | 公式 | 用途 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 主要性能指标 |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 辅助性能指标 |
| 可解释性得分 | 定性评估 | 创新点验证 |
| 推理速度 | ms/sample | 效率评估 |

### 4.4 实验设计

**实验一：基线复现**
- 任务：完整复现 Time-LLM 在 ETT 数据集上的性能
- 预期：MSE 与论文报告值误差 < 5%

**实验二：消融实验**
- 任务：逐一验证三个创新点的有效性
- 对比组：
  - Time-LLM (基线)
  - + 残差学习架构
  - + 分段自适应融合
  - + 知识蒸馏
  - + 全部创新点

**实验三：对比实验**
- 任务：与同期 SOTA 模型对比
- 对比模型：PatchTST, iTransformer, TimeMixer

**实验四：可解释性展示**
- 任务：可视化趋势/季节性分解、权重分配
- 形式：热力图、时序分解图、注意力可视化

### 4.5 预期结果

| 模型 | ETTh1 MSE | ETTm1 MSE | 可解释性 |
|------|-----------|-----------|---------|
| Time-LLM (基线) | 0.375 | 0.302 | 低 |
| + 残差学习 | 0.330 | 0.265 | 高 |
| + 分段融合 | 0.315 | 0.250 | 高 |
| + 知识蒸馏 | 0.305 | 0.240 | 高 |

---

## 五、研究计划与时间安排

### 5.1 总体进度安排

| 阶段 | 时间 | 主要任务 | 产出 |
|------|------|---------|------|
| 第一阶段 | 第1-3周 | 文献调研与论文精读 | 文献综述报告 |
| 第二阶段 | 第4-6周 | Time-LLM 复现与验证 | 基线实验结果 |
| 第三阶段 | 第7-10周 | 创新模块设计与实现 | 核心代码 |
| 第四阶段 | 第11-13周 | 实验验证与优化 | 实验报告 |
| 第五阶段 | 第14-16周 | 论文撰写与答辩准备 | 毕业论文 |

### 5.2 详细执行计划

**第一阶段：文献调研（第1-3周）**
- 第1周：精读 Time-LLM、PatchTST、iTransformer 论文
- 第2周：精读 TimeMixer、Time-MoE、N-BEATS 论文
- 第3周：整理文献综述，确定技术路线

**第二阶段：基线复现（第4-6周）**
- 第4周：搭建实验环境，配置 Qwen 2.5 3B 量化推理
- 第5周：运行 Time-LLM 官方代码，理解数据流
- 第6周：完成 ETTh1/ETTm1 数据集的基线复现

**第三阶段：创新实现（第7-10周）**
- 第7周：实现残差学习混合架构
- 第8周：实现分段自适应融合模块
- 第9周：实现知识蒸馏增强策略
- 第10周：集成测试与调参优化

**第四阶段：实验验证（第11-13周）**
- 第11周：完成消融实验
- 第12周：完成对比实验
- 第13周：完成可解释性展示实验

**第五阶段：论文撰写（第14-16周）**
- 第14周：撰写方法与实验章节
- 第15周：撰写引言、相关工作、结论
- 第16周：论文修改、答辩准备

### 5.3 风险控制

| 风险 | 应对措施 |
|------|---------|
| 显存不足 | 使用 4-bit 量化、梯度累积、减小 batch_size |
| 创新点效果不显著 | 准备备选方案（变量间注意力、频域增强） |
| 实验周期过长 | 优先在小数据集验证，确认有效后扩展 |

---

## 六、预期成果

1. **技术成果**
   - 可解释性增强的混合时序预测模型
   - 在 ETT 数据集上 MSE 降低 15-20%

2. **学术成果**
   - 完成毕业论文撰写
   - 具备投稿国内会议/期刊的研究基础

3. **工程成果**
   - 完整的代码实现与文档
   - 可复现的实验环境与脚本

---

## 七、参考文献

[1] Nie Y, Nguyen N H, Sinthong P, et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers[C]. ICLR, 2023.

[2] Wu H, Hu T, Liu Y, et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis[C]. ICLR, 2023.

[3] Liu Y, Hu T, Zhang H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting[C]. ICLR, 2024.

[4] Jin M, Wang S, Ma L, et al. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models[C]. ICLR, 2024.

[5] Wang S, Wu H, Shi X, et al. TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting[C]. ICLR, 2024.

[6] Oreshkin B N, Carpov D, Chapados N, et al. N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting[C]. ICLR, 2020.

[7] Liu Y, Zhang H, Li C, et al. AutoTimes: Autoregressive Time Series Forecasters via Large Language Models[C]. NeurIPS, 2024.

[8] Shi X, Wu H, Wang S, et al. Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts[C]. ICLR, 2025.

[9] Smyl S. A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Time Series Forecasting[J]. International Journal of Forecasting, 2020.

[10] Lim B, Arık S Ö, Loeff N, et al. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting[J]. International Journal of Forecasting, 2021.

[11] Zhang G P. Time Series Forecasting Using a Hybrid ARIMA and Neural Network Model[J]. Neurocomputing, 2003.

[12] Wang S, et al. Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting[J]. arXiv preprint, 2024.

[13] Chen Y, et al. Distillation Enhanced Time Series Forecasting via Multi-scale Contrastive Learning[J]. arXiv preprint, 2024.

---

## 八、总结

本课题以 Time-LLM 为基础，针对其可解释性不足、传统模型优势未充分利用等问题，提出残差学习混合架构、分段自适应融合、知识蒸馏增强三项创新。通过将传统统计模型的可解释性与 LLM 的强大模式识别能力深度融合，预期在保持预测精度的同时显著提升模型的可解释性。研究成果将为大语言模型在时序预测领域的应用提供新思路，具有重要的学术价值与实践意义。

---

**报告撰写日期**：2026年1月

**字数统计**：约 4500 字
