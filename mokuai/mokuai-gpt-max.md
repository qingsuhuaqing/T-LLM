# Time-LLM新增模块（顶级期刊/会议风格规范版）

> 论文题目：基于大模型语义融合的时间序列预测模型设计与实现  
> 基线：Time-LLM (ICLR 2024)  
> 文档版本：v3.0  
> 日期：2026年2月

---

## 一、基线框架定位与新增模块插入位置

Time-LLM通过“输入重编程 + 提示前缀 + 冻结LLM + 线性预测头”实现时序预测。新增模块以“学术裁缝+模块融合”为原则，即在不破坏主干结构的前提下，将已有SOTA验证有效的机制进行组合式融合，形成可插拔的改进模块。

统一数据流如下：

```
X -> RevIN -> MDFR -> Reprogramming -> Frozen LLM -> RAHF -> Y_hat
```

- MDFR：多尺度分解-混合重编程模块（输入侧增强）。
- RAHF：检索增强的层级预测模块（输出侧增强）。

---

## 二、新增模块一：多尺度分解-混合重编程模块（MDFR）

### 2.1 解决的问题

Time-LLM采用固定尺度patch进行重编程，面对多周期、多频段时序时存在尺度偏置，导致长视界预测误差累积；同时，季节性与趋势信息难以在统一尺度上稳定建模。

### 2.2 增加的目的

- 构建跨尺度一致的patch语义表示，增强对长周期与短期扰动的同步建模能力。
- 在不改动LLM主干的前提下，提升输入侧表示质量，减少尺度敏感性。
- 为后续提示与重编程提供更具可解释性的结构性先验。

### 2.3 模块原理

MDFR以“分解-混合-融合”为核心流程：

1. 多尺度金字塔构建：对输入序列进行多尺度下采样，形成多分辨率序列集合。
2. 尺度自适应patch：每个尺度配置不同patch长度与步长，保证局部语义一致性。
3. 分解与双向混合：对每尺度序列进行趋势/季节分解，并进行细→粗、粗→细双向混合。
4. 跨尺度门控融合：以门控或注意力权重将多尺度特征融合为统一patch表示。

形式化表达：

给定输入 $X \in \mathbb{R}^{B\times T\times N}$，尺度集合 $S=\{1,2,4\}$，则

$$X^{(s)}=\text{Downsample}(X, s), \quad P^{(s)}=\text{PatchEmbed}(X^{(s)})$$

$$H^{(s)}=\text{BiMix}(\text{Decomp}(P^{(s)})), \quad H=\sum_{s\in S}\alpha_s H^{(s)}$$

其中 $H$ 为送入重编程层的统一patch语义表示。

### 2.4 相应效果

- 强周期数据集（ETTh/Weather）下，周期与趋势解耦更充分，长预测误差降低。
- 高频数据集（ETTm）下，短期扰动与长期趋势可同时建模，提高稳定性。
- 多尺度语义增强提升LLM重编程阶段的鲁棒性与可解释性。

---

## 三、新增模块二：检索增强的层级预测模块（RAHF）

### 3.1 解决的问题

Time-LLM输出头直接回归数值，缺少显式趋势方向约束；同时未利用训练集中历史相似模式，导致对稀有模式与跨域数据的泛化不足。

### 3.2 增加的目的

- 通过“方向 → 幅度 → 数值”的层级预测流程，模拟人类专家的判断过程。
- 通过检索增强引入外部历史参考，提高稀有模式的预测稳定性。
- 在不修改LLM主干的前提下，形成可扩展的预测头。

### 3.3 模块原理

RAHF由两部分构成：检索增强子模块 + 层级预测子模块。

**A. 检索增强子模块**
- 建立历史模式记忆库，使用可检索时序编码器生成索引表示。
- 针对当前输入的表示向量，检索Top-k相似历史窗口，并提取其真实未来片段作为参考。
- 通过门控或交叉注意力将检索参考与当前输入融合。

**B. 层级预测子模块**
- Level-1：趋势方向分类（上涨/下降/平稳，或5类细化）。
- Level-2：幅度区间回归或分段分类（变化比例）。
- Level-3：残差细化回归，输出最终数值。

统一损失：

$$\mathcal{L}=\mathcal{L}_{mse}+\lambda_1 \mathcal{L}_{dir}+\lambda_2 \mathcal{L}_{mag}$$

其中 $\mathcal{L}_{dir}$ 为趋势分类损失，$\mathcal{L}_{mag}$ 为幅度回归损失。

### 3.4 相应效果

- 趋势方向准确率提升，拐点位置误差降低。
- 检索增强为模型提供“历史相似模式的未来参考”，提高少样本与跨域稳定性。
- 与MDFR结合后形成输入侧与输出侧的“双增强”结构。

---

## 四、工作量体现与实现要点

- MDFR新增：多尺度下采样、尺度自适应patch、趋势/季节分解、双向混合、门控融合等模块。
- RAHF新增：可检索时序编码器、记忆库构建与索引、Top-k检索、融合机制、多任务损失函数。
- 训练与评估新增：方向/幅度指标、检索规模消融、多尺度消融、长预测视界评测。
- 模型集成新增：`--use_mdfr`、`--use_rahf` 等模块开关与配置接口。

---

## 五、参考文献（顶级会议/期刊）

1. Jin M, Wang S, Ma L, et al. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. ICLR 2024.
2. Wang S, Wu H, Shi X, et al. TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. ICLR 2024.
3. Liu Y, Hu T, Zhang H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR 2024.
4. Han S, Lee S, Cha M, Arik S O, Yoon J. Retrieval Augmented Time Series Forecasting (RAFT). ICML 2025.
5. Nie Y, Nguyen N, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST). ICLR 2023.
6. Wu H, Hu T, Liu Y, et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR 2023.
7. Wu H, Xu J, Wang J, Long M. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS 2021.

