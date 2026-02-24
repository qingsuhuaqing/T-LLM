# Time-LLM增强框架技术文档 v4

> **Version**: 4.0 (Academic Edition)
> **Date**: 2026-02-24
> **Framework**: 多相多尺度预测 + 增强检索自适应融合
> **核心改变**: v3模块内部升级 — 多相采样 / 专家投票 / 增强检测 / 阈值门控

---

## 目录

1. [研究背景与问题分析](#1-研究背景与问题分析)
2. [设计原则与版本演进 (v3 → v4)](#2-设计原则与版本演进-v3--v4)
3. [模块一：TAPR v4 趋势感知尺度路由](#3-模块一tapr-v4-趋势感知尺度路由)
   - 3.1 [DM：多相交错多尺度预测](#31-dm多相交错多尺度预测polyphase-interleaved-multi-scale)
   - 3.2 [C2F：专家投票 + 层级粗到细融合](#32-c2f专家投票--层级粗到细融合expert-voting--hierarchical-coarse-to-fine-fusion)
4. [模块二：GRA-M v4 全局检索增强记忆](#4-模块二gra-m-v4-全局检索增强记忆)
   - 4.1 [PVDR：增强检测 + 双重检索](#41-pvdr增强检测--双重检索enhanced-detection--dual-retrieval)
   - 4.2 [AKG：阈值驱动的自适应门控](#42-akg阈值驱动的自适应门控threshold-driven-adaptive-gating)
5. [可训练参数：理论分析](#5-可训练参数理论分析)
6. [损失函数设计](#6-损失函数设计)
7. [系统架构与数据流](#7-系统架构与数据流)
8. [实验设计](#8-实验设计)
9. [参考文献](#9-参考文献)

---

## 1. 研究背景与问题分析

### 1.1 时序预测领域发展脉络

时间序列预测（Time Series Forecasting, TSF）是数据挖掘领域的经典问题。其方法论经历了三个主要阶段：

**第一阶段：统计学方法时代**
- ARIMA（差分整合移动平均自回归模型）与指数平滑法等统计学方法基于严格的数学假设，具有极强的可解释性
- 局限性：在面对非线性、非平稳及高维多变量数据时往往力不从心

**第二阶段：深度学习时代**
- 循环神经网络（RNN）及其变体LSTM通过门控机制有效捕捉了序列的短程依赖
- 2017年Transformer架构问世，Informer、Autoformer、FEDformer等变体通过自注意力机制实现了对长程依赖的并行捕捉
- 局限性：本质上仍属于"专用模型"，需要针对特定数据集从头训练，缺乏跨域泛化能力

**第三阶段：基础模型时代（2023-2025）**
- 随着ChatGPT、LLaMA等大型语言模型的爆发，TSF领域迎来了"基础模型"时代
- Time-LLM正是这一方向的杰出代表，它通过"重编程"（Reprogramming）技术，将连续的时间序列数据映射为LLM能够理解的文本原型

### 1.2 Time-LLM存在的核心问题

通过对Time-LLM v1-v3增强版的实验分析，我们识别出两个持续存在的技术瓶颈：

#### 问题A：多尺度感知缺失（Scale Blindness）

**现象描述**：
Time-LLM将时间序列切分为固定长度的Patch进行处理，缺乏对数据多分辨率特性的显式建模。

**v3进展**：v3引入了DM（可分解多尺度）模块，通过平均池化下采样 + 信号变换实现多尺度独立预测。

**v3残留问题**：平均池化下采样本质上是一种**有损变换**——对于下采样率 $k$，$k$ 个采样点被平均为1个点，高频细节被不可逆地抹除。这与多尺度分析的初衷（"在不同分辨率下保留不同频率的信息"）存在矛盾。

**文献支撑**：
- **多速率信号处理** (Vaidyanathan, 1993) 指出多相分解（Polyphase Decomposition）是实现完美重构滤波器组的基础，可以在不损失任何信息的前提下实现多速率分析
- **TimeMixer** (ICLR 2024) 的多尺度混合虽然有效，但其下采样策略同样基于平均池化，存在信息损失问题

#### 问题B：局部上下文局限（Local Context Limitation）

**现象描述**：
Time-LLM主要依赖输入窗口内的信息进行推理，缺乏对全局历史数据的回溯能力。

**v3进展**：v3引入了PVDR多尺度检索模块和AKG自适应门控，实现了检索增强预测。

**v3残留问题**：
1. 极端数据检测使用固定的3-sigma规则，对不同分布的数据缺乏适应性
2. 缺乏对**结构性变点**（regime change）的检测能力——时间序列的统计特性在某些时刻发生突变，此时模型需要特别依赖历史经验
3. AKG的sigmoid门控是静态的（仅依赖可学习参数C），未利用PVDR的检测信号进行动态调节

**文献支撑**：
- **Bayesian Online Change Point Detection** (Adams & MacKay, 2007) 提出了贝叶斯框架下的在线变点检测方法，通过递归更新后验分布实现高效检测
- **RAFT** (ICML 2025) 表明检索增强时序预测在极端事件和分布偏移场景下尤为有效，但需要精准的检测机制来触发适当的检索策略

### 1.3 v3版本实验分析与不足

**v3增强版已取得的进展**：

| 维度 | v2 (观察者模式) | v3 (主动融合) | 改善 |
|------|----------------|--------------|------|
| 辅助损失 | ~0.0002 (无效) | ~0.05 (有效) | ↑250× |
| 新模块参数 | ~5K | ~103K | 适中 |
| 架构完整性 | Baseline不变 | Baseline不变 | 保持 |

**v3仍存在的不足**：

1. **DM信息损失**：平均池化下采样率16时，每16个点只保留1个均值点，约93.75%的变异信息被丢弃（方差缩减为原始的1/16）
2. **C2F缺乏鲁棒性**：所有尺度的预测直接加权融合，单个异常尺度可能"拖累"整体
3. **PVDR检测单一**：仅有3-sigma极端检测，缺乏变点检测能力
4. **AKG门控静态**：sigmoid(C)在训练后固定，无法根据输入数据的特性动态调节

---

## 2. 设计原则与版本演进 (v3 → v4)

### 2.1 v4设计哲学："模块内部精细化"

v4的核心设计哲学与v3有本质区别：

- **v2 → v3**: 整体架构变革（观察者模式 → 主动预测融合）
- **v3 → v4**: 模块内部精细化（保持架构不变，升级每个模块的算法内核）

这一策略的优势在于：
1. **风险可控**：v3的整体流程已验证有效，v4仅替换内部算法
2. **消融清晰**：可以逐个模块对比v3 vs v4实现
3. **工程成本低**：模块接口基本保持不变，集成代码改动最小

### 2.2 四个模块的升级路线

| 模块 | v3 方法 | v3 不足 | v4 升级 | 理论依据 |
|------|---------|---------|---------|---------|
| DM | 平均池化下采样 | 信息损失 | 多相交错采样 | 多速率信号处理 |
| C2F | 直接softmax加权 | 异常分支干扰 | MAD投票 + DWT趋势约束 | 鲁棒统计 + 小波分析 |
| PVDR | 3-sigma极端检测 | 固定规则 + 无变点检测 | 逻辑回归 + 贝叶斯变点 | 判别学习 + 贝叶斯推断 |
| AKG | 固定sigmoid门控 | 无动态调节 | PVDR信号驱动的阈值门控 | 条件门控 |

### 2.3 保持不变的原则

1. **Baseline完整性**：原始Time-LLM的forward pass完全不修改
2. **整体流程不变**：DM → C2F → PVDR → AKG 的顺序和数据流向不变（架构图不变）
3. **模块可剥离**：`--use_tapr` 和 `--use_gram` 独立控制
4. **参数可控**：v4新增参数仅~12个，计算开销可通过命令参数调节

### 2.4 学术裁缝方法论 v4

本研究严格遵循"学术裁缝"方法论进行增量创新：

1. **有源可溯**：多相分解源自信号处理经典理论，贝叶斯变点检测源自Adams & MacKay (2007)
2. **组合创新**：将信号处理、鲁棒统计、贝叶斯推断、条件门控四种理论在Time-LLM框架下有机整合
3. **逐步验证**：每个模块可独立对比v3实现，消融矩阵覆盖所有升级组合

---

## 3. 模块一：TAPR v4 趋势感知尺度路由

**TAPR (Trend-Aware Patch Router)** v4版本在保持v3"多尺度独立预测+跨尺度融合"架构的基础上，升级DM的下采样策略和C2F的融合机制。

### 3.1 DM：多相交错多尺度预测（Polyphase Interleaved Multi-Scale）

#### 3.1.1 理论基础

**核心理论：多相分解（Polyphase Decomposition）**

多相分解是多速率信号处理的基础理论，最早由Vaidyanathan (1993)在滤波器组设计中系统化。其核心思想是将信号按相位分解为多个子信号，每个子信号包含原信号的一个交错子集。

对于离散信号 $x[n]$，其粒度为 $k$ 的多相分解定义为：

$$x_p[m] = x[k \cdot m + p], \quad p = 0, 1, \ldots, k-1; \quad m = 0, 1, \ldots, \lfloor T/k \rfloor - 1$$

其中 $p$ 称为**相位索引**（phase index），$k$ 个多相分支的并集**恰好覆盖全部原始数据**：

$$\bigcup_{p=0}^{k-1} \{x_p[m]\}_{m=0}^{T/k-1} = \{x[n]\}_{n=0}^{T-1}$$

**关键性质**：
- **完美覆盖**：所有数据被使用，无信息丢失
- **无冗余**：每个数据点仅属于一个分支
- **等长分支**：每个分支长度为 $T/k$
- **保频特性**：每个分支保留了原信号在对应相位上的采样特性

**与平均池化的对比**：

| 特性 | 平均池化 (v3) | 多相分解 (v4) |
|------|--------------|--------------|
| 输出长度 | $T/k$ (1个序列) | $T/k$ × $k$ 个序列 |
| 信息保留 | 仅均值，丢失变异 | 100%保留 |
| 高频信息 | 被平均抹除 | 保留在各分支中 |
| 计算量 | 1次前向传播 | $k$ 次前向传播 (共享编码器) |
| 可逆性 | 不可逆 | 完全可逆 |

**信号处理角度的解释**：

平均池化等价于一个低通滤波器后跟下采样，而多相分解等价于**直接下采样**后的多路并行处理。根据Noble恒等式（Noble Identity），多相分解在滤波器组设计中可以实现完美重构（Perfect Reconstruction），这意味着从多相分支可以无损地恢复原始信号。

**在时序预测中的适用性**：

TimeMixer (ICLR 2024) 指出"不同下采样尺度呈现不同模式"，但其使用的平均池化下采样在高采样率时会严重损失信息。多相分解解决了这一问题——每个分支虽然只有 $T/k$ 个点，但这些点是原始信号的真实采样值（而非均值），保留了完整的局部特征。

**共享编码器的理论依据**：

在平稳性假设下，同一尺度的不同相位分支在统计上是同分布的（因为它们是同一信号的等间隔采样）。因此，使用同一个ScaleEncoder处理所有分支不仅在参数效率上有优势，而且在统计学上是合理的。$k$ 个分支相当于 $k$ 倍数据增强，有助于编码器的泛化。

#### 3.1.2 数学表述

**多相提取**：

$$X_p^{(s)} = \{x[k_s \cdot m + p]\}_{m=0}^{T/k_s - 1} \in \mathbb{R}^{B \times (T/k_s) \times N}, \quad p = 0, \ldots, k_s-1$$

**信号变换**（与v3相同）：

$$\tilde{X}_p^{(s)} = \text{Transform}_s(X_p^{(s)})$$

| 尺度 | 粒度 $k_s$ | 变换 $\text{Transform}_s$ | 分支数 | 每分支长度 ($T=512$) |
|------|-----------|--------------------------|--------|---------------------|
| S1 | 1 | 无 (baseline LLM) | 1 | 512 |
| S2 | 2 | 一阶差分 | 2 | 256→255 |
| S3 | 4 | 恒等映射 | 4 | 128 |
| S4 | 4 | 移动平均平滑 (kernel=25) | 4 | 128 |
| S5 | 4 | 大窗口趋势 (kernel=32) | 4 | 128 |
| 总计 | | | 15 | |

**统一 k=4 的设计决策**：

多相分解存在刚性约束：分支数 $k$ = 分支内采样间距。这意味着 $k=16$ 时每分支仅32点（~3 patches），ScaleEncoder输入严重不足。v4的解决方案是**解耦"粒度控制"和"下采样率"**：
- S3/S4/S5统一使用 $k=4$，保证每分支128点（~16 patches），编码器输入充足
- "多尺度"效果通过**信号变换**实现：identity保留中频、smooth提取低频、trend提取宏观趋势
- 变换本身完成了频率分离，不需要依赖不同的下采样率

这一设计保证了每个分支的预测质量（充足的patch输入），同时通过变换的频率特性差异实现了有效的多尺度分析。

**分支预测**（共享ScaleEncoder）：

$$\hat{Y}_p^{(s)} = \Theta_s(\text{PatchEmbed}_s(\text{CreatePatch}(\tilde{X}_p^{(s)}))), \quad p = 0, \ldots, k_s-1$$

其中 $\Theta_s$ 为第 $s$ 个尺度的共享编码器参数（所有 $k_s$ 个分支共享同一组参数）。

**输出**：每个辅助尺度产生 $k_s$ 个分支预测 $\{\hat{Y}_p^{(s)}\}_{p=0}^{k_s-1}$，每个预测形状为 $[B, T_{pred}, N]$。

#### 3.1.3 实现架构

**文件位置**：`layers/TAPR_v4.py` — `PolyphaseMultiScale` 类

```python
class PolyphaseMultiScale(nn.Module):
    """
    DM v4：多相交错多尺度预测

    核心升级：平均池化 → 多相分解
    - 每个尺度(粒度k)产生k个分支预测
    - k个分支共享同一ScaleEncoder
    - 保证100%数据利用，无信息丢失

    输出与v3不同：返回分支预测字典，由C2F负责合并
    """
    SCALE_CFG = [
        (2,  'diff'),      # S2: 2分支
        (4,  'identity'),  # S3: 4分支
        (4,  'smooth'),    # S4: 4分支 (kernel=25)
        (4,  'trend'),     # S5: 4分支 (kernel=32)
    ]

    def forward(self, x_normed, pred_s1):
        branch_preds_dict = {}

        for idx, (k, transform_name) in enumerate(self.scale_cfgs):
            branch_preds = []
            for p in range(k):
                x_p = x_normed[:, p::k, :]  # 多相提取
                x_t, _ = TRANSFORM_REGISTRY[transform_name][0](x_p)
                # ... patch → embed → encode
                pred_p = encoder_s(embedded)
                branch_preds.append(pred_p)
            branch_preds_dict[idx] = branch_preds

        return pred_s1, branch_preds_dict
```

### 3.2 C2F：专家投票 + 层级粗到细融合（Expert Voting + Hierarchical Coarse-to-Fine Fusion）

#### 3.2.1 理论基础

**Stage 1 理论：鲁棒统计与专家投票**

v3的C2F直接对所有尺度预测做softmax加权，隐含假设所有预测均有参考价值。然而，某些分支可能由于数据噪声或模型偏差产生显著偏离的预测，直接加权融合会被这些"异常专家"拖累。

v4引入**专家投票机制**，基于鲁棒统计学中的MAD（Median Absolute Deviation）方法：

**MAD的优势**：
- MAD的崩溃点（breakdown point）为50%，即最多可容忍50%的异常值（而均值+标准差的崩溃点仅为0%）
- MAD是中位数绝对偏差的中位数，天然对称且对极端值不敏感
- 计算复杂度为 $O(k \log k)$（排序），远低于迭代式鲁棒估计方法

**文献支撑**：
- **MoE (Mixture of Experts)** 框架中的"门控"思想：不是所有专家对每个输入都有效，需要选择性激活 (Shazeer et al., 2017)
- **鲁棒统计** (Huber & Ronchetti, 2009)：中位数和MAD是最常用的鲁棒位置和尺度估计量

**Stage 2 理论：小波一致性约束**

v3使用KL散度比较各尺度预测的趋势方向分布。v4升级为基于小波变换的趋势一致性约束：

**DWT（离散小波变换）的优势**：
- 将信号分解为**低频趋势近似（approximation）**和**高频细节（detail）**两个分量
- 趋势近似直接反映信号的长期走向，比v3的"首尾差值+sigmoid"更准确
- 小波分析在信号处理中具有严格的数学基础（多分辨率分析理论，MRA）

**文献支撑**：
- **Autoformer** (NeurIPS 2021) 使用移动平均实现序列分解，而DWT提供了更精确的频率分离
- **FEDformer** (ICML 2022) 使用傅里叶变换实现频率分解，DWT在时间-频率局部化方面更优

#### 3.2.2 Stage 1 数学表述：尺度内专家投票

对每个辅助尺度 $s$ 的 $k_s$ 个分支预测 $\{\hat{Y}_p^{(s)}\}_{p=0}^{k_s-1}$：

**Step 1**：计算中位数参考预测

$$\hat{Y}_{med}^{(s)} = \text{median}(\{\hat{Y}_p^{(s)}\}_{p=0}^{k_s-1})$$

**Step 2**：计算偏差

$$D_p^{(s)} = |\hat{Y}_p^{(s)} - \hat{Y}_{med}^{(s)}|, \quad p = 0, \ldots, k_s-1$$

**Step 3**：MAD计算

$$\text{MAD}^{(s)} = \text{median}(\{D_p^{(s)}\}_{p=0}^{k_s-1})$$

**Step 4**：异常判断

$$\text{is\_outlier}_p^{(s)} = \mathbb{1}\left[\bar{D}_p^{(s)} > 3 \times \overline{\text{MAD}}^{(s)}\right]$$

其中 $\bar{D}_p^{(s)}$ 和 $\overline{\text{MAD}}^{(s)}$ 分别为对 $[B, T_{pred}, N]$ 维度取均值后的标量。

**Step 5**：投票合并

$$\hat{Y}_{voted}^{(s)} = \frac{1}{|\mathcal{V}_s|} \sum_{p \in \mathcal{V}_s} \hat{Y}_p^{(s)}, \quad \mathcal{V}_s = \{p : \neg \text{is\_outlier}_p^{(s)}\}$$

若 $|\mathcal{V}_s| = 0$（所有分支均为异常），退化为全部平均。

**修正率追踪**：

$$r_s = \frac{\sum_t n_{outliers}^{(s,t)}}{\sum_t k_s}, \quad \text{unreliable}_s = \mathbb{1}[r_s > 0.3]$$

#### 3.2.3 Stage 2 数学表述：跨尺度小波趋势约束

**Step 1**：DWT分解（Haar小波）

$$\text{approx}_s, \text{detail}_s = \text{DWT}(\hat{Y}_{voted}^{(s)})$$

Haar小波的一级分解：
$$\text{approx}[m] = \frac{x[2m] + x[2m+1]}{\sqrt{2}}, \quad \text{detail}[m] = \frac{x[2m] - x[2m+1]}{\sqrt{2}}$$

**Step 2**：趋势一致性评分

以最粗尺度（S5）的趋势近似为基准：

$$c_s = \cos(\text{approx}_s, \text{approx}_{S5}) = \frac{\langle \text{approx}_s, \text{approx}_{S5} \rangle}{\|\text{approx}_s\| \cdot \|\text{approx}_{S5}\|}$$

**Step 3**：权重调节

$$w_s^{adj} = \frac{\exp(W_{s,\cdot})}{\sum_j \exp(W_{j,\cdot})} \times \begin{cases} \gamma & \text{if } c_s < -0.3 \\ 1 & \text{otherwise} \end{cases} \times \begin{cases} \rho & \text{if unreliable}_s \\ 1 & \text{otherwise} \end{cases}$$

其中 $\gamma = 0.5$ 为趋势不一致衰减因子，$\rho = 0.3$ 为不可靠尺度惩罚因子。

**Step 4**：层级加权融合

$$\hat{Y}_{C2F} = \sum_{s=1}^{S} \frac{w_s^{adj}}{\sum_j w_j^{adj}} \odot \hat{Y}_{voted}^{(s)}$$

#### 3.2.4 实现架构

**文件位置**：`layers/TAPR_v4.py` — `ExpertVotingFusion` 类

```python
class ExpertVotingFusion(nn.Module):
    """
    C2F v4: 专家投票 + 层级趋势约束

    Stage 1: 尺度内MAD异常检测 + 投票合并
    Stage 2: DWT趋势一致性 + 可靠性调节 + softmax加权

    "权重矩阵" W: [n_scales, pred_len] — 与v3相同
    新增: 修正率追踪器、DWT趋势约束
    """
    def __init__(self, n_scales, pred_len, decay_factor=0.5,
                 reliability_penalty=0.3, mad_threshold=3.0):
        self.weight_matrix = nn.Parameter(
            torch.ones(n_scales, pred_len) / n_scales
        )

    def forward(self, pred_s1, branch_preds_dict, training=True):
        # Stage 1: 投票
        consolidated = [pred_s1]
        reliability_flags = [False]
        for s_idx, branches in branch_preds_dict.items():
            voted, n_outliers = self.intra_scale_expert_voting(branches)
            consolidated.append(voted)
            reliability_flags.append(...)

        # Stage 2: 趋势约束融合
        weighted_pred = self.cross_scale_trend_constraint(
            consolidated, reliability_flags
        )
        return weighted_pred
```

---

## 4. 模块二：GRA-M v4 全局检索增强记忆

**GRA-M (Global Retrieval-Augmented Memory)** v4版本在保持v3"多尺度检索+双矩阵门控"架构的基础上，升级PVDR的检测能力和AKG的门控灵活性。

### 4.1 PVDR：增强检测 + 双重检索（Enhanced Detection + Dual Retrieval）

#### 4.1.1 理论基础

**理论一：逻辑回归极端分类**

v3使用固定的 $3\sigma$ 规则检测极端数据：$\text{has\_extreme} = \exists_{t,n}: |z_{t,n}| > 3$。这一规则隐含假设数据服从正态分布，但实际时序数据往往具有厚尾分布（如金融数据的尖峰厚尾特性），导致 $3\sigma$ 规则的假阳性率和假阴性率不稳定。

v4将极端检测建模为一个**二分类问题**，使用逻辑回归（Logistic Regression）从5个统计特征中学习判别边界：

$$P(\text{extreme} | \mathbf{f}) = \sigma(\mathbf{w}^T \mathbf{f} + b)$$

其中 $\mathbf{f} = [\mu, \sigma, x_{max}, x_{min}, x_T - x_1] \in \mathbb{R}^5$ 为统计特征向量。

**逻辑回归的适用性**：
- 仅6个可学习参数（5个权重 + 1个偏置），不会增加过拟合风险
- 决策边界是特征空间中的超平面，具有良好的可解释性
- 通过训练数据中的 $z$-score 标签进行监督学习，能够适应不同数据分布

**理论二：贝叶斯变点检测（Bayesian Change Point Detection）**

时间序列中的**结构性变点**（change point）指数据生成过程的统计特性在某一时刻发生突变。例如：电力负荷在极端天气时的突变、股票在重大事件后的走势反转。

Adams & MacKay (2007) 提出的贝叶斯在线变点检测框架基于以下模型：

假设序列被划分为若干段（segment），每段内数据服从同一分布，段间分布不同。对于候选分割点 $\tau$，计算**贝叶斯因子**：

$$\text{BF}(\tau) = \frac{p(\text{data} | \text{分割于}\tau)}{p(\text{data} | \text{不分割})} = \frac{p(x_{1:\tau}) \cdot p(x_{\tau+1:T})}{p(x_{1:T})}$$

当 $\text{BF}(\tau) > 1$ 时，认为在 $\tau$ 处分割优于不分割。

v4使用**递归二分**策略：在每个候选段内寻找最优分割点，若贝叶斯因子 > 1则分割并递归处理子段，最大递归深度为4。

**正态-正态共轭模型**：

假设每段数据服从正态分布 $x_i \sim \mathcal{N}(\mu, \sigma^2)$，先验为 $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$，则边际似然有解析形式：

$$\log p(x_{1:n}) = -\frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2}\log\frac{\sigma_{post}^2}{\sigma_0^2} - \frac{1}{2}\left(\frac{n \cdot s^2}{\sigma^2} + \frac{n\bar{x}^2}{\sigma^2} - \frac{\mu_{post}^2}{\sigma_{post}^2} + \frac{\mu_0^2}{\sigma_0^2}\right)$$

其中 $\sigma_{post}^2 = (1/\sigma_0^2 + n/\sigma^2)^{-1}$，$\mu_{post} = \sigma_{post}^2 \cdot (\mu_0/\sigma_0^2 + n\bar{x}/\sigma^2)$。

先验超参 $\mu_0, \sigma_0^2, \sigma^2$ 设为**可学习参数**（~3个），通过主损失的反向传播自动调节。

**变点检测在时序预测中的意义**：

当检测到序列末尾附近存在变点时，意味着数据的生成过程刚刚发生了转变。此时，模型基于历史窗口内旧分布学到的模式可能不再适用，应该更多依赖检索到的类似"转折点后"的历史片段。

#### 4.1.2 增强检索策略

基于逻辑回归和变点检测的结果，PVDR v4实现四种检索模式：

| 模式 | 触发条件 | 检索策略 | 理论依据 |
|------|---------|---------|---------|
| **直接修正** | $\max_{sim} > 0.95$ | Top-1直接使用 | 极高相似度意味着历史几乎重现 |
| **极端检索** | $P(\text{extreme}) > \tau_{ext}$ | 降低阈值，扩大检索量，优先"extreme"标签 | 极端场景下模型推理不可靠 |
| **转折检索** | $\text{near\_end\_score} > \tau_{cp}$ | 从"turning_point"子集优先检索 | 转折后的走势需要特殊历史参考 |
| **标准检索** | 其它情况 | 标准Top-K | v3默认行为 |

#### 4.1.3 记忆库标签机制

v4的记忆库在v3的key-value基础上增加label字段：

$$\mathcal{M}_s = \{(k_i^{(s)}, v_i^{(s)}, l_i^{(s)})\}_{i=1}^{M}$$

其中 $l_i \in \{0: \text{normal}, 1: \text{extreme}, 2: \text{turning\_point}\}$。

标签在记忆库构建阶段由逻辑回归分类器和变点检测器自动标注。

#### 4.1.4 PVDR v4信号输出

PVDR v4除返回 `hist_refs`（与v3兼容）外，还返回丰富的检测信号供AKG使用：

$$\text{pvdr\_signals} = \{e_b, c_b, \mathbf{q}_{b,s}\}_{b=1}^{B}$$

- $e_b \in [0,1]$：样本 $b$ 的极端概率
- $c_b \in [0,1]$：样本 $b$ 末尾附近有变点的概率
- $q_{b,s} \in [0,1]$：样本 $b$ 在尺度 $s$ 上的检索质量（最高相似度）

#### 4.1.5 实现架构

**文件位置**：`layers/GRAM_v4.py` — `EnhancedDualRetriever` 类

```python
class EnhancedDualRetriever(nn.Module):
    """
    PVDR v4: 增强检测 + 双重检索

    核心升级:
    1. 3-sigma → LogisticExtremeClassifier (可学习边界)
    2. 新增 BayesianChangePointDetector (结构性变点)
    3. 检索策略分化: direct/extreme/turning_point/normal
    4. 记忆库标签机制: normal/extreme/turning_point
    """
    def retrieve_all_scales(self, x_normed, downsample_rates):
        # 增强检测
        extreme_prob = self.extreme_classifier(x_normed)
        _, _, near_end_score = self.cp_detector(x_normed)

        # 分模式检索
        hist_refs = []
        retrieval_conf = []
        for s, ds_rate in enumerate(downsample_rates):
            # ... 根据检测结果选择检索模式
            hist_refs.append(hist_ref)
            retrieval_conf.append(max_sim)

        pvdr_signals = {
            'extreme_score': extreme_prob,
            'change_point_score': near_end_score,
            'retrieval_confidence': torch.stack(retrieval_conf, dim=-1),
        }
        return hist_refs, pvdr_signals
```

### 4.2 AKG：阈值驱动的自适应门控（Threshold-driven Adaptive Gating）

#### 4.2.1 理论基础

**核心论文**：TS-RAG (NeurIPS 2025)、TFT (IJF 2021)

v3的AKG使用固定的 $\sigma(C)$ 门控，训练完成后门控值不再随输入变化。这意味着无论输入是正常数据还是极端数据，门控行为完全相同。

v4引入**条件门控**（Conditional Gating）的思想：门控值不仅取决于可学习参数 $C$，还取决于当前输入的检测信号。具体地，PVDR的三种信号（极端概率、变点概率、检索置信度）通过**软阈值函数**调节基础门控：

$$g_{eff} = \sigma(C) \times (1 - a_{ext}) \times (1 - a_{cp}) \times a_{conf}$$

其中：
$$a_{ext} = \sigma(\beta \cdot (e - \tau_{ext}))$$
$$a_{cp} = \sigma(\beta \cdot (c - \tau_{cp}))$$
$$a_{conf} = \sigma(\beta \cdot (q - \tau_{conf}))$$

$\beta = 10$ 为温度参数（使软阈值接近硬阈值但保持可微），$\tau_{ext}, \tau_{cp}, \tau_{conf}$ 为可学习阈值参数。

**门控语义**：

| 信号 | 调节效果 | 直觉解释 |
|------|---------|---------|
| 极端概率 $e$ 高 | $a_{ext} \to 1$, $g_{eff}$ 降低 | 极端数据时更信任历史经验 |
| 变点概率 $c$ 高 | $a_{cp} \to 1$, $g_{eff}$ 降低 | 转折点附近更信任历史参考 |
| 检索置信度 $q$ 高 | $a_{conf} \to 1$, $g_{eff}$ 保持 | 检索质量好时允许历史影响 |
| 检索置信度 $q$ 低 | $a_{conf} \to 0$, $g_{eff}$ 降低 | 检索质量差时减少历史影响 |

**设计原则**：
- 当PVDR信号表明当前输入"特殊"（极端或转折）且检索到了可靠的历史参考时，门控倾向于更多信任历史
- 当PVDR信号表明当前输入"正常"或检索质量差时，退化为v3的基础门控行为

#### 4.2.2 数学表述

**Stage 1：联系矩阵 $C$（增强版）**

基础门控（与v3相同）：
$$g_{base} = \sigma(C), \quad C \in \mathbb{R}^{S \times T_{pred}}$$

PVDR信号调节：
$$g_{eff}^{(s,t)} = g_{base}^{(s,t)} \times (1 - \sigma(\beta(e - \tau_{ext}))) \times (1 - \sigma(\beta(c - \tau_{cp}))) \times \sigma(\beta(q_s - \tau_{conf}))$$

门控融合：
$$Y_{conn}^{(s)} = g_{eff}^{(s)} \odot \hat{Y}^{(s)} + (1 - g_{eff}^{(s)}) \odot V^{(s)}$$

**Stage 2：融合矩阵 $F$（与v3完全相同）**

$$Y_{final} = \sum_{i=1}^{2S} \text{softmax}(F)_i \odot Z_i, \quad Z = [\hat{Y}^{(1..S)}, Y_{conn}^{(1..S)}]$$

#### 4.2.3 门控正则化（与v3相同 + 新增阈值约束）

$$\mathcal{L}_{gate} = -\text{mean}(|\sigma(C) - 0.5|)$$

阈值参数 $\tau_{ext}, \tau_{cp}, \tau_{conf}$ 通过主损失的反向传播自动学习最优阈值位置。

#### 4.2.4 实现架构

**文件位置**：`layers/GRAM_v4.py` — `ThresholdAdaptiveGating` 类

```python
class ThresholdAdaptiveGating(nn.Module):
    """
    AKG v4: 阈值驱动的自适应门控

    核心升级: sigmoid(C) → sigmoid(C) × PVDR信号调节

    新增参数:
    - extreme_threshold: τ_extreme (可学习)
    - cp_threshold: τ_cp (可学习)
    - conf_threshold: τ_conf (可学习)
    """
    def __init__(self, n_scales, pred_len):
        self.connection_matrix = nn.Parameter(torch.zeros(n_scales, pred_len))
        self.fusion_matrix = nn.Parameter(torch.zeros(2 * n_scales, pred_len))
        # NEW
        self.extreme_threshold = nn.Parameter(torch.tensor(0.5))
        self.cp_threshold = nn.Parameter(torch.tensor(0.5))
        self.conf_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, scale_preds, hist_refs, pvdr_signals=None):
        base_gate = torch.sigmoid(self.connection_matrix)

        if pvdr_signals is not None:
            effective_gate = self.compute_effective_gate(base_gate, pvdr_signals)
        else:
            effective_gate = base_gate  # 降级为v3行为

        # Stage 1: 门控
        connected = [g * pred + (1-g) * hist for ...]

        # Stage 2: 融合 (与v3完全相同)
        all_sources = scale_preds + connected
        final_pred = softmax_weighted_sum(all_sources, self.fusion_matrix)
        return final_pred
```

---

## 5. 可训练参数：理论分析

### 5.1 参数总览

v4框架的可训练参数分为以下类别：

| 参数类型 | 符号 | 形状 | 数量 | 来源 |
|----------|------|------|------|------|
| **参数矩阵** | $\Theta_s$ | Conv/Linear参数集合 | ~25K × 4 = ~100K | v3沿用 |
| **权重矩阵** | $W$ | $[S, T_{pred}]$ | 480 | v3沿用 |
| **联系矩阵** | $C$ | $[S, T_{pred}]$ | 480 | v3沿用 |
| **融合矩阵** | $F$ | $[2S, T_{pred}]$ | 960 | v3沿用 |
| 逻辑回归 | $\mathbf{w}, b$ | Linear(5,1) | 6 | v4新增 |
| 贝叶斯CP先验 | $\mu_0, \sigma_0^2, \sigma^2$ | 3标量 | 3 | v4新增 |
| AKG阈值 | $\tau_{ext}, \tau_{cp}, \tau_{conf}$ | 3标量 | 3 | v4新增 |
| 记忆库阈值 | threshold per scale | $[S]$ | 5 | v3沿用 |
| **总可训练** | | | **~103K + ~12** | |

### 5.2 参数共享分析

v4的关键设计是**ScaleEncoder在分支间共享**：

| 尺度 | 分支数 $k$ | ScaleEncoder数 | 共享方式 | 等效数据增强倍数 |
|------|-----------|---------------|---------|----------------|
| S2 | 2 | 1 | 2分支共享 | 2× |
| S3 | 4 | 1 | 4分支共享 | 4× |
| S4 | 4 | 1 | 4分支共享 | 4× |
| S5 | 4 | 1 | 4分支共享 | 4× |

这意味着v4与v3具有**完全相同的可训练参数量**（ScaleEncoder层面），但每个编码器接收到了更多的训练样本（通过多相分支的"数据增强"效应）。

### 5.3 v3 vs v4 参数对比

| 组件 | v3 | v4 | 变化 |
|------|-----|-----|------|
| ScaleEncoders | ~100K | ~100K | 不变 (共享) |
| C2F W | 480 | 480 | 不变 |
| 极端检测 | 0 (规则) | 6 | +6 |
| 变点检测 | 0 | 3 | +3 |
| AKG C | 480 | 480 | 不变 |
| AKG F | 960 | 960 | 不变 |
| AKG阈值 | 0 | 3 | +3 |
| **总新增** | 0 | **~12** | **可忽略** |

### 5.4 计算量分析

| 操作 | v3 | v4 | 分析 |
|------|-----|-----|------|
| ScaleEncoder前向传播 | 4次 × $T/k$ | 14次 × $T/k$ | 次数增加 (2+4+4+4)，但每次数据量相同 |
| 总处理数据量 | $\sum T/k_s$ | $\sum k_s \times (T/k_s) = 4T$ | **相同** |
| MAD投票 | 0 | 4次排序 | $O(k \log k)$，轻量 |
| DWT分解 | 0 | 5次 | $O(T_{pred})$，轻量 |
| 逻辑回归 | 0 | 1次 5→1线性 | 可忽略 |
| 贝叶斯变点 | 0 | 递归二分 | $O(T \log T)$，中等 |
| AKG门控调节 | 0 | 3次逐元素运算 | 可忽略 |

---

## 6. 损失函数设计

### 6.1 总损失公式

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \alpha \cdot \left(\lambda_{consist} \cdot \mathcal{L}_{consist} + \lambda_{gate} \cdot \mathcal{L}_{gate} + \lambda_{expert} \cdot \mathcal{L}_{expert}\right)$$

其中 $\alpha = \min(1.0, \text{step}/500)$ 为warmup因子。

### 6.2 各损失分量

**主损失（不变）**：

$$\mathcal{L}_{main} = \frac{1}{B \cdot T_{pred} \cdot N} \sum_{b,t,n} (\hat{Y}_{b,t,n} - Y_{b,t,n})^2$$

**一致性损失（v4升级：KL散度 → 小波趋势一致性）**：

$$\mathcal{L}_{consist} = \frac{1}{S} \sum_{s=1}^{S} \max\left(0, -\cos(\text{approx}_s, \text{approx}_{S5})\right)$$

v3使用KL散度比较趋势方向分布（down/flat/up三类概率），v4直接使用小波趋势近似的余弦相似度。优势在于：
- 直接衡量连续趋势向量的方向一致性，而非离散化的三类概率
- 仅在余弦相似度为负（趋势明显相反）时产生惩罚，允许合理的细微差异

**门控正则化（不变）**：

$$\mathcal{L}_{gate} = -\frac{1}{S \cdot T_{pred}} \sum_{s,t} |\sigma(C_{s,t}) - 0.5|$$

**专家修正损失（v4新增）**：

$$\mathcal{L}_{expert} = \frac{1}{S-1} \sum_{s=2}^{S} r_s$$

其中 $r_s$ 为尺度 $s$ 的修正率（被MAD投票剔除的分支比例）。此损失鼓励各分支预测趋于一致，间接促进ScaleEncoder的学习质量。

### 6.3 损失权重

| 超参数 | 符号 | 推荐值 | 作用 |
|--------|------|--------|------|
| 一致性权重 | $\lambda_{consist}$ | 0.1 | 跨尺度趋势一致性 |
| 门控正则权重 | $\lambda_{gate}$ | 0.01 | 门控决断性 |
| 专家修正权重 | $\lambda_{expert}$ | 0.05 | 分支一致性 |
| Warmup步数 | - | 500 | 辅助损失渐进启动 |

---

## 7. 系统架构与数据流

### 7.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Time-LLM Enhanced v4                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  原始Time-LLM主流程（完全保持，不做任何修改）                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Input → Norm → CI → PatchEmbed → Reprogram → LLM →     │    │
│  │                            FlattenHead → pred_s1        │    │
│  └──────────────────────────────────┬──────────────────────┘    │
│                                     │                            │
│  v4新增模块（在baseline输出后处理）  │                            │
│  ┌──────────────────────────────────┴──────────────────────┐    │
│  │                                                         │    │
│  │  ┌───────────────────────┐                              │    │
│  │  │ DM v4: 多相多尺度预测  │                              │    │
│  │  │ S2: 2分支 (diff)      │                              │    │
│  │  │ S3: 4分支 (identity)  │                              │    │
│  │  │ S4: 4分支 (smooth)    │                              │    │
│  │  │ S5: 4分支 (trend)     │                              │    │
│  │  │ "参数矩阵" Θ_s (共享) │                              │    │
│  │  └───────────┬───────────┘                              │    │
│  │              │ 14个分支预测                               │    │
│  │              ▼                                          │    │
│  │  ┌───────────────────────┐                              │    │
│  │  │ C2F v4 Stage1: 投票   │                              │    │
│  │  │ MAD异常检测 → 剔除     │                              │    │
│  │  │ 修正率追踪             │                              │    │
│  │  └───────────┬───────────┘                              │    │
│  │              │ consolidated: [s1, voted_s2..s5]          │    │
│  │              ▼                                          │    │
│  │  ┌───────────────────────┐                              │    │
│  │  │ C2F v4 Stage2: 趋势   │                              │    │
│  │  │ DWT分解 + 一致性检查   │                              │    │
│  │  │ "权重矩阵" W [5, 96]  │                              │    │
│  │  │ × reliability × cosine│                              │    │
│  │  └───────────┬───────────┘                              │    │
│  │              │ weighted_pred                             │    │
│  │              │                                          │    │
│  │  ┌───────────┴───────────┐                              │    │
│  │  │ PVDR v4: 增强双重检索  │                              │    │
│  │  │ 逻辑回归极端分类       │                              │    │
│  │  │ 贝叶斯变点检测         │                              │    │
│  │  │ 分模式检索             │                              │    │
│  │  │ → hist_refs + signals │                              │    │
│  │  └───────────┬───────────┘                              │    │
│  │              │                                          │    │
│  │  ┌───────────┴───────────┐                              │    │
│  │  │ AKG v4: 阈值驱动门控  │                              │    │
│  │  │ base_gate = σ(C)      │                              │    │
│  │  │ × (1-extreme_adj)     │                              │    │
│  │  │ × (1-cp_adj)          │                              │    │
│  │  │ × conf_adj            │                              │    │
│  │  │ "融合矩阵" F [10, 96] │                              │    │
│  │  │ → final_pred          │                              │    │
│  │  └───────────┬───────────┘                              │    │
│  └──────────────┼──────────────────────────────────────────┘    │
│                 │                                                │
│                 ▼                                                │
│           Instance Denorm → Output [B, pred_len, N]             │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 数据流公式

完整的v4预测过程可用如下数学表达：

**Baseline（不变）**：
$$\hat{Y}^{(1)} = f_{baseline}(X)$$

**DM v4 多相分解**：
$$X_p^{(s)} = \{x[k_s m + p]\}_{m}, \quad \hat{Y}_p^{(s)} = \Theta_s(\text{Transform}_s(X_p^{(s)})), \quad p = 0, \ldots, k_s-1$$

**C2F v4 Stage1 专家投票**：
$$\hat{Y}_{voted}^{(s)} = \frac{1}{|\mathcal{V}_s|} \sum_{p \in \mathcal{V}_s} \hat{Y}_p^{(s)}, \quad \mathcal{V}_s = \{p : \bar{D}_p \leq 3 \cdot \overline{\text{MAD}}\}$$

**C2F v4 Stage2 趋势约束融合**：
$$\hat{Y}_{C2F} = \sum_{s=1}^{S} \frac{w_s^{adj}}{\sum_j w_j^{adj}} \odot \hat{Y}_{voted}^{(s)}$$

**PVDR v4 增强检索**：
$$e = \sigma(\mathbf{w}^T\mathbf{f} + b), \quad c = \text{BayesCP}(X), \quad V^{(s)} = \text{Retrieve}(X, s, e, c)$$

**AKG v4 阈值门控**：
$$g_{eff} = \sigma(C) \times (1 - \sigma(\beta(e - \tau_{ext}))) \times (1 - \sigma(\beta(c - \tau_{cp}))) \times \sigma(\beta(q - \tau_{conf}))$$

$$Y_{conn}^{(s)} = g_{eff}^{(s)} \odot \hat{Y}^{(s)} + (1 - g_{eff}^{(s)}) \odot V^{(s)}$$

$$\hat{Y}_{final} = \sum_{i=1}^{2S} \text{softmax}(F)_i \odot Z_i, \quad Z = [\hat{Y}^{(1..S)}, Y_{conn}^{(1..S)}]$$

### 7.3 数据形状流 (ETTh1, B=4, T=512, N=7, P=96)

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 | [4, 512, 7] | B=4, T=512, N=7 |
| Baseline输出 | [4, 96, 7] | pred_s1 |
| DM S2分支 | 2 × [4, 96, 7] | 2个多相分支预测 |
| DM S3分支 | 4 × [4, 96, 7] | 4个多相分支预测 |
| DM S4分支 | 4 × [4, 96, 7] | 4个多相分支预测 |
| DM S5分支 | 4 × [4, 96, 7] | 4个多相分支预测 |
| C2F Stage1投票后 | 5 × [4, 96, 7] | consolidated_preds |
| C2F Stage2融合后 | [4, 96, 7] | weighted_pred |
| PVDR极端分类 | [4] | extreme_prob |
| PVDR变点检测 | [4] | near_end_score |
| PVDR检索 | 5 × [4, 96, 7] | hist_refs |
| AKG有效门控 | [4, 5, 96] | effective_gate (per-batch) |
| AKG最终输出 | [4, 96, 7] | final_pred |
| 反归一化 | [4, 96, 7] | 输出 |

---

## 8. 实验设计

### 8.1 主消融实验矩阵

| 实验编号 | TAPR | GRA-M | 预期效果 |
|----------|------|-------|----------|
| E1 (Baseline) | ✗ | ✗ | 原始Time-LLM性能 |
| E2 (+TAPR v4) | ✓ | ✗ | 验证多相预测+投票融合效果 |
| E3 (+GRAM v4) | ✗ | ✓ | 验证增强检索+阈值门控效果 |
| E4 (Full v4) | ✓ | ✓ | 完整增强模型 |

### 8.2 v3 vs v4 模块对比消融

| 实验 | DM版本 | C2F版本 | PVDR版本 | AKG版本 | 预期观察 |
|------|--------|---------|----------|---------|---------|
| E5a | v3 (avg_pool) | v3 | v3 | v3 | v3 baseline |
| E5b | **v4 (polyphase)** | v3 | v3 | v3 | 多相采样增益 |
| E6a | v4 | v3 (直接加权) | v3 | v3 | 无投票 |
| E6b | v4 | **v4 (投票+趋势)** | v3 | v3 | 投票+趋势约束增益 |
| E7a | v4 | v4 | v3 (3-sigma) | v3 | 无增强检测 |
| E7b | v4 | v4 | **v4 (logistic+CP)** | v3 | 增强检测增益 |
| E8a | v4 | v4 | v4 | v3 (固定sigmoid) | 无阈值门控 |
| E8b | v4 | v4 | v4 | **v4 (阈值门控)** | 阈值门控增益 |

### 8.3 参数敏感性实验

| 实验 | 参数 | 测试值 | 预期观察 |
|------|------|--------|---------|
| P1 | mad_threshold | 2.0, 3.0, 4.0 | MAD异常检测灵敏度 |
| P2 | reliability_threshold | 0.2, 0.3, 0.5 | 尺度可靠性判断标准 |
| P3 | cp_max_depth | 2, 4, 6 | 变点检测精细度 |
| P4 | lambda_expert | 0.01, 0.05, 0.1 | 专家修正损失强度 |

### 8.4 关键评估指标

- **主任务指标**：MSE, MAE
- **辅助指标**：
  - 一致性损失值：反映各尺度趋势一致程度
  - 门控分布统计：反映AKG是否做出有意义的决策
  - 权重矩阵可视化：反映各尺度在不同时间步的贡献
  - **修正率统计**（v4新增）：每个尺度被剔除分支的比例
  - **检索模式分布**（v4新增）：四种检索模式的触发频率
  - **变点检测准确率**（v4新增）：与事后标注变点的重合度

### 8.5 预期结果

| 实验 | 预期MSE (ETTh1, 96步) | 相对v3变化 |
|------|----------------------|-----------|
| E1 (Baseline) | ~0.38 | - |
| v3 Full | ~0.32-0.34 | ↓10-15% |
| v4 Full | ~0.28-0.32 | ↓15-25% (相对baseline) |
| v4 vs v3 | - | ↓5-10% (相对v3) |

**v4相对v3的增益来源**：
1. DM多相采样：消除信息损失 → ↓1-3%
2. C2F投票+趋势：提高融合鲁棒性 → ↓1-2%
3. PVDR增强检测：更精准的检索策略 → ↓1-3%
4. AKG阈值门控：动态适应输入特性 → ↓1-2%

---

## 9. 参考文献

### 9.1 核心引用

| 技术点 | 论文 | 引用要点 |
|--------|------|---------|
| 多相分解理论 | **Vaidyanathan** (1993), Multirate Systems and Filter Banks | 完美重构滤波器组的理论基础 |
| 多尺度下采样+独立预测+融合 | **TimeMixer** (ICLR 2024) | 不同尺度呈现不同模式 |
| 插值对齐到统一预测长度 | **N-HiTS** (AAAI 2023) | 分层插值+多速率采样 |
| 跨尺度归一化 | **Scaleformer** (ICLR 2023) | 防止跨尺度分布漂移 |
| 粗到细条件分解 | **C2FAR** (NeurIPS 2022) | 细粒度落在粗粒度区间内 |
| 贝叶斯变点检测 | **Adams & MacKay** (2007) | 在线贝叶斯变点检测框架 |
| 鲁棒统计方法 | **Huber & Ronchetti** (2009) | MAD等鲁棒估计量 |
| MoE门控选择 | **Shazeer et al.** (2017) | 混合专家的选择性激活 |
| 检索增强时序预测 | **RAFT** (ICML 2025) | 检索相似历史+后续值作为参考 |
| 自适应检索混合器 | **TS-RAG** (NeurIPS 2025) | ARM模块自适应分配权重 |
| GLU门控融合 | **TFT** (IJF 2021) | 门控残差网络选择性融合 |
| 信号分解 | **Autoformer** (NeurIPS 2021) | 序列分解为趋势+季节性 |
| 频率分解 | **FEDformer** (ICML 2022) | 傅里叶增强分解 |
| LLM重编程 | **Time-LLM** (ICLR 2024) | 冻结LLM的时序重编程 |

### 9.2 与现有工作的区别

| 方法 | 核心思想 | 与本文区别 |
|------|----------|-----------|
| TimeMixer (ICLR 2024) | 多尺度混合 | 本文使用多相分解替代平均池化，保证信息完整性 |
| N-HiTS (AAAI 2023) | 分层插值 | 本文增加专家投票剔除异常分支 |
| RAFT (ICML 2025) | 检索增强 | 本文结合逻辑回归+贝叶斯变点实现分模式检索 |
| TS-RAG (NeurIPS 2025) | 自适应检索 | 本文使用PVDR信号驱动的阈值门控 |
| Time-LLM (ICLR 2024) | LLM重编程 | 本文保持baseline不变，增加多相多尺度+增强检索 |
| Adams & MacKay (2007) | 在线变点检测 | 本文将变点检测嵌入检索增强预测框架 |

**本文核心贡献**：
1. 首次将多相分解引入LLM时序预测框架的多尺度分析
2. 提出专家投票+小波趋势约束的两阶段鲁棒融合机制
3. 将贝叶斯变点检测与检索增强预测有机结合
4. 设计PVDR信号驱动的条件门控机制

---

## 附录A：关键术语对照

| 英文 | 中文 | 说明 |
|------|------|------|
| Polyphase Decomposition | 多相分解 | 信号处理中的多速率采样理论 |
| MAD (Median Absolute Deviation) | 中位数绝对偏差 | 鲁棒尺度估计量 |
| DWT (Discrete Wavelet Transform) | 离散小波变换 | 信号的时频分析工具 |
| Bayesian Change Point Detection | 贝叶斯变点检测 | 序列结构性变化点的概率检测 |
| Logistic Regression | 逻辑回归 | 二分类判别模型 |
| Bayes Factor | 贝叶斯因子 | 模型比较的概率度量 |
| Expert Voting | 专家投票 | 多模型融合的鲁棒策略 |
| Conditional Gating | 条件门控 | 根据输入信号动态调节门控 |

---

**文档版本**: 4.0 (Academic Edition)
**最后更新**: 2026-02-24
**生成工具**: Claude Code (Opus 4.6)
