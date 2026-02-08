# 新增模块统一合并与整理（基于现有 mokuai*.md）

> 目标：将现有文档中的多种命名与设计思路统一为两个可实施模块，形成可直接进入实现与实验阶段的规范版本。

---

## 一、统一命名与来源映射

**统一模块名称**
- 模块一：多尺度分解-混合重编程模块（Multi-scale Decomposition & Fusion Reprogramming, MDFR）
- 模块二：检索增强的层级预测模块（Retrieval-Augmented Hierarchical Forecasting, RAHF）

**来源映射（学术裁缝式合并）**

| 统一模块 | 旧命名与来源 | 统一后的功能定位 |
|---|---|---|
| MDFR | MTDM（mokuai.md）、MSARM（mokuaii.md）、多尺度金字塔/分解混合（mokuai-1.md） | 多尺度特征分解、跨尺度混合、统一patch语义输出 |
| RAHF | HTAM/HTPM/HPPM/PGPM（mokuai-1.md/mokuaii.md/mokuai.md/mokuaiii.md） + RATM/RAPM（mokuai-1.md/mokuaiii.md） | 方向-幅度-数值的层级预测 + 历史相似模式检索增强 |

---

## 二、统一架构定位与数据流

**基线管线**
```
X -> RevIN -> Patching -> Reprogramming -> Frozen LLM -> FlattenHead -> Y_hat
```

**统一后的插入位置**
```
X -> RevIN -> MDFR -> Reprogramming -> Frozen LLM -> RAHF -> Y_hat
```

- MDFR 位于输入侧，用于多尺度表达与跨尺度融合，再进入重编程层。
- RAHF 位于输出侧，用于检索增强与层级预测，替代或增强FlattenHead。

---

## 三、模块一：MDFR（多尺度分解-混合重编程）

**解决的问题**
- 单尺度patch导致跨尺度信息缺失，长视界预测误差累积。
- 周期、趋势、高频扰动无法同时建模。

**核心设计（融合自mokuai.md与mokuaii.md）**
1. 多尺度金字塔输入，尺度集合如 `{1, 2, 4}` 或 `{1, 4, 16}`。
2. 尺度自适应Patch，保证不同尺度下的局部语义可比。
3. 先分解趋势与季节，再进行双向混合（细→粗与粗→细）。
4. 使用门控或注意力融合为统一patch表示，送入重编程层。

**输入输出约定**
- 输入：`X [B, T, N]`
- 输出：`H_fused [B*N, L, D]`，与原重编程层输入对齐。

**建议可控超参**
- `scales`、`patch_len_base`、`stride_base`
- `use_decomp`、`use_bimix`、`fusion_type`

---

## 四、模块二：RAHF（检索增强层级预测）

**解决的问题**
- 直接回归缺乏趋势方向约束，拐点误差大。
- 历史相似模式未被利用，泛化能力受限。

**核心设计（融合自mokuai-1.md与mokuaiiii.md）**
1. 层级预测子模块
- Level-1：趋势方向分类（上涨/下降/平稳或5类细化）。
- Level-2：幅度区间回归或分段分类。
- Level-3：残差细化回归，输出最终数值。

2. 检索增强子模块
- 构建历史模式记忆库（Pattern Memory Bank），支持Top-k检索。
- 使用可检索时序编码器生成查询向量。
- 检索得到的历史未来片段作为参考，使用交叉注意力或门控融合。

**输入输出约定**
- 输入：`Z [B, L, d_llm]`（LLM输出特征）与检索参考 `M`。
- 输出：最终预测 `Y_hat [B, pred_len, N]`。

**建议可控超参**
- `retrieval_top_k`、`memory_size`
- `trend_bins`、`tau1/tau2`、`loss_weights`
- `fusion_type`（gate / cross-attn / concat）

---

## 五、统一损失与训练策略

**统一损失表达**
- 主损失：`L_mse`（或`L_mae`）
- 层级损失：`L_dir` + `L_mag`
- 组合：`L = L_mse + λ1 L_dir + λ2 L_mag`

**训练流程建议**
1. 固定LLM主干，仅训练MDFR与RAHF参数。
2. 先单模块预热，再联合训练。
3. 检索记忆库仅由训练集构建，避免信息泄漏。

---

## 六、统一实现准备清单

**推荐文件结构**
- `layers/MDFR.py`
- `layers/RAHF.py`
- `models/TimeLLM.py` 中新增模块开关与调用路径

**推荐配置开关**
- `--use_mdfr`
- `--use_rahf`
- `--scales 1,2,4`
- `--retrieval_top_k 5`
- `--trend_bins 5`

---

## 七、统一结论与后续准备

- 现有文档中的多个命名已被统一为 MDFR 与 RAHF 两个模块，避免后续实现混乱。
- MDFR主攻输入侧多尺度表达，RAHF主攻输出侧检索与层级推理，二者互补且低侵入。
- 下一步可直接进入代码实现、消融实验与论文撰写。

