# Time-LLM：从数据集到跨模态对齐前的处理推导

本记录聚焦“数据集取样 → 批处理 → Patch 分块与嵌入 → 跨模态对齐前”的全流程，结合项目源码逐步解释。

---

## 0. 符号与维度约定

- `B`：batch size（一次迭代处理的样本数）
- `T` / `SeqLen`：输入序列长度（`args.seq_len`）
- `N` / `N_vars`：变量数（多变量通道数）
- `patch_len`：每个 patch 的长度
- `stride`：patch 之间的滑动步长
- `num_patches`：patch 数量
- `d_model`：patch embedding 维度
- `llm_dim`：LLM 词向量维度
- `prompt_len`：prompt token 数（每个样本对应一个文本提示）

**重要现实：**
当前数据集实现中（`Dataset_ETT_*` / `Dataset_Custom`），**每个样本只返回 1 个变量**，即 `N=1`。多变量设置通过 `feat_id` 把不同变量当作不同样本（见 `data_provider/data_loader.py:90`）。

---

## 1. 数据集 → DataLoader（一次处理多少数据）

### 1.1 数据集选择与 DataLoader 组装
- 入口：`data_provider/data_factory.py:16`
- 根据 `args.data` 选择 Dataset 类，按 `args.batch_size` 构建 DataLoader。

```python
# data_provider/data_factory.py:16-63
Data = data_dict[args.data]
...
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)
```

**结论：一次处理的数据量就是 `batch_size = args.batch_size` 条样本。**

### 1.2 数据读取与标准化
- `Dataset_ETT_hour.__read_data__`（`data_provider/data_loader.py:46-87`）：
  - 读 CSV → 选择特征列 → 用训练集统计量做 `StandardScaler` 标准化
  - 构造时间特征 `data_stamp`（月、日、星期、小时等）

### 1.3 单个样本的组成（滑动窗口取样）
- `__getitem__`（`data_provider/data_loader.py:90-102`）

```python
feat_id = index // self.tot_len
s_begin = index % self.tot_len
s_end = s_begin + self.seq_len
r_begin = s_end - self.label_len
r_end = r_begin + self.label_len + self.pred_len

seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
seq_x_mark = self.data_stamp[s_begin:s_end]
seq_y_mark = self.data_stamp[r_begin:r_end]
```

**得到的形式：**
- `seq_x`: `[SeqLen, 1]`
- `seq_y`: `[LabelLen + PredLen, 1]`
- `seq_x_mark`: `[SeqLen, time_feat_dim]`
- `seq_y_mark`: `[LabelLen + PredLen, time_feat_dim]`

**说明：** `feat_id` 使得每个样本只包含一个变量的窗口序列；即使 `features='M'`，也会拆成多个单变量样本。

### 1.4 批处理后形状
在训练循环中（`run_main.py:180-204`），DataLoader 返回批次：
- `batch_x`: `[B, SeqLen, 1]`
- `batch_y`: `[B, LabelLen + PredLen, 1]`
- `batch_x_mark`: `[B, SeqLen, time_feat_dim]`
- `batch_y_mark`: `[B, LabelLen + PredLen, time_feat_dim]`

TimeLLM 实际只使用 `batch_x`（见下一节），`x_mark_enc` 在当前实现中未参与计算。

---

## 2. TimeLLM：从 batch 到 Patch Embedding 前的处理

### 2.1 归一化（RevIN 风格）
- `models/TimeLLM.py:259`
- `layers/StandardNorm.py:13-68`

```python
x_enc = self.normalize_layers(x_enc, 'norm')
```

**形状不变**：`[B, SeqLen, N]`。

### 2.2 Prompt 统计信息构建
- `models/TimeLLM.py:261-288`

处理步骤：
1. `x_enc` 先变形为 `[B*N, T, 1]`，逐变量计算统计量
2. 计算 min / max / median / trend / top-k lags
3. 拼接成文本 prompt

```python
B, T, N = x_enc.size()
x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
min_values = torch.min(x_enc, dim=1)[0]
max_values = torch.max(x_enc, dim=1)[0]
medians = torch.median(x_enc, dim=1).values
lags = self.calcute_lags(x_enc)
trends = x_enc.diff(dim=1).sum(dim=1)
```

**输出形式：**
- 文本 prompt 列表长度为 `B*N`

### 2.3 Prompt 嵌入（NLP 侧）
- `models/TimeLLM.py:291-293`

```python
prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))
```

**输出形状：**
- `prompt_embeddings`: `[B*N, prompt_len, llm_dim]`

---

## 3. Patch 分块与嵌入（核心）

### 3.1 PatchEmbedding 入口
- `models/TimeLLM.py:296-298`

```python
x_enc = x_enc.permute(0, 2, 1).contiguous()  # [B, N, T]
enc_out, n_vars = self.patch_embedding(x_enc.float())
```

### 3.2 Patching 过程
- `layers/Embed.py:177-184`

```python
n_vars = x.shape[1]
x = self.padding_patch_layer(x)
x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
```

**形状变化：**
1. 输入：`[B, N, T]`
2. Padding（复制末尾值，长度 + stride）
3. `unfold` → `[B, N, num_patches, patch_len]`
4. reshape → `[B*N, num_patches, patch_len]`

**num_patches 公式：**
- 由于 padding：`T' = T + stride`
- `num_patches = floor((T' - patch_len) / stride) + 1`
- 等价于 `floor((T - patch_len) / stride) + 2`

### 3.3 TokenEmbedding（1D Conv 映射到 d_model）
- `layers/Embed.py:30-43` + `layers/Embed.py:184`

```python
# TokenEmbedding.forward
x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
```

**核心理解：**
- 输入为 `[B*N, num_patches, patch_len]`
- `permute` 后变成 `[B*N, patch_len, num_patches]`
- Conv1d 以 `patch_len` 作为 in_channels，输出 `d_model`
- kernel_size=3 在 patch 序列维度上做局部融合

**输出：**
- `enc_out`: `[B*N, num_patches, d_model]`

---

## 4. 跨模态对齐前的“数据形式”

在进入 `ReprogrammingLayer` 之前，模型已得到以下两类输入：

### 4.1 时间序列侧（Patch Embedding）
- `enc_out`: `[B*N, num_patches, d_model]`
- 来源：`PatchEmbedding`（`layers/Embed.py:160-185`）

### 4.2 NLP 侧（可学习的 Source Embeddings）
- `models/TimeLLM.py:233-295`

```python
self.word_embeddings = self.llm_model.get_input_embeddings().weight
self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
```

**输出：**
- `source_embeddings`: `[num_tokens, llm_dim]`

这两个张量是跨模态对齐（Reprogramming）的输入：
- **Query**：`enc_out`（时序 patch）
- **Key/Value**：`source_embeddings`（LLM 词向量经过映射）

---

## 5. 具体数值例子（贴近默认配置）

假设：
- `B=32`, `T=96`, `patch_len=16`, `stride=8`, `d_model=16`
- 数据集按当前实现，每个样本只含 1 个变量（`N=1`）

```text
输入 batch_x: [32, 96, 1]
归一化后:     [32, 96, 1]
Patching 输入: [32, 1, 96]
Padding 后:    [32, 1, 104]
Unfold:        [32, 1, 12, 16]
Reshape:       [32, 12, 16]
TokenEmbedding:[32, 12, 16]
```

如果你将数据集改成“每个样本包含全部变量”，则会得到：
- `N=7` → patch embedding 输出为 `[B*N, num_patches, d_model] = [224, 12, 16]`

---

## 6. 关键结论（回答你的问题）

1. **分块 patch 怎么做**：先复制末尾补齐，再 `unfold` 滑窗切片，得到 `[B, N, num_patches, patch_len]`，再 reshape 为 `[B*N, num_patches, patch_len]`。
2. **数据集传入后的嵌入处理**：从 `batch_x` → 归一化 → patching → Conv1d 映射到 `d_model`。
3. **一次处理多少数据**：DataLoader 一次取 `batch_size` 个样本，每个样本是一个变量的窗口序列；patching 后按 `B*N` 展平。
4. **对齐前的形式**：
   - 时序侧：`enc_out [B*N, num_patches, d_model]`
   - 文本侧：`source_embeddings [num_tokens, llm_dim]`
   - prompt：`prompt_embeddings [B*N, prompt_len, llm_dim]`

---

## 7. 快速索引（源码位置）
- 数据集与采样：`data_provider/data_loader.py:46-105`
- DataLoader 构建：`data_provider/data_factory.py:16-63`
- 训练 batch 进入模型：`run_main.py:180-204`
- TimeLLM 处理与 prompt：`models/TimeLLM.py:257-299`
- PatchEmbedding：`layers/Embed.py:160-185`
- TokenEmbedding：`layers/Embed.py:30-43`
- Normalize（RevIN）：`layers/StandardNorm.py:13-68`

