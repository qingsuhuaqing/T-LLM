# Time-LLM 训练命令详解 (mingling_2.md)

> **适用配置:** Qwen 2.5 3B + 4-bit 量化 | 6GB 显存 | ETTm1 数据集

---

## 一、完整训练命令

```powershell
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

---

## 二、参数逐条解析

### 2.1 基础配置参数

| 参数 | 值 | 代码位置 | 作用 |
|------|------|----------|------|
| `--task_name` | `long_term_forecast` | `run_main.py` L30 | 任务类型，指定为**长期预测**任务 |
| `--is_training` | `1` | `run_main.py` L32 | 训练模式开关，1=训练，0=仅推理 |
| `--model` | `TimeLLM` | `run_main.py` L35 | 使用的模型，触发 `models/TimeLLM.py` 加载 |
| `--model_id` | `ETTm1_512_96` | `run_main.py` L33 | 实验标识符，用于 checkpoint 命名 |
| `--model_comment` | `Qwen3B` | `run_main.py` L34 | 实验备注，附加在 checkpoint 路径后 |

### 2.2 数据集配置

| 参数 | 值 | 代码位置 | 作用 |
|------|------|----------|------|
| `--root_path` | `./dataset/ETT-small/` | `run_main.py` L41 | 数据集根目录 |
| `--data_path` | `ETTm1.csv` | `run_main.py` L42 | 数据文件名 |
| `--data` | `ETTm1` | `run_main.py` L40 | 数据集名称，触发 `Dataset_ETT_minute` 加载器 |
| `--features` | `M` | `run_main.py` L43 | **M**=多变量→多变量，S=单变量，MS=多变量→单变量 |

### 2.3 序列长度配置

| 参数 | 值 | 代码位置 | 作用 |
|------|------|----------|------|
| `--seq_len` | `512` | `run_main.py` L56 | **输入序列长度**，历史观测窗口 |
| `--label_len` | `48` | `run_main.py` L57 | Decoder 起始重叠长度，用于自回归解码 |
| `--pred_len` | `96` | `run_main.py` L58 | **预测长度**，输出未来 96 个时间步 |

**数据流说明:**
```
输入: 过去 512 步 → 模型处理 → 输出: 未来 96 步
```

### 2.4 模型架构参数

| 参数 | 值 | 代码位置 | 作用 |
|------|------|----------|------|
| `--e_layers` | `2` | `run_main.py` L67 | 编码器层数（Time-LLM 未使用，保留兼容性） |
| `--d_layers` | `1` | `run_main.py` L68 | 解码器层数（Time-LLM 未使用，保留兼容性） |
| `--factor` | `3` | `run_main.py` L71 | 注意力稀疏因子（Time-LLM 未使用） |
| `--d_model` | `32` | `run_main.py` L65 | **Patch 嵌入维度**，影响 `PatchEmbedding` 输出 |
| `--d_ff` | `32` | `run_main.py` L69 | **FFN 隐藏维度**，影响 `FlattenHead` 输入 |
| `--dropout` | `0.1` | `run_main.py` L72 | Dropout 比例，防止过拟合 |

### 2.5 变量数量配置

| 参数 | 值 | 代码位置 | 作用 |
|------|------|----------|------|
| `--enc_in` | `7` | `run_main.py` L62 | 编码器输入变量数 = ETTm1 的 7 列特征 |
| `--dec_in` | `7` | `run_main.py` L63 | 解码器输入变量数 |
| `--c_out` | `7` | `run_main.py` L64 | 输出变量数 |

**ETTm1 的 7 个变量:**
- HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (油温)

### 2.6 训练配置

| 参数 | 值 | 代码位置 | 作用 |
|------|------|----------|------|
| `--batch_size` | `8` | `run_main.py` L92 | **批大小**，⚠️ 6GB 显存可能需降至 4 |

### 2.7 ⭐ LLM 配置 (核心参数)

| 参数 | 值 | 代码位置 | 作用 |
|------|------|----------|------|
| `--llm_model` | `QWEN` | `run_main.py` L80 | LLM 类型标识（实际未被使用，见说明） |
| `--llm_dim` | `2048` | `run_main.py` L81 | **Qwen 2.5 3B 隐藏层维度** |
| `--llm_model_path` | 本地路径 | `run_main.py` L83 | **⭐ 核心参数**，指向本地模型文件夹 |
| `--load_in_4bit` | 启用 | `run_main.py` L84 | **⭐ 启用 4-bit 量化**，显存降至 ~1.5GB |

---

## 三、参数如何被使用

### 3.1 数据加载流程

```python
# run_main.py L129-131
train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')

# data_provider 内部使用:
# args.data → 选择 Dataset_ETT_minute
# args.root_path + args.data_path → 加载 CSV 文件
# args.seq_len, args.pred_len → 切分滑动窗口
```

### 3.2 模型创建流程

```python
# run_main.py L137-138
model = TimeLLM.Model(args).float()

# TimeLLM.Model.__init__ 内部使用:
# args.llm_model_path → 触发通用模型加载 (L43)
# args.load_in_4bit → 启用 BitsAndBytesConfig (L50-57)
# args.llm_dim → 设置嵌入空间维度 (L39)
# args.d_model, args.d_ff → 配置 Patch/FFN 维度
```

### 3.3 4-bit 量化配置

```python
# models/TimeLLM.py L50-57
if configs.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
```

---

## 四、Checkpoint 输出

**保存路径格式:**
```
./checkpoints/{task_name}_{model_id}_{model}_{data}_ft{features}_..._{model_comment}/checkpoint
```

**本命令的实际路径:**
```
./checkpoints/long_term_forecast_ETTm1_512_96_TimeLLM_ETTm1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-Qwen3B/checkpoint
```

---

## 五、预期输出

```
Loading generic model from: e:\timellm\Time-LLM\base_models\Qwen2.5-3B
Enabling 4-bit quantization...

iters: 100, epoch: 1 | loss: 0.4523
speed: 0.12s/iter; left time: 1234.5s
Epoch: 1 cost time: 123.45
Epoch: 1 | Train Loss: 0.4523 Vali Loss: 0.3912 Test Loss: 0.4012 MAE Loss: 0.4567

Validation loss decreased (inf --> 0.391200). Saving model ...
```

---

## 六、注意事项

> [!WARNING]
> **显存警告:** `seq_len=512` + `batch_size=8` 可能导致 OOM
> 建议首次运行使用 `--batch_size 4`

> [!TIP]
> `--llm_model QWEN` 参数在当前代码中**未被使用**（因为优先检查 `llm_model_path`），可以删除但不影响运行
