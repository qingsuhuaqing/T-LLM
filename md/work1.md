# Time-LLM 项目执行指南 (Qwen 2.5 3B 4-bit 量化版)

## 一、项目简介

Time-LLM 是 ICLR'24 发表的一个创新项目，其核心功能是：通过 **"重编程" (Reprogramming)** 技术将预训练的大语言模型 (LLM) 适配为时间序列预测模型，无需从头训练即可利用 LLM 强大的模式识别能力进行时序预测。

**本版本已适配 6GB 显存**，使用 Qwen 2.5 3B 配合 4-bit 量化运行。

---

## 二、已完成的配置

### 2.1 数据集 ✅
已下载并放置在 `./dataset/ETT-small/` 目录下：
- `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`

### 2.2 基座模型 ✅
已下载 **Qwen 2.5 3B Instruct** 到本地：
- **位置**: `./base_models/Qwen2.5-3B/`
- **文件清单**:
  - `config.json` ✅
  - `model-00001-of-00002.safetensors` ✅
  - `model-00002-of-00002.safetensors` ✅
  - `model.safetensors.index.json` ✅
  - `tokenizer.json` ✅
  - `tokenizer_config.json` ✅
  - `vocab.json` ✅
  - `merges.txt` ✅
  - `generation_config.json` ✅

---

## 三、代码修改记录

### 3.1 `run_main.py` 修改
**新增参数（第 82-84 行）**：
```python
# ========== 新增参数：支持本地模型路径和4-bit量化 ==========
parser.add_argument('--llm_model_path', type=str, default='', help='LLM model path (local or HuggingFace ID)')
parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit quantization to save VRAM')
# =========================================================
```

### 3.2 `models/TimeLLM.py` 修改
**新增通用模型加载逻辑（第 43-96 行）**：
- 使用 `AutoModel`, `AutoTokenizer`, `AutoConfig` 支持任意 HuggingFace 模型
- 使用 `BitsAndBytesConfig` 实现 4-bit NF4 量化
- 支持本地路径和在线 HuggingFace ID 两种加载方式

---

## 四、运行命令

### 4.1 前置依赖安装
```bash
pip install bitsandbytes accelerate
```

### 4.2 训练命令 (Qwen 2.5 3B + 4-bit)
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

**关键参数说明**：
| 参数 | 值 | 说明 |
|------|------|------|
| `--llm_model_path` | 本地路径 | 指向 Qwen 2.5 3B 模型文件夹 |
| `--load_in_4bit` | 开启 | 启用 4-bit 量化，模型仅占 ~1.5GB 显存 |
| `--llm_dim` | 2048 | Qwen 2.5 3B 的隐藏层维度 |
| `--batch_size` | 8 | 3B 模型显存宽裕，可调大 |

---

## 五、显存估算

| 组件 | 显存占用 |
|------|----------|
| Qwen 2.5 3B (4-bit) | ~1.5 GB |
| Time-LLM 可训练参数 | ~0.5 GB |
| 中间变量 & 梯度 | ~2.0 GB |
| 系统占用 | ~1.0 GB |
| **总计** | **~5.0 GB** ✅ |

> 6GB 显存完全够用！如遇 OOM，可将 `batch_size` 降至 4。

---

## 六、故障排除

### 6.1 CUDA out of memory
- 降低 `--batch_size` 为 4 或 2
- 降低 `--seq_len` 为 256

### 6.2 Model files not found
- 确认 `base_models/Qwen2.5-3B/` 下有完整的 `.safetensors` 文件
- 确认 `config.json` 和 `tokenizer.json` 存在

### 6.3 bitsandbytes 安装失败 (Windows)
```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

---

## 七、项目架构

```
Time-LLM/
├── base_models/
│   ├── gpt2/                    # 旧版 GPT-2 模型 (可选)
│   └── Qwen2.5-3B/              # ★ 新增：Qwen 2.5 3B 模型
├── models/
│   └── TimeLLM.py               # ★ 已修改：支持 AutoModel + 4-bit
├── dataset/
│   ├── ETT-small/               # 训练数据
│   └── prompt_bank/             # 领域描述
├── run_main.py                  # ★ 已修改：新增 llm_model_path, load_in_4bit
├── CLAUDE.md                    # 项目文档
└── work1.md                     # 本文档
```