# Colab + Drive 使用指南（Llama 3 8B）

本文档整理了：已实现的功能、改动点、脚本、命令与使用流程，帮助你在 Colab 上稳定训练、断点续训与日志保存。

---

## 1. 目标与结论

你现在可以在 Colab 中实现：
- ✅ **固定步保存**（每 N step 保存一次）
- ✅ **断点续训**（从任意 checkpoint 恢复）
- ✅ **最优模型保存（EarlyStopping）**
- ✅ **日志写入 Google Drive**
- ✅ **推理可加载权重**

所有输出（checkpoint/log）都落在 Google Drive，断电/断连不丢。

---

## 2. 项目结构要点（与你需求相关）

- 训练入口：`run_main.py`
- 早停保存与最优模型（完整 checkpoint）：`utils/tools.py`
- 步级保存 + 断点续训：`run_main.py`
- Colab 脚本：`scripts/TimeLLM_ETTh1_llama3_colab_drive.sh`
- Checkpoint 输出：`/content/drive/MyDrive/T-L/checkpoints/...`
- 日志输出：`/content/drive/MyDrive/T-L/logs/...`

---

## 3. 已做的代码改动（实现断点续训与完整保存）

### 3.1 `run_main.py`
- 新增参数：
  - `--save_steps`：每 N step 保存一次
  - `--save_total_limit`：只保留最近 N 个 step checkpoint
  - `--resume_from_checkpoint`：从指定路径恢复训练
- 新增逻辑：
- 训练开始时加载完整 checkpoint（模型 + optimizer + scheduler + epoch + step）
- 训练中每 `save_steps` 保存一次完整 checkpoint
- 超过 `save_total_limit` 时自动删除最老的 step checkpoint
- 支持在 epoch 内恢复：会自动跳过已训练的 batch
- 兼容旧 checkpoint（仅 `model.state_dict()`）加载

### 3.2 `utils/tools.py`
- EarlyStopping 保存改为 **完整 dict**：
  - `model + optimizer + scheduler + epoch + global_step`
- 文件名保持 `checkpoint` 不变

### 3.3 兼容改动
- `run_m4.py`：读取 `checkpoint` 时兼容 dict
- `run_pretrain.py`：EarlyStopping 改为保存完整 dict

---

## 4. 新增脚本

### 4.1 Colab + Drive 训练脚本
路径：`scripts/TimeLLM_ETTh1_llama3_colab_drive.sh`

**手动修改区（都在脚本最上方）**：
- `PROJECT_DIR="/content/drive/MyDrive/T-L"`
- `LLM_PATH="${PROJECT_DIR}/base_models/LLM-Research/Meta-Llama-3-8B-Instruct"`
- `CHECKPOINTS="${PROJECT_DIR}/checkpoints"`
- `LOG_DIR="${PROJECT_DIR}/logs"`
- `SAVE_STEPS=1000`
- `RESUME_FROM=""`（第一次留空）
- `BATCH / LLM_LAYERS / SEQ_LEN / D_FF`

---

## 5. Colab 使用流程（推荐）

### 5.1 挂载 Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5.2 检查路径（可选）
```bash
!ls /content/drive/MyDrive/T-L
!ls /content/drive/MyDrive/T-L/base_models/LLM-Research/Meta-Llama-3-8B-Instruct
```

### 5.3 启动训练（Drive 上直接运行）
```bash
!bash /content/drive/MyDrive/T-L/scripts/TimeLLM_ETTh1_llama3_colab_drive.sh
```

---

## 6. 断点续训

将脚本中的 `RESUME_FROM` 指向某个保存点：

示例（目录或具体文件都可）：
```
RESUME_FROM="/content/drive/MyDrive/T-L/checkpoints/<setting>-<comment>/checkpoint_step_1000"
# 或
RESUME_FROM="/content/drive/MyDrive/T-L/checkpoints/<setting>-<comment>/checkpoint_step_1000/checkpoint.pt"
```

然后重新运行脚本即可继续训练。

---

## 7. 推理加载权重

早停保存的 best checkpoint 现在也是完整 dict，可用于推理：
```python
ckpt = torch.load(".../checkpoint", map_location="cpu")
model.load_state_dict(ckpt["model"])
```

---

## 8. OOM（显存/内存不足）处理策略

依次降低以下参数（优先级从高到低）：
1. `BATCH`: 6 -> 4 -> 2
2. `LLM_LAYERS`: 32 -> 24 -> 16
3. `SEQ_LEN`: 512 -> 384 -> 256
4. `D_FF`: 64 -> 32

---

## 9. 日志输出位置

训练日志会写入：
```
/content/drive/MyDrive/T-L/logs/train_llama3_8b_YYYYMMDD_HHMMSS.log
```

---

## 10. 关键文件清单

```
run_main.py                         # 训练入口 + save_steps/resume
utils/tools.py                      # EarlyStopping 保存完整 dict
scripts/TimeLLM_ETTh1_llama3_colab_drive.sh  # Colab + Drive 脚本
```

---

如果你需要进一步：
- 自动打印每个 epoch 的 step 数
- 增加“按分钟保存”
- 生成推理专用脚本  
告诉我，我可以继续补充。
