# Time-LLM 运行复现指南

> 针对 TimeLLM_ETTh1_2.sh 脚本的详细运行说明

---

## 目录

1. [如何运行脚本](#1-如何运行脚本)
2. [参数顺序要求](#2-参数顺序要求)
3. [运行过程中的终端输出](#3-运行过程中的终端输出)
4. [训练与推理的区别](#4-训练与推理的区别)
5. [训练效果分析](#5-训练效果分析)
6. [硬件资源占用分析](#6-硬件资源占用分析)

---

## 1. 如何运行脚本

### 1.1 运行命令

在您的目录下执行以下命令：

```bash
# 方法一：直接使用 bash 运行
bash ./scripts/TimeLLM_ETTh1_2.sh

# 方法二：赋予执行权限后运行
chmod +x ./scripts/TimeLLM_ETTh1_2.sh
./scripts/TimeLLM_ETTh1_2.sh

# 方法三：使用 source 运行（会在当前 shell 中执行）
source ./scripts/TimeLLM_ETTh1_2.sh
```

### 1.2 推荐方式

```bash
(time-llm) suhuaqing@DESKTOP-4BSH3A5:/mnt/e/timellm-chuangxin/Time-LLM$ bash ./scripts/TimeLLM_ETTh1_2.sh
```

### 1.3 注意事项

**重要**：脚本中第 10 行有路径设置：
```bash
cd /mnt/e/timellm/Time-LLM
```

您当前的目录是 `/mnt/e/timellm-chuangxin/Time-LLM`，需要修改脚本中的路径，或者直接在当前目录运行 Python 命令：

```bash
# 直接运行 Python 命令（推荐）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model_comment Qwen3B_fast \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 4 \
  --d_model 32 \
  --n_heads 8 \
  --d_ff 32 \
  --llm_dim 2048 \
  --llm_layers 6 \
  --num_workers 2 \
  --prompt_domain 1 \
  --train_epochs 3 \
  --itr 1 \
  --dropout 0.1 \
  --llm_model QWEN \
  --llm_model_path /mnt/e/timellm-chuangxin/Time-LLM/base_models/Qwen2.5-3B \
  --load_in_4bit
```

---

## 2. 参数顺序要求

### 2.1 答案：参数顺序没有要求

```bash
--label_len 48 \
--pred_len 96 \
--e_layers 2 \
```

这些参数的顺序**可以任意调换**，不影响运行结果。

### 2.2 原因

Python 的 `argparse` 模块使用**命名参数**方式解析命令行参数：

```python
# run_main.py 中的参数定义
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--e_layers', type=int, default=2)
```

`argparse` 通过参数名（如 `--label_len`）来识别参数值，而非位置。

### 2.3 等价写法

以下三种写法完全等价：

```bash
# 写法 1（原脚本）
--label_len 48 --pred_len 96 --e_layers 2

# 写法 2（调换顺序）
--e_layers 2 --label_len 48 --pred_len 96

# 写法 3（完全打乱）
--pred_len 96 --e_layers 2 --label_len 48
```

### 2.4 唯一约束

**参数名和参数值必须紧邻**，即 `--label_len` 后面必须紧跟 `48`：

```bash
# 正确
--label_len 48

# 错误
--label_len --pred_len 48 96
```

---

## 3. 运行过程中的终端输出

### 3.1 完整输出流程预测

运行脚本后，您将看到以下阶段的输出：

#### 阶段 1：模型加载（约 10-30 秒）

```
Loading checkpoint shards: 100%|████████████████████████| 2/2 [00:15<00:00,  7.52s/it]
```

这是加载 Qwen 2.5 3B 模型的进度条，4-bit 量化版本约 1.5GB，需要加载 2 个分片。

#### 阶段 2：数据集信息

```
train 8545
val 2881
test 2881
```

显示训练集、验证集、测试集的样本数量。

#### 阶段 3：训练迭代（核心输出）

```
  0%|          | 0/2136 [00:00<?, ?it/s]
iters: 100, epoch: 1 | loss: 0.4523156
        speed: 1.1234s/iter; left time: 7234.5678s
iters: 200, epoch: 1 | loss: 0.3892341
        speed: 1.0987s/iter; left time: 6543.2345s
iters: 300, epoch: 1 | loss: 0.3567823
        speed: 1.1012s/iter; left time: 5876.4321s
...
```

**输出内容解释：**

| 输出项 | 含义 | 示例值 |
|--------|------|--------|
| `iters` | 当前迭代次数 | 100, 200, 300... |
| `epoch` | 当前训练轮数 | 1, 2, 3 |
| `loss` | **训练损失（MSE）** | 0.4523156 |
| `speed` | 每次迭代耗时 | 1.1234s/iter |
| `left time` | 预估剩余时间 | 7234.5678s (~2 小时) |

#### 阶段 4：每轮结束汇总

```
Epoch: 1 cost time: 3542.34
Epoch: 1 | Train Loss: 0.3245 Vali Loss: 0.2987 Test Loss: 0.3012 MAE Loss: 0.4123
```

**汇总指标解释：**

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| `Train Loss` | 训练集 MSE 损失 | 逐轮下降 |
| `Vali Loss` | 验证集 MSE 损失 | 逐轮下降（用于 Early Stopping） |
| `Test Loss` | 测试集 MSE 损失 | 最终评估指标 |
| `MAE Loss` | 测试集 MAE 损失 | 最终评估指标 |

#### 阶段 5：Early Stopping 提示

```
EarlyStopping counter: 0 out of 10
Validation loss decreased (inf --> 0.298700). Saving model ...
```

或

```
EarlyStopping counter: 1 out of 10
EarlyStopping counter: 2 out of 10
...
Early stopping
```

#### 阶段 6：训练完成

```
Epoch: 3 cost time: 3456.78
Epoch: 3 | Train Loss: 0.2156 Vali Loss: 0.2345 Test Loss: 0.2456 MAE Loss: 0.3567

Checkpoints saved at: ./checkpoints
```

### 3.2 是否有损失函数输出？

**是的，有损失函数输出。** 主要包括：

1. **实时训练损失**：每 100 次迭代输出一次 `loss: x.xxxxxxx`
2. **每轮汇总损失**：
   - `Train Loss`：训练集 MSE
   - `Vali Loss`：验证集 MSE
   - `Test Loss`：测试集 MSE
   - `MAE Loss`：测试集 MAE

### 3.3 其他可能的输出

1. **警告信息**（可忽略）：
```
UserWarning: torch.nn.utils.weight_norm is deprecated...
Some weights of the model checkpoint were not used...
```

2. **GPU 相关信息**：
```
Using device: cuda:0
GPU Memory: 5.8GB / 6.0GB
```

3. **tqdm 进度条**：
```
100%|██████████| 2136/2136 [39:15<00:00,  1.10s/it]
```

---

## 4. 训练与推理的区别

### 4.1 当前项目状态

**是的，项目目前没有独立的推理脚本。**

当前的 `run_main.py` 主要用于训练，虽然在训练过程中会进行验证和测试，但没有单独的推理入口。

### 4.2 训练模式 vs 推理模式

| 模式 | 参数设置 | 数据使用 | 输出 |
|------|----------|----------|------|
| **训练** | `--is_training 1` | train + val + test | 更新模型参数，保存 checkpoint |
| **推理** | `--is_training 0` | 仅 test | 加载 checkpoint，输出预测结果 |

### 4.3 如何进行推理

虽然没有独立脚本，但可以通过修改参数实现推理：

```bash
python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \          # 关键：设为 0 表示测试模式
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model_comment Qwen3B_fast \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  ... # 其他参数保持与训练时一致
```

### 4.4 Checkpoint 的作用

**训练后保存的 checkpoint 包含：**
- PatchEmbedding 参数
- Mapping Layer 参数
- Reprogramming Layer 参数
- FlattenHead 参数

**推理时：**
1. 重新初始化模型结构
2. 加载 checkpoint 中的参数
3. 重新加载 LLM（因为 LLM 参数冻结，不在 checkpoint 中）
4. 对测试数据进行预测

### 4.5 推理脚本建议

如果需要独立的推理脚本，可以创建 `inference.py`：

```python
# 简化的推理逻辑
import torch
from models import TimeLLM
from data_provider.data_factory import data_provider

# 1. 加载模型
model = TimeLLM.Model(args).float()
model.load_state_dict(torch.load('checkpoints/.../checkpoint'))
model.eval()

# 2. 加载测试数据
test_data, test_loader = data_provider(args, 'test')

# 3. 推理
with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # 保存或分析 outputs
```

---

## 5. 训练效果分析

### 5.1 当前配置评估

| 参数 | 当前值 | 论文推荐值 | 影响 |
|------|--------|------------|------|
| `train_epochs` | **3** | 100 | ⚠️ 严重不足 |
| `seq_len` | 512 | 512 | ✅ 符合 |
| `batch_size` | 4 | 24 | ⚠️ 较小但可接受 |
| `llm_layers` | 6 | 32 | ⚠️ 减少但可接受 |

### 5.2 3 个 epoch 的训练效果预测

**预测结果：训练效果会较差，但不是完全无效。**

| 指标 | 3 epochs 预估 | 论文结果 (100 epochs) | 差距 |
|------|---------------|----------------------|------|
| Test MSE | 0.35 - 0.45 | 0.375 | +0% ~ +20% |
| Test MAE | 0.45 - 0.55 | 0.395 | +14% ~ +39% |

**原因分析：**

1. **损失下降曲线**：
   - Epoch 1：损失快速下降（学习率高，梯度大）
   - Epoch 2-3：损失继续下降但速度变慢
   - Epoch 4-100：损失缓慢收敛到最优

2. **3 epochs 的问题**：
   - 模型可能刚进入收敛阶段
   - 可训练参数（~160M）还未充分优化
   - Early Stopping 可能还未触发

### 5.3 是否可用于推理？

**可以用于推理，但准确率不会很高。**

- **损失函数不会"很大"**：3 epochs 后损失已经明显下降
- **准确率会"偏低"**：比充分训练的模型差 10%-30%
- **实际使用**：可用于快速验证和调试，不建议用于生产环境

### 5.4 更换显卡（服务器）的效果

**是的，更换更好的显卡可以显著提升训练效果。**

| 显卡配置 | 可用参数 | 预期效果 |
|----------|----------|----------|
| **6GB (当前)** | batch_size=4, epochs=3 | Test MSE ~0.40 |
| **24GB (RTX 3090)** | batch_size=24, epochs=50 | Test MSE ~0.38 |
| **80GB (A100)** | batch_size=64, epochs=100 | Test MSE ~0.375 (论文水平) |

**服务器训练建议：**

```bash
# 服务器配置（假设 24GB 显存）
python run_main.py \
  --batch_size 16 \
  --train_epochs 50 \
  --llm_layers 12 \
  --seq_len 512 \
  --d_model 32 \
  --d_ff 128 \
  ... # 其他参数
```

---

## 6. 硬件资源占用分析

### 6.1 训练时间预估

根据脚本注释和实际测试：

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 模型加载 | 15-30 秒 | 加载 Qwen 2.5 3B (4-bit) |
| 数据加载 | 5-10 秒 | 读取 ETTh1.csv |
| **每轮训练** | **55-60 分钟** | ~2136 迭代 × ~1.1秒/迭代 |
| 验证 + 测试 | 5-10 分钟 | 每轮结束后 |
| **总时间 (3 epochs)** | **约 2.5-3 小时** | - |

### 6.2 显存占用

**主要占用的是显存（GPU Memory），而非内存（RAM）。**

| 组件 | 显存占用 | 说明 |
|------|----------|------|
| Qwen 2.5 3B (4-bit) | ~1.5 GB | 量化后的 LLM |
| 可训练参数 | ~0.5 GB | PatchEmbedding + Mapping + Reprogramming + FlattenHead |
| 中间变量 (batch_size=4) | ~2.5 GB | 前向传播的 tensor |
| CUDA 上下文 | ~0.5 GB | PyTorch/CUDA 开销 |
| **总计** | **~5.0-5.5 GB** | 接近 6GB 上限 |

### 6.3 内存（RAM）占用

**会占用部分内存，但不多。**

| 组件 | 内存占用 | 说明 |
|------|----------|------|
| Python 进程 | ~2-3 GB | 基础开销 |
| 数据加载 (num_workers=2) | ~1-2 GB | DataLoader 子进程 |
| 数据集缓存 | ~0.5 GB | ETTh1.csv 数据 |
| **总计** | **~4-6 GB** | 32GB 内存绰绰有余 |

### 6.4 对本机的影响

#### 6.4.1 显卡占用

- **显存占用率**：~85-95%（5.0-5.7 GB / 6 GB）
- **GPU 利用率**：~90-100%（持续高负载）
- **温度**：可能达到 80-90°C（建议监控）

```bash
# 监控 GPU 状态
watch -n 1 nvidia-smi
```

#### 6.4.2 CPU 占用

- **CPU 利用率**：~20-40%（主要是数据加载）
- **核心使用**：主要使用 2 个核心（num_workers=2）

#### 6.4.3 内存占用

- **内存占用**：~4-6 GB / 32 GB
- **剩余可用**：~26-28 GB

### 6.5 本机能否处理其他内容？

**可以，但有限制。**

| 任务类型 | 是否可行 | 说明 |
|----------|----------|------|
| 浏览网页 | ✅ 可以 | 内存充足，不影响 |
| 文档编辑 | ✅ 可以 | 内存充足，不影响 |
| 代码编辑 | ✅ 可以 | 内存充足，不影响 |
| **其他 GPU 任务** | ❌ 不可以 | 显存已满 |
| 大型程序 (如 IDE) | ⚠️ 可能卡顿 | 建议先关闭 |
| 视频播放 | ⚠️ 可能卡顿 | 占用 GPU 解码 |

### 6.6 系统稳定性

**只要不超出内存，系统可以正常运行。**

关键约束是**显存**而非**内存**：

| 资源 | 超出后果 |
|------|----------|
| **显存超出** | CUDA OOM 错误，训练崩溃 |
| **内存超出** | 系统 swap，严重卡顿 |

您的配置（6GB 显存 + 32GB 内存）：
- 显存：接近上限，需要保持当前参数不变
- 内存：非常充足，可以同时运行其他程序

### 6.7 建议操作

训练期间的建议：

```bash
# 1. 启动训练前关闭不必要的 GPU 程序
nvidia-smi  # 检查是否有其他 GPU 进程

# 2. 在另一个终端监控资源
watch -n 1 nvidia-smi  # 监控 GPU
htop                   # 监控 CPU 和内存

# 3. 训练命令（在 tmux 或 screen 中运行，防止终端断开）
tmux new -s training
bash ./scripts/TimeLLM_ETTh1_2.sh

# 断开 tmux（训练继续）
Ctrl+B, D

# 重新连接
tmux attach -t training
```

---

## 总结

### 快速回答汇总

| 问题 | 回答 |
|------|------|
| 运行命令 | `bash ./scripts/TimeLLM_ETTh1_2.sh` |
| 参数顺序 | 无要求，可任意调换 |
| 终端输出 | 有损失函数、迭代进度、epoch 汇总 |
| 推理脚本 | 目前没有独立脚本，需设置 `--is_training 0` |
| 3 epochs 效果 | 可用但不理想，MSE 约 0.35-0.45 |
| 服务器效果 | 更好，epochs 更多，batch_size 更大 |
| 训练时间 | 约 2.5-3 小时 |
| 显存占用 | ~5.0-5.5 GB（主要瓶颈） |
| 内存占用 | ~4-6 GB（充足） |
| 本机可用性 | 可以处理轻量任务，避免 GPU 任务 |

---

**文档生成时间**: 2025-01-09
**适用脚本**: TimeLLM_ETTh1_2.sh
**硬件环境**: 6GB 显存 + 32GB 内存
