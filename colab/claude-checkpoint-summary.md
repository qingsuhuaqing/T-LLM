# Checkpoint 机制完整总结

> **目的**: 回答关于断点保存、早停保存的具体内容，以及在算力有限条件下的训练完整性
● 总结回答

  1. checkpoint_step_N.pt 保存了什么？

  | 字段           | 内容                 | 功能             |
  |--------------|--------------------|----------------|
  | model        | 所有可训练参数 (~56M)     | 恢复模型，继续训练或推理   |
  | optimizer    | Adam 的动量、历史梯度      | 恢复优化器状态，保持训练连续 |
  | scheduler    | 学习率曲线位置            | 恢复学习率，保持曲线连续   |
  | epoch        | 当前 epoch 数         | 确定从哪继续         |
  | global_step  | 总 step 数           | 跳过已训练的 batch   |
  | best_score   | EarlyStopping 最佳分数 | 防止覆盖更好的模型      |
  | val_loss_min | 最小验证损失             | 同上             |

  2. checkpoint（早停保存）保存了什么？

  内容完全相同，区别是：
  - 触发时机: 每个 epoch 结束后，仅当 vali_loss 创新低时
  - 用途: 保存验证集上表现最好的模型，用于最终推理

  3. 在算力有限条件下，是否完整？

  ✅ 是的，完整了！

  修复后的检查表：

  | 检查项                | 状态      |
  |--------------------|---------|
  | 模型参数连续性            | ✅       |
  | 优化器状态连续性           | ✅       |
  | 学习率曲线连续性           | ✅       |
  | 训练进度连续性（不重复 batch） | ✅       |
  | EarlyStopping 状态恢复 | ✅ (刚修复) |
  | 最佳模型独立保护           | ✅       |

  你可以放心地分段训练，最终效果与一次性训练完成是一致的。
---

## 1. 两种 Checkpoint 的详细内容对比

### 1.1 checkpoint_step_N.pt（断点保存）

**保存位置**: `checkpoints/<setting>-<comment>/checkpoint_step_N/checkpoint.pt`

**触发时机**: 每 `save_steps` 个训练 step

**保存内容** (`run_main.py:303-311`):

```python
ckpt_payload = {
    'model': model.state_dict(),           # 模型参数
    'optimizer': model_optim.state_dict(), # 优化器状态
    'scheduler': scheduler.state_dict(),   # 学习率调度器状态
    'epoch': epoch,                        # 当前 epoch 数
    'global_step': global_step,            # 全局 step 数
    'best_score': early_stopping.best_score,     # EarlyStopping 最佳分数
    'val_loss_min': early_stopping.val_loss_min, # EarlyStopping 最小验证损失
}
```

| 字段 | 类型 | 功能 |
|------|------|------|
| `model` | dict | **模型所有可训练参数**（PatchEmbedding, Mapping Layer, Reprogramming Layer, FlattenHead） |
| `optimizer` | dict | Adam 优化器状态（每个参数的动量、历史梯度等） |
| `scheduler` | dict | OneCycleLR 的内部状态（当前学习率位置、step 计数） |
| `epoch` | int | 当前在第几个 epoch |
| `global_step` | int | 总共训练了多少个 step |
| `best_score` | float/None | EarlyStopping 记录的最佳验证分数 |
| `val_loss_min` | float | EarlyStopping 记录的最小验证损失 |

---

### 1.2 checkpoint（早停保存/最佳模型）

**保存位置**: `checkpoints/<setting>-<comment>/checkpoint`（无扩展名）

**触发时机**: 每个 epoch 结束后，**仅当验证损失创新低时**

**保存内容** (`utils/tools.py:83-91`):

```python
ckpt = {
    'model': model.state_dict(),           # 模型参数
    'optimizer': optimizer.state_dict(),   # 优化器状态
    'scheduler': scheduler.state_dict(),   # 学习率调度器状态
    'epoch': epoch,                        # 当前 epoch 数
    'global_step': global_step,            # 全局 step 数
    'best_score': self.best_score,         # EarlyStopping 最佳分数
    'val_loss_min': val_loss,              # 当前验证损失（即新的最小值）
}
```

**内容与 checkpoint_step_N 完全相同！** 区别只是触发时机和用途。

---

## 2. 两种 Checkpoint 的区别总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Checkpoint 文件结构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  checkpoints/                                                       │
│  └── long_term_forecast_ETTh1_...-Llama3_8B_4bit/                  │
│      │                                                              │
│      ├── checkpoint                    ← 早停保存（最佳模型）        │
│      │   └── 内容: model + optimizer + scheduler + epoch +          │
│      │            global_step + best_score + val_loss_min           │
│      │   └── 用途: 最终推理、保存验证集上表现最好的模型              │
│      │                                                              │
│      ├── checkpoint_step_1000/         ← 断点保存                   │
│      │   └── checkpoint.pt                                          │
│      │       └── 内容: 同上                                         │
│      │       └── 用途: 断点续训、防止意外中断丢失进度                │
│      │                                                              │
│      └── checkpoint_step_2000/         ← 断点保存                   │
│          └── checkpoint.pt                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| 特性 | checkpoint_step_N (断点) | checkpoint (早停/最佳) |
|------|--------------------------|------------------------|
| **触发条件** | 每 N 个 step | 验证损失创新低 |
| **保存内容** | 完整状态 | 完整状态 |
| **主要用途** | 断点续训 | 最终推理 |
| **会被覆盖** | 被 save_total_limit 管理 | 只被更好的模型覆盖 |
| **保存的模型** | 任意时刻的模型 | 验证集上最好的模型 |

---

## 3. 各字段的具体功能

### 3.1 model（模型参数）

```python
model.state_dict() 包含：
├── patch_embedding.value_embedding.tokenConv.weight  # Patch 嵌入层
├── patch_embedding.value_embedding.tokenConv.bias
├── patch_embedding.position_embedding                 # 位置编码
├── mapping_layer.weight                               # 词表映射层 (~50M 参数)
├── mapping_layer.bias
├── reprogramming_layer.query_projection.weight        # 重编程层
├── reprogramming_layer.key_projection.weight
├── reprogramming_layer.value_projection.weight
├── reprogramming_layer.out_projection.weight
├── output_projection.flatten.weight                   # 输出层
└── output_projection.flatten.bias
```

**恢复后**: 模型可以从断点处继续训练，或直接用于推理

### 3.2 optimizer（优化器状态）

```python
optimizer.state_dict() 包含：
├── state: {
│   0: {exp_avg, exp_avg_sq, step},  # 第 0 个参数的 Adam 状态
│   1: {exp_avg, exp_avg_sq, step},  # 第 1 个参数的 Adam 状态
│   ...
│ }
└── param_groups: [{lr, betas, eps, weight_decay, ...}]
```

**功能**: Adam 优化器需要记录每个参数的一阶/二阶动量估计

**恢复后**: 优化器从断点状态继续，不会"忘记"之前的梯度历史

### 3.3 scheduler（学习率调度器状态）

```python
scheduler.state_dict() 包含：
├── last_epoch: 500        # 已执行的 step 数
├── _step_count: 501       # 内部计数器
├── base_lrs: [0.0001]     # 基础学习率
└── _last_lr: [0.00005]    # 当前学习率
```

**功能**: OneCycleLR 需要知道当前在学习率曲线的哪个位置

**恢复后**: 学习率从断点处继续，保持曲线连续

### 3.4 epoch 和 global_step

```python
epoch: 0         # 当前是第几个 epoch（从 0 开始）
global_step: 500 # 总共训练了多少个 step
```

**功能**:
- 确定从哪个 epoch 继续
- 计算需要跳过多少个 batch

### 3.5 best_score 和 val_loss_min

```python
best_score: -0.45      # = -val_loss（越大越好）
val_loss_min: 0.45     # 最小验证损失
```

**功能**:
- 断点续训后，EarlyStopping 知道之前的最佳分数
- 不会错误地保存更差的模型

---

## 4. 你的任务完整性评估

### 4.1 在算力有限条件下，你的方案是否完整？

**✅ 是的，你的方案是完整的。**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    断点续训完整性检查表                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [✅] 模型参数保存与恢复                                            │
│       └── model.state_dict() → model.load_state_dict()             │
│                                                                     │
│  [✅] 优化器状态保存与恢复                                          │
│       └── 动量、历史梯度等完整恢复                                  │
│                                                                     │
│  [✅] 学习率调度器保存与恢复                                        │
│       └── 学习率曲线从断点继续                                      │
│                                                                     │
│  [✅] 训练进度保存与恢复                                            │
│       └── epoch + global_step → 跳过已训练的 batch                 │
│                                                                     │
│  [✅] EarlyStopping 状态保存与恢复 (你刚刚修复的)                   │
│       └── best_score + val_loss_min → 不会覆盖更好的模型            │
│                                                                     │
│  [✅] 最佳模型独立保存                                              │
│       └── checkpoint 文件不受 save_total_limit 影响                │
│                                                                     │
│  [✅] 数据不重复学习                                                │
│       └── resume_step_in_epoch 跳过已训练的 batch                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 你的训练流程

```
第 1 次运行
│
├── Step 0-999: 训练
├── Step 1000: 保存 checkpoint_step_1000
├── Step 1001-1999: 训练
├── Step 2000: 保存 checkpoint_step_2000
├── 中断（算力不足/Colab 断连）
│
└── checkpoint_step_2000 保存了完整状态

第 2 次运行（断点续训）
│
├── 加载 checkpoint_step_2000
├── 恢复: model, optimizer, scheduler, epoch, global_step, best_score
├── 跳过 Step 0-1999
├── 继续 Step 2000+
├── ...
├── Epoch 1 结束 → vali_loss = 0.45 → 保存 checkpoint (best)
├── Step 3000: 保存 checkpoint_step_3000
├── 中断
│
└── checkpoint_step_3000 保存了完整状态（含 best_score = -0.45）

第 3 次运行（断点续训）
│
├── 加载 checkpoint_step_3000
├── 恢复: best_score = -0.45 ✅
├── 继续训练...
├── Epoch 2 结束 → vali_loss = 0.50 → 不保存（没有改善）✅
├── Epoch 3 结束 → vali_loss = 0.40 → 保存 checkpoint (new best)
│
└── ... 直到训练完成

最终结果
│
├── checkpoint: 验证集上表现最好的模型 → 用于推理
└── checkpoint_step_N: 最后一个断点 → 备用
```

### 4.3 总结

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 模型参数连续性 | ✅ | 完整保存和恢复 |
| 优化器状态连续性 | ✅ | 动量等完整恢复 |
| 学习率曲线连续性 | ✅ | scheduler 状态恢复 |
| 训练进度连续性 | ✅ | 不重复训练 batch |
| 最佳模型保护 | ✅ | best_score 恢复（你修复的） |
| 数据一致性 | ✅ | 固定随机种子 |

**你的方案在算力有限、需要断点续训的条件下是完整的。可以放心使用！**

---

## 5. 推荐的使用流程

```bash
# 1. 首次训练
RESUME_FROM=""
bash scripts/TimeLLM_ETTh1_llama3_colab_drive.sh

# 2. 中断后续训
RESUME_FROM="/path/to/checkpoint_step_N"
bash scripts/TimeLLM_ETTh1_llama3_colab_drive.sh

# 3. 训练完成后推理
# 加载 checkpoint (不是 checkpoint_step_N)
ckpt = torch.load('checkpoints/.../checkpoint')
model.load_state_dict(ckpt['model'])
model.eval()
```

---

*文档生成于 2026-01-26*
