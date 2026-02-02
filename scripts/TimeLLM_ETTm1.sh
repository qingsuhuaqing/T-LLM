#!/bin/bash

# ========================================================================
# Time-LLM Training Script for ETTm1 Dataset
# ========================================================================
# 模型: Qwen 2.5 3B + 4-bit NF4 量化
# 硬件: 6GB VRAM GPU
# 数据集: ETTm1 (15分钟级别, 7变量)
# 任务: 长期预测 (512步输入 → 96步预测)
# ========================================================================

# 切换到项目目录 (WSL 路径)
cd /mnt/e/timellm-chuangxin/Time-LLM

# 设置显存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# 基础配置
# --task_name: 任务类型 (长期预测)
# --is_training: 训练模式开关 (1=训练, 0=测试)
# --model_id: 模型标识符 (用于生成checkpoint名称)
# --model_comment: 模型备注 (附加到checkpoint名称末尾)
# --model: 模型架构 (TimeLLM)
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTm1_512_96 \
  --model_comment Qwen3B \
  --model TimeLLM \
  \
  `# 数据配置` \
  `# --data: 数据集名称` \
  `# --root_path: 数据集根目录` \
  `# --data_path: 数据文件名` \
  `# --features: M=多变量预测多变量 (7→7)` \
  --data ETTm1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --features M \
  \
  `# 时序长度配置` \
  `# --seq_len: 输入序列长度 (历史观测窗口)` \
  `# --label_len: Decoder起始token长度` \
  `# --pred_len: 预测长度 (未来预测窗口)` \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  \
  `# 模型结构配置` \
  `# --enc_in: Encoder输入变量数 (ETT有7个特征)` \
  `# --dec_in: Decoder输入变量数` \
  `# --c_out: 输出变量数` \
  `# --d_model: 模型隐藏层维度 (Patch Embedding维度)` \
  `# --n_heads: 多头注意力的头数` \
  `# --e_layers: Encoder层数 (保留兼容性)` \
  `# --d_layers: Decoder层数 (保留兼容性)` \
  `# --d_ff: Feed-Forward网络维度` \
  `# --factor: 注意力因子 (保留兼容性)` \
  `# --dropout: Dropout概率` \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --factor 3 \
  --dropout 0.1 \
  \
  `# LLM配置 (最关键部分)` \
  `# --llm_model: LLM基座模型类型 (QWEN标识)` \
  `# --llm_dim: LLM隐藏层维度 (Qwen 2.5 3B=2048)` \
  `# --llm_model_path: 本地模型路径` \
  `# --llm_layers: 使用的LLM层数 (⭐必需! 不设置会使用32层导致OOM)` \
  `# --load_in_4bit: 启用4-bit量化 (⭐显存从6GB降至1.5GB)` \
  --llm_model QWEN \
  --llm_dim 2048 \
  --llm_model_path /mnt/e/timellm-chuangxin/Time-LLM/base_models/Qwen2.5-3B \
  --llm_layers 6 \
  --load_in_4bit \
  \
  `# 训练优化配置` \
  `# --batch_size: 训练批大小 (6GB显存推荐8, OOM时降至4或2)` \
  `# --num_workers: 数据加载器进程数 (⭐默认10太高, 推荐2)` \
  `# --train_epochs: 训练轮数` \
  `# --itr: 实验重复次数` \
  `# --prompt_domain: 启用领域提示词 (⭐必需! 设为1才会加载ETT.txt)` \
  --batch_size 8 \
  --num_workers 2 \
  --train_epochs 10 \
  --itr 1 \
  --prompt_domain 1

# ========================================================================
# 显存占用估算 (Qwen 2.5 3B + 4-bit, 6层)
# ========================================================================
# Qwen 2.5 3B (4-bit, 6层):  ~1.5 GB
# Time-LLM 可训练参数:       ~0.5 GB
# 中间变量 (batch_size=8):   ~2.0 GB
# 梯度缓存:                  ~0.5 GB
# 系统占用:                  ~0.5 GB
# ----------------------------------------
# 总计:                      ~5.0 GB ✅
# ========================================================================

# ========================================================================
# 生成的 Checkpoint 路径
# ========================================================================
# checkpoints/long_term_forecast_ETTm1_512_96_TimeLLM_ETTm1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-Qwen3B/checkpoint
# ========================================================================

# ========================================================================
# 常见问题与解决
# ========================================================================
# 1. OOM (显存不足):
#    - 降低 --batch_size 为 4 或 2
#    - 降低 --seq_len 为 256
#    - 降低 --llm_layers 为 4
#
# 2. 提示词未加载:
#    - 确保 --prompt_domain 1
#    - 检查 dataset/prompt_bank/ETT.txt 存在
#
# 3. 模型加载失败:
#    - 确认 base_models/Qwen2.5-3B/ 下有完整文件
#    - 确认 config.json 和 tokenizer.json 存在
# ========================================================================
