#!/bin/bash
# ============================================================
# Time-LLM 训练脚本 (Qwen 2.5 3B + 4-bit 量化)
# 数据集: ETTm1
# 显存要求: 6GB (推荐 8GB+)
# ============================================================

# 切换到项目目录
cd /mnt/e/timellm-chuangxin/Time-LLM

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# ============================================================
# 训练配置 (建议值，可根据显存调整)
# ============================================================
# --batch_size: 首次运行建议 4，成功后可尝试 8
# --seq_len: 512 为论文配置，显存不足时可降至 256
# ============================================================

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model_comment Qwen3B \
  --model TimeLLM \
  --data ETTm1 \
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
  --d_ff 32 \
  --llm_dim 2048 \
  --llm_layers 6 \
  --num_workers 2 \
  --prompt_domain 1 \
  --train_epochs 10 \
  --itr 1 \
  --dropout 0.1 \
  --llm_model_path "/mnt/e/timellm-chuangxin/Time-LLM/base_models/Qwen2.5-3B" \
  --load_in_4bit
