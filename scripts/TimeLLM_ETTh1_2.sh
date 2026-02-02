#!/bin/bash
# ============================================================
# Time-LLM 快速训练脚本 (Qwen 2.5 3B + 4-bit 量化)
# 数据集: ETTh1 (小时级数据，训练更快)
# 目标: 3小时内完成训练
# 显存要求: 6GB
# ============================================================

# 切换到项目目录 (WSL 路径)
cd /mnt/e/timellm-chuangxin/Time-LLM

# 设置显存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# ============================================================
# 训练时间估算
# ============================================================
# ETTh1 训练集: ~12,000 行 (vs ETTm1 的 ~48,000 行)
# 每个 epoch: ~3,000 迭代 (batch_size=4)
# 每个迭代: ~1.1 秒
# 每个 epoch: ~55 分钟
# 3 个 epoch: ~2.75 小时 ✅
# ============================================================

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

# ============================================================
# 训练完成后
# ============================================================
# Checkpoint 保存在:
# ./checkpoints/long_term_forecast_ETTh1_512_96_TimeLLM_ETTh1_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-Qwen3B_fast/
#
# 注意: run_main.py 中的 del_files 已被注释，checkpoint 会保留
# ============================================================
