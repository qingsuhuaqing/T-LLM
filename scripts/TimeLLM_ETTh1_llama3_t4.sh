#!/bin/bash
# ============================================================
# Time-LLM 训练脚本 (Llama 3 8B + 4-bit, 面向 T4 16GB)
# 目标: 在可运行的前提下尽量提高显存利用
# ============================================================

cd /mnt/e/timellm-chuangxin/Time-LLM

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# ---------- 可调参数 ----------
# 本地 Llama 3 8B 权重目录 (按你的实际目录改)
LLM_PATH="/mnt/e/timellm-chuangxin/Time-LLM/base_models/Meta-Llama-3-8B"
# Llama 3 8B: hidden size = 4096, layers = 32
LLM_DIM=4096
# 先给一个“高但通常可跑”的层数，想更吃满显存可逐步加到 24/28/32
LLM_LAYERS=32
# 显存允许时可逐步加到 8/12/16；若 OOM 就回退
BATCH=8

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model_comment Llama3_8B_4bit_T4 \
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
  --batch_size ${BATCH} \
  --d_model 32 \
  --n_heads 8 \
  --d_ff 64 \
  --llm_dim ${LLM_DIM} \
  --llm_layers ${LLM_LAYERS} \
  --num_workers 2 \
  --prompt_domain 1 \
  --train_epochs 3 \
  --itr 1 \
  --dropout 0.1 \
  --llm_model LLAMA \
  --llm_model_path ${LLM_PATH} \
  --load_in_4bit

# 如果显存溢出:
# 1) 先降 BATCH (8 -> 4)
# 2) 再降 LLM_LAYERS (20 -> 16 -> 12)
