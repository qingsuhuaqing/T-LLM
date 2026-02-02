#!/bin/bash
# ============================================================
# Colab 运行脚本 (Llama 3 8B + 4-bit, 15GB GPU)
# ============================================================

# ====== 手动修改区 (路径/策略) ======
# 项目目录（如果已“全盘云端化”，改成 Drive 路径）
PROJECT_DIR="/content/drive/MyDrive/Time-LLM-Project"
# 如仍在临时盘，则用：/content/T-L
# PROJECT_DIR="/content/T-L"

# 模型目录（提前下载到此处）
LLM_PATH="${PROJECT_DIR}/base_models/Meta-Llama-3-8B-Instruct"

# Checkpoint 保存到 Drive（避免断电丢失）
CHECKPOINTS="/content/drive/MyDrive/Time-LLM-Backup/checkpoints"

# 每 N step 保存一次（0=关闭）
SAVE_STEPS=1000

# 断点续训（留空=不启用）
RESUME_FROM=""

# 训练参数
LLM_DIM=4096
LLM_LAYERS=32
BATCH=6  # 显存紧张就降到 4
# ===================================

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model_comment Llama3_8B_4bit_Colab \
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
  --llm_model_path "${LLM_PATH}" \
  --load_in_4bit \
  --checkpoints "${CHECKPOINTS}" \
  --save_steps ${SAVE_STEPS} \
  --resume_from_checkpoint "${RESUME_FROM}"

# 说明:
# 1) --save_steps 会在训练中每 N step 额外保存一次 checkpoint。
# 2) --resume_from_checkpoint 指向 checkpoint_step_xxx 或 checkpoint.pt 即可继续训练。
