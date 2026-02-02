#!/bin/bash
# ============================================================
# Colab + Drive 运行脚本 (Llama 3 8B + 4-bit)
# 目标: 15GB GPU / 12GB RAM 下尽量吃满显存
# ============================================================

# ====== 手动修改区（路径/策略）======
# 项目根目录（你已固定在 Drive）
PROJECT_DIR="/content/drive/MyDrive/T-L"

# Llama 3 8B 模型目录（相对路径保持一致）
LLM_PATH="${PROJECT_DIR}/base_models/LLM-Research/Meta-Llama-3-8B-Instruct"

# Checkpoint 保存到 Drive（断电不丢）
CHECKPOINTS="${PROJECT_DIR}/checkpoints"

# 日志保存目录（Drive）
LOG_DIR="${PROJECT_DIR}/logs"

# 每 N step 保存一次（0=关闭）
SAVE_STEPS=1000
# 只保留最近 N 个 step checkpoint（0=关闭）
SAVE_TOTAL_LIMIT=2

# 断点续训（第一次留空）
RESUME_FROM=""

# 训练参数（可按显存调整）
LLM_DIM=4096
LLM_LAYERS=16
BATCH=2       # OOM 时降到 4
D_MODEL=32
D_FF=32        # OOM 时降到 32
SEQ_LEN=256    # OOM 时可降到 384/256
PRED_LEN=96
# ===================================

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_llama3_8b_$(date +%Y%m%d_%H%M%S).log"

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
  --model_comment Llama3_8B_4bit_ColabDrive \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len 48 \
  --pred_len ${PRED_LEN} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size ${BATCH} \
  --d_model ${D_MODEL} \
  --n_heads 8 \
  --d_ff ${D_FF} \
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
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --resume_from_checkpoint "${RESUME_FROM}" \
  2>&1 | tee -a "${LOG_FILE}"

# OOM 处理建议:
# 1) 降 BATCH (6->4->2)
# 2) 降 LLM_LAYERS (32->24->16)
# 3) 降 SEQ_LEN (512->384->256)
# 4) 降 D_FF (64->32)
