#!/bin/bash
# ============================================================
# Colab + Drive 运行脚本 (GPT-2)
# 目标: 15GB GPU / 12GB RAM 下尽量吃满显存
# ============================================================

# ====== 手动修改区（路径/策略/显存）======
# 项目根目录（你已固定在 Drive）
PROJECT_DIR="/content/drive/MyDrive/T-L-GPT2"

# GPT-2 模型目录（相对路径保持一致）
LLM_PATH="${PROJECT_DIR}/base_models/openai-community/gpt2"

# Checkpoint 保存到 Drive（断电不丢）
CHECKPOINTS="${PROJECT_DIR}/checkpoints"

# 日志保存目录（Drive）
LOG_DIR="${PROJECT_DIR}/logs"

# 每 N step 保存一次（0=关闭）
SAVE_STEPS=1757
# 只保留最近 N 个 step checkpoint（0=关闭）
SAVE_TOTAL_LIMIT=20

# 断点续训（第一次留空）
# RESUME_FROM=""
# 断点续训（第二次设置3000step）
# （第三次设置6000step）
# （第四次设置9000step）
# （第五次设置11000step）
RESUME_FROM="/content/drive/MyDrive/T-L-GPT2/checkpoints/long_term_forecast_ETTh1_512_96_TimeLLM_ETTh1_ftM_sl512_ll48_pl96_dm64_nh8_el2_dl1_df128_fc3_ebtimeF_test_0-GPT2_ColabDrive_15GB/checkpoint_step_11000/checkpoint.pt"
# 仅首次恢复时手动覆盖 EarlyStopping 计数（-1=不用）
# 示例：如果断点前已连续 3 次未提升，可设 RESUME_COUNTER=3
# 考虑手动调整,阻止之前的不良影响
RESUME_COUNTER=-1

# 随机种子与确定性（压缩“断点震荡”）
SEED=2021
# 1=开启确定性（更慢但更稳定）；0=关闭
DETERMINISTIC=0


# ====== 训练参数（15GB 显存优先）======
# GPT-2 固定规格
LLM_DIM=768
LLM_LAYERS=12

# 可适当加大以吃满显存
BATCH=32
D_MODEL=64
D_FF=128
SEQ_LEN=512
LABEL_LEN=48
PRED_LEN=96
TRAIN_EPOCHS=50
#适当保留验证集的严谨和容错
PATIENCE=10
# ===================================

# ====== 断点续训可改/不可改说明 ======
# ✅ 可改（不影响模型形状）：
#   RESUME_FROM / SAVE_STEPS / SAVE_TOTAL_LIMIT / LOG_DIR / CHECKPOINTS / BATCH / NUM_WORKERS
#   （注意: BATCH 改大/改小只影响训练速度，不影响权重形状）
#
# ❌ 不可改（会导致形状不匹配或语义不一致）：
#   LLM_PATH / LLM_DIM / LLM_LAYERS / LLM_MODEL
#   SEQ_LEN / LABEL_LEN / PRED_LEN / PATCH_LEN / STRIDE
#   D_MODEL / D_FF / N_HEADS / ENC_IN / DEC_IN / C_OUT / FEATURES / DATA
#
# ⚠️ 谨慎：
#   learning_rate / lradj / pct_start / train_epochs
#   断点恢复会加载 optimizer/scheduler 状态，改了也可能不生效。
# ===================================

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_gpt2_$(date +%Y%m%d_%H%M%S).log"

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
  --model_comment GPT2_ColabDrive_15GB \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
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
  --seed ${SEED} \
  $( [ "${DETERMINISTIC}" -eq 1 ] && echo "--deterministic" ) \
  --prompt_domain 1 \
  --train_epochs ${TRAIN_EPOCHS} \
  --patience ${PATIENCE} \
  --itr 1 \
  --dropout 0.1 \
  --llm_model GPT2 \
  --llm_model_path "${LLM_PATH}" \
  --checkpoints "${CHECKPOINTS}" \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --resume_from_checkpoint "${RESUME_FROM}" \
  --resume_counter ${RESUME_COUNTER} \
  2>&1 | tee -a "${LOG_FILE}"

# OOM 处理建议:
# 1) 降 BATCH (16->12->8)
# 2) 降 SEQ_LEN (512->384->256)
# 3) 降 D_FF (128->64)
# 4) 降 D_MODEL (64->32)
# 5) 必要时启用 --load_in_4bit
