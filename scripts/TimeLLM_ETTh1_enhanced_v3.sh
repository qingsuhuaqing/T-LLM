#!/bin/bash

# ========================================================================
# Time-LLM Enhanced v3 Training Script — ETTh1 Dataset
# ========================================================================
# Model:    Qwen 2.5 3B + 4-bit NF4 quantization
# Hardware: 6GB VRAM GPU (GTX 1660 Ti)
# Dataset:  ETTh1 (hourly, 7 variables)
# Task:     Long-term forecast (512 -> 96)
# ========================================================================

cd /mnt/e/timellm-colab-fuxian-chaungxin-bishe/Time-LLM

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# ========================================================================
# Configuration presets
# ========================================================================
# E1: Baseline only (no TAPR, no GRAM)
# E2: +TAPR (DM + C2F)
# E3: +GRAM (PVDR + AKG)
# E4: Full  (DM + C2F + PVDR + AKG)
# ========================================================================

# Default: E4 (Full)
USE_TAPR=${USE_TAPR:-1}
USE_GRAM=${USE_GRAM:-1}

TAPR_FLAG=""
GRAM_FLAG=""
MEMORY_FLAG=""
COMMENT="v3"

if [ "$USE_TAPR" -eq 1 ]; then
    TAPR_FLAG="--use_tapr"
    COMMENT="${COMMENT}_TAPR"
fi

if [ "$USE_GRAM" -eq 1 ]; then
    GRAM_FLAG="--use_gram --build_memory"
    COMMENT="${COMMENT}_GRAM"
fi

echo "=========================================="
echo "Running: ${COMMENT}"
echo "  TAPR (DM+C2F): $([ "$USE_TAPR" -eq 1 ] && echo 'ON' || echo 'OFF')"
echo "  GRAM (PVDR+AKG): $([ "$USE_GRAM" -eq 1 ] && echo 'ON' || echo 'OFF')"
echo "=========================================="

python run_main_enhanced_v3.py \
  `# Basic config` \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_512_96 \
  --model_comment "${COMMENT}" \
  --model TimeLLM \
  \
  `# Data config` \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  \
  `# Sequence config` \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  \
  `# Model structure` \
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
  `# LLM config` \
  --llm_model QWEN \
  --llm_dim 2048 \
  --llm_model_path /mnt/e/timellm-colab-fuxian-chaungxin-bishe/Time-LLM/base_models/Qwen2.5-3B \
  --llm_layers 6 \
  --load_in_4bit \
  \
  `# v3 TAPR (DM + C2F) parameters` \
  ${TAPR_FLAG} \
  --n_scales 5 \
  --downsample_rates "1,2,4,8,16" \
  --lambda_consist 0.1 \
  --consistency_mode hybrid \
  --decay_factor 0.5 \
  \
  `# v3 GRAM (PVDR + AKG) parameters` \
  ${GRAM_FLAG} \
  --lambda_gate 0.01 \
  --similarity_threshold 0.8 \
  --extreme_sigma 3.0 \
  --extreme_threshold_reduction 0.2 \
  --top_k 5 \
  --d_repr 64 \
  \
  `# Training config` \
  --batch_size 4 \
  --num_workers 2 \
  --train_epochs 10 \
  --itr 1 \
  --prompt_domain 1 \
  --warmup_steps 500 \
  --patience 10 \
  --test_epochs "final"

# ========================================================================
# Ablation experiments:
#
# E1 (Baseline only):
#   USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh
#
# E2 (+TAPR only):
#   USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh
#
# E3 (+GRAM only):
#   USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh
#
# E4 (Full):
#   USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v3.sh
# ========================================================================

# ========================================================================
# VRAM estimate (~5.3 GB):
#   Qwen 2.5 3B (4-bit, 6 layers):  ~1.5 GB
#   Time-LLM trainable params:      ~0.5 GB
#   DM (4 ScaleEncoders):           ~2 MB
#   C2F weight matrix:              < 1 MB
#   PVDR memory banks (5 scales):   ~5 MB
#   AKG matrices:                   < 1 MB
#   Intermediate tensors:           ~0.3 GB
#   System overhead:                ~1.0 GB
#   Total:                          ~5.3 GB
# ========================================================================
