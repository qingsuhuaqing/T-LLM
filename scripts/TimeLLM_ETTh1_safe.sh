#!/bin/bash
# ============================================================
# Time-LLM 安全训练脚本 (带保护措施)
# 数据集: ETTh1
# 显存要求: 6GB
# 特点: 更保守的参数 + 温度监控提示
# ============================================================

cd /mnt/e/timellm-chuangxin/Time-LLM

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

# ============================================================
# 安全配置说明:
# - batch_size=2 (更保守，显存更安全)
# - seq_len=256 (缩短序列，减少计算量)
# - train_epochs=2 (先跑2轮验证稳定性)
# - 预计时间: ~40-50 分钟
# ============================================================

echo "=========================================="
echo "开始训练前检查..."
echo "=========================================="
echo ""
echo "建议操作:"
echo "1. 打开另一个终端运行: watch -n 2 nvidia-smi"
echo "2. 观察 GPU 温度，保持在 80°C 以下"
echo "3. 如果温度过高，可按 Ctrl+C 中断"
echo ""
echo "5秒后开始训练..."
sleep 5

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_256_96 \
  --model_comment Qwen3B_safe \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 256 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size 2 \
  --d_model 32 \
  --n_heads 8 \
  --d_ff 32 \
  --llm_dim 2048 \
  --llm_layers 6 \
  --num_workers 2 \
  --prompt_domain 1 \
  --train_epochs 2 \
  --itr 1 \
  --dropout 0.1 \
  --llm_model QWEN \
  --llm_model_path /mnt/e/timellm-chuangxin/Time-LLM/base_models/Qwen2.5-3B \
  --load_in_4bit

echo ""
echo "=========================================="
echo "训练完成！"
echo "如果一切正常，可以运行完整版本:"
echo "bash ./scripts/TimeLLM_ETTh1_2.sh"
echo "=========================================="
