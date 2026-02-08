#!/bin/bash
# ============================================================
# Time-LLM Enhanced 训练脚本 (GPT-2 + TAPR + GRA-M)
# ============================================================
#
# 本脚本基于原始 GPT-2 配置，新增了两个创新模块的参数：
#
# 【创新模块一】TAPR (Trend-Aware Patch Router) 趋势感知尺度路由
#   ├── DM (Decomposable Multi-scale): 可分解多尺度混合
#   │   - 通过平均池化将输入下采样为多个分辨率
#   │   - 在每个尺度上进行季节性-趋势分解
#   │   - 从粗到细的跨尺度注意力融合
#   │
#   └── C2F (Coarse-to-Fine Head): 先粗后细预测头
#       - Level 1: 方向分类 (上涨/下跌/平稳)
#       - Level 2: 幅度分类 (强涨/弱涨/弱跌/强跌)
#       - Level 3: 条件回归 (具体数值)
#       - 层次化一致性损失 (HCL) 约束
#
# 【创新模块二】GRA-M (Global Retrieval-Augmented Memory) 全局检索增强记忆
#   ├── PVDR (Pattern-Value Dual Retriever): 模式-数值双重检索器
#   │   - 检索与当前输入相似的历史片段
#   │   - 同时获取历史片段后续的"未来值"作为参考
#   │   - 结合形状(Shape)和数值(Value)进行相似度度量
#   │
#   └── AKG (Adaptive Knowledge Gating): 自适应知识门控
#       - 动态计算检索内容与当前输入的权重
#       - 决定"听从历史经验"还是"依赖当前推理"
#       - 交叉注意力融合检索结果
#
# 架构流程:
#   Input → Normalize → [TAPR: Multi-Scale + C2F] → Patch Embedding →
#          [GRA-M: Retrieval + Gating] → Reprogramming → LLM → Output
#
# 修改日志 (2026-02-08):
#   - 增大辅助损失权重 (lambda_trend: 0.1 -> 0.5, lambda_retrieval: 0.1 -> 0.3)
#   - 降低学习率 (0.001 -> 0.0005) 防止过拟合
#   - 增加 dropout (0.1 -> 0.15) 增强正则化
#   - 增加 patience (10 -> 15) 给模型更多收敛时间
#
# ============================================================

# ====== 手动修改区（路径/策略/显存）======

# 项目根目录
# PROJECT_DIR="/mnt/e/timellm-colab-fuxian-chaungxin-bishe/Time-LLM"
PROJECT_DIR="/content/drive/MyDrive/T-L-GPT2-Enhanced"

# GPT-2 模型目录
# LLM_PATH="${PROJECT_DIR}/base_models/gpt2"
LLM_PATH="${PROJECT_DIR}/base_models/openai-community/gpt2"
# Checkpoint 保存目录
CHECKPOINTS="${PROJECT_DIR}/checkpoints"

# 日志保存目录
LOG_DIR="${PROJECT_DIR}/logs"

# 每 N step 保存一次（0=关闭）
# 建议: 根据数据集大小设置，ETTh1约8500*2样本，batch_size=32时约1757步/epoch
SAVE_STEPS=1755
# 只保留最近 N 个 step checkpoint（0=关闭）
SAVE_TOTAL_LIMIT=10

# 断点续训（第一次留空）
RESUME_FROM=""
# 断点续训示例:
RESUME_FROM="${CHECKPOINTS}/checkpoints/long_term_forecast_ETTh1_512_96_TimeLLM_ETTh1_ftM_sl512_ll48_pl96_dm64_nh8_el2_dl1_df128_fc3_ebtimeF_test_0-GPT2_Enhanced/checkpoint"

# 仅首次恢复时手动覆盖 EarlyStopping 计数（-1=不用）
RESUME_COUNTER=-1

# 随机种子与确定性
SEED=2021
# 1=开启确定性（更慢但更稳定）；0=关闭
DETERMINISTIC=0


# ====== GPT-2 模型参数（固定）======
# GPT-2 词向量维度
LLM_DIM=768
# 使用的 GPT-2 层数（6GB显存建议6层，15GB可用12层）
LLM_LAYERS=12


# ====== 数据与预测参数 ======
# 输入序列长度（历史窗口）
SEQ_LEN=512
# 标签长度（解码器输入）
LABEL_LEN=48
# 预测长度
PRED_LEN=96
# 批次大小（6GB显存建议8-16，15GB可用32）
BATCH=32


# ====== 模型结构参数 ======
# Patch嵌入维度
D_MODEL=64
# FFN隐藏层维度
D_FF=128
# 注意力头数
N_HEADS=8
# 编码器层数
E_LAYERS=2
# 解码器层数
D_LAYERS=1


# ====== 训练参数 ======
# 训练轮数
TRAIN_EPOCHS=50
# 早停耐心（连续N轮验证损失不降则停止）
# 修改: 从10增加到15，给模型更多收敛时间
PATIENCE=15
# 学习率
# 修改: 从0.001降低到0.0005，防止过拟合
LEARNING_RATE=0.0005
# Dropout比例
# 修改: 从0.1增加到0.15，增强正则化
DROPOUT=0.15


# ====== 新增: TAPR 模块参数 ======
# 是否启用 TAPR 模块
# --use_tapr: 添加此标志启用，不添加则禁用
USE_TAPR=1

# 多尺度分解的尺度数量（默认3: 原始/4x下采样/16x下采样）
# 更多尺度可捕获更丰富的时间模式，但增加计算量
N_SCALES=3

# 趋势辅助损失权重
# 控制趋势分类损失对总损失的贡献
# 修改: 从0.1增加到0.5，让趋势分类真正起作用
LAMBDA_TREND=0.5


# ====== 新增: GRA-M 模块参数 ======
# 是否启用 GRA-M 模块
USE_GRAM=1

# 检索的相似模式数量（Top-K）
# 更多检索结果提供更丰富参考，但可能引入噪声
TOP_K=5

# 检索辅助损失权重
# 控制门控一致性损失对总损失的贡献
# 修改: 从0.1增加到0.3，让检索模块真正起作用
LAMBDA_RETRIEVAL=0.3

# 是否构建检索记忆库
# 首次训练时启用，从训练数据构建历史模式库
BUILD_MEMORY=0

# 模式表示向量维度（用于检索）
# 较大维度可表示更复杂模式，但增加内存占用
D_REPR=128


# ====== 断点续训可改/不可改说明 ======
# ✅ 可改（不影响模型形状）：
#   RESUME_FROM / SAVE_STEPS / SAVE_TOTAL_LIMIT / LOG_DIR / CHECKPOINTS
#   BATCH / NUM_WORKERS / LEARNING_RATE / TRAIN_EPOCHS / PATIENCE
#   LAMBDA_TREND / LAMBDA_RETRIEVAL (辅助损失权重)
#
# ❌ 不可改（会导致形状不匹配或语义不一致）：
#   LLM_PATH / LLM_DIM / LLM_LAYERS
#   SEQ_LEN / LABEL_LEN / PRED_LEN / PATCH_LEN / STRIDE
#   D_MODEL / D_FF / N_HEADS / ENC_IN / DEC_IN / C_OUT
#   USE_TAPR / USE_GRAM / N_SCALES / TOP_K / D_REPR
#   (模块开关和结构参数一旦训练开始不可更改)
#
# ⚠️ 谨慎：
#   BUILD_MEMORY: 断点续训时应设为0，避免重建记忆库
# ====================================


# ====== 执行训练 ======

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_enhanced_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "Time-LLM Enhanced 训练开始"
echo "============================================================"
echo "TAPR模块: $( [ "${USE_TAPR}" -eq 1 ] && echo '启用' || echo '禁用' )"
echo "  - 尺度数量: ${N_SCALES}"
echo "  - 趋势损失权重: ${LAMBDA_TREND}"
echo "GRA-M模块: $( [ "${USE_GRAM}" -eq 1 ] && echo '启用' || echo '禁用' )"
echo "  - Top-K: ${TOP_K}"
echo "  - 检索损失权重: ${LAMBDA_RETRIEVAL}"
echo "  - 构建记忆库: $( [ "${BUILD_MEMORY}" -eq 1 ] && echo '是' || echo '否' )"
echo "============================================================"
echo "训练参数调整:"
echo "  - 学习率: ${LEARNING_RATE} (降低防止过拟合)"
echo "  - Dropout: ${DROPOUT} (增加正则化)"
echo "  - Patience: ${PATIENCE} (增加收敛时间)"
echo "============================================================"

# 构建命令参数
CMD_ARGS=""

# TAPR参数
if [ "${USE_TAPR}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --use_tapr"
  CMD_ARGS="${CMD_ARGS} --n_scales ${N_SCALES}"
  CMD_ARGS="${CMD_ARGS} --lambda_trend ${LAMBDA_TREND}"
fi

# GRA-M参数
if [ "${USE_GRAM}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --use_gram"
  CMD_ARGS="${CMD_ARGS} --top_k ${TOP_K}"
  CMD_ARGS="${CMD_ARGS} --lambda_retrieval ${LAMBDA_RETRIEVAL}"
  CMD_ARGS="${CMD_ARGS} --d_repr ${D_REPR}"
  if [ "${BUILD_MEMORY}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --build_memory"
  fi
fi

# 断点续训参数
if [ -n "${RESUME_FROM}" ]; then
  CMD_ARGS="${CMD_ARGS} --resume_from_checkpoint ${RESUME_FROM}"
  CMD_ARGS="${CMD_ARGS} --resume_counter ${RESUME_COUNTER}"
fi

# 确定性训练
if [ "${DETERMINISTIC}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --deterministic"
fi

# 执行训练
python run_main_enhanced.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
  --model_comment GPT2_Enhanced \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --e_layers ${E_LAYERS} \
  --d_layers ${D_LAYERS} \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --batch_size ${BATCH} \
  --d_model ${D_MODEL} \
  --n_heads ${N_HEADS} \
  --d_ff ${D_FF} \
  --llm_dim ${LLM_DIM} \
  --llm_layers ${LLM_LAYERS} \
  --num_workers 2 \
  --seed ${SEED} \
  --prompt_domain 1 \
  --learning_rate ${LEARNING_RATE} \
  --train_epochs ${TRAIN_EPOCHS} \
  --patience ${PATIENCE} \
  --itr 1 \
  --dropout ${DROPOUT} \
  --llm_model GPT2 \
  --llm_model_path "${LLM_PATH}" \
  --checkpoints "${CHECKPOINTS}" \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  ${CMD_ARGS} \
  2>&1 | tee -a "${LOG_FILE}"

echo "============================================================"
echo "训练完成，日志保存至: ${LOG_FILE}"
echo "============================================================"


# ============================================================
# 消融实验指南
# ============================================================
#
# 【实验1】仅TAPR模块 (验证多尺度分解效果)
#   USE_TAPR=1
#   USE_GRAM=0
#
# 【实验2】仅GRA-M模块 (验证检索增强效果)
#   USE_TAPR=0
#   USE_GRAM=1
#
# 【实验3】完整模型 (TAPR + GRA-M)
#   USE_TAPR=1
#   USE_GRAM=1
#
# 【实验4】基线模型 (原始Time-LLM)
#   USE_TAPR=0
#   USE_GRAM=0
#   (此时可直接使用 run_main.py 而非 run_main_enhanced.py)
#
# 【参数敏感性实验】
# - N_SCALES: 测试 2, 3, 4 对比多尺度效果
# - TOP_K: 测试 3, 5, 10 对比检索数量影响
# - LAMBDA_TREND: 测试 0.3, 0.5, 0.7 对比辅助损失权重
# - LAMBDA_RETRIEVAL: 测试 0.2, 0.3, 0.5 对比辅助损失权重
#
# ============================================================


# ============================================================
# OOM 处理建议 (显存不足时)
# ============================================================
#
# 优先级从高到低:
# 1) 降 BATCH (16->12->8->4)
# 2) 降 SEQ_LEN (512->384->256->128)
# 3) 降 LLM_LAYERS (6->4->2)
# 4) 降 D_FF (128->64->32)
# 5) 降 D_MODEL (64->32->16)
# 6) 降 N_SCALES (3->2)
# 7) 降 TOP_K (5->3)
# 8) 禁用 GRA-M (BUILD_MEMORY 消耗额外显存)
# 9) 启用 --load_in_4bit (如果使用Qwen等大模型)
#
# 6GB 显存推荐配置:
#   BATCH=8, SEQ_LEN=256, LLM_LAYERS=4, D_MODEL=32, D_FF=64
#
# 15GB 显存推荐配置:
#   BATCH=32, SEQ_LEN=512, LLM_LAYERS=12, D_MODEL=64, D_FF=128
#
# ============================================================
