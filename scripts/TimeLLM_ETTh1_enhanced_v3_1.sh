#!/bin/bash
# ============================================================
# Time-LLM Enhanced v3 训练脚本 (Qwen 2.5 3B + TAPR + GRA-M)
# ============================================================
#
# 本脚本基于 Qwen 2.5 3B 配置，集成 v3 创新模块参数：
#
# 【创新模块一】TAPR (Trend-Aware Patch Router) 趋势感知尺度路由
#   ├── DM (Decomposable Multi-scale): 可分解多尺度独立预测
#   │   - 5个尺度 (S1=baseline, S2..S5=辅助)
#   │   - 下采样率: 1, 2, 4, 8, 16
#   │   - 信号变换: identity, diff, smooth, trend
#   │   - 每个尺度产生独立的 [B, pred_len, N] 预测
#   │   - "参数矩阵" Θ_s: 每个ScaleEncoder的Conv+Linear参数
#   │
#   └── C2F (Coarse-to-Fine Fusion): 粗到细融合
#       - "权重矩阵" W: [n_scales, pred_len]
#       - 训练时: softmax归一化 + KL散度一致性约束
#       - 推理时: 硬一致性检查 + 不一致尺度降权
#
# 【创新模块二】GRA-M (Global Retrieval-Augmented Memory) 全局检索增强记忆
#   ├── PVDR (Pattern-Value Dual Retriever): 模式-数值双重检索器
#   │   - 每个尺度独立记忆库 (key-value存储)
#   │   - 轻量统计编码器 (mean, std, max, min, trend)
#   │   - 极端数据检测 (3-sigma规则)
#   │   - 可学习per-scale阈值
#   │   - Top-K=5 加权检索
#   │
#   └── AKG (Adaptive Knowledge Gating): 自适应知识门控
#       - "联系矩阵" C: [n_scales, pred_len]
#         → sigmoid门控：预测 vs 历史检索
#       - "融合矩阵" F: [2*n_scales, pred_len]
#         → softmax融合：所有来源 → 最终预测
#
# 四类可训练矩阵:
#   1. 参数矩阵 Θ_s (每个尺度的ScaleEncoder参数, ~25K×4)
#   2. 权重矩阵 W  [5, 96] (C2F跨尺度融合权重, 480参数)
#   3. 联系矩阵 C  [5, 96] (AKG历史-预测门控, 480参数)
#   4. 融合矩阵 F [10, 96] (AKG最终融合权重, 960参数)
#   总新增参数: ~103K
#
# 架构流程:
#   Input → [Baseline Time-LLM → pred_s1]
#        → [DM: S2..S5 → pred_s2..s5]
#        → [C2F: 权重矩阵W融合 → weighted_pred]
#        → [PVDR: 多尺度检索 → hist_refs]
#        → [AKG: 联系矩阵C + 融合矩阵F → final_pred]
#        → Denorm → Output
#
# ============================================================


# ====== 手动修改区（路径/策略/显存）======

# 项目根目录
PROJECT_DIR="/mnt/e/timellm-colab-fuxian-chaungxin-bishe/Time-LLM"

# LLM 模型目录 (Qwen 2.5 3B)
LLM_PATH="${PROJECT_DIR}/base_models/Qwen2.5-3B"

# Checkpoint 保存目录
CHECKPOINTS="${PROJECT_DIR}/checkpoints"

# 日志保存目录
LOG_DIR="${PROJECT_DIR}/logs"

# 每 N step 保存一次（0=关闭）
# ETTh1: ~8500样本, batch_size=4时约2125步/epoch
SAVE_STEPS=0

# 只保留最近 N 个 step checkpoint（0=关闭）
SAVE_TOTAL_LIMIT=5

# 断点续训（第一次留空）
RESUME_FROM=""
# 断点续训示例:
# RESUME_FROM="${CHECKPOINTS}/long_term_forecast_ETTh1_512_96_.../checkpoint"

# 仅首次恢复时手动覆盖 EarlyStopping 计数（-1=不用）
RESUME_COUNTER=-1

# 随机种子
SEED=2021


# ====== LLM 模型参数 ======
# Qwen 2.5 3B 词向量维度
LLM_DIM=2048
# 使用的 LLM 层数
# 6GB显存: 6层 | 16GB: 12层 | 24GB+: 24层
LLM_LAYERS=6
# 是否启用 4-bit 量化
# 6GB显存: 必须启用 | 16GB+: 可选
LOAD_4BIT="--load_in_4bit"


# ====== 数据与预测参数 ======
# 输入序列长度（历史窗口）
SEQ_LEN=512
# 标签长度（解码器输入）
LABEL_LEN=48
# 预测长度
PRED_LEN=96
# 批次大小
# 6GB: 4 | 16GB: 16 | 24GB: 32 | 32GB: 48 | 64GB: 64
BATCH=4


# ====== 模型结构参数 ======
# Patch嵌入维度
# 6GB: 32 | 16GB+: 64
D_MODEL=32
# FFN隐藏层维度
# 6GB: 32 | 16GB+: 128
D_FF=32
# 注意力头数
N_HEADS=8
# 编码器层数
E_LAYERS=2
# 解码器层数
D_LAYERS=1


# ====== 训练参数 ======
# 训练轮数
TRAIN_EPOCHS=10
# 早停耐心
PATIENCE=10
# 学习率
LEARNING_RATE=0.0001
# Dropout比例
DROPOUT=0.1
# 并行加载线程数
NUM_WORKERS=2


# ====== v3 TAPR 模块参数 (DM + C2F) ======
# 是否启用 TAPR 模块
# 设为 1 启用, 0 禁用
USE_TAPR=1

# 总尺度数量 (包含baseline)
# 默认5: S1(baseline) + S2(ds=2) + S3(ds=4) + S4(ds=8) + S5(ds=16)
# 减少到3可降低计算量: S1 + S2 + S3
N_SCALES=5

# 各尺度下采样率 (逗号分隔)
DOWNSAMPLE_RATES="1,2,4,8,16"

# 一致性损失权重
# 控制KL散度一致性损失对总损失的贡献
# 过大: 强制各尺度预测相同，失去多尺度意义
# 过小: 各尺度预测可能严重矛盾
# 建议范围: 0.05 ~ 0.3
LAMBDA_CONSIST=0.1

# 一致性模式: soft | hard | hybrid (推荐hybrid)
# soft: 仅训练时KL约束
# hard: 仅推理时降权
# hybrid: 训练时soft + 推理时hard
CONSISTENCY_MODE="hybrid"

# 推理时不一致尺度的降权因子
# 0.5: 不一致尺度权重减半
DECAY_FACTOR=0.5


# ====== v3 GRAM 模块参数 (PVDR + AKG) ======
# 是否启用 GRA-M 模块
USE_GRAM=1

# 门控正则化损失权重
# 鼓励联系矩阵的门控做出明确决策 (偏向0或1)
# 过大: 强制门控极端化，可能损害灵活性
# 过小: 门控始终停留在0.5附近（无效）
LAMBDA_GATE=0.01

# 基础检索阈值 (sigmoid前的原始值)
# sigmoid(0.8) ≈ 0.69 → 余弦相似度需>0.69才触发
# 是nn.Parameter，训练中会自动调整
SIMILARITY_THRESHOLD=0.8

# 极端数据检测σ阈值 (3-sigma规则)
EXTREME_SIGMA=3.0

# 极端数据时阈值降低量
# 检测到极端数据时: threshold -= 0.2 → 更容易匹配历史
EXTREME_THRESHOLD_REDUCTION=0.2

# 检索的相似模式数量 (Top-K)
TOP_K=5

# 模式表示向量维度 (用于检索key编码)
D_REPR=64

# 是否构建检索记忆库
# 首次训练: 必须设为 1
# 断点续训: 必须设为 0 (记忆库已在checkpoint中)
BUILD_MEMORY=1


# ====== Warmup 参数 ======
# 辅助损失warmup步数
# 前500步: 辅助损失 × (step / 500)
# 确保模型先学好主任务
WARMUP_STEPS=500


# ====== 测试时机 ======
# "final": 仅最终epoch测试
# "all": 每个epoch都测试
# "5,10": 指定epoch测试
TEST_EPOCHS="final"


# ====== 消融实验快捷设置 ======
# 通过环境变量覆盖 (优先级高于上面的设置)
USE_TAPR=${USE_TAPR:-${USE_TAPR}}
USE_GRAM=${USE_GRAM:-${USE_GRAM}}


# ============================================================
# 不同显存下的参数推荐
# ============================================================
#
# ┌──────────┬─────────┬────────┬──────────┬────────┬─────────┬──────────┐
# │ 参数     │ 6GB     │ 16GB   │ 24GB     │ 32GB   │ 64GB    │ 说明     │
# ├──────────┼─────────┼────────┼──────────┼────────┼─────────┼──────────┤
# │ BATCH    │ 4       │ 16     │ 32       │ 48     │ 64-128  │ 批次大小 │
# │ LLM_LAYER│ 6       │ 12     │ 24       │ 32     │ 全部    │ LLM层数  │
# │ D_MODEL  │ 32      │ 64     │ 64       │ 64     │ 128     │ 嵌入维度 │
# │ D_FF     │ 32      │ 128    │ 128      │ 256    │ 256     │ FFN维度  │
# │ SEQ_LEN  │ 512     │ 512    │ 512      │ 720    │ 720     │ 输入长度 │
# │ LOAD_4BIT│ 必须    │ 可选   │ 不需要   │ 不需要 │ 不需要  │ 4bit量化 │
# │ N_SCALES │ 5       │ 5      │ 5        │ 5      │ 5       │ 尺度数   │
# │ TOP_K    │ 5       │ 5      │ 10       │ 10     │ 10      │ 检索数   │
# │ D_REPR   │ 64      │ 64     │ 128      │ 128    │ 256     │ 表示维度 │
# │ NUM_WORK │ 2       │ 4      │ 8        │ 8      │ 16      │ 数据线程 │
# └──────────┴─────────┴────────┴──────────┴────────┴─────────┴──────────┘
#
# 显存估算 (batch_size=4, Qwen 2.5 3B):
#   Qwen 2.5 3B (4-bit, 6层):  ~1.5 GB
#   Trainable params:          ~0.5 GB
#   DM (4 ScaleEncoders):      ~2 MB
#   C2F weight matrix:         < 1 KB
#   PVDR memory banks:         ~5 MB
#   AKG matrices:              < 1 KB
#   Intermediate tensors:      ~0.3 GB
#   System overhead:           ~1.0 GB
#   Total:                     ~5.3 GB ✅ (6GB GPU)
#
# ============================================================
# 各显存配置的详细命令示例
# ============================================================
#
# 【6GB GPU (GTX 1660 Ti / RTX 2060)】
#   BATCH=4 LLM_LAYERS=6 D_MODEL=32 D_FF=32 \
#   LOAD_4BIT="--load_in_4bit" NUM_WORKERS=2 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【16GB GPU (RTX 4060 Ti / V100 16GB / T4)】
#   BATCH=16 LLM_LAYERS=12 D_MODEL=64 D_FF=128 \
#   LOAD_4BIT="" NUM_WORKERS=4 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【24GB GPU (RTX 3090 / RTX 4090 / A5000)】
#   BATCH=32 LLM_LAYERS=24 D_MODEL=64 D_FF=128 \
#   LOAD_4BIT="" TOP_K=10 D_REPR=128 NUM_WORKERS=8 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【32GB GPU (V100 32GB / A100 40GB)】
#   BATCH=48 LLM_LAYERS=32 D_MODEL=64 D_FF=256 SEQ_LEN=720 \
#   LOAD_4BIT="" TOP_K=10 D_REPR=128 NUM_WORKERS=8 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【64GB GPU (A100 80GB / H100)】
#   BATCH=128 LLM_LAYERS=36 D_MODEL=128 D_FF=256 SEQ_LEN=720 \
#   LOAD_4BIT="" TOP_K=10 D_REPR=256 NUM_WORKERS=16 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# ============================================================


# ====== 断点续训可改/不可改说明 ======
# ✅ 可改（不影响模型形状）：
#   RESUME_FROM / SAVE_STEPS / SAVE_TOTAL_LIMIT / LOG_DIR / CHECKPOINTS
#   BATCH / NUM_WORKERS / LEARNING_RATE / TRAIN_EPOCHS / PATIENCE
#   LAMBDA_CONSIST / LAMBDA_GATE (辅助损失权重)
#   CONSISTENCY_MODE / DECAY_FACTOR (一致性模式和衰减因子)
#   WARMUP_STEPS / TEST_EPOCHS
#
# ❌ 不可改（会导致形状不匹配或语义不一致）：
#   LLM_PATH / LLM_DIM / LLM_LAYERS
#   SEQ_LEN / LABEL_LEN / PRED_LEN
#   D_MODEL / D_FF / N_HEADS / ENC_IN / DEC_IN / C_OUT
#   USE_TAPR / USE_GRAM / N_SCALES / D_REPR
#   DOWNSAMPLE_RATES / SIMILARITY_THRESHOLD (结构参数)
#
# ⚠️ 谨慎：
#   BUILD_MEMORY: 断点续训时 **必须设为 0**，避免重建记忆库
# ====================================


# ====== 执行训练 ======

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_v3_$(date +%Y%m%d_%H%M%S).log"

# 构建模块标志
TAPR_FLAG=""
GRAM_FLAG=""
MEMORY_FLAG=""
COMMENT="v3"

if [ "${USE_TAPR}" -eq 1 ]; then
    TAPR_FLAG="--use_tapr"
    COMMENT="${COMMENT}_TAPR"
fi

if [ "${USE_GRAM}" -eq 1 ]; then
    GRAM_FLAG="--use_gram"
    COMMENT="${COMMENT}_GRAM"
    if [ "${BUILD_MEMORY}" -eq 1 ]; then
        MEMORY_FLAG="--build_memory"
    fi
fi

echo "============================================================"
echo "Time-LLM Enhanced v3 训练开始"
echo "============================================================"
echo "TAPR模块 (DM+C2F): $([ "${USE_TAPR}" -eq 1 ] && echo '启用' || echo '禁用')"
if [ "${USE_TAPR}" -eq 1 ]; then
    echo "  - 尺度数量: ${N_SCALES}"
    echo "  - 下采样率: ${DOWNSAMPLE_RATES}"
    echo "  - 一致性损失权重: ${LAMBDA_CONSIST}"
    echo "  - 一致性模式: ${CONSISTENCY_MODE}"
    echo "  - 推理降权因子: ${DECAY_FACTOR}"
fi
echo "GRA-M模块 (PVDR+AKG): $([ "${USE_GRAM}" -eq 1 ] && echo '启用' || echo '禁用')"
if [ "${USE_GRAM}" -eq 1 ]; then
    echo "  - Top-K: ${TOP_K}"
    echo "  - 表示维度: ${D_REPR}"
    echo "  - 门控损失权重: ${LAMBDA_GATE}"
    echo "  - 检索阈值: ${SIMILARITY_THRESHOLD} (sigmoid=${SIMILARITY_THRESHOLD})"
    echo "  - 极端检测σ: ${EXTREME_SIGMA}"
    echo "  - 极端阈值降低: ${EXTREME_THRESHOLD_REDUCTION}"
    echo "  - 构建记忆库: $([ "${BUILD_MEMORY}" -eq 1 ] && echo '是' || echo '否')"
fi
echo "============================================================"
echo "LLM: Qwen 2.5 3B | Layers: ${LLM_LAYERS} | 4-bit: $([ -n "${LOAD_4BIT}" ] && echo '是' || echo '否')"
echo "Batch: ${BATCH} | Seq: ${SEQ_LEN} | Pred: ${PRED_LEN}"
echo "D_model: ${D_MODEL} | D_ff: ${D_FF} | LR: ${LEARNING_RATE}"
echo "Warmup: ${WARMUP_STEPS}步 | Patience: ${PATIENCE}"
echo "============================================================"

# 构建命令参数
CMD_ARGS=""

# TAPR参数
if [ "${USE_TAPR}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --use_tapr"
    CMD_ARGS="${CMD_ARGS} --n_scales ${N_SCALES}"
    CMD_ARGS="${CMD_ARGS} --downsample_rates ${DOWNSAMPLE_RATES}"
    CMD_ARGS="${CMD_ARGS} --lambda_consist ${LAMBDA_CONSIST}"
    CMD_ARGS="${CMD_ARGS} --consistency_mode ${CONSISTENCY_MODE}"
    CMD_ARGS="${CMD_ARGS} --decay_factor ${DECAY_FACTOR}"
fi

# GRA-M参数
if [ "${USE_GRAM}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --use_gram"
    CMD_ARGS="${CMD_ARGS} --lambda_gate ${LAMBDA_GATE}"
    CMD_ARGS="${CMD_ARGS} --similarity_threshold ${SIMILARITY_THRESHOLD}"
    CMD_ARGS="${CMD_ARGS} --extreme_sigma ${EXTREME_SIGMA}"
    CMD_ARGS="${CMD_ARGS} --extreme_threshold_reduction ${EXTREME_THRESHOLD_REDUCTION}"
    CMD_ARGS="${CMD_ARGS} --top_k ${TOP_K}"
    CMD_ARGS="${CMD_ARGS} --d_repr ${D_REPR}"
    if [ -n "${MEMORY_FLAG}" ]; then
        CMD_ARGS="${CMD_ARGS} ${MEMORY_FLAG}"
    fi
fi

# 断点续训参数
if [ -n "${RESUME_FROM}" ]; then
    CMD_ARGS="${CMD_ARGS} --resume_from_checkpoint ${RESUME_FROM}"
    CMD_ARGS="${CMD_ARGS} --resume_counter ${RESUME_COUNTER}"
fi

# 执行训练
python run_main_enhanced_v3.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
  --model_comment "${COMMENT}" \
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
  --llm_model QWEN \
  --llm_dim ${LLM_DIM} \
  --llm_model_path "${LLM_PATH}" \
  --llm_layers ${LLM_LAYERS} \
  ${LOAD_4BIT} \
  --num_workers ${NUM_WORKERS} \
  --seed ${SEED} \
  --prompt_domain 1 \
  --learning_rate ${LEARNING_RATE} \
  --train_epochs ${TRAIN_EPOCHS} \
  --patience ${PATIENCE} \
  --itr 1 \
  --dropout ${DROPOUT} \
  --checkpoints "${CHECKPOINTS}" \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --warmup_steps ${WARMUP_STEPS} \
  --test_epochs "${TEST_EPOCHS}" \
  ${CMD_ARGS} \
  2>&1 | tee -a "${LOG_FILE}"

echo "============================================================"
echo "训练完成，日志保存至: ${LOG_FILE}"
echo "============================================================"


# ============================================================
# 消融实验指南
# ============================================================
#
# 【E1】Baseline only (原始Time-LLM)
#   USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【E2】+TAPR only (DM + C2F)
#   USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【E3】+GRAM only (PVDR + AKG)
#   USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【E4】Full (DM + C2F + PVDR + AKG)
#   USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【参数敏感性实验】
# - n_scales: 修改 N_SCALES=3 (仅S1+S2+S3)
# - lambda_consist: 测试 0.05, 0.1, 0.3
# - top_k: 测试 3, 5, 10
# - lambda_gate: 测试 0.005, 0.01, 0.05
#
# ============================================================


# ============================================================
# OOM 处理建议 (显存不足时)
# ============================================================
#
# 优先级从高到低:
# 1) 降 BATCH (4 → 2)
# 2) 降 SEQ_LEN (512 → 384 → 256)
# 3) 降 LLM_LAYERS (6 → 4 → 2)
# 4) 降 D_FF (32 → 16)
# 5) 降 D_MODEL (32 → 16)
# 6) 降 N_SCALES (5 → 3)
# 7) 降 TOP_K (5 → 3)
# 8) 禁用 GRAM (USE_GRAM=0)
# 9) 启用 LOAD_4BIT="--load_in_4bit"
#
# ============================================================


# ============================================================
# 断点续训示例
# ============================================================
#
# 1. 从 step checkpoint 恢复:
#    RESUME_FROM="${CHECKPOINTS}/.../.../checkpoint_step_1000/checkpoint.pt"
#    BUILD_MEMORY=0  # 重要!
#
# 2. 从 EarlyStopping checkpoint 恢复:
#    RESUME_FROM="${CHECKPOINTS}/.../checkpoint"
#    BUILD_MEMORY=0  # 重要!
#
# 3. 可选重置EarlyStopping计数:
#    RESUME_COUNTER=0  # 重置为0
#    RESUME_COUNTER=-1  # 使用checkpoint中保存的值(默认)
#
# ============================================================
