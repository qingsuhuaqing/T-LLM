#!/bin/bash
# ============================================================
# Time-LLM Enhanced v4 训练脚本 (GPT-2 + TAPR v4 + GRAM v4)
# ============================================================
#
# 本脚本基于 GPT-2 配置，集成 v4 创新模块参数。
# v4 相对于 v3 的核心升级:
#   - DM: 平均池化下采样 → 统一k=4多相分解 (数据100%保留)
#   - C2F: 简单加权融合 → 专家投票(MAD) + DWT趋势约束
#   - PVDR: 3-sigma规则 → 逻辑回归极端分类 + 贝叶斯变点检测
#   - AKG: 固定sigmoid门控 → PVDR信号驱动的阈值自适应门控
#
# ┌────────────────────────────────────────────────────────────┐
# │            v3 vs v4 模块差异对照                            │
# ├────────────┬─────────────────────┬─────────────────────────┤
# │ 模块       │ v3                  │ v4                      │
# ├────────────┼─────────────────────┼─────────────────────────┤
# │ DM下采样   │ avg_pool(k不同)     │ 多相x[:, p::k, :](统一k)│
# │ DM尺度区分 │ 不同下采样率2/4/8/16│ 统一k=4+信号变换区分    │
# │ DM前向次数 │ 4次(每尺度1次)      │ 14次(2+4+4+4分支)       │
# │ C2F融合    │ softmax(W)直接加权  │ Stage1 MAD + Stage2 DWT │
# │ PVDR检测   │ ExtremeDetector     │ LogisticExtreme+BayesCP  │
# │ PVDR检索   │ 统一Top-K           │ 4模式(直接/极端/转折/正常)│
# │ AKG门控    │ sigmoid(C)          │ sigmoid(C)×adj(pvdr)    │
# │ 新增损失   │ 无                  │ L_expert(修正率)         │
# └────────────┴─────────────────────┴─────────────────────────┘
#
# 加载逻辑 (TimeLLM_Enhanced_v4.py):
#   if llm_model_path 非空:       → AutoModel (Qwen路径)
#   elif llm_model == 'GPT2':     → GPT2Model (硬编码 ./base_models/gpt2)
#   elif llm_model == 'LLAMA':    → LlamaModel
#   elif llm_model == 'BERT':     → BertModel
#
# 因此使用 GPT-2 时:
#   1. --llm_model 必须为 GPT2
#   2. --llm_dim 必须为 768 (GPT-2隐藏维度)
#   3. 不需要 --load_in_4bit (模型很小)
#
#
# 【创新模块一】TAPR v4 (Trend-Aware Patch Router)
#   ├── DM v4 (PolyphaseMultiScale): 多相交错多尺度预测
#   │   - 5个尺度 (S1=baseline, S2..S5=辅助)
#   │   - 多相分支: S2(k=2,2分支) S3(k=4,4分支) S4(k=4,4分支) S5(k=4,4分支)
#   │   - 信号变换: diff, identity, smooth, trend (区分频率聚焦)
#   │   - 关键: 统一k=4保证每分支≥16 patches, ScaleEncoder输入充足
#   │   - 总14个分支, 编码器分支内共享(仅4个ScaleEncoder)
#   │
#   └── C2F v4 (ExpertVotingFusion): 专家投票+趋势约束
#       ├── Stage1: 尺度内MAD投票
#       │   - 中位数偏差(MAD)检测异常分支
#       │   - S2(k=2)跳过MAD用简单平均(2分支MAD退化)
#       │   - S3/S4/S5各4分支投票→1个consolidated预测
#       │   - 追踪修正率(correction_rate)
#       │
#       └── Stage2: 跨尺度DWT趋势约束
#           - Haar小波分解每尺度预测
#           - S5趋势近似为基准方向
#           - cosine_similarity < -0.3 → 降权(趋势相反)
#           - correction_rate > 0.3 → 降权(不可靠尺度)
#           - "权重矩阵" W [5, pred_len]
#
# 【创新模块二】GRAM v4 (Global Retrieval-Augmented Memory)
#   ├── PVDR v4 (EnhancedDualRetriever): 增强检测+双重检索
#   │   ├── LogisticExtremeClassifier: 可学习极端分类(6参数)
#   │   ├── BayesianChangePointDetector: 贝叶斯变点检测(3参数)
#   │   ├── EnhancedMultiScaleMemoryBank: key+value+label
#   │   └── 4模式检索: direct/extreme/turning_point/normal
#   │
#   └── AKG v4 (ThresholdAdaptiveGating): 阈值自适应门控
#       ├── base_gate = sigmoid(C) [5, 96]
#       ├── effective_gate = base × (1-extreme_adj) × (1-cp_adj) × conf_adj
#       ├── 3个可学习阈值: τ_extreme, τ_cp, τ_conf
#       └── "融合矩阵" F [10, pred_len]
#
# 五类可训练矩阵/参数:
#   1. 参数矩阵 Θ_s (4个ScaleEncoder, 分支共享, ~100K)
#   2. 权重矩阵 W [5, 96] (C2F跨尺度融合, 480参数)
#   3. 联系矩阵 C [5, 96] (AKG基础门控, 480参数)
#   4. 融合矩阵 F [10, 96] (AKG最终融合, 960参数)
#   5. 阈值参数 τ (AKG 3个 + LogisticRegression 6个 + BayesianCP 3个 = 12参数)
#   总新增参数: ~103K + 12 ≈ v3同量级
#
# 损失函数:
#   L_total = L_main + warmup × (λ_consist × L_consist + λ_gate × L_gate + λ_expert × L_expert)
#   - L_consist: DWT小波趋势一致性 (v4替换v3的KL散度)
#   - L_gate: 门控决断性正则化 (与v3相同)
#   - L_expert: 专家修正率损失 (v4新增)
#
# ============================================================


# ====== 手动修改区（路径/策略/显存）======

# 项目根目录
#PROJECT_DIR="/mnt/e/timellm-colab-fuxian-chaungxin-bishe/Time-LLM"
PROJECT_DIR="/content/drive/MyDrive/T-L-GPT2-Enhanced"  # Colab

# ────────────────────────────────────────────────────────────
# LLM 模型路径 (GPT-2)
# ────────────────────────────────────────────────────────────
LLM_PATH="${PROJECT_DIR}/base_models/openai-community/gpt2"

# Checkpoint 保存目录
CHECKPOINTS="${PROJECT_DIR}/checkpoints_v4"

# 日志保存目录
LOG_DIR="${PROJECT_DIR}/logs"

# 每 N step 保存一次（0=关闭）
SAVE_STEPS=1756

# 只保留最近 N 个 step checkpoint（0=关闭）
SAVE_TOTAL_LIMIT=5

# 断点续训（第一次留空）
RESUME_FROM=""

# 仅首次恢复时手动覆盖 EarlyStopping 计数（-1=不用）
RESUME_COUNTER=-1

# 随机种子
SEED=2021


# ====== LLM 模型参数 ======

# GPT-2 词向量维度
LLM_DIM=768

# 使用的 LLM 层数 (GPT-2 总共12层)
LLM_LAYERS=12

# 4-bit 量化开关 (GPT-2不需要量化)
LOAD_4BIT=""


# ====== 数据与预测参数 ======
# 输入序列长度（历史窗口）
SEQ_LEN=512
# 标签长度（解码器输入）
LABEL_LEN=48
# 预测长度
PRED_LEN=96

# 批次大小 (GPT-2显存充裕, 可用较大batch)
BATCH=32


# ====== 模型结构参数 ======

# Patch嵌入维度 (GPT-2显存充裕)
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
TRAIN_EPOCHS=50
PATIENCE=10
LEARNING_RATE=0.0001
DROPOUT=0.1
NUM_WORKERS=4


# ====== v4 TAPR 模块参数 (DM Polyphase + C2F Voting) ======

# 是否启用 TAPR 模块
USE_TAPR=1

# 总尺度数量 (包含baseline)
# S1(baseline) + S2(k=2,diff) + S3(k=4,id) + S4(k=4,smooth) + S5(k=4,trend)
N_SCALES=5

# 各尺度下采样率 (用于PVDR内存库构建和检索时的输入下采样)
# v4关键: 必须与DM多相分解的k值对齐
#   S1=1(baseline), S2=2(k=2), S3=4(k=4), S4=4(k=4), S5=4(k=4)
# 若仍用v3的"1,2,4,8,16", S4/S5的PVDR检索粒度(T/8, T/16)
# 与DM预测粒度(T/4)不匹配, AKG融合时信号分辨率错位
DOWNSAMPLE_RATES="1,2,4,4,4"

# ────────────────────────────────────────────────────────────
# 一致性损失权重
# v4: 使用DWT小波趋势一致性(替代v3的KL散度)
# 含义: 各尺度趋势方向应与最粗尺度(S5)一致
# ────────────────────────────────────────────────────────────
LAMBDA_CONSIST=0.1

# 推理时不一致尺度的降权因子
DECAY_FACTOR=0.5

# ────────────────────────────────────────────────────────────
# v4 新增: MAD异常检测倍数
# MAD (Median Absolute Deviation) 用于尺度内专家投票
# 分支偏差 > MAD_THRESHOLD × MAD 的分支被判定为异常
# 过小: 过多分支被剔除, 信息损失
# 过大: 异常分支不被检测, 影响融合质量
# ────────────────────────────────────────────────────────────
MAD_THRESHOLD=3.0

# ────────────────────────────────────────────────────────────
# v4 新增: 尺度不可靠判定阈值
# 某尺度的累计修正率(被剔除分支数/总分支数) > 此阈值时
# 该尺度被标记为"不可靠", 在C2F Stage2中额外降权
# ────────────────────────────────────────────────────────────
RELIABILITY_THRESHOLD=0.3

# ────────────────────────────────────────────────────────────
# v4 新增: 不可靠尺度降权因子
# 被标记为不可靠的尺度, 其权重 × 此因子
# 0.3 = 权重降为原来的30%
# ────────────────────────────────────────────────────────────
RELIABILITY_PENALTY=0.3

# ────────────────────────────────────────────────────────────
# v4 新增: 专家修正损失权重
# L_expert = mean(correction_rate_s) 跨辅助尺度
# 惩罚修正率高的尺度, 鼓励各分支预测趋于一致
# ────────────────────────────────────────────────────────────
LAMBDA_EXPERT=0.05


# ====== v4 GRAM 模块参数 (PVDR Enhanced + AKG Threshold) ======

# 是否启用 GRAM 模块
USE_GRAM=1

# 门控正则化损失权重 (与v3相同)
LAMBDA_GATE=0.01

# 基础检索阈值
SIMILARITY_THRESHOLD=0.8

# 极端数据时阈值降低量
EXTREME_THRESHOLD_REDUCTION=0.2

# 检索的相似模式数量 (Top-K)
TOP_K=5

# 模式表示向量维度
D_REPR=64

# 是否构建检索记忆库
# 首次训练: 必须设为 1
# 断点续训: 必须设为 0 (记忆库已在checkpoint中)
BUILD_MEMORY=1

# ────────────────────────────────────────────────────────────
# v4 新增: 贝叶斯变点检测器最小段长
# 递归二分时, 段长 < CP_MIN_SEGMENT 则停止
# 过小: 检测过于敏感, 噪声被误判为变点
# 过大: 漏检短期结构变化
# ────────────────────────────────────────────────────────────
CP_MIN_SEGMENT=16

# ────────────────────────────────────────────────────────────
# v4 新增: 贝叶斯变点检测器最大递归深度
# 控制递归二分的最大层数
# 4 = 最多检测 2^4-1=15 个变点 (实际远少于此)
# ────────────────────────────────────────────────────────────
CP_MAX_DEPTH=4


# ====== Warmup 参数 ======
WARMUP_STEPS=500


# ====== 数据划分策略 ======
SWAP_VAL_TEST=0
DISCARD_ORIGINAL_VAL=0
DATA_SPLIT="6:2:2"


# ====== 测试时机 ======
TEST_EPOCHS="2,4,5,8,9,10,11,13,15"


# ====== 消融实验快捷设置 ======
USE_TAPR=${USE_TAPR:-${USE_TAPR}}
USE_GRAM=${USE_GRAM:-${USE_GRAM}}


# ============================================================
# v4 显存分析 (14分支多相, GPT-2)
# ============================================================
#
# v4相对v3的额外显存:
#   - 14个分支预测中间张量: 14×[B,96,7]×4byte ≈ ~150KB
#   - DWT分解结果: 5×[B,48,7]×4byte ≈ ~40KB
#   - 逻辑回归/变点检测: < 1KB
#   - 记忆库标签: 5000×5×4byte = ~100KB
#   - 总新增: ~0.3MB (可忽略)
#
# GPT-2 总显存估算 (batch=32):
#   GPT-2 (FP32, 12层):        ~0.5 GB
#   Trainable params:           ~0.3 GB
#   DM v4 (4 ScaleEncoders):   ~2 MB
#   C2F v4 (weight matrix+DWT): < 1 KB
#   PVDR v4 (memory+detectors): ~5.3 MB
#   AKG v4 (matrices+thresholds): < 1 KB
#   Intermediate tensors:       ~0.8 GB
#   System overhead:            ~1.0 GB
#   Total:                      ~2.6 GB
#
# 结论: GPT-2 + v4 在 6GB 显存上完全充裕
#
# ============================================================


# ====== 执行训练 ======

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_v4_gpt2_$(date +%Y%m%d_%H%M%S).log"

# 构建模块标志
TAPR_FLAG=""
GRAM_FLAG=""
MEMORY_FLAG=""
COMMENT="v4_gpt2"

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
echo "Time-LLM Enhanced v4 训练开始 (GPT-2)"
echo "============================================================"
echo "LLM: GPT-2 (124M) | Layers: ${LLM_LAYERS}/12 | Dim: ${LLM_DIM}"
echo "4-bit量化: 否 (GPT-2不需要)"
echo "------------------------------------------------------------"
echo "TAPR v4 (DM Polyphase + C2F Voting): $([ "${USE_TAPR}" -eq 1 ] && echo '启用' || echo '禁用')"
if [ "${USE_TAPR}" -eq 1 ]; then
    echo "  - 尺度数量: ${N_SCALES}"
    echo "  - 下采样率: ${DOWNSAMPLE_RATES}"
    echo "  - 一致性损失权重(DWT): ${LAMBDA_CONSIST}"
    echo "  - 推理降权因子: ${DECAY_FACTOR}"
    echo "  - MAD异常检测倍数: ${MAD_THRESHOLD}"
    echo "  - 不可靠修正率阈值: ${RELIABILITY_THRESHOLD}"
    echo "  - 不可靠降权因子: ${RELIABILITY_PENALTY}"
    echo "  - 专家修正损失权重: ${LAMBDA_EXPERT}"
fi
echo "GRAM v4 (PVDR Enhanced + AKG Threshold): $([ "${USE_GRAM}" -eq 1 ] && echo '启用' || echo '禁用')"
if [ "${USE_GRAM}" -eq 1 ]; then
    echo "  - Top-K: ${TOP_K}"
    echo "  - 表示维度: ${D_REPR}"
    echo "  - 门控损失权重: ${LAMBDA_GATE}"
    echo "  - 检索阈值: ${SIMILARITY_THRESHOLD}"
    echo "  - 极端阈值降低: ${EXTREME_THRESHOLD_REDUCTION}"
    echo "  - 变点最小段长: ${CP_MIN_SEGMENT}"
    echo "  - 变点最大递归深度: ${CP_MAX_DEPTH}"
    echo "  - 构建记忆库: $([ "${BUILD_MEMORY}" -eq 1 ] && echo '是' || echo '否')"
fi
echo "------------------------------------------------------------"
echo "数据划分策略:"
echo "  - 交换验证/测试集: $([ "${SWAP_VAL_TEST}" -eq 1 ] && echo '是' || echo '否')"
echo "  - 丢弃原验证集: $([ "${DISCARD_ORIGINAL_VAL}" -eq 1 ] && echo "是 (重划分为 ${DATA_SPLIT})" || echo '否 (原始划分)')"
echo "  - 测试时机: ${TEST_EPOCHS}"
echo "------------------------------------------------------------"
echo "Batch: ${BATCH} | Seq: ${SEQ_LEN} | Pred: ${PRED_LEN}"
echo "D_model: ${D_MODEL} | D_ff: ${D_FF} | LR: ${LEARNING_RATE}"
echo "Warmup: ${WARMUP_STEPS}步 | Patience: ${PATIENCE}"
echo "============================================================"

# 构建命令参数
CMD_ARGS=""

# TAPR v4 参数
if [ "${USE_TAPR}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --use_tapr"
    CMD_ARGS="${CMD_ARGS} --n_scales ${N_SCALES}"
    CMD_ARGS="${CMD_ARGS} --downsample_rates ${DOWNSAMPLE_RATES}"
    CMD_ARGS="${CMD_ARGS} --lambda_consist ${LAMBDA_CONSIST}"
    CMD_ARGS="${CMD_ARGS} --decay_factor ${DECAY_FACTOR}"
    # v4 新增参数
    CMD_ARGS="${CMD_ARGS} --mad_threshold ${MAD_THRESHOLD}"
    CMD_ARGS="${CMD_ARGS} --reliability_threshold ${RELIABILITY_THRESHOLD}"
    CMD_ARGS="${CMD_ARGS} --reliability_penalty ${RELIABILITY_PENALTY}"
    CMD_ARGS="${CMD_ARGS} --lambda_expert ${LAMBDA_EXPERT}"
fi

# GRAM v4 参数
if [ "${USE_GRAM}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --use_gram"
    CMD_ARGS="${CMD_ARGS} --lambda_gate ${LAMBDA_GATE}"
    CMD_ARGS="${CMD_ARGS} --similarity_threshold ${SIMILARITY_THRESHOLD}"
    CMD_ARGS="${CMD_ARGS} --extreme_threshold_reduction ${EXTREME_THRESHOLD_REDUCTION}"
    CMD_ARGS="${CMD_ARGS} --top_k ${TOP_K}"
    CMD_ARGS="${CMD_ARGS} --d_repr ${D_REPR}"
    # v4 新增参数
    CMD_ARGS="${CMD_ARGS} --cp_min_segment ${CP_MIN_SEGMENT}"
    CMD_ARGS="${CMD_ARGS} --cp_max_depth ${CP_MAX_DEPTH}"
    if [ -n "${MEMORY_FLAG}" ]; then
        CMD_ARGS="${CMD_ARGS} ${MEMORY_FLAG}"
    fi
fi

# 断点续训参数
if [ -n "${RESUME_FROM}" ]; then
    CMD_ARGS="${CMD_ARGS} --resume_from_checkpoint ${RESUME_FROM}"
    CMD_ARGS="${CMD_ARGS} --resume_counter ${RESUME_COUNTER}"
fi

# 数据划分参数
if [ "${SWAP_VAL_TEST}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --swap_val_test"
fi
if [ "${DISCARD_ORIGINAL_VAL}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --discard_original_val"
    CMD_ARGS="${CMD_ARGS} --data_split ${DATA_SPLIT}"
fi

# ────────────────────────────────────────────────────────────
# 执行训练 (v4: 使用 run_main_enhanced_v4.py)
# ────────────────────────────────────────────────────────────
python run_main_enhanced_v4.py \
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
  --llm_model GPT2 \
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
# 消融实验指南 (v4, GPT-2)
# ============================================================
#
# 【E1】Baseline only (原始Time-LLM + GPT-2)
#   USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【E2】+TAPR v4 only (DM Polyphase + C2F Voting)
#   USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【E3】+GRAM v4 only (PVDR Enhanced + AKG Threshold)
#   USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【E4】Full v4 (DM Polyphase + C2F Voting + PVDR Enhanced + AKG Threshold)
#   USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【E5】DM v4 (polyphase) vs DM v3 (avg_pool)
#   分别用 v4 和 v3 脚本, 仅启用TAPR, 对比多相采样增益
#   USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#   USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v3_2.sh
#
# 【E6】C2F v4 (voting+trend) vs C2F v3 (直接加权)
#   同E5, 对比投票+趋势约束 vs 简单加权的增益
#
# 【E7】PVDR v4 (logistic+CP) vs PVDR v3 (3-sigma)
#   分别用 v4 和 v3 脚本, 仅启用GRAM, 对比增强检测增益
#   USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#   USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v3_2.sh
#
# 【E8】AKG v4 (threshold) vs AKG v3 (fixed sigmoid)
#   同E7, 对比阈值门控 vs 固定门控的增益
#
# 【E9】L_expert enabled vs disabled
#   LAMBDA_EXPERT=0 USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#   LAMBDA_EXPERT=0.05 USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v4.sh
#
# 【参数敏感性实验】
# - MAD_THRESHOLD: 测试 2.0, 3.0, 4.0
# - RELIABILITY_THRESHOLD: 测试 0.2, 0.3, 0.5
# - LAMBDA_EXPERT: 测试 0.01, 0.05, 0.1
# - CP_MIN_SEGMENT: 测试 8, 16, 32
# - CP_MAX_DEPTH: 测试 2, 4, 6
#
# ============================================================


# ============================================================
# 断点续训说明
# ============================================================
#
# ✅ 可改（不影响模型形状）：
#   RESUME_FROM / SAVE_STEPS / SAVE_TOTAL_LIMIT / LOG_DIR / CHECKPOINTS
#   BATCH / NUM_WORKERS / LEARNING_RATE / TRAIN_EPOCHS / PATIENCE
#   LAMBDA_CONSIST / LAMBDA_GATE / LAMBDA_EXPERT (损失权重)
#   DECAY_FACTOR / MAD_THRESHOLD / RELIABILITY_THRESHOLD / RELIABILITY_PENALTY
#   WARMUP_STEPS / TEST_EPOCHS
#   SWAP_VAL_TEST / DISCARD_ORIGINAL_VAL / DATA_SPLIT
#
# ❌ 不可改（会导致形状不匹配）：
#   LLM_PATH / LLM_DIM / LLM_LAYERS / --llm_model
#   SEQ_LEN / LABEL_LEN / PRED_LEN
#   D_MODEL / D_FF / N_HEADS / ENC_IN / DEC_IN / C_OUT
#   USE_TAPR / USE_GRAM / N_SCALES / D_REPR
#   DOWNSAMPLE_RATES / SIMILARITY_THRESHOLD
#   CP_MIN_SEGMENT / CP_MAX_DEPTH (影响BayesianCP结构)
#
# ⚠️ 谨慎：
#   BUILD_MEMORY: 断点续训时 **必须设为 0**
#
# ⚠️ 重要: v3 checkpoint 和 v4 checkpoint 不兼容!
#   v3→v4: DM输出格式变更, C2F接口变更, PVDR新增检测器
#   不同版本之间不能跨版本续训
#
# ============================================================


# ============================================================
# OOM 处理建议
# ============================================================
#
# GPT-2 模型极小(~0.5GB), OOM 多因 batch/序列过大
# 优先级从高到低:
# 1) 降 BATCH (32 → 16 → 8 → 4)
# 2) 降 SEQ_LEN (512 → 384 → 256)
# 3) 降 D_FF (128 → 64 → 32)
# 4) 降 D_MODEL (64 → 32)
# 5) 降 N_SCALES (5 → 3)  — 注意: v4中S3/S4/S5统一k=4
# 6) 降 TOP_K (5 → 3)
# 7) 禁用 GRAM (USE_GRAM=0)
# 8) 降 LLM_LAYERS (12 → 6 → 4)
#
# ============================================================
