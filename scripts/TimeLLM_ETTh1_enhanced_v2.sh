#!/bin/bash
# ============================================================
# Time-LLM Enhanced v2 训练脚本 (GPT-2 + TAPR_v2 + GRAM_v2)
# ============================================================
#
# 版本: 2.0 (观察者模式)
# 日期: 2026-02-10
#
# ============================================================
# 【重要更新】v2版本核心改进
# ============================================================
#
# v1问题诊断:
#   - Train Loss 恶化: 0.33 → 0.65 (+100%)
#   - 辅助损失过高: ~0.28，干扰主任务
#   - 模块直接修改 enc_out，破坏重编程机制
#
# v2解决方案 (观察者模式):
#   - 模块仅"观察"数据流，不修改 enc_out
#   - 辅助损失降低50倍: 0.5 → 0.01
#   - 检索阈值极高: 0.95，仅1-3%样本触发
#   - 趋势冲突阈值: 0.7，极少介入输出调整
#
# ============================================================
#
# 【创新模块架构】
#
# ┌─────────────────────────────────────────────────────────┐
# │  TAPR_v2 (Trend-Aware Patch Router) 趋势感知尺度路由     │
# │                                                          │
# │  ├── DM (Decomposable Multi-scale) 可分解多尺度混合      │
# │  │   - 多尺度下采样: kernel_sizes = [1, 4, 16]          │
# │  │   - 分组卷积特征提取 (减少参数量)                     │
# │  │   - 移动平均趋势提取                                  │
# │  │   - 输出: trend_embed [B*N, d_model]                 │
# │  │   【关键】不修改 enc_out，仅观察提取趋势嵌入           │
# │  │                                                       │
# │  └── C2F (Coarse-to-Fine Head) 先粗后细预测头            │
# │      - Level 1: 方向分类 (上涨/平稳/下跌)                │
# │      - Level 2: 幅度分类 (强涨/弱涨/弱跌/强跌)           │
# │      - 层次化一致性损失 (HCL)                            │
# │      【关键】辅助任务，不影响主预测                       │
# │                                                          │
# │  冲突感知融合 (Conflict-Aware Fusion):                   │
# │      - 冲突阈值 > 0.7 时才进行输出微调                   │
# │      - 调整幅度初始化为 0.01 (极小)                      │
# └─────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────┐
# │  GRAM_v2 (Global Retrieval-Augmented Memory) 检索增强    │
# │                                                          │
# │  ├── PVDR (Pattern-Value Dual Retriever) 严格检索器      │
# │  │   - 轻量级模式编码: 5维统计特征 (mean,std,max,min,trend)│
# │  │   - 余弦相似度度量                                    │
# │  │   - 【关键】相似度阈值 > 0.95 才触发检索              │
# │  │   - 预估触发率: 仅 1-3% 样本                          │
# │  │                                                       │
# │  └── AKG (Adaptive Knowledge Gating) 保守门控            │
# │      - 门控偏置初始化为 2.0 → sigmoid(2) ≈ 0.88          │
# │      - 默认 88% 信任当前模型推理                         │
# │      - 仅极度可靠时才使用历史信息                        │
# └─────────────────────────────────────────────────────────┘
#
# 数据流 (v2 观察者模式):
#
#   enc_out ───────────────────────────→ Reprogramming → LLM
#       │                                      ↑
#       ├→ [TAPR观察] → trend_embed ──────────┘ (可选prompt token)
#       │       │
#       │       └→ DM: 多尺度特征提取 (不修改enc_out!)
#       │       └→ C2F: 趋势方向分类 (辅助任务)
#       │
#       └→ [GRAM观察] → context_embed ────────┘ (极少触发)
#               │
#               └→ PVDR: 相似度>0.95才检索 (绝大多数跳过)
#               └→ AKG: 默认88%信任当前模型
#
# ============================================================


# ====== 手动修改区（路径/策略/显存）======

# 项目根目录 (请根据实际环境修改)
# PROJECT_DIR="/mnt/e/timellm-colab-fuxian-chaungxin-bishe/Time-LLM"  # 本地
PROJECT_DIR="/content/drive/MyDrive/T-L-GPT2-Enhanced"  # Colab

# GPT-2 模型目录
LLM_PATH="${PROJECT_DIR}/base_models/openai-community/gpt2"

# Checkpoint 保存目录
CHECKPOINTS="${PROJECT_DIR}/checkpoints"

# 日志保存目录
LOG_DIR="${PROJECT_DIR}/logs"

# 每 N step 保存一次（0=关闭）
# ETTh1约8500*2样本，batch_size=32时约1757步/epoch
SAVE_STEPS=3512

# 只保留最近 N 个 step checkpoint（0=关闭）
SAVE_TOTAL_LIMIT=10


# ====== 断点续训配置 ======

# 断点续训（第一次训练留空）
RESUME_FROM=""
# 断点续训示例:
# RESUME_FROM="${CHECKPOINTS}/long_term_forecast_ETTh1_512_96_TimeLLM_ETTh1_ftM_sl512_ll48_pl96_dm64_nh8_el2_dl1_df128_fc3_ebtimeF_test_0-GPT2_Enhanced_v2/checkpoint"

# 仅首次恢复时手动覆盖 EarlyStopping 计数（-1=不用）
RESUME_COUNTER=-1

# 随机种子与确定性
SEED=2021
# 1=开启确定性（更慢但结果可复现）；0=关闭
DETERMINISTIC=0


# ====== GPT-2 模型参数（固定）======

# GPT-2 词向量维度
LLM_DIM=768
# 使用的 GPT-2 层数
# - 6GB显存: 建议 6 层
# - 15GB显存: 可用 12 层
LLM_LAYERS=12


# ====== 数据与预测参数 ======

# 输入序列长度（历史窗口）
SEQ_LEN=512
# 标签长度（解码器输入）
LABEL_LEN=48
# 预测长度
PRED_LEN=96
# 批次大小
# - 6GB显存: 建议 8-16
# - 15GB显存: 可用 32
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
PATIENCE=15
# 学习率
LEARNING_RATE=0.0005
# Dropout比例
DROPOUT=0.15


# ============================================================
# 【新增】v2版本数据划分参数
# ============================================================
#
# 原始ETT数据集时间分布:
#   训练集: 2016年7月 - 2017年6月 (12个月)
#   验证集: 2017年7月 - 2017年10月 (4个月) ← 可能存在分布偏移
#   测试集: 2017年11月 - 2018年2月 (4个月)
#
# 观察到的问题:
#   Vali Loss ≈ 2 × Test Loss (验证集损失异常高)
#
# 可能原因:
#   1. 验证集时段 (7-10月) 包含极端季节性变化
#   2. 数据采集质量问题
#   3. 分布偏移
#
# 解决策略:
#   --swap_val_test: 交换验证集和测试集
#   --discard_original_val: 丢弃原验证集，使用data_split重新划分
#
# ============================================================

# 是否交换验证集和测试集
# 0=正常顺序, 1=交换
SWAP_VAL_TEST=0

# 是否丢弃原验证集（使用新划分比例）
# 0=使用原始划分, 1=丢弃原验证集并重新划分
DISCARD_ORIGINAL_VAL=1

# 重新划分比例 (仅当 DISCARD_ORIGINAL_VAL=1 时生效)
# 格式: "train:val:test"
DATA_SPLIT="6:2:2"

# 测试时机控制
# "final": 仅最后一个epoch测试
# "all": 每个epoch都测试
# "5,10,15": 在指定epoch测试 (会自动包含最后一个epoch)
TEST_EPOCHS="2,4,5"


# ============================================================
# 【新增】TAPR_v2 模块参数 (DM + C2F)
# ============================================================
#
# DM (Decomposable Multi-scale) 可分解多尺度:
#   - 仅观察输入序列，提取多尺度趋势嵌入
#   - 不修改 enc_out，保持重编程机制完整性
#
# C2F (Coarse-to-Fine) 先粗后细:
#   - Level 1: 方向分类 (上涨/平稳/下跌)
#   - Level 2: 幅度分类 (强涨/弱涨/弱跌/强跌)
#   - 辅助任务，仅作正则化
#
# ============================================================

# 是否启用 TAPR_v2 模块
USE_TAPR=1

# DM 多尺度数量
# 更多尺度捕获更丰富时间模式，但增加计算量
# 默认3: kernel_sizes = [1, 4, 16]
N_SCALES=3

# C2F 趋势辅助损失权重
# 【v2关键改进】从 0.5 降到 0.01，仅作正则化
# 不再干扰主任务优化
LAMBDA_TREND=0.01

# 是否应用趋势融合 (冲突感知输出调整)
# 0=禁用 (推荐), 1=启用
# 仅当检测到趋势冲突 > 0.7 时才微调输出
APPLY_TREND_FUSION=0


# ============================================================
# 【新增】GRAM_v2 模块参数 (PVDR + AKG)
# ============================================================
#
# PVDR (Pattern-Value Dual Retriever) 严格检索器:
#   - 相似度阈值 > 0.95 才触发检索
#   - 绝大多数样本完全跳过，实现"透明旁路"
#   - 预估仅 1-3% 样本触发检索
#
# AKG (Adaptive Knowledge Gating) 保守门控:
#   - 门控初始化偏置 = 2.0 → sigmoid(2) ≈ 0.88
#   - 默认 88% 信任当前模型推理
#   - 仅极度可靠的历史模式才会被使用
#
# ============================================================

# 是否启用 GRAM_v2 模块
USE_GRAM=1

# PVDR Top-K 检索数量
TOP_K=3

# PVDR 相似度阈值
# 【v2关键改进】极高阈值，仅 1-3% 样本触发
SIMILARITY_THRESHOLD=0.95

# AKG 检索辅助损失权重
# 【v2关键改进】从 0.3 降到 0.001
LAMBDA_RETRIEVAL=0.001

# 是否构建检索记忆库
# 首次训练时启用 (1)，断点续训时禁用 (0)
BUILD_MEMORY=1

# 模式表示向量维度
D_REPR=64


# ============================================================
# 断点续训可改/不可改说明
# ============================================================
#
# ✅ 可改（不影响模型形状）：
#   RESUME_FROM / SAVE_STEPS / SAVE_TOTAL_LIMIT / LOG_DIR / CHECKPOINTS
#   BATCH / NUM_WORKERS / LEARNING_RATE / TRAIN_EPOCHS / PATIENCE
#   LAMBDA_TREND / LAMBDA_RETRIEVAL (辅助损失权重)
#   TEST_EPOCHS (测试时机)
#   SWAP_VAL_TEST / DISCARD_ORIGINAL_VAL (数据划分策略)
#
# ❌ 不可改（会导致形状不匹配）：
#   LLM_PATH / LLM_DIM / LLM_LAYERS
#   SEQ_LEN / LABEL_LEN / PRED_LEN / PATCH_LEN / STRIDE
#   D_MODEL / D_FF / N_HEADS / ENC_IN / DEC_IN / C_OUT
#   N_SCALES / TOP_K / D_REPR / SIMILARITY_THRESHOLD
#
# ⚠️ 谨慎：
#   USE_TAPR / USE_GRAM: 模块开关一旦训练开始不可更改
#   BUILD_MEMORY: 断点续训时应设为 0，避免重建记忆库
#
# ============================================================


# ====== 执行训练 ======

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_enhanced_v2_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "Time-LLM Enhanced v2 训练开始 (观察者模式)"
echo "============================================================"
echo ""
echo "【v2核心改进】"
echo "  - 模块设计: 观察者模式，不修改 enc_out"
echo "  - 辅助损失: 极低权重 (0.01)，仅作正则化"
echo "  - 检索触发: 相似度 > ${SIMILARITY_THRESHOLD} (极高阈值)"
echo ""
echo "【TAPR_v2 (DM + C2F)】"
echo "  - 启用状态: $( [ "${USE_TAPR}" -eq 1 ] && echo '✓ 启用' || echo '✗ 禁用' )"
if [ "${USE_TAPR}" -eq 1 ]; then
echo "  - DM 尺度数量: ${N_SCALES} (kernel_sizes: [1, 4, 16])"
echo "  - C2F 损失权重: ${LAMBDA_TREND} (极低，仅正则化)"
echo "  - 趋势融合: $( [ "${APPLY_TREND_FUSION}" -eq 1 ] && echo '启用 (冲突>0.7时微调)' || echo '禁用 (推荐)' )"
fi
echo ""
echo "【GRAM_v2 (PVDR + AKG)】"
echo "  - 启用状态: $( [ "${USE_GRAM}" -eq 1 ] && echo '✓ 启用' || echo '✗ 禁用' )"
if [ "${USE_GRAM}" -eq 1 ]; then
echo "  - PVDR Top-K: ${TOP_K}"
echo "  - PVDR 相似度阈值: ${SIMILARITY_THRESHOLD} (极高，~1-3%触发)"
echo "  - AKG 损失权重: ${LAMBDA_RETRIEVAL} (极低)"
echo "  - 构建记忆库: $( [ "${BUILD_MEMORY}" -eq 1 ] && echo '是' || echo '否' )"
fi
echo ""
echo "【数据划分策略】"
echo "  - 交换验证/测试集: $( [ "${SWAP_VAL_TEST}" -eq 1 ] && echo '是' || echo '否' )"
echo "  - 丢弃原验证集: $( [ "${DISCARD_ORIGINAL_VAL}" -eq 1 ] && echo "是 (重划分为 ${DATA_SPLIT})" || echo '否' )"
echo "  - 测试时机: ${TEST_EPOCHS}"
echo ""
echo "【训练参数】"
echo "  - 学习率: ${LEARNING_RATE}"
echo "  - Dropout: ${DROPOUT}"
echo "  - Patience: ${PATIENCE}"
echo "  - 批次大小: ${BATCH}"
echo "============================================================"

# 构建命令参数
CMD_ARGS=""

# TAPR_v2 参数
if [ "${USE_TAPR}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --use_tapr"
  CMD_ARGS="${CMD_ARGS} --n_scales ${N_SCALES}"
  CMD_ARGS="${CMD_ARGS} --lambda_trend ${LAMBDA_TREND}"
  if [ "${APPLY_TREND_FUSION}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --apply_trend_fusion"
  fi
fi

# GRAM_v2 参数
if [ "${USE_GRAM}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --use_gram"
  CMD_ARGS="${CMD_ARGS} --top_k ${TOP_K}"
  CMD_ARGS="${CMD_ARGS} --similarity_threshold ${SIMILARITY_THRESHOLD}"
  CMD_ARGS="${CMD_ARGS} --lambda_retrieval ${LAMBDA_RETRIEVAL}"
  CMD_ARGS="${CMD_ARGS} --d_repr ${D_REPR}"
  if [ "${BUILD_MEMORY}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --build_memory"
  fi
fi

# 数据划分参数
if [ "${SWAP_VAL_TEST}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --swap_val_test"
fi

if [ "${DISCARD_ORIGINAL_VAL}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --discard_original_val"
  CMD_ARGS="${CMD_ARGS} --data_split ${DATA_SPLIT}"
fi

CMD_ARGS="${CMD_ARGS} --test_epochs ${TEST_EPOCHS}"

# 断点续训参数
if [ -n "${RESUME_FROM}" ]; then
  CMD_ARGS="${CMD_ARGS} --resume_from_checkpoint ${RESUME_FROM}"
  CMD_ARGS="${CMD_ARGS} --resume_counter ${RESUME_COUNTER}"
fi

# 确定性训练
if [ "${DETERMINISTIC}" -eq 1 ]; then
  CMD_ARGS="${CMD_ARGS} --deterministic"
fi

# 执行训练 (使用 run_main_enhanced_v2.py)
python run_main_enhanced_v2.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${SEQ_LEN}_${PRED_LEN} \
  --model_comment GPT2_Enhanced_v2 \
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
# 【实验矩阵】
#
# | 实验   | TAPR | GRAM | 预期效果                     |
# |--------|------|------|------------------------------|
# | Base   | ✗    | ✗    | 原始Time-LLM性能 (基准)      |
# | Exp-1  | ✓    | ✗    | 验证多尺度观察是否有益       |
# | Exp-2  | ✗    | ✓    | 验证检索增强是否有益         |
# | Exp-3  | ✓    | ✓    | 完整增强模型                 |
#
# 【子模块消融】
#
# TAPR子模块:
#   - 仅DM: 观察多尺度特征，无趋势分类
#   - 仅C2F: 趋势分类，无多尺度
#   - Full: DM + C2F
#
# GRAM子模块:
#   - 仅PVDR: 检索，固定权重
#   - 仅AKG: 门控，无检索 (无意义)
#   - Full: PVDR + AKG
#
# 【参数敏感性实验】
#
# DM:
#   - N_SCALES: 2, 3, 4 (默认3)
#
# C2F:
#   - LAMBDA_TREND: 0.005, 0.01, 0.02 (默认0.01)
#
# PVDR:
#   - SIMILARITY_THRESHOLD: 0.90, 0.95, 0.98 (默认0.95)
#   - TOP_K: 3, 5, 10 (默认3)
#
# AKG:
#   - LAMBDA_RETRIEVAL: 0.0005, 0.001, 0.002 (默认0.001)
#
# 【预期结果】
#
# | 实验   | 预期MSE变化 | 说明                         |
# |--------|-------------|------------------------------|
# | Base   | 基准        | 原版性能                     |
# | Exp-1  | ±0.5%       | 趋势嵌入可能略有帮助         |
# | Exp-2  | ±0.5%       | 极少触发，几乎无影响         |
# | Exp-3  | -1~2%       | 综合效果                     |
#
# 核心验证目标: Exp-1/2/3 不应显著低于 Baseline
#
# ============================================================


# ============================================================
# OOM 处理建议 (显存不足时)
# ============================================================
#
# 优先级从高到低:
# 1) 降 BATCH (32->16->8->4)
# 2) 降 SEQ_LEN (512->384->256->128)
# 3) 降 LLM_LAYERS (12->6->4->2)
# 4) 降 D_FF (128->64->32)
# 5) 降 D_MODEL (64->32->16)
# 6) 降 N_SCALES (3->2)
# 7) 禁用 GRAM (BUILD_MEMORY 消耗额外显存)
#
# 6GB 显存推荐配置:
#   BATCH=8, SEQ_LEN=256, LLM_LAYERS=4
#   D_MODEL=32, D_FF=64, N_SCALES=2
#
# 15GB 显存推荐配置 (Colab T4):
#   BATCH=32, SEQ_LEN=512, LLM_LAYERS=12
#   D_MODEL=64, D_FF=128, N_SCALES=3
#
# ============================================================


# ============================================================
# v1 vs v2 对比速查
# ============================================================
#
# | 设计维度       | v1版本              | v2版本 (观察者模式)   |
# |----------------|---------------------|-----------------------|
# | 数据流修改     | 直接修改 enc_out    | 仅观察，不修改        |
# | TAPR辅助损失   | lambda=0.5          | lambda=0.01 (降低50倍)|
# | GRAM辅助损失   | lambda=0.3          | lambda=0.001 (降低300倍)|
# | 检索触发条件   | 所有样本            | 相似度>0.95 (~1-3%)   |
# | 趋势融合策略   | 强制融合            | 冲突>0.7时才微调      |
# | 门控初始化     | 随机                | 偏置=2.0 (88%信任模型)|
# | 预期Train Loss | 0.65 (恶化100%)     | ~0.35 (接近原版)      |
#
# ============================================================
