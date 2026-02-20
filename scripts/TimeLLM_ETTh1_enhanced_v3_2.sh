#!/bin/bash
# ============================================================
# Time-LLM Enhanced v3 训练脚本 (GPT-2 + TAPR + GRA-M)
# ============================================================
#
# 本脚本基于 GPT-2 配置，集成 v3 创新模块参数。
# 与 v4 脚本的区别: v4 使用 Qwen 2.5 3B，本脚本使用 GPT-2。
#
# ┌────────────────────────────────────────────────────────────┐
# │            GPT-2 vs Qwen 2.5 3B 关键差异                  │
# ├──────────────┬──────────────┬──────────────────────────────┤
# │ 属性         │ GPT-2        │ Qwen 2.5 3B                  │
# ├──────────────┼──────────────┼──────────────────────────────┤
# │ 参数量       │ 124M         │ 3B (30亿)                    │
# │ 隐藏维度     │ 768          │ 2048                         │
# │ 总层数       │ 12           │ ~36                          │
# │ 词表大小     │ 50257        │ ~151K                        │
# │ 显存(FP32)   │ ~0.5 GB      │ ~12 GB (需4-bit→1.5GB)       │
# │ 4-bit量化    │ 不需要       │ 6GB显存必须                   │
# │ 加载方式     │ GPT2Model    │ AutoModel                    │
# │              │ (专用分支)   │ (通用分支)                    │
# └──────────────┴──────────────┴──────────────────────────────┘
#
# 加载逻辑 (TimeLLM_Enhanced_v3.py:155-214):
#   if llm_model_path 非空:       → AutoModel (Qwen路径)
#   elif llm_model == 'GPT2':     → GPT2Model (硬编码 ./base_models/gpt2)
#   elif llm_model == 'LLAMA':    → LlamaModel
#   elif llm_model == 'BERT':     → BertModel
#
# 因此使用 GPT-2 时:
#   1. LLM_PATH 必须为空 ""  (否则走 AutoModel 分支)
#   2. --llm_model 必须为 GPT2 (触发专用分支)
#   3. --llm_dim 必须为 768  (GPT-2 隐藏维度)
#   4. 不需要 --load_in_4bit (模型很小)
#
#
# 【创新模块一】TAPR (Trend-Aware Patch Router) 趋势感知尺度路由
#   ├── DM (Decomposable Multi-scale): 可分解多尺度独立预测
#   │   - 5个尺度 (S1=baseline, S2..S5=辅助)
#   │   - 下采样率: 1, 2, 4, 8, 16
#   │   - 信号变换: identity, diff, smooth, trend
#   │   - 每个尺度产生独立的 [B, pred_len, N] 预测
#   │   - "参数矩阵" Theta_s: 每个ScaleEncoder的Conv+Linear参数
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
#         → sigmoid门控: 预测 vs 历史检索
#       - "融合矩阵" F: [2*n_scales, pred_len]
#         → softmax融合: 所有来源 → 最终预测
#
# 四类可训练矩阵:
#   1. 参数矩阵 Theta_s (每个尺度的ScaleEncoder参数, ~25K*4)
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
#PROJECT_DIR="/mnt/e/timellm-colab-fuxian-chaungxin-bishe/Time-LLM"
PROJECT_DIR="/content/drive/MyDrive/T-L-GPT2-Enhanced"  # Colab

# ────────────────────────────────────────────────────────────
# LLM 模型路径 (GPT-2)
# ────────────────────────────────────────────────────────────
# TimeLLM_Enhanced_v3.py 加载优先级链:
#   (1) 若 llm_model_path 非空 → 走 AutoModel 通用分支
#   (2) 若 llm_model_path 为空 且 llm_model=='GPT2' → 走 GPT2Model 专用分支
#
# 两种方式均可加载 GPT-2:
#
# 【方式A】指定路径 (推荐，与 Qwen v4 脚本风格统一)
#   LLM_PATH="${PROJECT_DIR}/base_models/gpt2"
#   → 走 AutoModel 通用分支
#   → AutoConfig/AutoModel/AutoTokenizer 自动识别为 GPT-2
#   → 只要 LOAD_4BIT="" (不启用量化)，行为与专用分支完全一致
#
# 【方式B】留空 (走专用分支)
#   LLM_PATH=""
#   → 走 GPT2Model 专用分支
#   → 从硬编码路径 ./base_models/gpt2 加载
#   → 注意: 此路径是相对于项目根目录的，cd 到其他位置会找不到
#
# 两种方式的实际效果完全相同:
#   - 都调用 GPT2Config + GPT2Model + GPT2Tokenizer
#   - 都应用 num_hidden_layers = llm_layers
#   - 都冻结 LLM 参数
#
# 当前使用方式A (指定路径):
# ────────────────────────────────────────────────────────────
LLM_PATH="${PROJECT_DIR}/base_models/openai-community/gpt2"

# Checkpoint 保存目录
CHECKPOINTS="${PROJECT_DIR}/checkpoints_v3"

# 日志保存目录
LOG_DIR="${PROJECT_DIR}/logs"

# 每 N step 保存一次（0=关闭）
# ETTh1: ~8500样本, batch_size=16时约531步/epoch
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

# ────────────────────────────────────────────────────────────
# GPT-2 词向量维度
# ────────────────────────────────────────────────────────────
# 【关键差异 vs v4】768 (Qwen 是 2048)
#
# GPT-2 的 hidden_size = 768，此值决定:
#   - Reprogramming Layer 输出投影维度
#   - Prompt 嵌入维度
#   - FlattenHead 计算: head_nf = d_model * patch_num
#     (注: llm_dim 影响 ReprogrammingLayer 的 d_llm 参数，
#      而非直接影响 head_nf; 但 llm_dim 决定了 LLM 输出
#      到 FlattenHead 之间的投影层尺寸)
#
# 如果设错 (比如仍用 2048)，会导致:
#   RuntimeError: mat1 and mat2 shapes cannot be multiplied
# ────────────────────────────────────────────────────────────
LLM_DIM=768

# ────────────────────────────────────────────────────────────
# 使用的 LLM 层数
# ────────────────────────────────────────────────────────────
# GPT-2 总共 12 层 (Qwen 有 ~36 层)
# 可选范围: 1 ~ 12
#
# 推荐:
#   6GB显存: 6层 (与Qwen v4配置一致，便于对比实验)
#   任意显存: 6~12层均可 (GPT-2很小，层数对显存影响微弱)
#
# 注意: 如果之前用 Qwen 训练了 checkpoint，切换到 GPT-2 后
# 不能断点续训 (LLM 结构完全不同，权重不兼容)
# ────────────────────────────────────────────────────────────
LLM_LAYERS=12

# ────────────────────────────────────────────────────────────
# 4-bit 量化开关
# ────────────────────────────────────────────────────────────
# 【关键差异 vs v4】必须设为空 ""
#
# GPT-2 仅 124M 参数，FP32 下仅占 ~0.5 GB 显存
# 完全不需要量化，且 GPT-2 专用加载分支不支持
# BitsAndBytesConfig 量化配置
#
# Qwen v4 中此项为 "--load_in_4bit"，因为 Qwen 3B
# FP32 需 ~12 GB，4-bit 后降到 ~1.5 GB
# ────────────────────────────────────────────────────────────
LOAD_4BIT=""


# ====== 数据与预测参数 ======
# 输入序列长度（历史窗口）
SEQ_LEN=512
# 标签长度（解码器输入）
LABEL_LEN=48
# 预测长度
PRED_LEN=96

# ────────────────────────────────────────────────────────────
# 批次大小
# ────────────────────────────────────────────────────────────
# GPT-2 省出 ~1 GB 显存 (vs Qwen 4-bit)，可增大 batch
#
# 显存对比 (6GB GPU):
#   Qwen 4-bit: 模型~1.5GB → 剩余~3.5GB → BATCH=4
#   GPT-2 FP32: 模型~0.5GB → 剩余~4.5GB → BATCH=16~32
#
# 更大的 batch 通常让训练更稳定，梯度估计更准确
# 6GB显存推荐: 16 | 实在OOM则降回8
# ────────────────────────────────────────────────────────────
BATCH=32


# ====== 模型结构参数 ======

# ────────────────────────────────────────────────────────────
# Patch嵌入维度
# ────────────────────────────────────────────────────────────
# GPT-2 显存充裕，可从 32 提升到 64
# 这影响 ScaleEncoder 的特征容量
# v4 (Qwen, 6GB): d_model=32 (显存紧张)
# v5 (GPT-2, 6GB): d_model=64 (显存充裕)
# ────────────────────────────────────────────────────────────
D_MODEL=64

# ────────────────────────────────────────────────────────────
# FFN隐藏层维度
# ────────────────────────────────────────────────────────────
# 同理，GPT-2 显存充裕可提升
# v4 (Qwen, 6GB): d_ff=32
# v5 (GPT-2, 6GB): d_ff=128
# ────────────────────────────────────────────────────────────
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
# 早停耐心
PATIENCE=10
# 学习率
LEARNING_RATE=0.0001
# Dropout比例
DROPOUT=0.1

# ────────────────────────────────────────────────────────────
# 并行加载线程数
# ────────────────────────────────────────────────────────────
# batch 增大后，数据加载可能成为瓶颈
# 可适当增加 num_workers
# ────────────────────────────────────────────────────────────
NUM_WORKERS=4


# ====== v3 TAPR 模块参数 (DM + C2F) ======
# 这些参数与 LLM 选择无关，保持与 v4 一致

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
# 这些参数与 LLM 选择无关，保持与 v4 一致

# 是否启用 GRA-M 模块
USE_GRAM=1

# 门控正则化损失权重
# 鼓励联系矩阵的门控做出明确决策 (偏向0或1)
# 过大: 强制门控极端化，可能损害灵活性
# 过小: 门控始终停留在0.5附近（无效）
LAMBDA_GATE=0.01

# 基础检索阈值 (sigmoid前的原始值)
# sigmoid(0.8) = 0.69 → 余弦相似度需>0.69才触发
# 是nn.Parameter，训练中会自动调整
SIMILARITY_THRESHOLD=0.8

# 极端数据检测sigma阈值 (3-sigma规则)
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
# 前500步: 辅助损失 * (step / 500)
# 确保模型先学好主任务
WARMUP_STEPS=500


# ====== 数据划分策略 ======
# ────────────────────────────────────────────────────────────
# 原始 ETTh1 数据集时间分布:
#   训练集: 2016年7月 - 2017年6月 (12个月, ~8544条)
#   验证集: 2017年7月 - 2017年10月 (4个月, ~2880条)
#   测试集: 2017年11月 - 2018年2月 (4个月, ~2880条)
#
# v2 训练中观察到的问题:
#   Vali Loss ≈ 2 × Test Loss (验证集损失异常偏高)
#
# 可能原因:
#   1. 验证集时段 (7-10月) 包含极端季节性变化 (夏→秋)
#   2. 数据采集质量问题 (传感器漂移等)
#   3. 训练/验证分布偏移
#
# 解决策略:
#   --swap_val_test:        交换验证集和测试集
#   --discard_original_val: 丢弃原验证集，用 train+test 按比例重新划分
#   --data_split:           指定重新划分比例 "train:val:test"
#
# 三种策略对比:
#   ┌─────────────────────┬─────────────────────────────────────┐
#   │ 策略                │ 效果                                │
#   ├─────────────────────┼─────────────────────────────────────┤
#   │ 默认 (不修改)       │ 保持原始划分, vali可能偏高          │
#   │ swap_val_test=1     │ 交换验证/测试集, 用测试集做早停     │
#   │ discard_original_val│ 丢弃原验证集, 按比例重新三分        │
#   │  + data_split=6:2:2 │ 最均匀, 但改变了原论文划分方式      │
#   └─────────────────────┴─────────────────────────────────────┘
# ────────────────────────────────────────────────────────────

# 是否交换验证集和测试集
# 0=正常顺序, 1=交换 (用原测试集做早停验证)
SWAP_VAL_TEST=0

# 是否丢弃原验证集（使用新划分比例）
# 0=使用原始划分, 1=丢弃原验证集并重新划分
DISCARD_ORIGINAL_VAL=0

# 重新划分比例 (仅当 DISCARD_ORIGINAL_VAL=1 时生效)
# 格式: "train:val:test"
# "6:2:2" = 60%训练, 20%验证, 20%测试
DATA_SPLIT="6:2:2"


# ====== 测试时机 ======
# "final": 仅最终epoch测试
# "all":   每个epoch都测试
# "5,10":  指定epoch测试 (final会自动追加，无需手写)
# 注意: 不可混写数字和"final"，如 "4,5,final" 会降级为仅final
TEST_EPOCHS="2,4,5,8,9"


# ====== 消融实验快捷设置 ======
# 通过环境变量覆盖 (优先级高于上面的设置)
USE_TAPR=${USE_TAPR:-${USE_TAPR}}
USE_GRAM=${USE_GRAM:-${USE_GRAM}}


# ============================================================
# GPT-2 各显存下的参数推荐
# ============================================================
#
# GPT-2 (124M) 远小于 Qwen 2.5 3B，显存压力大幅降低
# 因此各档位的 BATCH / D_MODEL / D_FF 均可提升
#
# ┌──────────┬──────────┬────────┬──────────┬────────┬─────────┬──────────┐
# │ 参数     │ 6GB      │ 16GB   │ 24GB     │ 32GB   │ 64GB    │ 说明     │
# ├──────────┼──────────┼────────┼──────────┼────────┼─────────┼──────────┤
# │ BATCH    │ 16       │ 64     │ 128      │ 256    │ 512     │ 批次大小 │
# │ LLM_LAYER│ 6        │ 12     │ 12       │ 12     │ 12      │ LLM层数  │
# │ D_MODEL  │ 64       │ 64     │ 128      │ 128    │ 128     │ 嵌入维度 │
# │ D_FF     │ 128      │ 128    │ 256      │ 256    │ 256     │ FFN维度  │
# │ SEQ_LEN  │ 512      │ 512    │ 720      │ 720    │ 720     │ 输入长度 │
# │ LOAD_4BIT│ 不需要   │ 不需要 │ 不需要   │ 不需要 │ 不需要  │ 4bit量化 │
# │ N_SCALES │ 5        │ 5      │ 5        │ 5      │ 5       │ 尺度数   │
# │ TOP_K    │ 5        │ 5      │ 10       │ 10     │ 10      │ 检索数   │
# │ D_REPR   │ 64       │ 64     │ 128      │ 128    │ 256     │ 表示维度 │
# │ NUM_WORK │ 4        │ 8      │ 8        │ 16     │ 16      │ 数据线程 │
# └──────────┴──────────┴────────┴──────────┴────────┴─────────┴──────────┘
#
# 显存估算 (batch_size=16, GPT-2):
#   GPT-2 (FP32, 6层):         ~0.3 GB   ← (Qwen 4-bit: ~1.5 GB)
#   Trainable params:           ~0.3 GB   ← (Qwen: ~0.5 GB, 因映射层与llm_dim相关)
#   DM (4 ScaleEncoders):       ~2 MB
#   C2F weight matrix:          < 1 KB
#   PVDR memory banks:          ~5 MB
#   AKG matrices:               < 1 KB
#   Intermediate tensors:       ~0.8 GB   ← (batch更大，d_model更大)
#   System overhead:             ~1.0 GB
#   Total:                      ~2.4 GB   ← (Qwen v4: ~5.3 GB)
#
# 结论: GPT-2 在 6GB 显存上极为充裕 (仅用 ~40%)
#
# ============================================================
# 各显存配置的详细命令示例 (GPT-2)
# ============================================================
#
# 【6GB GPU (GTX 1660 Ti / RTX 2060)】
#   BATCH=16 LLM_LAYERS=6 D_MODEL=64 D_FF=128 \
#   NUM_WORKERS=4 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# 【16GB GPU (RTX 4060 Ti / V100 16GB / T4)】
#   BATCH=64 LLM_LAYERS=12 D_MODEL=64 D_FF=128 \
#   NUM_WORKERS=8 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# 【24GB GPU (RTX 3090 / RTX 4090 / A5000)】
#   BATCH=128 LLM_LAYERS=12 D_MODEL=128 D_FF=256 SEQ_LEN=720 \
#   TOP_K=10 D_REPR=128 NUM_WORKERS=8 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# 【32GB+ GPU (V100 32GB / A100)】
#   BATCH=256 LLM_LAYERS=12 D_MODEL=128 D_FF=256 SEQ_LEN=720 \
#   TOP_K=10 D_REPR=128 NUM_WORKERS=16 \
#   bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# ============================================================


# ====== v4 (Qwen) vs v5 (GPT-2) 参数差异总结 ======
#
# ┌─────────────┬───────────────────────────────┬──────────────────────┬──────────┐
# │ 参数        │ v4 (Qwen 2.5 3B)              │ v5 (GPT-2)           │ 必要性   │
# ├─────────────┼───────────────────────────────┼──────────────────────┼──────────┤
# │ LLM_PATH    │ "${PROJECT_DIR}/.../Qwen2.5-3B"│ "${PROJECT_DIR}/.../gpt2"│ 必须改│
# │ LLM_DIM     │ 2048                          │ 768                  │ 必须改   │
# │ LOAD_4BIT   │ "--load_in_4bit"              │ "" (不需要)          │ 必须改   │
# │ --llm_model │ QWEN                          │ GPT2                 │ 必须改   │
# │ LLM_LAYERS  │ 6 (最多~36)                   │ 6 (最多12)           │ 保持/可调│
# │ BATCH       │ 4                             │ 16                   │ 建议改   │
# │ D_MODEL     │ 32                            │ 64                   │ 建议改   │
# │ D_FF        │ 32                            │ 128                  │ 建议改   │
# │ NUM_WORKERS │ 2                             │ 4                    │ 建议改   │
# │ TAPR/GRAM   │ 不变                          │ 不变                 │ 无需改   │
# │ 数据参数    │ 不变                          │ 不变                 │ 无需改   │
# │ 训练参数    │ 不变                          │ 不变                 │ 无需改   │
# └─────────────┴───────────────────────────────┴──────────────────────┴──────────┘
#
# ============================================================


# ====== 断点续训可改/不可改说明 ======
# ✅ 可改（不影响模型形状）：
#   RESUME_FROM / SAVE_STEPS / SAVE_TOTAL_LIMIT / LOG_DIR / CHECKPOINTS
#   BATCH / NUM_WORKERS / LEARNING_RATE / TRAIN_EPOCHS / PATIENCE
#   LAMBDA_CONSIST / LAMBDA_GATE (辅助损失权重)
#   CONSISTENCY_MODE / DECAY_FACTOR (一致性模式和衰减因子)
#   WARMUP_STEPS / TEST_EPOCHS
#   SWAP_VAL_TEST / DISCARD_ORIGINAL_VAL / DATA_SPLIT (数据划分策略)
#
# ❌ 不可改（会导致形状不匹配或语义不一致）：
#   LLM_PATH / LLM_DIM / LLM_LAYERS / --llm_model
#   SEQ_LEN / LABEL_LEN / PRED_LEN
#   D_MODEL / D_FF / N_HEADS / ENC_IN / DEC_IN / C_OUT
#   USE_TAPR / USE_GRAM / N_SCALES / D_REPR
#   DOWNSAMPLE_RATES / SIMILARITY_THRESHOLD (结构参数)
#
# ⚠️ 谨慎：
#   BUILD_MEMORY: 断点续训时 **必须设为 0**，避免重建记忆库
#
# ⚠️ 重要: Qwen checkpoint 和 GPT-2 checkpoint 不兼容!
#   用 v4 (Qwen) 训练的 checkpoint 不能用 v5 (GPT-2) 续训，反之亦然。
#   LLM 结构、隐藏维度、词表大小完全不同，权重无法对齐。
# ====================================


# ====== 执行训练 ======

cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_v3_gpt2_$(date +%Y%m%d_%H%M%S).log"

# 构建模块标志
TAPR_FLAG=""
GRAM_FLAG=""
MEMORY_FLAG=""
COMMENT="v3_gpt2"

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
echo "Time-LLM Enhanced v3 训练开始 (GPT-2)"
echo "============================================================"
echo "LLM: GPT-2 (124M) | Layers: ${LLM_LAYERS}/12 | Dim: ${LLM_DIM}"
echo "4-bit量化: 否 (GPT-2不需要)"
echo "------------------------------------------------------------"
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
    echo "  - 检索阈值: ${SIMILARITY_THRESHOLD}"
    echo "  - 极端检测sigma: ${EXTREME_SIGMA}"
    echo "  - 极端阈值降低: ${EXTREME_THRESHOLD_REDUCTION}"
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

# 数据划分参数
if [ "${SWAP_VAL_TEST}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --swap_val_test"
fi
if [ "${DISCARD_ORIGINAL_VAL}" -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --discard_original_val"
    CMD_ARGS="${CMD_ARGS} --data_split ${DATA_SPLIT}"
fi

# ────────────────────────────────────────────────────────────
# 执行训练
# ────────────────────────────────────────────────────────────
# 与 v4 (Qwen) 脚本的 4 处关键差异已标注 [GPT-2]
# ────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────
# 训练命令中与 v4 (Qwen) 的差异说明:
#
# 第416行 --llm_model GPT2        ← [GPT-2] 原为 QWEN
#   触发 TimeLLM_Enhanced_v3.py 中的 GPT2Model 专用加载分支
#
# 第417行 --llm_dim ${LLM_DIM}    ← [GPT-2] LLM_DIM=768 (原2048)
#   GPT-2 隐藏维度 768，影响 ReprogrammingLayer 维度
#
# 第418行 --llm_model_path "${LLM_PATH}"  ← [GPT-2] 指向 base_models/gpt2
#   非空路径走 AutoModel 通用分支，自动识别为 GPT-2
#   (原 v4 指向 Qwen2.5-3B 路径)
#
# 第420行 ${LOAD_4BIT}            ← [GPT-2] 为空 (原 --load_in_4bit)
#   GPT-2 不需要量化，此变量展开为空字符串
# ────────────────────────────────────────────────────────────

echo "============================================================"
echo "训练完成，日志保存至: ${LOG_FILE}"
echo "============================================================"


# ============================================================
# 消融实验指南 (GPT-2 版)
# ============================================================
#
# 【E1】Baseline only (原始Time-LLM + GPT-2)
#   USE_TAPR=0 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# 【E2】+TAPR only (DM + C2F)
#   USE_TAPR=1 USE_GRAM=0 bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# 【E3】+GRAM only (PVDR + AKG)
#   USE_TAPR=0 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# 【E4】Full (DM + C2F + PVDR + AKG)
#   USE_TAPR=1 USE_GRAM=1 bash scripts/TimeLLM_ETTh1_enhanced_v5.sh
#
# 【参数敏感性实验】
# - n_scales: 修改 N_SCALES=3 (仅S1+S2+S3)
# - lambda_consist: 测试 0.05, 0.1, 0.3
# - top_k: 测试 3, 5, 10
# - lambda_gate: 测试 0.005, 0.01, 0.05
#
# 【LLM 对比实验】
# - GPT-2:     bash scripts/TimeLLM_ETTh1_enhanced_v5.sh  (本脚本)
# - Qwen 2.5:  bash scripts/TimeLLM_ETTh1_enhanced_v4.sh  (v4脚本)
# 对比相同 v3 模块在不同 LLM 骨干下的表现
#
# ============================================================


# ============================================================
# OOM 处理建议 (显存不足时)
# ============================================================
#
# GPT-2 模型本身极小 (~0.5GB)，OOM 多因 batch/序列过大
# 优先级从高到低:
# 1) 降 BATCH (16 → 8 → 4)
# 2) 降 SEQ_LEN (512 → 384 → 256)
# 3) 降 D_FF (128 → 64 → 32)
# 4) 降 D_MODEL (64 → 32)
# 5) 降 N_SCALES (5 → 3)
# 6) 降 TOP_K (5 → 3)
# 7) 禁用 GRAM (USE_GRAM=0)
# 8) 降 LLM_LAYERS (6 → 4 → 2)
#
# 注意: GPT-2 下不需要也不应该启用 LOAD_4BIT
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
# ⚠️ 不能跨 LLM 续训:
#    v4 (Qwen) 的 checkpoint 不能用 v5 (GPT-2) 加载，反之亦然
#    原因: LLM隐藏维度不同 (768 vs 2048)，
#          ReprogrammingLayer/MappingLayer 权重形状完全不同
#
# ============================================================
