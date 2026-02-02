# Time-LLM 云服务器复现指南 (Cloud Server Reproduction Guide)

> **目标**: 帮助复现者在云服务器上以最优性价比复现 Time-LLM 项目
> **适用场景**: 阿里云、腾讯云、AutoDL、Vast.ai 等 GPU 云服务

---

## 一、云服务器配置推荐

### 1.1 GPU 配置选择

| 配置等级 | GPU 型号 | 显存 | 参考价格 | 推荐场景 |
|---------|---------|------|---------|---------|
| **经济型** | RTX 3060/3070 | 8-12GB | ¥1-2/小时 | 学习验证、小规模实验 |
| **性价比型** ⭐推荐 | RTX 3090/4090 | 24GB | ¥3-5/小时 | 完整训练、论文复现 |
| **专业型** | A100 40GB | 40GB | ¥10-15/小时 | 大规模实验、多任务并行 |
| **入门型** | RTX 2080Ti | 11GB | ¥0.8-1.5/小时 | 预算有限的学习者 |

### 1.2 其他配置要求

| 配置项 | 最低要求 | 推荐配置 |
|-------|---------|---------|
| CPU | 4核 | 8核+ |
| 内存 | 16GB | 32GB |
| 磁盘 | 50GB SSD | 100GB SSD |
| 系统 | Ubuntu 20.04/22.04 | Ubuntu 22.04 |
| CUDA | 11.8+ | 12.1 |
| Python | 3.10+ | 3.11 |

---

## 二、云服务器平台推荐

### 2.1 国内平台

| 平台 | 特点 | 适用人群 | 链接 |
|------|------|---------|------|
| **AutoDL** ⭐ | 按秒计费，性价比最高 | 学生、个人研究者 | [autodl.com](https://autodl.com) |
| **阿里云 PAI** | 稳定可靠，企业级 | 企业用户 | [pai.console.aliyun.com](https://pai.console.aliyun.com) |
| **腾讯云 GPU** | 新用户优惠多 | 新手尝试 | [cloud.tencent.com](https://cloud.tencent.com) |
| **矩池云** | 专注深度学习 | 科研团队 | [matpool.com](https://matpool.com) |

### 2.2 国外平台

| 平台 | 特点 | 适用人群 | 链接 |
|------|------|---------|------|
| **Vast.ai** | 价格透明，资源丰富 | 预算有限 | [vast.ai](https://vast.ai) |
| **Lambda Labs** | 高性能 A100 | 专业研究 | [lambdalabs.com](https://lambdalabs.com) |
| **RunPod** | 灵活计费 | 短期使用 | [runpod.io](https://runpod.io) |

---

## 三、环境搭建步骤

### 3.1 基础环境配置 (推荐 AutoDL)

```bash
# ============================================================
# Step 1: 选择镜像 (AutoDL 为例)
# ============================================================
# 推荐镜像: PyTorch 2.1.0 + CUDA 12.1 + Python 3.10
# 或自定义安装

# Step 2: 连接服务器后，创建工作目录
mkdir -p ~/workspace && cd ~/workspace

# Step 3: 克隆项目
git clone https://github.com/KimMeen/Time-LLM.git
cd Time-LLM

# Step 4: 创建虚拟环境 (推荐)
conda create -n timellm python=3.11 -y
conda activate timellm
```

### 3.2 依赖安装

```bash
# ============================================================
# PyTorch 安装 (根据 CUDA 版本选择)
# ============================================================
# CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (备选)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# ============================================================
# 项目依赖安装
# ============================================================
pip install -r requirements.txt

# 额外依赖 (4-bit 量化支持)
pip install bitsandbytes accelerate

# 升级 transformers (Qwen2 支持)
pip install "transformers>=4.40.0" --upgrade
```

### 3.3 数据集下载

```bash
# ============================================================
# 方法 1: Google Drive 手动下载 (推荐)
# ============================================================
# 下载链接: https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view
# 解压到 ./dataset/

# ============================================================
# 方法 2: 使用 gdown 自动下载
# ============================================================
pip install gdown
gdown "https://drive.google.com/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP" -O datasets.zip
unzip datasets.zip -d ./dataset/
```

### 3.4 模型下载

```bash
# ============================================================
# 方案 A: 使用 Qwen 2.5 3B (推荐，性能更好)
# ============================================================
mkdir -p ./base_models/Qwen2.5-3B
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir ./base_models/Qwen2.5-3B \
  --local-dir-use-symlinks False

# ============================================================
# 方案 B: 使用 GPT-2 (备选，显存更省)
# ============================================================
python -c "
from transformers import GPT2Model, GPT2Tokenizer
GPT2Model.from_pretrained('openai-community/gpt2', cache_dir='./base_models/gpt2')
GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir='./base_models/gpt2')
"

# ============================================================
# 方案 C: 使用 Llama-7B (需要申请访问权限)
# ============================================================
huggingface-cli login  # 先登录
huggingface-cli download huggyllama/llama-7b --local-dir ./base_models/llama-7b
```

---

## 四、训练命令

### 4.1 快速验证 (RTX 3060 8GB / RTX 3070 8GB)

```bash
# 使用 GPT-2，显存友好
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model_comment GPT2_Cloud \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --batch_size 8 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 6 \
  --train_epochs 10 \
  --num_workers 4 \
  --prompt_domain 1 \
  --itr 1
```

### 4.2 性价比配置 (RTX 3090/4090 24GB) ⭐推荐

```bash
# 使用 Qwen 2.5 3B + 4-bit 量化
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model_comment Qwen3B_Cloud \
  --model TimeLLM \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size 16 \
  --llm_dim 2048 \
  --llm_layers 12 \
  --llm_model_path ./base_models/Qwen2.5-3B \
  --load_in_4bit \
  --train_epochs 50 \
  --num_workers 8 \
  --prompt_domain 1 \
  --itr 1
```

### 4.3 完整复现 (A100 40GB)

```bash
# 使用 Llama-7B 全精度，论文完整配置
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_512_96 \
  --model_comment Llama7B_Full \
  --model TimeLLM \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 24 \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --llm_layers 32 \
  --train_epochs 100 \
  --num_workers 10 \
  --prompt_domain 1 \
  --learning_rate 0.01 \
  --itr 1
```

---

## 五、不同配置的成本估算

### 5.1 时间与成本对比

| GPU 配置 | 数据集 | 训练时间 | 单价 | 总成本 |
|---------|-------|---------|------|-------|
| RTX 3060 8GB | ETTh1 (3 epochs) | ~4 小时 | ¥1.5/h | ¥6 |
| RTX 3090 24GB | ETTm1 (10 epochs) | ~8 小时 | ¥4/h | ¥32 |
| RTX 4090 24GB | ETTm1 (50 epochs) | ~15 小时 | ¥5/h | ¥75 |
| A100 40GB | ETTm1 (100 epochs) | ~10 小时 | ¥12/h | ¥120 |

### 5.2 推荐配置 (性价比最优)

**目标**: 完成论文复现 + 获得可发表的实验结果

| 阶段 | GPU | 时长 | 成本 | 说明 |
|------|-----|------|------|------|
| 环境调试 | RTX 3060 | 2小时 | ¥3 | 验证代码能否运行 |
| 快速训练 | RTX 3090 | 5小时 | ¥20 | ETTh1 完整训练 |
| 完整实验 | RTX 3090 | 20小时 | ¥80 | 所有 ETT 数据集 |
| **总计** | | **27小时** | **¥103** | |

---

## 六、训练脚本 (一键执行)

### 6.1 创建训练脚本

```bash
# 创建 train_cloud.sh
cat > train_cloud.sh << 'EOF'
#!/bin/bash
# ============================================================
# Time-LLM 云服务器训练脚本
# 适用: RTX 3090/4090 (24GB VRAM)
# ============================================================

# 激活环境
source ~/miniconda3/bin/activate timellm

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# 训练参数
MODEL_PATH="./base_models/Qwen2.5-3B"
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2")
PRED_LENS=(96 192 336 720)

# 循环训练
for DATA in "${DATASETS[@]}"; do
  for PRED_LEN in "${PRED_LENS[@]}"; do
    echo "=========================================="
    echo "Training: ${DATA} with pred_len=${PRED_LEN}"
    echo "=========================================="

    python run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ${DATA}.csv \
      --model_id ${DATA}_512_${PRED_LEN} \
      --model_comment Qwen3B_Cloud \
      --model TimeLLM \
      --data ${DATA} \
      --features M \
      --seq_len 512 \
      --label_len 48 \
      --pred_len ${PRED_LEN} \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 32 \
      --d_ff 32 \
      --batch_size 16 \
      --llm_dim 2048 \
      --llm_layers 6 \
      --llm_model_path ${MODEL_PATH} \
      --load_in_4bit \
      --train_epochs 20 \
      --num_workers 8 \
      --prompt_domain 1 \
      --itr 1
  done
done

echo "All training completed!"
EOF

chmod +x train_cloud.sh
```

### 6.2 后台运行

```bash
# 使用 nohup 后台运行
nohup bash train_cloud.sh > training.log 2>&1 &

# 查看日志
tail -f training.log

# 使用 screen (推荐)
screen -S timellm
bash train_cloud.sh
# Ctrl+A+D 分离，screen -r timellm 恢复
```

---

## 七、性能基准 (预期指标)

### 7.1 ETT 数据集预期 MSE

| 数据集 | pred_len=96 | pred_len=192 | pred_len=336 | pred_len=720 |
|-------|------------|--------------|--------------|--------------|
| ETTh1 | 0.375 | 0.421 | 0.461 | 0.490 |
| ETTh2 | 0.288 | 0.349 | 0.382 | 0.420 |
| ETTm1 | 0.302 | 0.350 | 0.390 | 0.448 |
| ETTm2 | 0.175 | 0.235 | 0.290 | 0.375 |

### 7.2 验证训练效果

```python
# 评估脚本 eval_results.py
import numpy as np
import torch
from utils.metrics import metric

# 加载预测结果
preds = np.load('./results/pred.npy')
trues = np.load('./results/true.npy')

# 计算指标
mae, mse, rmse, mape, mspe = metric(preds, trues)
print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
```

---

## 八、常见问题排查

### 8.1 CUDA Out of Memory

```bash
# 方案 1: 减小 batch_size
--batch_size 8  # 从 16 降到 8

# 方案 2: 减少 LLM 层数
--llm_layers 4  # 从 6 降到 4

# 方案 3: 缩短序列长度
--seq_len 256  # 从 512 降到 256

# 方案 4: 清理显存缓存
import torch
torch.cuda.empty_cache()
```

### 8.2 模型下载失败

```bash
# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 使用代理
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 离线下载后上传
# 在本地下载模型，打包上传到服务器
```

### 8.3 训练速度慢

```bash
# 增加 num_workers
--num_workers 8

# 使用混合精度
--use_amp

# 检查 CPU/GPU 利用率
watch -n 1 nvidia-smi
htop
```

---

## 九、省钱技巧

1. **竞价实例**: AutoDL 等平台的竞价实例价格可低至正常价格的 20-30%
2. **非高峰时段**: 凌晨/周末价格通常更低
3. **预付费套餐**: 长期使用考虑包月/包年
4. **中间保存**: 每个 epoch 后保存 checkpoint，避免中断损失
5. **数据预处理本地化**: 在本地完成数据预处理，上传后直接训练

---

## 十、完整复现清单

- [ ] 1. 选择云服务器平台和 GPU 配置
- [ ] 2. 创建实例并连接 SSH
- [ ] 3. 安装 Miniconda 和创建虚拟环境
- [ ] 4. 安装 PyTorch 和项目依赖
- [ ] 5. 下载数据集到 `./dataset/`
- [ ] 6. 下载基座模型到 `./base_models/`
- [ ] 7. 修改代码支持 4-bit 量化 (如需)
- [ ] 8. 运行快速验证脚本
- [ ] 9. 运行完整训练
- [ ] 10. 导出实验结果和 checkpoint

---

**文档更新时间**: 2026-01-02
**适用版本**: Time-LLM v1.0 + Qwen 2.5 3B
**维护者**: Claude Code
