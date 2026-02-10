"""
GRAM_v2 (Global Retrieval-Augmented Memory) - 全局检索增强记忆网络 (极简版)

核心设计原则 (学术裁缝思想):
1. 极苛刻触发条件: 仅当历史模式相似度极高时才介入
2. 透明旁路: 不满足条件时完全不影响主流程
3. 最小权重: 即使触发，影响也被严格控制

改进要点:
- 相似度阈值设置为极高值 (>0.95)
- 检索结果仅作为参考嵌入，不直接修改输出
- 门控机制默认偏向"信任当前推理"

修改日志 (2026-02-10):
- 重新设计触发机制，相似度<0.95时完全跳过
- 简化记忆库结构，减少存储开销
- 门控权重初始化为0.95 (默认信任模型推理)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LightweightPatternEncoder(nn.Module):
    """
    轻量级模式编码器

    设计目标:
    - 快速编码输入序列为检索键
    - 参数量极小，不增加训练负担
    - 使用简单统计特征 + 轻量投影
    """

    def __init__(self, d_model, d_repr=64):
        super(LightweightPatternEncoder, self).__init__()

        self.d_model = d_model
        self.d_repr = d_repr

        # 统计特征维度: 5 (mean, std, max, min, trend)
        self.stat_dim = 5

        # 轻量投影
        self.proj = nn.Sequential(
            nn.Linear(self.stat_dim, d_repr),
            nn.LayerNorm(d_repr)
        )

    def forward(self, x):
        """
        Args:
            x: [B, L, D] 或 [B, L]

        Returns:
            key: [B, d_repr]
        """
        if x.dim() == 3:
            # 取均值降到 [B, L]
            x_flat = x.mean(dim=-1)
        else:
            x_flat = x

        B = x_flat.shape[0]
        device = x_flat.device

        # 提取统计特征
        mean_val = x_flat.mean(dim=1, keepdim=True)
        std_val = x_flat.std(dim=1, keepdim=True)
        max_val = x_flat.max(dim=1, keepdim=True)[0]
        min_val = x_flat.min(dim=1, keepdim=True)[0]
        trend_val = (x_flat[:, -1:] - x_flat[:, :1])  # 首尾差

        stats = torch.cat([mean_val, std_val, max_val, min_val, trend_val], dim=1)  # [B, 5]

        # 投影到表示空间
        key = self.proj(stats)

        return key


class StrictRetriever(nn.Module):
    """
    严格检索器

    核心特点:
    - 极高的相似度阈值 (默认0.95)
    - 不满足阈值时返回None，完全跳过检索
    - 使用余弦相似度而非欧氏距离
    """

    def __init__(self, d_repr=64, top_k=3, similarity_threshold=0.95):
        """
        Args:
            d_repr: 表示维度
            top_k: 检索数量
            similarity_threshold: 相似度阈值 (极高!)
        """
        super(StrictRetriever, self).__init__()

        self.d_repr = d_repr
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # 模式编码器
        self.encoder = LightweightPatternEncoder(d_model=d_repr, d_repr=d_repr)

        # 记忆库 (使用空tensor初始化，避免checkpoint加载问题)
        self.register_buffer('memory_keys', torch.zeros(0, d_repr))
        self.register_buffer('memory_values', torch.zeros(0, 1))  # 存储标签的简单表示
        self.register_buffer('memory_size', torch.tensor(0))

        self.max_memory_size = 5000

    def build_memory(self, train_loader, device, max_samples=5000):
        """
        从训练数据构建记忆库

        Args:
            train_loader: 训练数据加载器
            device: 设备
            max_samples: 最大样本数
        """
        all_keys = []
        all_values = []
        count = 0

        with torch.no_grad():
            for batch_x, batch_y, _, _ in train_loader:
                if count >= max_samples:
                    break

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                B, T, C = batch_x.shape

                # 对每个变量独立编码
                for c in range(C):
                    x_c = batch_x[:, :, c]  # [B, T]
                    key = self.encoder(x_c)  # [B, d_repr]

                    # 存储标签的简单表示 (末尾值)
                    y_c = batch_y[:, -1, c:c+1]  # [B, 1]

                    all_keys.append(key.cpu())
                    all_values.append(y_c.cpu())
                    count += B

                if count >= max_samples:
                    break

        if all_keys:
            self.memory_keys = torch.cat(all_keys, dim=0)[:max_samples].to(device)
            self.memory_values = torch.cat(all_values, dim=0)[:max_samples].to(device)
            self.memory_size = torch.tensor(self.memory_keys.shape[0])

    def retrieve(self, query):
        """
        严格检索 - 仅在高相似度时返回结果

        Args:
            query: [B, L, D] 输入序列

        Returns:
            retrieved: 检索结果 (可能为None)
            similarity: 相似度分数 (可能为None)
            is_valid: 是否有效检索
        """
        # 检查记忆库是否为空
        if self.memory_keys.shape[0] == 0:
            return None, None, False

        # 编码查询
        query_key = self.encoder(query)  # [B, d_repr]

        # 计算余弦相似度
        query_norm = F.normalize(query_key, dim=-1)  # [B, d_repr]
        memory_norm = F.normalize(self.memory_keys, dim=-1)  # [M, d_repr]

        similarity = torch.mm(query_norm, memory_norm.t())  # [B, M]

        # 获取最高相似度
        max_sim, max_idx = similarity.max(dim=-1)  # [B]

        # 严格阈值检查
        valid_mask = max_sim > self.similarity_threshold  # [B]

        # 如果没有任何样本超过阈值，返回None
        if not valid_mask.any():
            return None, None, False

        # 获取TopK结果
        top_k = min(self.top_k, self.memory_keys.shape[0])
        top_sim, top_idx = similarity.topk(k=top_k, dim=-1)  # [B, K]

        # 检索值
        retrieved_values = self.memory_values[top_idx]  # [B, K, 1]

        return retrieved_values, top_sim, True


class ConservativeGating(nn.Module):
    """
    保守门控模块

    核心设计:
    - 默认偏向信任模型推理 (gate初始化为高值)
    - 仅在检索结果极度可靠时才使用历史信息
    """

    def __init__(self, d_model):
        super(ConservativeGating, self).__init__()

        self.d_model = d_model

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4),  # +1 for similarity score
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # 初始化偏置，使门控默认输出高值 (信任模型)
        self._init_conservative()

    def _init_conservative(self):
        """
        初始化为保守模式 (默认信任模型)
        """
        # 最后一层偏置设为正值，使sigmoid输出偏大
        with torch.no_grad():
            self.gate[-2].bias.fill_(2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, current_feat, retrieved_values, similarity):
        """
        Args:
            current_feat: 当前特征 [B, D]
            retrieved_values: 检索值 [B, K, 1]
            similarity: 相似度 [B, K]

        Returns:
            gate_weight: 门控权重 [B, 1]，越高越信任当前模型
        """
        B = current_feat.shape[0]

        # 使用最高相似度
        max_sim = similarity.max(dim=-1, keepdim=True)[0]  # [B, 1]

        # 计算门控
        gate_input = torch.cat([current_feat, max_sim], dim=-1)
        gate_weight = self.gate(gate_input)  # [B, 1]

        return gate_weight


class GRAM_v2(nn.Module):
    """
    GRAM_v2 (Global Retrieval-Augmented Memory) - 极简版全局检索增强

    核心设计原则:
    1. 极苛刻触发: 相似度>0.95才启用检索
    2. 最小影响: 门控默认偏向信任模型推理
    3. 完全透明: 不满足条件时对主流程完全无影响

    与原GRAM的关键区别:
    - 原版: 所有样本都经过检索增强
    - 新版: 仅极少数高相似度样本才触发检索
    """

    def __init__(self, configs):
        super(GRAM_v2, self).__init__()

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.d_repr = getattr(configs, 'd_repr', 64)
        self.pred_len = configs.pred_len
        self.top_k = getattr(configs, 'top_k', 3)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 极高的相似度阈值
        self.similarity_threshold = getattr(configs, 'similarity_threshold', 0.95)

        # 极低的辅助损失权重
        self.default_lambda_retrieval = 0.001

        # 严格检索器
        self.retriever = StrictRetriever(
            d_repr=self.d_repr,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )

        # 保守门控
        self.gating = ConservativeGating(d_model=self.d_model)

        # 检索上下文投影 (仅用于prompt增强，不影响主流程)
        self.context_proj = nn.Linear(self.d_repr, self.d_model)

        # 检索启用标志
        self.retrieval_enabled = True

    def build_memory_from_dataloader(self, dataloader, model, device, max_samples=5000):
        """
        从数据加载器构建记忆库
        """
        self.retriever.build_memory(dataloader, device, max_samples)

    def forward(self, x, return_retrieval_info=False):
        """
        前向传播 - 仅观察，极少数情况下才返回检索信息

        核心原则: 不修改输入x

        Args:
            x: [B*N, L, d_model]
            return_retrieval_info: 是否返回检索信息

        Returns:
            x: 原样返回 (不修改!)
            context_embed: 检索上下文嵌入 (可能为零向量)
            retrieval_info: 检索信息 (可选)
        """
        device = x.device
        B = x.shape[0]

        # 默认返回零向量
        context_embed = torch.zeros(B, self.d_model, device=device)

        # 检查是否启用检索
        if not self.retrieval_enabled or self.retriever.memory_keys.shape[0] == 0:
            if return_retrieval_info:
                return x, context_embed, {
                    'gate_weights': torch.ones(B, 1, device=device),
                    'retrieval_triggered': False
                }
            return x, context_embed, None

        # 尝试检索
        retrieved, similarity, is_valid = self.retriever.retrieve(x)

        if not is_valid:
            # 未触发检索，返回原样
            if return_retrieval_info:
                return x, context_embed, {
                    'gate_weights': torch.ones(B, 1, device=device),
                    'retrieval_triggered': False
                }
            return x, context_embed, None

        # 检索触发，计算门控
        current_pooled = x.mean(dim=1)  # [B, d_model]
        gate_weights = self.gating(current_pooled, retrieved, similarity)

        # 生成检索上下文嵌入 (加权平均)
        sim_weights = F.softmax(similarity, dim=-1)  # [B, K]
        retrieved_feat = (retrieved.squeeze(-1) * sim_weights).sum(dim=-1, keepdim=True)  # [B, 1]

        # 简单扩展到d_model维度
        retrieved_expanded = retrieved_feat.expand(-1, self.d_repr)
        context_embed = self.context_proj(retrieved_expanded)  # [B, d_model]

        if return_retrieval_info:
            return x, context_embed, {
                'gate_weights': gate_weights,
                'similarity': similarity,
                'retrieval_triggered': True,
                'num_triggered': is_valid
            }

        return x, context_embed, None

    def compute_retrieval_loss(self, retrieval_info, predictions, y_true,
                                lambda_ref=0.001, warmup_steps=1000, current_step=None):
        """
        计算检索辅助损失 (极低权重)

        设计原则:
        - 损失权重极低，仅作为正则化
        - 鼓励门控的稳定性
        """
        if y_true is not None:
            device = y_true.device
        elif predictions is not None:
            device = predictions.device
        else:
            device = 'cpu'

        if retrieval_info is None or not retrieval_info.get('retrieval_triggered', False):
            return torch.tensor(0.0, device=device), {'retrieval_triggered': False}

        gate_weights = retrieval_info.get('gate_weights')

        if gate_weights is None:
            return torch.tensor(0.0, device=device), {'retrieval_triggered': False}

        # 门控稳定性损失: 鼓励门控远离0.5 (做出明确决策)
        gate_stability = -((gate_weights - 0.5).abs()).mean()

        # Warmup
        if current_step is not None:
            warmup_factor = min(1.0, current_step / warmup_steps)
        else:
            warmup_factor = 1.0

        total_loss = warmup_factor * lambda_ref * gate_stability

        return total_loss, {
            'gate_stability': gate_stability.item(),
            'warmup_factor': warmup_factor,
            'retrieval_triggered': True
        }


class RetrievalContext(nn.Module):
    """
    检索上下文构建器

    将检索信息转换为可用于prompt增强的嵌入
    """

    def __init__(self, d_model, llm_dim, dropout=0.1):
        super(RetrievalContext, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_model, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, context_embed, gate_weights=None):
        """
        Args:
            context_embed: [B, d_model]
            gate_weights: [B, 1] (可选)

        Returns:
            context_llm: [B, 1, llm_dim]
        """
        out = self.proj(context_embed)  # [B, llm_dim]
        return out.unsqueeze(1)  # [B, 1, llm_dim]
