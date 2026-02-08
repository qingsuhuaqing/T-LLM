"""
GRA-M (Global Retrieval-Augmented Memory) - 全局检索增强记忆网络

该模块包含两个子模块:
1. PVDR (Pattern-Value Dual Retriever) - 模式-数值双重检索器
   - 检索与当前输入相似的历史片段
   - 同时检索历史片段紧随其后的"未来值"作为参考
   - 结合形状(Shape)和数值(Value)进行相似度度量

2. AKG (Adaptive Knowledge Gating) - 自适应知识门控
   - 动态计算检索内容与当前输入的权重
   - 决定"听从历史经验"还是"依赖当前推理"
   - 解决LLM在时序预测中的不稳定性

参考论文:
- RAFT: Retrieval-Augmented Forecasting for Time Series
- TS-RAG: Time Series Retrieval-Augmented Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np


class PatternEncoder(nn.Module):
    """
    模式编码器
    将时间序列片段编码为可检索的向量表示
    """

    def __init__(self, d_model, d_repr=128, num_layers=2, dropout=0.1):
        """
        Args:
            d_model: 输入特征维度
            d_repr: 表示向量维度 (用于检索)
            num_layers: 编码器层数
            dropout: Dropout比例
        """
        super(PatternEncoder, self).__init__()

        self.d_model = d_model
        self.d_repr = d_repr

        # 多层Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 投影到表示空间
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_repr),
            nn.LayerNorm(d_repr)
        )

        # 可学习的[CLS] token用于聚合序列信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        """
        Args:
            x: 输入序列 [B, L, d_model]

        Returns:
            repr: 表示向量 [B, d_repr]
            seq_out: 序列输出 [B, L, d_model]
        """
        B, L, D = x.shape

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, D]

        # Transformer编码
        seq_out = self.encoder(x_with_cls)

        # 使用CLS token的输出作为表示
        cls_out = seq_out[:, 0, :]  # [B, D]

        # 投影到表示空间
        repr_vec = self.proj(cls_out)  # [B, d_repr]

        # 返回序列输出(不含CLS)用于后续处理
        return repr_vec, seq_out[:, 1:, :]


class PatternValueDualRetriever(nn.Module):
    """
    模式-数值双重检索器 (PVDR)

    核心机制:
    1. 检索与当前输入形状相似的历史模式
    2. 同时获取这些历史模式后续的真实值作为参考
    3. 结合形状和数值两个维度进行相似度计算
    """

    def __init__(self, d_model, d_repr=128, top_k=5, temperature=0.1, dropout=0.1):
        """
        Args:
            d_model: 特征维度
            d_repr: 表示向量维度
            top_k: 检索的相似模式数量
            temperature: 相似度softmax温度
            dropout: Dropout比例
        """
        super(PatternValueDualRetriever, self).__init__()

        self.d_model = d_model
        self.d_repr = d_repr
        self.top_k = top_k
        self.temperature = temperature

        # 模式编码器
        self.pattern_encoder = PatternEncoder(d_model, d_repr, dropout=dropout)

        # 数值编码器 (捕获数值特征)
        self.value_encoder = nn.Sequential(
            nn.Linear(d_model, d_repr),
            nn.LayerNorm(d_repr),
            nn.GELU(),
            nn.Linear(d_repr, d_repr)
        )

        # 形状-数值融合权重
        self.shape_weight = nn.Parameter(torch.tensor(0.5))

        # 历史模式库 (在线构建)
        # 这些在训练时会动态更新
        self.register_buffer('memory_keys', None)  # [M, d_repr]
        self.register_buffer('memory_values', None)  # [M, pred_len, d_model]
        self.register_buffer('memory_size', torch.tensor(0))
        self.max_memory_size = 10000  # 最大记忆容量

    def build_memory(self, train_data, train_labels):
        """
        离线阶段: 从训练数据构建历史模式库

        Args:
            train_data: 训练输入 [N, seq_len, d_model]
            train_labels: 训练标签 (未来值) [N, pred_len, d_model]
        """
        with torch.no_grad():
            # 编码所有训练样本
            repr_vecs, _ = self.pattern_encoder(train_data)  # [N, d_repr]
            value_vecs = self.value_encoder(train_data.mean(dim=1))  # [N, d_repr]

            # 融合形状和数值特征
            combined_keys = self.shape_weight * repr_vecs + (1 - self.shape_weight) * value_vecs

            # 存储到记忆库
            self.memory_keys = combined_keys
            self.memory_values = train_labels
            self.memory_size = torch.tensor(train_data.shape[0])

    def update_memory(self, new_data, new_labels, update_ratio=0.1):
        """
        在线更新记忆库 (可选)

        Args:
            new_data: 新样本输入
            new_labels: 新样本标签
            update_ratio: 更新比例
        """
        if self.memory_keys is None:
            self.build_memory(new_data, new_labels)
            return

        with torch.no_grad():
            # 编码新样本
            repr_vecs, _ = self.pattern_encoder(new_data)
            value_vecs = self.value_encoder(new_data.mean(dim=1))
            new_keys = self.shape_weight * repr_vecs + (1 - self.shape_weight) * value_vecs

            # 随机替换部分旧记忆
            num_replace = int(self.memory_size * update_ratio)
            replace_idx = torch.randperm(self.memory_size)[:num_replace]

            if num_replace > 0 and new_keys.shape[0] >= num_replace:
                self.memory_keys[replace_idx] = new_keys[:num_replace]
                self.memory_values[replace_idx] = new_labels[:num_replace]

    def retrieve(self, query, return_indices=False):
        """
        在线阶段: 检索相似历史模式

        Args:
            query: 查询输入 [B, L, d_model]
            return_indices: 是否返回检索索引

        Returns:
            retrieved_refs: 检索到的历史未来值 [B, top_k, pred_len, d_model]
            similarity_weights: 相似度权重 [B, top_k]
            indices: 检索索引 (可选)
        """
        if self.memory_keys is None:
            # 记忆库为空时返回None
            return None, None, None

        B = query.shape[0]

        # 编码查询
        query_repr, _ = self.pattern_encoder(query)  # [B, d_repr]
        query_value = self.value_encoder(query.mean(dim=1))  # [B, d_repr]
        query_key = self.shape_weight * query_repr + (1 - self.shape_weight) * query_value

        # 计算与记忆库的相似度
        # 使用欧几里得距离 (时序数据的形状相似性)
        # similarity = -||q - k||^2
        query_key_expanded = query_key.unsqueeze(1)  # [B, 1, d_repr]
        memory_keys_expanded = self.memory_keys.unsqueeze(0)  # [1, M, d_repr]

        # 欧几里得距离
        distances = torch.sum((query_key_expanded - memory_keys_expanded) ** 2, dim=-1)  # [B, M]
        similarity = -distances / self.temperature

        # 选择Top-K
        top_k = min(self.top_k, self.memory_keys.shape[0])
        top_scores, top_indices = torch.topk(similarity, k=top_k, dim=-1)  # [B, top_k]

        # 归一化权重
        similarity_weights = F.softmax(top_scores, dim=-1)  # [B, top_k]

        # 检索对应的历史未来值
        # memory_values: [M, pred_len, d_model]
        retrieved_refs = self.memory_values[top_indices]  # [B, top_k, pred_len, d_model]

        if return_indices:
            return retrieved_refs, similarity_weights, top_indices

        return retrieved_refs, similarity_weights, None


class AdaptiveKnowledgeGating(nn.Module):
    """
    自适应知识门控 (AKG)

    动态决定"听从历史经验"还是"依赖当前推理":
    1. 评估当前输入与检索模式的匹配度
    2. 自适应融合模型预测和检索参考
    3. 处理检索噪声，提升预测稳定性
    """

    def __init__(self, d_model, pred_len, dropout=0.1):
        """
        Args:
            d_model: 特征维度
            pred_len: 预测长度
            dropout: Dropout比例
        """
        super(AdaptiveKnowledgeGating, self).__init__()

        self.d_model = d_model
        self.pred_len = pred_len

        # 当前特征编码器
        self.current_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 检索特征编码器
        self.retrieved_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 匹配度评估网络
        self.match_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # 门控网络: 决定依赖历史还是当前
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # 交叉注意力融合
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )

        # 残差连接后的投影
        self.output_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, current_feat, retrieved_refs, similarity_weights, model_prediction=None):
        """
        Args:
            current_feat: 当前模型特征 [B, L, d_model]
            retrieved_refs: 检索到的历史参考 [B, top_k, pred_len, d_model] 或 [B, top_k, d_model]
            similarity_weights: 相似度权重 [B, top_k]
            model_prediction: 模型预测 (可选) [B, pred_len, d_model]

        Returns:
            fused_output: 融合后的输出 [B, L, d_model]
            gate_weights: 门控权重 [B, 1]
            match_scores: 匹配度分数 [B, top_k]
        """
        if retrieved_refs is None:
            # 没有检索结果时直接返回原始特征
            return current_feat, torch.ones(current_feat.shape[0], 1, device=current_feat.device), None

        B = current_feat.shape[0]

        # 1. 编码当前特征
        current_pooled = current_feat.mean(dim=1)  # [B, d_model]
        current_encoded = self.current_encoder(current_pooled)  # [B, d_model]

        # 2. 处理检索结果
        if retrieved_refs.dim() == 4:
            # [B, top_k, pred_len, d_model] -> [B, top_k, d_model]
            retrieved_pooled = retrieved_refs.mean(dim=2)
        else:
            retrieved_pooled = retrieved_refs  # [B, top_k, d_model]

        retrieved_encoded = self.retrieved_encoder(retrieved_pooled)  # [B, top_k, d_model]

        # 3. 计算每个检索结果的匹配度
        current_expanded = current_encoded.unsqueeze(1).expand(-1, retrieved_encoded.shape[1], -1)
        match_input = torch.cat([current_expanded, retrieved_encoded], dim=-1)  # [B, top_k, d_model*2]
        match_scores = self.match_scorer(match_input).squeeze(-1)  # [B, top_k]

        # 4. 加权聚合检索结果
        # 结合相似度权重和匹配度
        combined_weights = similarity_weights * match_scores  # [B, top_k]
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 加权平均得到参考值
        weighted_ref = torch.einsum('bk,bkd->bd', combined_weights, retrieved_pooled)  # [B, d_model]
        weighted_ref_encoded = self.retrieved_encoder(weighted_ref)  # [B, d_model]

        # 5. 门控决策: 听从历史还是依赖当前
        gate_input = torch.cat([current_encoded, weighted_ref_encoded], dim=-1)  # [B, d_model*2]
        gate_weights = self.gate_network(gate_input)  # [B, 1]

        # 6. 交叉注意力融合
        # 当前特征作为Query，检索结果作为Key/Value
        attn_out, _ = self.cross_attention(
            current_feat,  # [B, L, d_model]
            retrieved_encoded,  # [B, top_k, d_model]
            retrieved_encoded
        )

        # 7. 门控融合
        fused = gate_weights.unsqueeze(1) * current_feat + (1 - gate_weights.unsqueeze(1)) * attn_out

        # 8. 残差连接
        fused_output = torch.cat([current_feat, fused], dim=-1)
        fused_output = self.output_proj(fused_output)

        return fused_output, gate_weights, match_scores


class GRAM(nn.Module):
    """
    全局检索增强记忆网络 (Global Retrieval-Augmented Memory)

    整合PVDR和AKG模块，实现:
    1. 历史模式检索增强
    2. 自适应知识门控
    3. 突破LLM上下文窗口限制
    """

    def __init__(self, configs):
        """
        Args:
            configs: 配置对象，包含:
                - d_model: 模型维度
                - d_ff: FFN维度 (用作d_repr)
                - pred_len: 预测长度
                - top_k: 检索数量 (默认5)
                - dropout: Dropout比例
        """
        super(GRAM, self).__init__()

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.d_repr = getattr(configs, 'd_repr', min(128, configs.d_ff))
        self.pred_len = configs.pred_len
        self.top_k = getattr(configs, 'top_k', 5)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 模式-数值双重检索器
        self.retriever = PatternValueDualRetriever(
            d_model=self.d_model,
            d_repr=self.d_repr,
            top_k=self.top_k,
            dropout=self.dropout
        )

        # 自适应知识门控
        self.gating = AdaptiveKnowledgeGating(
            d_model=self.d_model,
            pred_len=self.pred_len,
            dropout=self.dropout
        )

        # 检索上下文投影 (用于增强Prompt)
        self.context_proj = nn.Linear(self.d_model, self.d_model)

        # 是否启用检索 (训练时可能需要disable)
        self.retrieval_enabled = True

    @staticmethod
    def _resize_last_dim(tensor, target_dim):
        """
        将任意形状张量的“最后一维”重采样到 target_dim，前面的维度保持不变。

        设计目的:
        1) memory_values 期望是 [..., d_model]，否则后续 Linear(d_model, d_model) 会报维度错误。
        2) 旧实现在构建记忆库时把插值用在了错误维度，可能产生 [..., 1] 或 [..., 7] 等形状。

        实现方式:
        - 先把张量折叠成 [N, 1, C]（C=原最后一维）
        - 在 C 这个轴上线性插值到 target_dim
        - 再恢复回原始前缀维度
        """
        if tensor is None or tensor.shape[-1] == target_dim:
            return tensor

        original_shape = tensor.shape
        original_dtype = tensor.dtype

        # F.interpolate(linear) 需要 3D 输入 [N, C, L]；此处固定通道数为1，
        # 把“最后一维”当作可插值的长度轴处理。
        tensor_3d = tensor.reshape(-1, 1, original_shape[-1]).float()
        resized = F.interpolate(
            tensor_3d,
            size=target_dim,
            mode='linear',
            align_corners=False
        )
        return resized.reshape(*original_shape[:-1], target_dim).to(original_dtype)

    def build_memory_from_dataloader(self, dataloader, model, device, max_samples=5000):
        """
        从数据加载器构建记忆库

        Args:
            dataloader: 训练数据加载器
            model: 用于获取特征的模型
            device: 设备
            max_samples: 最大样本数
        """
        all_features = []
        all_labels = []
        sample_count = 0

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, _ in dataloader:
                if sample_count >= max_samples:
                    break

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                # 简单使用输入的均值作为特征
                # 在实际使用中，可以使用模型的中间特征
                features = batch_x.mean(dim=-1, keepdim=True).expand(-1, -1, self.d_model)

                # 取预测部分作为标签
                labels = batch_y[:, -self.pred_len:, :]

                # 关键修复:
                # 记忆库里检索值 memory_values 必须是 [N, pred_len, d_model]。
                # 旧代码把 labels.permute 后按 size=d_model 插值，实际上改的是“时间维”，
                # 会导致最后一维仍是原通道数(如1/7)，进而在 AKG 的 Linear(d_model, d_model)
                # 处触发:
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (...x1 and 64x64)
                #
                # 这里统一按“最后一维”对齐，确保 features/labels 最后一维都是 d_model。
                features = self._resize_last_dim(features, self.d_model)
                labels = self._resize_last_dim(labels, self.d_model)

                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
                sample_count += features.shape[0]

        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)[:max_samples].to(device)
            all_labels = torch.cat(all_labels, dim=0)[:max_samples].to(device)
            self.retriever.build_memory(all_features, all_labels)

        model.train()

    def forward(self, x, return_retrieval_info=False):
        """
        Args:
            x: 输入特征 [B*N, L, d_model]
            return_retrieval_info: 是否返回检索详情

        Returns:
            enhanced_x: 检索增强后的特征 [B*N, L, d_model]
            context_embed: 检索上下文嵌入 (用于Prompt增强)
            retrieval_info: 检索信息字典 (可选)
        """
        if not self.retrieval_enabled or self.retriever.memory_keys is None:
            # 检索未启用或记忆库为空
            context_embed = torch.zeros(x.shape[0], self.d_model, device=x.device)
            if return_retrieval_info:
                return x, context_embed, {'gate_weights': torch.ones(x.shape[0], 1, device=x.device)}
            return x, context_embed, None

        # 1. 检索相似历史模式
        retrieved_refs, similarity_weights, indices = self.retriever.retrieve(x, return_indices=True)

        if retrieved_refs is None:
            context_embed = torch.zeros(x.shape[0], self.d_model, device=x.device)
            if return_retrieval_info:
                return x, context_embed, {'gate_weights': torch.ones(x.shape[0], 1, device=x.device)}
            return x, context_embed, None

        # 兼容性兜底:
        # 如果加载的是旧 checkpoint，里面可能已经保存了错误形状的 memory_values
        # （最后一维不是 d_model）。这里在前向阶段再次对齐，避免训练/推理直接崩溃。
        retrieved_refs = self._resize_last_dim(retrieved_refs, self.d_model)

        # 2. 自适应门控融合
        enhanced_x, gate_weights, match_scores = self.gating(
            x, retrieved_refs, similarity_weights
        )

        # 3. 生成检索上下文嵌入 (用于增强Prompt)
        # 使用加权检索结果的均值
        if retrieved_refs.dim() == 4:
            ref_summary = (retrieved_refs * similarity_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1).mean(dim=1)
        else:
            ref_summary = (retrieved_refs * similarity_weights.unsqueeze(-1)).sum(dim=1)

        context_embed = self.context_proj(ref_summary)  # [B*N, d_model]

        if return_retrieval_info:
            return enhanced_x, context_embed, {
                'gate_weights': gate_weights,
                'match_scores': match_scores,
                'similarity_weights': similarity_weights,
                'retrieved_indices': indices
            }

        return enhanced_x, context_embed, None

    def compute_retrieval_loss(self, retrieval_info, predictions, y_true, lambda_ref=0.1):
        """
        计算检索增强的辅助损失

        Args:
            retrieval_info: forward返回的检索信息
            predictions: 模型预测
            y_true: 真实值
            lambda_ref: 参考一致性损失权重

        Returns:
            total_loss: 辅助损失
            loss_dict: 损失详情
        """
        if retrieval_info is None or 'gate_weights' not in retrieval_info:
            return torch.tensor(0.0), {}

        gate_weights = retrieval_info['gate_weights']

        # 鼓励门控权重的多样性 (避免总是选择同一种策略)
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8) +
                        (1 - gate_weights) * torch.log(1 - gate_weights + 1e-8)).mean()

        # 检索一致性: 高置信度时门控权重应该更稳定
        if 'match_scores' in retrieval_info and retrieval_info['match_scores'] is not None:
            match_scores = retrieval_info['match_scores']
            confidence = match_scores.mean(dim=-1, keepdim=True)
            gate_consistency = ((gate_weights - 0.5).abs() * confidence).mean()
        else:
            gate_consistency = torch.tensor(0.0)

        total_loss = -0.01 * gate_entropy + lambda_ref * gate_consistency

        return total_loss, {
            'gate_entropy': gate_entropy.item(),
            'gate_consistency': gate_consistency.item() if isinstance(gate_consistency, torch.Tensor) else gate_consistency
        }


class RetrievalAugmentedPrompt(nn.Module):
    """
    检索增强的Prompt构建器

    将检索到的历史模式信息融入Prompt中:
    1. 历史相似模式的统计信息
    2. 历史模式后续发展的参考
    3. 置信度指示
    """

    def __init__(self, d_model, llm_dim, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            llm_dim: LLM嵌入维度
            dropout: Dropout比例
        """
        super(RetrievalAugmentedPrompt, self).__init__()

        self.d_model = d_model
        self.llm_dim = llm_dim

        # 检索上下文到LLM空间的投影
        self.context_to_llm = nn.Sequential(
            nn.Linear(d_model, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 置信度编码
        self.confidence_embed = nn.Embedding(10, llm_dim)  # 10个离散置信度级别

    def forward(self, context_embed, gate_weights=None):
        """
        Args:
            context_embed: 检索上下文 [B, d_model]
            gate_weights: 门控权重 [B, 1]

        Returns:
            retrieval_prompt_embed: 检索增强的Prompt嵌入 [B, 2, llm_dim]
        """
        B = context_embed.shape[0]

        # 1. 投影检索上下文到LLM空间
        context_llm = self.context_to_llm(context_embed)  # [B, llm_dim]

        # 2. 编码置信度
        if gate_weights is not None:
            # 离散化置信度到0-9
            confidence_level = (gate_weights.squeeze(-1) * 9).long().clamp(0, 9)
        else:
            confidence_level = torch.full((B,), 5, dtype=torch.long, device=context_embed.device)

        confidence_llm = self.confidence_embed(confidence_level)  # [B, llm_dim]

        # 3. 组合成Prompt嵌入
        retrieval_prompt_embed = torch.stack([context_llm, confidence_llm], dim=1)  # [B, 2, llm_dim]

        return retrieval_prompt_embed
