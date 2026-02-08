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

修改日志 (2026-02-08):
- 修复 compute_retrieval_loss 中 device 不匹配问题
- 修复记忆库构建时的维度错误
- 添加 warmup 机制防止过拟合
- 优化特征提取方式，使用更有意义的表示
- 添加详细注释说明每个组件的作用
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
    使用Transformer编码器捕获序列中的时序依赖关系

    设计理念:
    - 使用CLS token聚合整个序列的信息
    - 投影到低维表示空间便于高效检索
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
        # 捕获序列中的时序依赖关系
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
        # 使用LayerNorm稳定训练
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_repr),
            nn.LayerNorm(d_repr)
        )

        # 可学习的[CLS] token用于聚合序列信息
        # 类似BERT的设计，CLS token的输出表示整个序列
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

        # 添加CLS token到序列开头
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, D]

        # Transformer编码
        seq_out = self.encoder(x_with_cls)

        # 使用CLS token的输出作为整个序列的表示
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

    创新点:
    - 形状相似性: 捕获时序模式的形态特征
    - 数值相似性: 考虑实际数值范围的匹配
    - 双重检索: 综合两种相似性进行最终匹配
    """

    def __init__(self, d_model, d_repr=128, top_k=5, temperature=0.1, dropout=0.1, pred_len=96):
        """
        Args:
            d_model: 特征维度
            d_repr: 表示向量维度
            top_k: 检索的相似模式数量
            temperature: 相似度softmax温度 (越小越尖锐)
            dropout: Dropout比例
            pred_len: 预测长度 (用于初始化 memory_values 的形状)
        """
        super(PatternValueDualRetriever, self).__init__()

        self.d_model = d_model
        self.d_repr = d_repr
        self.top_k = top_k
        self.temperature = temperature
        self.pred_len = pred_len

        # 模式编码器 - 提取形状特征
        self.pattern_encoder = PatternEncoder(d_model, d_repr, dropout=dropout)

        # 数值编码器 - 提取数值统计特征
        self.value_encoder = nn.Sequential(
            nn.Linear(d_model, d_repr),
            nn.LayerNorm(d_repr),
            nn.GELU(),
            nn.Linear(d_repr, d_repr)
        )

        # 形状-数值融合权重
        # 可学习的权重决定两种相似性的相对重要性
        self.shape_weight = nn.Parameter(torch.tensor(0.5))

        # 历史模式库 (在线构建)
        # 使用register_buffer确保这些tensor随模型保存/加载
        # 修复: 初始化为空tensor而非None，确保checkpoint加载时shape兼容
        self.register_buffer('memory_keys', torch.zeros(0, d_repr))  # [M, d_repr] - 检索键
        self.register_buffer('memory_values', torch.zeros(0, pred_len, d_model))  # [M, pred_len, d_model] - 检索值
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
            # shape_weight控制两者的相对重要性
            combined_keys = self.shape_weight * repr_vecs + (1 - self.shape_weight) * value_vecs

            # 存储到记忆库
            self.memory_keys = combined_keys
            self.memory_values = train_labels
            self.memory_size = torch.tensor(train_data.shape[0])

    def update_memory(self, new_data, new_labels, update_ratio=0.1):
        """
        在线更新记忆库 (可选)

        支持增量更新，随机替换部分旧记忆

        Args:
            new_data: 新样本输入
            new_labels: 新样本标签
            update_ratio: 更新比例
        """
        # 修复: 检查记忆库是否为空 (shape[0] == 0 而非 None)
        if self.memory_keys.shape[0] == 0:
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

        使用欧几里得距离计算相似度（时序数据的形状相似性）

        Args:
            query: 查询输入 [B, L, d_model]
            return_indices: 是否返回检索索引

        Returns:
            retrieved_refs: 检索到的历史未来值 [B, top_k, pred_len, d_model]
            similarity_weights: 相似度权重 [B, top_k]
            indices: 检索索引 (可选)
        """
        # 修复: 检查记忆库是否为空 (shape[0] == 0 而非 None)
        if self.memory_keys.shape[0] == 0:
            # 记忆库为空时返回None
            return None, None, None

        B = query.shape[0]

        # 编码查询
        query_repr, _ = self.pattern_encoder(query)  # [B, d_repr]
        query_value = self.value_encoder(query.mean(dim=1))  # [B, d_repr]
        query_key = self.shape_weight * query_repr + (1 - self.shape_weight) * query_value

        # 计算与记忆库的相似度
        # 使用欧几里得距离 (时序数据的形状相似性更适合用距离度量)
        query_key_expanded = query_key.unsqueeze(1)  # [B, 1, d_repr]
        memory_keys_expanded = self.memory_keys.unsqueeze(0)  # [1, M, d_repr]

        # 欧几里得距离转换为相似度
        distances = torch.sum((query_key_expanded - memory_keys_expanded) ** 2, dim=-1)  # [B, M]
        similarity = -distances / self.temperature

        # 选择Top-K最相似的模式
        top_k = min(self.top_k, self.memory_keys.shape[0])
        top_scores, top_indices = torch.topk(similarity, k=top_k, dim=-1)  # [B, top_k]

        # 归一化权重 (softmax确保权重和为1)
        similarity_weights = F.softmax(top_scores, dim=-1)  # [B, top_k]

        # 检索对应的历史未来值
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

    设计理念:
    - 当检索到的历史模式与当前输入高度匹配时，更多依赖历史经验
    - 当匹配度低或当前输入有独特特征时，更多依赖模型推理
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
        # 评估每个检索结果与当前输入的匹配程度
        self.match_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # 门控网络: 决定依赖历史还是当前
        # 输出接近1表示依赖当前推理，接近0表示依赖历史经验
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # 交叉注意力融合
        # 当前特征作为Query，检索结果作为Key/Value
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

        # 1. 编码当前特征 (全局池化)
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
        # 将当前特征与每个检索结果拼接，评估匹配程度
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
        # gate_weight高: 更依赖当前特征
        # gate_weight低: 更依赖检索结果
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

    创新点:
    - 通过检索历史相似模式，为预测提供额外参考
    - 自适应门控机制避免错误检索结果的干扰
    - 检索上下文可以作为额外Prompt增强LLM理解
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
            dropout=self.dropout,
            pred_len=self.pred_len  # 传入 pred_len 用于初始化 memory_values 形状
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

        # 训练步数计数器，用于warmup
        self.register_buffer('train_step', torch.tensor(0))

    @staticmethod
    def _resize_last_dim(tensor, target_dim):
        """
        将任意形状张量的"最后一维"重采样到 target_dim

        设计目的:
        1) memory_values 期望是 [..., d_model]
        2) 确保维度一致，避免 Linear 层报错

        Args:
            tensor: 输入张量
            target_dim: 目标维度

        Returns:
            resized_tensor: 调整后的张量
        """
        if tensor is None or tensor.shape[-1] == target_dim:
            return tensor

        original_shape = tensor.shape
        original_dtype = tensor.dtype

        # F.interpolate 需要 3D 输入 [N, C, L]
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

        修复说明:
        - 使用更有意义的特征表示
        - 确保维度正确匹配 d_model
        - 添加详细注释

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

                # ========== 修复: 使用更有意义的特征表示 ==========
                B, T, C = batch_x.shape  # [batch, seq_len, n_vars]

                # 方案: 对每个变量独立处理，与模型内部Channel Independence一致
                # [B, T, C] -> [B*C, T]
                batch_x_flat = batch_x.permute(0, 2, 1).reshape(B * C, T)

                # 提取统计特征作为表示
                # 使用多种统计量捕获序列特征
                mean_feat = batch_x_flat.mean(dim=1, keepdim=True)  # 均值
                std_feat = batch_x_flat.std(dim=1, keepdim=True)   # 标准差
                max_feat = batch_x_flat.max(dim=1, keepdim=True)[0]  # 最大值
                min_feat = batch_x_flat.min(dim=1, keepdim=True)[0]  # 最小值

                # 计算趋势特征
                diff = batch_x_flat[:, 1:] - batch_x_flat[:, :-1]
                trend_feat = diff.sum(dim=1, keepdim=True)  # 总体趋势

                # 拼接统计特征
                stat_features = torch.cat([mean_feat, std_feat, max_feat, min_feat, trend_feat], dim=1)  # [B*C, 5]

                # 扩展到 d_model 维度
                # 使用重复填充确保维度一致
                repeat_times = (self.d_model + 4) // 5
                features = stat_features.repeat(1, repeat_times)[:, :self.d_model]  # [B*C, d_model]

                # 扩展时间维度用于检索
                features = features.unsqueeze(1).expand(-1, T, -1)  # [B*C, T, d_model]

                # 处理标签
                labels = batch_y[:, -self.pred_len:, :]  # [B, pred_len, C]
                labels = labels.permute(0, 2, 1).reshape(B * C, self.pred_len)  # [B*C, pred_len]

                # 扩展标签到 d_model 维度
                labels_expanded = labels.unsqueeze(-1).expand(-1, -1, self.d_model)  # [B*C, pred_len, d_model]

                all_features.append(features.cpu())
                all_labels.append(labels_expanded.cpu())
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
        device = x.device

        # 修复: 检查记忆库是否为空 (shape[0] == 0 而非 None)
        if not self.retrieval_enabled or self.retriever.memory_keys.shape[0] == 0:
            # 检索未启用或记忆库为空
            context_embed = torch.zeros(x.shape[0], self.d_model, device=device)
            if return_retrieval_info:
                return x, context_embed, {'gate_weights': torch.ones(x.shape[0], 1, device=device)}
            return x, context_embed, None

        # 1. 检索相似历史模式
        retrieved_refs, similarity_weights, indices = self.retriever.retrieve(x, return_indices=True)

        if retrieved_refs is None:
            context_embed = torch.zeros(x.shape[0], self.d_model, device=device)
            if return_retrieval_info:
                return x, context_embed, {'gate_weights': torch.ones(x.shape[0], 1, device=device)}
            return x, context_embed, None

        # 维度兼容处理
        retrieved_refs = self._resize_last_dim(retrieved_refs, self.d_model)

        # 2. 自适应门控融合
        enhanced_x, gate_weights, match_scores = self.gating(
            x, retrieved_refs, similarity_weights
        )

        # 3. 生成检索上下文嵌入 (用于增强Prompt)
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

    def compute_retrieval_loss(self, retrieval_info, predictions, y_true, lambda_ref=0.1,
                                warmup_steps=500, current_step=None):
        """
        计算检索增强的辅助损失

        修复说明:
        1. 添加 device 参数确保返回值在正确设备上
        2. 添加 warmup 机制防止训练初期干扰
        3. 优化损失函数设计

        Args:
            retrieval_info: forward返回的检索信息
            predictions: 模型预测
            y_true: 真实值
            lambda_ref: 参考一致性损失权重
            warmup_steps: warmup步数
            current_step: 当前训练步数

        Returns:
            total_loss: 辅助损失
            loss_dict: 损失详情
        """
        # 获取设备
        if y_true is not None:
            device = y_true.device
        elif predictions is not None:
            device = predictions.device
        else:
            device = 'cpu'

        if retrieval_info is None or 'gate_weights' not in retrieval_info:
            return torch.tensor(0.0, device=device), {}

        gate_weights = retrieval_info['gate_weights']

        # 1. 门控熵损失
        # 鼓励门控权重的多样性 (避免总是选择同一种策略)
        # 使用二元熵: -p*log(p) - (1-p)*log(1-p)
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8) +
                        (1 - gate_weights) * torch.log(1 - gate_weights + 1e-8)).mean()

        # 2. 门控一致性损失
        # 高置信度匹配时，门控权重应该更稳定
        if 'match_scores' in retrieval_info and retrieval_info['match_scores'] is not None:
            match_scores = retrieval_info['match_scores']
            confidence = match_scores.mean(dim=-1, keepdim=True)
            # 高置信度时鼓励门控权重远离0.5
            gate_consistency = ((gate_weights - 0.5).abs() * confidence).mean()
        else:
            gate_consistency = torch.tensor(0.0, device=device)

        # ========== Warmup 机制 ==========
        if current_step is not None:
            warmup_factor = min(1.0, current_step / warmup_steps)
        else:
            self.train_step += 1
            warmup_factor = min(1.0, self.train_step.item() / warmup_steps)

        # 总损失
        # -gate_entropy: 鼓励多样性 (最大化熵)
        # +gate_consistency: 高置信度时鼓励决断性
        total_loss = warmup_factor * (-0.01 * gate_entropy + lambda_ref * gate_consistency)

        return total_loss, {
            'gate_entropy': gate_entropy.item(),
            'gate_consistency': gate_consistency.item() if isinstance(gate_consistency, torch.Tensor) else gate_consistency,
            'warmup_factor': warmup_factor
        }


class RetrievalAugmentedPrompt(nn.Module):
    """
    检索增强的Prompt构建器

    将检索到的历史模式信息融入Prompt中:
    1. 历史相似模式的统计信息
    2. 历史模式后续发展的参考
    3. 置信度指示

    设计理念:
    - 检索上下文作为额外的Prompt token
    - 置信度编码让模型知道检索结果的可靠程度
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

        # 置信度编码 (10个离散级别)
        self.confidence_embed = nn.Embedding(10, llm_dim)

    def forward(self, context_embed, gate_weights=None):
        """
        Args:
            context_embed: 检索上下文 [B, d_model]
            gate_weights: 门控权重 [B, 1]

        Returns:
            retrieval_prompt_embed: 检索增强的Prompt嵌入 [B, 2, llm_dim]
        """
        B = context_embed.shape[0]
        device = context_embed.device

        # 1. 投影检索上下文到LLM空间
        context_llm = self.context_to_llm(context_embed)  # [B, llm_dim]

        # 2. 编码置信度
        if gate_weights is not None:
            # 离散化置信度到0-9
            # gate_weight接近0表示高置信度(依赖检索)
            # gate_weight接近1表示低置信度(依赖当前)
            confidence_level = ((1 - gate_weights.squeeze(-1)) * 9).long().clamp(0, 9)
        else:
            confidence_level = torch.full((B,), 5, dtype=torch.long, device=device)

        confidence_llm = self.confidence_embed(confidence_level)  # [B, llm_dim]

        # 3. 组合成Prompt嵌入
        retrieval_prompt_embed = torch.stack([context_llm, confidence_llm], dim=1)  # [B, 2, llm_dim]

        return retrieval_prompt_embed
