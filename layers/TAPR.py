"""
TAPR (Trend-Aware Patch Router) - 趋势感知尺度路由模块

该模块包含两个子模块:
1. DM (Decomposable Multi-scale) - 可分解多尺度混合
   - 通过平均池化将输入序列下采样为多个分辨率
   - 在每个尺度上进行季节性-趋势分解
   - 允许信息在不同尺度间流动（从粗到细）

2. C2F (Coarse-to-Fine Head) - 先粗后细预测头
   - 层级决策机制：先判断趋势方向，再细化预测
   - 层次化一致性损失（HCL）约束不同层级分类器的输出分布
   - 实现"先判涨跌再细分"的思想

参考论文:
- TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024)
- HCAN: Hierarchical Classification with Alignment Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MultiScaleDecomposition(nn.Module):
    """
    可分解多尺度模块 (Decomposable Multi-scale Module)

    通过多尺度分解捕获不同时间分辨率的模式:
    - 微观尺度: 高频波动、噪声、短期突变
    - 中观尺度: 日内周期、工作日模式
    - 宏观尺度: 长期趋势、季节性变化
    """

    def __init__(self, d_model, n_scales=3, kernel_sizes=None, dropout=0.1):
        """
        Args:
            d_model: 嵌入维度
            n_scales: 尺度数量 (默认3: 原始/4x下采样/16x下采样)
            kernel_sizes: 每个尺度的下采样核大小
            dropout: Dropout比例
        """
        super(MultiScaleDecomposition, self).__init__()

        self.n_scales = n_scales
        self.d_model = d_model

        # 默认下采样率: 1x, 4x, 16x
        if kernel_sizes is None:
            kernel_sizes = [1, 4, 16]
        self.kernel_sizes = kernel_sizes[:n_scales]

        # 每个尺度的特征提取器
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Dropout(dropout)
            ) for _ in range(n_scales)
        ])

        # 趋势-季节性分解模块
        self.decomp_layers = nn.ModuleList([
            SeriesDecomposition(kernel_size=max(3, ks // 2 * 2 + 1))
            for ks in self.kernel_sizes
        ])

        # 跨尺度融合层 (从粗到细)
        self.cross_scale_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
            for _ in range(n_scales - 1)
        ])

        # 尺度自适应权重
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # 输出投影
        self.output_proj = nn.Linear(d_model * n_scales, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [B*N, num_patches, d_model]

        Returns:
            多尺度融合后的特征 [B*N, num_patches, d_model]
            各尺度的趋势分量 (用于趋势分类)
        """
        B, L, D = x.shape

        # 转换为 [B, D, L] 用于卷积操作
        x_conv = x.permute(0, 2, 1)

        scale_features = []
        trend_features = []
        seasonal_features = []

        for i, (encoder, decomp, ks) in enumerate(
            zip(self.scale_encoders, self.decomp_layers, self.kernel_sizes)
        ):
            # 1. 下采样到当前尺度
            if ks > 1:
                x_scale = F.avg_pool1d(x_conv, kernel_size=ks, stride=ks,
                                       padding=ks//2 if L % ks != 0 else 0)
            else:
                x_scale = x_conv

            # 2. 特征提取
            x_scale = encoder(x_scale)

            # 3. 转换回 [B, L_scale, D]
            x_scale = x_scale.permute(0, 2, 1)

            # 4. 趋势-季节性分解
            trend, seasonal = decomp(x_scale)

            # 5. 上采样回原始长度
            if ks > 1:
                trend = F.interpolate(trend.permute(0, 2, 1), size=L, mode='linear', align_corners=False)
                trend = trend.permute(0, 2, 1)
                seasonal = F.interpolate(seasonal.permute(0, 2, 1), size=L, mode='linear', align_corners=False)
                seasonal = seasonal.permute(0, 2, 1)
                x_scale = F.interpolate(x_scale.permute(0, 2, 1), size=L, mode='linear', align_corners=False)
                x_scale = x_scale.permute(0, 2, 1)

            scale_features.append(x_scale)
            trend_features.append(trend)
            seasonal_features.append(seasonal)

        # 6. 从粗到细的跨尺度注意力融合
        # 粗尺度信息指导细尺度预测
        for i in range(self.n_scales - 2, -1, -1):
            if i < len(self.cross_scale_attn):
                # 使用粗尺度作为Key/Value，细尺度作为Query
                coarse = scale_features[i + 1] if i + 1 < len(scale_features) else scale_features[i]
                fine = scale_features[i]
                attn_out, _ = self.cross_scale_attn[i](fine, coarse, coarse)
                scale_features[i] = fine + attn_out

        # 7. 自适应权重融合
        weights = F.softmax(self.scale_weights, dim=0)

        # 拼接所有尺度特征
        multi_scale_out = torch.cat(scale_features, dim=-1)  # [B, L, D*n_scales]
        multi_scale_out = self.output_proj(multi_scale_out)  # [B, L, D]

        # 加权趋势特征 (用于趋势分类)
        weighted_trend = sum(w * t for w, t in zip(weights, trend_features))

        return self.dropout(multi_scale_out + x), weighted_trend, trend_features


class SeriesDecomposition(nn.Module):
    """
    季节性-趋势分解模块
    使用移动平均提取趋势，残差作为季节性成分
    """
    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            trend: [B, L, D]
            seasonal: [B, L, D]
        """
        # 转换为 [B, D, L] 用于池化
        x_t = x.permute(0, 2, 1)

        # 移动平均提取趋势
        trend = self.avg(x_t)

        # 处理边界
        if trend.shape[-1] != x_t.shape[-1]:
            trend = F.interpolate(trend, size=x_t.shape[-1], mode='linear', align_corners=False)

        # 转换回 [B, L, D]
        trend = trend.permute(0, 2, 1)

        # 残差为季节性成分
        seasonal = x - trend

        return trend, seasonal


class CoarseToFineHead(nn.Module):
    """
    先粗后细预测头 (Coarse-to-Fine Head)

    层级决策机制:
    - Level 1 (二分类): 上涨 vs 下跌
    - Level 2 (四分类): 强上涨/弱上涨/弱下跌/强下跌
    - Level 3 (回归): 具体数值预测

    通过层次化一致性损失(HCL)强制细粒度预测符合粗粒度趋势判断
    """

    def __init__(self, d_model, n_vars, pred_len, num_trend_classes=4, dropout=0.1):
        """
        Args:
            d_model: 特征维度
            n_vars: 变量数量
            pred_len: 预测长度
            num_trend_classes: 趋势分类数 (默认4: 强涨/弱涨/弱跌/强跌)
            dropout: Dropout比例
        """
        super(CoarseToFineHead, self).__init__()

        self.d_model = d_model
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.num_trend_classes = num_trend_classes

        # Level 1: 方向分类器 (上涨/下跌/平稳 -> 3类)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 上涨/下跌/平稳
        )

        # Level 2: 幅度分类器 (强/弱 * 涨/跌 -> 4类)
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2),  # 条件输入: 特征 + 方向概率
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_trend_classes)
        )

        # Level 3: 条件回归头
        # 基于趋势类别条件的幅度预测
        self.trend_embeddings = nn.Embedding(num_trend_classes, d_model // 4)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

        # 可学习的趋势划分阈值
        self.trend_thresholds = nn.Parameter(torch.tensor([-0.02, -0.005, 0.005, 0.02]))

        # 门控融合层: 决定趋势引导的影响程度
        self.gate = nn.Sequential(
            nn.Linear(d_model + num_trend_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x, trend_features=None):
        """
        Args:
            x: LLM输出特征 [B, n_vars, d_model] 或 [B*n_vars, L, d_model]
            trend_features: 趋势特征 (可选)

        Returns:
            predictions: 最终预测 [B, pred_len, n_vars]
            direction_logits: 方向分类logits
            magnitude_logits: 幅度分类logits
            trend_class: 预测的趋势类别
        """
        # 如果输入是 [B*N, L, D]，需要先池化
        if x.dim() == 3 and x.shape[1] > 1:
            # 使用注意力池化聚合序列信息
            attn_weights = F.softmax(x.mean(dim=-1), dim=-1)  # [B, L]
            x_pooled = torch.einsum('bl,bld->bd', attn_weights, x)  # [B, D]
        elif x.dim() == 3:
            x_pooled = x.squeeze(1)  # [B, D]
        else:
            x_pooled = x  # [B, D]

        # Level 1: 方向分类
        direction_logits = self.direction_head(x_pooled)  # [B, 3]
        direction_probs = F.softmax(direction_logits, dim=-1)

        # Level 2: 幅度分类 (条件于方向)
        magnitude_input = torch.cat([x_pooled, direction_probs], dim=-1)  # [B, D+3]
        magnitude_logits = self.magnitude_head(magnitude_input)  # [B, num_classes]

        # 获取预测的趋势类别
        trend_class = torch.argmax(magnitude_logits, dim=-1)  # [B]

        # Level 3: 条件回归
        trend_embed = self.trend_embeddings(trend_class)  # [B, D//4]
        regression_input = torch.cat([x_pooled, trend_embed], dim=-1)  # [B, D + D//4]
        predictions = self.regression_head(regression_input)  # [B, pred_len]

        # 门控机制: 动态调整趋势引导的影响
        gate_input = torch.cat([x_pooled, F.softmax(magnitude_logits, dim=-1)], dim=-1)
        gate_weight = self.gate(gate_input)  # [B, 1]

        return {
            'predictions': predictions,
            'direction_logits': direction_logits,
            'magnitude_logits': magnitude_logits,
            'trend_class': trend_class,
            'gate_weight': gate_weight
        }

    def compute_trend_labels(self, y_true):
        """
        根据真实值计算趋势标签

        Args:
            y_true: 真实预测值 [B, pred_len, n_vars]

        Returns:
            direction_labels: 方向标签 [B] (0=下跌, 1=平稳, 2=上涨)
            magnitude_labels: 幅度标签 [B] (0=强跌, 1=弱跌, 2=弱涨, 3=强涨)
        """
        # 计算预测期间的总变化率
        if y_true.dim() == 3:
            change_rate = (y_true[:, -1, :] - y_true[:, 0, :]).mean(dim=-1) / (y_true[:, 0, :].mean(dim=-1).abs() + 1e-6)
        else:
            change_rate = (y_true[:, -1] - y_true[:, 0]) / (y_true[:, 0].abs() + 1e-6)

        # 使用可学习阈值划分类别
        thresholds = self.trend_thresholds.detach()

        # 方向标签
        direction_labels = torch.zeros_like(change_rate, dtype=torch.long)
        direction_labels[change_rate < thresholds[1]] = 0  # 下跌
        direction_labels[(change_rate >= thresholds[1]) & (change_rate <= thresholds[2])] = 1  # 平稳
        direction_labels[change_rate > thresholds[2]] = 2  # 上涨

        # 幅度标签
        magnitude_labels = torch.zeros_like(change_rate, dtype=torch.long)
        magnitude_labels[change_rate < thresholds[0]] = 0  # 强跌
        magnitude_labels[(change_rate >= thresholds[0]) & (change_rate < thresholds[1])] = 1  # 弱跌
        magnitude_labels[(change_rate >= thresholds[1]) & (change_rate <= thresholds[2])] = 2  # 弱涨
        magnitude_labels[change_rate > thresholds[2]] = 3  # 强涨

        return direction_labels, magnitude_labels


class TAPR(nn.Module):
    """
    趋势感知尺度路由模块 (Trend-Aware Patch Router)

    整合多尺度分解和先粗后细预测头，实现:
    1. 多尺度特征提取和融合
    2. 趋势感知的层级预测
    3. 层次化一致性约束
    """

    def __init__(self, configs):
        """
        Args:
            configs: 配置对象，包含:
                - d_model: 模型维度
                - d_ff: FFN维度
                - enc_in: 变量数量
                - pred_len: 预测长度
                - n_scales: 尺度数量 (默认3)
                - dropout: Dropout比例
        """
        super(TAPR, self).__init__()

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_vars = configs.enc_in
        self.pred_len = configs.pred_len
        self.n_scales = getattr(configs, 'n_scales', 3)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 多尺度分解模块
        self.multi_scale = MultiScaleDecomposition(
            d_model=self.d_model,
            n_scales=self.n_scales,
            dropout=self.dropout
        )

        # 先粗后细预测头
        self.c2f_head = CoarseToFineHead(
            d_model=self.d_model,
            n_vars=self.n_vars,
            pred_len=self.pred_len,
            dropout=self.dropout
        )

        # 趋势嵌入投影 (用于增强Prompt)
        self.trend_proj = nn.Linear(self.d_model, self.d_model)

        # 是否启用趋势分类辅助损失
        self.use_trend_loss = True

    def forward(self, x, return_trend_info=False):
        """
        Args:
            x: Patch嵌入 [B*N, num_patches, d_model]
            return_trend_info: 是否返回趋势信息

        Returns:
            enhanced_x: 多尺度增强后的特征 [B*N, num_patches, d_model]
            trend_embed: 趋势嵌入 (可选)
            trend_info: 趋势分类信息字典 (可选)
        """
        # 1. 多尺度分解和融合
        multi_scale_out, weighted_trend, all_trends = self.multi_scale(x)

        # 2. 生成趋势嵌入 (用于增强Prompt)
        # 对趋势特征进行池化
        trend_pooled = weighted_trend.mean(dim=1)  # [B*N, d_model]
        trend_embed = self.trend_proj(trend_pooled)  # [B*N, d_model]

        if return_trend_info:
            # 3. 获取趋势分类信息 (用于辅助损失)
            c2f_output = self.c2f_head(weighted_trend)

            return multi_scale_out, trend_embed, {
                'direction_logits': c2f_output['direction_logits'],
                'magnitude_logits': c2f_output['magnitude_logits'],
                'trend_class': c2f_output['trend_class'],
                'gate_weight': c2f_output['gate_weight'],
                'all_trends': all_trends
            }

        return multi_scale_out, trend_embed, None

    def compute_auxiliary_loss(self, trend_info, y_true, lambda_dir=0.1, lambda_mag=0.1, lambda_hcl=0.05):
        """
        计算趋势分类辅助损失

        Args:
            trend_info: forward返回的趋势信息字典
            y_true: 真实预测值 [B, pred_len, n_vars]
            lambda_dir: 方向分类损失权重
            lambda_mag: 幅度分类损失权重
            lambda_hcl: 层次一致性损失权重

        Returns:
            total_loss: 总辅助损失
            loss_dict: 各项损失字典
        """
        if trend_info is None:
            return torch.tensor(0.0), {}

        direction_logits = trend_info['direction_logits']
        magnitude_logits = trend_info['magnitude_logits']

        # 计算趋势标签
        direction_labels, magnitude_labels = self.c2f_head.compute_trend_labels(y_true)

        # 1. 方向分类损失
        loss_direction = F.cross_entropy(direction_logits, direction_labels)

        # 2. 幅度分类损失
        loss_magnitude = F.cross_entropy(magnitude_logits, magnitude_labels)

        # 3. 层次一致性损失 (HCL)
        # 确保幅度分类与方向分类一致
        # 例如: 预测"强涨"时，方向应该是"上涨"
        direction_probs = F.softmax(direction_logits, dim=-1)
        magnitude_probs = F.softmax(magnitude_logits, dim=-1)

        # 方向到幅度的映射: 下跌->强跌/弱跌, 平稳->弱跌/弱涨, 上涨->弱涨/强涨
        # 使用KL散度约束
        expected_mag_from_dir = torch.zeros_like(magnitude_probs)
        expected_mag_from_dir[:, 0] = direction_probs[:, 0] * 0.7  # 下跌 -> 强跌
        expected_mag_from_dir[:, 1] = direction_probs[:, 0] * 0.3 + direction_probs[:, 1] * 0.5  # 弱跌
        expected_mag_from_dir[:, 2] = direction_probs[:, 1] * 0.5 + direction_probs[:, 2] * 0.3  # 弱涨
        expected_mag_from_dir[:, 3] = direction_probs[:, 2] * 0.7  # 上涨 -> 强涨

        loss_hcl = F.kl_div(
            F.log_softmax(magnitude_logits, dim=-1),
            expected_mag_from_dir + 1e-8,
            reduction='batchmean'
        )

        # 总损失
        total_loss = lambda_dir * loss_direction + lambda_mag * loss_magnitude + lambda_hcl * loss_hcl

        return total_loss, {
            'loss_direction': loss_direction.item(),
            'loss_magnitude': loss_magnitude.item(),
            'loss_hcl': loss_hcl.item()
        }
