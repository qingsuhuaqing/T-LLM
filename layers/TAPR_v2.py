"""
TAPR_v2 (Trend-Aware Patch Router) - 趋势感知尺度路由模块 (极简版)

核心设计原则 (学术裁缝思想):
1. 最小干扰原则: 新增模块不能影响原模型性能
2. 条件触发原则: 仅在明确有益时才介入，否则完全透明
3. 可消融原则: 每个子模块可独立开关

改进要点:
- 多尺度分解仅作为辅助信息提取，不修改主数据流
- C2F趋势分类仅在相邻层预测不一致时才触发调整
- 辅助损失权重极低，仅作为正则化
- 保持原模型的完整数据流

修改日志 (2026-02-10):
- 重新设计为"观察者模式"，最小化对主流程的干扰
- 新增趋势冲突检测机制
- 辅助损失降至极低权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MultiScaleObserver(nn.Module):
    """
    多尺度观察器 (Multi-Scale Observer)

    核心思想: 观察而非修改
    - 从输入序列中提取多尺度特征作为辅助信息
    - 生成趋势嵌入供后续模块参考
    - 不修改原始数据流，保持重编程机制的完整性

    与原版区别:
    - 原版: enc_out = multi_scale(enc_out) + enc_out (直接修改)
    - 新版: trend_embed = observe(enc_out) (仅观察提取)
    """

    def __init__(self, d_model, n_scales=3, dropout=0.1):
        """
        Args:
            d_model: 嵌入维度
            n_scales: 观察尺度数量 (默认3: 原始/4x/16x)
            dropout: Dropout比例
        """
        super(MultiScaleObserver, self).__init__()

        self.n_scales = n_scales
        self.d_model = d_model

        # 多尺度下采样核大小
        self.kernel_sizes = [1, 4, 16][:n_scales]

        # 轻量级特征提取器 (仅用于观察，不参与主流程)
        # 使用分组卷积减少参数量
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=max(1, d_model // 4)),
                nn.GELU(),
            ) for _ in range(n_scales)
        ])

        # 趋势提取: 简单的移动平均
        self.trend_kernels = nn.ModuleList([
            nn.AvgPool1d(kernel_size=max(3, ks), stride=1, padding=max(3, ks) // 2)
            for ks in self.kernel_sizes
        ])

        # 尺度融合: 简单线性组合
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # 趋势方向检测器
        self.trend_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3)  # 上涨/平稳/下跌
        )

    def forward(self, x):
        """
        观察输入序列，提取多尺度趋势信息

        Args:
            x: 输入 [B*N, num_patches, d_model]

        Returns:
            trend_embed: 趋势嵌入 [B*N, d_model]
            scale_trends: 各尺度趋势方向 [B*N, n_scales, 3]
        """
        B, L, D = x.shape

        # 转换为 [B, D, L] 用于卷积
        x_conv = x.permute(0, 2, 1)

        scale_features = []
        scale_trends = []

        for i, (proj, trend_pool, ks) in enumerate(
            zip(self.scale_projectors, self.trend_kernels, self.kernel_sizes)
        ):
            # 1. 下采样到当前尺度
            if ks > 1 and L >= ks:
                x_scale = F.avg_pool1d(x_conv, kernel_size=ks, stride=ks,
                                       padding=0 if L % ks == 0 else ks // 2)
            else:
                x_scale = x_conv

            # 2. 特征提取
            feat = proj(x_scale)  # [B, D, L_scale]

            # 3. 提取趋势 (移动平均)
            if feat.shape[-1] >= 3:
                trend = trend_pool(feat)
                # 处理边界
                if trend.shape[-1] != feat.shape[-1]:
                    trend = F.interpolate(trend, size=feat.shape[-1], mode='linear', align_corners=False)
            else:
                trend = feat

            # 4. 上采样回原长度
            if ks > 1 and feat.shape[-1] != L:
                feat = F.interpolate(feat, size=L, mode='linear', align_corners=False)
                trend = F.interpolate(trend, size=L, mode='linear', align_corners=False)

            scale_features.append(feat.permute(0, 2, 1))  # [B, L, D]

            # 5. 检测该尺度的趋势方向
            trend_pooled = trend.mean(dim=-1)  # [B, D]
            trend_direction = self.trend_detector(trend_pooled)  # [B, 3]
            scale_trends.append(trend_direction)

        # 6. 加权融合各尺度特征生成趋势嵌入
        weights = F.softmax(self.scale_weights, dim=0)

        # 池化得到全局表示
        trend_embed = sum(
            w * f.mean(dim=1)  # [B, D]
            for w, f in zip(weights, scale_features)
        )

        # 堆叠各尺度趋势方向
        scale_trends = torch.stack(scale_trends, dim=1)  # [B, n_scales, 3]

        return trend_embed, scale_trends


class ConflictAwareFusion(nn.Module):
    """
    冲突感知融合模块 (Conflict-Aware Fusion)

    核心思想: 仅在检测到趋势冲突时介入
    - 当粗尺度和细尺度的趋势预测一致时，完全信任原模型
    - 仅当检测到明显冲突时，才进行微调

    这体现了"学术裁缝"思想：
    - 新模块是"补丁"，不是"替换"
    - 只在原模型可能出错时才介入
    """

    def __init__(self, d_model, pred_len, n_scales=3, conflict_threshold=0.5):
        """
        Args:
            d_model: 特征维度
            pred_len: 预测长度
            n_scales: 尺度数量
            conflict_threshold: 冲突阈值 (0-1)，越高越不容易触发调整
        """
        super(ConflictAwareFusion, self).__init__()

        self.d_model = d_model
        self.pred_len = pred_len
        self.n_scales = n_scales
        self.conflict_threshold = conflict_threshold

        # 冲突检测器: 判断各尺度趋势是否一致
        self.conflict_detector = nn.Sequential(
            nn.Linear(n_scales * 3, 16),  # 输入: 各尺度的趋势概率
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 微调层: 仅在冲突时使用，调整幅度极小
        self.adjustment_layer = nn.Sequential(
            nn.Linear(d_model + 3, 16),  # 趋势嵌入 + 主导趋势方向
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Tanh()  # 输出 [-1, 1]，作为调整系数
        )

        # 调整幅度控制 (初始化为极小值)
        self.adjustment_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, predictions, trend_embed, scale_trends):
        """
        根据趋势冲突情况决定是否调整预测

        Args:
            predictions: 原模型预测 [B, pred_len, N]
            trend_embed: 趋势嵌入 [B*N, d_model]
            scale_trends: 各尺度趋势 [B*N, n_scales, 3]

        Returns:
            adjusted_predictions: 调整后的预测 [B, pred_len, N]
            conflict_info: 冲突检测信息
        """
        B, pred_len, N = predictions.shape
        BN = trend_embed.shape[0]

        # 检查维度匹配
        if BN != B * N:
            return predictions, {'conflict_score': 0.0, 'adjustment_applied': False}

        # 1. 检测各尺度间的趋势冲突
        scale_trends_flat = scale_trends.reshape(BN, -1)  # [B*N, n_scales*3]
        conflict_score = self.conflict_detector(scale_trends_flat)  # [B*N, 1]

        # 2. 判断是否需要调整
        # 只有当冲突分数超过阈值时才调整
        need_adjustment = (conflict_score > self.conflict_threshold).float()

        # 如果没有任何样本需要调整，直接返回
        if need_adjustment.sum() < 0.5:
            return predictions, {
                'conflict_score': conflict_score.mean().item(),
                'adjustment_applied': False,
                'num_adjusted': 0
            }

        # 3. 计算主导趋势方向 (取粗尺度的预测)
        coarse_trend = scale_trends[:, -1, :]  # [B*N, 3] 最粗尺度

        # 4. 计算调整系数
        adjustment_input = torch.cat([trend_embed, coarse_trend], dim=-1)  # [B*N, d_model+3]
        adjustment_factor = self.adjustment_layer(adjustment_input)  # [B*N, 1]

        # 5. 应用调整 (乘以冲突触发掩码)
        # 调整幅度 = adjustment_factor * adjustment_scale * need_adjustment
        effective_adjustment = adjustment_factor * self.adjustment_scale * need_adjustment

        # 重塑为 [B, N]
        effective_adjustment = effective_adjustment.reshape(B, N)

        # 6. 渐进式调整 (预测后期调整更大)
        time_weights = torch.linspace(0.5, 1.0, pred_len, device=predictions.device)
        time_weights = time_weights.view(1, pred_len, 1)

        # 调整预测
        adjustment = effective_adjustment.unsqueeze(1) * time_weights  # [B, pred_len, N]
        adjusted_predictions = predictions * (1 + adjustment)

        return adjusted_predictions, {
            'conflict_score': conflict_score.mean().item(),
            'adjustment_applied': True,
            'num_adjusted': need_adjustment.sum().item(),
            'mean_adjustment': effective_adjustment.abs().mean().item()
        }


class TrendClassifier(nn.Module):
    """
    趋势分类器 (Trend Classifier)

    核心功能:
    - 将连续预测问题辅助为离散分类问题
    - 提供粗粒度和细粒度的趋势判断
    - 仅作为辅助任务，不影响主预测流程

    层级结构:
    - Level 1: 方向分类 (上涨/平稳/下跌)
    - Level 2: 幅度分类 (强涨/弱涨/弱跌/强跌)
    """

    def __init__(self, d_model, dropout=0.1):
        super(TrendClassifier, self).__init__()

        self.d_model = d_model

        # Level 1: 方向分类
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 3)
        )

        # Level 2: 幅度分类 (条件于方向)
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 4)
        )

        # 可学习阈值
        self.register_buffer('trend_thresholds',
                            torch.tensor([-0.02, -0.005, 0.005, 0.02]))

    def forward(self, trend_embed):
        """
        Args:
            trend_embed: 趋势嵌入 [B*N, d_model]

        Returns:
            direction_logits: 方向分类 [B*N, 3]
            magnitude_logits: 幅度分类 [B*N, 4]
        """
        # Level 1
        direction_logits = self.direction_head(trend_embed)
        direction_probs = F.softmax(direction_logits, dim=-1)

        # Level 2 (条件于方向)
        magnitude_input = torch.cat([trend_embed, direction_probs], dim=-1)
        magnitude_logits = self.magnitude_head(magnitude_input)

        return direction_logits, magnitude_logits

    def compute_labels(self, y_true):
        """
        根据真实值计算趋势标签

        Args:
            y_true: [B, pred_len, N] 或 [B*N, pred_len]

        Returns:
            direction_labels, magnitude_labels
        """
        # 统一维度处理
        if y_true.dim() == 3:
            B, P, N = y_true.shape
            y_flat = y_true.permute(0, 2, 1).reshape(B * N, P)
        else:
            y_flat = y_true

        # 计算变化率
        change_rate = (y_flat[:, -1] - y_flat[:, 0]) / (y_flat[:, 0].abs() + 1e-6)

        device = y_true.device
        thresholds = self.trend_thresholds

        # 方向标签: 0=下跌, 1=平稳, 2=上涨
        direction = torch.ones_like(change_rate, dtype=torch.long)
        direction[change_rate < thresholds[1]] = 0
        direction[change_rate > thresholds[2]] = 2

        # 幅度标签: 0=强跌, 1=弱跌, 2=弱涨, 3=强涨
        magnitude = torch.ones_like(change_rate, dtype=torch.long)
        magnitude[change_rate < thresholds[0]] = 0
        magnitude[(change_rate >= thresholds[0]) & (change_rate < thresholds[1])] = 1
        magnitude[(change_rate >= thresholds[1]) & (change_rate <= thresholds[2])] = 2
        magnitude[change_rate > thresholds[2]] = 3

        return direction, magnitude


class TAPR_v2(nn.Module):
    """
    TAPR_v2 (Trend-Aware Patch Router) - 极简版趋势感知模块

    设计原则:
    1. 观察而非修改: 多尺度模块仅观察输入，不修改数据流
    2. 冲突时才介入: 仅在检测到趋势冲突时进行微调
    3. 最小辅助损失: 趋势分类损失权重极低，仅作正则化

    与原TAPR的关键区别:
    - 原版: enc_out = multi_scale(enc_out) + enc_out (直接修改)
    - 新版: trend_info = observe(enc_out), 主流程不变
    """

    def __init__(self, configs):
        super(TAPR_v2, self).__init__()

        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.n_scales = getattr(configs, 'n_scales', 3)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 极低的辅助损失权重 (不影响主任务)
        self.default_lambda_trend = 0.01  # 从0.5降到0.01

        # 多尺度观察器 (仅观察，不修改)
        self.observer = MultiScaleObserver(
            d_model=self.d_model,
            n_scales=self.n_scales,
            dropout=self.dropout
        )

        # 趋势分类器 (辅助任务)
        self.classifier = TrendClassifier(
            d_model=self.d_model,
            dropout=self.dropout
        )

        # 冲突感知融合 (仅在输出阶段可选使用)
        self.fusion = ConflictAwareFusion(
            d_model=self.d_model,
            pred_len=self.pred_len,
            n_scales=self.n_scales,
            conflict_threshold=0.7  # 高阈值，不轻易触发
        )

        # 趋势嵌入投影 (用于prompt增强)
        self.trend_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, return_trend_info=False):
        """
        前向传播 - 仅观察，不修改输入

        Args:
            x: Patch嵌入 [B*N, num_patches, d_model]
            return_trend_info: 是否返回趋势信息

        Returns:
            x: 原样返回输入 (不修改!)
            trend_embed: 趋势嵌入 [B*N, d_model]
            trend_info: 趋势分类信息 (可选)
        """
        # 1. 观察多尺度趋势 (不修改x)
        trend_embed, scale_trends = self.observer(x)

        # 2. 投影趋势嵌入
        trend_embed_proj = self.trend_proj(trend_embed)

        if return_trend_info:
            # 3. 趋势分类 (辅助任务)
            direction_logits, magnitude_logits = self.classifier(trend_embed)

            return x, trend_embed_proj, {
                'direction_logits': direction_logits,
                'magnitude_logits': magnitude_logits,
                'scale_trends': scale_trends,
                'trend_embed': trend_embed
            }

        # 不返回趋势信息时，原样返回输入
        return x, trend_embed_proj, None

    def apply_output_fusion(self, predictions, trend_info):
        """
        在输出阶段应用冲突感知融合 (可选)

        Args:
            predictions: 原模型预测 [B, pred_len, N]
            trend_info: forward返回的趋势信息

        Returns:
            predictions: 调整后的预测
            fusion_info: 融合信息
        """
        if trend_info is None:
            return predictions, {'applied': False}

        trend_embed = trend_info.get('trend_embed')
        scale_trends = trend_info.get('scale_trends')

        if trend_embed is None or scale_trends is None:
            return predictions, {'applied': False}

        return self.fusion(predictions, trend_embed, scale_trends)

    def compute_auxiliary_loss(self, trend_info, y_true,
                                lambda_dir=0.01, lambda_mag=0.01,
                                warmup_steps=1000, current_step=None):
        """
        计算趋势分类辅助损失 (极低权重)

        Args:
            trend_info: forward返回的趋势信息
            y_true: 真实值 [B, pred_len, N]
            lambda_dir: 方向损失权重 (默认0.01，极低)
            lambda_mag: 幅度损失权重 (默认0.01，极低)
            warmup_steps: warmup步数 (较长，避免早期干扰)
            current_step: 当前步数

        Returns:
            total_loss, loss_dict
        """
        device = y_true.device

        if trend_info is None:
            return torch.tensor(0.0, device=device), {}

        direction_logits = trend_info.get('direction_logits')
        magnitude_logits = trend_info.get('magnitude_logits')

        if direction_logits is None:
            return torch.tensor(0.0, device=device), {}

        # 计算真实标签
        direction_labels, magnitude_labels = self.classifier.compute_labels(y_true)

        # 维度对齐
        BN = direction_logits.shape[0]
        direction_labels = direction_labels[:BN].to(device)
        magnitude_labels = magnitude_labels[:BN].to(device)

        # 使用 label smoothing 防止过拟合
        loss_dir = F.cross_entropy(direction_logits, direction_labels, label_smoothing=0.2)
        loss_mag = F.cross_entropy(magnitude_logits, magnitude_labels, label_smoothing=0.2)

        # Warmup机制 (较长的warmup避免早期干扰)
        if current_step is not None:
            warmup_factor = min(1.0, current_step / warmup_steps)
        else:
            warmup_factor = 1.0

        total_loss = warmup_factor * (lambda_dir * loss_dir + lambda_mag * loss_mag)

        return total_loss, {
            'loss_direction': loss_dir.item(),
            'loss_magnitude': loss_mag.item(),
            'warmup_factor': warmup_factor
        }
