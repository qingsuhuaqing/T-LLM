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

修改日志 (2026-02-08):
- 修复 compute_auxiliary_loss 中 device 不匹配问题
- 修复趋势标签计算时的维度不匹配问题
- 增加辅助损失的渐进式warmup机制防止过拟合
- 添加详细注释说明每个组件的作用
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

    设计理念:
    时间序列包含多种频率成分，单一尺度难以同时捕获。
    通过多尺度分解，可以让模型在不同分辨率上学习不同的模式。
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
        # 1x: 原始分辨率，捕获高频细节
        # 4x: 中等分辨率，捕获日内模式
        # 16x: 低分辨率，捕获长期趋势
        if kernel_sizes is None:
            kernel_sizes = [1, 4, 16]
        self.kernel_sizes = kernel_sizes[:n_scales]

        # 每个尺度的特征提取器
        # 使用深度可分离卷积减少参数量，同时保持特征提取能力
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                # 深度卷积: 每个通道独立卷积，捕获局部模式
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                # 逐点卷积: 跨通道信息融合
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.Dropout(dropout)
            ) for _ in range(n_scales)
        ])

        # 趋势-季节性分解模块
        # 每个尺度使用不同大小的移动平均窗口
        # 较大的尺度使用较大的窗口，捕获更平滑的趋势
        self.decomp_layers = nn.ModuleList([
            SeriesDecomposition(kernel_size=max(3, ks // 2 * 2 + 1))
            for ks in self.kernel_sizes
        ])

        # 跨尺度融合层 (从粗到细)
        # 使用注意力机制让粗尺度的全局信息指导细尺度的局部预测
        self.cross_scale_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
            for _ in range(n_scales - 1)
        ])

        # 尺度自适应权重 - 可学习的权重决定各尺度的重要性
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # 输出投影 - 将多尺度特征融合为单一表示
        self.output_proj = nn.Linear(d_model * n_scales, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [B*N, num_patches, d_model]
               B*N: batch_size * n_variables (通道独立处理)
               num_patches: patch数量
               d_model: 特征维度

        Returns:
            multi_scale_out: 多尺度融合后的特征 [B*N, num_patches, d_model]
            weighted_trend: 加权趋势分量 (用于趋势分类)
            trend_features: 各尺度的趋势分量列表
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
            # 使用平均池化实现平滑下采样，保留主要趋势信息
            if ks > 1:
                x_scale = F.avg_pool1d(x_conv, kernel_size=ks, stride=ks,
                                       padding=ks//2 if L % ks != 0 else 0)
            else:
                x_scale = x_conv

            # 2. 特征提取 - 在当前尺度上提取局部模式
            x_scale = encoder(x_scale)

            # 3. 转换回 [B, L_scale, D]
            x_scale = x_scale.permute(0, 2, 1)

            # 4. 趋势-季节性分解
            # 将序列分解为平滑趋势和高频季节性成分
            trend, seasonal = decomp(x_scale)

            # 5. 上采样回原始长度
            # 使用线性插值保持平滑性
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
        # 核心思想: 粗尺度包含全局趋势信息，可以指导细尺度的局部预测
        for i in range(self.n_scales - 2, -1, -1):
            if i < len(self.cross_scale_attn):
                # 使用粗尺度作为Key/Value，细尺度作为Query
                # 这样细尺度可以"查询"粗尺度的全局信息
                coarse = scale_features[i + 1] if i + 1 < len(scale_features) else scale_features[i]
                fine = scale_features[i]
                attn_out, _ = self.cross_scale_attn[i](fine, coarse, coarse)
                # 残差连接保留原始细尺度信息
                scale_features[i] = fine + attn_out

        # 7. 自适应权重融合
        # 使用softmax确保权重和为1
        weights = F.softmax(self.scale_weights, dim=0)

        # 拼接所有尺度特征
        multi_scale_out = torch.cat(scale_features, dim=-1)  # [B, L, D*n_scales]
        multi_scale_out = self.output_proj(multi_scale_out)  # [B, L, D]

        # 加权趋势特征 (用于趋势分类)
        weighted_trend = sum(w * t for w, t in zip(weights, trend_features))

        # 残差连接 + dropout
        return self.dropout(multi_scale_out + x), weighted_trend, trend_features


class SeriesDecomposition(nn.Module):
    """
    季节性-趋势分解模块

    使用移动平均提取趋势，残差作为季节性成分
    这是时间序列分析中经典的STL分解的简化版本

    优点:
    - 计算效率高，无需学习参数
    - 物理意义明确，趋势=平滑成分，季节性=高频成分
    """
    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        # 使用平均池化实现移动平均
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            trend: 趋势成分 [B, L, D]
            seasonal: 季节性成分 [B, L, D]
        """
        # 转换为 [B, D, L] 用于池化
        x_t = x.permute(0, 2, 1)

        # 移动平均提取趋势
        trend = self.avg(x_t)

        # 处理边界 - 确保输出长度与输入一致
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

    设计理念:
    1. 人类预测时也是先判断大方向，再细化具体数值
    2. 层级约束可以减少错误传播，提高预测稳定性
    3. 通过层次化一致性损失(HCL)强制细粒度预测符合粗粒度趋势判断
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
        # 这是最粗粒度的判断，决定整体趋势方向
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 上涨/下跌/平稳
        )

        # Level 2: 幅度分类器 (强/弱 * 涨/跌 -> 4类)
        # 在方向基础上进一步细分幅度
        # 输入包含方向概率，实现条件建模
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
        # 这些阈值会在训练中自动调整，适应不同数据集的分布
        self.trend_thresholds = nn.Parameter(torch.tensor([-0.02, -0.005, 0.005, 0.02]))

        # 门控融合层: 决定趋势引导的影响程度
        # 输出接近1表示强趋势，应该更相信趋势分类
        # 输出接近0表示弱趋势或不确定，应该更依赖回归
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
            dict: 包含predictions, direction_logits, magnitude_logits, trend_class, gate_weight
        """
        # 如果输入是 [B*N, L, D]，需要先池化
        if x.dim() == 3 and x.shape[1] > 1:
            # 使用注意力池化聚合序列信息
            # 这比简单平均更能关注重要时间点
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
        # 将方向概率作为额外输入，实现条件建模
        magnitude_input = torch.cat([x_pooled, direction_probs], dim=-1)  # [B, D+3]
        magnitude_logits = self.magnitude_head(magnitude_input)  # [B, num_classes]

        # 获取预测的趋势类别
        trend_class = torch.argmax(magnitude_logits, dim=-1)  # [B]

        # Level 3: 条件回归
        # 使用趋势类别的嵌入作为条件信息
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

        修复说明:
        - 支持多种输入维度 [B, pred_len], [B, pred_len, 1], [B*N, pred_len]
        - 使用相对变化率而非绝对值，适应不同量级的数据

        Args:
            y_true: 真实预测值，支持多种形状

        Returns:
            direction_labels: 方向标签 [B] (0=下跌, 1=平稳, 2=上涨)
            magnitude_labels: 幅度标签 [B] (0=强跌, 1=弱跌, 2=弱涨, 3=强涨)
        """
        # 统一处理输入维度
        if y_true.dim() == 3:
            # [B, pred_len, n_vars] -> 取所有变量的平均
            change_rate = (y_true[:, -1, :] - y_true[:, 0, :]).mean(dim=-1)
            base_value = y_true[:, 0, :].mean(dim=-1).abs() + 1e-6
            change_rate = change_rate / base_value
        elif y_true.dim() == 2:
            # [B, pred_len] 或 [B*N, pred_len]
            change_rate = (y_true[:, -1] - y_true[:, 0])
            base_value = y_true[:, 0].abs() + 1e-6
            change_rate = change_rate / base_value
        else:
            # 标量情况，返回平稳
            batch_size = y_true.shape[0] if y_true.dim() > 0 else 1
            return (torch.ones(batch_size, dtype=torch.long, device=y_true.device),
                    torch.ones(batch_size, dtype=torch.long, device=y_true.device) * 2)

        # 使用可学习阈值划分类别
        # detach() 确保阈值不参与这个分支的梯度计算
        thresholds = self.trend_thresholds.detach()

        # 方向标签: 0=下跌, 1=平稳, 2=上涨
        direction_labels = torch.ones_like(change_rate, dtype=torch.long)  # 默认平稳
        direction_labels[change_rate < thresholds[1]] = 0  # 下跌
        direction_labels[change_rate > thresholds[2]] = 2  # 上涨

        # 幅度标签: 0=强跌, 1=弱跌, 2=弱涨, 3=强涨
        magnitude_labels = torch.ones_like(change_rate, dtype=torch.long)  # 默认弱跌
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

    创新点:
    - 多尺度分解捕获不同时间粒度的模式
    - C2F头实现层级决策，提高预测稳定性
    - 趋势嵌入增强Prompt，让LLM更好理解时序趋势
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

        # 训练步数计数器，用于warmup
        self.register_buffer('train_step', torch.tensor(0))

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
        # 对趋势特征进行池化，得到全局趋势表示
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

    def compute_auxiliary_loss(self, trend_info, y_true, lambda_dir=0.3, lambda_mag=0.3, lambda_hcl=0.1,
                                warmup_steps=500, current_step=None):
        """
        计算趋势分类辅助损失

        修复说明:
        1. 添加 device 参数确保返回值在正确设备上
        2. 修复维度匹配问题，正确处理 [B, pred_len, n_vars] 输入
        3. 添加 warmup 机制，防止训练初期辅助损失干扰主任务
        4. 使用 label smoothing 防止过拟合

        Args:
            trend_info: forward返回的趋势信息字典
            y_true: 真实预测值 [B, pred_len, n_vars]
            lambda_dir: 方向分类损失权重 (默认0.3)
            lambda_mag: 幅度分类损失权重 (默认0.3)
            lambda_hcl: 层次一致性损失权重 (默认0.1)
            warmup_steps: warmup步数 (默认500)
            current_step: 当前训练步数

        Returns:
            total_loss: 总辅助损失
            loss_dict: 各项损失字典
        """
        # 获取设备
        device = y_true.device

        if trend_info is None:
            return torch.tensor(0.0, device=device), {}

        direction_logits = trend_info['direction_logits']  # [B*N, 3]
        magnitude_logits = trend_info['magnitude_logits']  # [B*N, 4]

        # ========== 维度匹配修复 ==========
        B_N = direction_logits.shape[0]  # B * N

        # y_true: [B, pred_len, n_vars] -> 需要reshape为 [B*N, pred_len]
        if y_true.dim() == 3:
            B, P, N = y_true.shape
            # 将每个变量独立处理
            y_true_reshaped = y_true.permute(0, 2, 1).reshape(B * N, P)  # [B*N, pred_len]
        elif y_true.dim() == 2:
            y_true_reshaped = y_true
        else:
            return torch.tensor(0.0, device=device), {}

        # 确保维度匹配
        min_size = min(y_true_reshaped.shape[0], B_N)
        y_true_reshaped = y_true_reshaped[:min_size]
        direction_logits = direction_logits[:min_size]
        magnitude_logits = magnitude_logits[:min_size]

        # 计算趋势标签
        direction_labels, magnitude_labels = self.c2f_head.compute_trend_labels(y_true_reshaped)

        # 确保标签在正确的设备上
        direction_labels = direction_labels.to(device)
        magnitude_labels = magnitude_labels.to(device)

        # ========== Label Smoothing 防止过拟合 ==========
        label_smoothing = 0.1

        # 1. 方向分类损失
        loss_direction = F.cross_entropy(
            direction_logits, direction_labels,
            label_smoothing=label_smoothing
        )

        # 2. 幅度分类损失
        loss_magnitude = F.cross_entropy(
            magnitude_logits, magnitude_labels,
            label_smoothing=label_smoothing
        )

        # 3. 层次一致性损失 (HCL)
        # 确保幅度分类与方向分类一致
        direction_probs = F.softmax(direction_logits, dim=-1)
        magnitude_probs = F.softmax(magnitude_logits, dim=-1)

        # 方向到幅度的映射矩阵
        # 下跌 -> 强跌/弱跌, 平稳 -> 弱跌/弱涨, 上涨 -> 弱涨/强涨
        expected_mag_from_dir = torch.zeros_like(magnitude_probs)
        expected_mag_from_dir[:, 0] = direction_probs[:, 0] * 0.7  # 下跌 -> 强跌
        expected_mag_from_dir[:, 1] = direction_probs[:, 0] * 0.3 + direction_probs[:, 1] * 0.5  # 弱跌
        expected_mag_from_dir[:, 2] = direction_probs[:, 1] * 0.5 + direction_probs[:, 2] * 0.3  # 弱涨
        expected_mag_from_dir[:, 3] = direction_probs[:, 2] * 0.7  # 上涨 -> 强涨

        # KL散度约束
        loss_hcl = F.kl_div(
            F.log_softmax(magnitude_logits, dim=-1),
            expected_mag_from_dir + 1e-8,
            reduction='batchmean'
        )

        # ========== Warmup 机制 ==========
        # 训练初期降低辅助损失权重，让模型先学习主任务
        if current_step is not None:
            warmup_factor = min(1.0, current_step / warmup_steps)
        else:
            # 使用内部计数器
            self.train_step += 1
            warmup_factor = min(1.0, self.train_step.item() / warmup_steps)

        # 计算总损失（带warmup）
        total_loss = warmup_factor * (
            lambda_dir * loss_direction +
            lambda_mag * loss_magnitude +
            lambda_hcl * loss_hcl
        )

        return total_loss, {
            'loss_direction': loss_direction.item(),
            'loss_magnitude': loss_magnitude.item(),
            'loss_hcl': loss_hcl.item(),
            'warmup_factor': warmup_factor
        }
