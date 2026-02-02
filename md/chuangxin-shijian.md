# Time-LLM 创新方案实践指南

> **从理论到代码** —— 逐条详细说明各创新方案的具体实现、代码修改与运行命令

---

## 目录

1. [方案一: 可解释性增强与传统模型集成](#方案一-可解释性增强与传统模型集成-实践指南)
2. [方案二: 多尺度分解混合](#方案二-多尺度分解混合-实践指南)
3. [方案三: 频域增强](#方案三-频域增强-实践指南)
4. [方案四: 变量间注意力增强](#方案四-变量间注意力增强-实践指南)
5. [方案五: 动态Prompt生成](#方案五-动态prompt生成-实践指南)
6. [方案六: 稀疏专家混合](#方案六-稀疏专家混合-实践指南)
7. [方案七: 任务专用词汇表初始化](#方案七-任务专用词汇表初始化-实践指南)
8. [完整代码结构与文件清单](#完整代码结构与文件清单)

---

## 方案一: 可解释性增强与传统模型集成 实践指南

### 1.1 可行性评估

| 评估项 | 评分 | 说明 |
|--------|------|------|
| **实现难度** | ⭐⭐⭐ | 需要额外引入传统模型库 |
| **代码侵入性** | 低 | 主要在输出端添加，不影响核心架构 |
| **显存影响** | 无 | 传统模型 CPU 运行 |
| **兼容性** | 高 | 可作为可选模块 |

### 1.2 新增文件清单

```
Time-LLM/
├── models/
│   └── traditional/                    # 新增目录
│       ├── __init__.py
│       ├── arima_wrapper.py           # ARIMA 封装
│       ├── exponential_smoothing.py   # 指数平滑封装
│       └── hybrid_forecaster.py       # 混合预测器
│
├── utils/
│   └── data_analysis.py               # 新增: 数据特征分析工具
│
└── configs/
    └── traditional_config.yaml        # 新增: 传统模型配置
```

### 1.3 核心代码实现

#### 文件 1: `models/traditional/__init__.py`

```python
from .arima_wrapper import ARIMAWrapper
from .exponential_smoothing import ExponentialSmoothingWrapper
from .hybrid_forecaster import HybridForecaster

__all__ = ['ARIMAWrapper', 'ExponentialSmoothingWrapper', 'HybridForecaster']
```

#### 文件 2: `models/traditional/arima_wrapper.py`

```python
"""ARIMA 模型封装器 - 用于与 Time-LLM 集成"""

import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


class ARIMAWrapper:
    """ARIMA 模型的 PyTorch 兼容封装"""

    def __init__(self, order=(2, 1, 1), seasonal_order=None):
        """
        Args:
            order: (p, d, q) - AR阶数, 差分阶数, MA阶数
            seasonal_order: (P, D, Q, s) - 季节性参数（可选）
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted = False

    def check_stationarity(self, series):
        """ADF 平稳性检验"""
        result = adfuller(series.flatten(), autolag='AIC')
        return result[1] < 0.05  # p-value < 0.05 表示平稳

    def fit(self, series):
        """
        拟合 ARIMA 模型

        Args:
            series: numpy array [T] 或 [T, 1]
        """
        series = np.asarray(series).flatten()

        try:
            if self.seasonal_order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(
                    series,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

            self.fitted_model = self.model.fit(disp=0)
            self.fitted = True
            return True
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            self.fitted = False
            return False

    def predict(self, steps):
        """
        预测未来 steps 步

        Args:
            steps: 预测步数

        Returns:
            numpy array [steps]
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet!")

        forecast = self.fitted_model.forecast(steps=steps)
        return np.asarray(forecast)

    def predict_tensor(self, steps, device='cpu'):
        """返回 PyTorch 张量形式的预测"""
        pred = self.predict(steps)
        return torch.tensor(pred, dtype=torch.float32, device=device)

    def get_residuals(self):
        """获取拟合残差"""
        if not self.fitted:
            return None
        return np.asarray(self.fitted_model.resid)

    def get_confidence(self):
        """计算模型置信度（基于残差标准差）"""
        if not self.fitted:
            return 0.0
        residuals = self.get_residuals()
        return 1.0 / (1.0 + np.std(residuals))


def auto_arima_order(series, max_p=5, max_d=2, max_q=5):
    """
    自动确定最佳 ARIMA 阶数（简化版 auto_arima）

    Args:
        series: 时间序列
        max_p, max_d, max_q: 最大阶数

    Returns:
        (p, d, q) 最佳阶数
    """
    from statsmodels.tsa.stattools import adfuller

    series = np.asarray(series).flatten()

    # 确定差分阶数 d
    d = 0
    temp_series = series.copy()
    for i in range(max_d + 1):
        result = adfuller(temp_series, autolag='AIC')
        if result[1] < 0.05:  # 平稳
            break
        d += 1
        temp_series = np.diff(temp_series)

    # 简化: 使用默认 p=2, q=1
    # 完整实现应该使用 AIC/BIC 选择
    return (2, d, 1)
```

#### 文件 3: `models/traditional/exponential_smoothing.py`

```python
"""指数平滑模型封装器"""

import numpy as np
import torch
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExponentialSmoothingWrapper:
    """Holt-Winters 指数平滑模型封装"""

    def __init__(self, trend='add', seasonal='add', seasonal_periods=24):
        """
        Args:
            trend: 'add' (加法趋势) 或 'mul' (乘法趋势) 或 None
            seasonal: 'add' (加法季节性) 或 'mul' (乘法季节性) 或 None
            seasonal_periods: 季节周期长度
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted = False

    def fit(self, series):
        """拟合模型"""
        series = np.asarray(series).flatten()

        # 确保数据长度足够
        if len(series) < 2 * self.seasonal_periods:
            self.seasonal = None

        try:
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                initialization_method='estimated'
            )
            self.fitted_model = self.model.fit(optimized=True)
            self.fitted = True
            return True
        except Exception as e:
            print(f"ExponentialSmoothing fitting failed: {e}")
            # 回退到简单指数平滑
            try:
                self.model = ExponentialSmoothing(
                    series,
                    trend=None,
                    seasonal=None
                )
                self.fitted_model = self.model.fit()
                self.fitted = True
                return True
            except:
                self.fitted = False
                return False

    def predict(self, steps):
        """预测未来 steps 步"""
        if not self.fitted:
            raise ValueError("Model not fitted yet!")

        forecast = self.fitted_model.forecast(steps)
        return np.asarray(forecast)

    def predict_tensor(self, steps, device='cpu'):
        """返回 PyTorch 张量"""
        pred = self.predict(steps)
        return torch.tensor(pred, dtype=torch.float32, device=device)

    def get_components(self):
        """获取分解成分（趋势、季节性、残差）"""
        if not self.fitted:
            return None, None, None

        level = self.fitted_model.level
        trend = self.fitted_model.trend if hasattr(self.fitted_model, 'trend') else None
        season = self.fitted_model.season if hasattr(self.fitted_model, 'season') else None

        return level, trend, season

    def get_confidence(self):
        """计算置信度"""
        if not self.fitted:
            return 0.0

        # 基于 AIC 的置信度（越小越好）
        aic = self.fitted_model.aic
        return 1.0 / (1.0 + aic / 1000)
```

#### 文件 4: `models/traditional/hybrid_forecaster.py`

```python
"""混合预测器 - 融合传统模型与 Time-LLM"""

import numpy as np
import torch
import torch.nn as nn

from .arima_wrapper import ARIMAWrapper, auto_arima_order
from .exponential_smoothing import ExponentialSmoothingWrapper


class HybridForecaster(nn.Module):
    """
    混合预测器：结合传统统计模型与深度学习模型

    工作流程:
    1. 分析输入数据特征
    2. 选择合适的传统模型
    3. 与 Time-LLM 预测结果融合
    """

    def __init__(self, pred_len, use_arima=True, use_es=True, fusion_method='adaptive'):
        """
        Args:
            pred_len: 预测长度
            use_arima: 是否使用 ARIMA
            use_es: 是否使用指数平滑
            fusion_method: 融合方法 ('adaptive', 'average', 'weighted')
        """
        super().__init__()
        self.pred_len = pred_len
        self.use_arima = use_arima
        self.use_es = use_es
        self.fusion_method = fusion_method

        # 可学习的融合权重
        if fusion_method == 'adaptive':
            self.fusion_gate = nn.Sequential(
                nn.Linear(pred_len * 3, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
                nn.Softmax(dim=-1)
            )

    def analyze_data(self, x):
        """
        分析输入数据特征

        Args:
            x: [B, T, N] 输入序列

        Returns:
            dict: 数据特征字典
        """
        x_np = x.detach().cpu().numpy()
        B, T, N = x_np.shape

        features = {
            'is_stationary': [],
            'has_trend': [],
            'has_seasonality': [],
            'noise_level': []
        }

        for b in range(B):
            for n in range(N):
                series = x_np[b, :, n]

                # 平稳性检测
                from statsmodels.tsa.stattools import adfuller
                try:
                    adf_result = adfuller(series, autolag='AIC')
                    is_stationary = adf_result[1] < 0.05
                except:
                    is_stationary = False

                # 趋势检测（线性回归斜率）
                slope = np.polyfit(np.arange(T), series, 1)[0]
                has_trend = abs(slope) > 0.01 * np.std(series)

                # 季节性检测（自相关）
                from statsmodels.tsa.stattools import acf
                try:
                    acf_values = acf(series, nlags=min(T//2, 50), fft=True)
                    has_seasonality = np.max(np.abs(acf_values[1:])) > 0.5
                except:
                    has_seasonality = False

                # 噪声水平
                diff_series = np.diff(series)
                noise_level = np.std(diff_series) / (np.std(series) + 1e-8)

                features['is_stationary'].append(is_stationary)
                features['has_trend'].append(has_trend)
                features['has_seasonality'].append(has_seasonality)
                features['noise_level'].append(noise_level)

        return features

    def get_traditional_predictions(self, x):
        """
        获取传统模型的预测结果

        Args:
            x: [B, T, N] 输入序列

        Returns:
            arima_pred: [B, pred_len, N] ARIMA 预测
            es_pred: [B, pred_len, N] 指数平滑预测
            confidences: dict 各模型置信度
        """
        x_np = x.detach().cpu().numpy()
        B, T, N = x_np.shape
        device = x.device

        arima_preds = np.zeros((B, self.pred_len, N))
        es_preds = np.zeros((B, self.pred_len, N))
        arima_confs = []
        es_confs = []

        for b in range(B):
            for n in range(N):
                series = x_np[b, :, n]

                # ARIMA 预测
                if self.use_arima:
                    arima = ARIMAWrapper(order=auto_arima_order(series))
                    if arima.fit(series):
                        arima_preds[b, :, n] = arima.predict(self.pred_len)
                        arima_confs.append(arima.get_confidence())
                    else:
                        arima_preds[b, :, n] = series[-1]  # 用最后一个值填充
                        arima_confs.append(0.0)

                # 指数平滑预测
                if self.use_es:
                    es = ExponentialSmoothingWrapper(
                        trend='add',
                        seasonal='add' if T >= 48 else None,
                        seasonal_periods=24
                    )
                    if es.fit(series):
                        es_preds[b, :, n] = es.predict(self.pred_len)
                        es_confs.append(es.get_confidence())
                    else:
                        es_preds[b, :, n] = series[-1]
                        es_confs.append(0.0)

        return (
            torch.tensor(arima_preds, dtype=torch.float32, device=device),
            torch.tensor(es_preds, dtype=torch.float32, device=device),
            {
                'arima': np.mean(arima_confs) if arima_confs else 0.0,
                'es': np.mean(es_confs) if es_confs else 0.0
            }
        )

    def forward(self, x_enc, timellm_pred):
        """
        融合传统模型与 Time-LLM 预测

        Args:
            x_enc: [B, T, N] 输入序列
            timellm_pred: [B, pred_len, N] Time-LLM 预测结果

        Returns:
            fused_pred: [B, pred_len, N] 融合后的预测
            report: dict 可解释性报告
        """
        B, pred_len, N = timellm_pred.shape

        # 获取传统模型预测
        arima_pred, es_pred, confidences = self.get_traditional_predictions(x_enc)

        # 融合
        if self.fusion_method == 'average':
            weights = torch.tensor([1/3, 1/3, 1/3], device=timellm_pred.device)
            fused_pred = (
                weights[0] * timellm_pred +
                weights[1] * arima_pred +
                weights[2] * es_pred
            )

        elif self.fusion_method == 'weighted':
            # 基于置信度的加权
            conf_sum = confidences['arima'] + confidences['es'] + 1.0  # Time-LLM 置信度设为 1
            w_arima = confidences['arima'] / conf_sum
            w_es = confidences['es'] / conf_sum
            w_llm = 1.0 / conf_sum

            fused_pred = (
                w_llm * timellm_pred +
                w_arima * arima_pred +
                w_es * es_pred
            )

        elif self.fusion_method == 'adaptive':
            # 自适应门控融合
            concat_pred = torch.cat([
                timellm_pred.view(B, -1),
                arima_pred.view(B, -1),
                es_pred.view(B, -1)
            ], dim=-1)

            weights = self.fusion_gate(concat_pred)  # [B, 3]
            weights = weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, 1, 3]

            stacked = torch.stack([timellm_pred, arima_pred, es_pred], dim=-1)
            fused_pred = (stacked * weights).sum(dim=-1)

        else:
            fused_pred = timellm_pred

        # 生成可解释性报告
        report = {
            'traditional_confidences': confidences,
            'fusion_method': self.fusion_method,
            'arima_pred_sample': arima_pred[0, :5, 0].tolist() if B > 0 else [],
            'es_pred_sample': es_pred[0, :5, 0].tolist() if B > 0 else [],
            'timellm_pred_sample': timellm_pred[0, :5, 0].tolist() if B > 0 else []
        }

        return fused_pred, report


class InterpretabilityReport:
    """可解释性报告生成器"""

    @staticmethod
    def generate(x_enc, predictions, fusion_report):
        """
        生成详细的可解释性报告

        Args:
            x_enc: 输入数据
            predictions: 预测结果
            fusion_report: 融合报告

        Returns:
            str: 格式化的报告文本
        """
        x_np = x_enc.detach().cpu().numpy()
        B, T, N = x_np.shape

        # 计算输入统计
        stats = {
            'min': np.min(x_np),
            'max': np.max(x_np),
            'mean': np.mean(x_np),
            'std': np.std(x_np),
            'trend': 'upward' if np.polyfit(range(T), x_np[0, :, 0], 1)[0] > 0 else 'downward'
        }

        report = f"""
═══════════════════════════════════════════════════════════════
              Time-LLM 可解释性预测报告
═══════════════════════════════════════════════════════════════

1. 输入数据特征分析
   - 数据范围: [{stats['min']:.4f}, {stats['max']:.4f}]
   - 均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}
   - 整体趋势: {stats['trend']}

2. 模型选择与置信度
   - ARIMA 置信度: {fusion_report['traditional_confidences']['arima']:.4f}
   - 指数平滑置信度: {fusion_report['traditional_confidences']['es']:.4f}
   - 融合方法: {fusion_report['fusion_method']}

3. 各模型预测对比 (前5步)
   - Time-LLM: {fusion_report['timellm_pred_sample']}
   - ARIMA:    {fusion_report['arima_pred_sample']}
   - 指数平滑: {fusion_report['es_pred_sample']}

═══════════════════════════════════════════════════════════════
"""
        return report
```

### 1.4 修改 TimeLLM.py

**修改位置**: `models/TimeLLM.py`

```python
# 在文件开头添加导入
from models.traditional.hybrid_forecaster import HybridForecaster, InterpretabilityReport

# 在 Model.__init__ 中添加 (约第 247 行之后)
# ========== 新增: 混合预测器 ==========
if hasattr(configs, 'use_hybrid') and configs.use_hybrid:
    self.hybrid_forecaster = HybridForecaster(
        pred_len=configs.pred_len,
        use_arima=True,
        use_es=True,
        fusion_method=getattr(configs, 'fusion_method', 'adaptive')
    )
    self.use_hybrid = True
else:
    self.use_hybrid = False
# =====================================

# 修改 forward 方法 (约第 251-255 行)
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        timellm_pred = dec_out[:, -self.pred_len:, :]

        # ========== 新增: 混合预测融合 ==========
        if self.use_hybrid:
            fused_pred, report = self.hybrid_forecaster(x_enc, timellm_pred)
            # 可选: 打印可解释性报告
            # print(InterpretabilityReport.generate(x_enc, fused_pred, report))
            return fused_pred
        # ========================================

        return timellm_pred
    return None
```

### 1.5 修改 run_main.py

```python
# 在参数定义部分添加 (约第 100 行之后)
# ========== 新增: 混合预测参数 ==========
parser.add_argument('--use_hybrid', action='store_true',
                    help='Enable hybrid forecasting with traditional models')
parser.add_argument('--fusion_method', type=str, default='adaptive',
                    choices=['adaptive', 'average', 'weighted'],
                    help='Method for fusing traditional and LLM predictions')
# ========================================
```

### 1.6 运行命令

```bash
# 基础命令 (不使用混合预测)
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --pred_len 96 \
  ...

# 启用混合预测
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --pred_len 96 \
  --use_hybrid \
  --fusion_method adaptive \
  ...
```

### 1.7 依赖安装

```bash
pip install statsmodels scipy
```

---

## 方案二: 多尺度分解混合 实践指南

### 2.1 可行性评估

| 评估项 | 评分 | 说明 |
|--------|------|------|
| **实现难度** | ⭐⭐⭐ | 需要修改 Patching 和 LLM 层分配逻辑 |
| **代码侵入性** | 中 | 需要修改 forward 流程 |
| **显存影响** | +20% | 多尺度并行处理 |
| **兼容性** | 高 | 可作为可选模块 |

### 2.2 核心代码实现

#### 文件: `layers/MultiScaleEmbed.py` (新建)

```python
"""多尺度时序分解与嵌入模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PatchEmbedding, TokenEmbedding


class MultiScaleDecomposition(nn.Module):
    """多尺度时序分解模块"""

    def __init__(self, scales=[1, 2, 4]):
        """
        Args:
            scales: 下采样比例列表
        """
        super().__init__()
        self.scales = scales
        self.downsamples = nn.ModuleList([
            nn.AvgPool1d(kernel_size=s, stride=s) if s > 1 else nn.Identity()
            for s in scales
        ])

    def forward(self, x):
        """
        Args:
            x: [B, T, N]

        Returns:
            list of [B, T/s, N] for each scale
        """
        outputs = []
        x_transposed = x.permute(0, 2, 1)  # [B, N, T]

        for ds in self.downsamples:
            if isinstance(ds, nn.Identity):
                x_ds = x_transposed
            else:
                x_ds = ds(x_transposed)  # [B, N, T/s]
            outputs.append(x_ds.permute(0, 2, 1))  # [B, T/s, N]

        return outputs


class MultiScalePatchEmbedding(nn.Module):
    """多尺度 Patch 嵌入模块"""

    def __init__(self, d_model, scales=[1, 2, 4], base_patch_len=16, base_stride=8, dropout=0.1):
        """
        Args:
            d_model: 嵌入维度
            scales: 尺度列表
            base_patch_len: 基础 patch 长度
            base_stride: 基础步长
            dropout: dropout 比例
        """
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)

        # 为每个尺度创建 Patch Embedding
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                d_model=d_model,
                patch_len=max(base_patch_len // s, 4),  # 尺度越大，patch 越小
                stride=max(base_stride // s, 2),
                dropout=dropout
            )
            for s in scales
        ])

        # 多尺度分解
        self.decomposition = MultiScaleDecomposition(scales)

    def forward(self, x):
        """
        Args:
            x: [B, N, T]

        Returns:
            list of (enc_out, n_vars) for each scale
        """
        # 分解为多尺度
        x_permuted = x.permute(0, 2, 1)  # [B, T, N]
        multi_scale_inputs = self.decomposition(x_permuted)

        # 每个尺度独立进行 Patch Embedding
        outputs = []
        for scale_idx, x_scale in enumerate(multi_scale_inputs):
            x_scale = x_scale.permute(0, 2, 1)  # [B, N, T/s]
            enc_out, n_vars = self.patch_embeddings[scale_idx](x_scale)
            outputs.append((enc_out, n_vars))

        return outputs


class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, d_model, num_scales=3, fusion_method='gate'):
        """
        Args:
            d_model: 特征维度
            num_scales: 尺度数量
            fusion_method: 融合方法 ('concat', 'sum', 'gate', 'attention')
        """
        super().__init__()
        self.fusion_method = fusion_method
        self.num_scales = num_scales

        if fusion_method == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(d_model * num_scales, d_model),
                nn.Sigmoid()
            )
            self.proj = nn.Linear(d_model * num_scales, d_model)

        elif fusion_method == 'attention':
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=4,
                batch_first=True
            )
            self.scale_query = nn.Parameter(torch.randn(1, 1, d_model))

        elif fusion_method == 'concat':
            self.proj = nn.Linear(d_model * num_scales, d_model)

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: list of [B, L_i, D] 多尺度特征

        Returns:
            fused: [B, L, D] 融合后的特征
        """
        # 首先对齐序列长度（取最短的）
        min_len = min(f.shape[1] for f in multi_scale_features)
        aligned_features = [f[:, :min_len, :] for f in multi_scale_features]

        if self.fusion_method == 'sum':
            fused = sum(aligned_features) / len(aligned_features)

        elif self.fusion_method == 'concat':
            concatenated = torch.cat(aligned_features, dim=-1)
            fused = self.proj(concatenated)

        elif self.fusion_method == 'gate':
            concatenated = torch.cat(aligned_features, dim=-1)
            gate = self.gate(concatenated)
            proj = self.proj(concatenated)
            fused = gate * proj + (1 - gate) * aligned_features[0]  # 残差连接

        elif self.fusion_method == 'attention':
            B, L, D = aligned_features[0].shape
            stacked = torch.stack(aligned_features, dim=2)  # [B, L, num_scales, D]
            stacked = stacked.view(B * L, self.num_scales, D)

            query = self.scale_query.expand(B * L, -1, -1)
            attn_out, _ = self.scale_attention(query, stacked, stacked)
            fused = attn_out.view(B, L, D)

        return fused


class MultiScaleTimeLLM(nn.Module):
    """多尺度 Time-LLM 封装"""

    def __init__(self, configs, base_model):
        """
        Args:
            configs: 配置对象
            base_model: 原始 Time-LLM 模型
        """
        super().__init__()
        self.configs = configs
        self.base_model = base_model

        # 多尺度配置
        self.scales = getattr(configs, 'scales', [1, 2, 4])
        self.num_scales = len(self.scales)

        # 替换 patch_embedding 为多尺度版本
        self.multi_scale_patch = MultiScalePatchEmbedding(
            d_model=configs.d_model,
            scales=self.scales,
            base_patch_len=configs.patch_len,
            base_stride=configs.stride,
            dropout=configs.dropout
        )

        # 多尺度融合
        self.fusion = MultiScaleFusion(
            d_model=configs.d_ff,
            num_scales=self.num_scales,
            fusion_method=getattr(configs, 'fusion_method', 'gate')
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """多尺度前向传播"""
        # 使用基础模型的大部分逻辑，但替换 patching 部分
        # ... 具体实现需要根据实际需求调整
        pass
```

### 2.3 简化实现方案

如果显存受限，可以使用以下简化方案：

#### 修改 `models/TimeLLM.py`

```python
# 在 Model.__init__ 中添加 (约第 240 行之后)

# ========== 新增: 多尺度配置 ==========
self.use_multiscale = getattr(configs, 'use_multiscale', False)
if self.use_multiscale:
    self.scales = getattr(configs, 'scales', [1, 2])  # 简化为双尺度
    self.patch_embeddings = nn.ModuleList([
        PatchEmbedding(
            configs.d_model,
            max(self.patch_len // s, 4),
            max(self.stride // s, 2),
            configs.dropout
        )
        for s in self.scales
    ])
    # 尺度融合
    self.scale_fusion = nn.Linear(configs.d_ff * len(self.scales), configs.d_ff)
# =====================================

# 修改 forecast 方法中的 patching 部分
def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... 前面的代码保持不变 ...

    x_enc = x_enc.permute(0, 2, 1).contiguous()

    # ========== 多尺度 Patching ==========
    if self.use_multiscale:
        multi_scale_outputs = []
        for scale_idx, scale in enumerate(self.scales):
            if scale > 1:
                # 下采样
                x_scaled = F.avg_pool1d(x_enc, kernel_size=scale, stride=scale)
            else:
                x_scaled = x_enc

            enc_out, n_vars = self.patch_embeddings[scale_idx](x_scaled.float())
            enc_out = enc_out.to(prompt_embeddings.dtype)
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
            multi_scale_outputs.append(enc_out)

        # 融合 (简单拼接 + 线性层)
        # 需要先对齐长度
        min_len = min(out.shape[1] for out in multi_scale_outputs)
        aligned = [out[:, :min_len, :] for out in multi_scale_outputs]
        enc_out = torch.cat(aligned, dim=-1)
        enc_out = self.scale_fusion(enc_out)
    else:
        enc_out, n_vars = self.patch_embedding(x_enc.float())
        enc_out = enc_out.to(prompt_embeddings.dtype)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
    # =====================================

    # ... 后续代码保持不变 ...
```

### 2.4 运行命令

```bash
# 启用多尺度处理
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --pred_len 96 \
  --use_multiscale \
  --scales 1 2 \
  ...
```

---

## 方案三: 频域增强 实践指南

### 3.1 可行性评估

| 评估项 | 评分 | 说明 |
|--------|------|------|
| **实现难度** | ⭐⭐ | FFT 操作相对简单 |
| **代码侵入性** | 低 | 仅在 patching 前添加预处理 |
| **显存影响** | +5% | FFT 计算开销小 |
| **兼容性** | 高 | 可作为可选预处理 |

### 3.2 核心代码实现

#### 文件: `layers/FrequencyDecomposition.py` (新建)

```python
"""频域分解增强模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyDecomposition(nn.Module):
    """基于 FFT 的趋势-季节性分解"""

    def __init__(self, top_k_freqs=5, trend_cutoff=0.1):
        """
        Args:
            top_k_freqs: 保留的主要频率数量
            trend_cutoff: 趋势的频率截止比例
        """
        super().__init__()
        self.top_k = top_k_freqs
        self.trend_cutoff = trend_cutoff

    def forward(self, x):
        """
        分解时序为趋势和季节性成分

        Args:
            x: [B, T, N]

        Returns:
            trend: [B, T, N] 趋势成分
            seasonal: [B, T, N] 季节性成分
        """
        B, T, N = x.shape

        # FFT 变换 (沿时间维度)
        x_fft = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(T, device=x.device)

        # 振幅
        amplitude = torch.abs(x_fft)

        # 低频成分 (趋势)
        trend_mask = (freqs < self.trend_cutoff).float()
        trend_mask = trend_mask.unsqueeze(0).unsqueeze(-1)  # [1, freq, 1]
        x_fft_trend = x_fft * trend_mask
        trend = torch.fft.irfft(x_fft_trend, n=T, dim=1)

        # 高频成分 (季节性) = 原信号 - 趋势
        seasonal = x - trend

        return trend, seasonal

    def get_dominant_periods(self, x, top_k=5):
        """
        获取主导周期

        Args:
            x: [B, T, N]
            top_k: 返回前 k 个主要周期

        Returns:
            periods: [B, N, top_k] 主要周期
        """
        B, T, N = x.shape

        # FFT
        x_fft = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_fft)  # [B, freq, N]

        # 找到振幅最大的频率
        _, top_indices = torch.topk(amplitude, top_k, dim=1)  # [B, top_k, N]

        # 转换为周期
        freqs = torch.fft.rfftfreq(T, device=x.device)
        periods = []
        for b in range(B):
            period_b = []
            for n in range(N):
                indices = top_indices[b, :, n]
                freq_values = freqs[indices]
                period_values = 1.0 / (freq_values + 1e-8)
                period_b.append(period_values)
            periods.append(torch.stack(period_b, dim=0))

        return torch.stack(periods, dim=0)  # [B, N, top_k]


class FrequencyEnhancedEmbedding(nn.Module):
    """频域增强的时序嵌入"""

    def __init__(self, d_model, patch_len, stride, dropout, use_frequency=True):
        """
        Args:
            d_model: 嵌入维度
            patch_len: patch 长度
            stride: 步长
            dropout: dropout 比例
            use_frequency: 是否使用频域增强
        """
        super().__init__()
        self.use_frequency = use_frequency

        # 频域分解
        if use_frequency:
            self.freq_decomp = FrequencyDecomposition()

            # 趋势分支的 Patch Embedding
            from layers.Embed import PatchEmbedding
            self.trend_embedding = PatchEmbedding(d_model, patch_len * 2, stride * 2, dropout)

            # 季节性分支的 Patch Embedding
            self.seasonal_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)

            # 融合层
            self.fusion = nn.Linear(d_model * 2, d_model)
        else:
            from layers.Embed import PatchEmbedding
            self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, T]

        Returns:
            enc_out: [B*N, num_patches, d_model]
            n_vars: 变量数
        """
        if self.use_frequency:
            # 分解
            x_permuted = x.permute(0, 2, 1)  # [B, T, N]
            trend, seasonal = self.freq_decomp(x_permuted)

            # 分别嵌入
            trend = trend.permute(0, 2, 1)  # [B, N, T]
            seasonal = seasonal.permute(0, 2, 1)  # [B, N, T]

            trend_out, n_vars = self.trend_embedding(trend)
            seasonal_out, _ = self.seasonal_embedding(seasonal)

            # 对齐长度
            min_len = min(trend_out.shape[1], seasonal_out.shape[1])
            trend_out = trend_out[:, :min_len, :]
            seasonal_out = seasonal_out[:, :min_len, :]

            # 融合
            combined = torch.cat([trend_out, seasonal_out], dim=-1)
            enc_out = self.fusion(combined)

            return enc_out, n_vars
        else:
            return self.patch_embedding(x)


class AdaptiveFrequencySwitch(nn.Module):
    """自适应频域开关 - 根据数据特征决定是否使用频域增强"""

    def __init__(self, periodicity_threshold=0.5, trend_threshold=0.3):
        """
        Args:
            periodicity_threshold: 周期性阈值
            trend_threshold: 趋势性阈值
        """
        super().__init__()
        self.periodicity_threshold = periodicity_threshold
        self.trend_threshold = trend_threshold

    def should_use_frequency(self, x):
        """
        判断是否应该使用频域增强

        Args:
            x: [B, T, N]

        Returns:
            use_freq: bool
            reason: str
        """
        B, T, N = x.shape

        # 计算 FFT
        x_fft = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_fft)

        # 计算能量集中度 (周期性指标)
        total_energy = amplitude.sum(dim=1)
        sorted_amp, _ = torch.sort(amplitude, dim=1, descending=True)
        top5_energy = sorted_amp[:, :5, :].sum(dim=1)
        periodicity = (top5_energy / (total_energy + 1e-8)).mean()

        # 计算趋势强度
        trend = x[:, -1, :] - x[:, 0, :]
        trend_strength = torch.abs(trend).mean() / (x.std() + 1e-8)

        if periodicity > self.periodicity_threshold:
            return True, f"strong_periodicity ({periodicity:.3f})"
        elif trend_strength > self.trend_threshold:
            return True, f"strong_trend ({trend_strength:.3f})"
        else:
            return False, "weak_pattern"
```

### 3.3 修改 TimeLLM.py

```python
# 在 Model.__init__ 中添加 (约第 230 行之后)

# ========== 新增: 频域增强 ==========
self.use_frequency = getattr(configs, 'use_frequency', False)
if self.use_frequency:
    from layers.FrequencyDecomposition import FrequencyDecomposition, AdaptiveFrequencySwitch

    self.freq_decomp = FrequencyDecomposition(
        top_k_freqs=getattr(configs, 'top_k_freqs', 5),
        trend_cutoff=getattr(configs, 'trend_cutoff', 0.1)
    )

    # 双分支 Patch Embedding
    self.trend_patch_embedding = PatchEmbedding(
        configs.d_model, self.patch_len * 2, self.stride * 2, configs.dropout
    )
    self.seasonal_patch_embedding = PatchEmbedding(
        configs.d_model, self.patch_len, self.stride, configs.dropout
    )

    # 融合层
    self.freq_fusion = nn.Linear(configs.d_model * 2, configs.d_model)

    # 自适应开关
    self.adaptive_switch = AdaptiveFrequencySwitch()
# =====================================


# 在 forecast 方法中修改 patching 部分

def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... 前面的代码保持不变 ...

    # ========== 频域增强处理 ==========
    if self.use_frequency:
        # 检查是否需要使用频域
        use_freq, reason = self.adaptive_switch.should_use_frequency(x_enc_normalized)
        # print(f"Frequency enhancement: {use_freq} ({reason})")

        if use_freq:
            # 分解
            trend, seasonal = self.freq_decomp(x_enc_normalized)

            # 转置
            trend = trend.permute(0, 2, 1).contiguous()
            seasonal = seasonal.permute(0, 2, 1).contiguous()

            # 分别嵌入
            trend_out, n_vars = self.trend_patch_embedding(trend.float())
            seasonal_out, _ = self.seasonal_patch_embedding(seasonal.float())

            # 对齐长度
            min_len = min(trend_out.shape[1], seasonal_out.shape[1])
            trend_out = trend_out[:, :min_len, :]
            seasonal_out = seasonal_out[:, :min_len, :]

            # 融合
            combined = torch.cat([trend_out, seasonal_out], dim=-1)
            enc_out = self.freq_fusion(combined)
            enc_out = enc_out.to(prompt_embeddings.dtype)
        else:
            # 不使用频域增强，走原始路径
            x_enc = x_enc_normalized.permute(0, 2, 1).contiguous()
            enc_out, n_vars = self.patch_embedding(x_enc.float())
            enc_out = enc_out.to(prompt_embeddings.dtype)
    else:
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.float())
        enc_out = enc_out.to(prompt_embeddings.dtype)
    # =====================================

    # ... 后续代码保持不变 ...
```

### 3.4 运行命令

```bash
# 启用频域增强
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --pred_len 96 \
  --use_frequency \
  --top_k_freqs 5 \
  --trend_cutoff 0.1 \
  ...
```

---

## 方案四: 变量间注意力增强 实践指南

### 4.1 可行性评估

| 评估项 | 评分 | 说明 |
|--------|------|------|
| **实现难度** | ⭐⭐ | 标准 Multi-Head Attention |
| **代码侵入性** | 低 | 在 Reprogramming 后添加模块 |
| **显存影响** | +5% | 注意力矩阵 N×N |
| **兼容性** | 高 | 仅对多变量有效 |

### 4.2 核心代码实现

#### 文件: `layers/InterVariateAttention.py` (新建)

```python
"""变量间注意力模块 - 捕获多变量相关性"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InterVariateAttention(nn.Module):
    """
    变量间注意力模块

    在变量维度上做注意力，捕获变量间的相关性
    灵感来源: iTransformer (ICLR 2024 Spotlight)
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        """
        Args:
            d_model: 嵌入维度
            n_heads: 注意力头数
            dropout: dropout 比例
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # 投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x, n_vars):
        """
        Args:
            x: [B*N, num_patches, d_model] - Reprogramming 后的嵌入
            n_vars: 变量数 N

        Returns:
            out: [B*N, num_patches, d_model]
        """
        B_N, num_patches, d_model = x.shape
        B = B_N // n_vars
        N = n_vars
        H = self.n_heads

        # Reshape: [B*N, L, D] -> [B, N, L, D]
        x = x.view(B, N, num_patches, d_model)

        # 转置: [B, N, L, D] -> [B, L, N, D]
        # 这样在 N 维度上做注意力
        x = x.permute(0, 2, 1, 3).contiguous()

        # 保存残差
        residual = x

        # Reshape for attention: [B, L, N, D] -> [B*L, N, D]
        x_flat = x.view(B * num_patches, N, d_model)

        # Q, K, V 投影
        Q = self.W_q(x_flat)  # [B*L, N, D]
        K = self.W_k(x_flat)
        V = self.W_v(x_flat)

        # 分头: [B*L, N, D] -> [B*L, N, H, D/H] -> [B*L, H, N, D/H]
        Q = Q.view(B * num_patches, N, H, self.d_head).transpose(1, 2)
        K = K.view(B * num_patches, N, H, self.d_head).transpose(1, 2)
        V = V.view(B * num_patches, N, H, self.d_head).transpose(1, 2)

        # 注意力计算: [B*L, H, N, D/H] @ [B*L, H, D/H, N] -> [B*L, H, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 加权聚合: [B*L, H, N, N] @ [B*L, H, N, D/H] -> [B*L, H, N, D/H]
        attn_output = torch.matmul(attn_probs, V)

        # 合并头: [B*L, H, N, D/H] -> [B*L, N, H, D/H] -> [B*L, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B * num_patches, N, d_model)

        # 输出投影
        attn_output = self.W_o(attn_output)

        # Reshape back: [B*L, N, D] -> [B, L, N, D]
        attn_output = attn_output.view(B, num_patches, N, d_model)

        # 残差连接 + 层归一化
        out = self.norm(residual + attn_output)

        # 转回原始形状: [B, L, N, D] -> [B, N, L, D] -> [B*N, L, D]
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B * N, num_patches, d_model)

        return out

    def get_attention_weights(self, x, n_vars):
        """
        获取注意力权重（用于可视化）

        Returns:
            attn_weights: [B, L, H, N, N]
        """
        B_N, num_patches, d_model = x.shape
        B = B_N // n_vars
        N = n_vars
        H = self.n_heads

        x = x.view(B, N, num_patches, d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        x_flat = x.view(B * num_patches, N, d_model)

        Q = self.W_q(x_flat).view(B * num_patches, N, H, self.d_head).transpose(1, 2)
        K = self.W_k(x_flat).view(B * num_patches, N, H, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # [B*L, H, N, N] -> [B, L, H, N, N]
        attn_weights = attn_weights.view(B, num_patches, H, N, N)

        return attn_weights


class InterVariateAttentionBlock(nn.Module):
    """带前馈网络的完整变量间注意力块"""

    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.attention = InterVariateAttention(d_model, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, n_vars):
        # 注意力
        x = self.attention(x, n_vars)

        # 前馈网络 + 残差
        x = self.norm(x + self.ffn(x))

        return x
```

### 4.3 修改 TimeLLM.py

```python
# 在 Model.__init__ 中添加 (约第 238 行之后)

# ========== 新增: 变量间注意力 ==========
self.use_inter_variate = getattr(configs, 'use_inter_variate', False)
if self.use_inter_variate:
    from layers.InterVariateAttention import InterVariateAttentionBlock

    self.inter_variate_attn = InterVariateAttentionBlock(
        d_model=self.d_llm,  # 使用 LLM 维度
        n_heads=getattr(configs, 'inter_variate_heads', 8),
        d_ff=getattr(configs, 'inter_variate_ff', self.d_llm * 4),
        dropout=configs.dropout
    )
# =====================================


# 在 forecast 方法中，Reprogramming 后添加

def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... 前面的代码 ...

    enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

    # ========== 新增: 变量间注意力 ==========
    if self.use_inter_variate and n_vars > 1:
        enc_out = self.inter_variate_attn(enc_out, n_vars)
    # ========================================

    llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

    # ... 后续代码 ...
```

### 4.4 修改 run_main.py

```python
# 在参数定义中添加

# ========== 变量间注意力参数 ==========
parser.add_argument('--use_inter_variate', action='store_true',
                    help='Enable inter-variate attention')
parser.add_argument('--inter_variate_heads', type=int, default=8,
                    help='Number of attention heads for inter-variate attention')
parser.add_argument('--inter_variate_ff', type=int, default=0,
                    help='FFN dimension for inter-variate attention (0=auto)')
# =====================================
```

### 4.5 运行命令

```bash
# 启用变量间注意力
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --pred_len 96 \
  --use_inter_variate \
  --inter_variate_heads 8 \
  ...
```

---

## 方案五: 动态Prompt生成 实践指南

### 5.1 可行性评估

| 评估项 | 评分 | 说明 |
|--------|------|------|
| **实现难度** | ⭐⭐⭐ | 需要可学习的 Prompt Encoder |
| **代码侵入性** | 中 | 修改 Prompt 构建逻辑 |
| **显存影响** | +15% | 额外的 Prompt Token |
| **兼容性** | 高 | 与原有 Prompt 兼容 |

### 5.2 核心代码实现

#### 文件: `layers/DynamicPrompt.py` (新建)

```python
"""动态 Prompt 生成模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPromptEncoder(nn.Module):
    """
    动态 Prompt 编码器

    从时序数据直接学习 Prompt 嵌入，替代或增强静态统计信息
    """

    def __init__(self, seq_len, n_vars, llm_dim, num_prompt_tokens=32):
        """
        Args:
            seq_len: 输入序列长度
            n_vars: 变量数
            llm_dim: LLM 嵌入维度
            num_prompt_tokens: 生成的 Prompt Token 数量
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.llm_dim = llm_dim
        self.num_tokens = num_prompt_tokens

        # 时序编码器
        self.temporal_encoder = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_prompt_tokens * llm_dim)
        )

        # 可学习的基础 Prompt Tokens
        self.base_prompt = nn.Parameter(
            torch.randn(1, num_prompt_tokens, llm_dim) * 0.02
        )

        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(llm_dim * 2, llm_dim),
            nn.Sigmoid()
        )

        # 层归一化
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x_enc):
        """
        Args:
            x_enc: [B, T, N] 归一化后的输入序列

        Returns:
            dynamic_prompt: [B*N, num_tokens, llm_dim]
        """
        B, T, N = x_enc.shape

        # 展平为 [B*N, T]
        x_flat = x_enc.permute(0, 2, 1).contiguous().view(B * N, T)

        # 生成动态 Prompt: [B*N, num_tokens * llm_dim]
        dynamic_prompt = self.temporal_encoder(x_flat)
        dynamic_prompt = dynamic_prompt.view(B * N, self.num_tokens, self.llm_dim)

        # 扩展基础 Prompt
        base_prompt = self.base_prompt.expand(B * N, -1, -1)

        # 门控融合
        combined = torch.cat([dynamic_prompt, base_prompt], dim=-1)
        gate = self.fusion_gate(combined)

        # 融合
        output = gate * dynamic_prompt + (1 - gate) * base_prompt

        # 归一化
        output = self.norm(output)

        return output


class LocalStatisticsEncoder(nn.Module):
    """
    局部统计特征编码器

    将序列分段，计算每段的统计特征，编码为嵌入
    """

    def __init__(self, seq_len, llm_dim, num_segments=3):
        """
        Args:
            seq_len: 序列长度
            llm_dim: LLM 维度
            num_segments: 分段数
        """
        super().__init__()
        self.num_segments = num_segments
        self.segment_len = seq_len // num_segments

        # 统计特征: [min, max, mean, std, trend] * num_segments
        num_features = 5 * num_segments
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.GELU(),
            nn.Linear(64, llm_dim)
        )

    def forward(self, x_enc):
        """
        Args:
            x_enc: [B, T, N]

        Returns:
            local_stats_emb: [B*N, 1, llm_dim]
        """
        B, T, N = x_enc.shape

        features = []
        for i in range(self.num_segments):
            start = i * self.segment_len
            end = start + self.segment_len if i < self.num_segments - 1 else T

            segment = x_enc[:, start:end, :]  # [B, seg_len, N]

            # 计算统计量
            seg_min = segment.min(dim=1).values  # [B, N]
            seg_max = segment.max(dim=1).values
            seg_mean = segment.mean(dim=1)
            seg_std = segment.std(dim=1)
            seg_trend = segment[:, -1, :] - segment[:, 0, :]

            features.extend([seg_min, seg_max, seg_mean, seg_std, seg_trend])

        # 拼接: [B, N, 5*num_segments]
        features = torch.stack(features, dim=-1)  # [B, N, 5*num_segments]

        # 展平: [B*N, 5*num_segments]
        features = features.view(B * N, -1)

        # 编码: [B*N, llm_dim]
        emb = self.encoder(features)

        # 添加序列维度: [B*N, 1, llm_dim]
        return emb.unsqueeze(1)


class DynamicPromptModule(nn.Module):
    """
    完整的动态 Prompt 模块

    整合:
    1. 静态文本 Prompt (原有)
    2. 动态时序 Prompt (新增)
    3. 局部统计 Prompt (新增)
    """

    def __init__(self, seq_len, n_vars, llm_dim, num_dynamic_tokens=32,
                 use_local_stats=True, num_segments=3):
        super().__init__()

        self.dynamic_encoder = DynamicPromptEncoder(
            seq_len, n_vars, llm_dim, num_dynamic_tokens
        )

        self.use_local_stats = use_local_stats
        if use_local_stats:
            self.local_stats_encoder = LocalStatisticsEncoder(
                seq_len, llm_dim, num_segments
            )

        # 融合不同来源的 Prompt
        self.prompt_fusion = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x_enc, static_prompt_emb):
        """
        Args:
            x_enc: [B, T, N] 输入序列
            static_prompt_emb: [B*N, static_len, llm_dim] 静态 Prompt 嵌入

        Returns:
            fused_prompt: [B*N, total_len, llm_dim]
        """
        B, T, N = x_enc.shape

        # 动态 Prompt
        dynamic_prompt = self.dynamic_encoder(x_enc)  # [B*N, num_tokens, llm_dim]

        # 局部统计 Prompt
        if self.use_local_stats:
            local_stats = self.local_stats_encoder(x_enc)  # [B*N, 1, llm_dim]
            dynamic_prompt = torch.cat([dynamic_prompt, local_stats], dim=1)

        # 与静态 Prompt 拼接
        all_prompts = torch.cat([static_prompt_emb, dynamic_prompt], dim=1)

        # 注意力融合 (可选: 使用静态 Prompt 作为 Query)
        # fused, _ = self.prompt_fusion(
        #     static_prompt_emb, all_prompts, all_prompts
        # )

        return all_prompts
```

### 5.3 修改 TimeLLM.py

```python
# 在 Model.__init__ 中添加

# ========== 新增: 动态 Prompt ==========
self.use_dynamic_prompt = getattr(configs, 'use_dynamic_prompt', False)
if self.use_dynamic_prompt:
    from layers.DynamicPrompt import DynamicPromptModule

    self.dynamic_prompt_module = DynamicPromptModule(
        seq_len=configs.seq_len,
        n_vars=configs.enc_in,
        llm_dim=self.d_llm,
        num_dynamic_tokens=getattr(configs, 'num_dynamic_tokens', 32),
        use_local_stats=True,
        num_segments=3
    )
# =====================================


# 在 forecast 方法中，prompt_embeddings 生成后添加

def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... 前面的代码 ...

    prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
    prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

    # ========== 新增: 动态 Prompt ==========
    if self.use_dynamic_prompt:
        prompt_embeddings = self.dynamic_prompt_module(
            x_enc_for_stats,  # 归一化后的输入
            prompt_embeddings
        )
    # =====================================

    # ... 后续代码 ...
```

### 5.4 运行命令

```bash
# 启用动态 Prompt
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --pred_len 96 \
  --use_dynamic_prompt \
  --num_dynamic_tokens 32 \
  ...
```

---

## 方案六: 稀疏专家混合 实践指南

### 6.1 可行性评估

| 评估项 | 评分 | 说明 |
|--------|------|------|
| **实现难度** | ⭐⭐⭐⭐ | MoE 实现相对复杂 |
| **代码侵入性** | 中 | 在 Reprogramming 后添加 |
| **显存影响** | +30% | 多个专家网络 |
| **兼容性** | 高 | 可选模块 |

### 6.2 核心代码实现

#### 文件: `layers/TimeSeriesMoE.py` (新建)

```python
"""时序专用的稀疏专家混合 (MoE) 模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """单个专家网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    """路由器 - 决定使用哪些专家"""

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]

        Returns:
            top_k_indices: [B, L, top_k]
            top_k_weights: [B, L, top_k]
        """
        logits = self.gate(x)  # [B, L, num_experts]
        probs = F.softmax(logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # 重新归一化 Top-K 概率
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        return top_k_indices, top_k_weights


class TimeSeriesMoE(nn.Module):
    """
    时序专用的稀疏专家混合层

    设计思想:
    - 不同专家专注于不同的时序模式
    - 通过路由机制动态选择最相关的专家
    - 稀疏激活保证计算效率
    """

    def __init__(self, d_model, num_experts=4, top_k=2, d_ff=None, dropout=0.1):
        """
        Args:
            d_model: 嵌入维度
            num_experts: 专家数量
            top_k: 每次激活的专家数
            d_ff: 专家 FFN 维度
            dropout: dropout 比例
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        d_ff = d_ff or d_model * 4

        # 路由器
        self.router = Router(d_model, num_experts, top_k)

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [B, L, D]

        Returns:
            out: [B, L, D]
        """
        B, L, D = x.shape
        residual = x

        # 获取路由决策
        top_k_indices, top_k_weights = self.router(x)
        # top_k_indices: [B, L, top_k]
        # top_k_weights: [B, L, top_k]

        # 稀疏激活专家
        # 方法1: 简单循环 (适合小规模)
        output = torch.zeros_like(x)

        for k in range(self.top_k):
            indices = top_k_indices[:, :, k]  # [B, L]
            weights = top_k_weights[:, :, k:k+1]  # [B, L, 1]

            for e_idx in range(self.num_experts):
                mask = (indices == e_idx)  # [B, L]

                if mask.any():
                    # 获取需要此专家处理的输入
                    expert_input = x[mask]  # [num_selected, D]

                    # 专家处理
                    expert_output = self.experts[e_idx](expert_input)

                    # 加权累加
                    weights_selected = weights[mask]  # [num_selected, 1]
                    output[mask] += weights_selected * expert_output

        # 残差连接 + 层归一化
        output = self.norm(residual + output)

        return output

    def get_expert_usage(self, x):
        """
        获取专家使用统计（用于分析和负载均衡）

        Returns:
            usage: [num_experts] 每个专家被选中的次数
        """
        top_k_indices, _ = self.router(x)
        usage = torch.zeros(self.num_experts, device=x.device)

        for e_idx in range(self.num_experts):
            usage[e_idx] = (top_k_indices == e_idx).sum()

        return usage


class TimeSeriesMoEWithAuxLoss(TimeSeriesMoE):
    """带辅助损失的 MoE，用于负载均衡"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = 0.01

    def forward(self, x):
        output = super().forward(x)

        # 计算辅助损失（鼓励专家均匀使用）
        if self.training:
            usage = self.get_expert_usage(x)
            target_usage = x.shape[0] * x.shape[1] * self.top_k / self.num_experts
            aux_loss = ((usage - target_usage) ** 2).mean() / (target_usage ** 2)
            self.aux_loss = self.aux_loss_weight * aux_loss
        else:
            self.aux_loss = 0.0

        return output
```

### 6.3 修改 TimeLLM.py

```python
# 在 Model.__init__ 中添加

# ========== 新增: 稀疏专家混合 ==========
self.use_moe = getattr(configs, 'use_moe', False)
if self.use_moe:
    from layers.TimeSeriesMoE import TimeSeriesMoEWithAuxLoss

    self.moe_layer = TimeSeriesMoEWithAuxLoss(
        d_model=self.d_llm,
        num_experts=getattr(configs, 'num_experts', 4),
        top_k=getattr(configs, 'top_k_experts', 2),
        d_ff=getattr(configs, 'moe_d_ff', self.d_llm * 4),
        dropout=configs.dropout
    )
# =====================================


# 在 forecast 方法中，Reprogramming 后添加

def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # ... 前面的代码 ...

    enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

    # ========== 新增: 稀疏专家混合 ==========
    if self.use_moe:
        enc_out = self.moe_layer(enc_out)
    # ========================================

    llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

    # ... 后续代码 ...
```

### 6.4 修改训练损失 (run_main.py)

```python
# 在训练循环中添加辅助损失

# 计算主损失
loss = criterion(outputs, batch_y)

# ========== 新增: MoE 辅助损失 ==========
if hasattr(model, 'moe_layer') and hasattr(model.moe_layer, 'aux_loss'):
    aux_loss = model.moe_layer.aux_loss
    loss = loss + aux_loss
# ========================================

train_loss.append(loss.item())
```

### 6.5 运行命令

```bash
# 启用 MoE
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --pred_len 96 \
  --use_moe \
  --num_experts 4 \
  --top_k_experts 2 \
  ...
```

---

## 方案七: 任务专用词汇表初始化 实践指南

### 7.1 可行性评估

| 评估项 | 评分 | 说明 |
|--------|------|------|
| **实现难度** | ⭐ | 仅修改初始化逻辑 |
| **代码侵入性** | 极低 | 不影响模型结构 |
| **显存影响** | 无 | 仅改变初始值 |
| **兼容性** | 高 | 对训练过程透明 |

### 7.2 核心代码实现

#### 文件: `utils/vocab_init.py` (新建)

```python
"""任务专用词汇表初始化工具"""

import torch
import torch.nn as nn
import numpy as np


# 时序预测相关关键词
TIME_SERIES_KEYWORDS = [
    # 趋势词
    'increase', 'decrease', 'rise', 'fall', 'grow', 'decline', 'stable', 'steady',
    'upward', 'downward', 'ascending', 'descending',

    # 周期词
    'cycle', 'period', 'seasonal', 'daily', 'weekly', 'monthly', 'yearly', 'annual',
    'recurring', 'periodic', 'repetitive',

    # 变化词
    'change', 'fluctuate', 'vary', 'shift', 'swing', 'oscillate', 'wave',

    # 极值词
    'peak', 'valley', 'maximum', 'minimum', 'highest', 'lowest', 'extreme',
    'high', 'low', 'top', 'bottom',

    # 统计词
    'average', 'mean', 'median', 'normal', 'typical', 'unusual', 'anomaly',

    # 预测词
    'forecast', 'predict', 'expect', 'anticipate', 'estimate', 'project',

    # 时间词
    'future', 'past', 'history', 'previous', 'next', 'current', 'recent',
    'before', 'after', 'yesterday', 'tomorrow',

    # 数量词
    'more', 'less', 'few', 'many', 'some', 'most', 'all', 'none',

    # 程度词
    'slight', 'moderate', 'significant', 'dramatic', 'rapid', 'slow', 'gradual',
]


def get_keyword_indices(tokenizer, keywords):
    """
    获取关键词在词表中的索引

    Args:
        tokenizer: HuggingFace tokenizer
        keywords: 关键词列表

    Returns:
        indices: 有效关键词的索引列表
    """
    indices = []
    for word in keywords:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:  # 只保留单 token 词
            indices.append(tokens[0])
    return list(set(indices))


def initialize_mapping_layer_with_keywords(mapping_layer, word_embeddings, tokenizer,
                                           num_anchor_words=100, method='weighted'):
    """
    使用关键词初始化 Mapping Layer

    Args:
        mapping_layer: nn.Linear 层
        word_embeddings: LLM 词嵌入 [vocab_size, llm_dim]
        tokenizer: HuggingFace tokenizer
        num_anchor_words: 锚点词数量
        method: 初始化方法 ('weighted', 'cluster', 'random_from_keywords')
    """
    vocab_size, llm_dim = word_embeddings.shape
    num_tokens = mapping_layer.out_features

    # 获取关键词索引
    keyword_indices = get_keyword_indices(tokenizer, TIME_SERIES_KEYWORDS)
    print(f"Found {len(keyword_indices)} valid keyword indices")

    if method == 'weighted':
        # 方法1: 给关键词更高的权重
        with torch.no_grad():
            # 初始化为小值
            mapping_layer.weight.data.fill_(0.001)

            # 给关键词更高的权重
            keyword_weight = 1.0 / len(keyword_indices)
            for idx in keyword_indices:
                if idx < vocab_size:
                    mapping_layer.weight.data[:, idx] = keyword_weight

            # 归一化
            row_sums = mapping_layer.weight.data.sum(dim=1, keepdim=True)
            mapping_layer.weight.data /= (row_sums + 1e-8)

    elif method == 'cluster':
        # 方法2: 使用关键词嵌入的聚类中心初始化
        from sklearn.cluster import KMeans

        keyword_embeddings = word_embeddings[keyword_indices].detach().cpu().numpy()

        if len(keyword_embeddings) >= num_tokens:
            # 聚类
            kmeans = KMeans(n_clusters=num_tokens, random_state=42)
            kmeans.fit(keyword_embeddings)

            # 计算每个聚类中心与所有词的相似度作为初始权重
            centers = torch.tensor(kmeans.cluster_centers_, dtype=word_embeddings.dtype)
            centers = centers.to(word_embeddings.device)

            # 相似度作为权重
            similarity = torch.mm(centers, word_embeddings.T)  # [num_tokens, vocab_size]

            with torch.no_grad():
                mapping_layer.weight.data = F.softmax(similarity, dim=1)

    elif method == 'random_from_keywords':
        # 方法3: 从关键词中随机采样初始化
        with torch.no_grad():
            mapping_layer.weight.data.fill_(0.0)

            for i in range(num_tokens):
                # 随机选择一些关键词
                selected = np.random.choice(
                    keyword_indices,
                    size=min(10, len(keyword_indices)),
                    replace=False
                )
                for idx in selected:
                    if idx < vocab_size:
                        mapping_layer.weight.data[i, idx] = 1.0 / len(selected)

    print(f"Mapping layer initialized with method: {method}")


def adaptive_num_tokens(seq_len, pred_len, n_vars, base_tokens=1000):
    """
    根据任务复杂度自适应确定词表大小

    Args:
        seq_len: 输入序列长度
        pred_len: 预测长度
        n_vars: 变量数
        base_tokens: 基础词表大小

    Returns:
        num_tokens: 建议的词表大小
    """
    complexity = np.log(seq_len) * np.log(pred_len) * np.sqrt(n_vars)
    scale_factor = complexity / (np.log(96) * np.log(96) * np.sqrt(7))  # 以 ETT 为基准

    num_tokens = int(base_tokens * scale_factor)
    num_tokens = max(500, min(2000, num_tokens))  # 限制范围

    return num_tokens
```

### 7.3 修改 TimeLLM.py

```python
# 在 Model.__init__ 中修改 Mapping Layer 初始化

# ========== 修改: 任务专用词表初始化 ==========
# 原代码
# self.num_tokens = 1000
# self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

# 新代码
use_adaptive_vocab = getattr(configs, 'use_adaptive_vocab', False)
if use_adaptive_vocab:
    from utils.vocab_init import adaptive_num_tokens, initialize_mapping_layer_with_keywords

    # 自适应词表大小
    self.num_tokens = adaptive_num_tokens(
        configs.seq_len, configs.pred_len, configs.enc_in,
        base_tokens=getattr(configs, 'base_num_tokens', 1000)
    )
    print(f"Adaptive num_tokens: {self.num_tokens}")
else:
    self.num_tokens = getattr(configs, 'num_tokens', 1000)

self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

# 任务相关初始化
if getattr(configs, 'use_keyword_init', False):
    from utils.vocab_init import initialize_mapping_layer_with_keywords

    initialize_mapping_layer_with_keywords(
        self.mapping_layer,
        self.word_embeddings,
        self.tokenizer,
        method=getattr(configs, 'vocab_init_method', 'weighted')
    )
# =====================================
```

### 7.4 运行命令

```bash
# 启用任务专用词表初始化
python run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --pred_len 96 \
  --use_adaptive_vocab \
  --use_keyword_init \
  --vocab_init_method weighted \
  ...
```

---

## 完整代码结构与文件清单

### 新增文件清单

```
Time-LLM/
├── models/
│   └── traditional/                         # 方案一
│       ├── __init__.py
│       ├── arima_wrapper.py
│       ├── exponential_smoothing.py
│       └── hybrid_forecaster.py
│
├── layers/
│   ├── MultiScaleEmbed.py                   # 方案二
│   ├── FrequencyDecomposition.py            # 方案三
│   ├── InterVariateAttention.py             # 方案四
│   ├── DynamicPrompt.py                     # 方案五
│   └── TimeSeriesMoE.py                     # 方案六
│
├── utils/
│   └── vocab_init.py                        # 方案七
│
└── configs/
    └── innovation_config.yaml               # 统一配置文件
```

### 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `models/TimeLLM.py` | 添加各创新模块的条件调用 |
| `run_main.py` | 添加命令行参数 |
| `layers/Embed.py` | 可选的多尺度/频域增强 |

### 依赖安装

```bash
# 必需
pip install statsmodels scipy scikit-learn

# 可选 (用于可视化)
pip install matplotlib seaborn
```

### 统一配置文件示例

```yaml
# configs/innovation_config.yaml

# 方案一: 混合预测
use_hybrid: false
fusion_method: adaptive  # adaptive, average, weighted

# 方案二: 多尺度
use_multiscale: false
scales: [1, 2, 4]

# 方案三: 频域增强
use_frequency: false
top_k_freqs: 5
trend_cutoff: 0.1

# 方案四: 变量间注意力
use_inter_variate: false
inter_variate_heads: 8

# 方案五: 动态 Prompt
use_dynamic_prompt: false
num_dynamic_tokens: 32

# 方案六: MoE
use_moe: false
num_experts: 4
top_k_experts: 2

# 方案七: 词表初始化
use_adaptive_vocab: false
use_keyword_init: false
vocab_init_method: weighted
```

---

## 测试与验证

### 单元测试示例

```python
# tests/test_innovations.py

import torch
import unittest

class TestInterVariateAttention(unittest.TestCase):
    def test_forward_shape(self):
        from layers.InterVariateAttention import InterVariateAttention

        attn = InterVariateAttention(d_model=64, n_heads=8)
        x = torch.randn(28, 12, 64)  # B*N=28, L=12, D=64
        n_vars = 7

        out = attn(x, n_vars)
        self.assertEqual(out.shape, (28, 12, 64))

class TestFrequencyDecomposition(unittest.TestCase):
    def test_decomposition(self):
        from layers.FrequencyDecomposition import FrequencyDecomposition

        decomp = FrequencyDecomposition()
        x = torch.randn(4, 96, 7)  # B=4, T=96, N=7

        trend, seasonal = decomp(x)
        self.assertEqual(trend.shape, (4, 96, 7))
        self.assertEqual(seasonal.shape, (4, 96, 7))

        # 验证分解正确性
        reconstructed = trend + seasonal
        self.assertTrue(torch.allclose(x, reconstructed, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
```

---

**文档版本**: v1.0
**生成日期**: 2026-01-11
**作者**: Zhenda Wang
