"""
TAPR_v3 — Decomposable Multi-scale (DM) + Coarse-to-Fine Fusion (C2F)

v3 core change: modules now produce independent predictions that are fused
via learnable per-timestep matrices, instead of the v2 "observer" pattern.

Design references:
  - TimeMixer (ICLR 2024): multi-scale independent prediction + additive fusion
  - N-HiTS (AAAI 2023): interpolation coefficients for uniform pred_len alignment
  - Scaleformer (ICLR 2023): cross-scale normalization to prevent distribution drift
  - C2FAR (NeurIPS 2022): coarse-to-fine conditional probability decomposition
  - Autoformer (NeurIPS 2021): series decomposition (diff / smoothing / trend)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Signal transforms (forward + inverse) for each scale
# ---------------------------------------------------------------------------

class SignalTransform:
    """Stateless signal transform / inverse-transform pairs."""

    @staticmethod
    def identity(x):
        return x, None

    @staticmethod
    def identity_inv(x, _aux):
        return x

    @staticmethod
    def diff(x):
        """First-order differencing: captures rate-of-change."""
        offset = x[:, :1, :]  # keep first value for inverse
        return x[:, 1:, :] - x[:, :-1, :], offset

    @staticmethod
    def diff_inv(x, offset):
        """Cumulative sum + offset to invert differencing."""
        return torch.cumsum(torch.cat([offset, x], dim=1), dim=1)

    @staticmethod
    def smooth(x, kernel_size=25):
        """Moving-average smoothing (de-noising)."""
        B, T, N = x.shape
        if T < kernel_size:
            kernel_size = max(3, T)
        # pad symmetrically
        pad = kernel_size // 2
        x_t = x.permute(0, 2, 1)  # [B, N, T]
        x_padded = F.pad(x_t, (pad, pad), mode='replicate')
        weight = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
        # apply per-channel
        smoothed = F.conv1d(
            x_padded.reshape(B * N, 1, -1), weight, padding=0
        ).reshape(B, N, -1)[:, :, :T].permute(0, 2, 1)
        return smoothed, None

    @staticmethod
    def smooth_inv(x, _aux):
        return x  # smoothing is lossy; use as-is

    @staticmethod
    def trend(x):
        """Extract macro trend via large-window moving average."""
        B, T, N = x.shape
        kernel_size = max(3, T // 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        x_t = x.permute(0, 2, 1)
        x_padded = F.pad(x_t, (pad, pad), mode='replicate')
        weight = torch.ones(1, 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
        trend_out = F.conv1d(
            x_padded.reshape(B * N, 1, -1), weight, padding=0
        ).reshape(B, N, -1)[:, :, :T].permute(0, 2, 1)
        return trend_out, None

    @staticmethod
    def trend_inv(x, _aux):
        return x


# Maps: (forward_fn, inverse_fn)
TRANSFORM_REGISTRY = {
    'identity': (SignalTransform.identity, SignalTransform.identity_inv),
    'diff':     (SignalTransform.diff,     SignalTransform.diff_inv),
    'smooth':   (SignalTransform.smooth,   SignalTransform.smooth_inv),
    'trend':    (SignalTransform.trend,    SignalTransform.trend_inv),
}


# ---------------------------------------------------------------------------
# Lightweight per-scale encoder  ("parameter matrix" Theta_s)
# ---------------------------------------------------------------------------

class ScaleEncoder(nn.Module):
    """Lightweight encoder for a single auxiliary scale.

    Each scale has its own PatchEmbedding -> 2-layer Conv -> FlattenHead.
    """

    def __init__(self, d_model, patch_len, stride, head_nf, pred_len, dropout=0.1):
        super().__init__()
        # Conv feature extractor
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # FlattenHead
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_nf, pred_len)

    def forward(self, x):
        """
        Args:
            x: [B*N, num_patches, d_model]
        Returns:
            pred: [B*N, pred_len]
        """
        # x: [B*N, L, D] -> [B*N, D, L]
        h = x.permute(0, 2, 1)
        h = self.dropout(self.act(self.conv1(h)))
        h = self.dropout(self.act(self.conv2(h)))
        # flatten & project
        h = self.flatten(h)  # [B*N, D*L]
        return self.linear(h)  # [B*N, pred_len]


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _create_patches(x, patch_len, stride):
    """Create patches from [B*N, T, 1] input.

    Returns:
        patches: [B*N, num_patches, patch_len]
        num_patches: int
    """
    B, T, _ = x.shape
    # Replication padding at the end
    pad_len = stride  # same as baseline
    x_padded = F.pad(x.squeeze(-1), (0, pad_len), mode='replicate')  # [B, T+pad]
    patches = x_padded.unfold(dimension=-1, size=patch_len, step=stride)  # [B, num_patches, patch_len]
    return patches, patches.shape[1]


# ---------------------------------------------------------------------------
# DM: Decomposable Multi-scale module
# ---------------------------------------------------------------------------

class DecomposableMultiScale(nn.Module):
    """DM module: produces independent predictions at multiple scales.

    Scale 1 (ds=1):  handled by baseline Time-LLM (not in this module)
    Scale 2 (ds=2):  diff transform
    Scale 3 (ds=4):  identity (raw downsampled)
    Scale 4 (ds=8):  smooth transform
    Scale 5 (ds=16): trend transform
    """

    SCALE_CFG = [
        # (downsample_rate, transform_name)
        (2,  'diff'),
        (4,  'identity'),
        (8,  'smooth'),
        (16, 'trend'),
    ]

    def __init__(self, d_model, patch_len, stride, pred_len, seq_len,
                 n_aux_scales=4, dropout=0.1):
        """
        Args:
            d_model: patch embedding dimension
            patch_len: patch length (default 16)
            stride: patch stride (default 8)
            pred_len: prediction length
            seq_len: input sequence length
            n_aux_scales: number of auxiliary scales (default 4, for S2-S5)
            dropout: dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.n_aux_scales = min(n_aux_scales, len(self.SCALE_CFG))

        self.scale_cfgs = self.SCALE_CFG[:self.n_aux_scales]

        # Build per-scale components
        self.patch_embeddings = nn.ModuleList()
        self.scale_encoders = nn.ModuleList()

        for ds_rate, _ in self.scale_cfgs:
            # After downsampling + optional transform, effective length
            eff_len = max(seq_len // ds_rate, patch_len + 1)  # at least one patch

            # For diff transform, length decreases by 1
            # We compute num_patches conservatively
            num_patches = int((eff_len - patch_len) / stride + 2)
            head_nf = d_model * num_patches

            # TokenEmbedding for patches (Conv1d: patch_len -> d_model)
            # Input: [B, num_patches, patch_len] -> permute -> Conv1d -> permute back
            patch_embed = nn.Conv1d(
                patch_len, d_model, kernel_size=3, padding=1,
                padding_mode='circular', bias=False)
            self.patch_embeddings.append(patch_embed)

            # ScaleEncoder with per-scale head
            encoder = ScaleEncoder(
                d_model=d_model,
                patch_len=patch_len,
                stride=stride,
                head_nf=head_nf,
                pred_len=pred_len,
                dropout=dropout,
            )
            self.scale_encoders.append(encoder)

        self.dropout = nn.Dropout(dropout)

    def _downsample(self, x, rate):
        """Average-pool downsampling along time dimension.
        Args:
            x: [B, T, N]
            rate: downsampling rate
        Returns:
            x_ds: [B, T//rate, N]
        """
        if rate <= 1:
            return x
        B, T, N = x.shape
        # Truncate to multiple of rate
        trunc_T = (T // rate) * rate
        x_trunc = x[:, :trunc_T, :]
        x_ds = x_trunc.reshape(B, T // rate if trunc_T == T else trunc_T // rate, rate, N).mean(dim=2)
        return x_ds

    def forward(self, x_normed, pred_s1):
        """
        Args:
            x_normed: [B, T, N] — normalized input (after instance norm)
            pred_s1:  [B, pred_len, N] — baseline prediction (already computed)

        Returns:
            scale_preds: list of [B, pred_len, N], length = 1 + n_aux_scales
                         index 0 = pred_s1 (baseline), index 1..n = auxiliary scales
        """
        B, T, N = x_normed.shape
        device = x_normed.device
        dtype = x_normed.dtype

        scale_preds = [pred_s1]  # S1 = baseline

        for idx, (ds_rate, transform_name) in enumerate(self.scale_cfgs):
            # 1. Downsample
            x_ds = self._downsample(x_normed, ds_rate)  # [B, T_ds, N]

            # 2. Apply signal transform
            fwd_fn, inv_fn = TRANSFORM_REGISTRY[transform_name]
            x_transformed, aux = fwd_fn(x_ds)  # [B, T_trans, N]
            T_trans = x_transformed.shape[1]

            # Handle case where transformed sequence is too short
            if T_trans < self.patch_len:
                # Fallback: pad to minimum length
                pad_needed = self.patch_len - T_trans + 1
                x_transformed = F.pad(x_transformed.permute(0, 2, 1),
                                       (0, pad_needed), mode='replicate').permute(0, 2, 1)
                T_trans = x_transformed.shape[1]

            # 3. Channel-independent: reshape to [B*N, T_trans, 1]
            x_ci = x_transformed.permute(0, 2, 1).contiguous().reshape(B * N, T_trans, 1)

            # 4. Create patches
            patches, num_patches = _create_patches(x_ci, self.patch_len, self.stride)
            # patches: [B*N, num_patches, patch_len]

            # 5. Patch embedding (lightweight)
            # patches: [B*N, num_patches, patch_len]
            # Conv1d expects [B, in_channels, seq_len] = [B*N, patch_len, num_patches]
            patch_embed = self.patch_embeddings[idx]
            embedded = patch_embed(patches.permute(0, 2, 1))  # [B*N, d_model, num_patches]
            embedded = embedded.permute(0, 2, 1)  # [B*N, num_patches, d_model]
            embedded = self.dropout(embedded)

            # 6. Encode -> predict
            encoder = self.scale_encoders[idx]
            # encoder expects [B*N, num_patches, d_model]
            # but head_nf was computed with a fixed num_patches, so we need to match
            expected_patches = encoder.linear.in_features // self.d_model
            cur_patches = embedded.shape[1]

            if cur_patches != expected_patches:
                # Interpolate to match expected size
                embedded_t = embedded.permute(0, 2, 1)  # [B*N, D, cur]
                embedded_t = F.interpolate(embedded_t, size=expected_patches, mode='linear',
                                           align_corners=False)
                embedded = embedded_t.permute(0, 2, 1)  # [B*N, expected, D]

            pred_scale = encoder(embedded)  # [B*N, pred_len]

            # 7. Reshape back to [B, pred_len, N]
            pred_scale = pred_scale.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()

            scale_preds.append(pred_scale)

        return scale_preds  # list of n_scales [B, pred_len, N]


# ---------------------------------------------------------------------------
# C2F: Coarse-to-Fine Fusion  ("weight matrix" W)
# ---------------------------------------------------------------------------

class CoarseToFineFusion(nn.Module):
    """Fuses multi-scale predictions via a learnable per-timestep weight matrix.

    Training:   soft consistency (KL-based loss between scale trend distributions)
    Inference:  hard consistency (decay weights of scales that disagree with coarsest)
    """

    def __init__(self, n_scales, pred_len, decay_factor=0.5, consistency_mode='hybrid'):
        """
        Args:
            n_scales: total number of scales (including baseline)
            pred_len: prediction horizon length
            decay_factor: weight decay for inconsistent scales at inference
            consistency_mode: 'soft' | 'hard' | 'hybrid'
        """
        super().__init__()

        self.n_scales = n_scales
        self.pred_len = pred_len
        self.decay_factor = decay_factor
        self.consistency_mode = consistency_mode

        # "Weight Matrix" W: [n_scales, pred_len]
        # Initialized uniformly
        self.weight_matrix = nn.Parameter(
            torch.ones(n_scales, pred_len) / n_scales
        )

    def forward(self, scale_preds, training=True):
        """
        Args:
            scale_preds: list of [B, pred_len, N], length = n_scales
            training: whether in training mode

        Returns:
            weighted_pred: [B, pred_len, N]
        """
        # Stack: [n_scales, B, pred_len, N]
        stacked = torch.stack(scale_preds, dim=0)

        # Softmax over scales for each timestep
        weights = F.softmax(self.weight_matrix, dim=0)  # [n_scales, pred_len]

        # Apply hard consistency at inference
        if not training and self.consistency_mode in ('hard', 'hybrid'):
            weights = self._apply_hard_consistency(scale_preds, weights)

        # Weighted fusion: [n_scales, pred_len] -> broadcast over [n_scales, B, pred_len, N]
        w = weights.unsqueeze(1).unsqueeze(-1)  # [n_scales, 1, pred_len, 1]
        weighted_pred = (w * stacked).sum(dim=0)  # [B, pred_len, N]

        return weighted_pred

    def _apply_hard_consistency(self, scale_preds, weights):
        """At inference, decay weights of scales whose trend disagrees with the coarsest scale."""
        with torch.no_grad():
            # Coarsest scale = last element (S5)
            coarse_trend = scale_preds[-1][:, -1, :] - scale_preds[-1][:, 0, :]  # [B, N]
            coarse_sign = (coarse_trend > 0).float()  # [B, N] — 1 for up, 0 for down

            adjusted_weights = weights.clone()
            for s in range(self.n_scales - 1):
                fine_trend = scale_preds[s][:, -1, :] - scale_preds[s][:, 0, :]
                fine_sign = (fine_trend > 0).float()
                # Fraction of (batch, var) that disagree
                disagreement = (fine_sign != coarse_sign).float().mean()
                if disagreement > 0.5:
                    adjusted_weights[s] *= self.decay_factor

            # Re-normalize
            adjusted_weights = adjusted_weights / (adjusted_weights.sum(dim=0, keepdim=True) + 1e-8)
            return adjusted_weights

    def consistency_loss(self, scale_preds):
        """Soft consistency loss: encourage scales to agree on trend direction.

        Uses KL divergence between trend distributions of different scale pairs.
        Returns scalar loss.
        """
        device = scale_preds[0].device
        n = len(scale_preds)
        if n < 2:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Compute trend direction probabilities for each scale
                trend_i = self._trend_distribution(scale_preds[i])  # [B, N, 3]
                trend_j = self._trend_distribution(scale_preds[j])  # [B, N, 3]
                # Symmetric KL
                kl_ij = F.kl_div(trend_i.log(), trend_j, reduction='batchmean')
                kl_ji = F.kl_div(trend_j.log(), trend_i, reduction='batchmean')
                loss = loss + (kl_ij + kl_ji) / 2
                count += 1

        return loss / max(count, 1)

    def _trend_distribution(self, pred):
        """Convert predictions to trend direction soft-distribution.

        Args:
            pred: [B, pred_len, N]
        Returns:
            dist: [B, N, 3] — probabilities for [down, flat, up]
        """
        # Overall trend: end - start
        diff = pred[:, -1, :] - pred[:, 0, :]  # [B, N]
        # Magnitude relative to std
        std = pred.std(dim=1).clamp(min=1e-6)  # [B, N]
        normalized_diff = diff / std

        # 3-class soft labels via sigmoid gates
        down_prob = torch.sigmoid(-normalized_diff - 0.5)
        up_prob = torch.sigmoid(normalized_diff - 0.5)
        flat_prob = 1.0 - down_prob - up_prob
        flat_prob = flat_prob.clamp(min=1e-6)

        dist = torch.stack([down_prob, flat_prob, up_prob], dim=-1)  # [B, N, 3]
        # Normalize to valid distribution
        dist = dist / dist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return dist
