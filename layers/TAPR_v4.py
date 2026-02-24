"""
TAPR_v4 — PolyphaseMultiScale (DM) + ExpertVotingFusion (C2F)

v4 core changes from v3:
  - DM: avg-pool downsampling -> unified k=4 polyphase decomposition
        + signal transforms differentiate scales (not downsampling rates)
  - C2F: simple weighted sum -> two-stage fusion:
        Stage 1: intra-scale MAD voting (outlier branch removal)
        Stage 2: cross-scale DWT trend constraint + reliability penalty

Design references:
  - TimeMixer (ICLR 2024): multi-scale independent prediction + additive fusion
  - N-HiTS (AAAI 2023): interpolation coefficients for uniform pred_len alignment
  - Polyphase Decomposition: classic signal processing for lossless multi-rate analysis
  - MAD (Median Absolute Deviation): robust outlier detection
  - Haar DWT: efficient wavelet decomposition for trend extraction
  - Autoformer (NeurIPS 2021): series decomposition (diff / smoothing / trend)
"""

import math

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
# DWT helper (Haar wavelet)
# ---------------------------------------------------------------------------

def dwt_haar_1d(x):
    """One-level Haar wavelet decomposition.

    Args:
        x: [B, T, N]
    Returns:
        approx: [B, T//2, N] — low-frequency approximation
        detail: [B, T//2, N] — high-frequency detail
    """
    if x.shape[1] % 2 != 0:
        x = x[:, :-1, :]
    approx = (x[:, 0::2, :] + x[:, 1::2, :]) / math.sqrt(2)
    detail = (x[:, 0::2, :] - x[:, 1::2, :]) / math.sqrt(2)
    return approx, detail


# ---------------------------------------------------------------------------
# DM v4: PolyphaseMultiScale
# ---------------------------------------------------------------------------

class PolyphaseMultiScale(nn.Module):
    """DM v4: produces independent branch predictions at multiple scales
    using polyphase decomposition instead of average-pool downsampling.

    Scale 1 (k=1): handled by baseline Time-LLM (not in this module)
    Scale 2 (k=2): diff transform, 2 polyphase branches
    Scale 3 (k=4): identity transform, 4 polyphase branches
    Scale 4 (k=4): smooth transform, 4 polyphase branches
    Scale 5 (k=4): trend transform, 4 polyphase branches

    Total branches: 2 + 4 + 4 + 4 = 14
    """

    SCALE_CFG = [
        # (k, transform_name)
        (2, 'diff'),       # S2: k=2
        (4, 'identity'),   # S3: k=4
        (4, 'smooth'),     # S4: k=4
        (4, 'trend'),      # S5: k=4
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

        for k, _ in self.scale_cfgs:
            # Polyphase: effective length is seq_len // k
            eff_len = max(seq_len // k, patch_len + 1)

            # Compute num_patches conservatively
            num_patches = int((eff_len - patch_len) / stride + 2)
            head_nf = d_model * num_patches

            # PatchEmbedding (Conv1d: patch_len -> d_model)
            patch_embed = nn.Conv1d(
                patch_len, d_model, kernel_size=3, padding=1,
                padding_mode='circular', bias=False)
            self.patch_embeddings.append(patch_embed)

            # ScaleEncoder (shared across all k branches of this scale)
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

    def forward(self, x_normed, pred_s1):
        """
        Args:
            x_normed: [B, T, N] — normalized input (after instance norm)
            pred_s1:  [B, pred_len, N] — baseline prediction (already computed)

        Returns:
            pred_s1: [B, pred_len, N] — passed through unchanged
            branch_preds_dict: dict {scale_idx: list of k [B, pred_len, N]}
        """
        B, T, N = x_normed.shape

        branch_preds_dict = {}

        for idx, (k, transform_name) in enumerate(self.scale_cfgs):
            branch_preds = []

            for p in range(k):
                # 1. Polyphase extraction: x[:, p::k, :]
                x_p = x_normed[:, p::k, :]  # [B, ~T/k, N]

                # 2. Apply signal transform
                fwd_fn, inv_fn = TRANSFORM_REGISTRY[transform_name]
                x_transformed, aux = fwd_fn(x_p)  # [B, T_trans, N]
                T_trans = x_transformed.shape[1]

                # Handle case where transformed sequence is too short
                if T_trans < self.patch_len:
                    pad_needed = self.patch_len - T_trans + 1
                    x_transformed = F.pad(
                        x_transformed.permute(0, 2, 1),
                        (0, pad_needed), mode='replicate'
                    ).permute(0, 2, 1)
                    T_trans = x_transformed.shape[1]

                # 3. Channel-independent: reshape to [B*N, T_trans, 1]
                x_ci = x_transformed.permute(0, 2, 1).contiguous().reshape(B * N, T_trans, 1)

                # 4. Create patches
                patches, num_patches = _create_patches(x_ci, self.patch_len, self.stride)

                # 5. Patch embedding (shared within this scale)
                patch_embed = self.patch_embeddings[idx]
                embedded = patch_embed(patches.permute(0, 2, 1))  # [B*N, d_model, num_patches]
                embedded = embedded.permute(0, 2, 1)  # [B*N, num_patches, d_model]
                embedded = self.dropout(embedded)

                # 6. Align patch count via interpolation if needed
                encoder = self.scale_encoders[idx]
                expected_patches = encoder.linear.in_features // self.d_model
                cur_patches = embedded.shape[1]

                if cur_patches != expected_patches:
                    embedded_t = embedded.permute(0, 2, 1)  # [B*N, D, cur]
                    embedded_t = F.interpolate(
                        embedded_t, size=expected_patches, mode='linear',
                        align_corners=False)
                    embedded = embedded_t.permute(0, 2, 1)  # [B*N, expected, D]

                # 7. ScaleEncoder (shared within this scale) -> predict
                pred_p = encoder(embedded)  # [B*N, pred_len]

                # 8. Reshape back to [B, pred_len, N]
                pred_p = pred_p.reshape(B, N, self.pred_len).permute(0, 2, 1).contiguous()
                branch_preds.append(pred_p)

            branch_preds_dict[idx] = branch_preds

        return pred_s1, branch_preds_dict


# ---------------------------------------------------------------------------
# C2F v4: ExpertVotingFusion
# ---------------------------------------------------------------------------

class ExpertVotingFusion(nn.Module):
    """Fuses multi-scale predictions via two-stage process:

    Stage 1 — Intra-scale expert voting:
      For each scale's k branches, use MAD to detect outlier branches,
      then average non-outlier branches into a single consolidated prediction.
      Track correction rates for reliability assessment.

    Stage 2 — Cross-scale trend constraint:
      DWT decompose each consolidated prediction, compare trends against
      the coarsest scale (S5) via cosine similarity, penalize weights of
      inconsistent or unreliable scales, then weighted-sum.
    """

    def __init__(self, n_scales, pred_len, decay_factor=0.5,
                 mad_threshold=3.0, reliability_threshold=0.3,
                 reliability_penalty=0.3):
        """
        Args:
            n_scales: total number of scales (including baseline)
            pred_len: prediction horizon length
            decay_factor: weight decay for trend-inconsistent scales
            mad_threshold: MAD multiplier for outlier detection
            reliability_threshold: correction rate threshold for unreliable flag
            reliability_penalty: weight multiplier for unreliable scales
        """
        super().__init__()

        self.n_scales = n_scales
        self.pred_len = pred_len
        self.decay_factor = decay_factor
        self.mad_threshold = mad_threshold
        self.reliability_threshold = reliability_threshold
        self.reliability_penalty = reliability_penalty

        # "Weight Matrix" W: [n_scales, pred_len]
        self.weight_matrix = nn.Parameter(
            torch.ones(n_scales, pred_len) / n_scales
        )

        # Non-parameter tracking counters
        self.correction_counts = {}
        self.total_counts = {}

        # Cache for consolidated_preds (set during forward)
        self._consolidated_preds = None

    def forward(self, pred_s1, branch_preds_dict, training=True):
        """
        Args:
            pred_s1: [B, pred_len, N] — baseline prediction
            branch_preds_dict: dict {scale_idx: list of k [B, pred_len, N]}
            training: whether in training mode

        Returns:
            weighted_pred: [B, pred_len, N]
        """
        # Stage 1: Intra-scale voting
        consolidated = [pred_s1]  # S1 needs no voting
        reliability_flags = [False]  # S1 is always reliable

        for s_idx in sorted(branch_preds_dict.keys()):
            branches = branch_preds_dict[s_idx]
            k = len(branches)

            if k <= 2:
                # For k=2, MAD is degenerate (only 2 points); use simple mean
                stacked = torch.stack(branches, dim=0)
                c_pred = stacked.mean(dim=0)
                n_outliers = 0
            else:
                c_pred, n_outliers = self._intra_scale_vote(branches)

            consolidated.append(c_pred)

            # Update tracking counters
            if s_idx not in self.correction_counts:
                self.correction_counts[s_idx] = 0
                self.total_counts[s_idx] = 0

            if training:
                self.correction_counts[s_idx] += n_outliers
                self.total_counts[s_idx] += k

            # Compute reliability flag
            if self.total_counts[s_idx] > 0:
                rate = self.correction_counts[s_idx] / self.total_counts[s_idx]
                reliability_flags.append(rate > self.reliability_threshold)
            else:
                reliability_flags.append(False)

        # Cache for later use (auxiliary loss, AKG)
        self._consolidated_preds = consolidated

        # Stage 2: Cross-scale trend constraint
        weighted_pred = self._cross_scale_trend_constraint(
            consolidated, reliability_flags)
        return weighted_pred

    def get_consolidated_preds(self):
        """Return cached consolidated predictions from last forward pass."""
        return self._consolidated_preds

    def _intra_scale_vote(self, branches):
        """MAD-based expert voting for k >= 3 branches.

        Args:
            branches: list of k [B, pred_len, N]
        Returns:
            consolidated: [B, pred_len, N]
            num_outliers: int
        """
        stacked = torch.stack(branches, dim=0)  # [k, B, P, N]

        # 1. Median prediction as reference
        median_pred = stacked.median(dim=0).values  # [B, P, N]

        # 2. Per-branch deviation from median
        deviations = (stacked - median_pred.unsqueeze(0)).abs()  # [k, B, P, N]

        # 3. MAD (Median Absolute Deviation)
        MAD = deviations.median(dim=0).values  # [B, P, N]

        # 4. Outlier detection: aggregate deviation per branch
        per_branch_deviation = deviations.mean(dim=(1, 2, 3))  # [k]
        MAD_scalar = MAD.mean()

        is_outlier = per_branch_deviation > self.mad_threshold * MAD_scalar  # [k] bool

        # 5. Average non-outlier branches
        valid_mask = ~is_outlier
        if valid_mask.sum() == 0:
            # All branches are outliers -> fallback to full average
            consolidated = stacked.mean(dim=0)
        else:
            consolidated = stacked[valid_mask].mean(dim=0)

        num_outliers = is_outlier.sum().item()
        return consolidated, num_outliers

    def _cross_scale_trend_constraint(self, consolidated_preds, reliability_flags):
        """Stage 2: DWT trend constraint + reliability penalty.

        Args:
            consolidated_preds: list of n_scales [B, pred_len, N]
            reliability_flags: list of n_scales bool
        Returns:
            weighted_pred: [B, pred_len, N]
        """
        n = len(consolidated_preds)
        B = consolidated_preds[0].shape[0]

        # DWT decompose each consolidated prediction
        approx_list = []
        for pred in consolidated_preds:
            approx, _ = dwt_haar_1d(pred)
            approx_list.append(approx)

        # S5 (last) as reference trend
        ref_trend = approx_list[-1]

        # Compute trend consistency scores
        consistency_scores = []
        for s in range(n):
            score = F.cosine_similarity(
                approx_list[s].reshape(B, -1),
                ref_trend.reshape(B, -1),
                dim=-1
            ).mean()
            consistency_scores.append(score)

        # Softmax weights (differentiable)
        weights = F.softmax(self.weight_matrix, dim=0)  # [n_scales, pred_len]

        # Build penalty multiplier
        penalty = torch.ones_like(weights)
        for s in range(n):
            # Trend inconsistency penalty (trend clearly opposite)
            if consistency_scores[s] < -0.3:
                penalty[s] = penalty[s] * self.decay_factor
            # Reliability penalty
            if reliability_flags[s]:
                penalty[s] = penalty[s] * self.reliability_penalty

        # Apply penalty and re-normalize
        weights = weights * penalty
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        # Weighted fusion
        stacked = torch.stack(consolidated_preds, dim=0)  # [n, B, P, N]
        w = weights.unsqueeze(1).unsqueeze(-1)  # [n, 1, P, 1]
        weighted_pred = (w * stacked).sum(dim=0)  # [B, P, N]

        return weighted_pred

    def consistency_loss(self, consolidated_preds):
        """Wavelet-based consistency loss.

        L_consist = mean(max(0, -cosine_sim(approx_s, approx_S5))) for s=0..S-2

        Penalizes scales whose trend is clearly opposite to the coarsest scale.
        """
        device = consolidated_preds[0].device
        n = len(consolidated_preds)
        if n < 2:
            return torch.tensor(0.0, device=device)

        # DWT decompose all
        approx_list = []
        for pred in consolidated_preds:
            approx, _ = dwt_haar_1d(pred)
            approx_list.append(approx)

        ref_trend = approx_list[-1]  # S5 reference
        B = ref_trend.shape[0]

        loss = torch.tensor(0.0, device=device)
        for s in range(n - 1):
            cos_sim = F.cosine_similarity(
                approx_list[s].reshape(B, -1),
                ref_trend.reshape(B, -1),
                dim=-1
            ).mean()
            loss = loss + torch.clamp(-cos_sim, min=0.0)

        return loss / max(n - 1, 1)

    def expert_loss(self):
        """Expert correction loss: mean correction rate across scales.

        Non-gradient (value from tracking counters), returns detached tensor.
        Penalizes scales with high correction rates, encouraging branch consistency.
        """
        device = self.weight_matrix.device
        if not self.total_counts:
            return torch.tensor(0.0, device=device)

        rates = []
        for s_idx in sorted(self.total_counts.keys()):
            if self.total_counts[s_idx] > 0:
                rate = self.correction_counts[s_idx] / self.total_counts[s_idx]
                rates.append(rate)

        if not rates:
            return torch.tensor(0.0, device=device)

        return torch.tensor(sum(rates) / len(rates), device=device, dtype=torch.float32)
