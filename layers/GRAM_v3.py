"""
GRAM_v3 — Pattern-Value Dual Retriever (PVDR) + Adaptive Knowledge Gating (AKG)

v3 core change: multi-scale retrieval with per-scale memory banks, extreme-data
detection, and dual-matrix (connection + fusion) gating that directly produces
the final prediction.

Design references:
  - RAFT (ICML 2025): retrieve similar history + subsequent values as reference
  - TS-RAG (NeurIPS 2025): ARM module for adaptive retrieval/current mixing
  - TFT (IJF 2021): GLU gated residual network for selective fusion
  - SJTU 2025 retrieval-augmented gating framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Lightweight pattern encoder (shared across scales)
# ---------------------------------------------------------------------------

class LightweightPatternEncoder(nn.Module):
    """Encodes a 1-D sequence segment into a compact representation vector."""

    def __init__(self, d_repr=64):
        super().__init__()
        # 5 statistical features: mean, std, max, min, trend
        self.proj = nn.Sequential(
            nn.Linear(5, d_repr),
            nn.LayerNorm(d_repr),
        )

    def forward(self, x):
        """
        Args:
            x: [B, L] or [B, L, D] — if 3D, mean over D first
        Returns:
            key: [B, d_repr]
        """
        if x.dim() == 3:
            x_flat = x.mean(dim=-1)  # [B, L]
        else:
            x_flat = x

        mean_val = x_flat.mean(dim=1, keepdim=True)
        std_val = x_flat.std(dim=1, keepdim=True).clamp(min=1e-6)
        max_val = x_flat.max(dim=1, keepdim=True)[0]
        min_val = x_flat.min(dim=1, keepdim=True)[0]
        trend_val = x_flat[:, -1:] - x_flat[:, :1]

        stats = torch.cat([mean_val, std_val, max_val, min_val, trend_val], dim=1)
        return self.proj(stats)


# ---------------------------------------------------------------------------
# Extreme data detector
# ---------------------------------------------------------------------------

class ExtremeDetector(nn.Module):
    """Detects extreme data points using z-score (3-sigma rule)."""

    def __init__(self, sigma_threshold=3.0):
        super().__init__()
        self.sigma_threshold = sigma_threshold

    def forward(self, x):
        """
        Args:
            x: [B, T, N]
        Returns:
            has_extreme: [B] bool tensor — True if any extreme point detected
            z_scores: [B, T, N]
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        z_scores = (x - mean) / std
        has_extreme = (z_scores.abs() > self.sigma_threshold).any(dim=1).any(dim=1)  # [B]
        return has_extreme, z_scores


# ---------------------------------------------------------------------------
# Multi-scale memory bank
# ---------------------------------------------------------------------------

class MultiScaleMemoryBank(nn.Module):
    """Maintains independent memory banks per scale.

    Each bank stores (key, value) pairs:
      - key:   [M, d_repr] — pattern representation
      - value: [M, pred_len, n_vars] — the actual future values following that pattern
    """

    def __init__(self, n_scales, d_repr, pred_len, max_memory_per_scale=5000):
        super().__init__()
        self.n_scales = n_scales
        self.d_repr = d_repr
        self.pred_len = pred_len
        self.max_memory = max_memory_per_scale

        # Learnable per-scale similarity thresholds
        self.thresholds = nn.Parameter(torch.full((n_scales,), 0.8))

        # Register empty buffers — will be filled during build_memory
        for s in range(n_scales):
            self.register_buffer(f'keys_{s}', torch.zeros(0, d_repr))
            self.register_buffer(f'values_{s}', torch.zeros(0, pred_len))

    def get_keys(self, scale_idx):
        return getattr(self, f'keys_{scale_idx}')

    def get_values(self, scale_idx):
        return getattr(self, f'values_{scale_idx}')

    def set_bank(self, scale_idx, keys, values):
        """Set memory bank for a given scale."""
        device = self.thresholds.device
        setattr(self, f'keys_{scale_idx}', keys.to(device))
        setattr(self, f'values_{scale_idx}', values.to(device))

    def is_empty(self, scale_idx):
        return self.get_keys(scale_idx).shape[0] == 0


# ---------------------------------------------------------------------------
# PVDR: Pattern-Value Dual Retriever (multi-scale)
# ---------------------------------------------------------------------------

class PatternValueDualRetriever(nn.Module):
    """Multi-scale retrieval module.

    For each scale, retrieves top-K similar historical patterns from the
    corresponding memory bank, using cosine similarity with a learnable
    per-scale threshold.
    """

    def __init__(self, n_scales, d_repr, pred_len, n_vars, top_k=5,
                 extreme_sigma=3.0, extreme_threshold_reduction=0.2,
                 max_memory_per_scale=5000):
        super().__init__()
        self.n_scales = n_scales
        self.d_repr = d_repr
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.top_k = top_k
        self.extreme_threshold_reduction = extreme_threshold_reduction

        self.encoder = LightweightPatternEncoder(d_repr=d_repr)
        self.memory_bank = MultiScaleMemoryBank(
            n_scales=n_scales,
            d_repr=d_repr,
            pred_len=pred_len,
            max_memory_per_scale=max_memory_per_scale,
        )
        self.extreme_detector = ExtremeDetector(sigma_threshold=extreme_sigma)

    def build_memory(self, train_loader, device, downsample_rates, max_samples=5000):
        """Build memory banks for all scales from training data.

        For each (input, target) pair, we store:
          key   = encoder(downsampled_input_mean_over_vars)
          value = target_mean_over_vars (simplified for memory efficiency)
        """
        self.eval()
        all_inputs = []
        all_targets = []
        count = 0

        with torch.no_grad():
            for batch_x, batch_y, _, _ in train_loader:
                if count >= max_samples:
                    break
                batch_x = batch_x.float().to(device)  # [B, T, N]
                batch_y = batch_y.float().to(device)  # [B, T_out, N]
                all_inputs.append(batch_x)
                all_targets.append(batch_y[:, -self.pred_len:, :])  # [B, pred_len, N]
                count += batch_x.shape[0]

        if not all_inputs:
            return

        all_x = torch.cat(all_inputs, dim=0)[:max_samples]  # [M, T, N]
        all_y = torch.cat(all_targets, dim=0)[:max_samples]  # [M, pred_len, N]

        # Mean over variables for compact storage
        y_mean = all_y.mean(dim=-1)  # [M, pred_len]

        with torch.no_grad():
            for s, ds_rate in enumerate(downsample_rates):
                # Downsample input
                if ds_rate > 1:
                    T = all_x.shape[1]
                    trunc_T = (T // ds_rate) * ds_rate
                    x_ds = all_x[:, :trunc_T, :].reshape(
                        all_x.shape[0], trunc_T // ds_rate, ds_rate, all_x.shape[2]
                    ).mean(dim=2)
                else:
                    x_ds = all_x

                # Encode: mean over vars -> [M, T_ds]
                x_flat = x_ds.mean(dim=-1)  # [M, T_ds]
                keys = self.encoder(x_flat)  # [M, d_repr]

                # Normalize keys
                keys = F.normalize(keys, dim=-1)

                self.memory_bank.set_bank(s, keys, y_mean)

        self.train()

    def retrieve(self, x_normed, scale_idx, has_extreme=None):
        """Retrieve top-K similar historical patterns for one scale.

        Args:
            x_normed: [B, T_ds, N] — downsampled + normalized input at this scale
            scale_idx: which scale's memory bank to use
            has_extreme: [B] bool — whether this sample has extreme data

        Returns:
            hist_ref: [B, pred_len] — weighted average of retrieved future values
            valid_mask: [B] bool — True if retrieval succeeded
        """
        if self.memory_bank.is_empty(scale_idx):
            B = x_normed.shape[0]
            device = x_normed.device
            return torch.zeros(B, self.pred_len, device=device), \
                   torch.zeros(B, dtype=torch.bool, device=device)

        # Encode query
        query_flat = x_normed.mean(dim=-1)  # [B, T_ds]
        query_key = self.encoder(query_flat)  # [B, d_repr]
        query_key = F.normalize(query_key, dim=-1)

        # Cosine similarity
        mem_keys = self.memory_bank.get_keys(scale_idx)  # [M, d_repr]
        similarity = torch.mm(query_key, mem_keys.t())  # [B, M]

        # Threshold (adjusted for extreme data)
        threshold = torch.sigmoid(self.memory_bank.thresholds[scale_idx])
        if has_extreme is not None:
            threshold_per_sample = threshold.expand(x_normed.shape[0])
            # Reduce threshold for extreme samples
            extreme_mask = has_extreme.float()
            threshold_per_sample = threshold_per_sample - extreme_mask * self.extreme_threshold_reduction
            threshold_per_sample = threshold_per_sample.clamp(min=0.1)
        else:
            threshold_per_sample = threshold.expand(x_normed.shape[0])

        # Max similarity per sample
        max_sim = similarity.max(dim=-1)[0]  # [B]
        valid_mask = max_sim > threshold_per_sample  # [B]

        # Top-K retrieval
        k = min(self.top_k, mem_keys.shape[0])
        top_sim, top_idx = similarity.topk(k=k, dim=-1)  # [B, K]

        # Retrieved values
        mem_values = self.memory_bank.get_values(scale_idx)  # [M, pred_len]
        retrieved = mem_values[top_idx]  # [B, K, pred_len]

        # Weighted average by similarity
        sim_weights = F.softmax(top_sim, dim=-1)  # [B, K]
        hist_ref = (retrieved * sim_weights.unsqueeze(-1)).sum(dim=1)  # [B, pred_len]

        # Zero out invalid retrievals
        hist_ref = hist_ref * valid_mask.float().unsqueeze(-1)

        return hist_ref, valid_mask

    def retrieve_all_scales(self, x_normed, downsample_rates):
        """Retrieve historical references for all scales.

        Args:
            x_normed: [B, T, N] — full normalized input
            downsample_rates: list of int, one per scale

        Returns:
            hist_refs: list of [B, pred_len, N], length = n_scales
                       (expanded from mean to match N dimensions)
        """
        B, T, N = x_normed.shape
        device = x_normed.device

        # Detect extreme data once
        has_extreme, _ = self.extreme_detector(x_normed)  # [B]

        hist_refs = []
        for s, ds_rate in enumerate(downsample_rates):
            # Downsample
            if ds_rate > 1 and T >= ds_rate:
                trunc_T = (T // ds_rate) * ds_rate
                x_ds = x_normed[:, :trunc_T, :].reshape(
                    B, trunc_T // ds_rate, ds_rate, N
                ).mean(dim=2)
            else:
                x_ds = x_normed

            # Retrieve for this scale
            hist_ref, _ = self.retrieve(x_ds, s, has_extreme)  # [B, pred_len]

            # Expand to [B, pred_len, N] by repeating across variables
            hist_ref = hist_ref.unsqueeze(-1).expand(B, self.pred_len, N)
            hist_refs.append(hist_ref)

        return hist_refs


# ---------------------------------------------------------------------------
# AKG: Adaptive Knowledge Gating
# ---------------------------------------------------------------------------

class AdaptiveKnowledgeGating(nn.Module):
    """Fuses scale predictions with historical references via two matrices:

    1. Connection Matrix C [n_scales, pred_len]:
       gate = sigmoid(C[s, t]) controls each scale's history vs prediction balance

    2. Fusion Matrix F [2*n_scales, pred_len]:
       final weighted combination of all sources (predictions + gated results)
    """

    def __init__(self, n_scales, pred_len):
        super().__init__()
        self.n_scales = n_scales
        self.pred_len = pred_len

        # "Connection Matrix": [n_scales, pred_len]
        # Initialized to 0 => sigmoid(0) = 0.5 (balanced)
        self.connection_matrix = nn.Parameter(torch.zeros(n_scales, pred_len))

        # "Fusion Matrix": [2*n_scales, pred_len]
        # First n_scales: weight for scale_preds
        # Last n_scales: weight for connected (gated) results
        self.fusion_matrix = nn.Parameter(torch.zeros(2 * n_scales, pred_len))

    def forward(self, scale_preds, hist_refs):
        """
        Args:
            scale_preds: list of [B, pred_len, N], length = n_scales
            hist_refs:   list of [B, pred_len, N], length = n_scales

        Returns:
            final_pred: [B, pred_len, N]
        """
        # 1. Connection Matrix: per-scale history-prediction gating
        gates = torch.sigmoid(self.connection_matrix)  # [n_scales, pred_len]

        connected = []
        for s in range(self.n_scales):
            g = gates[s].unsqueeze(0).unsqueeze(-1)  # [1, pred_len, 1]
            # gate=1 -> trust prediction, gate=0 -> trust history
            connected_s = g * scale_preds[s] + (1 - g) * hist_refs[s]
            connected.append(connected_s)

        # 2. Fusion Matrix: combine all sources
        # Sources: scale_preds (n_scales) + connected (n_scales) = 2*n_scales
        all_sources = scale_preds + connected
        fusion_weights = F.softmax(self.fusion_matrix, dim=0)  # [2*n_scales, pred_len]

        # Weighted sum
        stacked = torch.stack(all_sources, dim=0)  # [2*n_scales, B, pred_len, N]
        w = fusion_weights.unsqueeze(1).unsqueeze(-1)  # [2*n_scales, 1, pred_len, 1]
        final_pred = (w * stacked).sum(dim=0)  # [B, pred_len, N]

        return final_pred

    def gate_loss(self):
        """Gate decisiveness regularization: encourage gates away from 0.5.

        L_gate = -mean(|connection_matrix - 0.5|)
        Negative sign because we want to maximize distance from 0.5.
        """
        gates = torch.sigmoid(self.connection_matrix)
        return -(gates - 0.5).abs().mean()
