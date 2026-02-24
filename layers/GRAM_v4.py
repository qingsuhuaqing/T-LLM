"""
GRAM_v4 — Enhanced Pattern-Value Dual Retriever (PVDR) + Threshold Adaptive Gating (AKG)

v4 core changes from v3:
  - PVDR: 3-sigma ExtremeDetector -> LogisticExtremeClassifier (learnable, 6 params)
  - PVDR: new BayesianChangePointDetector (3 learnable prior params)
  - PVDR: unified Top-K -> 4-mode retrieval (direct/extreme/turning_point/normal)
  - PVDR: memory bank gains per-sample labels (normal/extreme/turning_point)
  - AKG: fixed sigmoid(C) -> threshold-driven dynamic gating via PVDR signals
  - AKG: 3 new learnable threshold parameters (tau_extreme, tau_cp, tau_conf)

Design references:
  - RAFT (ICML 2025): retrieve similar history + subsequent values as reference
  - TS-RAG (NeurIPS 2025): ARM module for adaptive retrieval/current mixing
  - TFT (IJF 2021): GLU gated residual network for selective fusion
  - Bayesian Online Changepoint Detection (Adams & MacKay, 2007)
"""

import math

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
# Logistic Extreme Classifier (replaces ExtremeDetector)
# ---------------------------------------------------------------------------

class LogisticExtremeClassifier(nn.Module):
    """Learnable extreme data classifier using logistic regression.

    Input: 5 statistical features [mean, std, max, min, trend]
    Output: extreme_probability in [0, 1]
    Total parameters: 6 (5 weights + 1 bias)
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, T, N]
        Returns:
            extreme_prob: [B] — probability of being extreme data
        """
        # Extract statistical features (mean across variables)
        features = torch.stack([
            x.mean(dim=1).mean(dim=-1),                      # mean: [B]
            x.std(dim=1).mean(dim=-1).clamp(min=1e-6),       # std: [B]
            x.max(dim=1).values.mean(dim=-1),                 # max: [B]
            x.min(dim=1).values.mean(dim=-1),                 # min: [B]
            (x[:, -1, :] - x[:, 0, :]).mean(dim=-1),         # trend: [B]
        ], dim=-1)  # [B, 5]

        logits = self.linear(features)        # [B, 1]
        extreme_prob = self.sigmoid(logits).squeeze(-1)  # [B]
        return extreme_prob


# ---------------------------------------------------------------------------
# Bayesian Change Point Detector
# ---------------------------------------------------------------------------

class BayesianChangePointDetector(nn.Module):
    """Bayesian recursive binary-partition change point detector.

    Core idea: At each candidate split point, evaluate the Bayesian Factor
    comparing "two-segment model" vs "single-segment model".

    Parameters: 3 learnable priors (prior_mean, prior_var, noise_var)
    Uses softplus for variance params to maintain positivity (differentiable).
    """

    def __init__(self, min_segment=16, max_depth=4):
        super().__init__()
        self.min_segment = min_segment
        self.max_depth = max_depth
        # Learnable prior hyperparameters
        self.prior_mean = nn.Parameter(torch.zeros(1))
        self.prior_var = nn.Parameter(torch.ones(1))
        self.noise_var = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        Args:
            x: [B, T, N]
        Returns:
            near_end_score: [B] — probability of change point near sequence end
        """
        x_flat = x.mean(dim=-1)  # [B, T]
        B, T = x_flat.shape

        if T < 2 * self.min_segment:
            return torch.zeros(B, device=x.device)

        scores = []
        for b in range(B):
            segment = x_flat[b]  # [T]
            score = self._detect_near_end(segment, T)
            scores.append(score)

        return torch.stack(scores)  # [B]

    def _detect_near_end(self, segment, T):
        """Detect change points and return near-end score for one sample.

        Uses soft-max over Bayesian Factor scores at candidate positions
        for differentiability.
        """
        start = 0
        end = T
        near_end_threshold = int(T * 0.8)

        # Collect Bayesian factors at all candidate positions
        candidate_positions = list(range(
            start + self.min_segment,
            end - self.min_segment
        ))

        if len(candidate_positions) == 0:
            return torch.tensor(0.0, device=segment.device)

        bf_scores = []
        for pos in candidate_positions:
            left = segment[start:pos]
            right = segment[pos:end]
            whole = segment[start:end]

            log_bf = (self._log_marginal(left) + self._log_marginal(right)
                      - self._log_marginal(whole))
            bf_scores.append(log_bf)

        bf_scores = torch.stack(bf_scores)  # [num_candidates]

        # Soft-max weighted score (differentiable alternative to argmax)
        weights = F.softmax(bf_scores, dim=0)  # [num_candidates]

        # Compute near-end indicator for each position
        positions_tensor = torch.tensor(
            candidate_positions, device=segment.device, dtype=torch.float32)
        near_end_mask = (positions_tensor > near_end_threshold).float()

        # Score = sum of weights for near-end positions, scaled by max BF
        near_end_weight = (weights * near_end_mask).sum()

        # Overall confidence: sigmoid of best Bayesian factor
        best_bf = bf_scores.max()
        confidence = torch.sigmoid(best_bf)

        # Near-end score: confidence * proportion of weight in near-end region
        score = confidence * near_end_weight

        return score

    def _log_marginal(self, segment):
        """Compute log marginal likelihood for a segment (Normal-Normal conjugate).

        Args:
            segment: [L] — 1D time series segment
        Returns:
            log_ml: scalar tensor
        """
        n = segment.shape[0]
        if n == 0:
            return torch.tensor(0.0, device=segment.device)

        s_mean = segment.mean()
        s_var = segment.var().clamp(min=1e-8)

        prior_var = F.softplus(self.prior_var)
        noise_var = F.softplus(self.noise_var)

        # Normal-Normal conjugate marginal likelihood
        post_var = 1.0 / (1.0 / prior_var + n / noise_var)
        post_mean = post_var * (self.prior_mean / prior_var + n * s_mean / noise_var)

        log_ml = (-n / 2.0 * torch.log(2 * math.pi * noise_var)
                  + 0.5 * torch.log(post_var / prior_var)
                  - 0.5 * (n * s_var / noise_var
                           + s_mean ** 2 * n / noise_var
                           - post_mean ** 2 / post_var
                           + self.prior_mean ** 2 / prior_var))
        return log_ml


# ---------------------------------------------------------------------------
# Enhanced Multi-scale Memory Bank (extends v3 with labels)
# ---------------------------------------------------------------------------

class EnhancedMultiScaleMemoryBank(nn.Module):
    """Maintains independent memory banks per scale with label annotations.

    Each bank stores (key, value, label) triplets:
      - key:   [M, d_repr] — pattern representation
      - value: [M, pred_len] — the actual future values following that pattern
      - label: [M] (long) — 0=normal, 1=extreme, 2=turning_point
    """

    def __init__(self, n_scales, d_repr, pred_len, max_memory_per_scale=5000):
        super().__init__()
        self.n_scales = n_scales
        self.d_repr = d_repr
        self.pred_len = pred_len
        self.max_memory = max_memory_per_scale

        # Learnable per-scale similarity thresholds
        self.thresholds = nn.Parameter(torch.full((n_scales,), 0.8))

        # Register empty buffers
        for s in range(n_scales):
            self.register_buffer(f'keys_{s}', torch.zeros(0, d_repr))
            self.register_buffer(f'values_{s}', torch.zeros(0, pred_len))
            self.register_buffer(f'labels_{s}', torch.zeros(0, dtype=torch.long))

    def get_keys(self, scale_idx):
        return getattr(self, f'keys_{scale_idx}')

    def get_values(self, scale_idx):
        return getattr(self, f'values_{scale_idx}')

    def get_labels(self, scale_idx):
        return getattr(self, f'labels_{scale_idx}')

    def set_bank(self, scale_idx, keys, values, labels=None):
        """Set memory bank for a given scale."""
        device = self.thresholds.device
        setattr(self, f'keys_{scale_idx}', keys.to(device))
        setattr(self, f'values_{scale_idx}', values.to(device))
        if labels is not None:
            setattr(self, f'labels_{scale_idx}', labels.to(device))
        else:
            setattr(self, f'labels_{scale_idx}',
                    torch.zeros(keys.shape[0], dtype=torch.long, device=device))

    def is_empty(self, scale_idx):
        return self.get_keys(scale_idx).shape[0] == 0

    def get_subset_by_label(self, scale_idx, label):
        """Get indices of memory entries with a specific label.

        Args:
            scale_idx: which scale's bank
            label: 0=normal, 1=extreme, 2=turning_point
        Returns:
            indices: [K] long tensor of matching indices
        """
        labels = self.get_labels(scale_idx)
        if labels.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=labels.device)
        return (labels == label).nonzero(as_tuple=True)[0]

    def build_with_labels(self, all_x, all_y, extreme_classifier, cp_detector,
                          encoder, downsample_rates):
        """Build memory banks with automatic label annotation.

        Args:
            all_x: [M, T, N] — all training inputs
            all_y: [M, pred_len, N] — all training targets
            extreme_classifier: LogisticExtremeClassifier instance
            cp_detector: BayesianChangePointDetector instance
            encoder: LightweightPatternEncoder instance
            downsample_rates: list of int, one per scale
        """
        device = self.thresholds.device
        M = all_x.shape[0]

        # Mean over variables for compact storage
        y_mean = all_y.mean(dim=-1)  # [M, pred_len]

        with torch.no_grad():
            # Classify all samples
            extreme_probs = extreme_classifier(all_x)  # [M]
            near_end_scores = cp_detector(all_x)        # [M]

            # Assign labels
            labels = torch.zeros(M, dtype=torch.long, device=device)
            labels[extreme_probs > 0.5] = 1       # extreme
            labels[near_end_scores > 0.5] = 2     # turning_point
            # turning_point takes priority when both flags are set

            # Build per-scale banks
            for s, ds_rate in enumerate(downsample_rates):
                # Downsample input
                if ds_rate > 1:
                    T = all_x.shape[1]
                    trunc_T = (T // ds_rate) * ds_rate
                    x_ds = all_x[:, :trunc_T, :].reshape(
                        M, trunc_T // ds_rate, ds_rate, all_x.shape[2]
                    ).mean(dim=2)
                else:
                    x_ds = all_x

                # Encode
                x_flat = x_ds.mean(dim=-1)  # [M, T_ds]
                keys = encoder(x_flat)       # [M, d_repr]
                keys = F.normalize(keys, dim=-1)

                self.set_bank(s, keys, y_mean, labels)


# ---------------------------------------------------------------------------
# PVDR v4: Enhanced Dual Retriever
# ---------------------------------------------------------------------------

class EnhancedDualRetriever(nn.Module):
    """Multi-scale retrieval module with enhanced detection.

    Upgrades from v3:
    - LogisticExtremeClassifier replaces 3-sigma rule
    - BayesianChangePointDetector adds turning point awareness
    - 4 retrieval modes: direct, extreme, turning_point, normal
    - Returns pvdr_signals for AKG threshold gating
    """

    def __init__(self, n_scales, d_repr, pred_len, n_vars, top_k=5,
                 extreme_threshold_reduction=0.2,
                 max_memory_per_scale=5000,
                 cp_min_segment=16, cp_max_depth=4):
        super().__init__()
        self.n_scales = n_scales
        self.d_repr = d_repr
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.top_k = top_k
        self.extreme_threshold_reduction = extreme_threshold_reduction

        self.encoder = LightweightPatternEncoder(d_repr=d_repr)
        self.memory_bank = EnhancedMultiScaleMemoryBank(
            n_scales=n_scales,
            d_repr=d_repr,
            pred_len=pred_len,
            max_memory_per_scale=max_memory_per_scale,
        )
        self.extreme_classifier = LogisticExtremeClassifier()
        self.cp_detector = BayesianChangePointDetector(
            min_segment=cp_min_segment,
            max_depth=cp_max_depth,
        )

    def build_memory(self, train_loader, device, downsample_rates, max_samples=5000):
        """Build memory banks from training data with label annotation."""
        self.eval()
        all_inputs = []
        all_targets = []
        count = 0

        with torch.no_grad():
            for batch_x, batch_y, _, _ in train_loader:
                if count >= max_samples:
                    break
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                all_inputs.append(batch_x)
                all_targets.append(batch_y[:, -self.pred_len:, :])
                count += batch_x.shape[0]

        if not all_inputs:
            return

        all_x = torch.cat(all_inputs, dim=0)[:max_samples]
        all_y = torch.cat(all_targets, dim=0)[:max_samples]

        with torch.no_grad():
            self.memory_bank.build_with_labels(
                all_x, all_y,
                self.extreme_classifier,
                self.cp_detector,
                self.encoder,
                downsample_rates,
            )

        self.train()

    def retrieve(self, x_normed, scale_idx, extreme_prob=None, near_end_score=None):
        """Retrieve historical patterns for one scale with mode-based logic.

        Args:
            x_normed: [B, T_ds, N] — downsampled + normalized input
            scale_idx: which scale's memory bank to use
            extreme_prob: [B] — extreme probability per sample (optional)
            near_end_score: [B] — change point score per sample (optional)

        Returns:
            hist_ref: [B, pred_len] — weighted average of retrieved future values
            valid_mask: [B] bool — True if retrieval succeeded
            confidence: [B] — retrieval confidence score
        """
        B = x_normed.shape[0]
        device = x_normed.device

        if self.memory_bank.is_empty(scale_idx):
            return (torch.zeros(B, self.pred_len, device=device),
                    torch.zeros(B, dtype=torch.bool, device=device),
                    torch.zeros(B, device=device))

        # Encode query
        query_flat = x_normed.mean(dim=-1)  # [B, T_ds]
        query_key = self.encoder(query_flat)  # [B, d_repr]
        query_key = F.normalize(query_key, dim=-1)

        # Cosine similarity with all memory entries
        mem_keys = self.memory_bank.get_keys(scale_idx)   # [M, d_repr]
        mem_values = self.memory_bank.get_values(scale_idx)  # [M, pred_len]
        similarity = torch.mm(query_key, mem_keys.t())  # [B, M]

        # Base threshold
        threshold = torch.sigmoid(self.memory_bank.thresholds[scale_idx])

        # Max similarity per sample
        max_sim = similarity.max(dim=-1)[0]  # [B]

        # Per-sample retrieval
        hist_ref = torch.zeros(B, self.pred_len, device=device)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=device)
        confidence = torch.zeros(B, device=device)

        for b in range(B):
            sim_b = similarity[b]  # [M]
            max_sim_b = max_sim[b]

            # Mode selection
            if max_sim_b > 0.95:
                # Mode A: Direct — extremely similar, use Top-1
                k_retrieve = 1
                effective_threshold = 0.0  # no threshold filtering
            elif extreme_prob is not None and extreme_prob[b] > 0.5:
                # Mode B: Extreme — lower threshold, retrieve more
                k_retrieve = min(self.top_k * 2, mem_keys.shape[0])
                effective_threshold = max(0.1, threshold.item() - self.extreme_threshold_reduction)
            elif near_end_score is not None and near_end_score[b] > 0.5:
                # Mode C: Turning point — standard K from turning_point subset
                k_retrieve = min(self.top_k, mem_keys.shape[0])
                effective_threshold = threshold.item()
                # Prefer turning_point labeled entries
                tp_indices = self.memory_bank.get_subset_by_label(scale_idx, 2)
                if len(tp_indices) > 0:
                    tp_sim = sim_b[tp_indices]
                    tp_values = mem_values[tp_indices]
                    k_tp = min(k_retrieve, len(tp_indices))
                    top_sim_tp, top_idx_tp = tp_sim.topk(k=k_tp, dim=-1)
                    retrieved = tp_values[top_idx_tp]
                    sim_weights = F.softmax(top_sim_tp, dim=-1)
                    hist_ref[b] = (retrieved * sim_weights.unsqueeze(-1)).sum(dim=0)
                    valid_mask[b] = max_sim_b > effective_threshold
                    confidence[b] = max_sim_b
                    continue
            else:
                # Mode D: Normal — standard Top-K
                k_retrieve = min(self.top_k, mem_keys.shape[0])
                effective_threshold = threshold.item()

            # Standard retrieval
            k_actual = min(k_retrieve, mem_keys.shape[0])
            top_sim, top_idx = sim_b.topk(k=k_actual, dim=-1)
            retrieved = mem_values[top_idx]  # [K, pred_len]
            sim_weights = F.softmax(top_sim, dim=-1)
            hist_ref[b] = (retrieved * sim_weights.unsqueeze(-1)).sum(dim=0)
            valid_mask[b] = max_sim_b > effective_threshold
            confidence[b] = max_sim_b

        # Zero out invalid retrievals
        hist_ref = hist_ref * valid_mask.float().unsqueeze(-1)

        return hist_ref, valid_mask, confidence

    def retrieve_all_scales(self, x_normed, downsample_rates):
        """Retrieve historical references for all scales.

        Args:
            x_normed: [B, T, N] — full normalized input
            downsample_rates: list of int, one per scale

        Returns:
            hist_refs: list of [B, pred_len, N], length = n_scales
            pvdr_signals: dict with detection signals for AKG
        """
        B, T, N = x_normed.shape
        device = x_normed.device

        # Run detectors once
        extreme_prob = self.extreme_classifier(x_normed)   # [B]
        near_end_score = self.cp_detector(x_normed)        # [B]

        hist_refs = []
        confidences = []

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
            hist_ref, _, conf = self.retrieve(
                x_ds, s,
                extreme_prob=extreme_prob,
                near_end_score=near_end_score)

            # Expand to [B, pred_len, N]
            hist_ref = hist_ref.unsqueeze(-1).expand(B, self.pred_len, N)
            hist_refs.append(hist_ref)
            confidences.append(conf)

        # Build pvdr_signals
        pvdr_signals = {
            'extreme_score': extreme_prob,                             # [B]
            'change_point_score': near_end_score,                      # [B]
            'retrieval_confidence': torch.stack(confidences, dim=-1),   # [B, n_scales]
        }

        return hist_refs, pvdr_signals


# ---------------------------------------------------------------------------
# AKG v4: Threshold Adaptive Gating
# ---------------------------------------------------------------------------

class ThresholdAdaptiveGating(nn.Module):
    """Fuses scale predictions with historical references via threshold-driven gating.

    Upgrades from v3 AdaptiveKnowledgeGating:
    - base_gate = sigmoid(C) is modulated by PVDR signals
    - effective_gate = sigmoid(C) * (1-extreme_adj) * (1-cp_adj) * conf_adj
    - 3 learnable threshold parameters control modulation sensitivity
    - Per-batch gating (not just per-scale)
    - Falls back to v3 behavior when pvdr_signals=None

    Matrices (same as v3):
    1. Connection Matrix C [n_scales, pred_len]: history vs prediction balance
    2. Fusion Matrix F [2*n_scales, pred_len]: final weighted combination
    """

    def __init__(self, n_scales, pred_len):
        super().__init__()
        self.n_scales = n_scales
        self.pred_len = pred_len

        # "Connection Matrix": [n_scales, pred_len]
        self.connection_matrix = nn.Parameter(torch.zeros(n_scales, pred_len))

        # "Fusion Matrix": [2*n_scales, pred_len]
        self.fusion_matrix = nn.Parameter(torch.zeros(2 * n_scales, pred_len))

        # v4 NEW: learnable threshold parameters
        self.extreme_threshold = nn.Parameter(torch.tensor(0.5))
        self.cp_threshold = nn.Parameter(torch.tensor(0.5))
        self.conf_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, scale_preds, hist_refs, pvdr_signals=None):
        """
        Args:
            scale_preds: list of [B, pred_len, N], length = n_scales
            hist_refs:   list of [B, pred_len, N], length = n_scales
            pvdr_signals: dict with extreme_score, change_point_score,
                          retrieval_confidence (optional, None => v3 behavior)

        Returns:
            final_pred: [B, pred_len, N]
        """
        B = scale_preds[0].shape[0]

        # Base gate
        base_gate = torch.sigmoid(self.connection_matrix)  # [n_scales, pred_len]

        if pvdr_signals is not None:
            # v4: compute effective gate with threshold adjustments
            effective_gate = self._compute_effective_gate(
                base_gate, pvdr_signals, B)  # [B, n_scales, pred_len]
        else:
            # v3 fallback: broadcast to [B, n_scales, pred_len]
            effective_gate = base_gate.unsqueeze(0).expand(
                B, -1, -1)  # [B, n_scales, pred_len]

        # Stage 1: Per-scale gating (history vs prediction balance)
        connected = []
        for s in range(self.n_scales):
            g = effective_gate[:, s, :].unsqueeze(-1)  # [B, pred_len, 1]
            # gate=1 -> trust prediction, gate=0 -> trust history
            connected_s = g * scale_preds[s] + (1 - g) * hist_refs[s]
            connected.append(connected_s)

        # Stage 2: Fusion Matrix — combine all sources (same as v3)
        all_sources = list(scale_preds) + connected
        fusion_weights = F.softmax(self.fusion_matrix, dim=0)  # [2*n_scales, pred_len]

        stacked = torch.stack(all_sources, dim=0)  # [2*n_scales, B, pred_len, N]
        w = fusion_weights.unsqueeze(1).unsqueeze(-1)  # [2*n_scales, 1, pred_len, 1]
        final_pred = (w * stacked).sum(dim=0)  # [B, pred_len, N]

        return final_pred

    def _compute_effective_gate(self, base_gate, pvdr_signals, B):
        """Compute threshold-driven effective gate.

        Args:
            base_gate: [n_scales, pred_len]
            pvdr_signals: dict with signals
            B: batch size
        Returns:
            effective_gate: [B, n_scales, pred_len]
        """
        extreme_score = pvdr_signals['extreme_score']           # [B]
        cp_score = pvdr_signals['change_point_score']           # [B]
        conf = pvdr_signals['retrieval_confidence']              # [B, n_scales]

        # Soft thresholds (differentiable via sigmoid)
        tau_ext = torch.sigmoid(self.extreme_threshold)
        tau_cp = torch.sigmoid(self.cp_threshold)
        tau_conf = torch.sigmoid(self.conf_threshold)

        # Extreme data adjustment: extreme -> lower gate -> more history trust
        extreme_adj = torch.sigmoid(
            10.0 * (extreme_score - tau_ext)
        )  # [B]

        # Change point adjustment: near change point -> lower gate
        cp_adj = torch.sigmoid(
            10.0 * (cp_score - tau_cp)
        )  # [B]

        # Retrieval confidence adjustment: high confidence -> allow more history
        conf_adj = torch.sigmoid(
            10.0 * (conf - tau_conf)
        )  # [B, n_scales]

        # Build effective gate: [B, n_scales, pred_len]
        effective_gate = base_gate.unsqueeze(0).expand(B, -1, -1).clone()
        # [B, n_scales, pred_len]

        # Apply adjustments (broadcast)
        effective_gate = effective_gate * (1.0 - extreme_adj.unsqueeze(1).unsqueeze(-1))
        effective_gate = effective_gate * (1.0 - cp_adj.unsqueeze(1).unsqueeze(-1))
        effective_gate = effective_gate * conf_adj.unsqueeze(-1)

        return effective_gate

    def gate_loss(self):
        """Gate decisiveness regularization: encourage gates away from 0.5.

        L_gate = -mean(|connection_matrix - 0.5|)
        Negative sign because we want to maximize distance from 0.5.
        """
        gates = torch.sigmoid(self.connection_matrix)
        return -(gates - 0.5).abs().mean()
