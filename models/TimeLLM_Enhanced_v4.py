"""
TimeLLM_Enhanced_v4 — Polyphase Multi-scale Prediction + Enhanced Retrieval Fusion

v4 design: upgrades v3 module internals while preserving the overall architecture:
  1. Baseline Time-LLM produces pred_s1  (frozen forward pass, identical to v3)
  2. DM v4 produces branch_preds_dict   (polyphase multi-scale, replaces avg-pool)
  3. C2F v4 votes + fuses branches      (MAD voting + DWT trend constraint)
  4. PVDR v4 retrieves historical refs   (logistic extreme + Bayesian changepoint)
  5. AKG v4 gates + fuses everything     (threshold-driven gating via PVDR signals)

Loss = L_main + warmup * (lambda_consist * L_consistency
                         + lambda_gate * L_gate
                         + lambda_expert * L_expert)
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
)
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

from layers.TAPR_v4 import PolyphaseMultiScale, ExpertVotingFusion
from layers.GRAM_v4 import EnhancedDualRetriever, ThresholdAdaptiveGating

transformers.logging.set_verbosity_error()


# ---------------------------------------------------------------------------
# FlattenHead (identical to baseline)
# ---------------------------------------------------------------------------

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# ReprogrammingLayer (identical to baseline)
# ---------------------------------------------------------------------------

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self._reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def _reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,she->blhe", A, value_embedding)


# ---------------------------------------------------------------------------
# Main enhanced model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    TimeLLM_Enhanced_v4

    v4 upgrades (from v3):
        DM:   PolyphaseMultiScale (unified k=4 polyphase + signal transforms)
        C2F:  ExpertVotingFusion (MAD intra-scale voting + DWT cross-scale constraint)
        PVDR: EnhancedDualRetriever (logistic extreme + Bayesian changepoint)
        AKG:  ThresholdAdaptiveGating (PVDR-signal-driven dynamic gating)

    New config attrs (all with defaults):
        lambda_expert (float):          expert correction loss weight
        mad_threshold (float):          MAD outlier detection multiplier
        reliability_threshold (float):  correction rate threshold for unreliable flag
        reliability_penalty (float):    weight multiplier for unreliable scales
        cp_min_segment (int):           Bayesian CP min segment length
        cp_max_depth (int):             Bayesian CP max recursion depth

    Inherited from v3 (all with defaults):
        use_tapr (bool):       enable DM + C2F
        use_gram (bool):       enable PVDR + AKG
        n_scales (int):        total number of scales incl. baseline (default 5)
        lambda_consist (float): consistency loss weight
        lambda_gate (float):   gate regularization weight
        decay_factor (float):  C2F inference weight decay
        similarity_threshold (float): base PVDR retrieval threshold
        extreme_sigma (float): extreme-data sigma threshold
        extreme_threshold_reduction (float): threshold reduction for extreme data
        top_k (int):           PVDR top-K
        d_repr (int):          PVDR pattern representation dimension
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()

        # ── basic config ──────────────────────────────────────────────
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # ── module switches ───────────────────────────────────────────
        self.use_tapr = getattr(configs, 'use_tapr', False)
        self.use_gram = getattr(configs, 'use_gram', False)

        # ── loss weights ──────────────────────────────────────────────
        self.lambda_consist = getattr(configs, 'lambda_consist', 0.1)
        self.lambda_gate = getattr(configs, 'lambda_gate', 0.01)
        self.lambda_expert = getattr(configs, 'lambda_expert', 0.05)

        # ── v4 config ────────────────────────────────────────────────
        self.mad_threshold = getattr(configs, 'mad_threshold', 3.0)
        self.reliability_threshold = getattr(configs, 'reliability_threshold', 0.3)
        self.reliability_penalty = getattr(configs, 'reliability_penalty', 0.3)
        self.cp_min_segment = getattr(configs, 'cp_min_segment', 16)
        self.cp_max_depth = getattr(configs, 'cp_max_depth', 4)

        # ── scale config ──────────────────────────────────────────────
        n_scales = getattr(configs, 'n_scales', 5)
        self.n_scales = n_scales
        self.n_aux_scales = n_scales - 1  # baseline is scale 1
        ds_str = getattr(configs, 'downsample_rates', '1,2,4,4,4')
        if isinstance(ds_str, str):
            self.downsample_rates = [int(x) for x in ds_str.split(',')]
        else:
            self.downsample_rates = list(ds_str)
        # Ensure length matches n_scales
        self.downsample_rates = self.downsample_rates[:n_scales]
        while len(self.downsample_rates) < n_scales:
            self.downsample_rates.append(self.downsample_rates[-1] * 2)

        # ══════════════════════════════════════════════════════════════
        # LLM loading (identical to baseline TimeLLM.py)
        # ══════════════════════════════════════════════════════════════
        if configs.llm_model_path:
            from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig

            print(f"[v4] Loading model from: {configs.llm_model_path}")
            quantization_config = None
            if configs.load_in_4bit:
                print("[v4] Enabling 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            self.llm_config = AutoConfig.from_pretrained(configs.llm_model_path)
            self.llm_config.num_hidden_layers = configs.llm_layers
            self.llm_config.output_attentions = True
            self.llm_config.output_hidden_states = True

            try:
                self.llm_model = AutoModel.from_pretrained(
                    configs.llm_model_path,
                    trust_remote_code=True, local_files_only=True,
                    config=self.llm_config,
                    quantization_config=quantization_config,
                    device_map="auto" if configs.load_in_4bit else None,
                )
            except EnvironmentError:
                print("[v4] Local not found, downloading...")
                self.llm_model = AutoModel.from_pretrained(
                    configs.llm_model_path,
                    trust_remote_code=True, local_files_only=False,
                    config=self.llm_config,
                    quantization_config=quantization_config,
                    device_map="auto" if configs.load_in_4bit else None,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    configs.llm_model_path, trust_remote_code=True, local_files_only=True)
            except EnvironmentError:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    configs.llm_model_path, trust_remote_code=True, local_files_only=False)

        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('./base_models/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    './base_models/gpt2', trust_remote_code=True,
                    local_files_only=True, config=self.gpt2_config)
            except EnvironmentError:
                raise
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    './base_models/gpt2', trust_remote_code=True, local_files_only=True)
            except EnvironmentError:
                raise

        elif configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b', trust_remote_code=True,
                    local_files_only=True, config=self.llama_config)
            except EnvironmentError:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b', trust_remote_code=True,
                    local_files_only=False, config=self.llama_config)
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b', trust_remote_code=True, local_files_only=True)
            except EnvironmentError:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b', trust_remote_code=True, local_files_only=False)

        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased', trust_remote_code=True,
                    local_files_only=True, config=self.bert_config)
            except EnvironmentError:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased', trust_remote_code=True,
                    local_files_only=False, config=self.bert_config)
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased', trust_remote_code=True, local_files_only=True)
            except EnvironmentError:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased', trust_remote_code=True, local_files_only=False)
        else:
            raise Exception('LLM model is not defined')

        # Pad token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'

        # Freeze LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Prompt
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = ('The Electricity Transformer Temperature (ETT) is a '
                                'crucial indicator in the electric power long-term deployment.')

        self.dropout = nn.Dropout(configs.dropout)

        # ══════════════════════════════════════════════════════════════
        # Baseline Time-LLM layers (identical, not modified)
        # ══════════════════════════════════════════════════════════════
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.output_projection = FlattenHead(
                configs.enc_in, self.head_nf, self.pred_len,
                head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # ══════════════════════════════════════════════════════════════
        # v4 new modules
        # ══════════════════════════════════════════════════════════════
        if self.use_tapr:
            print(f"[v4] Initializing DM (PolyphaseMultiScale, {self.n_aux_scales} aux scales) + C2F (ExpertVotingFusion)")
            self.dm = PolyphaseMultiScale(
                d_model=configs.d_model,
                patch_len=self.patch_len,
                stride=self.stride,
                pred_len=self.pred_len,
                seq_len=configs.seq_len,
                n_aux_scales=self.n_aux_scales,
                dropout=configs.dropout,
            )
            self.c2f = ExpertVotingFusion(
                n_scales=self.n_scales,
                pred_len=self.pred_len,
                decay_factor=getattr(configs, 'decay_factor', 0.5),
                mad_threshold=self.mad_threshold,
                reliability_threshold=self.reliability_threshold,
                reliability_penalty=self.reliability_penalty,
            )
        else:
            self.dm = None
            self.c2f = None

        if self.use_gram:
            print(f"[v4] Initializing PVDR (EnhancedDualRetriever) + AKG (ThresholdAdaptiveGating) ({self.n_scales} scales)")
            self.pvdr = EnhancedDualRetriever(
                n_scales=self.n_scales,
                d_repr=getattr(configs, 'd_repr', 64),
                pred_len=self.pred_len,
                n_vars=configs.enc_in,
                top_k=getattr(configs, 'top_k', 5),
                extreme_threshold_reduction=getattr(configs, 'extreme_threshold_reduction', 0.2),
                cp_min_segment=self.cp_min_segment,
                cp_max_depth=self.cp_max_depth,
            )
            self.akg = ThresholdAdaptiveGating(
                n_scales=self.n_scales,
                pred_len=self.pred_len,
            )
        else:
            self.pvdr = None
            self.akg = None

        # Cache for auxiliary loss computation
        self._aux_cache = {}

    # ------------------------------------------------------------------
    # Baseline forward  (identical to TimeLLM.py forecast())
    # ------------------------------------------------------------------

    def _baseline_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Run the original Time-LLM forward pass and return prediction."""
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc_ci = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # Statistics
        min_values = torch.min(x_enc_ci, dim=1)[0]
        max_values = torch.max(x_enc_ci, dim=1)[0]
        medians = torch.median(x_enc_ci, dim=1).values
        lags = self._calcute_lags(x_enc_ci)
        trends = x_enc_ci.diff(dim=1).sum(dim=1)

        # Build prompt
        prompt = []
        for b in range(x_enc_ci.shape[0]):
            prompt.append(
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {self.pred_len} steps given "
                f"the previous {self.seq_len} steps information; "
                f"Input statistics: "
                f"min value {min_values[b].tolist()[0]}, "
                f"max value {max_values[b].tolist()[0]}, "
                f"median value {medians[b].tolist()[0]}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags[b].tolist()}<|<end_prompt>|>"
            )

        x_enc_ci = x_enc_ci.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt_ids.to(x_enc.device))

        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc_ci = x_enc_ci.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc_ci.float())
        enc_out = enc_out.to(prompt_embeddings.dtype)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Return prediction BEFORE denormalization (we denorm after fusion)
        return dec_out, x_enc  # pred_s1 (normed space), x_enc (normed)

    def _calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Full v4 pipeline:
          baseline -> DM(polyphase) -> C2F(voting+trend) -> PVDR(enhanced) -> AKG(threshold) -> denorm
        Falls back to baseline-only when modules are disabled.
        """
        # ── Step 1: Baseline prediction ───────────────────────────────
        pred_s1, x_normed = self._baseline_forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec)
        # pred_s1: [B, pred_len, N]  (normalized space)
        # x_normed: [B, T, N]  (normalized input)

        # If no extra modules, just denorm and return
        if not self.use_tapr and not self.use_gram:
            return self.normalize_layers(pred_s1, 'denorm')

        # ── Step 2: DM v4 — polyphase multi-scale branch predictions ─
        if self.dm is not None:
            # v4: DM returns branch_preds_dict, C2F handles voting+fusion
            pred_s1_out, branch_preds_dict = self.dm(x_normed, pred_s1)

            # v4: C2F takes branch_preds_dict, does voting + fusion internally
            weighted_pred = self.c2f(
                pred_s1_out, branch_preds_dict, training=self.training)

            # Get consolidated_preds (cached during C2F forward)
            consolidated_preds = self.c2f.get_consolidated_preds()

            # Store for auxiliary loss computation
            if self.training:
                self._aux_cache['consolidated_preds'] = consolidated_preds
        else:
            consolidated_preds = [pred_s1]
            weighted_pred = pred_s1

        # ── Step 3: PVDR v4 — enhanced multi-scale retrieval ─────────
        if self.pvdr is not None:
            # v4: PVDR returns pvdr_signals alongside hist_refs
            hist_refs, pvdr_signals = self.pvdr.retrieve_all_scales(
                x_normed, self.downsample_rates[:self.n_scales])
        else:
            hist_refs = None
            pvdr_signals = None

        # ── Step 4: AKG v4 — threshold-driven gating ─────────────────
        if self.akg is not None and hist_refs is not None:
            # Use consolidated_preds as scale_preds for AKG
            akg_preds = list(consolidated_preds)
            if self.c2f is not None:
                # Replace baseline (index 0) with C2F weighted_pred
                # so C2F weight_matrix receives gradients
                akg_preds[0] = weighted_pred
            # v4: AKG receives pvdr_signals for threshold-driven gating
            final_pred = self.akg(akg_preds, hist_refs, pvdr_signals=pvdr_signals)
        else:
            final_pred = weighted_pred

        # ── Step 5: Denormalize ───────────────────────────────────────
        final_pred = self.normalize_layers(final_pred, 'denorm')
        return final_pred

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

    def compute_auxiliary_loss(self, y_true, current_step=None, warmup_steps=500):
        """Compute auxiliary losses for v4 modules.

        L_total = warmup * (lambda_consist * L_consistency
                           + lambda_gate * L_gate
                           + lambda_expert * L_expert)

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss values for logging
        """
        device = y_true.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # Warmup factor
        if current_step is not None:
            warmup_factor = min(1.0, current_step / max(warmup_steps, 1))
        else:
            warmup_factor = 1.0
        loss_dict['warmup_factor'] = warmup_factor

        # C2F consistency loss (v4: wavelet-based, replaces v3 KL divergence)
        if self.c2f is not None and 'consolidated_preds' in self._aux_cache:
            consist_loss = self.c2f.consistency_loss(
                self._aux_cache['consolidated_preds'])
            total_loss = total_loss + warmup_factor * self.lambda_consist * consist_loss
            loss_dict['consistency_loss'] = consist_loss.item()

        # AKG gate regularization loss (same as v3)
        if self.akg is not None:
            gate_loss = self.akg.gate_loss()
            total_loss = total_loss + warmup_factor * self.lambda_gate * gate_loss
            loss_dict['gate_loss'] = gate_loss.item()

        # C2F expert correction loss (v4 NEW)
        if self.c2f is not None:
            expert_loss = self.c2f.expert_loss()
            total_loss = total_loss + warmup_factor * self.lambda_expert * expert_loss
            loss_dict['expert_loss'] = expert_loss.item()

        # Clear cache
        self._aux_cache = {}

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Memory bank construction
    # ------------------------------------------------------------------

    def build_retrieval_memory(self, train_loader, device):
        """Build PVDR memory banks from training data."""
        if self.pvdr is not None:
            print("[v4] Building PVDR memory banks (with labels) for all scales...")
            self.pvdr.build_memory(
                train_loader, device,
                downsample_rates=self.downsample_rates[:self.n_scales],
            )
            print("[v4] Memory banks built successfully")
