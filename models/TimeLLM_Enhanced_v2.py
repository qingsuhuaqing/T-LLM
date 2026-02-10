"""
TimeLLM_Enhanced_v2 - 极简增强版Time-LLM模型

核心设计原则 (学术裁缝思想):
1. 最小干扰原则: 新增模块不影响原模型性能
2. 条件触发原则: 模块仅在明确有益时才介入
3. 可消融原则: 每个子模块可独立开关验证

改进要点:
- TAPR_v2: 仅观察不修改，趋势冲突时才微调
- GRAM_v2: 极高相似度阈值，绝大多数情况完全跳过
- 辅助损失权重极低 (0.01)，仅作正则化
- 数据流与原Time-LLM完全一致

与v1的关键区别:
- v1: 直接修改enc_out，辅助损失权重0.5
- v2: 不修改主流程，辅助损失权重0.01

修改日志 (2026-02-10):
- 完全重构模块集成方式
- 新增"观察者模式"设计
- 保持与原Time-LLM的完全兼容
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

# 导入v2版本模块
from layers.TAPR_v2 import TAPR_v2
from layers.GRAM_v2 import GRAM_v2, RetrievalContext

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    """
    输出投影头 - 将LLM输出映射为预测值
    与原版完全相同
    """
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


class Model(nn.Module):
    """
    TimeLLM_Enhanced_v2 - 极简增强版

    与原Time-LLM的数据流完全一致，新增模块作为"旁路观察者":
    1. TAPR_v2: 观察多尺度趋势，生成趋势嵌入 (不修改enc_out)
    2. GRAM_v2: 严格检索，仅极高相似度时才返回上下文 (默认跳过)

    新增参数:
        - use_tapr: 启用TAPR模块
        - use_gram: 启用GRA-M模块
        - lambda_trend: 趋势辅助损失权重 (默认0.01，极低)
        - lambda_retrieval: 检索辅助损失权重 (默认0.001，极低)
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # ========== 模块开关 ==========
        self.use_tapr = getattr(configs, 'use_tapr', False)
        self.use_gram = getattr(configs, 'use_gram', False)

        # 极低的辅助损失权重 (不影响主任务)
        self.lambda_trend = getattr(configs, 'lambda_trend', 0.01)
        self.lambda_retrieval = getattr(configs, 'lambda_retrieval', 0.001)

        # 是否在输出阶段应用趋势融合 (默认关闭)
        self.apply_trend_fusion = getattr(configs, 'apply_trend_fusion', False)

        # ========== LLM模型加载 (与原版完全相同) ==========
        if configs.llm_model_path:
            from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig

            print(f"Loading generic model from: {configs.llm_model_path}")

            quantization_config = None
            if configs.load_in_4bit:
                print("Enabling 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            self.llm_config = AutoConfig.from_pretrained(configs.llm_model_path)
            self.llm_config.num_hidden_layers = configs.llm_layers
            self.llm_config.output_attentions = True
            self.llm_config.output_hidden_states = True

            try:
                self.llm_model = AutoModel.from_pretrained(
                    configs.llm_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llm_config,
                    quantization_config=quantization_config,
                    device_map="auto" if configs.load_in_4bit else None
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    configs.llm_model_path,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llm_config,
                    quantization_config=quantization_config,
                    device_map="auto" if configs.load_in_4bit else None
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    configs.llm_model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    configs.llm_model_path,
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('./base_models/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    './base_models/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                raise
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    './base_models/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                raise
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 冻结LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # ========== 原有模块 (保持不变) ==========
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # ========== 新增: TAPR_v2模块 (观察者模式) ==========
        if self.use_tapr:
            print("[TimeLLM-Enhanced-v2] Initializing TAPR_v2 module (observer mode)...")
            self.tapr = TAPR_v2(configs)
            # 趋势嵌入到LLM空间的投影 (可选，作为额外prompt)
            self.trend_to_llm = nn.Linear(configs.d_model, self.d_llm)
        else:
            self.tapr = None

        # ========== 新增: GRAM_v2模块 (严格触发) ==========
        if self.use_gram:
            print("[TimeLLM-Enhanced-v2] Initializing GRAM_v2 module (strict mode)...")
            self.gram = GRAM_v2(configs)
            self.retrieval_context = RetrievalContext(configs.d_model, self.d_llm, configs.dropout)
        else:
            self.gram = None

        # 辅助损失信息
        self.aux_loss_info = {}

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        前向预测 - 与原Time-LLM数据流完全一致

        新增模块作为"旁路观察者"：
        - TAPR: 观察enc_out，生成趋势嵌入，但不修改enc_out
        - GRAM: 严格检索，绝大多数情况返回空，不影响流程
        """
        # ========== Step 1: 实例归一化 (与原版相同) ==========
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # ========== Step 2: 统计特征 (与原版相同) ==========
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # ========== Step 3: 构建Prompt (与原版相同) ==========
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # ========== Step 4: Prompt嵌入 (与原版相同) ==========
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # ========== Step 5: Source Embeddings (与原版相同) ==========
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # ========== Step 6: Patch Embedding (与原版相同) ==========
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.float())
        enc_out = enc_out.to(prompt_embeddings.dtype)

        # ========== 新增 Step 6.5: TAPR观察 (不修改enc_out!) ==========
        trend_embed_llm = None
        trend_info = None
        if self.tapr is not None:
            # 关键区别: enc_out不被修改，tapr只是"观察"
            _, trend_embed, trend_info = self.tapr(enc_out, return_trend_info=self.training)

            # 趋势嵌入作为可选的额外prompt token
            if trend_embed is not None:
                trend_embed_llm = self.trend_to_llm(trend_embed).unsqueeze(1)  # [B*N, 1, d_llm]

            if trend_info is not None:
                self.aux_loss_info['trend_info'] = trend_info

        # ========== 新增 Step 6.7: GRAM观察 (极少触发) ==========
        retrieval_context_llm = None
        if self.gram is not None:
            # 关键区别: enc_out不被修改，gram只在高相似度时返回上下文
            _, context_embed, retrieval_info = self.gram(enc_out, return_retrieval_info=self.training)

            # 如果检索触发，生成上下文嵌入
            if retrieval_info is not None and retrieval_info.get('retrieval_triggered', False):
                retrieval_context_llm = self.retrieval_context(context_embed)  # [B*N, 1, d_llm]
                self.aux_loss_info['retrieval_info'] = retrieval_info

        # ========== Step 7: Reprogramming (与原版相同) ==========
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # ========== Step 8: 拼接Embeddings (可选添加新模块嵌入) ==========
        llm_input_parts = []

        # 可选: 添加趋势嵌入
        if trend_embed_llm is not None:
            llm_input_parts.append(trend_embed_llm)

        # 可选: 添加检索上下文
        if retrieval_context_llm is not None:
            llm_input_parts.append(retrieval_context_llm)

        llm_input_parts.append(prompt_embeddings)
        llm_input_parts.append(enc_out)

        llama_enc_out = torch.cat(llm_input_parts, dim=1)

        # ========== Step 9: LLM Forward (与原版相同) ==========
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        # ========== Step 10: 输出投影 (与原版相同) ==========
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # ========== 可选 Step 10.5: 趋势冲突融合 (默认关闭) ==========
        if self.apply_trend_fusion and self.tapr is not None and trend_info is not None:
            dec_out, fusion_info = self.tapr.apply_output_fusion(dec_out, trend_info)
            if fusion_info.get('adjustment_applied', False):
                self.aux_loss_info['fusion_info'] = fusion_info

        # ========== Step 11: 反归一化 (与原版相同) ==========
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        """计算Top-5自相关lags (与原版相同)"""
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def compute_auxiliary_loss(self, y_true, current_step=None):
        """
        计算辅助损失 (极低权重)

        与v1的区别:
        - 权重降低50倍 (0.5 -> 0.01)
        - 仅作为正则化，不影响主任务
        """
        device = y_true.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # TAPR趋势辅助损失
        if self.tapr is not None and 'trend_info' in self.aux_loss_info:
            trend_loss, trend_dict = self.tapr.compute_auxiliary_loss(
                self.aux_loss_info['trend_info'],
                y_true,
                lambda_dir=0.01,  # 极低权重
                lambda_mag=0.01,
                warmup_steps=1000,  # 更长的warmup
                current_step=current_step
            )
            total_loss = total_loss + self.lambda_trend * trend_loss
            loss_dict.update({f'tapr_{k}': v for k, v in trend_dict.items()})

        # GRAM检索辅助损失
        if self.gram is not None and 'retrieval_info' in self.aux_loss_info:
            retrieval_loss, retrieval_dict = self.gram.compute_retrieval_loss(
                self.aux_loss_info['retrieval_info'],
                None,
                y_true,
                lambda_ref=0.001,  # 极低权重
                warmup_steps=1000,
                current_step=current_step
            )
            total_loss = total_loss + self.lambda_retrieval * retrieval_loss
            loss_dict.update({f'gram_{k}': v for k, v in retrieval_dict.items()})

        # 清空辅助信息
        self.aux_loss_info = {}

        return total_loss, loss_dict

    def build_retrieval_memory(self, train_loader, device):
        """构建GRA-M检索记忆库"""
        if self.gram is not None:
            print("[TimeLLM-Enhanced-v2] Building retrieval memory bank (strict mode)...")
            self.gram.build_memory_from_dataloader(train_loader, self, device)
            print("[TimeLLM-Enhanced-v2] Memory bank built successfully")


class ReprogrammingLayer(nn.Module):
    """
    重编程层 - 与原版完全相同
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

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

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
