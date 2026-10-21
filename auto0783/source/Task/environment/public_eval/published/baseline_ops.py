# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Extracted executable definitions from the author's published decoder.
import math
from typing import Optional, Tuple, Union, List
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import logging
from configuration_qwen2_5_vl import Qwen2_5_VLConfig
logger = logging.get_logger(__name__)
class Qwen2RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.variance_epsilon}'

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class Qwen2_5_VLRotaryEmbedding(nn.Module):

    def __init__(self, config: Qwen2_5_VLConfig, device=None):
        super().__init__()
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get('rope_type', config.rope_scaling.get('type'))
        else:
            self.rope_type = 'default'
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        (inv_freq, self.attention_scaling) = self.rope_init_fn(self.config, device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            (inv_freq, self.attention_scaling) = self.rope_init_fn(self.config, device, seq_len=seq_len, **self.rope_kwargs)
            self.register_buffer('inv_freq', inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer('inv_freq', self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        x = hidden_states
        self = self.rotary_emb
        """
        if 'dynamic' in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

class Qwen2MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for (i, m) in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for (i, m) in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    (batch, num_key_value_heads, slen, head_dim) = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Qwen2_5_VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int]=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(f'Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class.')
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.seperate_vision_attention = False

    def add_vision_attention(self):
        self.vision_branch_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.vision_branch_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.vision_branch_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.vision_branch_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.seperate_vision_attention = True

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Cache]=None, output_attentions: bool=False, use_cache: bool=False, cache_position: Optional[torch.LongTensor]=None, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, image_position_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        self = self.self_attn
        """
        if image_position_mask is not None:
            if image_position_mask.ndim == 2:
                image_position_mask = image_position_mask.unsqueeze(-1)
        (bsz, q_len, _) = hidden_states.size()
        if self.seperate_vision_attention == True:
            query_states = self.vision_branch_q_proj(hidden_states) * image_position_mask + self.q_proj(hidden_states) * (1 - image_position_mask)
            key_states = self.vision_branch_k_proj(hidden_states) * image_position_mask + self.k_proj(hidden_states) * (1 - image_position_mask)
            value_states = self.vision_branch_v_proj(hidden_states) * image_position_mask + self.v_proj(hidden_states) * (1 - image_position_mask)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        (cos, sin) = position_embeddings
        (query_states, key_states) = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, self.rope_scaling['mrope_section'])
        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            (key_states, value_states) = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}')
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        if self.seperate_vision_attention == True:
            attn_output = self.vision_branch_o_proj(attn_output) * image_position_mask + self.o_proj(attn_output) * (1 - image_position_mask)
        else:
            attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return (attn_output, attn_weights, past_key_value)

class Qwen2_5_VLDecoderLayer(nn.Module):

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.use_sliding_window and config._attn_implementation != 'flash_attention_2':
            logger.warning_once(f'Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; unexpected results may be encountered.')
        config._attn_implementation = 'eager'
        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.seperate_vision_mlp = False
        self.seperate_vision_layernorms = False
        self.seperate_vision_adaptive_layernorms = False
        self.config = config

    def add_vision_mlp(self):
        self.vision_branch_mlp = Qwen2MLP(self.config)
        self.seperate_vision_mlp = True

    def add_vision_layernorms(self):
        self.vision_branch_input_layernorm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.vision_branch_post_attention_layernorm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.seperate_vision_layernorms = True

    def add_vision_adaptive_layernorms(self):
        from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
        self.vision_branch_input_adaptive_layernorm = AdaLayerNormZero(self.hidden_size)
        self.vision_branch_post_attention_layernorm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.seperate_vision_adaptive_layernorms = True

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_value: Optional[Tuple[torch.Tensor]]=None, output_attentions: Optional[bool]=False, use_cache: Optional[bool]=False, cache_position: Optional[torch.LongTensor]=None, position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, image_position_mask: Optional[torch.Tensor]=None, t_emb: Optional[torch.Tensor]=None, **kwargs) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        '\n        past_key_value=None\n        attention_mask=causal_mask\n        self = decoder_layer\n        '
        if image_position_mask is not None:
            if image_position_mask.ndim == 2:
                image_position_mask = image_position_mask.unsqueeze(-1)
        residual = hidden_states
        if self.seperate_vision_layernorms == True:
            hidden_states = self.vision_branch_input_layernorm(hidden_states) * image_position_mask + self.input_layernorm(hidden_states) * (1 - image_position_mask)
        elif self.seperate_vision_adaptive_layernorms == True:
            (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.vision_branch_input_adaptive_layernorm(hidden_states, emb=t_emb)
            hidden_states = norm_hidden_states * image_position_mask + self.input_layernorm(hidden_states) * (1 - image_position_mask)
        else:
            hidden_states = self.input_layernorm(hidden_states)
        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, image_position_mask=image_position_mask)
        if self.seperate_vision_adaptive_layernorms:
            hidden_states = gate_msa.unsqueeze(1) * hidden_states * image_position_mask + hidden_states * (1 - image_position_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        if self.seperate_vision_layernorms == True:
            hidden_states = self.vision_branch_post_attention_layernorm(hidden_states) * image_position_mask + self.post_attention_layernorm(hidden_states) * (1 - image_position_mask)
        elif self.seperate_vision_adaptive_layernorms == True:
            hidden_states = self.vision_branch_post_attention_layernorm(hidden_states) * image_position_mask + self.post_attention_layernorm(hidden_states) * (1 - image_position_mask)
            norm_hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            hidden_states = norm_hidden_states * image_position_mask + hidden_states * (1 - image_position_mask)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        if self.seperate_vision_mlp == True:
            hidden_states = self.vision_branch_mlp(hidden_states) * image_position_mask + self.mlp(hidden_states) * (1 - image_position_mask)
        else:
            hidden_states = self.mlp(hidden_states)
        if self.seperate_vision_adaptive_layernorms:
            hidden_states = gate_mlp.unsqueeze(1) * hidden_states * image_position_mask + hidden_states * (1 - image_position_mask)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
QWEN2_5_VL_ATTENTION_CLASSES = {"eager": Qwen2_5_VLAttention}
