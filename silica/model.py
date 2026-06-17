"""Qwen3 decoder, built against `mx.fast.*`.

This module exists to be *read*. Every Qwen3-specific detail the audit flagged
as a parity-breaker is called out inline:

  * per-head QK-RMSNorm (`q_norm`/`k_norm`) over head_dim, BEFORE RoPE  <-- Qwen3
  * NO QKV bias (Qwen2 had it)
  * head_dim read from config, decoupled from hidden//n_heads
  * tied lm_head on the small models (embed_tokens.as_linear)

Module/parameter names match the HuggingFace Qwen3 checkpoint key layout so a
checkpoint loads with `model.load_weights(...)` and no renaming (see weights.py).

NOTE (unvalidated): this has not been run on device. The M0 parity gate
(tests/test_parity.py) is what proves these numerics.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .cache import KVCache
from .attention import sdpa


def causal_additive_mask(seq_len: int, offset: int, dtype) -> mx.array | None:
    """Offset-aware additive causal mask, shape (seq_len, offset+seq_len).

    Returns None for single-token decode (a query attends to all cached keys).
    For prefill at offset 0 this is the standard lower-triangular mask; the
    offset term keeps it correct for (future) chunked prefill.
    """
    if seq_len <= 1:
        return None
    total = offset + seq_len
    q_pos = mx.arange(offset, total).reshape(seq_len, 1)
    k_pos = mx.arange(total).reshape(1, total)
    allowed = k_pos <= q_pos
    neg_inf = mx.array(float("-inf"), dtype=dtype)
    return mx.where(allowed, mx.array(0.0, dtype=dtype), neg_inf)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim            # decoupled — from config
        self.scale = self.head_dim**-0.5

        bias = cfg.attention_bias               # False for Qwen3
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)

        # Qwen3 per-head QK normalization (over head_dim), applied before RoPE.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        # TODO(M1.5): honor cfg.rope_scaling (YaRN) for >native_context_length.
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def __call__(self, x, mask=None, cache: KVCache | None = None):
        b, l, _ = x.shape

        q = self.q_proj(x).reshape(b, l, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self.n_kv_heads, self.head_dim)

        # QK-norm per head (normalizes over the last dim == head_dim) BEFORE RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # -> (b, n_heads, l, head_dim) for RoPE + SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # fp KV -> mx.fast SDPA; quantized KV -> manual quantized_matmul path.
        out = sdpa(q, k, v, scale=self.scale, mask=mask, cache=cache)
        out = out.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x, mask=None, cache: KVCache | None = None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, inputs, cache: list[KVCache] | None = None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        offset = cache[0].offset if cache[0] is not None else 0
        mask = causal_additive_mask(h.shape[1], offset, h.dtype)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Qwen3ForCausalLM(nn.Module):
    """Top-level model. Returns logits over the vocabulary."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = Qwen3Model(cfg)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        h = self.model(inputs, cache)
        if self.config.tie_word_embeddings:
            # 0.6B: the embedding matrix IS the output projection.
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers
