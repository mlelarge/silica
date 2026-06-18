"""Qwen3 decoder. The Qwen3-specific bit is per-head QK-RMSNorm before RoPE."""

from __future__ import annotations

import mlx.nn as nn

from ..config import ModelConfig
from ..attention import sdpa
from .common import CausalLM, build_rope


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5

        bias = cfg.attention_bias                 # False for Qwen3
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=bias)

        # Qwen3: per-head QK-RMSNorm over head_dim, applied BEFORE RoPE.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rope = build_rope(cfg)

    def __call__(self, x, mask=None, cache=None):
        b, seq, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, seq, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = sdpa(q, k, v, scale=self.scale, mask=mask, cache=cache)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, seq, -1))


class Qwen3ForCausalLM(CausalLM):
    attention_cls = Qwen3Attention


ARCHITECTURES = ("Qwen3ForCausalLM",)
