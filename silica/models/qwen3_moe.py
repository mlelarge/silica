"""Qwen3-MoE (e.g. Qwen3-30B-A3B: 128 experts, top-8). Same attention as dense
Qwen3 (per-head QK-norm, GQA, llama-style RoPE θ=1e6); the MLP is the shared
MoEBlock with `moe_intermediate_size` experts. The headline bandwidth demo:
a 30B model that reads only ~3B of active weights per token.
"""

from __future__ import annotations

from .common import CausalLM, MoEBlock, moe_sanitize
from .qwen3 import Qwen3Attention


class Qwen3MoeForCausalLM(CausalLM):
    attention_cls = Qwen3Attention
    mlp_cls = MoEBlock

    def sanitize(self, weights):
        return moe_sanitize(weights, self.config.num_hidden_layers, self.config.num_experts)


ARCHITECTURES = ("Qwen3MoeForCausalLM",)
