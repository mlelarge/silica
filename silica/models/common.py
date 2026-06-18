"""Shared layer library for silica's model files (the SGLang-style split).

Per-architecture files (`qwen3.py`, `llama.py`) compose these blocks and supply
only what differs (their attention module). Everything model-agnostic — the MLP,
the decoder stack, the tied/untied lm_head, the causal mask, and RoPE
construction (incl. llama3 scaling) — lives here, so a new architecture is a
small file, not a re-implementation.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig


def causal_additive_mask(seq_len: int, offset: int, dtype) -> mx.array | None:
    """Offset-aware additive causal mask, shape (seq_len, offset+seq_len).

    None for single-token decode (a query attends to all cached keys).
    """
    if seq_len <= 1:
        return None
    total = offset + seq_len
    q_pos = mx.arange(offset, total).reshape(seq_len, 1)
    k_pos = mx.arange(total).reshape(1, total)
    allowed = k_pos <= q_pos
    neg_inf = mx.array(float("-inf"), dtype=dtype)
    return mx.where(allowed, mx.array(0.0, dtype=dtype), neg_inf)


def _llama3_freqs(dims: int, base: float, scaling: dict) -> mx.array:
    """Frequencies for llama3 RoPE scaling (Llama-3.1/3.2). Mirrors mlx-lm: the
    spectrum is rescaled by `factor` below a low-freq wavelength, kept above a
    high-freq one, and interpolated between — applied to ALL positions, so it is
    parity-critical, not just a long-context knob."""
    factor = scaling["factor"]
    low = scaling.get("low_freq_factor", 1.0)
    high = scaling.get("high_freq_factor", 4.0)
    old = scaling.get("original_max_position_embeddings", 8192)
    low_wl = old / low
    high_wl = old / high
    freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    wavelens = 2 * math.pi * freqs
    freqs = mx.where(wavelens > low_wl, freqs * factor, freqs)
    is_medium = (wavelens > high_wl) & (wavelens < low_wl)
    smooth = (old / wavelens - low) / (high - low)
    smooth_freqs = freqs / ((1 - smooth) / factor + smooth)
    return mx.where(is_medium, smooth_freqs, freqs)


def build_rope(cfg: ModelConfig):
    """Return a callable `rope(x, offset=0)`. Plain RoPE unless the config asks
    for llama3 scaling (then a closure with custom freqs — kept OUT of the module
    parameter tree, so it doesn't need to appear in the checkpoint)."""
    scaling = cfg.rope_scaling
    if not scaling or scaling.get("rope_type") != "llama3":
        return nn.RoPE(cfg.head_dim, traditional=False, base=cfg.rope_theta)
    dims = cfg.head_dim
    freqs = _llama3_freqs(dims, cfg.rope_theta, scaling)

    def rope(x, offset=0):
        return mx.fast.rope(x, dims, traditional=False, base=None, scale=1.0,
                            offset=offset, freqs=freqs)

    return rope


class MLP(nn.Module):
    """SwiGLU MLP (shared by Qwen3 and Llama)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# --- Mixture-of-Experts (sparse MLP) ---------------------------------------- #
# A transparent reimplementation of mlx-lm's SwitchGLU: experts are stacked into
# (num_experts, out, in) tensors and a token's top-k experts are computed with
# MLX's native gathered matmul (`mx.gather_mm` / `mx.gather_qmm` for quantized).

class SwitchLinear(nn.Module):
    """Per-expert Linear: weight (num_experts, output_dims, input_dims)."""

    def __init__(self, input_dims, output_dims, num_experts, bias=False):
        super().__init__()
        scale = (1.0 / input_dims) ** 0.5
        self.weight = mx.random.uniform(low=-scale, high=scale,
                                        shape=(num_experts, output_dims, input_dims))
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    def __call__(self, x, indices):
        x = mx.gather_mm(x, self["weight"].swapaxes(-1, -2), rhs_indices=indices)
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size=64, bits=4, mode="affine"):
        ne, od, idd = self.weight.shape
        ql = QuantizedSwitchLinear(idd, od, ne, group_size=group_size, bits=bits)
        ql.weight, ql.scales, *bs = mx.quantize(self.weight, group_size, bits, mode=mode)
        ql.biases = bs[0] if bs else None
        if "bias" in self:
            ql.bias = self.bias
        return ql


class QuantizedSwitchLinear(nn.Module):
    def __init__(self, input_dims, output_dims, num_experts, group_size=64, bits=4):
        super().__init__()
        w, s, *bs = mx.quantize(mx.zeros((num_experts, output_dims, input_dims)),
                                group_size, bits)
        self.weight, self.scales = w, s
        self.biases = bs[0] if bs else None
        self.group_size, self.bits = group_size, bits
        self.freeze()

    def __call__(self, x, indices):
        return mx.gather_qmm(x, self["weight"], self["scales"], self.get("biases"),
                             rhs_indices=indices, transpose=True,
                             group_size=self.group_size, bits=self.bits)


class SwitchGLU(nn.Module):
    """SwiGLU over gathered experts (gate/up/down are SwitchLinears)."""

    def __init__(self, input_dims, hidden_dims, num_experts, bias=False):
        super().__init__()
        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        x_up = self.up_proj(x, indices)
        x_gate = self.gate_proj(x, indices)
        x = self.down_proj(nn.silu(x_gate) * x_up, indices)
        return x.squeeze(-2)


class MoEBlock(nn.Module):
    """Router (softmax over experts) -> top-k -> weighted SwitchGLU sum."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.norm_topk_prob = cfg.norm_topk_prob
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(cfg.hidden_size, cfg.expert_intermediate_size, cfg.num_experts)

    def __call__(self, x):
        b, length, d = x.shape
        xf = x.reshape(-1, d)
        weights = mx.softmax(self.gate(xf), axis=-1, precise=True)
        k = self.top_k
        idx = mx.stop_gradient(mx.argpartition(-weights, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(weights, idx, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(xf, idx)
        y = (y * scores[..., None]).sum(axis=-2)
        return y.reshape(b, length, d)


def moe_sanitize(weights, n_layers: int, num_experts: int):
    """Stack HF per-expert weights into SwitchGLU's (num_experts, out, in) tensors."""
    if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
        return weights
    for layer in range(n_layers):
        p = f"model.layers.{layer}"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in ("weight", "scales", "biases"):
                if f"{p}.mlp.experts.0.{proj}.{suf}" in weights:
                    stacked = [weights.pop(f"{p}.mlp.experts.{e}.{proj}.{suf}")
                               for e in range(num_experts)]
                    weights[f"{p}.mlp.switch_mlp.{proj}.{suf}"] = mx.stack(stacked)
    return weights


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, attention_cls, mlp_cls):
        super().__init__()
        self.self_attn = attention_cls(cfg)
        self.mlp = mlp_cls(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Decoder(nn.Module):
    """Embedding -> N decoder layers -> final norm. Names match HF checkpoints."""

    def __init__(self, cfg: ModelConfig, attention_cls, mlp_cls):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg, attention_cls, mlp_cls)
                       for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        offset = cache[0].offset if cache[0] is not None else 0
        mask = causal_additive_mask(h.shape[1], offset, h.dtype)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class CausalLM(nn.Module):
    """Decoder-only LM base. Subclasses set `attention_cls` (and `mlp_cls` for
    MoE). Handles tied vs untied lm_head (tied -> the embedding matrix IS the
    output projection)."""

    attention_cls = None
    mlp_cls = MLP                 # dense by default; MoE models set MoEBlock

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = Decoder(cfg, type(self).attention_cls, type(self).mlp_cls)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        h = self.model(inputs, cache)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        """Hook for HF->silica weight remapping (MoE models stack experts)."""
        return weights
