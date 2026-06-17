"""Byte-traffic model for the achieved-bandwidth figure of merit.

The audit's central methodology correction: the decode figure of merit must
count *all* bytes read per token, not just weights.

    total_bytes/token = weight_bytes + kv_bytes(context_len)
                        + embedding_bytes + lm_head_bytes

  * weight_bytes use the on-device QUANTIZED footprint INCLUDING per-group
    scales+biases (affine stores both -> ~12.5% over packed bytes at 4-bit/g64,
    ~4.5 effective bits/weight), not params*bits/8.
  * kv_bytes grow linearly with context and dominate at long context (for
    Qwen3-4B, KV ~= weights by ~8k tokens, ~10x by 128k). Report bandwidth-%
    AS A FUNCTION OF context length.

These are analytic estimates for the scoreboard denominator; pair them with the
measured tok/s in bench/decode.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from silica.config import ModelConfig


def _affine_bits_per_weight(bits: int, group_size: int, scale_bias_bits: int = 16) -> float:
    """Effective bits/weight including a per-group scale AND bias (MLX affine)."""
    return bits + (2 * scale_bias_bits) / group_size


def linear_param_count(cfg: ModelConfig) -> dict[str, int]:
    """Per-token-relevant weight parameter counts by group (one decode step
    reads every weight exactly once at batch=1)."""
    h, i = cfg.hidden_size, cfg.intermediate_size
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    per_layer_attn = h * (nq * hd) + 2 * h * (nkv * hd) + (nq * hd) * h  # q,k,v,o
    per_layer_mlp = 3 * h * i                                            # gate,up,down
    body = cfg.num_hidden_layers * (per_layer_attn + per_layer_mlp)
    embed = cfg.vocab_size * h          # also the lm_head when tied
    return {"body": body, "embed_lm_head": embed}


def weight_bytes_per_token(
    cfg: ModelConfig, bits: int | None = None, group_size: int = 64,
    embed_bits: int | None = None,
) -> float:
    """Bytes of weights read per decode token. bits=None -> fp16 (2 bytes)."""
    counts = linear_param_count(cfg)
    if bits is None:
        body_bpw = embed_bpw = 16.0
    else:
        body_bpw = _affine_bits_per_weight(bits, group_size)
        eb = embed_bits if embed_bits is not None else 16  # often kept higher/fp
        embed_bpw = _affine_bits_per_weight(eb, group_size) if embed_bits else 16.0
    # tied lm_head: embedding read once for input, once as output projection
    return (counts["body"] * body_bpw + counts["embed_lm_head"] * embed_bpw * 2) / 8.0


def kv_bytes_per_token(cfg: ModelConfig, context_len: int, kv_bits: int | None = None) -> float:
    """Bytes of KV cache read per decode token at a given context length."""
    bpw = (kv_bits + 2 * 16 / 64) / 8.0 if kv_bits else 2.0  # fp16 default
    per_token_kv = cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2
    return per_token_kv * bpw * context_len


@dataclass
class ByteBudget:
    weights: float
    kv: float
    context_len: int

    @property
    def total(self) -> float:
        return self.weights + self.kv

    def achieved_bandwidth_gbps(self, tok_per_s: float) -> float:
        return self.total * tok_per_s / 1e9

    def pct_of_peak(self, tok_per_s: float, peak_gbps: float) -> float:
        return 100.0 * self.achieved_bandwidth_gbps(tok_per_s) / peak_gbps


def byte_budget(
    cfg: ModelConfig, context_len: int, *, bits: int | None = None,
    group_size: int = 64, embed_bits: int | None = None, kv_bits: int | None = None,
) -> ByteBudget:
    return ByteBudget(
        weights=weight_bytes_per_token(cfg, bits, group_size, embed_bits),
        kv=kv_bytes_per_token(cfg, context_len, kv_bits),
        context_len=context_len,
    )
