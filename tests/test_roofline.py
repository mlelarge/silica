"""Pure-Python tests for the corrected achieved-bandwidth byte model.

Encodes the audit's methodology corrections (KV counted; quantized footprint
includes scales+biases). Runs with no MLX installed.
"""

from silica.config import ModelConfig
from bench.roofline import byte_budget, weight_bytes_per_token, kv_bytes_per_token

QWEN3_0_6B = dict(
    hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8, head_dim=128, intermediate_size=3072, vocab_size=151936,
)
QWEN3_4B = dict(
    hidden_size=2560, num_hidden_layers=36, num_attention_heads=32,
    num_key_value_heads=8, head_dim=128, intermediate_size=9728, vocab_size=151936,
)


def test_quantized_weights_smaller_than_fp16():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    fp16 = weight_bytes_per_token(cfg, bits=None)
    q4 = weight_bytes_per_token(cfg, bits=4, group_size=64, embed_bits=6)
    assert q4 < fp16


def test_affine_overhead_above_naive_half():
    # 4-bit should be MORE than params*0.5 bytes because of per-group scale+bias.
    from bench.roofline import _affine_bits_per_weight
    bpw = _affine_bits_per_weight(4, 64)
    assert 4.4 < bpw < 4.6  # ~4.5 effective bits/weight, not 4.0


def test_kv_bytes_grow_with_context():
    cfg = ModelConfig.from_dict(QWEN3_4B)
    assert kv_bytes_per_token(cfg, 0) == 0
    assert kv_bytes_per_token(cfg, 8192) > 0
    assert kv_bytes_per_token(cfg, 131072) > kv_bytes_per_token(cfg, 8192)


def test_kv_dominates_weights_at_long_context_qwen3_4b():
    # The audit's headline: for Qwen3-4B, KV ~10x weights at 128k context.
    cfg = ModelConfig.from_dict(QWEN3_4B)
    b = byte_budget(cfg, 131072, bits=4, group_size=64, embed_bits=6)
    assert b.kv > 5 * b.weights


def test_total_is_weights_plus_kv():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    b = byte_budget(cfg, 4096, bits=4)
    assert abs(b.total - (b.weights + b.kv)) < 1.0


def test_pct_of_peak_formula():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    b = byte_budget(cfg, 0, bits=4)
    # achieved = bytes * tok/s ; pct = 100*achieved/peak
    tok_s, peak = 100.0, 400.0
    assert abs(b.pct_of_peak(tok_s, peak)
               - 100.0 * b.achieved_bandwidth_gbps(tok_s) / peak) < 1e-9
