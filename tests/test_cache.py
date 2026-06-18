"""KV cache, mask, and quantized-SDPA tests (need MLX, no checkpoint).

Closes the audit's biggest test gaps: multi-chunk KV growth (offset crossing the
step boundary), the offset-aware causal mask (chunked-prefill path), and the
quantized-KV GQA attention path — all with tiny synthetic tensors, no hardware
gating beyond MLX itself.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from silica.cache import KVCache, QuantizedKVCache, RotatingKVCache
from silica.model import causal_additive_mask
from silica.attention import quantized_sdpa


def test_kvcache_growth_across_step_boundary():
    """Feeding tokens one at a time past `step` must preserve all prior K/V."""
    B, n_kv, D, step = 1, 2, 8, 4
    c = KVCache(step=step)
    fed_k = []
    out_k = None
    for t in range(11):                       # crosses step=4 at 4 and 8
        k = mx.random.normal((B, n_kv, 1, D))
        v = mx.random.normal((B, n_kv, 1, D))
        fed_k.append(k)
        out_k, _ = c.update_and_fetch(k, v)
        assert c.offset == t + 1
        assert out_k.shape == (B, n_kv, t + 1, D)
    full = mx.concatenate(fed_k, axis=2)      # (B, n_kv, 11, D)
    mx.eval(out_k, full)
    assert mx.allclose(out_k, full), "growth across the step boundary lost K data"


def test_kvcache_multi_token_prefill_then_decode():
    """A multi-token prefill chunk + a later chunk grow correctly."""
    B, n_kv, D = 1, 1, 8
    c = KVCache(step=4)
    k1 = mx.random.normal((B, n_kv, 6, D))    # L=6 > step -> immediate grow to 8
    ok, _ = c.update_and_fetch(k1, k1)
    assert c.offset == 6 and ok.shape == (B, n_kv, 6, D)
    k2 = mx.random.normal((B, n_kv, 5, D))    # -> offset 11, grow to 12
    ok, _ = c.update_and_fetch(k2, k2)
    assert c.offset == 11
    mx.eval(ok)
    assert mx.allclose(ok, mx.concatenate([k1, k2], axis=2))


def test_causal_additive_mask_offset():
    """Offset-aware mask: query at abs-pos offset+i may attend columns 0..offset+i."""
    m = causal_additive_mask(seq_len=3, offset=2, dtype=mx.float32)
    assert m.shape == (3, 5)
    rows = m.tolist()
    for i in range(3):
        for j in range(5):
            allowed = j <= 2 + i
            assert (rows[i][j] == 0.0) if allowed else (rows[i][j] == float("-inf")), (i, j)
    assert causal_additive_mask(1, 7, mx.float32) is None   # single-token decode


def test_quantized_sdpa_matches_fp_on_dequantized_gqa():
    """The quantized GQA path (n_repeats=2) ~= fp SDPA on the dequantized K/V."""
    B, n_q, n_kv, L, S, D = 1, 4, 2, 3, 5, 64          # n_repeats=2, D divisible by g64
    q = mx.random.normal((B, n_q, L, D))
    k = mx.random.normal((B, n_kv, S, D))
    v = mx.random.normal((B, n_kv, S, D))
    scale = D**-0.5

    qc = QuantizedKVCache(group_size=64, bits=8)
    qk, qv = qc.update_and_fetch(k, v)
    k_deq = mx.dequantize(*qk, group_size=64, bits=8)
    v_deq = mx.dequantize(*qv, group_size=64, bits=8)

    out_q = quantized_sdpa(q, qk, qv, scale=scale, mask=None, group_size=64, bits=8)
    out_ref = mx.fast.scaled_dot_product_attention(q, k_deq, v_deq, scale=scale, mask=None)
    mx.eval(out_q, out_ref)

    a, b = out_q.reshape(-1), out_ref.reshape(-1)
    cos = float((a * b).sum().item() / (((a * a).sum().item() ** 0.5) * ((b * b).sum().item() ** 0.5)))
    assert cos > 0.999, f"quantized GQA SDPA diverged from dequant+fp SDPA (cos {cos:.4f})"


def test_quantized_kvcache_growth_across_step_boundary():
    """Quantized cache must preserve K/V (within 8-bit error) across growth."""
    B, n_kv, D, step = 1, 2, 64, 4
    c = QuantizedKVCache(group_size=64, bits=8, step=step)
    fed = []
    qk = None
    for t in range(11):                       # crosses step=4 at 4 and 8
        k = mx.random.normal((B, n_kv, 1, D))
        fed.append(k)
        qk, _ = c.update_and_fetch(k, k)
        assert c.offset == t + 1
    deq = mx.dequantize(*qk, group_size=64, bits=8)
    full = mx.concatenate(fed, axis=2)
    mx.eval(deq, full)
    assert mx.max(mx.abs(deq - full)).item() < 0.1   # 8-bit round-trip tolerance


def test_rotating_cache_is_an_unimplemented_stub():
    with pytest.raises(NotImplementedError):
        RotatingKVCache()
