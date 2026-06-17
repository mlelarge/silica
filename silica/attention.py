"""Attention dispatch: fp KV -> mx.fast SDPA; quantized KV -> manual path.

The audit established (mlx-1) that `mx.fast.scaled_dot_product_attention` takes
only fp16/bf16/fp32 — there is no native quantized-KV SDPA in mainline MLX. So a
quantized KV cache must go through a hand-written path built from two
`mx.quantized_matmul` calls + softmax, mirroring mlx-lm's
`quantized_scaled_dot_product_attention`. This is the ~0.5x-fp16 "memory for
speed" path the plan warns about (PLAN §7) — correct, not fast.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map

from .cache import QuantizedKVCache


def sdpa(queries, keys, values, *, scale, mask, cache=None):
    """GQA + causal attention. Branches on whether the cache is quantized.

    `keys`/`values` are plain arrays for an fp cache, or (packed, scales, biases)
    tuples for a QuantizedKVCache.
    """
    if isinstance(cache, QuantizedKVCache):
        return quantized_sdpa(
            queries, keys, values, scale=scale, mask=mask,
            group_size=cache.group_size, bits=cache.bits,
        )
    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)


def quantized_sdpa(queries, q_keys, q_values, *, scale, mask, group_size, bits):
    """Attention with a quantized KV cache (mirrors mlx-lm's base.py).

    `q_keys`/`q_values` are (packed_uint32, scales, biases) tuples. GQA is
    handled by reshaping queries into (B, n_kv, n_repeats, L, D) and broadcasting
    the kv heads with an inserted axis.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        scores = scores + mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out
