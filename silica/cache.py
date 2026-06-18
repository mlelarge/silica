"""KV cache.

M0 ships only the growing cache. The audit established two facts that shape the
later options and are documented here so they are not rediscovered the hard way:

  * `growing -> {quantized | rotating}` are ALTERNATIVES, not a composition.
    In mainline mlx-lm, `RotatingKVCache.to_quantized()` raises
    NotImplementedError — the ring-buffer's temporal reordering makes in-place
    quantization complex. A quantized *and* rotating cache is an open problem.
  * A rotating cache makes output a lossy function of history, so the M0/M1
    logits-parity gate is only exact for prompts shorter than the rotation
    threshold (see RotatingKVCache below / PLAN §5a).

The growing cache pre-allocates in `step`-sized chunks (mlx-lm pattern) so the
sequence dimension changes in coarse jumps rather than every token — relevant
for the M2 `mx.compile` experiment, which recompiles on shape change.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map


class KVCache:
    """Growing per-layer KV cache. One instance per decoder layer.

    Stored layout matches `mx.fast.scaled_dot_product_attention` inputs:
    `(batch, n_kv_heads, seq, head_dim)`.
    """

    def __init__(self, step: int = 256):
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset: int = 0          # logical length (valid tokens)
        self.step = step

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append `keys`/`values` for this step, return the full valid slice."""
        prev = self.offset
        n_new = keys.shape[2]
        need = prev + n_new

        if self.keys is None or need > self.keys.shape[2]:
            b, n_kv, _, hd = keys.shape
            n_steps = (need + self.step - 1) // self.step
            new_len = n_steps * self.step
            k_buf = mx.zeros((b, n_kv, new_len, hd), keys.dtype)
            v_buf = mx.zeros((b, n_kv, new_len, hd), values.dtype)
            if self.keys is not None:
                # carry over the already-valid portion, drop trailing padding
                k_buf[..., :prev, :] = self.keys[..., :prev, :]
                v_buf[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = k_buf, v_buf

        self.keys[..., prev:need, :] = keys
        self.values[..., prev:need, :] = values
        self.offset = need
        return self.keys[..., :need, :], self.values[..., :need, :]

    def to_quantized(self, group_size: int = 64, bits: int = 8) -> "QuantizedKVCache":
        """Convert the current fp cache into a quantized one (for quantized_kv_start)."""
        q = QuantizedKVCache(group_size=group_size, bits=bits)
        q.offset = self.offset
        if self.keys is not None:
            k = self.keys[..., : self.offset, :]
            v = self.values[..., : self.offset, :]
            q.keys = mx.quantize(k, group_size=group_size, bits=bits)
            q.values = mx.quantize(v, group_size=group_size, bits=bits)
        return q


class QuantizedKVCache:
    """KV cache that stores K and V quantized (memory-for-speed; PLAN §7).

    Mirrors mlx-lm's QuantizedKVCache: `keys`/`values` are (packed_uint32,
    scales, biases) tuples grown in `step`-sized chunks. `update_and_fetch`
    returns those tuples; the quantized-SDPA path in attention.py consumes them.

    NOTE: this does NOT compose with RotatingKVCache (the audit's strat-5 / the
    cache.py module docstring) — they are alternative long-context strategies.
    """

    def __init__(self, group_size: int = 64, bits: int = 8, step: int = 256):
        self.keys = None       # (packed, scales, biases)
        self.values = None
        self.offset = 0
        self.step = step
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        b, n_kv, n_new, k_dim = keys.shape
        v_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + n_new) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_len = (self.step + n_new - 1) // self.step * self.step  # padded total length
            shape = (b, n_kv, new_len)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand(x):
                return mx.concatenate(
                    [x, mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)], axis=-2
                )

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                self.keys, self.values = tree_map(expand, (self.keys, self.values))
            else:
                self.keys, self.values = init_quant(k_dim), init_quant(v_dim)

        self.offset += n_new
        qk = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        qv = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = qk[i]
            self.values[i][..., prev : self.offset, :] = qv[i]

        return (
            tree_map(lambda x: x[..., : self.offset, :], self.keys),
            tree_map(lambda x: x[..., : self.offset, :], self.values),
        )


def make_cache(
    n_layers: int, step: int = 256, *, kv_bits: int | None = None, kv_group_size: int = 64
):
    """Per-layer caches. `kv_bits=None` -> fp growing cache; else quantized KV."""
    if kv_bits is None:
        return [KVCache(step=step) for _ in range(n_layers)]
    return [QuantizedKVCache(group_size=kv_group_size, bits=kv_bits, step=step)
            for _ in range(n_layers)]


class RotatingKVCache:
    """Sliding-window cache with attention sinks (StreamingLLM).

    NOT used in M0. Stubbed here to pin the semantics the audit flagged as
    undefined: keep the first `keep` (sink) tokens plus a window of the most
    recent `max_size - keep` tokens; evict the oldest non-sink tokens.

    Parity caveat: eviction begins once offset > max_size; below that this is
    equivalent to the growing cache. Do not attempt to quantize this (see
    module docstring).
    """

    def __init__(self, max_size: int = 1024, keep: int = 4, step: int = 256):
        self.max_size = max_size
        self.keep = keep
        self.step = step
        self.offset = 0
        raise NotImplementedError(
            "RotatingKVCache is an M1.5 deliverable; see PLAN §5. "
            "It does not compose with quantization."
        )
