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

    @property
    def state(self):
        """For mx.compile inputs=/outputs= capture (M2)."""
        return self.keys, self.values, self.offset


def make_cache(n_layers: int, step: int = 256) -> list[KVCache]:
    return [KVCache(step=step) for _ in range(n_layers)]


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
