"""Token sampling.

M0 uses greedy only. The richer controls (temperature/top-k/top-p/min-p) land in
M1.5 — the Qwen3 card explicitly advises against greedy decoding for real use.

Design note (audit mlx-3): the sampler returns an `mx.array`, never a Python
int. Calling `.item()` inside the decode loop forces a host↔device sync and
collapses the `async_eval` overlap (M2). Keep the token lazy; read it only in
the detokenizer, after the next step has been enqueued.
"""

from __future__ import annotations

import mlx.core as mx

from .config import GenConfig


def make_sampler(cfg: GenConfig):
    """Return `sampler(logits) -> token_ids` for logits shaped (B, vocab).

    RNG is per-sampler, not global: we hold an `mx.random.key` derived from
    `cfg.seed` and split it each step, instead of calling the process-global
    `mx.random.seed`. So an explicit seed makes THIS run reproducible without
    clobbering global state, and `seed=None` draws from the default stream (which
    advances naturally — two generations differ). Seeding globally per call was a
    bug: with the old `seed=0` default every sampled generation was identical.
    """
    greedy = cfg.temperature <= 0.0
    key = [mx.random.key(cfg.seed) if cfg.seed is not None else None]

    def sampler(logits: mx.array) -> mx.array:
        if greedy:
            return mx.argmax(logits, axis=-1)

        logits = logits * (1.0 / cfg.temperature)

        if cfg.top_k and cfg.top_k > 0:
            logits = _top_k(logits, cfg.top_k)
        if cfg.min_p and cfg.min_p > 0.0:
            logits = _min_p(logits, cfg.min_p)
        if cfg.top_p and cfg.top_p < 1.0:
            logits = _top_p(logits, cfg.top_p)

        if key[0] is None:
            return mx.random.categorical(logits, axis=-1)
        key[0], sub = mx.random.split(key[0])
        return mx.random.categorical(logits, axis=-1, key=sub)

    return sampler


def _top_k(logits: mx.array, k: int) -> mx.array:
    k = min(k, logits.shape[-1])
    kth = mx.sort(logits, axis=-1)[..., -k][..., None]
    return mx.where(logits < kth, mx.array(float("-inf"), logits.dtype), logits)


def _min_p(logits: mx.array, min_p: float) -> mx.array:
    probs = mx.softmax(logits, axis=-1)
    top = mx.max(probs, axis=-1, keepdims=True)
    return mx.where(probs < min_p * top, mx.array(float("-inf"), logits.dtype), logits)


def _top_p(logits: mx.array, top_p: float) -> mx.array:
    # Nucleus: drop the low-probability tail beyond cumulative mass top_p.
    sorted_idx = mx.argsort(logits, axis=-1)            # ascending
    sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
    cumprobs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    # keep tokens whose cumulative-from-top mass is within top_p
    keep = cumprobs > (1.0 - top_p)
    masked = mx.where(keep, sorted_logits, mx.array(float("-inf"), logits.dtype))
    # scatter back to original vocab order
    inv = mx.argsort(sorted_idx, axis=-1)
    return mx.take_along_axis(masked, inv, axis=-1)
