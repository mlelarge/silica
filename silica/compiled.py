"""mx.compile experiment for the decode step (M2).

mlx-lm does NOT compile its decode loop, so this is the one perf lever the
baseline leaves on the table (PLAN §5 M2, audit strat-8). The audit flagged
stateful-cache friction as the risk; we sidestep it by:

  * threading the KV cache FUNCTIONALLY (concatenation), not via a mutating
    object — so the compiled graph is pure;
  * passing the RoPE position as a TRACED ARRAY offset (probed to work), so the
    growing position does not bake a constant into the graph;
  * compiling with shapeless=True so the per-step growth of the KV sequence
    dimension does not trigger recompilation.

Prefill stays eager. Greedy only (sampling would need to be compiled too).
Whether this actually beats the async_eval baseline is the M2 benchmark
question; this module is the correctness-tested implementation to measure.
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx

from .config import GenConfig
from .cache import make_cache


def _decode_forward(model, token, offset, k_cache, v_cache):
    """Pure single-token forward. Returns (logits[:, -1, :], new_k, new_v)."""
    h = model.model.embed_tokens(token)
    new_k, new_v = [], []
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        x = layer.input_layernorm(h)
        b, l, _ = x.shape
        q = a.q_norm(a.q_proj(x).reshape(b, l, a.n_heads, a.head_dim)).transpose(0, 2, 1, 3)
        k = a.k_norm(a.k_proj(x).reshape(b, l, a.n_kv_heads, a.head_dim)).transpose(0, 2, 1, 3)
        v = a.v_proj(x).reshape(b, l, a.n_kv_heads, a.head_dim).transpose(0, 2, 1, 3)
        q = a.rope(q, offset=offset)
        k = a.rope(k, offset=offset)
        kc = mx.concatenate([k_cache[i], k], axis=2)
        vc = mx.concatenate([v_cache[i], v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, kc, vc, scale=a.scale, mask=None)
        h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(b, l, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        new_k.append(kc)
        new_v.append(vc)
    h = model.model.norm(h)
    if model.config.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    return logits[:, -1, :], new_k, new_v


_COMPILED_STEPS: dict = {}


def make_compiled_step(model):
    """Return the shapeless-compiled per-token decode step, CACHED per model.

    mx.compile keys its cache on the wrapped function object, so building a fresh
    closure here on every call forced a full re-trace each generation — defeating
    the point of compiling. We memoize on id(model) so the trace is paid once and
    reused. (id() never reaped; fine for the few long-lived models we load.)
    """
    cached = _COMPILED_STEPS.get(id(model))
    if cached is not None:
        return cached

    def fn(token, offset, k_cache, v_cache):
        return _decode_forward(model, token, offset, k_cache, v_cache)

    step = mx.compile(fn, shapeless=True)
    _COMPILED_STEPS[id(model)] = step
    return step


def compiled_generate_step(model, prompt_ids, cfg: GenConfig, eos_ids) -> Iterator[int]:
    """Greedy decode with a compiled per-step forward. Prefill is eager."""
    cache = make_cache(len(model.layers))
    logits = model(mx.array(prompt_ids)[None], cache=cache)[:, -1, :]   # eager prefill
    y = mx.argmax(logits, axis=-1)

    # snapshot the prompt KV as plain fp arrays for the functional loop
    k_cache = [c.keys[..., : c.offset, :] for c in cache]
    v_cache = [c.values[..., : c.offset, :] for c in cache]
    offset = cache[0].offset
    step = make_compiled_step(model)

    n = 0
    while True:
        tok = int(y.item())
        yield tok
        n += 1
        if tok in eos_ids or n >= cfg.max_tokens:
            break
        logits, k_cache, v_cache = step(
            y.reshape(1, 1), mx.array(offset, dtype=mx.int32), k_cache, v_cache
        )
        offset += 1
        y = mx.argmax(logits, axis=-1)
        mx.async_eval(y)
