"""M2 baseline go/no-go (PLAN §5 M2, §7).

Measures the empirical usable-bandwidth ceiling and compares silica vs mlx-lm
decode at fp16 and 4-bit through ONE identical greedy+async_eval loop, charging
the same architecture-based byte model. The decision the plan gates on:

  * If mlx-lm is already near the usable ceiling (>~75-85%), there is little
    headroom — M3 custom kernels are unlikely to pay off and the project's value
    pivots to pedagogy/measurement.
  * If silica trails mlx-lm, that gap is silica's own to close in M2 (async_eval
    / mx.compile) before any kernel work.

Same-machine, sequential (GPU-bound) — not parallelizable.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from silica.weights import load_model, resolve_model_path
from silica.cache import make_cache as silica_make_cache
from silica.config import QuantConfig
from bench.roofline import byte_budget


def measure_peak_bandwidth(nbytes: int = 1_500_000_000, iters: int = 30) -> float:
    """Empirical GPU read bandwidth (GB/s) via a large reduction."""
    x = mx.ones((nbytes // 2,), dtype=mx.float16)
    mx.eval(x)
    for _ in range(3):
        mx.eval(mx.sum(x))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(mx.sum(x))
    return iters * nbytes / (time.perf_counter() - t0) / 1e9


def _timed_decode(model, cache, prompt_ids, n_tokens: int) -> float:
    """Greedy decode `n_tokens` (prefill excluded); returns tok/s. async_eval on."""
    def step(toks):
        return mx.argmax(model(toks, cache=cache)[:, -1, :], axis=-1)

    y = step(mx.array(prompt_ids)[None])
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        y = step(y.reshape(1, 1))
        mx.async_eval(y)
    mx.eval(y)
    return n_tokens / (time.perf_counter() - t0)


def _median_rate(model, cache_factory, prompt_ids, n_tokens, warmup, runs) -> float:
    rates = []
    for i in range(warmup + runs):
        r = _timed_decode(model, cache_factory(), prompt_ids, n_tokens)
        if i >= warmup:
            rates.append(r)
    return statistics.median(rates)


def _selective_pred(p, m):
    if not hasattr(m, "to_quantized"):
        return False
    if p.endswith("embed_tokens") or p.endswith("lm_head"):
        return {"group_size": 64, "bits": 6}
    w = getattr(m, "weight", None)
    if w is not None and w.shape[-1] % 64 != 0:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="silica vs mlx-lm baseline bandwidth")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--spec-bandwidth", type=float, default=400.0, help="chip rated GB/s")
    args = ap.parse_args()

    import mlx_lm
    try:
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError:
        from mlx_lm.cache import make_prompt_cache

    path = resolve_model_path(args.model)
    prompt_ids = list(range(1, 9))               # short prompt -> weights-bound
    eff_ctx = len(prompt_ids) + args.tokens // 2

    usable = measure_peak_bandwidth()
    print(f"chip spec BW      : {args.spec_bandwidth:.0f} GB/s")
    print(f"usable BW (meas.) : {usable:.0f} GB/s  ({100*usable/args.spec_bandwidth:.0f}% of spec)\n")

    def silica_models():
        fp, cfg = load_model(path, dtype=mx.bfloat16)
        q4, _ = load_model(path, quant=QuantConfig(bits=4, group_size=64, embed_bits=6),
                           dtype=mx.bfloat16)
        return [("silica fp16", fp, None, cfg), ("silica 4-bit", q4, 4, cfg)]

    def mlxlm_models():
        fp, _ = mlx_lm.load(str(path))
        q4, _ = mlx_lm.load(str(path))
        nn.quantize(q4, group_size=64, bits=4, class_predicate=_selective_pred)
        mx.eval(q4.parameters())
        # reuse silica's ModelConfig for the (architecture-identical) byte model
        _, cfg = load_model(path, dtype=mx.bfloat16)
        return [("mlx-lm fp16", fp, None, cfg), ("mlx-lm 4-bit", q4, 4, cfg)]

    rows = []
    for name, model, bits, cfg in silica_models() + mlxlm_models():
        is_silica = name.startswith("silica")
        factory = ((lambda m=model: silica_make_cache(len(m.layers))) if is_silica
                   else (lambda m=model: make_prompt_cache(m)))
        tok_s = _median_rate(model, factory, prompt_ids, args.tokens, args.warmup, args.runs)
        budget = byte_budget(cfg, eff_ctx, bits=bits)
        bw = budget.total * tok_s / 1e9
        rows.append((name, tok_s, bw, 100 * bw / usable, 100 * bw / args.spec_bandwidth))

    print(f"{'config':<14}{'tok/s':>9}{'GB/s':>9}{'% usable':>10}{'% spec':>9}")
    print("-" * 51)
    for name, tok_s, bw, pu, ps in rows:
        print(f"{name:<14}{tok_s:>9.1f}{bw:>9.0f}{pu:>9.0f}%{ps:>8.0f}%")


if __name__ == "__main__":
    main()
