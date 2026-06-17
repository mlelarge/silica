"""Contention-robust % of usable bandwidth (interleaved ceiling/decode).

Answers the M2 go/no-go's remaining question — is the decode bandwidth headroom a
small-model artifact? Run on 0.6B and on 4B and compare % usable.

% usable is the metric that needs a clean machine (achieved_BW / ceiling), but we
make it robust to background/desktop jitter on the shared memory bus by BRACKETING
each decode sample with a bandwidth-ceiling burst before AND after, then taking the
median per-sample ratio. Drift across the decode window cancels — so we get a
trustworthy % usable without a perfectly idle machine, the same trick that rescued
the compile ratio. (Caveat: a pure-GPU ceiling has no CPU-dispatch component, so
under heavy CPU contention the ratio can still slightly UNDER-read for a small,
dispatch-heavy model; it is tight for larger, GPU-bound models — exactly the 4B
regime this is for.)
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from silica.weights import load_model, resolve_model_path
from silica.config import QuantConfig
from silica.cache import make_cache
from bench.roofline import byte_budget
from bench.baseline import _selective_pred


def _ceiling(buf, nbytes, iters=20) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(mx.sum(buf))
    return iters * nbytes / (time.perf_counter() - t0) / 1e9


def _decode(model, cache_factory, prompt_ids, n_tokens) -> float:
    cache = cache_factory()

    def step(t):
        return mx.argmax(model(t, cache=cache)[:, -1, :], axis=-1)

    y = step(mx.array(prompt_ids)[None])
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        y = step(y.reshape(1, 1))
        mx.async_eval(y)
    mx.eval(y)
    return n_tokens / (time.perf_counter() - t0)


def measure(model, cfg, bits, cache_factory, prompt_ids, n_tokens, buf, nbytes, warmup, runs):
    total = byte_budget(cfg, len(prompt_ids) + n_tokens // 2, bits=bits).total
    for _ in range(warmup):
        _ceiling(buf, nbytes)
        _decode(model, cache_factory, prompt_ids, n_tokens)
    rates, pcts, ceils = [], [], []
    for _ in range(runs):
        cb = _ceiling(buf, nbytes)                       # bracket: ceiling before
        ts = _decode(model, cache_factory, prompt_ids, n_tokens)
        ca = _ceiling(buf, nbytes)                       # ceiling after
        ceil = 0.5 * (cb + ca)
        rates.append(ts)
        pcts.append(100 * (total * ts / 1e9) / ceil)
        ceils.append(ceil)
    return statistics.median(rates), statistics.median(pcts), statistics.median(ceils)


def main():
    ap = argparse.ArgumentParser(description="contention-robust roofline % usable")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--with-mlx-lm", action="store_true", help="also measure mlx-lm (parity check)")
    ap.add_argument("--spec-bandwidth", type=float, default=400.0)
    args = ap.parse_args()

    nbytes = 1_500_000_000
    buf = mx.ones((nbytes // 2,), dtype=mx.float16)
    mx.eval(buf)
    path = resolve_model_path(args.model)
    prompt_ids = list(range(1, 9))

    mlx_lm = make_prompt_cache = None
    if args.with_mlx_lm:
        import mlx_lm
        try:
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError:
            from mlx_lm.cache import make_prompt_cache

    # A reading is reliable only if the bracketing ceiling stayed near the quiet
    # value (bus not heavily contended) and the ratio is physically possible.
    ceiling_floor = 0.80 * 0.90 * args.spec_bandwidth

    def flag(pct, ceil):
        if pct > 98:
            return "  UNRELIABLE (>98%: ceiling contaminated)"
        if ceil < ceiling_floor:
            return f"  UNRELIABLE (ceiling {ceil:.0f}<{ceiling_floor:.0f}: bus contended)"
        return ""

    print(f"model: {args.model}   (% usable = median of bracketed achieved/ceiling)\n")
    print(f"{'engine':<9}{'config':<7}{'tok/s':>9}{'ceiling GB/s':>14}{'% usable':>11}")
    print("-" * 50)
    for name, bits, quant in [("fp16", None, None),
                              ("4-bit", 4, QuantConfig(bits=4, group_size=64, embed_bits=6))]:
        smodel, cfg = load_model(path, quant=quant, dtype=mx.bfloat16)
        ts, pct, ceil = measure(smodel, cfg, bits, lambda m=smodel: make_cache(len(m.layers)),
                                prompt_ids, args.tokens, buf, nbytes, args.warmup, args.runs)
        print(f"{'silica':<9}{name:<7}{ts:>9.1f}{ceil:>14.0f}{pct:>10.0f}%{flag(pct, ceil)}")
        del smodel

        if args.with_mlx_lm:
            rmodel, _ = mlx_lm.load(str(path))
            if bits:
                nn.quantize(rmodel, group_size=64, bits=4, class_predicate=_selective_pred)
                mx.eval(rmodel.parameters())
            ts2, pct2, _ = measure(rmodel, cfg, bits, lambda m=rmodel: make_prompt_cache(m),
                                   prompt_ids, args.tokens, buf, nbytes, args.warmup, args.runs)
            print(f"{'mlx-lm':<9}{name:<7}{ts2:>9.1f}{'':>14}{pct2:>10.0f}%{flag(pct2, ceil)}")
            del rmodel


if __name__ == "__main__":
    main()
