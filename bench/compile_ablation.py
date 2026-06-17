"""M2: mx.compile on/off decode ablation (run on a QUIET machine).

Compares eager (async_eval) vs compiled-decode tok/s and % of usable bandwidth,
for fp16 and 4-bit. Decides whether mx.compile clears the ~69% mlx-lm baseline
(GO for M3) or not (pivot). Both paths use greedy + async_eval; the only
difference is the compiled per-step forward, isolating mx.compile's effect.

WARNING: this is a bandwidth/launch-overhead measurement. On Apple Silicon the
CPU and GPU share memory bandwidth, so other busy processes contaminate it. The
script aborts-with-warning if the 1-min load average looks high.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import mlx.core as mx

from silica.weights import load_model, resolve_model_path
from silica.config import QuantConfig
from silica.cache import make_cache
from silica.compiled import make_compiled_step
from bench.roofline import byte_budget
from bench.baseline import measure_peak_bandwidth


def _time_eager(model, prompt_ids, n_tokens, warmup, runs) -> float:
    def once():
        cache = make_cache(len(model.layers))

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

    rates = [once() for _ in range(warmup + runs)]
    return statistics.median(rates[warmup:])


def _time_compiled(model, prompt_ids, n_tokens, warmup, runs) -> float:
    step = make_compiled_step(model)

    def once():
        cache = make_cache(len(model.layers))
        logits = model(mx.array(prompt_ids)[None], cache=cache)[:, -1, :]
        y = mx.argmax(logits, axis=-1)
        k = [c.keys[..., : c.offset, :] for c in cache]
        v = [c.values[..., : c.offset, :] for c in cache]
        off = cache[0].offset
        mx.eval(y, *k, *v)
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            logits, k, v = step(y.reshape(1, 1), mx.array(off, dtype=mx.int32), k, v)
            off += 1
            y = mx.argmax(logits, axis=-1)
            mx.async_eval(y)
        mx.eval(y)
        return n_tokens / (time.perf_counter() - t0)

    rates = [once() for _ in range(warmup + runs)]   # warmup absorbs JIT compile
    return statistics.median(rates[warmup:])


def main():
    ap = argparse.ArgumentParser(description="mx.compile on/off decode ablation")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--spec-bandwidth", type=float, default=400.0)
    ap.add_argument("--force", action="store_true", help="run even if the machine looks busy")
    args = ap.parse_args()

    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 1
    if load1 > 0.5 * ncpu and not args.force:
        raise SystemExit(
            f"load average {load1:.1f} on {ncpu} cpus looks high — bandwidth numbers "
            f"would be contaminated (shared CPU/GPU memory bus). Re-run when quiet, "
            f"or pass --force for throwaway numbers."
        )

    path = resolve_model_path(args.model)
    prompt_ids = list(range(1, 9))
    eff_ctx = len(prompt_ids) + args.tokens // 2
    usable = measure_peak_bandwidth()
    print(f"usable BW (meas.): {usable:.0f} GB/s  (load1={load1:.1f})\n")

    print(f"{'config':<20}{'tok/s':>9}{'GB/s':>9}{'% usable':>10}")
    print("-" * 48)
    for name, bits, quant in [("fp16", None, None),
                              ("4-bit", 4, QuantConfig(bits=4, group_size=64, embed_bits=6))]:
        model, cfg = load_model(path, quant=quant, dtype=mx.bfloat16)
        total = byte_budget(cfg, eff_ctx, bits=bits).total
        for tag, fn in [("eager", _time_eager), ("compiled", _time_compiled)]:
            ts = fn(model, prompt_ids, args.tokens, args.warmup, args.runs)
            bw = total * ts / 1e9
            print(f"{name+' '+tag:<20}{ts:>9.1f}{bw:>9.0f}{100*bw/usable:>9.0f}%")
        del model


if __name__ == "__main__":
    main()
