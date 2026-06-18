"""M2: mx.compile on/off decode ablation.

Decides whether mx.compile beats the eager (async_eval) decode that silica and
mlx-lm already share. Both paths use greedy + async_eval; the only difference is
the compiled per-step forward, isolating mx.compile's effect.

Robust-to-contention design: eager and compiled are measured INTERLEAVED (one
eager sample then one compiled sample, repeated), and the headline number is the
median of the per-pair ratio compiled/eager. Because each pair is taken back-to-
back, slow drift in background load cancels in the ratio — so the relative
verdict is trustworthy even when absolute tok/s is depressed. Absolute tok/s and
% usable still need a quiet machine (Apple Silicon shares the CPU/GPU memory bus
and batch=1 decode is partly dispatch-bound), so the script reports a quietness
check and flags when only the ratio should be trusted.
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


def _eager_sample(model, prompt_ids, n_tokens) -> float:
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


def _compiled_sample(model, step, prompt_ids, n_tokens) -> float:
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


def main():
    ap = argparse.ArgumentParser(description="mx.compile on/off decode ablation")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--spec-bandwidth", type=float, default=400.0)
    ap.add_argument("--force", action="store_true",
                    help="run despite contention; ratio stays valid, absolutes don't")
    args = ap.parse_args()

    load1 = os.getloadavg()[0]
    ceilings = [measure_peak_bandwidth() for _ in range(3)]   # >=3 samples, median-based
    usable = statistics.median(ceilings)
    spread = (max(ceilings) - min(ceilings)) / max(ceilings)
    expected = 0.90 * args.spec_bandwidth
    quiet = spread <= 0.06 and usable >= 0.85 * expected and load1 <= 2.0
    print(f"usable BW: {usable:.0f} GB/s (spread {spread*100:.0f}% over "
          f"{len(ceilings)} samples)  load1: {load1:.1f}  "
          f"-> {'QUIET (absolutes trustworthy)' if quiet else 'CONTENDED (trust only the ratio)'}")
    if not quiet and not args.force:
        raise SystemExit("not quiet — re-run idle, or --force (paired ratio is still valid).")
    print()

    path = resolve_model_path(args.model)
    prompt_ids = list(range(1, 9))
    eff_ctx = len(prompt_ids) + args.tokens // 2

    print(f"{'config':<8}{'eager t/s':>11}{'compiled':>10}{'compiled/eager':>16}{'%usable(e/c)':>14}")
    print("-" * 59)
    for name, bits, quant in [("fp16", None, None),
                              ("4-bit", 4, QuantConfig(bits=4, group_size=64, embed_bits=6))]:
        model, cfg = load_model(path, quant=quant, dtype=mx.bfloat16)
        step = make_compiled_step(model)
        for _ in range(args.warmup):                       # warm JIT + pipelines
            _eager_sample(model, prompt_ids, args.tokens)
            _compiled_sample(model, step, prompt_ids, args.tokens)
        pairs = []
        for _ in range(args.runs):                         # INTERLEAVED -> drift cancels
            e = _eager_sample(model, prompt_ids, args.tokens)
            c = _compiled_sample(model, step, prompt_ids, args.tokens)
            pairs.append((e, c))
        em = statistics.median(e for e, _ in pairs)
        cm = statistics.median(c for _, c in pairs)
        ratio = statistics.median(c / e for e, c in pairs)
        total = byte_budget(cfg, eff_ctx, bits=bits).total
        pe, pc = 100 * total * em / 1e9 / usable, 100 * total * cm / 1e9 / usable
        print(f"{name:<8}{em:>11.1f}{cm:>10.1f}{(ratio-1)*100:>+14.1f}%{pe:>7.0f}/{pc:<6.0f}")
        del model


if __name__ == "__main__":
    main()
