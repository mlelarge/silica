"""Cross-model bandwidth-utilization scaling (contention-robust).

Answers the M2 go/no-go's last question: does decode use more of the memory bus
on a bigger model? If yes, the 0.6B headroom is a small-model artifact (small
GEMVs under-utilize bandwidth) -> custom kernels (M3) won't help -> go to M4.

Measures two models' decode achieved bandwidth INTERLEAVED (one sample each,
back-to-back, repeated) and reports the median per-pair ratio. Because each pair
is back-to-back under the same bus conditions, the ratio cancels contention — so
this works WITHOUT a quiet machine (unlike an absolute % usable). Anchored on the
known-clean small-model % usable, it yields the big-model % usable:

    %usable_large  =  (achieved_large / achieved_small)  x  %usable_small_clean

Use fp16 by default: both models are then GPU/bandwidth-bound, minimizing the
CPU-dispatch confound that depresses a small 4-bit model's fast steps.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from silica.weights import load_model, resolve_model_path
from silica.config import QuantConfig
from silica.cache import make_cache
from bench.roofline import byte_budget


def _decode(model, prompt_ids, n_tokens) -> float:
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


def main():
    ap = argparse.ArgumentParser(description="cross-model bandwidth scaling (robust)")
    ap.add_argument("--small", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--large", default="Qwen/Qwen3-4B")
    ap.add_argument("--bits", type=int, default=None, help="weight bits (fp16 if unset)")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--anchor-usable", type=float, default=72.0,
                    help="known-CLEAN %% usable of the small model at this config "
                         "(0.6B: fp16 72, 4-bit 67)")
    args = ap.parse_args()

    quant = None if args.bits is None else QuantConfig(bits=args.bits, group_size=64, embed_bits=6)
    sm, scfg = load_model(resolve_model_path(args.small), quant=quant, dtype=mx.bfloat16)
    lg, lcfg = load_model(resolve_model_path(args.large), quant=quant, dtype=mx.bfloat16)
    ctx = 40
    sb = byte_budget(scfg, ctx, bits=args.bits).total
    lb = byte_budget(lcfg, ctx, bits=args.bits).total
    prompt = list(range(1, 9))

    for _ in range(args.warmup):
        _decode(sm, prompt, args.tokens)
        _decode(lg, prompt, args.tokens)

    ratios, s_bw, l_bw = [], [], []
    for _ in range(args.runs):                            # interleaved -> contention cancels
        ts_s = _decode(sm, prompt, args.tokens)
        ts_l = _decode(lg, prompt, args.tokens)
        a_s, a_l = sb * ts_s / 1e9, lb * ts_l / 1e9
        ratios.append(a_l / a_s)
        s_bw.append(a_s)
        l_bw.append(a_l)

    r = statistics.median(ratios)
    cfg_name = "fp16" if args.bits is None else f"{args.bits}-bit"
    print(f"config: {cfg_name}   (achieved BW = bytes/token x tok/s; ratio is back-to-back robust)\n")
    if args.bits is not None:
        print("CAVEAT: a quantized small model has very fast (dispatch-bound) steps, so "
              "external CPU load depresses it more than the large model -> this ratio is\n"
              "        INFLATED under contention. The fp16 ratio is the clean read.\n")
    print(f"small {args.small}:  {statistics.median(s_bw):.0f} GB/s achieved")
    print(f"large {args.large}:  {statistics.median(l_bw):.0f} GB/s achieved")
    print(f"\nlarge/small achieved-BW ratio (robust): {r:.2f}x")
    print(f"-> large % usable ~= {r:.2f} x {args.anchor_usable:.0f}% (small clean) = "
          f"{r * args.anchor_usable:.0f}% usable")
    if r > 1.15:
        print("\nVERDICT: big model uses the bus markedly better -> 0.6B headroom is a "
              "small-model artifact -> skip M3, go to M4.")
    elif r < 1.05:
        print("\nVERDICT: no improvement at scale -> headroom is real -> reconsider M3.")
    else:
        print("\nVERDICT: modest improvement -> leans skip-M3; confirm on a quiet machine.")


if __name__ == "__main__":
    main()
