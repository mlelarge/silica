"""Decode benchmark: tok/s + achieved-bandwidth % (PLAN §6).

Reports tok/s AND bandwidth-% side by side (quantization raises tok/s while
bytes*tok/s may stay flat — you need both to read an ablation). Includes the
rigor the audit asked for: discarded warmup, median + IQR over K runs, and a
REQUIRED device-bandwidth denominator tied to the exact chip SKU.

Caveat printed at runtime: on a laptop M3 Max sustained decode throttles, so
steady-state differs from burst; record thermal state alongside results.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from silica.config import BenchConfig, GenConfig
from silica.weights import load_model
from silica.generate import generate_step
from bench.roofline import byte_budget


def time_decode(model, prompt_ids, n_tokens: int, kv_bits=None) -> float:
    """Return seconds to decode `n_tokens` tokens (prefill excluded from rate)."""
    cfg = GenConfig(max_tokens=n_tokens, temperature=0.0, kv_bits=kv_bits)
    eos = tuple()  # ignore EOS so we always measure exactly n_tokens
    it = generate_step(model, prompt_ids, cfg, eos)
    first = next(it)  # consume prefill + first token
    mx.eval(mx.array(first))
    t0 = time.perf_counter()
    count = 0
    for _ in it:
        count += 1
    mx.synchronize() if hasattr(mx, "synchronize") else None
    return (time.perf_counter() - t0), count


def run(model_id: str, bcfg: BenchConfig, n_tokens: int, context_len: int,
        bits: int | None, kv_bits: int | None = None):
    model, cfg = load_model(model_id, dtype=mx.bfloat16)
    # Decode at the requested context: prefill `context_len` synthetic tokens so
    # the timed steps actually read that much KV. Charging KV bytes at a context
    # the run never reached is what let achieved-BW exceed the chip's peak.
    prompt_len = max(8, context_len)
    prompt_ids = list(range(1, prompt_len + 1))
    eff_ctx = prompt_len + n_tokens // 2  # mean KV depth over the timed window

    for _ in range(bcfg.warmup):
        time_decode(model, prompt_ids, min(n_tokens, 16), kv_bits=kv_bits)

    rates = []
    for _ in range(bcfg.runs):
        dt, count = time_decode(model, prompt_ids, n_tokens, kv_bits=kv_bits)
        if count > 0:
            rates.append(count / dt)

    med = statistics.median(rates)
    iqr = (statistics.quantiles(rates, n=4)[2] - statistics.quantiles(rates, n=4)[0]
           ) if len(rates) >= 4 else 0.0

    budget = byte_budget(cfg, eff_ctx, bits=bits, kv_bits=kv_bits)
    print(f"chip            : {bcfg.chip_name}")
    print(f"model           : {model_id}  (w={bits or 'fp16'}, kv={kv_bits or 'fp16'})")
    print(f"context (decode): prefill {prompt_len} + {n_tokens} dec -> mean {eff_ctx} tok")
    print(f"tok/s (median)  : {med:.1f}  (IQR {iqr:.1f}, n={len(rates)})")
    print(f"bytes/token     : weights={budget.weights/1e6:.1f}MB  "
          f"kv={budget.kv/1e6:.1f}MB  total={budget.total/1e6:.1f}MB")
    print(f"achieved BW     : {budget.achieved_bandwidth_gbps(med):.1f} GB/s")
    if bcfg.device_bandwidth_gbps:
        print(f"% of peak BW    : {budget.pct_of_peak(med, bcfg.device_bandwidth_gbps):.1f}%"
              f"  (peak {bcfg.device_bandwidth_gbps:.0f} GB/s)")
    else:
        print("% of peak BW    : (set --bandwidth to the chip's rated GB/s)")
    print("note            : laptop M3 Max throttles under sustained decode; "
          "record thermal state.")


def main():
    ap = argparse.ArgumentParser(description="silica decode benchmark")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--context-len", type=int, default=0,
                    help="context length to charge KV bytes at")
    ap.add_argument("--bits", type=int, default=None, help="weight quant bits (fp16 if unset)")
    ap.add_argument("--kv-bits", type=int, default=None, help="KV cache quant bits (fp16 if unset)")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="REQUIRED for %% of peak: chip rated GB/s (M3 Max 300 or 400)")
    ap.add_argument("--chip", default="unknown-apple-silicon")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    bcfg = BenchConfig(warmup=args.warmup, runs=args.runs,
                       device_bandwidth_gbps=args.bandwidth, chip_name=args.chip)
    run(args.model, bcfg, args.tokens, args.context_len, args.bits, args.kv_bits)


if __name__ == "__main__":
    main()
