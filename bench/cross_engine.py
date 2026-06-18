"""Cross-engine comparison: silica (MLX) vs llama.cpp (Metal), same model.

The M4 fairness baseline against a non-MLX engine. Uses the SAME Qwen3-0.6B and
matched quantization (llama.cpp Q8_0 ~= silica 8-bit/g64 ~8.5 bpw; Q4_K_M ~=
silica 4-bit/g64 ~4.5 bpw), so the achieved-bandwidth byte model is identical for
both engines and only decode tok/s differs.

Both engines are bandwidth-bound at batch=1, so the trustworthy output under load
is the **tok/s ratio** (silica / llama.cpp) measured close in time — it cancels
shared memory-bus contention; absolute % usable is flagged when the bracketing
ceiling shows the bus is contended.

Fairness caveats (documented, not hidden): llama.cpp Q4_K_M is a k-quant (block
scales+mins) vs MLX affine group-64 — close effective bpw, not identical schemes;
both run greedy, full Metal offload, default threads; quality (PPL) is not
compared here, only speed/bandwidth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time

import mlx.core as mx

from silica.weights import load_model, resolve_model_path
from silica.config import QuantConfig
from silica.cache import make_cache
from bench.roofline import byte_budget
from bench.baseline import measure_peak_bandwidth


def llama_tg_tokps(gguf: str, n_tokens: int, reps: int) -> float:
    """Decode (text-generation) tok/s from llama-bench (its own warmup/reps)."""
    out = subprocess.run(
        ["llama-bench", "-m", gguf, "-p", "0", "-n", str(n_tokens), "-r", str(reps), "-o", "json"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"llama-bench failed: {out.stderr[-300:]}")
    data = json.loads(out.stdout)
    tg = [float(r["avg_ts"]) for r in data if int(r.get("n_gen", 0)) > 0]
    return statistics.median(tg)


def silica_tg_tokps(model, prompt_ids, n_tokens, warmup, runs) -> float:
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

    for _ in range(warmup):
        once()
    return statistics.median(once() for _ in range(runs))


def main():
    ap = argparse.ArgumentParser(description="silica (MLX) vs llama.cpp (Metal)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="silica HF model id/path")
    ap.add_argument("--q8-gguf", required=True)
    ap.add_argument("--q4-gguf", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--llama-reps", type=int, default=4)
    ap.add_argument("--spec-bandwidth", type=float, default=400.0)
    args = ap.parse_args()

    path = resolve_model_path(args.model)
    prompt_ids = list(range(1, 9))
    eff_ctx = len(prompt_ids) + args.tokens // 2
    ceiling_floor = 0.85 * 0.90 * args.spec_bandwidth

    print(f"silica (MLX) vs llama.cpp (Metal) — {args.model}, decode {args.tokens} tok\n")
    header = f"{'config':<8}{'engine':<11}{'tok/s':>9}{'GB/s':>8}{'%usable':>9}"
    print(header + f"{'  silica/llama':>15}")
    print("-" * (len(header) + 15))

    for name, bits, gguf in [("8-bit", 8, args.q8_gguf), ("4-bit", 4, args.q4_gguf)]:
        model, cfg = load_model(path, quant=QuantConfig(bits=bits, group_size=64, embed_bits=6),
                                dtype=mx.bfloat16)
        total = byte_budget(cfg, eff_ctx, bits=bits).total

        c_before = measure_peak_bandwidth()
        lt = llama_tg_tokps(gguf, args.tokens, args.llama_reps)
        st = silica_tg_tokps(model, prompt_ids, args.tokens, args.warmup, args.runs)
        c_after = measure_peak_bandwidth()
        ceil = 0.5 * (c_before + c_after)
        contended = ceil < ceiling_floor

        la, sa = total * lt / 1e9, total * st / 1e9
        ratio = st / lt
        pu_s = f"{100*sa/ceil:.0f}%" + ("*" if contended else "")
        pu_l = f"{100*la/ceil:.0f}%" + ("*" if contended else "")
        print(f"{name:<8}{'silica':<11}{st:>9.1f}{sa:>8.0f}{pu_s:>9}{ratio:>14.2f}x")
        print(f"{name:<8}{'llama.cpp':<11}{lt:>9.1f}{la:>8.0f}{pu_l:>9}")
        del model

    print(f"\nceiling bracket avg per config; * = bus contended (% usable unreliable, "
          f"ratio still valid). spec {args.spec_bandwidth:.0f} GB/s.")


if __name__ == "__main__":
    main()
