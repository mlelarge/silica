# M2 baseline — go/no-go on perf work

Model **Qwen3-0.6B**, **M3 Max 40-core (400 GB/s spec)**, bf16 compute, 8-token
prompt + 128 greedy decode. Reproduce: `uv run python -m bench.baseline`.

- **Usable bandwidth ceiling (measured):** ~370 GB/s = **93% of the 400 GB/s spec**.

## silica vs mlx-lm (async_eval ON for both — the realistic config)

| config | tok/s | achieved GB/s | % of usable |
|---|---|---|---|
| silica fp16 | 222 | 267 | 72% |
| mlx-lm fp16 | 226 | 271 | 73% |
| silica 4-bit | 437 | 248 | 67% |
| mlx-lm 4-bit | 442 | 251 | 68% |

## async_eval ablation (silica)

| config | async OFF | async ON | gain |
|---|---|---|---|
| fp16 | 190 tok/s (62% usable) | 226 (73%) | +19% |
| 4-bit | 299 tok/s (46% usable) | 449 (69%) | **+50%** |

## Verdict: **GO on M2, with eyes open**

1. **silica is at parity with mlx-lm** (within ~1.5% at both fp16 and 4-bit) —
   the implementation is competitive; there is no silica-specific perf bug and
   nothing left on the table versus the reference.
2. **There is real headroom: ~69% of usable, not >85%.** Decode does not
   saturate bandwidth at batch=1 on this small model. The gap to the 370 GB/s
   ceiling is per-step launch overhead + small-GEMV inefficiency (the audit's
   perf-3), which is what `mx.compile` and M3 fusion target.
3. **async_eval is a big lever and is already captured** (+50% at 4-bit). The
   remaining 69%→93% gap is NOT recoverable by async_eval — both silica and
   mlx-lm already run it.
4. **The unexplored lever is `mx.compile`.** mlx-lm does NOT compile its decode
   loop (verified). If silica's compiled per-step forward pushes achieved BW
   above the ~69% mlx-lm baseline, that is a genuine, reference-beating M2 win
   and M3 fusion is justified. If it doesn't, the gap is fundamental small-model
   overhead → pivot toward pedagogy + larger-model roofline.

Quantization remains the dominant throughput lever (fp16→4-bit: +98% tok/s),
ahead of async_eval (+50%) — consistent with PLAN §1.

## mx.compile ablation (result)

Compiled decode step (`silica/compiled.py`): functional cache + traced array
RoPE offset + `shapeless=True`, prefill eager, greedy. Correct (compiled tokens
== eager, fp16 & 4-bit). Measured with INTERLEAVED eager/compiled pairs so the
per-pair ratio is robust to background load (absolute tok/s were depressed by a
persistent ~load-3 job, but the ratio is drift-cancelled):

| config | compiled/eager (2 runs) |
|---|---|
| fp16 | −1.3%, −1.1% → ~neutral (slightly negative) |
| 4-bit | +1.1%, +3.3% → ~+2% (marginal) |

**`mx.compile` does NOT clear the baseline** — it is within ±3% of eager, far
short of the ~31-point gap to the 370 GB/s usable ceiling. Mechanism: with
`async_eval` already on, per-step graph-build/dispatch is already hidden, so the
remaining gap is **small-matrix GEMV inefficiency** (a 0.6B model's GEMVs are too
small to saturate 370 GB/s), which compile cannot fix — it fuses launches, it
doesn't make small GEMVs bandwidth-efficient.

## Scaling: is the headroom a small-model artifact?

Measured robustly by comparing 0.6B vs 4B decode achieved BW **back-to-back**
(`bench/scaling.py`) so contention cancels in the ratio, anchored on 0.6B's
known-clean % usable. (Earlier *absolute* 4B roofline runs all came back
`UNRELIABLE` — the machine was never quiet; this ratio method sidesteps that.)

| config | 4B/0.6B achieved-BW ratio | implied 4B % usable |
|---|---|---|
| fp16 (clean read) | **0.95×** | ~69% (vs 0.6B 72%) |
| 4-bit (confounded) | 1.25× | ~84% — **inflated** (see caveat) |

The fp16 ratio is the trustworthy one: both models are GPU/bandwidth-bound, so
the dispatch confound is minimal. It says **bandwidth utilization is flat across
scale (~70%)** — the ~30% gap to the usable ceiling is **real and NOT a
small-model GEMV artifact** (my earlier hypothesis was wrong). The 4-bit 1.25×
is inflated because a quantized 0.6B's very fast (dispatch-bound) steps are
depressed more by external CPU load, deflating the denominator.

## Decision (synthesis)

The ~30% gap is **real and scale-independent** (fp16), and it is **not** the kind
M3 typically captures: it is not launch overhead (`mx.compile` neutral) and not
small-GEMV (flat at 4B). What's left is the intrinsic efficiency of real
quantized-GEMV/attention access patterns vs an idealized sequential read — which
Apple's already-tuned `mx.quantized_matmul` largely defines, and which a
from-scratch silica kernel is unlikely to beat (audit risk #1).

**Recommendation: lean M4 (write-up), not M3.** The dominant available lever
(`async_eval`) is captured, `mx.compile` adds nothing, and the residual gap is
structural and scale-independent. Before fully closing M3, the disciplined check
the plan already gates on is a **decode profile** (where does the per-step time
go — MLP GEMV / attention / norms / dispatch?) to confirm there is no fusible hot
spot. That profile, not more roofline, is the M3-vs-M4 decider.
