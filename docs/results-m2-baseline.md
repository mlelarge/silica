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

## Next M2 step
`mx.compile` the fixed-shape decode step (cache via `inputs=`/`outputs=`),
prefill kept eager; ablate compile on/off and re-measure % usable. Then decide
M3 on whether compile clears the mlx-lm baseline. Also worth a larger model
(4B/8B) measurement — bigger GEMVs should sit closer to the ceiling, shrinking
the apparent headroom.
