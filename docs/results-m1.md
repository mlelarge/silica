# M1 results — weight quantization quality

Model: **Qwen3-0.6B**. Corpus: `bench/data/corpus.txt` (378 scored tokens —
small, so absolute PPL is inflated/noisy; the *relative* deltas and the
cross-check vs mlx-lm are the trustworthy parts). Compute dtype bf16.
Reproduce with `uv run silica-ppl --ablate`.

| config | PPL | Δ vs fp16 |
|---|---|---|
| fp16 | 24.98 | — |
| 8-bit / g64 (selective) | 24.82 | **−0.7%** |
| 4-bit / g64 (selective, embed 6-bit) | 35.94 | **+43.9%** |
| 4-bit / **g32** (selective) | 29.89 | **+19.7%** |
| 4-bit / g64, embed also 4-bit | 35.96 | +44.0% |
| **mlx-lm** 4-bit / g64 (selective) | 35.94 | +43.9% |

## Takeaways

1. **silica's quantized forward is correct.** silica 4-bit/g64 PPL (35.940) is
   identical to three decimals to `mlx-lm`'s under the same recipe — the M0
   parity story extends to quantized weights. The +44% is inherent to 4-bit on
   a 0.6B model, not a silica defect.
2. **`group_size` is the dominant quality lever**, not embed precision: g32
   roughly halves the regression (+44% → +20%) for ~2× the scale/bias overhead
   (~25% vs 12.5% of packed bytes). Keeping the tied embed/lm_head at 6-bit
   barely moves PPL here (35.940 vs 35.962) — cheap insurance, not the driver.
3. **Recommended operating points for 0.6B:** 8-bit (essentially lossless) or
   4-bit/g32 (~20%); 4-bit/g64 is too lossy to recommend on a model this small.
   Larger models tolerate 4-bit/g64 far better — revisit per model size.

## Caveats / next

- Tiny corpus → noisy absolute PPL. Swap in a larger pinned corpus (or a
  WikiText slice) before treating absolute numbers as anything but directional.
- Pair this with the M2 tok/s + achieved-bandwidth-% table so the quality cost
  sits next to the speed benefit (the byte model already accounts for the
  scale/bias overhead that g32 doubles).
