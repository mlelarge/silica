# M4 cross-engine — silica (MLX) vs llama.cpp (Metal)

The non-MLX baseline the report previously listed as a gap. Same model, matched
quantization, same byte model — only the engine differs.

## Setup

- **Model:** Qwen3-0.6B (the exact model silica runs).
- **silica:** selective `mx.quantize` 8-bit and 4-bit, group 64, bf16 compute.
- **llama.cpp:** build 9270 (Homebrew, Metal), `llama-bench`, full GPU offload
  (`ngl=99`), GGUF **Q8_0** (639 MB) and **Q4_K_M** (397 MB).
- **Matched bpw → one byte model:** Q8_0 ≈ 8.5 bpw ≈ silica 8-bit/g64; Q4_K_M ≈
  4.5 bpw ≈ silica 4-bit/g64. So achieved bandwidth uses the *same* `byte_budget`
  for both engines and only decode tok/s differs.
- **Metric:** both engines are bandwidth-bound at batch=1; the trustworthy number
  under load is the **silica/llama tok/s ratio** measured close in time (cancels
  shared memory-bus contention). Reproduce: `bench/cross_engine.py`.

## Results

Two runs, deliberately under heavy background load (the test machine was busy);
absolute tok/s and % usable are contaminated (`*`), but the **ratio is the signal**:

| run | load | config | silica tok/s | llama tok/s | **silica/llama** |
|---|---|---|---|---|---|
| 1 | ~19 | 8-bit | 67.5 | 74.9 | **0.90×** |
| 1 | ~19 | 4-bit | 56.8 | 64.3 | **0.88×** |
| 2 | ~49 | 8-bit | 76.0 | 85.4 | **0.89×** |
| 2 | ~49 | 4-bit | 82.9 | 93.0 | **0.89×** |

**The ratio is stable at 0.88–0.90× across a 2.5× swing in load (19→49) and both
quant levels.** Absolute tok/s move wildly with load (silica 8-bit 67.5 → 76.0);
the ratio does not — strong evidence it is a real engine property, not a
contention artifact. (If Python dispatch overhead were being differentially
amplified by load, the ratio would shift between load 19 and 49; it doesn't.)

## Finding

**silica ≈ 0.89× llama.cpp decode speed — llama.cpp is ~12% faster** on the same
model. For a transparent ~1000-line Python/MLX engine versus mature, hand-tuned
C++/Metal, an ~12% gap is a strong result, and it is consistent with silica ==
mlx-lm (M2): the entire MLX Python stack sits ~12% behind llama.cpp here, most
plausibly the residual per-token Python/dispatch overhead that a compiled C++
generation loop avoids — not a kernel-efficiency gap (silica and llama both ride
near the same ~70% of usable bandwidth once normalized).

## Caveats (fairness, stated not hidden)

- **Speed only, not quality.** llama.cpp's **Q4_K_M is a k-quant** (per-block
  scales + mins) and is generally *higher quality* than silica's naive affine
  4-bit/g64 (which showed +44% PPL in M1). So at 4-bit llama.cpp is both slightly
  faster **and** likely higher quality — silica should adopt a better 4-bit recipe
  (k-quant-style, or g32 which already halves the PPL hit) before claiming 4-bit
  parity. The 8-bit comparison is the cleaner like-for-like.
- **Absolutes are contention-limited** (load 19–49). The % usable figures are
  flagged unreliable; only the ratio is reported as a result. A clean-machine
  re-run would firm the absolutes but the ratio is already stable.
- **Default llama.cpp config** (its out-of-box Metal threads/batch); no attempt to
  tune either engine beyond defaults.
