# silica — Performance & Correctness Report

*The honest scoreboard.* silica is a transparent, single-stream MLX LLM inference engine for Apple Silicon. This report records what was measured, how, and what it means — using only established results, with no projected numbers.

**Hardware under test.** Apple M3 Max, 40-core GPU, 400 GB/s spec memory bandwidth, 128 GB unified RAM. Empirical *usable* bandwidth ceiling ≈ **370 GB/s** (93% of spec). Apple Silicon shares **one** memory bus between CPU and GPU, which matters for both the figure of merit and the measurement methodology below.

---

## 1. Thesis & figure of merit

At **batch = 1**, autoregressive decode is **memory-bandwidth-bound**: each token streams the model weights (plus the KV cache and lm_head) through the GPU once, and arithmetic intensity is too low to hide that traffic. Raw `tok/s` therefore conflates the algorithm with the chip and the context length, and is not comparable across models or quant levels.

The figure of merit is **achieved bandwidth as a fraction of the usable ceiling** (~370 GB/s), where achieved bandwidth counts:

- **weights** — including per-group **scales + biases** (≈ 4.5 effective bits/weight at 4-bit / group-64),
- the **KV cache** for the *actual* context length,
- the **lm_head**,
- **but not** the input embedding as a full read — it is a one-row **gather**.

This makes "% of usable bandwidth" the scoreboard, and `tok/s` only a derived quantity. Crucially, batch=1 decode is *partly CPU-dispatch-bound* on a shared bus, so a healthy GPU bandwidth ceiling alone does **not** certify a decode number — you need a stable bus **and** an idle CPU (see §6).

---

## 2. M0 — Correctness

silica reimplements Qwen3-0.6B decode on `mx.fast.*` kernels. Correctness is gated by a **7/7-green parity gate**:

| # | Check | Oracle | What it proves |
|---|-------|--------|----------------|
| 1 | argmax-logits match | mlx-lm | same next-token pick |
| 2 | greedy-sequence match | mlx-lm | stable over generation |
| 3 | multibyte streaming decode | mlx-lm | detokenizer correctness |
| 4–7 | HF fp32 CPU oracle: teacher-forced per-position argmax **and** next-token logit **cosine > 0.999** | transformers / torch | **non-circular** numerical parity |

**Why two oracles.** mlx-lm shares the same `mx.fast` kernels as silica, so it is only a **same-backend** check. The HuggingFace **fp32 CPU** path (transformers / torch) is the **independent, non-circular oracle** — it certifies the math rather than the kernel wiring.

**Qwen3 specifics parity depends on** (getting any of these wrong breaks the oracle):

- per-head **QK-RMSNorm** (`q_norm` / `k_norm` over `head_dim`, applied **before** RoPE),
- **no QKV bias**,
- `head_dim = 128`, **decoupled** from hidden size and head count,
- **tied** `lm_head`,
- RoPE **theta = 1e6**.

---

## 3. M1 — Quantization quality

Quantization is **selective `mx.quantize` at load**: the body at 4-bit, the tied embedding / lm_head kept at 6-bit.

**Perplexity vs. bits** (Qwen3-0.6B, pinned corpus):

| Config | PPL change vs. fp |
|--------|-------------------|
| 8-bit | ≈ lossless (**−0.7%**) |
| 4-bit, group-64 | **+43.9%** |
| 4-bit, group-32 | **+19.7%** |

**Is the +44% a silica bug? No.** silica's 4-bit PPL matches mlx-lm **exactly** (**35.940**). The quantized forward is therefore correct; the +44% is **inherent to 4-bit on a 0.6B model**, not a silica defect. Group-32 (+19.7%) shows the expected quality recovery from finer grouping.

**Quantized KV cache.** Implemented via a separate **`quantized_matmul` ×2 + softmax** SDPA path, because `mx.fast` SDPA accepts only floating-point KV. `quantized_kv_start` keeps the **prefix fp** and quantizes only the tail. At 8-bit KV this is **~break-even on decode speed while halving KV bytes**.

---

## 4. M2 — Performance

**Usable ceiling.** ≈ **370 GB/s** (93% of the 400 GB/s spec).

**silica == mlx-lm.** silica is within **~1.5%** of mlx-lm at both precisions (`async_eval` ON for both):

| Precision | silica % of usable | vs. mlx-lm |
|-----------|--------------------|------------|
| fp16 | **~72%** | within ~1.5% |
| 4-bit | **~67–68%** | within ~1.5% |

**Ablation — what actually moves the needle:**

| Lever | Effect | Read |
|-------|--------|------|
| **Quantization** (fp16 → 4-bit) | **+98% tok/s** | dominant throughput lever |
| **`async_eval`** | **+50% tok/s** at 4-bit (299 → 449) | big lever; ON for all M2 numbers |
| **`mx.compile`** of the decode step | **neutral** (fp16 ~−1%, 4-bit ~+2%) | the gap is **not** launch overhead |

`mx.compile` (functional cache + traced-array RoPE offset + `shapeless`) is verified **correct** — compiled output equals eager at both fp16 and 4-bit — and being **neutral**, it rules out kernel-launch/dispatch overhead as the cause of the gap to the ceiling.

**Cross-model scaling.** Measured as a back-to-back **achieved-BW ratio** anchored on the clean 0.6B value:

| Pair | Ratio | Implication |
|------|-------|-------------|
| fp16, 4B / 0.6B | **0.95×** | 4B ≈ **69%** usable, **same** as 0.6B ≈ 72% |
| 4-bit, 4B / 0.6B | 1.25× | **inflated by a dispatch confound** — not trustworthy |

The fp16 result shows the **~30% gap to the usable ceiling is REAL and SCALE-INDEPENDENT** — it does not shrink at 4B, so it is **not a small-model artifact**.

**Cross-engine — vs llama.cpp (Metal).** Same Qwen3-0.6B, matched quantization (GGUF Q8_0 ≈ silica 8-bit, Q4_K_M ≈ silica 4-bit; same byte model), `llama-bench` with full Metal offload. Measured under heavy load, so the robust output is the tok/s **ratio** (close-in-time, cancels contention):

| config | silica / llama.cpp |
|--------|--------------------|
| 8-bit | **0.90×** |
| 4-bit | **0.89× / 0.88×** |

**silica ≈ 0.89× llama.cpp — llama.cpp is ~12% faster**, and the ratio is **stable across a 2.5× load swing** (load 19→49), so it is a real engine property, not noise. For a transparent ~1000-line Python/MLX engine vs hand-tuned C++/Metal, an ~12% gap is a strong result — and consistent with silica == mlx-lm, i.e. the entire MLX Python stack sits ~12% behind llama.cpp, most plausibly residual per-token Python/dispatch overhead a compiled generation loop avoids. *Caveat:* this is **speed only** — llama.cpp's Q4_K_M k-quant is generally higher quality than silica's affine 4-bit, so at 4-bit llama.cpp likely wins on both axes until silica adopts a better 4-bit recipe (see `docs/results-m4-cross-engine.md`).

---

## 5. Decision — M3 custom kernels gated OUT by evidence

Custom kernels (M3) are **declined**, because the evidence pins the ~30% gap to a cause that custom kernels cannot fix:

- **Not launch / dispatch overhead** — `mx.compile` is neutral (§4).
- **Not small-GEMV inefficiency** — the gap is flat at 4B, i.e. scale-independent (§4).
- It is the **intrinsic efficiency of the real quantized-GEMV / attention access patterns** that Apple's `mx.quantized_matmul` already defines.

Re-deriving those kernels would, at best, reproduce Apple's numbers while re-implementing the highest-risk part of the stack (**audit risk #1**). Decision: **skip M3, ship the M4 write-up.**

---

## 6. Honest limitations

- **llama.cpp comparison is speed-only.** The cross-engine baseline (§4) compares decode *throughput*, not quality; llama.cpp's Q4_K_M k-quant is likely higher-quality than silica's affine 4-bit. A like-for-like quality (PPL) comparison across engines was not done.
- **Absolute % usable was contention-limited.** The test machine carried unavoidable background load on the shared CPU/GPU bus, so absolute "% of usable" figures are **bracketed**, not exact. To get trustworthy numbers under non-stationary contention we **interleaved** the two arms of each A/B and reported the **per-pair ratio** (drift cancels) — this is what rescued the `mx.compile` and silica-vs-mlx-lm comparisons. Cross-model % usable is reported as a **back-to-back achieved-BW ratio anchored on a known-clean value**.

**Measurement lessons (bugs found by RUNNING, not static review):**

- A **weight double-count** made fp16 weight traffic read **1503 MB** instead of the true **1192 MB**.
- A **context mismatch** (tok/s timed at short context but KV charged at 4096) made achieved BW **exceed the chip peak** — **457 GB/s > 400** — an impossibility that flagged the bug.
- batch=1 decode is partly **CPU-dispatch-bound**: a healthy GPU bandwidth ceiling alone does **not** certify a decode number; you need a **stable bus AND an idle CPU**.

---

*Reproduce: `bench/baseline.py` (silica vs mlx-lm), `bench/compile_ablation.py` (compile on/off), `bench/scaling.py` (cross-model), `bench/eval_ppl.py --ablate` (quant quality). See `docs/results-m1.md` and `docs/results-m2-baseline.md` for the raw run logs, and `docs/AUDIT.md` for the pre-build audit that shaped the design.*
