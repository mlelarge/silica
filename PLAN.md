# silica — a transparent single-stream LLM engine for Apple Silicon

> Working codename. A compact, readable inference engine for small devices,
> in the spirit of `mini-sglang` but inverted for the constraints that actually
> bind on a Mac: memory bandwidth and quantization, not GPU scheduling.
>
> **Status:** pre-M0 design + scaffold. No results yet. The scaffold under
> `silica/` is unvalidated on device (correctness-first parity not yet run).

---

## 1. Motivation

`mini-sglang` (`sgl-project/mini-sglang`) is a ~5k-line Python *serving layer*
on top of CUDA kernels (FlashAttention/FlashInfer). Its headline wins — overlap
scheduling, tensor parallelism, optimized GPU kernels — all assume the GPU is
the scarce resource and the CPU just feeds it. On a Mac that premise inverts:

- The compute device *is* the bottleneck. There is **no large host↔device
  feeding gap** to hide as on a discrete GPU; a smaller per-step CPU-dispatch +
  sync bubble remains and is what `async_eval` targets (§4).
- We serve **one stream** (batch = 1), not high-throughput batches.
- Autoregressive **decode is memory-bandwidth bound**: throughput ≈
  `device_bandwidth / bytes_read_per_token`. Quantization (which shrinks the
  denominator) is therefore the central design decision, not an add-on.
- Unified memory means **no host↔device PCIe copies vs a discrete GPU**. Note:
  this is a property of the *platform*, shared by our own baselines (`mlx-lm`,
  `llama.cpp`-Metal), so it is not a `silica` differentiator — and it is not
  literally zero-sync (`mx.eval`/`mx.async_eval` boundaries remain).

The gap silica fills: a *pedagogically transparent* engine that **quantifies
where `mlx-lm`'s single-stream headroom is and tests whether custom fusion can
close it**. Transparency (an annotated, minimal reimplementation of the hot
path) is a first-class deliverable, not a side effect — so the project has
value even if the fusion kernels (M3) win nothing.

## 2. Scope

**v0 is (in scope):**
- Single-stream (batch = 1) inference of Qwen3-family decoder models.
- Apple Silicon / Metal GPU first. Record the exact chip + rated memory
  bandwidth per run (e.g. M3 Max ships as **300 *or* 400 GB/s**).
- 4-bit and 8-bit weight quantization (selective/mixed-precision, §5 M1).
- A correctness-first baseline, then targeted custom Metal kernel fusions.

**The batch=1 ceiling (owned, not hidden):** decode is bandwidth-bound
*because* batch=1 re-reads all weights per token. Batching is the canonical
fix that amortizes weight reads across streams — so by serving one stream we
**approach but can never beat** the bandwidth wall, only get closer to it. That
is a deliberate, teachable limitation of the whole perf story.

**Explicitly out of scope (for now):**
- High-throughput / continuous batching, an OpenAI server, request scheduling.
- Tensor parallelism, multi-device, distributed.
- Training, fine-tuning, autograd.
- The CPU backend and prefix/radix caching are **deferred**, not abandoned
  (see Roadmap → Later). "Few-stream batching" under Later is the deferred
  *bridge* toward the out-of-scope continuous batching, not a contradiction.

## 3. Design principles

1. **MLX is the substrate, not the engine.** Use `mx.array` (unified memory),
   lazy eval, and `mx.compile`. Don't reinvent the array runtime. **We
   reinvent nothing below the op level; our only original Metal is fusion of
   existing ops to cut traffic/launches.**
2. **Use `mx.fast.*` for the tuned-but-boring ops.** `rms_norm`, `rope`, and
   `scaled_dot_product_attention` — the last handles **GQA + causal for
   unquantized (fp16/bf16/fp32) KV**. *Quantized KV is NOT one fast call:*
   mainline MLX has no native quantized SDPA (feature request open), so
   quantized-KV attention is a separate hand-written path (`mx.quantized_matmul`
   ×2 + softmax), borrowed from `mlx-lm`. Plus `mx.quantized_matmul` for
   quantized weight GEMV. Apple has hand-tuned these; we will not beat their
   quantized GEMV on a first pass.
3. **Custom Metal kernels only where they earn it.** `mx.fast.metal_kernel`
   fusions that cut memory traffic / launch count — never a from-scratch matmul.
   Every kernel ships behind a pure-MLX fallback.
4. **Correctness before speed.** Numerical parity against a reference precedes
   any optimization. Optimizations are landed as measured ablations.
5. **Minimal and elegant, and readable.** Small surface area, no premature
   abstraction; the code doubles as the teaching artifact.

## 4. Architecture

```
silica/
  config.py      # typed config: ModelConfig (from HF config.json), Quant/Gen/Bench
  weights.py     # load Qwen3 safetensors (single + sharded/index.json), bf16->mlx
                 #   dtype, lazy load, SELECTIVE mx.quantize (keep embed/lm_head higher)
  model.py       # Qwen3 decoder (see block below), against mx.fast.*
  cache.py       # KV cache; growing -> {quantized | rotating}  (alternatives, not a stack)
  sample.py      # greedy + temperature/top-k/top-p/min-p, mx.random seeding
  detokenize.py  # incremental UTF-8-safe BPE detokenizer + EOS/EOT + stop sequences
  generate.py    # chat template -> prefill -> decode loop (async_eval) -> streamed text
  kernels/       # mx.fast.metal_kernel fusions, each with a Python fallback (M3, gated)
bench/           # decode tok/s + achieved-bandwidth %; roofline byte model; ablations
tests/           # parity (argmax gate + string-level), config, kernel correctness
```

**Qwen3 decoder block (the details parity depends on):**

```
RMSNorm                          # input_layernorm
  -> q/k/v_proj  (NO QKV bias; head_dim=128 read from config, ≠ hidden/n_heads)
  -> QK-RMSNorm  (per-head RMSNorm on Q and K over head_dim, BEFORE RoPE)   <-- Qwen3
  -> RoPE        (traditional=False, base = rope_theta = 1e6)
  -> GQA SDPA    (mx.fast.scaled_dot_product_attention, mask="causal")
  -> o_proj
RMSNorm                          # post_attention_layernorm
  -> SwiGLU MLP  (down(silu(gate(x)) * up(x)))
lm_head: tied to embed_tokens on 0.6B (tie_word_embeddings=true; no lm_head.weight);
         config-driven untied nn.Linear on larger members.
```

**The MLX division of labor (mirrors mini-sglang's Python + JIT-CUDA split):**

| Layer | mini-sglang | silica |
|---|---|---|
| Orchestration | Python | Python |
| Tuned ops | FlashInfer/FlashAttention | `mx.fast.*`, `mx.quantized_matmul` |
| Custom kernels | JIT CUDA | `mx.fast.metal_kernel` |
| Memory | discrete VRAM/HBM hierarchy; PCIe host↔device staging | unified memory; no VRAM, no PCIe staging, zero-copy CPU/GPU |

**Overlap, the one mini-sglang idea that ports:** their overlap scheduling
hides CPU scheduling behind GPU compute; the MLX analog is `mx.async_eval`,
which lets us **enqueue step t+1 without blocking on step t's GPU result**,
hiding per-step Python dispatch and the **host↔device eval/sync barrier**
(not "graph-build overhead" — `mx.compile` is what caches the trace). This is
the highest-leverage lever **that does not change model numerics**; the
*dominant* throughput lever remains quantization (§1), per the bandwidth model.
Correctness caveat: reading a sampled token (`.item()`, an EOS/stop check on a
host value) forces a sync and collapses the overlap — keep the token an
`mx.array` across the boundary.

## 5. Roadmap

### M0 — Correct baseline (no quantization, no custom kernels)
- [ ] Load Qwen3-0.6B safetensors into `mx.array` (single-shard; bf16→mlx dtype
      + chosen compute dtype recorded).
- [ ] Implement the decoder block in `model.py` using `mx.fast.*` — **including
      per-head QK-Norm, no QKV bias, head_dim=128 from config, tied lm_head.**
- [ ] Simple growing KV cache with explicit `offset`.
- [ ] Load tokenizer; **apply the ChatML chat template**; incremental UTF-8-safe
      detokenizer; **stop on EOS 151645 (`<|im_end|>`) + string stop-sequences.**
- [ ] Greedy `generate.py` that streams correct **text**, not just token IDs.
- [ ] **Parity test (acceptance criteria, §5a):** exact greedy/argmax-token
      match vs `mlx-lm` is the **hard gate**; per-layer hidden-state diff to
      localize failures; an independent **HF fp32 CPU oracle** (comparing only
      to `mlx-lm` is circular — same `mx.fast` kernels); plus a **string-level**
      decode test on a multibyte/emoji prompt. `allclose` only as a coarse,
      per-layer diagnostic (default rtol will never pass cross-backend bf16).

### M1 — Quantization
- [ ] **Selective** `mx.quantize` at load via a `class_predicate`: body 4-bit
      (group 64), **keep the (tied) embedding/lm_head at higher bits** and
      norms/RoPE in fp; verify each layer's last dim divides `group_size`.
- [ ] Add 8-bit path; expose `bits`/`group_size` as config (64 default;
      {32,64,128} as ablation knobs).
- [ ] Separate the two quantization stories: quantized **weights** (clear win)
      vs quantized **KV** (memory-for-speed tradeoff — quantized-KV decode is
      ~0.5× fp16; *measure*, don't assume). Quantized and rotating caches are
      **alternatives** (they do not compose in mainline MLX).
- [ ] **Quality eval harness** (`bench/eval_ppl.py`): perplexity on a fixed
      pinned corpus, reported as a column alongside the tok/s ablation, with a
      max-regression threshold gating quantization changes. Re-run parity at a
      stated looser tolerance.

### M1.5 — Sampler & long context
- [ ] Sampler beyond greedy: temperature / top-k / top-p / min-p / repetition
      penalty, with `mx.random` seeding (Qwen3 card advises against greedy).
- [ ] Pin `rope_theta` from config; state **32k native** context; YaRN
      (`rope_scaling`) only if/when >32k, else hard-cap to native and document.
- [ ] Rotating-cache eviction semantics: keep-first-k **attention-sink** tokens
      + sliding window as config; note parity holds only below the rotation
      threshold.

### M2 — Perf levers (no new kernels)
- [ ] **Baseline-first go/no-go:** measure `mlx-lm`'s achieved-bandwidth % first.
      If it is already >~75% of usable bandwidth, the project's value pivots to
      pedagogy/measurement and M3 is likely not worthwhile — decide here.
- [ ] `mx.async_eval` overlap in the decode loop (no per-token host sync).
- [ ] `mx.compile` the per-step *decode* forward (fixed shape; cache via
      `inputs=`/`outputs=`), prefill kept eager. Treat "does compile help with a
      stateful/quantized cache?" as an explicit experiment with a documented
      expected-null outcome (`mlx-lm` does **not** compile its loop).
- [ ] Ablation: async_eval on/off, compile on/off → tok/s **and** bandwidth-%
      table (a flat % with rising tok/s is expected and interpretable).

### M3 — Custom Metal kernels (gated on M2 profiling)
- [ ] Profile to confirm the actual hot path before writing anything.
- [ ] Quantitative gate: M3 only if baseline achieved-bandwidth < X% **and**
      profiling shows a fusible >Y% traffic/launch cost.
- [ ] First fusion candidate: dequant + GEMV + SwiGLU for the MLP block.
- [ ] Second candidate: fused (possibly quantized) attention iff our KV layout
      diverges from `mx.fast.scaled_dot_product_attention`.
- [ ] Each kernel: correctness test vs pure-MLX fallback + standalone microbench.

### M4 — Benchmarking & write-up
- [ ] Full roofline report (§6) across model sizes / quant settings.
- [ ] Comparison vs `mlx-lm` and `llama.cpp` (Metal) with a **fairness
      checklist**: identical weights/quant recipe (bits/group), prompt set,
      sampler, warmup, thermal steady state; report **bandwidth-%** as the
      primary cross-engine number, not raw tok/s.
- [ ] **Annotated reading guide / design narrative** (gated deliverable): the
      pedagogy artifact that justifies the project independent of M3's outcome.

### Later (deferred, not abandoned)
- CPU backend (Accelerate/AMX + NEON) behind the same op interface.
- Prefix/radix KV cache (system-prompt + multi-turn reuse).
- Speculative decoding (draft + target both in MLX).
- Optional thin OpenAI-compatible server + few-stream batching (the bridge to
  the out-of-scope continuous batching).

### 5a. Acceptance criteria (gates)
- **M0:** exact greedy-token match vs `mlx-lm` on a fixed prompt set AND a
  decoded-string match (incl. multibyte) AND HF-oracle argmax agreement. Commit
  a concrete per-layer `atol`/`rtol` for the diagnostic diff.
- **M1:** quantized greedy-token match within a stated divergence budget AND
  perplexity regression < threshold on the pinned corpus.
- **M2:** every ablation row carries N warmup discarded, median + IQR/CI over K
  runs, recorded chip SKU + thermal state.
- **M3:** each kernel beats its pure-MLX fallback in a microbench AND preserves
  the M0/M1 parity gate.

## 6. Performance methodology

The figure of merit for a bandwidth-bound engine is **% of peak memory
bandwidth achieved**, not raw tok/s in isolation. Always report tok/s
**alongside** bandwidth-% (quantization raises tok/s while `bytes×tok/s` may
stay flat — both numbers are needed to read an ablation).

- **Decode:** measure tok/s; compute achieved bandwidth =
  `total_bytes_read_per_token * tok/s`, where
  `total_bytes = weight_bytes + kv_bytes(context_len) + embedding/lm_head/activation_bytes`.
  - `weight_bytes` use the **on-device quantized footprint including per-group
    scales+biases** (affine stores both; ~12.5% over packed bytes at 4-bit/g64,
    ≈4.5 effective bits/weight) — not `params*bits/8`.
  - `kv_bytes` grow linearly with context and **dominate at long context**
    (for Qwen3-4B, KV ≈ weights by ~8k tokens and ~10× by 128k). Report
    bandwidth-% **as a function of context length**; keep a weights-only
    sub-metric only if explicitly labeled.
  - Report as a fraction of the **recorded chip SKU's rated bandwidth** (pin
    300 vs 400 GB/s, etc.) — never a single hardcoded "Apple Silicon" number.
- **Prefill:** tok/s vs prompt length. Mostly compute-bound for long prompts,
  but **sweep prompt length** and report the empirical arithmetic-intensity
  crossover; short-prompt TTFT on a small model may be latency/bandwidth-bound,
  not on the FLOPs roofline.
- **Latency:** TTFT and steady-state inter-token latency.
- **Memory:** peak RSS / unified-memory footprint (note quantize-at-load can
  transiently hold fp + quantized copies).
- **Rigor (else the ablation table is noise):** discard N warmup iterations
  (first calls pay JIT / Metal pipeline build); report median + IQR or 95% CI
  over K runs; record GPU/memory clock or note sustained-vs-burst **thermal**
  state (laptop M3 Max throttles under sustained decode); add **tok/s/W**.
- **Ablations (mini-sglang style):** async_eval, mx.compile, quant bits, group
  size — each a delta against the M0/M2 baseline, with the rigor above.

## 7. Risks & open questions

- **Baseline already near the wall (top unlisted risk).** If `mlx-lm` already
  achieves a high fraction of usable bandwidth on 4-bit decode, silica's best
  case is within measurement noise of the baseline it set out to improve. → M2
  baseline-first go/no-go turns this into an early decision.
- **Thesis collapse if M3 yields nothing.** M0–M2 (Qwen decode on `mx.fast`,
  4/8-bit weight quant, quantized + rotating KV, async_eval loop) already exist
  in `mlx-lm`; the only genuinely new artifact is the M3 fusion kernels, which
  may not beat `mx.fast`. → Pedagogy is a gated M4 deliverable so the project
  succeeds regardless.
- **Custom kernels may not beat `mx.fast.*`.** Likely for GEMV; the win must
  come from *fusion* (traffic/launch reduction). M3 is gated on M2 profiling.
- **`mx.compile` + stateful/quantized cache friction.** compile recompiles on
  shape/dtype change; the cache changes representation when quantization kicks
  in (and rotating + quantized do not compose). `mlx-lm` does not compile its
  decode loop — treat M2's compile step as an experiment, not a given.
- **Quantized-KV is memory-for-speed.** It buys long context but can cost
  decode throughput (~0.5× fp16) and spikes memory on full-cache dequant.
- **MLX API churn.** `mx.fast.*` and the `metal_kernel` API move fast; pin the
  MLX version (exact `==`) and track release notes.
- **Quantization quality.** A perplexity/task eval on a pinned corpus is a
  first-class deliverable (M1), not a one-off.
- **KV cache layout vs MLX SDPA.** Adopt MLX's layout (free, fast) or a custom
  one (needed for a custom attention kernel later) — decide before M3.

## 8. References

- `sgl-project/mini-sglang` — architecture & ablation style (GPU serving);
  LMSYS blog "Mini-SGLang: Efficient Inference Engine in a Nutshell".
- MLX custom Metal kernels — `mx.fast.metal_kernel` (Python source-string JIT).
- `mlx-lm` — KV cache patterns (`KVCache`, `QuantizedKVCache`, `RotatingKVCache`
  with sink+window), selective quantization (`class_predicate`, mixed recipes),
  `StreamingDetokenizer`, `generate_step` (`async_eval`), model loading.
- Qwen3 — model cards / `config.json` (head_dim 128, `attention_bias:false`,
  `tie_word_embeddings:true`, `rope_theta:1e6`, native 32k) and the Qwen3
  technical report (arXiv:2505.09388 — QKV-bias removed, QK-Norm added).
- WWDC25 "Get started with MLX for Apple silicon" — `mx.fast`, `mx.compile`,
  quantization overview.
- SGLang-on-MLX macOS backend (SGLang issue #17846 / the macOS-MLX gist) —
  existence proof of the *stack* (mlx-lm models + MLX caches + fused ops), but
  it is an **in-progress batched serving** backend doing no custom kernels, so
  it does not pre-empt silica's batch=1 + fusion niche.

---

### First commit checklist
- [x] `pyproject.toml`, pin `mlx`, `mlx-lm` (reference/dev only), `safetensors`,
      `transformers` (tokenizer + HF oracle), `numpy`.
- [x] `silica/config.py` typed config; `silica/model.py` Qwen3 block (QK-Norm,
      no bias, head_dim from config, tied lm_head).
- [x] `silica/weights.py`, `cache.py`, `sample.py`, `detokenize.py`,
      `generate.py` scaffolds.
- [x] `tests/test_parity.py` (argmax gate + string-level) against `mlx-lm` for
      Qwen3-0.6B; `tests/test_config.py`.
- [x] `bench/decode.py` printing tok/s + achieved-bandwidth %; `bench/roofline.py`
      byte model (weights+scales+KV+lm_head).
- [ ] Run M0 parity on device (Apple Silicon + MLX installed) — **not yet done.**
