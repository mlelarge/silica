# MoE — sparse activation, on-thesis for bandwidth-bound decode

silica was dense-only; this adds Mixture-of-Experts support and, with it, the
most on-thesis demonstration of the whole project: at batch=1, decode is
memory-bandwidth-bound, and MoE reads only its **active** experts per token — so a
big model decodes at the bandwidth cost of a small one.

## What was added

A registry extension (same pattern as Llama), reusing the entire runtime:

```
silica/models/
  common.py     + SwitchLinear / QuantizedSwitchLinear / SwitchGLU / MoEBlock
                  (transparent; uses MLX-native mx.gather_mm / mx.gather_qmm)
  olmoe.py      OLMoE: full-projection QK-norm + MoE, untied lm_head
  qwen3_moe.py  Qwen3-MoE: reuses Qwen3 per-head QK-norm attention + MoEBlock
```

- A token's top-k experts are computed with MLX's **gathered matmul** (`gather_mm`,
  or `gather_qmm` for quantized experts) — silica doesn't hand-roll the dispatch.
- `DecoderLayer`/`Decoder`/`CausalLM` now take an `mlp_cls`; a `sanitize()` hook
  stacks HF per-expert weights into `(num_experts, out, in)` tensors.
- Selective quant keeps the MoE **router (`mlp.gate`) in fp** (tiny, routing-sensitive).
- `bench/roofline.py` is MoE-aware: it counts **active** params (top-k of N + router)
  per token, not all experts.

## Correctness — OLMoE parity (validated on device)

`allenai/OLMoE-1B-7B-0125-Instruct` (64 experts, top-8): **exact next-token parity
vs mlx-lm** (argmax match + 5/5 top-5 overlap) on multiple prompts, plus clean
generation. mlx-lm's OLMoE is an independent implementation, so the match proves
silica's router, SwitchGLU/`gather_mm`, full-projection QK-norm, and the
expert-stacking `sanitize` are all correct. `tests/test_olmoe.py`; dense suite
unchanged (72/72).

## The roofline — a 30B model at ~3B of bandwidth

Analytic from each config (M3 Max-40c, ~370 GB/s usable, 4-bit weights):

| model | total | active/tok | sparsity | 4-bit bytes/tok | max tok/s @ 370 |
|---|---|---|---|---|---|
| Qwen3-0.6B (dense) | 0.8 B | 0.75 B | 100% | 374 MB | 989 |
| OLMoE-1B-7B | 6.9 B | 1.28 B | 19% | 689 MB | 537 |
| **Qwen3-30B-A3B** | **30.5 B** | **3.35 B** | **11%** | 1789 MB | **207** |

The byte model's active count (3.35 B) matches the model's own name — **A3B = ~3 B
active** — an independent check of the accounting. A *dense* 30B at 4-bit reads
~17 GB/token → ~22 tok/s; **Qwen3-30B-A3B reads only its active experts → ~207
tok/s, ~9× faster at batch=1** for the same total capacity. That is exactly the
bandwidth-bound thesis: MoE is the architecture that buys capability per byte read.

## Honest limits

- **Qwen3-30B-A3B is registered and roofline-analyzed but not empirically
  parity-run** — it reuses components already validated (Qwen3 attention on
  Qwen3-0.6B + the MoEBlock on OLMoE), and the real fp decode is download-gated
  (~60 GB fp / would need pre-quantized loading support, which silica doesn't have
  yet — it quantizes fp at load).
- The SwitchGLU `do_sort` token-reordering optimization (for large prefills) is
  omitted — correct but slower than mlx-lm on big batched prefills; irrelevant at
  batch=1 decode.
- Mixed dense/MoE layers (`mlp_only_layers`) aren't handled; Qwen3-30B-A3B is
  all-MoE (`decoder_sparse_step=1`) so it's fine.
