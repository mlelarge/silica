# Generality — a second architecture (Llama) through one runtime

The project was scoped to Qwen3 to keep the correctness bar exact (see
`results-m4-cross-engine.md` / the README). This adds a **second architecture
family (Llama)** to demonstrate that silica's design — not just its Qwen3 model
file — generalizes, mirroring how SGLang/vLLM handle many models.

## What changed

A SGLang-style split of the model layer:

```
silica/models/
  common.py    shared layer library: MLP (SwiGLU), DecoderLayer, Decoder,
               CausalLM base, causal mask, build_rope (incl. llama3 scaling)
  qwen3.py     Qwen3Attention (per-head QK-norm) + Qwen3ForCausalLM
  llama.py     LlamaAttention (no QK-norm)       + LlamaForCausalLM
  __init__.py  REGISTRY: HF `architectures` field -> model class
```

`weights.load_model` now calls `build_model(cfg)`, which dispatches on the
checkpoint's `architectures` field (`"Qwen3ForCausalLM"` / `"LlamaForCausalLM"`).
**Everything else is reused unchanged**: `cache.py`, `attention.py` (fp + quantized
SDPA), `sample.py`, `detokenize.py`, `generate.py`, and the whole `bench/`
harness. Adding Llama was the registry + one ~40-line attention block.

## The Qwen3 → Llama deltas silica handles (config-driven)

| | Qwen3-0.6B | Llama-3.2-1B |
|---|---|---|
| per-head QK-norm | **yes** | no |
| RoPE | plain, θ=1e6 | **llama3 scaling** (factor 32), θ=5e5 |
| GQA (q/kv heads) | 16 / 8 | 32 / 8 |
| head_dim | 128 (decoupled) | 64 (= hidden/heads) |
| rms_norm_eps | 1e-6 | 1e-5 |
| tied lm_head | yes | yes |

The **llama3 RoPE scaling** is the parity-critical piece (it rescales the
frequency spectrum at *all* positions, not just long context). silica computes
the adjusted frequencies and passes them to `mx.fast.rope(..., freqs=…)`, kept
out of the module's parameter tree so the checkpoint loads cleanly.

## Validation (`tests/test_llama.py`, device-gated)

Model: `unsloth/Llama-3.2-1B-Instruct` (ungated mirror of the gated Meta weights).

- **Registry dispatch**: `architectures=("LlamaForCausalLM",)` → `LlamaForCausalLM`.
- **Exact next-token parity vs mlx-lm** on multiple prompts (argmax match + 5/5
  top-5 overlap). mlx-lm's Llama is an *independent* implementation, so matching
  it proves silica's Llama wiring (incl. the llama3 RoPE) is correct.
- **Clean generation** through the shared `generate()` loop: e.g. "List three
  colors:" → coherent, replacement-char-free text.

Full suite: **69/69** green (65 Qwen3 + infra, 4 new Llama). The same Qwen3 path
is unchanged and still passes after the refactor.

## Notes / limits

- The Llama parity is against mlx-lm (a different implementation, but same MLX
  `mx.fast` kernels). An independent HF fp32 oracle (as for Qwen3 M0) could be
  added for a fully non-circular check.
- `compiled.py` (the M2 `mx.compile` experiment) is still Qwen3-specific
  (it hand-rolls the QK-norm forward); the normal generate path is model-agnostic.
- Other `rope_scaling` types (yarn/linear/dynamic) fall through to plain RoPE —
  add them in `build_rope` when a model needs them. SmolLM2 (plain-RoPE Llama)
  also works via the same `LlamaForCausalLM`.
