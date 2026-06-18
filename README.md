# silica

A transparent, single-stream LLM inference engine for Apple Silicon, built on
MLX. Inverted from `mini-sglang`: on a Mac the bottleneck is **memory bandwidth
and quantization**, not GPU scheduling.

> **Status: M0–M2 validated on device; M3 gated out by evidence; M4 written up.**
> Qwen3-0.6B decode is correctness-proven (parity gate 7/7, incl. an independent
> HuggingFace fp32 oracle), quantization + quantized-KV land in M1, and the M2
> roofline shows silica at parity with `mlx-lm` (~70% of usable bandwidth) with
> `async_eval` as the big lever and `mx.compile` neutral. Custom Metal kernels
> (M3) are **declined** — the ~30% gap to the ceiling is real, scale-independent,
> and already defined by Apple's `mx.quantized_matmul`.

**Docs:** [PLAN.md](PLAN.md) (design + roadmap) · [docs/REPORT.md](docs/REPORT.md)
(the performance & correctness scoreboard) · [docs/READING_GUIDE.md](docs/READING_GUIDE.md)
(annotated code tour + lessons learned) · [docs/AUDIT.md](docs/AUDIT.md) (pre-build
audit) · results: [m1](docs/results-m1.md) · [m2](docs/results-m2-baseline.md) ·
[m4 cross-engine](docs/results-m4-cross-engine.md) (vs llama.cpp).

## Layout

```
silica/
  config.py       typed config (ModelConfig from HF config.json, Quant/Gen/Bench)
  weights.py      load safetensors (single + sharded), selective quantize, registry dispatch
  models/         per-architecture model files + a registry (SGLang-style)
    common.py     shared layers: MLP, DecoderLayer, Decoder, CausalLM, mask, build_rope
    qwen3.py      Qwen3 (per-head QK-Norm)        llama.py   Llama (no QK-Norm, llama3 RoPE)
    __init__.py   REGISTRY: HF `architectures` field -> model class
  cache.py        growing KV cache (+ quantized); quant|rotating are alternatives
  sample.py       greedy + temp/top-k/top-p/min-p (per-sampler RNG key)
  detokenize.py   incremental UTF-8-safe BPE detok + stop sequences + flush
  generate.py     chat template -> prefill -> decode loop (async_eval) -> streamed text
  kernels/        custom Metal fusions (M3, gated out by evidence — empty by design)
bench/            decode tok/s + achieved-bandwidth %; cross-engine vs llama.cpp
tests/            pure-python (config, roofline, sampler, detok, cache) + device parity gates
```

**Supported models:** Qwen3 and Llama-3.x / SmolLM2 (both `LlamaForCausalLM`),
dispatched from the checkpoint's `architectures` field. Adding one is a ~40-line
attention block in `silica/models/` + a registry entry; the entire runtime and
benchmark harness are reused unchanged. Both families pass an exact next-token
parity gate vs `mlx-lm` (see [docs/results-generality.md](docs/results-generality.md)).

## Setup (Apple Silicon, [uv](https://docs.astral.sh/uv/))

```bash
cd silica
uv venv                                  # creates .venv (honors requires-python)
uv pip install -e ".[reference,dev]"     # mlx, transformers, ... + mlx-lm oracle
```

`uv run <cmd>` auto-uses the project venv, so you can skip `source .venv/bin/activate`.

## Run (once on device)

```bash
# Greedy generation (M0)
uv run silica-generate --model Qwen/Qwen3-0.6B --prompt "Explain RoPE in one sentence."

# Decode benchmark — bandwidth-% needs the chip's RATED bandwidth (M3 Max = 300 OR 400)
uv run silica-bench --model Qwen/Qwen3-0.6B --tokens 128 --context-len 4096 \
                    --bandwidth 400 --chip "M3 Max 40c"
```

## Test

```bash
# Pure-python (no MLX needed): config + corrected roofline byte model
uv run pytest tests/test_config.py tests/test_roofline.py

# M0 parity gate (needs MLX + mlx-lm + a Qwen3 checkpoint)
SILICA_PARITY_MODEL=Qwen/Qwen3-0.6B uv run pytest -m device
```

## What's deliberately corrected vs the first draft

The audit (see `docs/AUDIT.md`) caught issues now baked into this scaffold:

- **Qwen3 QK-Norm + no QKV bias + decoupled `head_dim` + tied `lm_head`** —
  without these the parity gate cannot pass (`model.py`).
- **A real output path** — incremental UTF-8-safe detokenizer, EOS/EOT + string
  stop-sequences, ChatML chat template (`detokenize.py`, `generate.py`).
- **Corrected figure of merit** — bytes/token counts weights *(incl. per-group
  scales+biases)* **+ KV(context) + lm_head**, reported vs context length, over
  the recorded chip SKU's rated bandwidth (`bench/roofline.py`).
- **Selective quantization** — keep the (tied) embedding/lm_head higher-precision
  (`weights.py`).
- **Parity gate = argmax/greedy-token match + string match**, not tight
  `allclose`; mlx-lm is a same-backend check, HF fp32 is the independent oracle
  (TODO) (`tests/test_parity.py`).

## Caveats

- CI cannot run the device gate on hosted runners (MLX is Metal-only); run it
  locally or on a self-hosted Apple-Silicon runner.
- Comparing only to `mlx-lm` is circular (shared `mx.fast` kernels); the HF
  fp32 CPU oracle is the non-circular check (still a TODO in `test_parity.py`).
- Quantized KV and rotating KV do **not** compose in mainline MLX.

## License

MIT (code). Model weights carry their own licenses — never committed here.
