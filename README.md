# silica

A transparent, single-stream LLM inference engine for Apple Silicon, built on
MLX. Inverted from `mini-sglang`: on a Mac the bottleneck is **memory bandwidth
and quantization**, not GPU scheduling. ~835 lines of engine, audited and
benchmarked against `mlx-lm` and `llama.cpp`.

> **Status: v0 complete and audited.** M0–M2 validated on device, M3 (custom
> kernels) gated out by evidence, M4 written up; an 8-reviewer code audit is
> fully resolved; two architecture families (Qwen3 + Llama) pass exact parity.

## Results

On Apple M3 Max (40-core, 400 GB/s), Qwen3-0.6B / Llama-3.2-1B:

| Dimension | Result |
|---|---|
| **Correctness** | parity gate vs `mlx-lm` **+ an independent HuggingFace fp32 oracle**; 69-test suite |
| **Models** | dense Qwen3 + Llama **and MoE (OLMoE, Qwen3-MoE)**, registry-dispatched; exact next-token parity vs `mlx-lm` |
| **MoE roofline** | **Qwen3-30B-A3B 4-bit decodes 110 tok/s** at batch=1 (reads ~3B active/token) — ~7× a dense 30B; 51% of usable BW |
| **Pre-quantized load** | loads mlx-community 4/8-bit checkpoints directly (e.g. 30B-A3B in ~17 GB, not ~60 GB fp) |
| **Quantization** | 8-bit ~lossless; 4-bit/g64 +17.8% PPL, **4-bit/g32 +8.0%** (vs llama.cpp Q4_K_M +6.2%) |
| **Decode perf** | silica **== `mlx-lm`** within ~1.5% (~70% of usable bandwidth); `async_eval` +50% at 4-bit; `mx.compile` neutral |
| **vs `llama.cpp`** | silica ≈ **0.89×** decode speed — ~12% behind hand-tuned C++/Metal |
| **Custom kernels (M3)** | **declined** — the ~30% gap to the ceiling is real but scale-independent and already defined by Apple's `mx.quantized_matmul` |

The headline lever is quantization (the bandwidth denominator); the headline
non-finding is that, once `async_eval` hides per-step dispatch, there's little
left for a transparent engine to win over Apple's tuned kernels.

**Docs:** [PLAN.md](PLAN.md) (design + roadmap) · [docs/REPORT.md](docs/REPORT.md)
(performance & correctness scoreboard) · [docs/READING_GUIDE.md](docs/READING_GUIDE.md)
(annotated code tour + lessons) · [docs/AUDIT.md](docs/AUDIT.md) (pre-build plan audit) ·
[docs/CODE_AUDIT.md](docs/CODE_AUDIT.md) (post-build code audit) · results:
[m1 quant](docs/results-m1.md) · [m2 perf](docs/results-m2-baseline.md) ·
[m4 cross-engine](docs/results-m4-cross-engine.md) · [generality](docs/results-generality.md) ·
[MoE](docs/results-moe.md).

## Layout

```
silica/
  config.py       typed config (ModelConfig from HF config.json, Quant/Gen/Bench)
  weights.py      load safetensors (single + sharded), selective quantize, registry dispatch
  models/         per-architecture model files + a registry (SGLang-style)
    common.py     shared layers: MLP, SwitchGLU/MoEBlock, Decoder, CausalLM, build_rope
    qwen3.py llama.py        dense (Qwen3 QK-Norm; Llama llama3-RoPE)
    olmoe.py qwen3_moe.py    MoE (router + gathered experts via mx.gather_qmm)
    __init__.py   REGISTRY: HF `architectures` field -> model class
  cache.py        growing + quantized KV cache; PrefixCache (single-stream prefix reuse)
  attention.py    sdpa() — fp -> mx.fast SDPA; quantized KV -> quantized_matmul path
  sample.py       greedy + temp/top-k/top-p/min-p (per-sampler RNG key, no global seed)
  detokenize.py   incremental UTF-8-safe BPE detok + stop sequences + flush
  generate.py     chat template -> prefill -> decode loop (async_eval) -> streamed text
  compiled.py     M2 mx.compile decode experiment (correct, perf-neutral)
  kernels/        custom Metal fusions (M3, gated out by evidence — empty by design)
bench/            decode tok/s + achieved-bandwidth %; quant-quality PPL; cross-engine vs llama.cpp
tests/            pure-python (config, roofline, sampler, detok, cache, ppl) + device parity gates
```

**Supported models:** dense **Qwen3** and **Llama-3.x / SmolLM2**, plus
**Mixture-of-Experts** (**OLMoE**, **Qwen3-MoE** like Qwen3-30B-A3B) — dispatched
from the checkpoint's `architectures` field. Adding one is a small attention
block + registry entry; the entire runtime and bench harness are reused unchanged.
Qwen3, Llama, OLMoE, and Qwen3-30B-A3B each pass an exact next-token parity gate vs
`mlx-lm` ([generality](docs/results-generality.md), [MoE](docs/results-moe.md)). MoE
is the on-thesis case: Qwen3-30B-A3B holds 30B weights but reads only its ~3B active
experts — **110 tok/s at batch=1** (~7× a dense 30B), loaded pre-quantized in ~17 GB.

## Setup (Apple Silicon, [uv](https://docs.astral.sh/uv/))

```bash
cd silica
uv venv                                  # creates .venv (honors requires-python)
uv pip install -e ".[reference,dev]"     # mlx, transformers, ... + mlx-lm oracle
```

`uv run <cmd>` auto-uses the project venv, so you can skip `source .venv/bin/activate`.

## Run (on device)

```bash
# Greedy generation — works for either architecture
uv run silica-generate --model Qwen/Qwen3-0.6B            --prompt "Explain RoPE in one sentence."
uv run silica-generate --model unsloth/Llama-3.2-1B-Instruct --prompt "Name two planets."

# Decode benchmark — bandwidth-% needs the chip's RATED bandwidth (M3 Max = 300 OR 400)
uv run silica-bench --model Qwen/Qwen3-0.6B --tokens 128 --context-len 4096 --bandwidth 400 --chip "M3 Max 40c"

# Quantization-quality perplexity ablation (fp16 / 8-bit / 4-bit)
uv run silica-ppl --ablate --model Qwen/Qwen3-0.6B
```

## Test

```bash
# Pure-python (no MLX needed): config, roofline byte model, sampler, detok, ppl windowing
uv run pytest tests/test_config.py tests/test_roofline.py tests/test_detokenize.py

# Full device gate (needs MLX + mlx-lm + a checkpoint) — Qwen3 by default
SILICA_PARITY_MODEL=Qwen/Qwen3-0.6B uv run pytest -m device

# Llama generality gate downloads the ungated unsloth/Llama-3.2-1B-Instruct mirror
```

## Design highlights (what the audits hardened)

A pre-build **plan audit** and a post-build **code audit** (both 8-reviewer,
findings adversarially verified) shaped the engine:

- **Architecture details that break parity** — per-head QK-Norm before RoPE, no
  QKV bias, decoupled `head_dim`, tied `lm_head`, the right RoPE θ / llama3
  scaling (`silica/models/`).
- **A real output path** — incremental UTF-8-safe detokenizer with a `finalize()`
  flush, EOS/EOT + string stop-sequences, chat template (`detokenize.py`, `generate.py`).
- **Two oracles** — `mlx-lm` (same-backend regression guard) **and** an
  independent HuggingFace fp32 CPU oracle (the non-circular correctness check).
- **Honest figure of merit** — bytes/token = weights *(incl. per-group
  scales+biases)* + KV(context) + lm_head, over the recorded chip SKU's bandwidth
  (`bench/roofline.py`); a weight double-count and a context mismatch were caught
  by *running* (achieved BW exceeded the chip's physical peak).
- **Per-sampler RNG** — a fixed seed makes one run reproducible without clobbering
  global state (`sample.py`).

## Caveats

- **MLX is Metal-only**, so the device test gate doesn't run on hosted CI —
  run it locally or on a self-hosted Apple-Silicon runner.
- Under unavoidable machine load, absolute bandwidth numbers are contention-
  limited; the benches report **relative/interleaved ratios** (which cancel the
  noise) and flag contaminated runs.
- Quantized KV and rotating KV do **not** compose in mainline MLX.
- The Llama parity is vs `mlx-lm` (an independent implementation, but same
  `mx.fast` kernels); an HF fp32 oracle for Llama is left as future work.

## License

[MIT](LICENSE) (code). Model weights carry their own licenses — never committed here.
