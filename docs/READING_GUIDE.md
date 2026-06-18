# silica — Reading Guide

A guided tour of the codebase for an engineer learning how a transparent,
single-stream MLX inference engine works. Read the modules in the order below
(roughly the data flow: config → load → model → cache/attention → sample →
detokenize → generate → compiled → bench), then the lessons section, which is
where the project's transferable insights live.

silica deliberately stays small and readable: it leans on `mx.fast.*` and
`mx.quantized_matmul` for the tuned-but-boring ops and reimplements only the
single-stream decode hot path, so the code doubles as the teaching artifact.

---

## Code walkthrough

### `silica/config.py`

`config.py` is the single source of typed truth for everything silica reads off a model and every knob the engine and benchmarks expose. It defines four frozen dataclasses — `ModelConfig`, `QuantConfig`, `GenConfig`, and `BenchConfig` — so configuration is validated once, immutably, rather than passed around as loose `dict`s.

`ModelConfig` mirrors the fields silica reads from a Qwen3 HuggingFace `config.json`, and its design choice is to encode the Qwen3 traps as *required, explicit* fields rather than derived quantities. The headline trap is `head_dim`: it is decoupled from `hidden_size // num_attention_heads` (Qwen3-0.6B uses 128, not 1024/16 = 64), so it is a mandatory field and `__post_init__` even recomputes the derived value to keep the divergence visible in stack traces. The other defaults bake in Qwen3 reality: `attention_bias=False` (Qwen2 had QKV bias, Qwen3 dropped it), `tie_word_embeddings=True` (the 0.6B has no separate `lm_head` weight — the embedding *is* the output projection), and `rope_theta=1_000_000.0` (the generic 1e4 default would break numerical parity). `from_dict`/`from_json` build it, filtering unknown keys and only falling back to a derived `head_dim` if absent. Convenience properties `n_rep` (the GQA query-per-KV repeat factor) and `eos_token_ids` (normalized to a tuple) round it out.

`QuantConfig` encodes the selective-quantization policy: `bits`/`group_size` for the body, but `embed_bits=6` keeps the tied embedding/lm_head at higher precision and `quantize_norms=False` keeps RMSNorm in fp. `GenConfig` holds sampling controls (greedy at `temperature=0.0`) plus quantized-KV settings (`kv_bits`, `quantized_kv_start`). `BenchConfig` carries harness rigor: `warmup`, `runs`, sweep `context_lengths`, and a deliberately defaulted-`None` `device_bandwidth_gbps` so a "% of peak" figure can never be silently mislabeled.

### `silica/weights.py`

`weights.py` is the bridge between an on-disk Qwen3 checkpoint and a live `Qwen3ForCausalLM` instance. Its public entry point, `load_model(model, *, quant=None, dtype=mx.bfloat16)`, returns a `(net, cfg)` pair and is what every bench harness calls to obtain a model.

`resolve_model_path` accepts either a local snapshot directory (used as-is) or an HF repo id, in which case it lazily imports `huggingface_hub.snapshot_download`. `_load_safetensors` handles both checkpoint layouts: if `model.safetensors.index.json` exists it loads the deduplicated set of shards named in its `weight_map`; otherwise it globs `*.safetensors` (the single-file Qwen3-0.6B case). Each file goes through `mx.load`, which is **lazy read-on-eval, not mmap** — the bytes aren't materialized until an `mx.eval`.

The ordering inside `load_model` is the crucial, non-obvious part. After building `net`, it loads the fp checkpoint and only then quantizes:

1. If `cfg.tie_word_embeddings`, a stray `lm_head.weight` is dropped (`weights.pop`).
2. Every tensor is cast with `v.astype(dtype)` (Qwen3 ships bf16).
3. `net.load_weights(...)` then `mx.eval(net.parameters())` forces the lazy reads to materialize.
4. **Only after that** does `nn.quantize` run, if `quant` is given.

The reason: `nn.quantize` quantizes each module's *current* in-place weights, so quantizing before loading would leave modules expecting packed `weight/scales/biases` tensors the fp checkpoint can't supply. Quantization is **selective**, driven by `_selective_predicate(qcfg)` (the `class_predicate` passed to `nn.quantize`): it keeps the tied `embed_tokens`/`lm_head` at the higher `qcfg.embed_bits`, and — enforcing a hard `mx.quantize` constraint — leaves any layer whose last dim isn't divisible by `group_size` in fp, recording it on `predicate.skipped`.

### `silica/model.py`

The Qwen3 decoder block is the heart of silica. `model.py` defines four nested `nn.Module` classes — `Attention`, `MLP`, `DecoderLayer`, `Qwen3Model` — plus the top-level `Qwen3ForCausalLM` and a free function `causal_additive_mask`. Module and parameter names mirror the HuggingFace Qwen3 checkpoint layout exactly, so `model.load_weights(...)` loads without key renaming.

Four Qwen3-specific subtleties are deliberately surfaced. First, **per-head QK-RMSNorm before RoPE**: `Attention` holds `self.q_norm` and `self.k_norm`, both `nn.RMSNorm(self.head_dim, ...)`. In `__call__`, `q` and `k` are reshaped to `(b, l, n_heads, head_dim)` and normalized over the trailing `head_dim` *before* being transposed and passed through `self.rope`. Get this order wrong and numerics diverge silently. Second, **no QKV bias**: the projections use `bias=cfg.attention_bias`, which is `False` for Qwen3. Third, **`head_dim` comes from config** (`self.head_dim = cfg.head_dim`), decoupled from `hidden_size // n_heads`, so `self.scale = self.head_dim**-0.5` stays correct. Fourth, **tied lm_head**: when `cfg.tie_word_embeddings` is set, `Qwen3ForCausalLM.__call__` projects with `self.model.embed_tokens.as_linear(h)` instead of a separate `lm_head`.

`causal_additive_mask(seq_len, offset, dtype)` is **offset-aware**: it returns `None` for single-token decode (`seq_len <= 1`), and otherwise builds a `(seq_len, offset+seq_len)` additive mask — the `offset` term (read from `cache[0].offset`) keeps prefill correct under KV-cache continuation. Attention delegates the actual product to `sdpa` (from `attention.py`).

### `silica/cache.py`

The KV cache stores the keys and values computed for every past token so that decoding step *t+1* never recomputes attention over tokens *0..t*. The module ships one concrete cache for M0 — `KVCache`, a *growing* per-layer cache. Its `update_and_fetch(keys, values)` appends the new step's K/V and returns the full valid slice. The key trick is **chunked preallocation**: instead of reallocating every token, it allocates `mx.zeros` buffers rounded up to the next multiple of `step` (default 256), so the sequence dimension grows in coarse jumps. The real length is tracked by `self.offset`; `self.keys.shape[2]` is the padded capacity, and the return slices `[..., :need, :]` hide the padding. This coarse growth matters for the M2 `mx.compile` work, which recompiles on shape change. The `state` property exposes `(keys, values, offset)` for compile capture.

`QuantizedKVCache` stores K and V compressed: each of `self.keys`/`self.values` is **not an array but a `(packed_uint32, scales, biases)` tuple** produced by `mx.quantize`. It grows the same chunked way (`init_quant` sizes the packed dim as `dim // el_per_int` and the scale/bias dims as `dim // group_size`), and its `update_and_fetch` returns `tree_map`-sliced tuples that the quantized-SDPA path consumes. `KVCache.to_quantized(group_size, bits)` converts a populated fp cache into one of these; `make_cache` is the factory (`kv_bits=None` → growing, else quantized).

The load-bearing subtlety, pinned by the `RotatingKVCache` stub: **growing → {quantized | rotating} are alternatives, not a composition**. In mainline mlx-lm a rotating cache's ring-buffer temporal reordering makes in-place quantization complex, so `RotatingKVCache.to_quantized()` is unimplemented — a quantized *and* rotating cache is an open problem. Rotation also makes output a lossy function of history, so the logits-parity gate is only exact for prompts shorter than the rotation threshold.

### `silica/attention.py`

`attention.py` implements attention dispatch for the decode loop, splitting cleanly on KV-cache dtype. Its public entry point is `sdpa(queries, keys, values, *, scale, mask, cache=None)`, which performs grouped-query (GQA) causal attention. The design hinges on one branch: if `cache` is a `QuantizedKVCache`, it delegates to `quantized_sdpa`, threading through `cache.group_size` and `cache.bits`; otherwise it calls MLX's fused `mx.fast.scaled_dot_product_attention` directly.

That branch encodes the module's central subtlety: `mx.fast.scaled_dot_product_attention` accepts only floating-point (fp16/bf16/fp32) keys and values — mainline MLX has no native quantized-KV SDPA. So when KV is quantized, `keys`/`values` arrive as `(packed_uint32, scales, biases)` tuples that the fused kernel cannot consume. `quantized_sdpa` reconstructs the math by hand, mirroring mlx-lm: it pre-scales `queries`, computes scores with `mx.quantized_matmul(queries, *q_keys, transpose=True, ...)`, adds the `mask`, applies `mx.softmax(..., axis=-1, precise=True)`, then a second `mx.quantized_matmul(scores, *q_values, transpose=False, ...)` for the weighted value sum.

GQA is handled explicitly via reshaping: with `n_repeats = n_q_heads // n_kv_heads > 1`, queries are reshaped to `(B, n_kv_heads, n_repeats, L, D)` and each component of the K/V tuples gets a broadcast axis inserted by `tree_map(lambda x: mx.expand_dims(x, axis=-3), ...)`; the output is reshaped back. The trade-off, per the docstring: this manual path is correct but roughly 0.5× fp16 throughput — it trades speed for the memory savings of a quantized cache.

### `silica/sample.py`

The sampler turns model logits into the next token. Its single public entry point is `make_sampler(cfg: GenConfig)`, a factory that reads a `GenConfig` once and returns a closure `sampler(logits) -> mx.array` operating on logits shaped `(B, vocab)`. Binding the strategy at construction time keeps the hot decode loop branch-light.

The seed is set once (`mx.random.seed(cfg.seed)`) inside the factory, and `greedy = cfg.temperature <= 0.0` is precomputed. In greedy mode it returns `mx.argmax(logits, axis=-1)`; otherwise it scales by `1.0 / cfg.temperature` and applies optional filters in order — `_top_k`, `_min_p`, `_top_p` — before `mx.random.categorical`. The filter helpers never drop tokens; they mask rejected logits to `-inf` via `mx.where`, preserving tensor shape. `_top_p` (nucleus) sorts ascending, takes cumulative softmax mass, keeps the top `top_p` fraction, then scatters back with `inv = mx.argsort(sorted_idx)`.

The load-bearing subtlety (audit `mlx-3`): the sampler returns an `mx.array`, **never** a Python int — no `.item()`. Calling `.item()` in the decode loop forces a host↔device sync and destroys the `async_eval` overlap from M2. The token stays lazy; it is materialized only later, in the detokenizer, after the next step has been enqueued.

### `silica/detokenize.py`

`detokenize.py` solves the output-side problem the original plan skipped: turning a stream of token IDs back into correct, user-visible text. Because Qwen3 uses a **byte-level BPE** tokenizer, a single multibyte UTF-8 character (an emoji, an accented letter) can straddle two tokens, so naively decoding each token in isolation would produce mojibake.

The module is one class, `IncrementalDetokenizer`, wrapping any HF tokenizer that exposes `.decode(list[int]) -> str`. It keeps the full ID history in `self._ids`, the cumulative decode in `self._text`, and a cursor `self._emitted`. The core method, `add_token(token_id)`, appends the id, re-decodes the *entire* buffer, and returns only the newly completed suffix. The key trick: if the fresh decode ends with `REPLACEMENT` (U+FFFD), the trailing character is still incomplete, so it returns `""` and withholds output until a later token finishes the bytes. Decoding the whole buffer each time (rather than incrementally) is what makes this correct.

It also owns termination via string stop-sequences: `_first_stop_index` finds the earliest occurrence of any `stop` string, and `add_token` emits only up to that cut, setting `self.finished = True`. (EOS/EOT *token-id* stops are handled separately in `generate.py`.)

### `silica/generate.py`

`generate.py` is the engine's top of the stack: it turns a prompt string into streamed decoded text, and is the clearest place to see the MLX async-decode idiom in isolation.

The public entry is `generate(model, tokenizer, prompt, cfg, *, stream=True)`. It assembles the stop set `eos_ids` from both `mcfg.eos_token_ids` and `tokenizer.eos_token_id`, encodes via `_encode_prompt`, then drives `generate_step` and feeds each token to an `IncrementalDetokenizer`. `_encode_prompt` is where ChatML lives: when `cfg.use_chat_template` and the tokenizer has a `chat_template`, it wraps the prompt as a user message and calls `apply_chat_template(..., add_generation_prompt=True)`, defensively unwrapping a `BatchEncoding` into `input_ids`.

The heart is `generate_step`, an `Iterator[int]`. A local `step(tokens)` runs `model(tokens, cache=cache)[:, -1, :]` through `sampler`. The first call on `mx.array(prompt_ids)[None]` is the eager **prefill**. The non-obvious MLX idiom is the pipelined loop, mirroring mlx-lm: before reading token `t`, it enqueues step `t+1` via `next_y = step(...)` then `mx.async_eval(next_y)`; only then does `int(y.item())` force the host↔device sync for token `t`, so that blocking read overlaps the GPU compute of `t+1`.

Finally, `maybe_quantize_kv_cache(cache, cfg)` is called *after prefill* and after each step: it leaves the prompt and first `quantized_kv_start` tokens as exact fp `KVCache`, then converts each layer to a quantized cache (`to_quantized`) once `c.offset > cfg.quantized_kv_start` — keeping the precision-sensitive prefix exact while compressing the long tail.

### `silica/compiled.py`

`compiled.py` is silica's M2 experiment in compiling the decode step with `mx.compile` — the one performance lever mlx-lm leaves untouched (it runs its decode loop eagerly). The challenge: a normal KV cache is a *mutating* object, which a pure compiled graph cannot tolerate. The module sidesteps this with three deliberate decisions, all in `_decode_forward`:

- **Functional cache.** Keys and values are threaded *through* the call as plain arrays and grown by `mx.concatenate([k_cache[i], k], axis=2)`, returning fresh `new_k`/`new_v` lists. The graph stays pure.
- **Traced-array RoPE offset.** The growing position is passed as `mx.array(offset, dtype=mx.int32)` to `a.rope(q, offset=offset)`, so the per-step position is a *traced input*, not a baked-in constant that would force recompilation each step.
- **`shapeless=True`.** `make_compiled_step` wraps `fn` in `mx.compile(fn, shapeless=True)` so the per-step growth of the KV sequence dimension does not retrigger compilation.

`compiled_generate_step` does **eager prefill**, then snapshots the prompt KV into plain fp arrays before entering the functional loop. It is **greedy only** (sampling would need to be compiled too). The honest punchline, measured by `bench/compile_ablation.py`: this is **correct but perf-neutral** — `mx.compile` does not beat the existing `async_eval` baseline for batch-1 decode.

### `bench/` — the measurement harness

The `bench/` package turns raw tok/s into a defensible *achieved-bandwidth* figure of merit and gates the M1/M2 milestones.

The shared substrate is `bench/roofline.py`'s **corrected byte model**. `byte_budget(cfg, context_len, bits, kv_bits)` returns a `ByteBudget(weights, kv)` whose `total` feeds `achieved_bandwidth_gbps(tok_per_s)` and `pct_of_peak(...)`. Two non-obvious corrections live here: `_affine_bits_per_weight` charges the *on-device* quantized footprint including per-group scales **and** biases (`bits + 2*16/group_size` → ~4.5 effective bits at 4-bit/g64), and the input embedding is counted as a single-row **gather**, not a full matrix read (the double-count that had pushed achieved BW above the chip's physical peak). Every script charges KV at `eff_ctx = prompt_len + n_tokens//2` (mean KV depth over the timed window), not the unreached final context.

On top sit four **contention-robust** measurement methods, each defeating shared-bus jitter differently:

- **Interleaved paired ratio** (`compile_ablation.py`): eager vs `make_compiled_step` sampled back-to-back; headline is the median per-pair `compiled/eager` ratio so drift cancels.
- **Bracketed % usable** (`roofline_compare.py`): sandwiches each decode between a ceiling burst before and after, averaging them before the ratio.
- **Cross-model anchored ratio** (`scaling.py`): interleaved small/large decode, `%usable_large = (achieved_large/achieved_small) × anchor_usable`, sidestepping any absolute ceiling.

Finally the **quietness/reliability gates**: `compile_ablation.py` computes a `quiet` flag from ceiling `spread`, fraction of `expected`, and `os.getloadavg()`, refusing absolutes unless `--force`; `roofline_compare.py`'s `flag()` marks readings `UNRELIABLE` when `pct > 98` (ceiling contaminated → impossible) or the bracketing ceiling fell well below the quiet value (bus contended). `baseline.py` drives the silica-vs-mlx-lm go/no-go through one identical greedy+`async_eval` loop, and `eval_ppl.py`/`decode.py` are the quality and speed analogs.

---

## Design decisions & lessons learned

silica is a single-stream MLX inference engine, so its design space is narrow and its lessons sharp. Two themes dominated: an up-front correctness audit, and the discovery that on Apple Silicon you only learn the truth by *running*.

### A. Correctness is an audit, not a hope

We treated bring-up as a model-specific audit before writing decode, because the failure modes of a transformer are silent: a wrong RoPE base or a missing norm still produces fluent-looking tokens. For Qwen3-0.6B the parity-critical details were:

- **Per-head QK-RMSNorm** (`q_norm`/`k_norm` over `head_dim`, applied *before* RoPE). Miss this and attention scores drift subtly — no crash, just wrong.
- **No QKV bias** (unlike older Qwen). Adding a phantom bias is an easy copy-paste error.
- **`head_dim=128` decoupled from hidden/heads.** You cannot infer head_dim by division here; hard-coding the usual assumption breaks the projections.
- **Tied `lm_head`** (shares the embedding weights) and **RoPE theta = 1e6**, not the common 1e4.

The transferable lesson: **list the architecture's deviations from the "default transformer" first, and turn each into a parity assertion.** Equally important was *what* we asserted against. mlx-lm shares the same `mx.fast` kernels as silica, so matching it only proves backend-consistency, not correctness. The non-circular oracle was an independent HuggingFace fp32 CPU run (teacher-forced per-position argmax plus next-token logit cosine > 0.999). The lesson: **a same-backend reference is a regression guard, not a correctness proof — you need at least one independent implementation.**

Two further decisions came from asking "what does a *real* engine need?" First, an engine emits text, not token ids: we needed UTF-8-safe incremental detokenization (multibyte characters split across token boundaries), EOS handling, and chat-template formatting. An output path is part of correctness, and the streaming-decode parity test exists precisely to catch multibyte breakage. Second, when adding a quantized KV cache we found that **`mx.fast` SDPA accepts only floating-point KV** — so quantized KV needs a separate path (two `quantized_matmul` calls plus a softmax SDPA), with a `quantized_kv_start` that keeps the prefix in fp. The lesson: **fast fused kernels constrain your data types; verify the kernel's contract before you design around it.** Likewise, quantization was *selective* — body to 4-bit but the tied embed/lm_head kept at 6-bit — because the output projection is quality-sensitive.

### B. Running surfaces bugs that static review cannot

Our figure of merit was *achieved memory bandwidth*: weights (including per-group scales and biases — ~4.5 effective bits/weight at 4-bit/g64) + KV(context) + lm_head, but **not** a full embedding read, since the input embedding is a one-row gather. Getting that accounting right was the whole game, and two bugs only appeared once numbers were on screen:

- A **weight double-count** reported fp16 weight traffic as 1503 MB instead of the true 1192 MB.
- A **context mismatch** — tok/s measured at short context while KV was charged at 4096 — produced an achieved bandwidth of 457 GB/s, *above the chip's 400 GB/s spec.* Exceeding the hardware ceiling is the tell that the measurement, not the chip, is wrong.

Neither was visible in static review; both fell out of running and sanity-checking against physical limits. The lesson: **derive a hard physical bound (peak bandwidth) and treat any result that beats it as a measurement bug.**

The deeper Apple Silicon lesson is architectural: **CPU and GPU share one memory bus.** Two consequences. First, batch=1 decode is partly **CPU-dispatch-bound**, so a healthy GPU bandwidth ceiling alone does *not* certify decode numbers — you need a stable bus *and* an idle CPU. Second, any background job contaminates the bandwidth figure directly.

The robust response was methodological. Under unavoidable background load we **interleaved the two arms of a relative A/B and reported the per-pair ratio**, so drift cancels — this is what rescued both the `mx.compile` (neutral) and silica-vs-mlx-lm (within ~1.5%) comparisons. For absolute "% usable" we **bracketed** the bandwidth ceiling (measured before *and* after each decode), accepting that this breaks under non-stationary contention. And cross-model "% usable" was best obtained as a **back-to-back achieved-bandwidth ratio anchored on a known-clean value** (anchoring 4B against a clean 0.6B). The transferable lesson: **when the environment drifts, measure ratios from interleaved pairs, not absolutes — relative measurements cancel the noise that absolute ones absorb.**

The meta-point tying both halves together: the audit told us *where to look*, but only running told us *what was true* — and it was running, against physical bounds and interleaved controls, that gated M3 custom kernels out.
