# silica — Code Audit

A post-implementation code review by 8 expert reviewers (model/numerics, KV cache,
generation+async, quantization, output+sampling, bench methodology, code quality,
test coverage), each reading the actual source. Every behavioral finding was then
**adversarially re-verified against the code** — which refuted one false positive
(a misread "KV growth-size bug") and confirmed the rest. 67 raw findings; the
consequential confirmed ones are below.

> **Resolution (follow-up):** all Tier 1 + Tier 2 findings (#1–#8) are **FIXED**
> with matching tests (commits `7e94607`, `ac92a73`; suite 49/49). The
> partially-correct #9/#10 and the Tier 3 test gaps / Tier 4 nits remain open.

**Verdict.** No correctness bug on the *proven* happy path — the model numerics,
GQA, RoPE/QK-norm order, attention scale, quantized-SDPA math, and the
load-before-quantize ordering all verified correct. The real issues are in the
**peripheral correctness paths the parity gate doesn't exercise**: the sampler's
global-RNG side effect, the detokenizer's stop-sequence and end-of-stream
handling, two robustness gaps in quantization config, and a wrong byte/quant
pairing in one (unused-by-the-report) bench. None invalidate the M0–M4 results
(those used `baseline.py`/`compile_ablation.py`/`scaling.py`, all correct).

---

## Tier 1 — confirmed correctness bugs (user-facing)

**1. Sampler reseeds the *global* MLX RNG every call → identical generations.**
`silica/sample.py:21` — `make_sampler` runs `mx.random.seed(cfg.seed)`, and
`GenConfig.seed` defaults to `0` (`config.py`). Since `generate_step` builds a
sampler per generation, **two back-to-back sampled (temperature>0) generations of
the same prompt produce byte-identical output**, and the call silently clobbers
any other process RNG state. A fixed seed should make *one* run reproducible, not
freeze *all* runs. → *Fix:* default `seed=None` (let the global stream advance),
or thread an explicit `mx.random.key` into `mx.random.categorical(..., key=...)`.
(Greedy M0 is unaffected — this bites the M1.5 sampler.)

**2. Stop-sequence spanning tokens leaks its prefix and mis-slices.**
`silica/detokenize.py:42-53` — `add_token` streams `decoded[_emitted:]` *before*
checking for a stop substring. If a stop string arrives across two tokens (`" EN"`
then `"D"` → `"END"`), the prefix `" EN"` is already emitted; on the completing
token `_first_stop_index` returns `cut < _emitted`, so `decoded[_emitted:cut]` is
an empty/negative slice and the partial stop text stays in the output. → *Fix:*
withhold a trailing window of `max(len(s) for s in stop) - 1` chars until
resolved, and clamp to `decoded[min(_emitted, cut):cut]`.

**3. Final partial multibyte character is dropped at end of generation.**
`silica/detokenize.py:38` — on the U+FFFD-withhold path `add_token` returns `""`
and never updates `_text`/`_emitted`, and there is no `flush`. If generation ends
mid-character (max_tokens hit, or EOS right after a partial emoji/accent), the
withheld bytes are **never emitted by anyone** — the last glyph is silently lost;
`.text` is also stale on that path. → *Fix:* add `finalize()` that emits the
remaining `decode(_ids)[_emitted:]` (replacement chars included) and call it after
the `generate()` loop.

**4. `bench/decode.py` charges a quantized byte budget against an fp16 model.**
`bench/decode.py:43,64` — `run()` calls `load_model(..., dtype=bf16)` with **no
`quant=`**, so the model is always fp16, yet `--bits N` is passed into
`byte_budget(..., bits=N)`, scaling the weight bytes down to ~4.5 bpw. So a
`--bits 4` invocation reports the *fp16* tok/s divided by a *4-bit* byte count —
a meaningless achieved-BW/%-peak. → *Fix:* pass a `QuantConfig` when `bits` is set
(as `baseline.py` does). *Not used for any reported result* (those came from
`baseline.py`/`scaling.py`), but the tool is wrong as written.

---

## Tier 2 — confirmed robustness gaps

**5. Quantizing embed/lm_head bypasses the group-size divisibility guard.**
`silica/weights.py:71-77` — the embed/lm_head branch returns its quant dict
*before* the `w.shape[-1] % group_size` check that gracefully skips every other
Linear. On a model whose `hidden_size` isn't divisible by `group_size`, `embed`
hard-crashes while every other layer degrades gracefully. (Qwen3-0.6B hidden=1024
is divisible, so untriggered.) → *Fix:* apply the same guard to the embed branch.

**6. `QuantConfig` doesn't validate `embed_bits`.**
`silica/config.py:108-112` — `bits` and `group_size` are validated, but
`embed_bits` (forwarded as MLX `bits`) isn't, so `embed_bits=7` constructs fine
then crashes deep in `nn.quantize`. → *Fix:* validate `embed_bits ∈ {2,3,4,5,6,8}`
in `__post_init__`.

**7. `compiled_generate_step` recompiles on every call.**
`silica/compiled.py:74` — it calls `make_compiled_step(model)` (a fresh closure →
`mx.compile`) *inside* the function, so each generation pays a full re-trace,
undercutting the perf the module exists to provide. → *Fix:* build the compiled
step once and cache it on the model. *(The M2 compile ablation is unaffected —
`compile_ablation.py` builds the step once and reuses it, so the "neutral" verdict
stands.)*

**8. `cross_engine.py` llama-bench JSON parsing is fragile.**
`bench/cross_engine.py:36-46` — `json.loads(out.stdout)` and `r["avg_ts"]` have no
guards; a non-JSON line, a renamed key, or zero tg-rows raises an opaque traceback
instead of a clear error. → *Fix:* wrap in try/except, use `.get`, and check for
an empty tg list.

---

## Tier 2b — confirmed-but-nuanced (partially-correct on verify)

**9. `eval_ppl` non-overlapping windows inflate absolute PPL.**
`bench/eval_ppl.py:36` — each window starts cache-free, so the first target token
of every chunk has ~no context, inflating NLL. Verified: this is real for
*absolute* PPL (already flagged "not a leaderboard number"), **but does not bias
the quant ablation delta** (identical windows across bits), so the M1/quality
conclusions hold. → *Optional:* sliding window scoring only the final `stride`
tokens for a cleaner absolute number.

**10. `measure_peak_bandwidth` is a single burst-mean; the quietness gate uses 2
samples.** `bench/baseline.py:31` + `compile_ablation.py:80` — one scheduling
hiccup can flip QUIET/CONTENDED. → *Fix:* per-iteration median (+IQR) and ≥3
ceiling samples.

---

## Tier 3 — test gaps (no bug, but unguarded)

The parity gate proves the happy path; these paths are **unexercised**, and #2/#3
above show that's where the bugs hid:

- **Non-greedy sampler** (temp/top-k/top-p/min-p) — pure-Python, no device needed.
- **Detokenizer stop-sequences + finish/flush** — directly the buggy code above.
- **Multi-chunk KV growth** (offset crossing the 256 step) — the buffer-grow path.
- **Chunked prefill at offset>0** + the offset-aware `causal_additive_mask` (a
  correct-by-inspection but never-called branch).
- **Quantized-KV under GQA + multi-chunk**, and `quantized_kv_start>0` conversion.
- **Error paths** (empty prompt, `max_tokens<=0`, missing config keys).
- **Threshold rigor:** `cosine>0.95`, `PPL<1.5×`, top-5≥4, 40-char prefix — loose
  enough to mask real regressions; tighten where the clean value is known.

A device-independent `quantized_sdpa` / KV-cache unit test (tiny synthetic tensors,
`n_kv=2,n_q=4` to force GQA) would cover much of this without hardware gating.

## Tier 4 — maintainability (low/nit)

Dead/inert code: `KVCache.state` (returns padded buffer, unused), the
`ModelConfig.__post_init__` head_dim divergence branch (computed then discarded),
the `quantize_norms` flag (never read), `RotatingKVCache.__init__` unconditionally
raises (so `make_cache` can't produce one). Annotations: `time_decode` is typed
`-> float` but returns a tuple; `generate`/`model` params under-typed. Misnamed
`n_steps` in `QuantizedKVCache` (it's the total padded length). `O(n²)` whole-buffer
re-decode in the detokenizer (fine at these lengths). `scaling.py` verdict uses a
hardcoded anchor constant.

## Strengths (verified)

- Correct Qwen3 ordering (per-head QK-norm over `head_dim`, before RoPE, before
  cache update) — the classic parity-breaker, done right.
- Attention scale applied exactly once across both fp and quantized paths.
- GQA correct in both SDPA paths; the 2D-mask-over-5D-scores broadcast is valid.
- In-place KV slice-assignment growth is sound under MLX value semantics.
- `quantized_sdpa` faithfully mirrors mlx-lm (scale, mask, precise softmax,
  transpose flags); load-fp-before-quantize and tied-lm_head pop are correct.
- The independent, non-circular HF fp32 oracle and the pure-Python config/roofline
  tests are genuinely good practice.

## Recommended fix order

1. ~~**#1 sampler RNG** and **#2/#3 detokenizer**~~ — **DONE** (`7e94607`): per-sampler
   key, detokenizer hold-back + `finalize()`; tests `test_sample.py`, `test_detokenize.py`.
2. ~~**#5/#6 quant guards** + **#8 cross_engine parse**~~ — **DONE** (`ac92a73`).
3. ~~**#4 decode.py** and **#7 compiled recompile**~~ — **DONE** (`ac92a73`).
4. **Open:** backfill Tier 3 tests (multi-chunk KV growth, offset>0 prefill,
   quantized-KV under GQA, error paths) and the partially-correct #9/#10
   (sliding-window PPL, median peak-bandwidth gate).
5. **Open:** Tier 4 cleanup opportunistically.
