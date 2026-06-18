"""Generation: chat template -> prefill -> decode loop -> streamed text.

The decode loop mirrors mlx-lm's `generate_step`: enqueue step t+1 with
`mx.async_eval` *before* reading token t, so the host↔device sync for token t
overlaps the GPU compute of t+1 (audit msgl-4 / mlx-3). The single `.item()`
read happens after the next step is already in flight.
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx

from .config import GenConfig, ModelConfig
from .cache import make_cache, KVCache
from .sample import make_sampler
from .detokenize import IncrementalDetokenizer


def maybe_quantize_kv_cache(cache, cfg: GenConfig) -> None:
    """Convert fp layer caches to quantized once past `quantized_kv_start`.

    Mirrors mlx-lm: the prompt (prefill) and the first `quantized_kv_start`
    tokens stay fp — exact — then KV is quantized in place for the long tail.
    """
    if cfg.kv_bits is None:
        return
    for i, c in enumerate(cache):
        if isinstance(c, KVCache) and c.offset > cfg.quantized_kv_start:
            cache[i] = c.to_quantized(group_size=cfg.kv_group_size, bits=cfg.kv_bits)


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path))


def _encode_prompt(tokenizer, prompt: str, cfg: GenConfig) -> list[int]:
    if cfg.use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        # Newer transformers may return a BatchEncoding (UserDict), not a list.
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return list(ids)
    return list(tokenizer.encode(prompt))


def generate_step(
    model,
    prompt_ids: list[int],
    cfg: GenConfig,
    eos_ids: tuple[int, ...],
) -> Iterator[int]:
    """Yield generated token ids one at a time (greedy/sampled)."""
    sampler = make_sampler(cfg)
    cache = make_cache(len(model.layers))     # fp; quantized after prefill if kv_bits

    def step(tokens: mx.array) -> mx.array:
        logits = model(tokens, cache=cache)[:, -1, :]
        return sampler(logits)

    y = step(mx.array(prompt_ids)[None])      # prefill -> first token, shape (1,)
    maybe_quantize_kv_cache(cache, cfg)
    mx.async_eval(y)

    n = 0
    while True:
        if n + 1 != cfg.max_tokens:
            next_y = step(y.reshape(1, 1))    # enqueue step t+1 ...
            maybe_quantize_kv_cache(cache, cfg)
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)

        token = int(y.item())                 # ... then read token t
        yield token
        n += 1
        if token in eos_ids or n >= cfg.max_tokens:
            break
        y = next_y


def generate(
    model,
    tokenizer,
    prompt: str,
    cfg: GenConfig | None = None,
    *,
    stream: bool = True,
) -> str:
    """Generate text for `prompt`. Returns the full decoded string."""
    cfg = cfg or GenConfig()
    mcfg: ModelConfig = model.config
    eos_ids = set(mcfg.eos_token_ids)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    prompt_ids = _encode_prompt(tokenizer, prompt, cfg)
    detok = IncrementalDetokenizer(tokenizer, stop=cfg.stop)

    out = []
    for token in generate_step(model, prompt_ids, cfg, tuple(eos_ids)):
        if token in eos_ids:
            break
        segment = detok.add_token(token)
        if segment:
            out.append(segment)
            if stream:
                print(segment, end="", flush=True)
        if detok.finished:
            break
    flush = detok.finalize()   # emit held-back window + any trailing partial char
    if flush:
        out.append(flush)
        if stream:
            print(flush, end="", flush=True)
    if stream:
        print()
    return "".join(out)


def main():
    import argparse

    from .weights import load_model, resolve_model_path

    ap = argparse.ArgumentParser(description="silica greedy generation (M0)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="Give me a short introduction to large language models.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--kv-bits", type=int, default=None, help="quantize KV cache to N bits")
    args = ap.parse_args()

    model, _ = load_model(args.model)
    tokenizer = load_tokenizer(resolve_model_path(args.model))
    cfg = GenConfig(max_tokens=args.max_tokens, temperature=args.temp, kv_bits=args.kv_bits)
    generate(model, tokenizer, args.prompt, cfg)


if __name__ == "__main__":
    main()
