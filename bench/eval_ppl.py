"""Perplexity harness + quant-bits ablation (PLAN §5 M1, §6).

The quality analog of bench/decode.py: a reusable perplexity measurement on a
fixed, pinned corpus (bench/data/corpus.txt — no download, fully reproducible),
reported as a column alongside a quant-bits sweep with a %-regression vs the
fp16 baseline. This is what gates M1 quantization changes on quality, not just
speed (audit comp-5).

Non-overlapping windows; the absolute PPL is a relative metric for the ablation
(consistent across bits), not a leaderboard number. Logits are upcast to fp32
for a numerically stable logsumexp regardless of the model's compute dtype.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx

from silica.config import QuantConfig
from silica.weights import load_model, resolve_model_path
from silica.generate import load_tokenizer

DEFAULT_CORPUS = Path(__file__).parent / "data" / "corpus.txt"


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> str:
    return Path(path).read_text(encoding="utf-8")


def token_nll(model, ids: list[int], max_seq: int = 1024) -> tuple[float, int]:
    """Return (sum negative-log-likelihood, num scored tokens)."""
    total, n = 0.0, 0
    for start in range(0, len(ids) - 1, max_seq):
        chunk = ids[start:start + max_seq + 1]
        if len(chunk) < 2:
            break
        x = mx.array(chunk[:-1])[None]
        y = mx.array(chunk[1:])
        logits = model(x)[0].astype(mx.float32)                 # (L, V)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ll = mx.take_along_axis(logp, y[:, None], axis=-1)[:, 0]  # (L,)
        mx.eval(ll)
        total += float((-ll).sum().item())
        n += int(y.size)
    return total, n


def perplexity(model, tokenizer, text: str, max_seq: int = 1024) -> tuple[float, int]:
    ids = tokenizer.encode(text)
    total, n = token_nll(model, ids, max_seq)
    return math.exp(total / n), n


def _free():
    import gc
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def ablate(model_id, text, bit_settings, group_size, embed_bits, max_seq):
    tokenizer = load_tokenizer(resolve_model_path(model_id))
    base = None
    print(f"{'config':<14}{'PPL':>10}{'Δ vs fp16':>12}{'tokens':>9}")
    print("-" * 45)
    for bits in bit_settings:
        quant = None if bits is None else QuantConfig(
            bits=bits, group_size=group_size, embed_bits=embed_bits)
        model, _ = load_model(model_id, quant=quant, dtype=mx.bfloat16)
        ppl, n = perplexity(model, tokenizer, text, max_seq)
        if base is None:
            base = ppl
        delta = "" if bits is None else f"{100 * (ppl / base - 1):+.1f}%"
        label = "fp16" if bits is None else f"{bits}-bit/g{group_size}"
        print(f"{label:<14}{ppl:>10.3f}{delta:>12}{n:>9}")
        del model
        _free()


def main():
    ap = argparse.ArgumentParser(description="silica perplexity / quant-quality ablation")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--bits", type=int, default=None, help="single run at this many bits (fp16 if unset)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--embed-bits", type=int, default=6, help="bits for tied embed/lm_head (selective)")
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--ablate", action="store_true", help="sweep fp16 / 8-bit / 4-bit")
    args = ap.parse_args()

    text = load_corpus(args.corpus)

    if args.ablate:
        ablate(args.model, text, [None, 8, 4], args.group_size, args.embed_bits, args.max_seq)
        return

    quant = None if args.bits is None else QuantConfig(
        bits=args.bits, group_size=args.group_size, embed_bits=args.embed_bits)
    model, _ = load_model(args.model, quant=quant, dtype=mx.bfloat16)
    tokenizer = load_tokenizer(resolve_model_path(args.model))
    ppl, n = perplexity(model, tokenizer, text, args.max_seq)
    print(f"model    : {args.model}  ({'fp16' if args.bits is None else f'{args.bits}-bit'})")
    print(f"corpus   : {args.corpus}  ({n} scored tokens)")
    print(f"PPL      : {ppl:.3f}")


if __name__ == "__main__":
    main()
