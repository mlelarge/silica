"""Prefix (prompt) cache tests — the single-stream slice of radix caching.

The contract: reusing a shared prefix and prefilling only the suffix must be
equivalent to a full prefill (causal attention makes the prefix KV
position-identical). Device-gated.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from silica.weights import load_model, resolve_model_path
from silica.generate import load_tokenizer, generate
from silica.config import GenConfig
from silica.cache import make_cache, PrefixCache


@pytest.fixture(scope="module")
def model_tok(parity_model_id):
    try:
        path = resolve_model_path(parity_model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")
    model, _ = load_model(path, dtype=mx.bfloat16)
    return model, load_tokenizer(path)


def test_reuse_len_leaves_one_token():
    pc = PrefixCache()
    pc.tokens = [1, 2, 3, 4, 5]
    assert pc.reuse_len([1, 2, 3, 9, 9]) == 3          # diverges at index 3
    assert pc.reuse_len([1, 2, 3, 4, 5]) == 4          # full match -> leave 1
    assert pc.reuse_len([9, 9]) == 0                   # no shared prefix


@pytest.mark.device
def test_prefix_reuse_logits_match_full_prefill(model_tok):
    model, tok = model_tok
    ids = tok.encode("The capital of France is Paris, and the capital of Italy is")
    n_layers = len(model.layers)

    cold = model(mx.array(ids)[None], cache=make_cache(n_layers))[:, -1, :]

    k = len(ids) // 2                                  # snapshot the first half
    warm_cache = make_cache(n_layers)
    model(mx.array(ids[:k])[None], cache=warm_cache)
    pc = PrefixCache()
    pc.update(ids[:k], warm_cache)
    reuse = pc.reuse_len(ids)
    assert reuse == min(k, len(ids) - 1)
    restored = pc.restore(n_layers, reuse)
    warm = model(mx.array(ids[reuse:])[None], cache=restored)[:, -1, :]

    mx.eval(cold, warm)
    # Mathematically equal; the reuse path uses different SDPA tiling, so bf16
    # logits aren't bit-identical (argmax + cosine are the robust checks).
    assert int(mx.argmax(cold, -1).item()) == int(mx.argmax(warm, -1).item())
    a = cold[0].astype(mx.float32)
    b = warm[0].astype(mx.float32)
    cos = float((a * b).sum().item() / ((a * a).sum().item() ** 0.5 * (b * b).sum().item() ** 0.5))
    assert cos > 0.999, f"prefix reuse diverged from full prefill (cosine {cos:.5f})"


@pytest.mark.device
def test_generate_with_prefix_cache_matches_cold(model_tok):
    """End-to-end: warming then reusing a prefix yields identical greedy text."""
    model, tok = model_tok
    cfg = GenConfig(max_tokens=20, temperature=0.0, use_chat_template=False)
    prompt = "The first three prime numbers are"

    cold = generate(model, tok, prompt, cfg, stream=False)
    pc = PrefixCache()
    generate(model, tok, prompt, cfg, stream=False, prefix_cache=pc)      # populate
    warm = generate(model, tok, prompt, cfg, stream=False, prefix_cache=pc)  # reuse
    assert warm == cold
    assert pc.reuse_len(tok.encode(prompt)) > 0          # something was actually cached
