"""M2: correctness of the mx.compile decode path.

This is a parity check (compiled tokens == eager tokens), NOT a timing test, so
it is valid regardless of machine load. The benchmark that decides whether
compile is worth it lives in bench/ and needs a quiet machine.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from silica.weights import load_model, resolve_model_path
from silica.generate import load_tokenizer, generate_step
from silica.compiled import compiled_generate_step
from silica.config import GenConfig, QuantConfig

PROMPT = "The capital of France is"


def _eager_vs_compiled(model, tok, max_tokens=24):
    ids = tok.encode(PROMPT)
    eos = tuple(model.config.eos_token_ids)
    cfg = GenConfig(max_tokens=max_tokens, temperature=0.0)
    eager = list(generate_step(model, ids, cfg, eos))
    compiled = list(compiled_generate_step(model, ids, cfg, eos))
    return eager, compiled


@pytest.fixture(scope="module")
def model_path(parity_model_id):
    try:
        return resolve_model_path(parity_model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")


@pytest.mark.device
def test_compiled_decode_matches_eager_fp16(model_path):
    model, _ = load_model(model_path, dtype=mx.bfloat16)
    tok = load_tokenizer(model_path)
    eager, compiled = _eager_vs_compiled(model, tok)
    assert compiled == eager, f"\neager:    {eager}\ncompiled: {compiled}"


@pytest.mark.device
def test_compiled_decode_matches_eager_quantized(model_path):
    model, _ = load_model(model_path, quant=QuantConfig(bits=4, group_size=64, embed_bits=6),
                          dtype=mx.bfloat16)
    tok = load_tokenizer(model_path)
    eager, compiled = _eager_vs_compiled(model, tok)
    assert compiled == eager, f"\neager:    {eager}\ncompiled: {compiled}"
