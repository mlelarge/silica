"""M1 quantization tests.

Verifies the audit-driven SELECTIVE quantization policy (keep the tied
embedding/lm_head at higher precision), that a 4-bit model loads and generates
clean text, and that quantization perturbs but does not break the logits
(cosine vs fp16) or the corpus perplexity (bounded regression).

Skips unless MLX + a Qwen3 checkpoint are present.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from silica.weights import load_model, resolve_model_path
from silica.config import QuantConfig, GenConfig
from silica.generate import load_tokenizer, generate
from bench.eval_ppl import perplexity, load_corpus


@pytest.fixture(scope="module")
def quant_models(parity_model_id):
    try:
        path = resolve_model_path(parity_model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")
    fp16, _ = load_model(path, dtype=mx.bfloat16)
    q4, _ = load_model(
        path, quant=QuantConfig(bits=4, group_size=64, embed_bits=6), dtype=mx.bfloat16
    )
    tok = load_tokenizer(path)
    return fp16, q4, tok


@pytest.mark.device
def test_selective_quant_keeps_embed_higher_precision(quant_models):
    """Body Linears -> 4-bit; tied embedding/lm_head -> 6-bit (not 4)."""
    _, q4, _ = quant_models
    embed = q4.model.embed_tokens
    body = q4.model.layers[0].self_attn.q_proj
    assert getattr(embed, "bits", None) == 6, "embedding must stay higher-precision"
    assert getattr(body, "bits", None) == 4, "body projections must be 4-bit"


@pytest.mark.device
def test_stronger_recipe_protects_sensitive_proj(parity_model_id):
    """high_bits_proj keeps protected projections (down_proj) at embed_bits while
    the rest of the body stays at the base bits (the stronger 4-bit recipe)."""
    try:
        path = resolve_model_path(parity_model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")
    model, _ = load_model(path, quant=QuantConfig(
        bits=4, group_size=32, embed_bits=6, high_bits_proj=("down_proj",)), dtype=mx.bfloat16)
    mlp = model.model.layers[0].mlp
    assert getattr(mlp.down_proj, "bits", None) == 6, "down_proj must be protected"
    assert getattr(mlp.gate_proj, "bits", None) == 4, "gate_proj stays at base bits"


@pytest.mark.device
def test_no_layers_skipped_for_qwen3_06b(quant_models):
    """All body projections divide group_size=64, so none fall back to fp."""
    _, q4, _ = quant_models
    mlp = q4.model.layers[0].mlp
    for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        assert getattr(proj, "bits", None) == 4


@pytest.mark.device
def test_quant_next_token_cosine_with_fp16(quant_models):
    """4-bit perturbs but does not break: next-token logit cosine vs fp16 ~1."""
    import numpy as np

    fp16, q4, tok = quant_models
    ids = tok.encode("The capital of France is")
    a = fp16(mx.array(ids)[None])[0, -1].astype(mx.float32)
    b = q4(mx.array(ids)[None])[0, -1].astype(mx.float32)
    mx.eval(a, b)
    a = np.asarray(a.tolist(), dtype=np.float64)
    b = np.asarray(b.tolist(), dtype=np.float64)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.95, f"4-bit logits diverged from fp16 (cosine {cos:.4f})"


@pytest.mark.device
def test_quant_generates_clean_text(quant_models):
    _, q4, tok = quant_models
    out = generate(q4, tok, "Name three primary colors.",
                   GenConfig(max_tokens=16, temperature=0.0), stream=False)
    assert out.strip(), "quantized model produced empty output"
    assert "�" not in out


@pytest.mark.device
def test_quant_perplexity_regression_bounded(quant_models):
    """4-bit corpus PPL must stay within a loose budget of fp16 (quality gate)."""
    fp16, q4, tok = quant_models
    text = load_corpus()
    ppl_fp16, _ = perplexity(fp16, tok, text, max_seq=512)
    ppl_q4, _ = perplexity(q4, tok, text, max_seq=512)
    assert ppl_q4 < ppl_fp16 * 1.5, (
        f"4-bit PPL {ppl_q4:.2f} regressed >50% vs fp16 {ppl_fp16:.2f}"
    )


# ---- quantized KV cache (the separate quantized-SDPA path) ----------------- #


@pytest.mark.device
def test_quantized_kv_8bit_sdpa_path_close_to_fp(quant_models):
    """The quantized_matmul SDPA path computes ~the same attention as fp KV:
    next-token logit cosine ~1 and argmax agree on an unambiguous context.
    Uses quantize-from-step-0 caches to exercise the quantized path on prefill."""
    import numpy as np
    from silica.cache import make_cache

    fp16, _, tok = quant_models
    ids = mx.array(tok.encode("List the first five prime numbers: 2, 3, 5,"))[None]

    a = fp16(ids, cache=make_cache(len(fp16.layers)))[0, -1].astype(mx.float32)
    b = fp16(ids, cache=make_cache(len(fp16.layers), kv_bits=8))[0, -1].astype(mx.float32)
    mx.eval(a, b)
    av = np.asarray(a.tolist(), dtype=np.float64)
    bv = np.asarray(b.tolist(), dtype=np.float64)
    cos = float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))
    assert cos > 0.99, f"8-bit KV SDPA diverged from fp (cosine {cos:.4f})"
    assert int(av.argmax()) == int(bv.argmax())


@pytest.mark.device
def test_quantized_kv_start_keeps_prefill_exact(quant_models):
    """With quantized_kv_start=0 the prompt stays fp, so the FIRST generated
    token matches fp KV exactly (only the long tail is quantized)."""
    fp16, _, tok = quant_models
    prompt = "List the first five prime numbers:"
    base = GenConfig(max_tokens=1, temperature=0.0)
    g_fp = generate(fp16, tok, prompt, base, stream=False)
    g_q8 = generate(fp16, tok, prompt,
                    GenConfig(max_tokens=1, temperature=0.0, kv_bits=8), stream=False)
    assert g_q8 == g_fp


@pytest.mark.device
def test_quantized_kv_4bit_generates_clean_text(quant_models):
    """4-bit KV cache must still produce clean (no replacement-char) text."""
    fp16, _, tok = quant_models
    out = generate(fp16, tok, "Name three primary colors.",
                   GenConfig(max_tokens=16, temperature=0.0, kv_bits=4, kv_group_size=32),
                   stream=False)
    assert out.strip() and "�" not in out
