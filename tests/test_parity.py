"""M0 parity gate — the most important test in the project.

Per the audit (parity-1), the PRIMARY gate is exact greedy/argmax-token match,
NOT a tight logits `allclose` (cross-backend bf16 will never pass at default
tolerance). We compare against mlx-lm for a fast same-backend check AND assert a
decoded-string match. An independent HF fp32 CPU oracle is a TODO below (it is
the only non-circular numerical check, since silica and mlx-lm share mx.fast
kernels).

Skips unless MLX + mlx-lm + a Qwen3 checkpoint are present.
"""

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from silica.weights import load_model, resolve_model_path
from silica.generate import load_tokenizer, generate, _encode_prompt
from silica.config import GenConfig

PROMPTS = [
    "The capital of France is",
    "Café crème costs €3.50 — déjà vu? 🤔",   # multibyte / emoji (detok stress)
]


@pytest.fixture(scope="module")
def models(parity_model_id):
    try:
        path = resolve_model_path(parity_model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")
    silica_model, _ = load_model(path, dtype=mx.bfloat16)
    ref_model, ref_tok = mlx_lm.load(str(path))
    tok = load_tokenizer(path)
    return silica_model, ref_model, tok, ref_tok


@pytest.mark.device
@pytest.mark.parametrize("prompt", PROMPTS)
def test_argmax_logits_match_mlx_lm(models, prompt):
    """Hard gate: next-token argmax agrees with mlx-lm on the prompt."""
    silica_model, ref_model, tok, _ = models
    ids = mx.array(tok.encode(prompt))[None]

    s_logits = silica_model(ids)[:, -1, :]
    r_logits = ref_model(ids)[:, -1, :]
    mx.eval(s_logits, r_logits)

    assert int(mx.argmax(s_logits, axis=-1).item()) == int(mx.argmax(r_logits, axis=-1).item())
    # top-5 overlap as a softer robustness signal
    s_top = set(mx.argsort(s_logits[0])[-5:].tolist())
    r_top = set(mx.argsort(r_logits[0])[-5:].tolist())
    assert len(s_top & r_top) >= 4


@pytest.mark.device
def test_greedy_sequence_matches_mlx_lm(models):
    """Greedy-decoded token sequence matches mlx-lm (the M0 acceptance gate)."""
    silica_model, ref_model, tok, ref_tok = models
    prompt = PROMPTS[0]
    cfg = GenConfig(max_tokens=24, temperature=0.0, use_chat_template=False)

    text = generate(silica_model, tok, prompt, cfg, stream=False)
    ref = mlx_lm.generate(ref_model, ref_tok, prompt, max_tokens=24, verbose=False)

    # Compare on the generated continuation (string-level — exercises detok too).
    assert text.strip()[:40] == ref.strip()[:40]


@pytest.mark.device
def test_string_decode_handles_multibyte(models):
    """Detokenizer must not corrupt multibyte characters (audit comp-1)."""
    silica_model, _, tok, _ = models
    cfg = GenConfig(max_tokens=16, temperature=0.0)
    out = generate(silica_model, tok, PROMPTS[1], cfg, stream=False)
    assert "�" not in out  # no stray replacement chars


# TODO(M0): add an HF fp32 CPU oracle (transformers + torch) as the independent,
# non-circular numerical check. Skip if torch is unavailable.
