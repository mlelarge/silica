"""M0 parity gate — the most important test in the project.

Per the audit (parity-1), the PRIMARY gate is exact greedy/argmax-token match,
NOT a tight logits `allclose` (cross-backend bf16 will never pass at default
tolerance). We compare against mlx-lm for a fast same-backend check AND assert a
decoded-string match.

The HF fp32 CPU oracle (transformers + torch) is the INDEPENDENT, non-circular
check: silica and mlx-lm both dispatch the same `mx.fast` Metal kernels, so a
silica/mlx-lm match only proves "same as mlx-lm". transformers is a different
implementation on a different backend, so agreement there is real evidence of
correctness. We run it teacher-forced (same input ids both sides) and compare
argmax at every position — no generation compounding.

mlx-lm checks skip unless MLX + mlx-lm + a checkpoint are present; the oracle
additionally skips unless torch + transformers are installed (`uv pip install
-e ".[oracle]"`).
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


# --------------------------------------------------------------------------- #
# Independent HF fp32 CPU oracle (the non-circular numerical check).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def oracle(parity_model_id):
    """silica (fp32, Metal) + HuggingFace (fp32, CPU) on the same checkpoint.

    silica is loaded in fp32 here so the only differences vs HF are kernel /
    accumulation order, not precision — making per-position argmax agreement a
    tight, meaningful test.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    try:
        path = resolve_model_path(parity_model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")

    silica_model, _ = load_model(path, dtype=mx.float32)
    hf_model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32)
    hf_model.eval()
    tok = load_tokenizer(path)
    return silica_model, hf_model, tok, torch


@pytest.mark.device
@pytest.mark.parametrize("prompt", PROMPTS)
def test_teacher_forced_argmax_matches_hf_fp32(oracle, prompt):
    """Independent check: silica-fp32 argmax agrees with HF-fp32 at every
    position of the same input (no generation compounding)."""
    silica_model, hf_model, tok, torch = oracle
    ids = tok.encode(prompt)

    s_logits = silica_model(mx.array(ids)[None])[0]          # (L, V)
    mx.eval(s_logits)
    with torch.no_grad():
        h_logits = hf_model(torch.tensor([ids])).logits[0]   # (L, V)

    s_arg = mx.argmax(s_logits, axis=-1).tolist()
    h_arg = h_logits.argmax(dim=-1).tolist()

    # The next-token prediction (the one that drives generation) must match.
    assert s_arg[-1] == h_arg[-1], (
        f"next-token argmax differs: silica={s_arg[-1]} hf={h_arg[-1]}"
    )
    # Teacher-forced agreement across all positions. A near-tie may flip at an
    # interior position between fp32-Metal and fp32-CPU, so require a high rate
    # rather than exact-all.
    rate = sum(a == b for a, b in zip(s_arg, h_arg)) / len(s_arg)
    assert rate >= 0.9, f"only {rate:.0%} of positions agree with HF fp32"


@pytest.mark.device
def test_last_logits_cosine_with_hf_fp32(oracle):
    """Numerical agreement (not just argmax): cosine similarity of the
    next-token logit vectors should be ~1.0 against the independent HF oracle."""
    import numpy as np

    silica_model, hf_model, tok, torch = oracle
    ids = tok.encode(PROMPTS[0])

    s_logits = silica_model(mx.array(ids)[None])[0, -1]
    mx.eval(s_logits)
    with torch.no_grad():
        h_logits = hf_model(torch.tensor([ids])).logits[0, -1]

    s = np.asarray(s_logits.tolist(), dtype=np.float64)
    h = h_logits.detach().to(torch.float64).numpy()
    cos = float(np.dot(s, h) / (np.linalg.norm(s) * np.linalg.norm(h)))
    assert cos > 0.999, f"logit cosine vs HF fp32 too low: {cos:.5f}"
