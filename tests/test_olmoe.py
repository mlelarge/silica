"""MoE generality: OLMoE-1B-7B (64 experts, top-8) through the same runtime.

OLMoE differs from the dense models in three ways silica must get right: a
Mixture-of-Experts MLP (router + gathered experts via mx.gather_mm), QK-norm over
the FULL q/k projection (not per-head), and an untied lm_head. Exact next-token
parity vs mlx-lm's independent OLMoE proves the router, SwitchGLU, weight-stacking
`sanitize`, and attention are correct — runtime/cache/sampler reused unchanged.

Device-gated; skips if the checkpoint or mlx-lm is unavailable.
"""

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from silica.weights import load_model, resolve_model_path
from silica.generate import load_tokenizer, generate
from silica.config import GenConfig
from silica.models import model_class
from silica.models.olmoe import OlmoeForCausalLM

OLMOE = "allenai/OLMoE-1B-7B-0125-Instruct"


@pytest.fixture(scope="module")
def olmoe():
    try:
        path = resolve_model_path(OLMOE)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{OLMOE} unavailable: {e}")
    silica_model, cfg = load_model(path, dtype=mx.bfloat16)
    ref_model, _ = mlx_lm.load(str(path))
    tok = load_tokenizer(path)
    return silica_model, ref_model, tok, cfg


@pytest.mark.device
def test_olmoe_is_moe_and_dispatches(olmoe):
    _, _, _, cfg = olmoe
    assert cfg.is_moe and cfg.num_experts == 64 and cfg.num_experts_per_tok == 8
    assert model_class(cfg.architectures) is OlmoeForCausalLM


@pytest.mark.device
@pytest.mark.parametrize("prompt", ["The capital of France is", "Once upon a time"])
def test_olmoe_argmax_matches_mlx_lm(olmoe, prompt):
    silica_model, ref_model, tok, _ = olmoe
    ids = mx.array(tok.encode(prompt))[None]
    s = silica_model(ids)[:, -1, :]
    r = ref_model(ids)[:, -1, :]
    mx.eval(s, r)
    assert int(mx.argmax(s, -1).item()) == int(mx.argmax(r, -1).item())
    s5 = set(mx.argsort(s[0])[-5:].tolist())
    r5 = set(mx.argsort(r[0])[-5:].tolist())
    assert len(s5 & r5) >= 4


@pytest.mark.device
def test_olmoe_generates_clean_text(olmoe):
    silica_model, _, tok, _ = olmoe
    out = generate(silica_model, tok, "Name three colors:",
                   GenConfig(max_tokens=16, temperature=0.0), stream=False)
    assert out.strip() and "�" not in out
