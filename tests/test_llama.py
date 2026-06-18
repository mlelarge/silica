"""Generality datapoint: a second architecture (Llama-3.2-1B) through the SAME
silica runtime, dispatched by the model registry.

Llama differs from Qwen3 (no QK-norm, llama3 RoPE scaling), so an exact
next-token match vs mlx-lm's independent Llama implementation proves the new
~40-line attention block + registry are correct, with the cache / attention
dispatch / sampler / detokenizer / generate loop all reused unchanged.

Device-gated; skips if the (ungated) checkpoint or mlx-lm is unavailable.
"""

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from silica.weights import load_model, resolve_model_path
from silica.generate import load_tokenizer, generate
from silica.config import GenConfig
from silica.models import model_class
from silica.models.llama import LlamaForCausalLM

LLAMA = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture(scope="module")
def llama():
    try:
        path = resolve_model_path(LLAMA)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{LLAMA} unavailable: {e}")
    silica_model, cfg = load_model(path, dtype=mx.bfloat16)
    ref_model, _ = mlx_lm.load(str(path))
    tok = load_tokenizer(path)
    return silica_model, ref_model, tok, cfg


@pytest.mark.device
def test_llama_registry_dispatch(llama):
    _, _, _, cfg = llama
    assert cfg.architectures == ("LlamaForCausalLM",)
    assert model_class(cfg.architectures) is LlamaForCausalLM


@pytest.mark.device
@pytest.mark.parametrize("prompt", ["The capital of France is", "Once upon a time,"])
def test_llama_argmax_matches_mlx_lm(llama, prompt):
    silica_model, ref_model, tok, _ = llama
    ids = mx.array(tok.encode(prompt))[None]
    s = silica_model(ids)[:, -1, :]
    r = ref_model(ids)[:, -1, :]
    mx.eval(s, r)
    assert int(mx.argmax(s, axis=-1).item()) == int(mx.argmax(r, axis=-1).item())
    s5 = set(mx.argsort(s[0])[-5:].tolist())
    r5 = set(mx.argsort(r[0])[-5:].tolist())
    assert len(s5 & r5) >= 4


@pytest.mark.device
def test_llama_generates_clean_text(llama):
    silica_model, _, tok, _ = llama
    out = generate(silica_model, tok, "List three colors:",
                   GenConfig(max_tokens=16, temperature=0.0), stream=False)
    assert out.strip() and "�" not in out
