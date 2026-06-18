"""Pre-quantized checkpoint loading: load an already-quantized mlx-community
model directly (packed weight/scales/biases) instead of quantizing fp at load.

Validates the `config["quantization"]` -> nn.quantize(class_predicate) path that
matches silica's module structure to the packed tensors. Exact next-token parity
vs mlx-lm loading the same checkpoint proves the quantized modules were
reconstructed and loaded correctly. Uses the small dense 4-bit model (cached);
device-gated.
"""

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")
import mlx.nn as nn

from silica.weights import load_model, resolve_model_path
from silica.generate import load_tokenizer

PREQUANT = "mlx-community/Qwen3-0.6B-4bit"


@pytest.fixture(scope="module")
def prequant():
    try:
        resolve_model_path(PREQUANT)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{PREQUANT} unavailable: {e}")
    silica_model, cfg = load_model(PREQUANT)
    ref_model, _ = mlx_lm.load(PREQUANT)
    tok = load_tokenizer(resolve_model_path(PREQUANT))
    return silica_model, ref_model, tok, cfg


@pytest.mark.device
def test_prequant_modules_are_quantized(prequant):
    """Loading a pre-quantized checkpoint yields QuantizedLinear, not fp Linear."""
    silica_model, _, _, _ = prequant
    q_proj = silica_model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, nn.QuantizedLinear)


@pytest.mark.device
@pytest.mark.parametrize("prompt", ["The capital of France is", "2 + 2 =", "Once upon a time"])
def test_prequant_argmax_matches_mlx_lm(prequant, prompt):
    silica_model, ref_model, tok, _ = prequant
    ids = mx.array(tok.encode(prompt))[None]
    s = silica_model(ids)[:, -1, :]
    r = ref_model(ids)[:, -1, :]
    mx.eval(s, r)
    assert int(mx.argmax(s, -1).item()) == int(mx.argmax(r, -1).item())
