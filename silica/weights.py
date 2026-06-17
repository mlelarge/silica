"""Load Qwen3 weights into a silica model.

Handles the loading specifics the audit flagged:

  * single-file (`model.safetensors`, the Qwen3-0.6B case) AND sharded
    checkpoints (`model.safetensors.index.json` + shards, for larger members);
  * MLX `mx.load` is lazy (read-on-eval), NOT mmap — we materialize explicitly;
  * tied embeddings: drop a stray `lm_head.weight` if present;
  * SELECTIVE quantization (M1): keep the (tied) embedding/lm_head at higher
    bits, leave norms in fp, and skip any layer whose last dim is not divisible
    by `group_size` (a hard `mx.quantize` constraint).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig, QuantConfig
from .model import Qwen3ForCausalLM


def resolve_model_path(model: str | Path) -> Path:
    """Local dir as-is; otherwise download the HF snapshot (weights + config)."""
    p = Path(model)
    if p.exists():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise FileNotFoundError(
            f"{model} not found locally and huggingface_hub is not installed. "
            f"Install it or pass a local snapshot directory."
        ) from e
    return Path(
        snapshot_download(
            repo_id=str(model),
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"],
        )
    )


def _load_safetensors(path: Path) -> dict[str, mx.array]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            shards = sorted({v for v in json.load(f)["weight_map"].values()})
        files = [path / s for s in shards]
    else:
        files = [Path(p) for p in glob.glob(str(path / "*.safetensors"))]
    if not files:
        raise FileNotFoundError(f"no .safetensors found under {path}")
    weights: dict[str, mx.array] = {}
    for f in files:
        weights.update(mx.load(str(f)))   # lazy until mx.eval
    return weights


def _selective_predicate(qcfg: QuantConfig):
    """class_predicate for nn.quantize implementing the mixed-precision policy."""
    skipped: list[str] = []

    def predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized"):
            return False
        # Keep embedding / lm_head higher-precision (they are tied on 0.6B).
        if path.endswith("embed_tokens") or path.endswith("lm_head"):
            if qcfg.embed_bits is None:
                return False
            return {"group_size": qcfg.group_size, "bits": qcfg.embed_bits}
        # Enforce the group_size divisibility constraint; keep fp otherwise.
        w = getattr(module, "weight", None)
        if w is not None and w.shape[-1] % qcfg.group_size != 0:
            skipped.append(path)
            return False
        return True

    predicate.skipped = skipped  # type: ignore[attr-defined]
    return predicate


def load_model(
    model: str | Path = "Qwen/Qwen3-0.6B",
    *,
    quant: QuantConfig | None = None,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[Qwen3ForCausalLM, ModelConfig]:
    """Build and load a silica Qwen3 model.

    `quant=None` -> M0 fp baseline. Pass a QuantConfig for M1 selective quant.
    `dtype` is the fp compute dtype (Qwen3 weights are stored bf16); record it
    in any parity/bench run since it shifts tolerances and speed.
    """
    path = resolve_model_path(model)
    cfg = ModelConfig.from_json(path / "config.json")

    net = Qwen3ForCausalLM(cfg)

    weights = _load_safetensors(path)
    if cfg.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    # Load fp weights FIRST, then quantize: nn.quantize quantizes each module's
    # *current* weights in place. Quantizing before load would leave the modules
    # expecting packed (weight/scales/biases) tensors the fp checkpoint lacks.
    net.load_weights(list(weights.items()))
    mx.eval(net.parameters())   # materialize (read-on-eval)

    if quant is not None:
        predicate = _selective_predicate(quant)
        nn.quantize(net, group_size=quant.group_size, bits=quant.bits,
                    class_predicate=predicate)
        mx.eval(net.parameters())
        if predicate.skipped:  # type: ignore[attr-defined]
            print(f"[silica] quant skipped (group_size): {predicate.skipped}")

    net.eval()
    return net, cfg
