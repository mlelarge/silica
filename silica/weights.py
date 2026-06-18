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
from .models import build_model


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
        # Enforce the group_size divisibility constraint for EVERY quantizable
        # module (incl. embed/lm_head); keep fp otherwise. This guard used to sit
        # AFTER the embed branch, so embed/lm_head bypassed it and could hard-crash
        # mx.quantize on models whose hidden_size isn't divisible by group_size.
        w = getattr(module, "weight", None)
        if w is not None and w.shape[-1] % qcfg.group_size != 0:
            skipped.append(path)
            return False
        # Keep embedding / lm_head higher-precision (they are tied on 0.6B).
        if path.endswith("embed_tokens") or path.endswith("lm_head"):
            if qcfg.embed_bits is None:
                return False
            return {"group_size": qcfg.group_size, "bits": qcfg.embed_bits}
        # Keep the MoE router (`mlp.gate`) in fp — tiny and routing-sensitive.
        if path.endswith("mlp.gate"):
            return False
        # Stronger recipe: keep quality-sensitive projections at higher precision.
        if qcfg.high_bits_proj and path.endswith(tuple(qcfg.high_bits_proj)):
            return {"group_size": qcfg.group_size, "bits": qcfg.embed_bits or 8}
        return True

    predicate.skipped = skipped  # type: ignore[attr-defined]
    return predicate


def _apply_checkpoint_quantization(net: nn.Module, weights: dict, ckpt_quant: dict) -> None:
    """Convert modules to match an already-quantized checkpoint's packed tensors.

    A module is quantized iff the checkpoint carries its ``<path>.scales`` (so this
    one rule covers uniform 4/8-bit, mixed-precision recipes, AND the stacked MoE
    experts). Per-module bits/group come from ``ckpt_quant[path]`` when present
    (mlx mixed-precision configs store per-module dicts), else the top-level.
    """
    g = ckpt_quant.get("group_size", 64)
    b = ckpt_quant.get("bits", 4)
    mode = ckpt_quant.get("mode", "affine")

    def class_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        spec = ckpt_quant.get(path)                 # per-module override (or False)
        if spec is False:
            return False
        if f"{path}.scales" not in weights:         # not quantized in this checkpoint
            return False
        if isinstance(spec, dict):
            return {"group_size": spec["group_size"], "bits": spec["bits"]}
        return {"group_size": g, "bits": b}

    kwargs = {"group_size": g, "bits": b, "class_predicate": class_predicate}
    if mode != "affine":
        kwargs["mode"] = mode
    nn.quantize(net, **kwargs)


def load_model(
    model: str | Path = "Qwen/Qwen3-0.6B",
    *,
    quant: QuantConfig | None = None,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[nn.Module, ModelConfig]:
    """Build and load a silica model (architecture chosen by the registry).

    `quant=None` -> fp (or, for a pre-quantized checkpoint, its stored precision).
    Pass a QuantConfig to quantize an fp checkpoint at load (M1 selective quant).
    A checkpoint whose config carries a `quantization` field is loaded already
    quantized (e.g. an mlx-community 4-bit model); `quant` is then ignored.
    `dtype` is the fp compute dtype (record it in parity/bench runs).
    """
    path = resolve_model_path(model)
    with open(path / "config.json") as f:
        config = json.load(f)
    cfg = ModelConfig.from_dict(config)

    net = build_model(cfg)              # registry dispatch on cfg.architectures

    weights = _load_safetensors(path)
    weights = net.sanitize(weights)         # MoE: stack per-expert weights
    if cfg.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    ckpt_quant = config.get("quantization")
    if ckpt_quant is not None:
        # Pre-quantized checkpoint: convert modules to match the packed tensors,
        # then load AS-IS -- no astype, since the packed weights are uint32 and
        # casting them to a float dtype would corrupt them.
        _apply_checkpoint_quantization(net, weights, ckpt_quant)
        net.load_weights(list(weights.items()))
        mx.eval(net.parameters())
        if quant is not None:
            print("[silica] checkpoint is pre-quantized; ignoring quant=...")
    else:
        # fp checkpoint: load fp FIRST, then optionally quantize. nn.quantize
        # quantizes each module's *current* weights in place; quantizing before
        # load would leave modules expecting packed tensors the fp checkpoint lacks.
        weights = {k: v.astype(dtype) for k, v in weights.items()}
        net.load_weights(list(weights.items()))
        mx.eval(net.parameters())
        if quant is not None:
            predicate = _selective_predicate(quant)
            nn.quantize(net, group_size=quant.group_size, bits=quant.bits,
                        class_predicate=predicate)
            mx.eval(net.parameters())
            if predicate.skipped:  # type: ignore[attr-defined]
                print(f"[silica] quant skipped (group_size): {predicate.skipped}")

    net.eval()
    return net, cfg
