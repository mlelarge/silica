"""Typed configuration.

`ModelConfig` mirrors the fields silica reads from a Qwen3 HuggingFace
`config.json`. The audit flagged several Qwen3-specific traps that are encoded
here as explicit, *required* fields rather than derived quantities:

  * `head_dim` is decoupled from hidden_size/num_heads (Qwen3-0.6B: 128, NOT
    1024/16 = 64). Always read it from config.
  * `attention_bias` is False on Qwen3 (Qwen2 used QKV bias).
  * `tie_word_embeddings` is True on 0.6B (no separate lm_head weight).
  * `rope_theta` is 1e6 (the RoPE default of 1e4 would break parity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    # --- architecture (read straight from config.json) ---
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int                 # decoupled; do NOT compute hidden//heads
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40960
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    hidden_act: str = "silu"
    # YaRN / long-context scaling, e.g. {"rope_type": "yarn", "factor": 4.0,
    # "original_max_position_embeddings": 32768}. None == native context only.
    rope_scaling: dict | None = None
    # native (pre-training) context length; YaRN extends beyond this.
    native_context_length: int = 32768

    # --- special tokens (Qwen3) ---
    bos_token_id: int | None = 151643      # <|endoftext|>
    eos_token_id: int | tuple[int, ...] = 151645  # <|im_end|>
    model_type: str = "qwen3"

    def __post_init__(self) -> None:
        # head_dim is intentionally decoupled from hidden_size//num_heads on Qwen3
        # (it is a required field for exactly that reason — see `head_dim`).
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads})."
            )

    @property
    def n_rep(self) -> int:
        """GQA repeat factor (query heads per kv head)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        e = self.eos_token_id
        return tuple(e) if isinstance(e, (list, tuple)) else (e,)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        # head_dim may be absent in some configs; fall back to the derived value
        # only as a last resort, and record nothing silently.
        if "head_dim" not in d:
            d = {**d, "head_dim": d["hidden_size"] // d["num_attention_heads"]}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True)
class QuantConfig:
    """Selective quantization policy (M1).

    The audit's key correction: do NOT quantize everything flatly. Keep the
    (tied) embedding / lm_head and the norms at higher precision, because on
    Qwen3-0.6B the embedding *is* the lm_head and crude 4-bit there degrades
    both the input representation and the output logits.
    """

    bits: int = 4
    group_size: int = 64            # 64 default; {32, 64, 128} as ablation knobs
    # bits to use for the embedding / lm_head (None == leave unquantized).
    # (Norms/RoPE are always left fp — nn.quantize only touches Linear/Embedding.)
    embed_bits: int | None = 6
    # Stronger-recipe knob: module-path suffixes to keep at higher precision
    # (`embed_bits`), e.g. ("down_proj",) — the quality-sensitive MLP output.
    # MLX has no k-quants, so finer group_size + protecting sensitive layers is
    # how silica narrows the quality gap to llama.cpp's Q4_K_M.
    high_bits_proj: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"unsupported bits={self.bits}")
        if self.group_size not in (32, 64, 128):
            raise ValueError(f"unsupported group_size={self.group_size}")
        if self.embed_bits is not None and self.embed_bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"unsupported embed_bits={self.embed_bits}")


@dataclass(frozen=True)
class GenConfig:
    """Sampling + generation controls (greedy for M0; rest land in M1.5)."""

    max_tokens: int = 256
    temperature: float = 0.0        # 0.0 == greedy (M0)
    top_k: int = 0                  # 0 == disabled
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    # None -> fresh randomness each generation (default stream advances);
    # an int makes a single run reproducible WITHOUT touching global RNG state.
    seed: int | None = None
    stop: tuple[str, ...] = ()      # extra string stop-sequences
    # apply the model's chat template before prefill (ChatML for Qwen3).
    use_chat_template: bool = True
    # Quantized KV cache (M1): None -> fp KV; else memory-for-speed (~0.5x fp16
    # decode). Quantized KV does NOT compose with a rotating cache.
    kv_bits: int | None = None
    kv_group_size: int = 64
    # Keep the first `quantized_kv_start` tokens (incl. the prompt) in fp, then
    # quantize — mirrors mlx-lm. 0 == quantize right after prefill; a larger
    # value keeps short generations exact and only pays off at long context.
    quantized_kv_start: int = 0


@dataclass(frozen=True)
class BenchConfig:
    """Measurement harness knobs (§6). Defaults encode the audit's rigor asks."""

    warmup: int = 3                 # discarded iterations (JIT / pipeline build)
    runs: int = 10                  # K runs for median + IQR
    # The denominator for "% of peak bandwidth". MUST be set to the exact chip
    # SKU's rated bandwidth — e.g. M3 Max is 300 OR 400 GB/s. No default that
    # could silently mislabel a result.
    device_bandwidth_gbps: float | None = None
    chip_name: str = "unknown-apple-silicon"
    context_lengths: tuple[int, ...] = field(default=(0, 2048, 8192, 32768))
