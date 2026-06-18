"""Backward-compat re-exports.

The model implementation moved to the `silica.models` package — a registry plus
one file per architecture (`qwen3.py`, `llama.py`) composed from a shared layer
library (`common.py`), mirroring how SGLang/vLLM handle many architectures.
Import model classes via `silica.models.build_model(cfg)`; this shim keeps the
old `silica.model` import paths working.
"""

from .models.common import causal_additive_mask  # noqa: F401
from .models.qwen3 import Qwen3Attention, Qwen3ForCausalLM  # noqa: F401
from .models.llama import LlamaAttention, LlamaForCausalLM  # noqa: F401
