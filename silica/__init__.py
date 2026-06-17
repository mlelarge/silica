"""silica — a transparent single-stream LLM engine for Apple Silicon.

See PLAN.md. v0 targets batch=1 inference of Qwen3-family decoders on MLX.

Status: pre-M0 scaffold. Unvalidated on device — correctness-first parity
(tests/test_parity.py) has NOT been run yet. Treat every module as a stub
whose API is correct against the verified MLX/mlx-lm surface but whose numerics
must be proven by the M0 parity gate before any optimization.
"""

__version__ = "0.0.0"

__all__ = [
    "config",
    "weights",
    "model",
    "cache",
    "sample",
    "detokenize",
    "generate",
]
