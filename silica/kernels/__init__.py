"""Custom `mx.fast.metal_kernel` fusions (M3 — gated on M2 profiling).

Empty by design until M2 proves there is headroom (PLAN §5 M3, §7). Discipline
for anything added here:

  * never a from-scratch matmul — only *fusions* that cut memory traffic or
    launch count (e.g. dequant + GEMV + SwiGLU);
  * every kernel ships behind a pure-MLX fallback selected at runtime;
  * every kernel has a correctness test vs that fallback and a standalone
    microbenchmark that must beat it.
"""
