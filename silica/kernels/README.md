# kernels/ — custom Metal fusions (M3)

Empty until M2 profiling proves headroom. See PLAN.md §5 (M3) and §7.

**Gate to add a kernel here:** baseline achieved-bandwidth `< X%` **and**
profiling shows a fusible `> Y%` traffic/launch cost on the measured hot path.

**Rules (non-negotiable):**
1. Fusion only — never a from-scratch matmul. We do not beat `mx.quantized_matmul`'s GEMV; we cut the traffic/launches *around* it.
2. Every kernel ships behind a pure-MLX fallback, selected at runtime.
3. Every kernel: a correctness test vs the fallback + a standalone microbench it must win.

First candidate: `dequant + GEMV + SwiGLU` for the MLP block.
Second candidate: fused (possibly quantized) attention — only if our KV layout
diverges from `mx.fast.scaled_dot_product_attention` (note: mainline MLX has no
native quantized-KV SDPA, so this is also where quantized-KV decode could be
made fast rather than the ~0.5× `quantized_matmul`-×2 path).
