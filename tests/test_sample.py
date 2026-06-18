"""Sampler tests (need MLX, but no checkpoint — not device-gated).

Covers the audit bug: make_sampler must NOT reseed the process-global RNG, so a
fixed seed makes one run reproducible without freezing all runs or clobbering
global state.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from silica.sample import make_sampler
from silica.config import GenConfig


def test_greedy_returns_argmax():
    s = make_sampler(GenConfig(temperature=0.0))
    logits = mx.array([[0.1, 5.0, 0.2, 0.3]])
    assert int(s(logits).item()) == 1


def test_seeded_sampler_is_reproducible():
    logits = mx.array([[0.5, 0.4, 0.6, 0.3, 0.55] * 10])  # fixed (1, 50) logits
    s1 = make_sampler(GenConfig(seed=7, temperature=1.0))
    s2 = make_sampler(GenConfig(seed=7, temperature=1.0))
    seq1 = [int(s1(logits).item()) for _ in range(8)]
    seq2 = [int(s2(logits).item()) for _ in range(8)]
    assert seq1 == seq2                      # same seed -> same draws
    assert len(set(seq1)) > 1                # and the per-step split actually varies draws


def test_unseeded_samplers_differ():
    logits = mx.zeros((1, 256))              # uniform -> draws spread across vocab
    s1 = make_sampler(GenConfig(seed=None, temperature=1.0))
    s2 = make_sampler(GenConfig(seed=None, temperature=1.0))
    seq1 = [int(s1(logits).item()) for _ in range(12)]
    seq2 = [int(s2(logits).item()) for _ in range(12)]
    assert seq1 != seq2                      # default stream advances; not reseeded


def test_make_sampler_does_not_touch_global_rng():
    """The regression guard: constructing a seeded sampler must not reseed global."""
    mx.random.seed(999)
    mx.eval(mx.random.uniform(shape=(8,)))
    ref = mx.random.uniform(shape=(8,)); mx.eval(ref)

    mx.random.seed(999)
    mx.eval(mx.random.uniform(shape=(8,)))
    make_sampler(GenConfig(seed=0, temperature=1.0))   # old bug: called mx.random.seed(0)
    got = mx.random.uniform(shape=(8,)); mx.eval(got)

    assert mx.allclose(ref, got), "make_sampler perturbed the global RNG stream"
