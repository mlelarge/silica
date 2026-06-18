"""Sampler tests (need MLX, but no checkpoint — not device-gated).

Covers the audit bug: make_sampler must NOT reseed the process-global RNG, so a
fixed seed makes one run reproducible without freezing all runs or clobbering
global state.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from silica.sample import make_sampler, _top_k, _top_p, _min_p
from silica.config import GenConfig

NEG = float("-inf")


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
    ref = mx.random.uniform(shape=(8,))
    mx.eval(ref)

    mx.random.seed(999)
    mx.eval(mx.random.uniform(shape=(8,)))
    make_sampler(GenConfig(seed=0, temperature=1.0))   # old bug: called mx.random.seed(0)
    got = mx.random.uniform(shape=(8,))
    mx.eval(got)

    assert mx.allclose(ref, got), "make_sampler perturbed the global RNG stream"


# ---- filter functions (the non-greedy path) ------------------------------- #


def test_top_k_keeps_only_top_k():
    out = _top_k(mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), 2)[0].tolist()
    assert out[4] == 5.0 and out[3] == 4.0          # top-2 kept
    assert out[0] == out[1] == out[2] == NEG        # rest masked


def test_top_p_always_keeps_the_top_token():
    # one dominant token (prob > top_p) -> nucleus is just that token
    out = _top_p(mx.array([[10.0, 0.0, 0.0, 0.0]]), 0.5)[0].tolist()
    assert out[0] == 10.0
    assert out[1] == out[2] == out[3] == NEG


def test_min_p_masks_low_probability_tail():
    out = _min_p(mx.array([[10.0, 9.0, 0.0, -10.0]]), 0.5)[0].tolist()
    assert out[0] == 10.0                            # top kept
    assert out[3] == NEG                             # far tail masked


def test_filters_never_mask_the_argmax():
    # every filter must leave the most likely token finite (categorical safety)
    logits = mx.array([[0.1, 9.0, 0.2, 0.3, 0.05]])
    for filt in (lambda x: _top_k(x, 1), lambda x: _top_p(x, 0.1), lambda x: _min_p(x, 0.9)):
        assert filt(logits)[0].tolist()[1] == 9.0
