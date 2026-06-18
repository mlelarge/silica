"""Sliding-window perplexity tests (audit #9), with a fake uniform model.

Deterministic (zero logits -> uniform -> per-token NLL = log(vocab)), so we can
check the windowing exactly: every token scored once, full-context tail only.
"""

import math

import pytest

mx = pytest.importorskip("mlx.core")

from bench.eval_ppl import token_nll


class _UniformModel:
    """Zero logits -> uniform over `vocab`; per-token NLL = log(vocab)."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, x, cache=None):
        b, length = x.shape
        return mx.zeros((b, length, self.vocab))


def test_sliding_window_scores_each_token_exactly_once():
    vocab = 64
    ids = list(range(30))                      # 30 tokens, all < vocab
    total, n = token_nll(_UniformModel(vocab), ids, max_seq=8, stride=4)
    assert n == len(ids) - 1                   # no double-scoring, no gaps
    assert abs(total / n - math.log(vocab)) < 1e-4


def test_single_window_when_corpus_fits():
    vocab = 64
    ids = list(range(5))
    total, n = token_nll(_UniformModel(vocab), ids, max_seq=16, stride=8)
    assert n == len(ids) - 1


def test_default_stride_covers_all_tokens():
    vocab = 32
    ids = list(range(20))
    _, n = token_nll(_UniformModel(vocab), ids, max_seq=6)   # stride defaults to 3
    assert n == len(ids) - 1                   # overlap (stride<max_seq) -> no gaps
