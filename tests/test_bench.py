"""Bench helper tests — the llama-bench JSON parser robustness (audit #8).

Needs MLX only because importing bench.cross_engine pulls it transitively; the
parser itself is pure Python.
"""

import pytest

pytest.importorskip("mlx.core")

from bench.cross_engine import _parse_llama_tg


def test_parse_good_json_returns_median_tg():
    s = '[{"n_gen":0,"avg_ts":500.0},{"n_gen":128,"avg_ts":60.0},{"n_gen":128,"avg_ts":62.0}]'
    assert _parse_llama_tg(s) == 61.0          # median of the two tg rows; prompt row ignored


def test_parse_non_json_raises_clear_error():
    with pytest.raises(RuntimeError):
        _parse_llama_tg("warming up the model ...\nnot json at all")


def test_parse_no_tg_rows_raises():
    with pytest.raises(RuntimeError):
        _parse_llama_tg('[{"n_gen":0,"avg_ts":500.0}]')   # only prompt-processing rows


def test_parse_missing_avg_ts_key_raises():
    with pytest.raises(RuntimeError):
        _parse_llama_tg('[{"n_gen":128}]')                # schema drift -> no usable rows


def test_measure_peak_bandwidth_returns_positive():
    """Smoke test for the median-per-iteration peak-bandwidth measurement (#10)."""
    from bench.baseline import measure_peak_bandwidth
    assert measure_peak_bandwidth(nbytes=50_000_000, iters=5) > 0
