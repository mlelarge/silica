"""Shared pytest fixtures / markers.

`test_config.py` and `test_roofline.py` are pure-Python and run anywhere.
`test_parity.py` is the on-device M0 gate — it skips unless MLX, mlx-lm and a
Qwen3 checkpoint are available (set SILICA_PARITY_MODEL to a path or repo id).
"""

import os

import pytest

PARITY_MODEL = os.environ.get("SILICA_PARITY_MODEL", "Qwen/Qwen3-0.6B")


def pytest_configure(config):
    config.addinivalue_line("markers", "device: requires MLX + a checkpoint (M0 gate)")


@pytest.fixture(scope="session")
def parity_model_id():
    return PARITY_MODEL
