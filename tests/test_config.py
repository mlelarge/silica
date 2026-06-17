"""Pure-Python config tests — encode the Qwen3 traps the audit flagged.

These run with no MLX installed.
"""

import pytest

from silica.config import ModelConfig, QuantConfig

# Qwen3-0.6B, as published.
QWEN3_0_6B = dict(
    hidden_size=1024,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=3072,
    vocab_size=151936,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    max_position_embeddings=40960,
    tie_word_embeddings=True,
    attention_bias=False,
)


def test_head_dim_is_decoupled_not_derived():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    # The trap: hidden//heads = 1024/16 = 64, but the real head_dim is 128.
    assert cfg.head_dim == 128
    assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads


def test_gqa_repeat_factor():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    assert cfg.n_rep == 2  # 16 query heads / 8 kv heads


def test_qwen3_has_no_attention_bias():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    assert cfg.attention_bias is False


def test_tied_embeddings_flag():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    assert cfg.tie_word_embeddings is True


def test_rope_theta_is_1e6_not_default_1e4():
    cfg = ModelConfig.from_dict(QWEN3_0_6B)
    assert cfg.rope_theta == 1_000_000.0


def test_eos_ids_normalized_to_tuple():
    cfg = ModelConfig.from_dict({**QWEN3_0_6B, "eos_token_id": 151645})
    assert cfg.eos_token_ids == (151645,)


def test_from_dict_ignores_unknown_keys():
    cfg = ModelConfig.from_dict({**QWEN3_0_6B, "architectures": ["Qwen3ForCausalLM"]})
    assert cfg.hidden_size == 1024


def test_from_dict_fills_missing_head_dim():
    d = {k: v for k, v in QWEN3_0_6B.items() if k != "head_dim"}
    cfg = ModelConfig.from_dict(d)
    assert cfg.head_dim == 64  # last-resort derivation when absent from config


def test_gqa_divisibility_enforced():
    with pytest.raises(ValueError):
        ModelConfig.from_dict({**QWEN3_0_6B, "num_key_value_heads": 7})


def test_quant_config_validation():
    QuantConfig(bits=4, group_size=64)         # ok
    with pytest.raises(ValueError):
        QuantConfig(bits=7)
    with pytest.raises(ValueError):
        QuantConfig(group_size=48)
