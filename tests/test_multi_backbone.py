"""
Unit tests for MultiBackboneEngine (Qwen3-8B + Qwen3-14B + OLMo2-7B).
Tests:
- Label alignment between OLMo2 alphabetical ordering and competition canonical order
- Masked pooling with attention-sink avoidance (mask[:, 0] = 0)
- VRAM hygiene and context manager teardown
- Configuration presets and dimensions
- Sequential multi-backbone fusion pipeline
- Representation fusion across backbones
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.multi_backbone_engine import (
    CANONICAL_LABELS,
    OLMO2_CLASSES,
    REORDER_OLMO_TO_CANONICAL,
    TRUNK_PRESETS,
    TrunkConfig,
    MultiBackboneEngine,
    align_to_canonical,
    cleanup_vram,
    assert_vram_hygiene,
    trunk_vram_scope,
    pool_intermediate_hidden_states,
    execute_mock_trunk,
    softmax_stable,
    log_softmax_stable,
)


def test_label_alignment_integrity():
    """Verifies label sets and bidirectional alignment mappings."""
    assert len(CANONICAL_LABELS) == 18
    assert len(OLMO2_CLASSES) == 18
    assert set(CANONICAL_LABELS) == set(OLMO2_CLASSES)

    # Test permutation alignment
    N = 5
    rng = np.random.RandomState(42)
    dummy_olmo_probs = rng.rand(N, 18).astype(np.float32)
    dummy_olmo_probs /= dummy_olmo_probs.sum(axis=1, keepdims=True)

    aligned = align_to_canonical(dummy_olmo_probs, OLMO2_CLASSES, CANONICAL_LABELS)
    assert aligned.shape == (N, 18)

    # Check a specific class mapping
    for k_target, target_name in enumerate(CANONICAL_LABELS):
        k_source = OLMO2_CLASSES.index(target_name)
        np.testing.assert_allclose(aligned[:, k_target], dummy_olmo_probs[:, k_source])

    # Invertible
    inverted = align_to_canonical(aligned, CANONICAL_LABELS, OLMO2_CLASSES)
    np.testing.assert_allclose(inverted, dummy_olmo_probs)


def test_attention_sink_avoidance_pooling():
    """Verifies attention-sink token masking: mask[:, 0] = 0."""
    B, S, H = 2, 6, 16
    hs = [torch.randn(B, S, H) for _ in range(5)]

    # Make token 0 an extreme outlier ("attention sink")
    for layer in hs:
        layer[:, 0, :] = 1000.0

    mask = torch.ones(B, S)

    # 1. Pooling WITH attention-sink avoidance
    pooled_clean = pool_intermediate_hidden_states(
        hidden_states=hs,
        attention_mask=mask,
        target_layers=[1, 3],
        avoid_attention_sink=True
    )
    assert pooled_clean.shape == (B, 2 * H)
    # The extreme values at token 0 must not contaminate the pooled vector
    assert (pooled_clean.abs() < 50.0).all()

    # 2. Pooling WITHOUT attention-sink avoidance (polluted)
    pooled_polluted = pool_intermediate_hidden_states(
        hidden_states=hs,
        attention_mask=mask,
        target_layers=[1, 3],
        avoid_attention_sink=False
    )
    assert (pooled_polluted > 100.0).any()


def test_pool_hidden_states_with_layernorm():
    """Verifies layer normalization applied during intermediate hidden state extraction."""
    B, S, H = 3, 8, 32
    target_layers = [0, 2, 4]
    hs = [torch.randn(B, S, H) * 5.0 + 3.0 for _ in range(5)]
    mask = torch.ones(B, S)

    norms = nn.ModuleList([nn.LayerNorm(H) for _ in target_layers])
    pooled = pool_intermediate_hidden_states(
        hidden_states=hs,
        attention_mask=mask,
        target_layers=target_layers,
        layer_norms=norms,
        avoid_attention_sink=True
    )

    assert pooled.shape == (B, len(target_layers) * H)
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()

    # Verify per-layer slice normalization
    for i in range(len(target_layers)):
        sub = pooled[:, i * H:(i + 1) * H]
        mean_val = sub.mean(dim=-1)
        std_val = sub.std(dim=-1, unbiased=False)
        np.testing.assert_allclose(mean_val.detach().numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(std_val.detach().numpy(), 1.0, atol=1e-5)


def test_trunk_presets_dimensions():
    """Validates configuration parameters for Qwen3-8B, Qwen3-14B, and OLMo2-7B."""
    assert "qwen3_8b" in TRUNK_PRESETS
    assert "qwen3_14b" in TRUNK_PRESETS
    assert "olmo2_7b" in TRUNK_PRESETS

    q8 = TRUNK_PRESETS["qwen3_8b"]
    assert q8.hidden_size == 4096
    assert q8.keep_layers == 36
    assert q8.pool_layers == [2, 4, 6, 8, 12, 16, 20, 24]
    assert q8.w_plain == 0.226
    assert q8.w_twin == 0.079

    q14 = TRUNK_PRESETS["qwen3_14b"]
    assert q14.hidden_size == 5120
    assert q14.keep_layers == 40
    assert q14.device_strategy == "dual_t4_balanced"
    assert q14.w_plain == 0.429
    assert q14.w_twin == 0.150

    olmo = TRUNK_PRESETS["olmo2_7b"]
    assert olmo.hidden_size == 4096
    assert olmo.keep_layers == 24
    assert olmo.pool_layers == [4, 8, 12, 16, 20, 24]
    assert olmo.w_plain == 0.030
    assert olmo.w_twin == 0.030


def test_vram_cleanup_and_scope():
    """Ensures VRAM cleanup and trunk scope context manager operate cleanly."""
    stats = cleanup_vram()
    assert isinstance(stats, dict)
    assert_vram_hygiene(max_reserved_gib=1.0)

    executed = False
    with trunk_vram_scope("test_scope", verbose=False):
        executed = True
    assert executed

    # Exception safety
    with pytest.raises(ValueError):
        with trunk_vram_scope("error_scope", verbose=False):
            raise ValueError("Forced error")


def test_execute_mock_trunk():
    """Tests synthetic fallback for offline execution without GPUs."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A transformer model processes tokens through multi-head attention.",
        "Short passage test."
    ]
    groups = np.array([1, 1, 2])
    twin_rows = np.array([True, True, False])

    cfg = TRUNK_PRESETS["olmo2_7b"]
    res = execute_mock_trunk(
        config=cfg,
        test_texts=texts,
        train_texts=["Sample train passage."],
        groups=groups,
        twin_rows=twin_rows
    )

    assert res.name == "olmo2_7b"
    assert res.plain_prob.shape == (3, 18)
    assert res.plain_logp.shape == (3, 18)
    np.testing.assert_allclose(res.plain_prob.sum(axis=1), np.ones(3), atol=1e-5)

    assert res.twin_prob is not None
    assert res.twin_prob.shape == (2, 18)  # twin_rows has 2 True entries

    expected_dim = cfg.hidden_size * len(cfg.pool_layers)
    assert res.features.shape == (3, expected_dim)
    assert res.train_features.shape == (1, expected_dim)


def test_multi_backbone_engine_orchestration():
    """Tests end-to-end execution of MultiBackboneEngine across all 3 trunks."""
    texts = [
        "Deep learning architectures rely on residual connections and normalization.",
        "Another sentence describing natural language processing and transformer models.",
        "A third passage evaluating style and typography in generated texts.",
        "Fourth short passage."
    ]
    groups = np.array([1, 1, 2, 2])
    twin_rows = np.array([True, True, True, True])

    engine = MultiBackboneEngine(smoke_mode=True, verbose=False)
    result = engine.execute_all_trunks(
        test_texts=texts,
        train_texts=["Train passage 1", "Train passage 2"],
        groups=groups,
        twin_rows=twin_rows,
        return_fused_features=True
    )

    assert result.fused_logits.shape == (4, 18)
    assert result.fused_probs.shape == (4, 18)
    assert not np.isnan(result.fused_logits).any()
    assert not np.isnan(result.fused_probs).any()
    np.testing.assert_allclose(result.fused_probs.sum(axis=1), np.ones(4), atol=1e-5)

    assert "qwen3_8b" in result.trunk_results
    assert "qwen3_14b" in result.trunk_results
    assert "olmo2_7b" in result.trunk_results

    # Representation fusion check
    assert result.fused_features is not None
    dim_8b = 4096 * 8
    dim_14b = 5120 * 8
    dim_olmo = 4096 * 6
    expected_fused_dim = dim_8b + dim_14b + dim_olmo
    assert result.fused_features.shape == (4, expected_fused_dim)
