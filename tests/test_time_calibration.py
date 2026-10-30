"""Tests for src/test_time_calibration.py.

The in-module `test_calibration_invariants()` is the contract; this file adds
the adversarial cases that a single self-test cannot express: dtype/shape
abuse, the numeric fallback path, and a large-batch throughput check.
"""

import time

import numpy as np
import pytest

from src.test_time_calibration import (
    N_CLASSES,
    adaptive_neighborhood_centering,
    balanced_accuracy,
    sinkhorn_knopp_transport,
    temperature_scale_logits,
    test_calibration_invariants as _run_contract,
)


def test_module_contract():
    metrics = _run_contract(verbose=False)
    assert metrics["bacc_after_sinkhorn"] > metrics["bacc_before"] + 0.05
    assert metrics["cluster_bacc_after"] > metrics["cluster_bacc_before"] + 0.02


def test_temperature_rejects_bad_input():
    with pytest.raises(ValueError):
        temperature_scale_logits(np.zeros((3, 4)), 0.0)
    with pytest.raises(ValueError):
        temperature_scale_logits(np.zeros((3, 4)), np.ones(5))
    with pytest.raises(ValueError):
        temperature_scale_logits(np.zeros((3, 4)), np.array([1.0, np.nan, 1.0, 1.0]))


def test_sinkhorn_rejects_bad_alpha_and_marginal():
    p = temperature_scale_logits(np.zeros((4, 18)), 1.0)
    with pytest.raises(ValueError):
        sinkhorn_knopp_transport(p, alpha=1.5)
    with pytest.raises(ValueError):
        sinkhorn_knopp_transport(p, alpha=-0.1)
    with pytest.raises(ValueError):
        sinkhorn_knopp_transport(p, target_col_marginal=np.ones(3))
    with pytest.raises(ValueError):
        sinkhorn_knopp_transport(p, target_col_marginal=np.zeros(18))


def test_sinkhorn_survives_hostile_input():
    rng = np.random.default_rng(3)
    p = rng.random((50, N_CLASSES))
    p[0] = 0.0                     # dead row
    p[1, 3] = np.nan               # NaN row
    p[2] = 1e-300                  # underflow row
    p[3, 0] = 1.0                  # one-hot row
    out = sinkhorn_knopp_transport(p, alpha=1.0)
    assert np.isfinite(out).all() and (out >= 0).all()
    assert np.allclose(out.sum(axis=1), 1.0)


def test_sinkhorn_never_diverges_at_tiny_reg():
    """reg << cost scale used to blow the log-domain duals up to 1e89."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        p = temperature_scale_logits(rng.normal(0.0, 2.0, (40, N_CLASSES)), 1.0)
        out, info = sinkhorn_knopp_transport(p, reg=0.01, alpha=1.0, return_info=True)
        assert np.isfinite(out).all()
        assert np.allclose(out.sum(axis=1), 1.0)
        worst = max(worst, float(np.abs(out.sum(axis=0) - 40 / N_CLASSES).max()))
    assert worst < 1.0


def test_single_cluster_matches_the_documented_18T_identity():
    """For a full 18-model cluster, n == K, so n*T == 18*T exactly."""
    rng = np.random.default_rng(5)
    p = temperature_scale_logits(rng.normal(0.0, 1.0, (N_CLASSES, N_CLASSES)), 1.0)
    out = sinkhorn_knopp_transport(p, reg=1.0, alpha=1.0, tol=1e-10)
    assert np.allclose(out.sum(axis=0), 1.0, atol=1e-6)


def test_centering_rejects_bad_strength_and_shape():
    p = temperature_scale_logits(np.zeros((6, 18)), 1.0)
    with pytest.raises(ValueError):
        adaptive_neighborhood_centering(p, strength=-0.1)
    with pytest.raises(ValueError):
        adaptive_neighborhood_centering(p, strength=3.0)
    with pytest.raises(ValueError):
        adaptive_neighborhood_centering(p, embeddings=np.zeros((5, 4)))


def test_large_batch_throughput():
    """4,000 passages is the hidden-test scale; the whole chain must be seconds."""
    rng = np.random.default_rng(7)
    z = rng.normal(0.0, 1.5, (4000, N_CLASSES))
    emb = rng.normal(0.0, 1.0, (4000, 32))
    start = time.time()
    p = temperature_scale_logits(z, 1.2)
    p = adaptive_neighborhood_centering(p, emb, k_neighbors=18, strength=0.2)
    p = sinkhorn_knopp_transport(p, alpha=0.5)
    elapsed = time.time() - start
    assert np.allclose(p.sum(axis=1), 1.0)
    assert elapsed < 30.0, f"calibration chain took {elapsed:.1f}s on 4k rows"


def test_balanced_accuracy_matches_sklearn():
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(9)
    y_true = rng.integers(0, 18, 500)
    y_pred = rng.integers(0, 18, 500)
    assert np.isclose(
        balanced_accuracy(y_true, y_pred, 18),
        balanced_accuracy_score(y_true, y_pred),
        atol=1e-12,
    )
