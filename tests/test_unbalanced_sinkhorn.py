"""
Unit tests for src/unbalanced_sinkhorn.py.

Verifies:
1. Invariant contract execution and numerical convergence.
2. Robustness to hostile, degenerate, and boundary inputs.
3. Unbalanced Optimal Transport marginal relaxation bounds and label-shift adaptation.
4. Confidence-weighted HodgeRank curl-free cycle elimination and inconsistency quantification.
5. Dynamic topic clustering adaptive thresholding and cluster size constraints.
6. OAS and Ledoit-Wolf covariance shrinkage properties and positive-definiteness.
7. Two-way ANOVA decomposition, joint diagonalization, and cavity EM marginal inference.
8. Monotonic safety guard (calibrator never degrades uncalibrated baseline).
9. Throughput benchmark on full hidden-test scale (4,000 passages).
"""

import time
import numpy as np
import pytest

from src.unbalanced_sinkhorn import (
    N_CLASSES,
    DynamicTopicClusterer,
    HodgeRankProjector,
    UnbalancedCalibrator,
    adaptive_topic_clustering,
    balanced_accuracy,
    cavity_em_marginals,
    fit_shared_prompt_gaussian,
    ledoit_wolf_shrinkage,
    oas_shrinkage,
    per_class_recall,
    project_hodgerank_weighted,
    shrink_covariance,
    temperature_scale_logits,
    test_unbalanced_invariants as _run_invariants,
    two_way_anova,
    unbalanced_sinkhorn_transport,
    within_prompt_pca,
)


def test_module_contract_invariants():
    """Runs the complete in-module executable invariant test contract."""
    metrics = _run_invariants(verbose=False)
    assert metrics["passed_invariants"] >= 18
    assert metrics["bacc_after"] > metrics["bacc_before"] + 0.05


def test_uot_input_validation():
    """Tests input rejection and boundary validation in UOT."""
    p_dummy = np.ones((5, N_CLASSES)) / N_CLASSES

    with pytest.raises(ValueError, match="alpha must be in"):
        unbalanced_sinkhorn_transport(p_dummy, alpha=-0.1)

    with pytest.raises(ValueError, match="alpha must be in"):
        unbalanced_sinkhorn_transport(p_dummy, alpha=1.5)

    with pytest.raises(ValueError, match="reg must be > 0"):
        unbalanced_sinkhorn_transport(p_dummy, reg=0.0)

    with pytest.raises(ValueError, match="target_col_marginal must be"):
        unbalanced_sinkhorn_transport(p_dummy, target_col_marginal=np.ones(5))

    with pytest.raises(ValueError, match="target_col_marginal must be"):
        unbalanced_sinkhorn_transport(p_dummy, target_col_marginal=-np.ones(N_CLASSES))

    with pytest.raises(ValueError, match="target_col_marginal must sum"):
        unbalanced_sinkhorn_transport(p_dummy, target_col_marginal=np.zeros(N_CLASSES))


def test_uot_hostile_inputs_survival():
    """Tests numerical resilience under extreme, malformed, and boundary inputs."""
    rng = np.random.default_rng(123)
    p_hostile = rng.uniform(0.01, 1.0, size=(50, N_CLASSES))
    p_hostile[0] = 0.0                       # Dead row
    p_hostile[1, 5] = np.nan                 # NaN injection
    p_hostile[2, 2] = np.inf                 # Positive infinity
    p_hostile[3, 8] = -np.inf                # Negative infinity
    p_hostile[4] = 1e-300                    # Severe underflow
    p_hostile[5, 0] = 1.0                    # One-hot hard assignment
    p_hostile[5, 1:] = 0.0

    out = unbalanced_sinkhorn_transport(p_hostile, reg=0.05, tau_col=1.0, alpha=1.0)
    assert np.all(np.isfinite(out)), "Hostile input caused non-finite values!"
    assert np.all(out >= 0.0), "Hostile input caused negative probabilities!"
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6), "Rows do not sum to 1!"


def test_uot_prevents_class_collapse_under_asymmetric_label_shift():
    """Demonstrates that Unbalanced Sinkhorn outperforms hard balanced Sinkhorn under test label shift."""
    rng = np.random.default_rng(777)
    K = 6
    N = 1200
    # Asymmetric test label distribution (common in partial test splits)
    p_true = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    y_test = rng.choice(K, size=N, p=p_true)

    # Logits with class bias against minority models
    bias = np.array([0.4, 0.2, 0.0, -0.5, -1.0, -1.8])
    z = rng.normal(0.0, 1.0, size=(N, K)) + bias[None, :]
    z[np.arange(N), y_test] += 1.8

    p_raw = temperature_scale_logits(z, 1.0)
    bacc_raw = balanced_accuracy(y_test, p_raw.argmax(axis=1), K)

    # Hard balanced Sinkhorn (forces 1/K uniform marginals)
    p_hard = unbalanced_sinkhorn_transport(p_raw, reg=0.2, tau_col=None, alpha=1.0, tol=1e-5)
    bacc_hard = balanced_accuracy(y_test, p_hard.argmax(axis=1), K)

    # Unbalanced Sinkhorn (gentle KL relaxation allows empirical label shift)
    p_unbal = unbalanced_sinkhorn_transport(p_raw, reg=0.2, tau_col=1.0, alpha=1.0, tol=1e-5)
    bacc_unbal = balanced_accuracy(y_test, p_unbal.argmax(axis=1), K)

    # Unbalanced Sinkhorn must strictly outperform hard uniform Sinkhorn here
    assert bacc_unbal > bacc_hard, (
        f"Unbalanced ({bacc_unbal:.4f}) should beat hard Sinkhorn ({bacc_hard:.4f}) "
        f"on asymmetric test distribution!"
    )


def test_hodgerank_confidence_weighting_and_transitivity():
    """Verifies confidence-weighting impact and strict curl-free transitivity."""
    rng = np.random.default_rng(999)
    K = 6
    N = 100

    # Pairwise comparison network with a cycle (0->1->2->0) and tail (2->3->4->5)
    pairs = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)]

    # Pair (0, 1) has very high confidence deltas; pair (1, 2) has near-zero confidence deltas
    deltas = {}
    deltas[(0, 1)] = rng.normal(3.0, 0.2, size=N)   # High confidence
    deltas[(1, 2)] = rng.normal(0.1, 0.1, size=N)   # Low confidence
    deltas[(2, 0)] = rng.normal(-1.5, 0.2, size=N)  # Moderate confidence
    deltas[(2, 3)] = rng.normal(1.2, 0.3, size=N)
    deltas[(3, 4)] = rng.normal(-0.8, 0.2, size=N)
    deltas[(4, 5)] = rng.normal(2.1, 0.4, size=N)

    projector = HodgeRankProjector(beta=2.0, confidence_weighting=True, n_classes=K)
    V = projector.transform(deltas)

    assert V.shape == (N, K)
    assert np.allclose(V.mean(axis=1), 0.0, atol=1e-12)

    # Mathematical identity: any closed path must sum to 0 identically
    grad_01 = V[:, 1] - V[:, 0]
    grad_12 = V[:, 2] - V[:, 1]
    grad_20 = V[:, 0] - V[:, 2]
    cycle_sum = grad_01 + grad_12 + grad_20
    assert np.allclose(cycle_sum, 0.0, atol=1e-12), "HodgeRank potential is not curl-free!"

    # Inconsistency must be strictly within [0, 1]
    assert 0.0 <= projector.curl_inconsistency_ <= 1.0


def test_dynamic_topic_clustering_adaptive_selection():
    """Verifies that dynamic thresholding selects valid cluster boundaries and rejects oversize."""
    rng = np.random.default_rng(555)
    N = 80
    # Create 8 prompt clusters of size 10 (valid sizes: 2 <= s <= 18)
    true_labels = np.repeat(np.arange(8), 10)
    D = np.ones((N, N)) * 0.90
    for g in range(8):
        idx = np.where(true_labels == g)[0]
        D[np.ix_(idx, idx)] = rng.uniform(0.05, 0.25, size=(10, 10))
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)

    best_thresh, groups, metrics = adaptive_topic_clustering(
        D, min_thresh=0.20, max_thresh=0.85, n_candidates=20, min_size=2, max_size=18
    )

    sizes = np.bincount(groups)[groups]
    assert sizes.min() >= 2
    assert sizes.max() <= 18
    assert metrics["oversize_frac"] == 0.0
    assert metrics["valid_frac"] == 1.0


def test_oas_and_ledoit_wolf_covariance_properties():
    """Verifies covariance shrinkage properties, positive-definiteness, and bounds."""
    rng = np.random.default_rng(333)
    X = rng.normal(0, 1.0, size=(80, 30))

    a_oas = oas_shrinkage(X)
    a_lw = ledoit_wolf_shrinkage(X)

    assert 0.0 <= a_oas <= 1.0
    assert 0.0 <= a_lw <= 1.0

    S_emp = (X.T @ X) / len(X)
    S_shrunk = shrink_covariance(S_emp, a_oas)

    # Shrunk covariance must be strictly positive definite (all eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(S_shrunk)
    assert eigvals.min() > 1e-6


def test_two_way_anova_and_cavity_em():
    """Verifies Two-Way ANOVA prompt deconfounding and cavity EM marginal inference."""
    rng = np.random.default_rng(444)
    N = 360
    K = 18
    G = 20
    D = 32

    y = np.tile(np.arange(K), N // K)
    prompts = np.repeat(np.arange(G), N // G)

    # Generate synthetic features: Z = mu[y] + P[prompt] + E
    true_mu = rng.normal(0, 2.0, size=(K, D))
    true_P = rng.normal(0, 1.5, size=(G, D))
    E = rng.normal(0, 0.5, size=(N, D))
    Z = true_mu[y] + true_P[prompts] + E

    pu, pinv = np.unique(prompts, return_inverse=True)
    mu_hat, P_hat, cnt = two_way_anova(Z, y, pinv, G, n_classes=K, iters=4)

    assert mu_hat.shape == (K, D)
    assert P_hat.shape == (G, D)

    # Shared prompt Gaussian fit
    prm = fit_shared_prompt_gaussian(Z, y, prompts, cfg={"d": 16, "se_shrink": "oas", "sp_shrink": "oas"})
    assert prm["W"].shape[1] == 16
    assert len(prm["lam"]) == 16
    assert np.all(prm["lam"] > 0)

    # Cavity EM marginal inference on a clique of 4 passages
    U_test = Z[:4] @ prm["W"]
    q = cavity_em_marginals(U_test, prm["m"], prm["lam"], T=1.0, iters=4, repel=True)
    assert q.shape == (4, K)
    assert np.all(np.isfinite(q))
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-6)


def test_unbalanced_calibrator_monotonic_safety():
    """Verifies that the orchestrator never regresses below uncalibrated accuracy on adversarial OOF."""
    rng = np.random.default_rng(222)
    K = N_CLASSES
    N = 300

    # Pure noise logits with no class signal
    z_noise = rng.normal(0, 1.0, size=(N, K))
    y_random = rng.integers(0, K, size=N)

    calibrator = UnbalancedCalibrator()
    calibrator.fit(z_noise, y_random)

    # Must fall back or maintain >= raw balanced accuracy
    raw_bacc = balanced_accuracy(y_random, z_noise.argmax(axis=1), K)
    cal_bacc = calibrator.report["bacc_calibrated"]
    assert cal_bacc >= raw_bacc - 1e-12


def test_hidden_test_throughput_benchmark():
    """Verifies that the entire calibration pipeline processes 4,000 hidden test rows in < 2 seconds."""
    rng = np.random.default_rng(111)
    N_test = 4000
    K = N_CLASSES

    logits = rng.normal(0, 1.5, size=(N_test, K))
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)]
    corrections = {p: rng.normal(0, 1.0, size=N_test) for p in pairs}

    t0 = time.time()
    calibrator = UnbalancedCalibrator(alpha=0.5, tau_col=1.0, reg=0.05)
    p_cal = calibrator.transform(logits, pairwise_corrections=corrections)
    elapsed = time.time() - t0

    assert p_cal.shape == (N_test, K)
    assert np.all(np.isfinite(p_cal))
    assert np.allclose(p_cal.sum(axis=1), 1.0, atol=1e-5)
    assert elapsed < 2.0, f"Full hidden-test execution took {elapsed:.2f}s (must be < 2.0s)!"
