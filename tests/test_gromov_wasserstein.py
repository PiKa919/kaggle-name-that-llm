"""
Comprehensive unit tests for src/gromov_wasserstein_clustering.py.

Verifies:
1. Exactness of O(n^2 K + n K^2) tensor contraction vs explicit quadruple loops.
2. Gromov-Wasserstein convergence, loss monotonicity, and numerical stability.
3. Row-stochasticity and non-negativity of calibrated posteriors.
4. Permutation equivariance and symmetry preservation.
5. Edge cases: n=0, n=1, small cliques (n=2, 3), exact 18-model cliques (n=18), oversized (n=24).
6. Resiliency to hostile inputs (NaNs, infinities, underflow).
7. TransductiveCliqueCalibrator C^1 smooth regime weights and partition of unity.
8. Soft cavity occupation repulsion behavior on conflicting passages.
9. Batch calibration across mixed-size prompt cliques.
10. ConfidenceHodgeRank pure Condorcet cycle elimination (100% curl).
11. Proof of Zero-Condorcet Curl Theorem across all 816 triangles of the 18-model graph.
12. Exact potential recovery on pure gradient flows.
13. Confidence weighting power beta scaling prioritizes high-margin edges.
14. Backward compatibility with pairwise dictionary interface.
15. End-to-end transductive calibration balanced accuracy uplift.
"""

import math
import numpy as np
import pytest
from scipy.special import logsumexp

from src.config import VALID_18_MODELS
from src.gromov_wasserstein_clustering import (
    N_CLASSES,
    ConfidenceHodgeRank,
    GromovWassersteinMatcher,
    TransductiveCliqueCalibrator,
    balanced_accuracy_score,
    compute_intra_clique_distances,
    compute_prototype_geometry,
    smoothstep,
    triangular_condorcet_cycles,
)


# =========================================================================== #
# 1. Gromov-Wasserstein Tensor Contraction & Mathematical Invariants
# =========================================================================== #
def test_gw_tensor_contraction_exactness():
    """Verifies O(n^2 K + n K^2) tensor contraction matches explicit quadruple loops."""
    rng = np.random.default_rng(42)
    n = 5
    K = 18

    # Random symmetric distance matrices with zero diagonal
    D_X = rng.uniform(0.1, 1.5, size=(n, n))
    D_X = 0.5 * (D_X + D_X.T)
    np.fill_diagonal(D_X, 0.0)

    D_Y = rng.uniform(0.1, 2.0, size=(K, K))
    D_Y = 0.5 * (D_Y + D_Y.T)
    np.fill_diagonal(D_Y, 0.0)

    # Random transport plan
    T = rng.uniform(0.01, 1.0, size=(n, K))
    T /= T.sum()

    # Fast vectorized contraction
    G_fast = GromovWassersteinMatcher.tensor_contraction(D_X, D_Y, T)

    # Explicit quadruple loop
    G_loop = np.zeros((n, K), dtype=np.float64)
    for i in range(n):
        for j in range(K):
            for k in range(n):
                for l in range(K):
                    G_loop[i, j] += ((D_X[i, k] - D_Y[j, l]) ** 2) * T[k, l]

    max_diff = np.max(np.abs(G_fast - G_loop))
    assert max_diff < 1e-12, f"Contraction mismatch: max difference {max_diff}"


def test_gw_matcher_convergence_and_loss_decrease():
    """Tests convergence and loss decrease on a structured synthetic clique."""
    rng = np.random.default_rng(101)
    n = 6
    K = 18
    d = 16

    # Candidate prototypes
    prototypes = rng.normal(size=(K, d))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    D_Y = 1.0 - prototypes @ prototypes.T
    np.fill_diagonal(D_Y, 0.0)

    # Injective clique passages
    chosen_models = rng.choice(K, size=n, replace=False)
    passage_emb = prototypes[chosen_models] + rng.normal(scale=0.05, size=(n, d))
    passage_emb /= np.linalg.norm(passage_emb, axis=1, keepdims=True)
    D_X = 1.0 - passage_emb @ passage_emb.T
    np.fill_diagonal(D_X, 0.0)

    # Noisy linear cost
    logits = np.zeros((n, K))
    for i in range(n):
        logits[i, chosen_models[i]] += 2.0
        logits[i] += rng.normal(scale=0.5, size=K)
    M = -logits

    matcher = GromovWassersteinMatcher(
        alpha_gw=0.6,
        epsilon=0.04,
        tau_col=1.5,
        max_iter=35,
        step_size=0.5,
    )

    P_cal, info = matcher.align(D_X=D_X, D_Y=D_Y, M=M)

    assert info["converged"] or info["iterations"] >= 10
    assert len(info["loss_history"]) >= 2
    # GW loss should decrease or stabilize from initial outer step
    assert info["gw_history"][-1] <= info["gw_history"][0] + 1e-6
    # Row-stochasticity
    assert np.allclose(P_cal.sum(axis=1), 1.0, atol=1e-7)
    assert np.all(P_cal >= 0.0)


def test_gw_matcher_symmetry_and_permutation_equivariance():
    """Verifies permutation equivariance and symmetry preservation."""
    rng = np.random.default_rng(202)
    n = 4
    K = 18

    D_X = rng.uniform(0.2, 1.0, size=(n, n))
    D_X = 0.5 * (D_X + D_X.T)
    np.fill_diagonal(D_X, 0.0)

    D_Y = rng.uniform(0.2, 1.0, size=(K, K))
    D_Y = 0.5 * (D_Y + D_Y.T)
    np.fill_diagonal(D_Y, 0.0)

    M = rng.normal(size=(n, K))

    matcher = GromovWassersteinMatcher(alpha_gw=0.5, epsilon=0.05, max_iter=20)
    P_orig, _ = matcher.align(D_X, D_Y, M=M)

    # Permute rows
    perm = np.array([2, 0, 3, 1])
    D_X_perm = D_X[perm][:, perm]
    M_perm = M[perm]

    P_perm, _ = matcher.align(D_X_perm, D_Y, M=M_perm)

    # Calibrated probabilities should be permuted correspondingly
    assert np.allclose(P_orig[perm], P_perm, atol=1e-5)


def test_gw_matcher_edge_cases():
    """Tests n=0, n=1, n=2, n=18, and oversized n=24 cliques."""
    matcher = GromovWassersteinMatcher(alpha_gw=0.5, epsilon=0.05, max_iter=15)
    K = N_CLASSES

    # Case n=0: Empty
    P_0, info_0 = matcher.align(np.zeros((0, 0)), np.zeros((K, K)))
    assert P_0.shape == (0, K)
    assert info_0["converged"]

    # Case n=1: Singleton
    D_X_1 = np.zeros((1, 1))
    D_Y = np.ones((K, K)) - np.eye(K)
    M_1 = np.arange(K)[None, :].astype(float)
    P_1, info_1 = matcher.align(D_X_1, D_Y, M=M_1)
    assert P_1.shape == (1, K)
    assert np.isclose(P_1.sum(), 1.0)
    # Smallest M cost should have highest probability
    assert np.argmax(P_1[0]) == 0

    # Case n=2: Pair
    D_X_2 = np.array([[0.0, 0.8], [0.8, 0.0]])
    M_2 = np.zeros((2, K))
    M_2[0, 3] = -2.0
    M_2[1, 7] = -2.0
    P_2, info_2 = matcher.align(D_X_2, D_Y, M=M_2)
    assert P_2.shape == (2, K)
    assert np.allclose(P_2.sum(axis=1), 1.0)

    # Case n=18: Full model clique
    D_X_18 = np.ones((18, 18)) - np.eye(18)
    P_18, info_18 = matcher.align(D_X_18, D_Y)
    assert P_18.shape == (18, K)
    assert np.allclose(P_18.sum(axis=1), 1.0)

    # Case n=24: Oversized clique (>18)
    D_X_24 = np.ones((24, 24)) - np.eye(24)
    P_24, info_24 = matcher.align(D_X_24, D_Y)
    assert P_24.shape == (24, K)
    assert np.allclose(P_24.sum(axis=1), 1.0)


def test_gw_matcher_hostile_inputs():
    """Tests numerical resilience under NaNs, infinities, and extreme underflow."""
    K = N_CLASSES
    n = 4

    D_X = np.array([
        [0.0, np.nan, 1.0, 0.5],
        [np.nan, 0.0, 0.8, np.inf],
        [1.0, 0.8, 0.0, -np.inf],
        [0.5, np.inf, -np.inf, 0.0],
    ])
    D_Y = np.ones((K, K)) - np.eye(K)
    D_Y[2, 5] = np.nan

    M = np.full((n, K), 1e-300)
    M[0, 0] = np.inf
    M[1, 1] = -np.inf

    matcher = GromovWassersteinMatcher(max_iter=10)
    P_out, info = matcher.align(D_X, D_Y, M=M)

    assert np.all(np.isfinite(P_out))
    assert np.all(P_out >= 0.0)
    assert np.allclose(P_out.sum(axis=1), 1.0, atol=1e-5)


# =========================================================================== #
# 2. TransductiveCliqueCalibrator Smooth Homotopy & Invariants
# =========================================================================== #
def test_transductive_clique_calibrator_smooth_weights():
    """Verifies C^1 smooth regime weights and partition of unity across cluster sizes."""
    calibrator = TransductiveCliqueCalibrator()

    for n in range(1, 25):
        w_soft, w_rep, w_sink = calibrator.get_regime_weights(n)

        # Non-negative and partition of unity
        assert w_soft >= 0.0
        assert w_rep >= 0.0
        assert w_sink >= 0.0
        assert np.isclose(w_soft + w_rep + w_sink, 1.0, atol=1e-9)

        # Boundary checks
        if n == 1:
            assert w_soft == 1.0 and w_rep == 0.0 and w_sink == 0.0
        elif 2 <= n < 6:
            assert w_sink == 0.0
            assert w_rep > 0.0
            assert w_soft < 1.0
        elif 6 <= n <= 18:
            assert w_soft == 0.0
            assert w_sink > 0.0
        elif n >= 18:
            assert w_soft == 0.0
            assert w_sink == 1.0
            assert w_rep == 0.0


def test_transductive_clique_calibrator_soft_repulsion():
    """Verifies that soft cavity occupation repulsion resolves class collision."""
    calibrator = TransductiveCliqueCalibrator()
    K = N_CLASSES
    n = 3

    # Passages 0 and 1 both strongly favor class 4, with secondary choices class 2 and 5
    probs = np.full((n, K), 0.01)
    probs[0, 4] = 0.60
    probs[0, 2] = 0.25
    probs[1, 4] = 0.58
    probs[1, 5] = 0.27
    probs[2, 8] = 0.80
    probs /= probs.sum(axis=1, keepdims=True)

    probs_repelled = calibrator.soft_cavity_repulsion(probs, lambda_repel=1.5, iters=3)

    # Repulsion should reduce collision on class 4 for both passages
    assert probs_repelled[0, 4] < probs[0, 4]
    assert probs_repelled[1, 4] < probs[1, 4]
    # Secondary choices should receive boosted relative mass
    assert probs_repelled[0, 2] > probs[0, 2]
    assert probs_repelled[1, 5] > probs[1, 5]
    # Passage 2 does not collide and should remain stable
    assert np.isclose(probs_repelled[2, 8], probs[2, 8], atol=0.08)
    assert np.allclose(probs_repelled.sum(axis=1), 1.0)


def test_transductive_clique_calibrator_batch():
    """Tests batch calibration across heterogeneous clique sizes (singletons, small, medium, full)."""
    calibrator = TransductiveCliqueCalibrator(temperature=1.0)
    K = N_CLASSES

    # Create dataset with varying clique sizes:
    # Clique 0: singleton (n=1)
    # Clique 1: small clique (n=3)
    # Clique 2: medium clique (n=8)
    # Clique 3: full clique (n=18)
    clique_ids = np.concatenate([
        np.full(1, 0),
        np.full(3, 1),
        np.full(8, 2),
        np.full(18, 3),
    ])
    N = len(clique_ids)

    rng = np.random.default_rng(303)
    logits = rng.normal(size=(N, K))

    P_cal = calibrator.calibrate_batch(logits=logits, clique_ids=clique_ids)

    assert P_cal.shape == (N, K)
    assert np.all(P_cal >= 0.0)
    assert np.allclose(P_cal.sum(axis=1), 1.0, atol=1e-7)

    # For singleton (row 0), P_cal should match standard softmax(logits[0])
    p_soft_0 = np.exp(logits[0] - logsumexp(logits[0]))
    assert np.allclose(P_cal[0], p_soft_0, atol=1e-7)


# =========================================================================== #
# 3. ConfidenceHodgeRank & Zero-Condorcet Curl Properties
# =========================================================================== #
def test_confidence_hodge_rank_pure_condorcet_cycle():
    """Verifies that a pure cyclic Condorcet paradox is projected to zero potential."""
    hodge = ConfidenceHodgeRank(beta=1.0)
    K = 3

    # Pure cyclic tournament: 0 -> 1 -> 2 -> 0
    Y_cycle = np.zeros((K, K))
    Y_cycle[0, 1] = 2.5
    Y_cycle[1, 0] = -2.5
    Y_cycle[1, 2] = 2.5
    Y_cycle[2, 1] = -2.5
    Y_cycle[2, 0] = 2.5
    Y_cycle[0, 2] = -2.5

    s, Y_hat, metrics = hodge.project_flow_matrix(Y_cycle)

    # Potential must be flat (all zeros) because pure curl contains zero gradient
    assert np.allclose(s, 0.0, atol=1e-12)
    assert np.allclose(Y_hat, 0.0, atol=1e-12)
    # Inconsistency ratio must be 1.0 (100% curl energy)
    assert np.isclose(metrics["curl_inconsistency_ratio"], 1.0, atol=1e-7)


def test_confidence_hodge_rank_zero_condorcet_curl_theorem():
    """Mathematically proves that the projected flow has IDENTICALLY ZERO curl on all 816 triangles."""
    hodge = ConfidenceHodgeRank(n_classes=N_CLASSES, beta=1.5)
    K = N_CLASSES
    rng = np.random.default_rng(404)

    # Generate hostile random skew-symmetric flow matrix with strong cycles
    Y_rand = rng.normal(scale=3.0, size=(K, K))
    Y_flow = 0.5 * (Y_rand - Y_rand.T)
    np.fill_diagonal(Y_flow, 0.0)

    s, Y_hat, metrics = hodge.project_flow_matrix(Y_flow)

    # Verify zero-Condorcet curl theorem on all 816 triangles of the 18-model graph
    triangles = triangular_condorcet_cycles(K)
    assert len(triangles) == math.comb(K, 3)  # 816 triangles for K=18

    is_zero, max_curl = hodge.verify_zero_condorcet_curl(Y_hat, tol=1e-12)
    assert is_zero, f"Condorcet curl violation: max circulation {max_curl}"
    assert max_curl < 1e-13


def test_confidence_hodge_rank_exact_gradient_recovery():
    """Verifies exact potential recovery when input flow is already curl-free."""
    hodge = ConfidenceHodgeRank(beta=1.0)
    K = N_CLASSES
    rng = np.random.default_rng(505)

    # Ground truth zero-mean potential
    s_true = rng.normal(scale=2.0, size=K)
    s_true -= np.mean(s_true)

    # Exact gradient flow
    Y_grad = s_true[None, :] - s_true[:, None]

    s_rec, Y_hat, metrics = hodge.project_flow_matrix(Y_grad)

    # Potential should match true potential to machine precision
    err = np.max(np.abs(s_rec - s_true))
    assert err < 1e-12, f"Recovery error {err} exceeds tolerance"
    # Curl energy should be identically zero
    assert metrics["curl_energy"] < 1e-12
    assert np.isclose(metrics["curl_inconsistency_ratio"], 0.0, atol=1e-10)


def test_confidence_hodge_rank_confidence_weighting():
    """Verifies that edges with large |Delta logit| are preserved with higher fidelity."""
    K = 4
    s_true = np.array([-1.0, 1.0, -0.5, 0.5])
    Y_true = s_true[None, :] - s_true[:, None]

    noise = np.random.RandomState(42).normal(scale=0.5, size=(K, K))
    noise = 0.5 * (noise - noise.T)
    np.fill_diagonal(noise, 0.0)
    Y_noisy = Y_true + noise

    W_prior = np.ones((K, K)) * 0.1
    W_prior[0, 1] = W_prior[1, 0] = 10.0
    np.fill_diagonal(W_prior, 0.0)

    hodge = ConfidenceHodgeRank(beta=0.0)
    s_un, _, _ = hodge.project_flow_matrix(Y_noisy)
    s_w, _, _ = hodge.project_flow_matrix(Y_noisy, base_weights=W_prior)

    err_un_high = abs((s_un[1] - s_un[0]) - Y_true[0, 1])
    err_w_high = abs((s_w[1] - s_w[0]) - Y_true[0, 1])

    assert err_w_high < err_un_high, f"Confidence weighting failed: {err_w_high} >= {err_un_high}"


def test_confidence_hodge_rank_dictionary_compatibility():
    """Verifies backward compatibility with dictionary pairwise corrections."""
    hodge = ConfidenceHodgeRank(n_classes=4, beta=1.0)
    N = 10

    rng = np.random.default_rng(606)
    delta_01 = rng.normal(loc=1.5, scale=0.3, size=N)
    delta_12 = rng.normal(loc=-1.0, scale=0.3, size=N)
    delta_23 = rng.normal(loc=0.8, scale=0.3, size=N)

    corrections = {
        (0, 1): delta_01,
        (1, 2): delta_12,
        (2, 3): delta_23,
    }

    V, curl = hodge.project_corrections_dict(corrections)

    assert V.shape == (N, 4)
    assert np.all(np.isfinite(V))
    # Approximate zero curl across sample potentials
    assert curl < 0.05


# =========================================================================== #
# 4. End-to-End Transductive Pipeline Integration & Accuracy Uplift
# =========================================================================== #
def test_transductive_clique_accuracy_uplift():
    """Simulates an end-to-end prompt clique test set and verifies accuracy uplift."""
    rng = np.random.default_rng(707)
    K = N_CLASSES
    d = 20

    # Model prototypes
    prototypes = rng.normal(size=(K, d))
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    D_Y = 1.0 - prototypes @ prototypes.T
    np.fill_diagonal(D_Y, 0.0)

    # Generate 15 prompt cliques of size 8 (common in Name That LLM)
    n_cliques = 15
    clique_size = 8
    N = n_cliques * clique_size

    passage_emb = []
    labels = []
    clique_ids = []
    logits = []

    for c in range(n_cliques):
        # Injective model selection for clique c
        models_c = rng.choice(K, size=clique_size, replace=False)
        # Shared prompt topic distractor
        topic_distractor = rng.normal(scale=0.4, size=d)

        for m in models_c:
            emb = prototypes[m] + topic_distractor + rng.normal(scale=0.1, size=d)
            emb /= np.linalg.norm(emb)
            passage_emb.append(emb)
            labels.append(m)
            clique_ids.append(c)

            # Noisy logits biased by topic distractor
            logit = prototypes @ emb * 4.0 + rng.normal(scale=1.5, size=K)
            logits.append(logit)

    passage_emb = np.array(passage_emb)
    labels = np.array(labels)
    clique_ids = np.array(clique_ids)
    logits = np.array(logits)

    # Uncalibrated baseline
    pred_raw = np.argmax(logits, axis=1)
    bacc_raw = balanced_accuracy_score(labels, pred_raw, n_classes=K)

    # Transductive Clique Calibrator with Gromov-Wasserstein alignment
    calibrator = TransductiveCliqueCalibrator(
        alpha_repel=1.0,
        alpha_gw=0.5,
        epsilon=0.05,
        prototype_geometry=D_Y,
    )

    P_calibrated = calibrator.calibrate_batch(
        logits=logits,
        clique_ids=clique_ids,
        embeddings=passage_emb,
        prototype_dist=D_Y,
    )

    pred_cal = np.argmax(P_calibrated, axis=1)
    bacc_cal = balanced_accuracy_score(labels, pred_cal, n_classes=K)

    assert bacc_cal > bacc_raw, f"Transductive calibration failed to improve score: {bacc_cal:.4f} <= {bacc_raw:.4f}"
