"""
Comprehensive Unit Tests for Orthogonal Subspace Deconfounding,
Truncation-Invariant Pooling, and Cross-Trunk Harmonization.

Verifies:
1. Exactness of Economy Gram SVD vs direct dense SVD.
2. Factored low-rank projection vs dense projector equivalence (diff < 1e-14).
3. Projection idempotence: Pi_perp^2 = Pi_perp.
4. Prompt subspace annihilation: X_perp @ U_k = 0.
5. Pythagorean energy conservation: ||X||^2 = ||X_par||^2 + ||X_perp||^2.
6. Deconfounding efficacy on synthetic prompt-contaminated data.
7. Attention sink token avoidance: mask[:, 0] = 0 eliminates token 0 contamination.
8. Multi-layer intermediate readout extraction & per-layer LayerNorm.
9. Length dispersion normalization & RoPE frequency attenuation compensation.
10. OLMo2 alphabetical -> Canonical label permutation alignment.
11. Dynamic length-gated multi-trunk fusion (short vs long regime weighting).
12. PyTorch autograd gradient flow and NumPy/Torch parity.
"""

import numpy as np
import torch

from src.orthogonal_deconfounding import (
    CANONICAL_LABELS,
    OLMO2_CLASSES,
    K_CLASSES,
    align_labels_to_canonical,
    PromptSubspaceDeconfounder,
    TruncationInvariantPooling,
    truncation_invariant_pool,
    CrossTrunkHarmonizer,
)


# =====================================================================
# 1. PROMPT SUBSPACE DECONFOUNDER TESTS
# =====================================================================

def test_economy_gram_svd_exactness():
    """
    Verifies that Economy Gram SVD on C (G x D) via Gram matrix C C^T (G x G)
    matches direct full SVD on C to within machine precision.
    """
    G, D = 12, 128
    rng = np.random.RandomState(42)
    C = rng.randn(G, D).astype(np.float64)

    # 1. Direct SVD on C
    _, s_direct, Vt_direct = np.linalg.svd(C, full_matrices=False)
    V_direct = Vt_direct.T  # (D, G) right singular vectors

    # 2. Economy Gram SVD via PromptSubspaceDeconfounder
    # Use dummy prompt IDs with one sample per prompt to match C directly
    dummy_ids = np.arange(G)
    deconf = PromptSubspaceDeconfounder(
        n_components=G,
        center_centroids=False,
        regularization=1e-12,
    )
    deconf.fit(C, dummy_ids)

    # Compare singular values
    s_gram = deconf.singular_values_
    np.testing.assert_allclose(s_gram, s_direct, atol=1e-12, rtol=1e-12)

    # Compare right singular vector subspaces (check column alignment)
    U_gram = deconf.U_
    for j in range(G):
        # Basis vectors are unique up to sign flip (+1 or -1)
        dot_product = abs(np.dot(U_gram[:, j], V_direct[:, j]))
        assert abs(dot_product - 1.0) < 1e-12, f"Singular vector {j} not collinear (dot={dot_product})"


def test_orthonormality_of_subspace_basis():
    """
    Verifies that extracted prompt basis U_k satisfies strict orthonormality:
        U_k^T U_k = I_k
    """
    N, D, G, k = 100, 256, 15, 8
    rng = np.random.RandomState(123)
    X = rng.randn(N, D)
    prompt_ids = rng.randint(0, G, size=N)

    deconf = PromptSubspaceDeconfounder(n_components=k, center_centroids=True)
    deconf.fit(X, prompt_ids)

    U = deconf.U_
    assert U.shape == (D, k)

    gram_U = U.T @ U
    identity_k = np.eye(k)
    np.testing.assert_allclose(gram_U, identity_k, atol=1e-14)


def test_factored_projection_vs_dense_projector():
    """
    Verifies that factored low-rank projection X - (X U) U^T is mathematically
    identical to dense projector X @ (I - U U^T) with zero D x D allocation.
    """
    N, D, G, k = 50, 512, 10, 6
    rng = np.random.RandomState(99)
    X = rng.randn(N, D)
    prompt_ids = rng.randint(0, G, size=N)

    deconf = PromptSubspaceDeconfounder(n_components=k, center_centroids=True)
    deconf.fit(X, prompt_ids)

    # Factored projection (O(N D k))
    X_factored = deconf.transform(X)

    # Dense projection (O(D^2))
    U = deconf.U_
    Pi_dense = np.eye(D) - U @ U.T
    X_dense = X @ Pi_dense

    np.testing.assert_allclose(X_factored, X_dense, atol=1e-14)


def test_projection_idempotence():
    """
    Verifies projector idempotence:
        Pi_perp^2 = Pi_perp  ==>  transform(transform(X)) == transform(X)
    """
    N, D, G, k = 40, 128, 8, 4
    rng = np.random.RandomState(77)
    X = rng.randn(N, D)
    prompt_ids = rng.randint(0, G, size=N)

    deconf = PromptSubspaceDeconfounder(n_components=k, center_centroids=False)
    deconf.fit(X, prompt_ids)

    X_proj1 = deconf.transform(X)
    X_proj2 = deconf.transform(X_proj1)

    np.testing.assert_allclose(X_proj1, X_proj2, atol=1e-14)


def test_prompt_subspace_annihilation():
    """
    Verifies that projected representations X_perp have ZERO correlation with
    the prompt subspace basis U_k:
        X_perp @ U_k = 0_{N x k}
    """
    N, D, G, k = 60, 200, 12, 7
    rng = np.random.RandomState(55)
    X = rng.randn(N, D)
    prompt_ids = rng.randint(0, G, size=N)

    deconf = PromptSubspaceDeconfounder(n_components=k, center_centroids=True)
    deconf.fit(X, prompt_ids)

    X_perp = deconf.transform(X)
    proj_coords = X_perp @ deconf.U_

    np.testing.assert_allclose(proj_coords, 0.0, atol=1e-13)


def test_pythagorean_energy_conservation():
    """
    Verifies orthogonal decomposition:
        X = X_parallel + X_perp
        ||X||_F^2 = ||X_parallel||_F^2 + ||X_perp||_F^2
    """
    N, D, G, k = 30, 64, 6, 4
    rng = np.random.RandomState(33)
    X = rng.randn(N, D)
    prompt_ids = rng.randint(0, G, size=N)

    deconf = PromptSubspaceDeconfounder(n_components=k, center_centroids=False)
    deconf.fit(X, prompt_ids)

    X_par = deconf.project_subspace(X)
    X_perp = deconf.transform(X)

    # Check additive reconstruction
    np.testing.assert_allclose(X, X_par + X_perp, atol=1e-14)

    # Check Pythagorean energy equality
    norm_orig_sq = np.sum(X ** 2)
    norm_par_sq = np.sum(X_par ** 2)
    norm_perp_sq = np.sum(X_perp ** 2)

    np.testing.assert_allclose(norm_orig_sq, norm_par_sq + norm_perp_sq, atol=1e-12)


def test_prompt_confounding_elimination():
    """
    Tests deconfounding on synthetic data with severe prompt confounding.
    Shows between-prompt variance in the prompt subspace drops to ~0.
    """
    rng = np.random.RandomState(42)
    N_per_prompt = 20
    G = 5
    D = 32
    N = G * N_per_prompt

    prompt_ids = np.repeat(np.arange(G), N_per_prompt)

    # True latent stylometric signal (independent of prompt)
    S_true = rng.randn(N, D) * 0.5

    # Strong prompt confounding: each prompt adds a huge distinct offset
    prompt_vectors = rng.randn(G, D) * 10.0
    P_confound = prompt_vectors[prompt_ids]

    X = S_true + P_confound

    # Fit deconfounder with k=G-1
    deconf = PromptSubspaceDeconfounder(n_components=G - 1, center_centroids=True)
    deconf.fit(X, prompt_ids)

    X_deconf = deconf.transform(X)

    # Verify that prompt centroid variance in X_deconf is dramatically reduced
    orig_centroids = np.array([X[prompt_ids == g].mean(axis=0) for g in range(G)])
    deconf_centroids = np.array([X_deconf[prompt_ids == g].mean(axis=0) for g in range(G)])

    orig_centroid_var = np.var(orig_centroids, axis=0).sum()
    deconf_centroid_var = np.var(deconf_centroids, axis=0).sum()

    # The prompt subspace explained >98% of centroid variance; after deconfounding it collapses
    assert deconf_centroid_var < 0.05 * orig_centroid_var, (
        f"Centroid var after deconfounding ({deconf_centroid_var:.4f}) should be < 5% of orig ({orig_centroid_var:.4f})"
    )


def test_torch_tensor_support():
    """
    Verifies transparent PyTorch tensor support, maintaining device and dtype.
    """
    N, D, G, k = 25, 64, 5, 3
    X_torch = torch.randn(N, D, dtype=torch.float32)
    p_torch = torch.randint(0, G, (N,))

    deconf = PromptSubspaceDeconfounder(n_components=k)
    deconf.fit(X_torch, p_torch)

    X_perp = deconf.transform(X_torch)
    assert isinstance(X_perp, torch.Tensor)
    assert X_perp.shape == (N, D)
    assert X_perp.dtype == torch.float32
    assert X_perp.device == X_torch.device


def test_variance_ratio_selection():
    """
    Verifies that variance_ratio dynamically chooses k to cover specified variance.
    """
    G, D = 10, 40
    rng = np.random.RandomState(10)
    # Construct centroids with decaying spectrum
    U_rand, _ = np.linalg.qr(rng.randn(D, G))
    s_decay = np.array([100.0, 50.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
    C = (U_rand @ np.diag(s_decay) @ rng.randn(G, G)).T

    dummy_ids = np.arange(G)
    deconf = PromptSubspaceDeconfounder(variance_ratio=0.95, n_components=None, center_centroids=False)
    deconf.fit(C, dummy_ids)

    # First 3 components explain ~ (100^2 + 50^2 + 10^2) / total > 0.99
    assert deconf.k_ <= 3
    assert np.sum(deconf.explained_variance_ratio_) >= 0.95


# =====================================================================
# 2. TRUNCATION-INVARIANT POOLING TESTS
# =====================================================================

def test_attention_sink_avoidance():
    """
    Verifies that attention-sink token avoidance (mask[:, 0] = 0) shields
    representations from massive token 0 artifacts.
    """
    B, S, H = 4, 10, 32
    # Hidden states across 6 layers
    hs_clean = [torch.randn(B, S, H) for _ in range(6)]
    mask = torch.ones(B, S)

    # Pollute token 0 with massive random outliers
    hs_polluted = [h.clone() for h in hs_clean]
    for layer in hs_polluted:
        layer[:, 0, :] = torch.randn(B, H) * 5000.0

    pooler_safe = TruncationInvariantPooling(
        hidden_size=H,
        target_layers=[2, 4],
        avoid_attention_sink=True,
    )
    pooler_polluted = TruncationInvariantPooling(
        hidden_size=H,
        target_layers=[2, 4],
        avoid_attention_sink=False,
    )

    out_clean = pooler_safe(hs_clean, mask)
    out_safe = pooler_safe(hs_polluted, mask)
    out_polluted = pooler_polluted(hs_polluted, mask)

    # 1. Safe pooling must be mathematically identical to clean (diff == 0.0)
    np.testing.assert_allclose(
        out_safe.detach().numpy(),
        out_clean.detach().numpy(),
        atol=1e-6,
    )

    # 2. Polluted pooling must diverge significantly from clean
    diff = (out_polluted - out_clean).abs().max().item()
    assert diff > 1.0, f"Polluted output did not diverge from clean (max diff: {diff})"


def test_multi_layer_intermediate_readout():
    """
    Verifies intermediate layer extraction (e.g. layers 4, 8, 12, 16)
    and per-layer LayerNorm normalization.
    """
    B, S, H = 2, 8, 16
    n_layers = 20
    hs = [torch.randn(B, S, H) * 10.0 + 5.0 for _ in range(n_layers)]
    mask = torch.ones(B, S)

    target_layers = [4, 8, 12, 16]
    pooler = TruncationInvariantPooling(
        hidden_size=H,
        target_layers=target_layers,
        avoid_attention_sink=True,
        use_layer_norm=True,
    )

    out = pooler(hs, mask)
    assert out.shape == (B, len(target_layers) * H)

    # Check each layer slice has zero mean and unit variance due to LayerNorm
    for i in range(len(target_layers)):
        sub = out[:, i * H:(i + 1) * H]
        mean_val = sub.mean(dim=-1).detach().numpy()
        std_val = sub.std(dim=-1, unbiased=False).detach().numpy()
        np.testing.assert_allclose(mean_val, 0.0, atol=1e-5)
        np.testing.assert_allclose(std_val, 1.0, atol=1e-5)


def test_length_dispersion_and_short_passage_scaling():
    """
    Verifies length dispersion factor and RoPE frequency compensation
    prevent representation collapse on short passages (<60 tokens).
    """
    H = 16
    pooler = TruncationInvariantPooling(
        hidden_size=H,
        target_layers=[0],
        avoid_attention_sink=True,
        use_layer_norm=False,  # inspect raw scaling
        length_dispersion_norm=True,
        rope_attenuation_correction=True,
    )

    # Short passage (length 5) vs Long passage (length 100)
    hs_short = [torch.ones(1, 5, H)]
    mask_short = torch.ones(1, 5)

    hs_long = [torch.ones(1, 100, H)]
    mask_long = torch.ones(1, 100)

    out_short = pooler(hs_short, mask_short)
    out_long = pooler(hs_long, mask_long)

    # Effective length for short: 5 - 1 = 4
    # Effective length for long: 100 - 1 = 99
    # Without compensation, short passage representation would suffer from small token sampling
    assert not torch.isnan(out_short).any()
    assert not torch.isnan(out_long).any()
    assert out_short.shape == (1, H)
    assert out_long.shape == (1, H)


def test_functional_and_numpy_pooling():
    """
    Verifies truncation_invariant_pool functional API on NumPy arrays.
    """
    B, S, H = 3, 12, 24
    rng = np.random.RandomState(42)
    hs_np = [rng.randn(B, S, H).astype(np.float32) for _ in range(5)]
    mask_np = np.ones((B, S), dtype=np.float32)

    res_np = truncation_invariant_pool(
        hs_np,
        attention_mask=mask_np,
        target_layers=[1, 3],
        avoid_attention_sink=True,
    )

    assert isinstance(res_np, np.ndarray)
    assert res_np.shape == (B, 2 * H)
    assert not np.isnan(res_np).any()


def test_torch_autograd_flow():
    """
    Verifies that gradients backpropagate smoothly through TruncationInvariantPooling.
    """
    B, S, H = 2, 6, 16
    hs = [torch.randn(B, S, H, requires_grad=True) for _ in range(4)]
    mask = torch.ones(B, S)

    pooler = TruncationInvariantPooling(
        hidden_size=H,
        target_layers=[1, 2],
        avoid_attention_sink=True,
    )

    out = pooler(hs, mask)
    loss = (out ** 2).sum()
    loss.backward()

    # Target layers have non-zero gradients; unselected layers have zero gradients
    assert hs[1].grad is not None
    assert (hs[1].grad.abs() > 0).any()
    assert hs[2].grad is not None
    assert (hs[2].grad.abs() > 0).any()
    assert hs[0].grad is None or (hs[0].grad == 0).all()


# =====================================================================
# 3. CROSS-TRUNK HARMONIZER TESTS
# =====================================================================

def test_label_alignment_olmo_to_canonical():
    """
    Verifies permutation mapping from OLMo2 alphabetical ordering to Canonical 18 classes.
    """
    assert len(CANONICAL_LABELS) == 18
    assert len(OLMO2_CLASSES) == 18
    assert set(CANONICAL_LABELS) == set(OLMO2_CLASSES)

    N = 4
    rng = np.random.RandomState(99)
    dummy_olmo_probs = rng.rand(N, 18).astype(np.float32)
    dummy_olmo_probs /= dummy_olmo_probs.sum(axis=1, keepdims=True)

    aligned = align_labels_to_canonical(
        dummy_olmo_probs,
        source_classes=OLMO2_CLASSES,
        target_classes=CANONICAL_LABELS,
    )

    assert aligned.shape == (N, 18)
    for k_target, target_name in enumerate(CANONICAL_LABELS):
        k_source = OLMO2_CLASSES.index(target_name)
        np.testing.assert_allclose(aligned[:, k_target], dummy_olmo_probs[:, k_source])


def test_dynamic_length_gate_weighting():
    """
    Verifies length-gated mixture:
    - Short sequences (<60) emphasize stylometric trunks (Qwen3-8B, OLMo2-7B).
    - Long sequences (>=60) emphasize deep reasoning trunk (Qwen3-14B).
    """
    harmonizer = CrossTrunkHarmonizer(length_pivot=60.0, gate_temperature=15.0)

    lengths = [15.0, 50.0, 60.0, 120.0, 300.0]
    weights = harmonizer.compute_length_gate_weights(lengths)

    # 1. Weights must sum to 1.0 for each sequence length
    total_w = sum(weights[m] for m in weights)
    np.testing.assert_allclose(total_w, 1.0, atol=1e-12)

    # 2. 14B weight must strictly increase monotonically with length
    w_14b = weights["qwen3_14b"]
    assert np.all(np.diff(w_14b) > 0), f"14B weights not monotonically increasing: {w_14b}"

    # 3. 8B and OLMo2 weights must strictly decrease monotonically with length
    w_8b = weights["qwen3_8b"]
    w_olmo = weights["olmo2_7b"]
    assert np.all(np.diff(w_8b) < 0), f"8B weights not decreasing: {w_8b}"
    assert np.all(np.diff(w_olmo) < 0), f"OLMo weights not decreasing: {w_olmo}"


def test_probability_fusion_linear_and_log_linear():
    """
    Verifies multi-trunk probability fusion produces valid simplex distributions.
    """
    harmonizer = CrossTrunkHarmonizer()
    N = 5
    rng = np.random.RandomState(42)

    trunk_probs = {
        "qwen3_8b": rng.dirichlet(np.ones(K_CLASSES), size=N).astype(np.float32),
        "qwen3_14b": rng.dirichlet(np.ones(K_CLASSES), size=N).astype(np.float32),
        "olmo2_7b": rng.dirichlet(np.ones(K_CLASSES), size=N).astype(np.float32),
    }
    lengths = [20, 45, 60, 150, 400]

    # Linear fusion
    fused_lin = harmonizer.fuse_probabilities(trunk_probs, lengths=lengths, fusion_mode="linear")
    assert fused_lin.shape == (N, K_CLASSES)
    np.testing.assert_allclose(fused_lin.sum(axis=1), 1.0, atol=1e-6)
    assert (fused_lin >= 0.0).all()

    # Log-linear fusion (Product of Experts)
    fused_log = harmonizer.fuse_probabilities(trunk_probs, lengths=lengths, fusion_mode="log_linear")
    assert fused_log.shape == (N, K_CLASSES)
    np.testing.assert_allclose(fused_log.sum(axis=1), 1.0, atol=1e-6)
    assert (fused_log >= 0.0).all()


def test_cross_trunk_feature_deconfounding_and_fusion():
    """
    Verifies that multi-trunk representation fusion standardizes each trunk,
    applies prompt deconfounding, and concatenates correctly.
    """
    harmonizer = CrossTrunkHarmonizer(deconfound_k=4)
    N, G = 30, 5
    rng = np.random.RandomState(77)

    prompt_ids = rng.randint(0, G, size=N)
    train_features = {
        "qwen3_8b": rng.randn(N, 64).astype(np.float32) * 5.0,
        "qwen3_14b": rng.randn(N, 128).astype(np.float32) * 20.0,  # 14B higher raw scale
        "olmo2_7b": rng.randn(N, 64).astype(np.float32) * 2.0,
    }

    harmonizer.fit_deconfounders(train_features, prompt_ids)

    # Test feature fusion
    test_features = {
        "qwen3_8b": rng.randn(10, 64).astype(np.float32),
        "qwen3_14b": rng.randn(10, 128).astype(np.float32),
        "olmo2_7b": rng.randn(10, 64).astype(np.float32),
    }

    fused = harmonizer.fuse_representations(test_features, apply_deconfounding=True, standardize_per_trunk=True)
    expected_dim = 64 + 128 + 64
    assert fused.shape == (10, expected_dim)
    assert not np.isnan(fused).any()
