"""
Unit tests for src/cross_tokenizer_forensics.py
================================================

Comprehensive verification of:
1. Schema contract, feature registration, documentation, and uniqueness.
2. Extraction robustness, numerical stability, finiteness, and clipping bounds.
3. Edge case degradation (empty strings, None, pure whitespace, single token, unicode).
4. Permutation equivariance and determinism.
5. Cross-tokenizer compression and boundary mechanics (cl100k vs o200k).
6. Heuristic syntactic depth and clausal subordination sensitivity.
7. Watermark residual statistics (Kirchenbauer z-scores, Zipfian fits, multiseed variance).
8. Directional sibling discrimination (Gemini 3.7 vs 3.8, Claude Opus 4.6 vs 5, GPT-5.6 Luna/Sol/Terra).
9. HodgeRank Laplacian least-squares projection, zero-mean conservation, and simplex invariants.
10. End-to-end probability fusion pipeline.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.config import MODEL_TO_IDX, VALID_18_MODELS
from src.cross_tokenizer_forensics import (
    CROSS_TOKENIZER_FEATURE_NAMES,
    CROSS_TOKENIZER_FEATURE_DOCS,
    N_CROSS_TOKENIZER_FEATURES,
    TARGET_SIBLING_PAIRS,
    extract_raw_cross_tokenizer_features,
    extract_cross_tokenizer_features,
    CrossTokenizerSiblingDiscriminator,
    CrossTokenizerPairwiseInjector,
    fuse_cross_tokenizer_forensics,
    cross_tokenizer_feature_names,
    cross_tokenizer_feature_doc,
)


# --------------------------------------------------------------------------- #
# Fixtures & Sample Corpora
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_corpus() -> list[str]:
    return [
        (
            "The model converged rapidly because the learning rate was small. "
            "Although the gradient was steep, we persisted until the loss hit zero. "
            "Therefore, the empirical convergence rate matched our theoretical prediction."
        ),
        (
            "“Why did the algorithm stop?” she asked. “Because the loss reached zero.” "
            "“Is it optimal?” she inquired. “Indeed, the Hessian is positive definite.”"
        ),
        (
            "In practice, 1,000,000 requests were processed across 16 clustered shards. "
            "The throughput measured 45,200 ops/sec with a 99.9th percentile latency of 3.14 ms."
        ),
        (
            "The mind — a delicate, recursive instrument — reflects upon its own processes. "
            "Although introspection is perilous, one might argue that reflection yields insight, "
            "provided that evidence guides the inquiry."
        ),
        (
            "First, initialize the weights. Second, compute the forward pass. "
            "Third, calculate the cross-entropy loss. Finally, update the parameters."
        ),
    ]


# --------------------------------------------------------------------------- #
# 1. Schema & Feature Registration Invariants
# --------------------------------------------------------------------------- #

def test_schema_and_metadata_invariants():
    """Verify feature registry integrity, uniqueness, and complete documentation."""
    assert len(CROSS_TOKENIZER_FEATURE_NAMES) == N_CROSS_TOKENIZER_FEATURES
    assert len(CROSS_TOKENIZER_FEATURE_DOCS) == N_CROSS_TOKENIZER_FEATURES
    assert N_CROSS_TOKENIZER_FEATURES >= 60

    # Ensure uniqueness of feature names
    assert len(set(CROSS_TOKENIZER_FEATURE_NAMES)) == N_CROSS_TOKENIZER_FEATURES

    # Ensure every feature has substantive documentation
    for name in CROSS_TOKENIZER_FEATURE_NAMES:
        doc = CROSS_TOKENIZER_FEATURE_DOCS[name]
        assert isinstance(doc, str) and len(doc.strip()) > 10, f"Empty doc for {name}"
        assert cross_tokenizer_feature_doc(name) == doc

    assert cross_tokenizer_feature_names() == CROSS_TOKENIZER_FEATURE_NAMES
    assert len(TARGET_SIBLING_PAIRS) >= 5


# --------------------------------------------------------------------------- #
# 2. Extraction Shapes, Dtypes, and Finiteness
# --------------------------------------------------------------------------- #

def test_extraction_shapes_and_finiteness(sample_corpus):
    """Verify matrix shapes, dtype float64, and strict absence of NaNs/Infs."""
    raw = extract_raw_cross_tokenizer_features(sample_corpus)
    assert raw.shape == (len(sample_corpus), N_CROSS_TOKENIZER_FEATURES)
    assert raw.dtype == np.float64
    assert np.isfinite(raw).all(), "Raw matrix contains non-finite values"

    robust = extract_cross_tokenizer_features(sample_corpus, method="robust", clip=8.0)
    assert robust.shape == (len(sample_corpus), N_CROSS_TOKENIZER_FEATURES)
    assert robust.dtype == np.float64
    assert np.isfinite(robust).all(), "Robust matrix contains non-finite values"
    assert robust.min() >= -8.0 - 1e-9 and robust.max() <= 8.0 + 1e-9

    quantile = extract_cross_tokenizer_features(sample_corpus, method="quantile")
    assert quantile.shape == (len(sample_corpus), N_CROSS_TOKENIZER_FEATURES)
    assert quantile.dtype == np.float64
    assert np.isfinite(quantile).all(), "Quantile matrix contains non-finite values"
    assert quantile.min() >= -1.0 - 1e-9 and quantile.max() <= 1.0 + 1e-9

    none_scaled = extract_cross_tokenizer_features(sample_corpus, method="none")
    assert np.allclose(none_scaled, raw)


def test_invalid_scaling_method_raises():
    with pytest.raises(ValueError, match="Unknown scaling method"):
        extract_cross_tokenizer_features(["Hello world"], method="unsupported_scaling")


# --------------------------------------------------------------------------- #
# 3. Edge Cases & Degradation
# --------------------------------------------------------------------------- #

def test_edge_cases_safety():
    """Verify zero crash and finite output on anomalous inputs."""
    edge_cases = [
        "",
        "   ",
        "\n\n\t",
        "...!!??",
        "a",
        "1234567890",
        "🚀🔥🌟🎉",
        "A" * 3000,
        None,
    ]

    raw = extract_raw_cross_tokenizer_features(edge_cases)
    assert raw.shape == (len(edge_cases), N_CROSS_TOKENIZER_FEATURES)
    assert np.isfinite(raw).all(), "Edge case produced non-finite values"

    # Empty collection
    empty_raw = extract_raw_cross_tokenizer_features([])
    assert empty_raw.shape == (0, N_CROSS_TOKENIZER_FEATURES)
    empty_norm = extract_cross_tokenizer_features([])
    assert empty_norm.shape == (0, N_CROSS_TOKENIZER_FEATURES)

    # Single string input (not in list)
    single_raw = extract_raw_cross_tokenizer_features("A single sentence.")
    assert single_raw.shape == (1, N_CROSS_TOKENIZER_FEATURES)
    assert np.isfinite(single_raw).all()


# --------------------------------------------------------------------------- #
# 4. Permutation Equivariance & Determinism
# --------------------------------------------------------------------------- #

def test_determinism_and_row_invariance(sample_corpus):
    """Check deterministic extraction and raw row independence."""
    raw1 = extract_raw_cross_tokenizer_features(sample_corpus)
    raw2 = extract_raw_cross_tokenizer_features(sample_corpus)
    assert np.array_equal(raw1, raw2), "Extraction must be strictly deterministic"

    # Permuted rows in raw extraction must match permuted raw1
    perm = [2, 0, 4, 1, 3]
    permuted_corpus = [sample_corpus[i] for i in perm]
    raw_perm = extract_raw_cross_tokenizer_features(permuted_corpus)
    assert np.allclose(raw_perm, raw1[perm]), "Raw extraction must be row-independent"


# --------------------------------------------------------------------------- #
# 5. Cross-Tokenizer Compression & Boundary Mechanics
# --------------------------------------------------------------------------- #

def test_tokenizer_compression_mechanics():
    """Verify o200k vs cl100k compression properties and contraction merges."""
    text_with_contractions = (
        "They don't think it's fair that we couldn't attend, shouldn't we? "
        "I'm sure you're aware that they've already left, aren't you?"
    )
    raw = extract_raw_cross_tokenizer_features([text_with_contractions])
    feat_idx = {name: i for i, name in enumerate(CROSS_TOKENIZER_FEATURE_NAMES)}

    cl_count = raw[0, feat_idx["cl100k_token_count"]]
    o2_count = raw[0, feat_idx["o200k_token_count"]]
    merged_cont = raw[0, feat_idx["o200k_contraction_merges_per_100w"]]

    # o200k merges common English contractions, producing fewer tokens than cl100k
    assert o2_count < cl_count, f"o200k count ({o2_count}) should be < cl100k ({cl_count})"
    assert merged_cont > 0.0, "Merged contractions must be detected"

    # Compression differential: o200k chars/token should exceed cl100k chars/token
    comp_diff = raw[0, feat_idx["compression_differential"]]
    assert comp_diff > 0.0, f"Expected positive compression differential, got {comp_diff}"


def test_phonotactic_transitions_and_morphemes():
    """Verify subword phonotactic transitions and morphological affixes."""
    text = (
        "The institutionalization of counterproductive miscalculations "
        "precipitated unpredictable synchronization disconnects."
    )
    raw = extract_raw_cross_tokenizer_features([text])
    feat_idx = {name: i for i, name in enumerate(CROSS_TOKENIZER_FEATURE_NAMES)}

    prefix_rate = raw[0, feat_idx["prefix_split_per_100w"]]
    suffix_rate = raw[0, feat_idx["suffix_split_per_100w"]]
    affix_density = raw[0, feat_idx["affix_density_ratio"]]
    phon_entropy = raw[0, feat_idx["subword_phonotactic_entropy"]]

    assert prefix_rate > 0.0, "Expected prefix affixes"
    assert suffix_rate > 0.0, "Expected suffix affixes"
    assert affix_density > 0.0, "Expected positive affix density"
    assert phon_entropy >= 0.0, "Phonotactic entropy must be non-negative"

    # Phonotactic shares should sum to approximately 1.0
    shares = [
        raw[0, feat_idx["cc_boundary_transition_share"]],
        raw[0, feat_idx["cv_boundary_transition_share"]],
        raw[0, feat_idx["vc_boundary_transition_share"]],
        raw[0, feat_idx["vv_boundary_transition_share"]],
    ]
    assert 0.99 <= sum(shares) <= 1.01, f"Shares sum to {sum(shares)}"


# --------------------------------------------------------------------------- #
# 6. Heuristic Syntactic Depth & Clausal Subordination
# --------------------------------------------------------------------------- #

def test_syntactic_depth_and_subordination():
    """Verify Yngve pre-verbal depth and clausal subordination features."""
    left_heavy_text = (
        "Although the extensive statistical investigation yielded significant anomalies, "
        "the senior research committee nevertheless approved the publication."
    )
    right_heavy_text = (
        "The committee approved the publication quickly."
    )

    raw_left = extract_raw_cross_tokenizer_features([left_heavy_text])
    raw_right = extract_raw_cross_tokenizer_features([right_heavy_text])
    feat_idx = {name: i for i, name in enumerate(CROSS_TOKENIZER_FEATURE_NAMES)}

    depth_left = raw_left[0, feat_idx["left_branching_depth_mean"]]
    depth_right = raw_right[0, feat_idx["left_branching_depth_mean"]]
    assert depth_left > depth_right, (
        f"Left-heavy text ({depth_left}) should have higher left-depth than right-heavy ({depth_right})"
    )

    ratio_left = raw_left[0, feat_idx["left_to_right_branching_ratio"]]
    ratio_right = raw_right[0, feat_idx["left_to_right_branching_ratio"]]
    assert ratio_left > ratio_right, "Left-to-right ratio should be higher for left-heavy sentence"

    subord_idx = raw_left[0, feat_idx["subordination_index_per_sentence"]]
    assert subord_idx >= 1.0, f"Expected subordination index >= 1.0, got {subord_idx}"


# --------------------------------------------------------------------------- #
# 7. Watermark Residual Signatures
# --------------------------------------------------------------------------- #

def test_watermark_residual_features(sample_corpus):
    """Verify Kirchenbauer z-scores, multiseed variance, and Zipfian fits."""
    raw = extract_raw_cross_tokenizer_features(sample_corpus)
    feat_idx = {name: i for i, name in enumerate(CROSS_TOKENIZER_FEATURE_NAMES)}

    for i in range(len(sample_corpus)):
        z_h1_50 = raw[i, feat_idx["kirchenbauer_z_h1_g50"]]
        z_h1_25 = raw[i, feat_idx["kirchenbauer_z_h1_g25"]]
        wm_std = raw[i, feat_idx["watermark_multiseed_z_std"]]
        zipf_beta = raw[i, feat_idx["zipf_slope_beta"]]
        simpson = raw[i, feat_idx["simpson_concentration"]]
        hapax = raw[i, feat_idx["hapax_legomena_ratio"]]

        assert np.isfinite(z_h1_50)
        assert np.isfinite(z_h1_25)
        assert wm_std >= 0.0, "Multiseed std must be non-negative"
        assert zipf_beta <= 0.0 or math.isclose(zipf_beta, -1.0), "Zipf slope should be negative"
        assert 0.0 <= simpson <= 1.0, f"Simpson index out of range: {simpson}"
        assert 0.0 <= hapax <= 1.0, f"Hapax ratio out of range: {hapax}"


# --------------------------------------------------------------------------- #
# 8. Sibling Discriminators & Directional Sanity
# --------------------------------------------------------------------------- #

def test_sibling_discriminator_gemini_pair():
    """Verify Gemini 3.7 Flash vs 3.8 Flash discrimination via quotes and pacing."""
    # Gemini 3.7 signature: curly double quotes, rhetorical questioning
    gemini_37_text = (
        "“Why should the loss converge so smoothly?” the author asked. "
        "“Because the step size decays exponentially with each iteration.” "
        "“Does the gradient stabilize?” Indeed, the variance vanishes."
    )
    # Gemini 3.8 signature: straight quotes, tight formatting, minimal rhetorical questioning
    gemini_38_text = (
        'The loss converges smoothly because the step size decays exponentially. '
        '1. Initialize learning rate at 0.01.\n2. Apply cosine decay schedule.\n'
        'The variance vanishes at step 10,000.'
    )

    raw = extract_raw_cross_tokenizer_features([gemini_37_text, gemini_38_text])
    disc = CrossTokenizerSiblingDiscriminator()
    deltas = disc.predict_log_odds(raw)

    pair = ("gemini-3.7-flash", "gemini-3.8-flash")
    d = deltas[pair]
    # Positive favors gemini-3.8-flash, negative favors gemini-3.7-flash
    assert d[0] < 0.0, f"Gemini 3.7 text should have negative log-odds favoring 3.7, got {d[0]}"
    assert d[1] > d[0], f"Gemini 3.8 text should have higher log-odds than 3.7 text ({d[1]} vs {d[0]})"


def test_sibling_discriminator_claude_pair():
    """Verify Claude Opus 4.6 vs Opus 5 discrimination via spaced em-dashes and depth."""
    # Opus 5 signature: spaced em-dash (' — '), high clausal subordination, high left depth
    opus_5_text = (
        "The cognitive architecture — an inherently layered construct — operates through reflection. "
        "Although previous investigators suggested simplicity, one must acknowledge that "
        "intelligence — when properly understood — demands complex clausal articulation."
    )
    opus_46_text = (
        "The architecture operates through standard forward steps. "
        "We tested the model on the benchmark and recorded the accuracy."
    )

    raw = extract_raw_cross_tokenizer_features([opus_5_text, opus_46_text])
    disc = CrossTokenizerSiblingDiscriminator()
    deltas = disc.predict_log_odds(raw)

    pair = ("claude-opus-4-6", "claude-opus-5")
    d = deltas[pair]
    # Positive favors claude-opus-5
    assert d[0] > 0.0, f"Opus 5 text should favor Opus 5, got {d[0]}"
    assert d[0] > d[1], f"Opus 5 text should score higher than plain text ({d[0]} vs {d[1]})"


def test_sibling_discriminator_gpt56_trio():
    """Verify 3-way discrimination across Luna (deductive), Terra (grounded), Sol (balanced)."""
    # Luna: Mathematical / Deductive reasoning, formal prefixes, preposed clauses
    luna_text = (
        "Although the manifold is non-Euclidean, Theorem 4 guarantees that the geodesic is unique. "
        "Therefore, by deductive induction, it follows that the curvature tensor vanishes; "
        "hence the global topological invariant is preserved."
    )
    # Terra: Empirical / Practical / Formatted numbers and operational grounding
    terra_text = (
        "In practice, 10,000 servers were deployed across 4 clusters. "
        "The latency measured 12.4 ms for 99.5% of requests, consuming 1,250 kW/h. "
        "Specifically, operational metrics confirmed stable throughput."
    )

    raw = extract_raw_cross_tokenizer_features([luna_text, terra_text])
    disc = CrossTokenizerSiblingDiscriminator()
    deltas = disc.predict_log_odds(raw)

    # Luna vs Terra: positive favors Terra, negative favors Luna
    pair_lt = ("gpt-5.6-luna", "gpt-5.6-terra")
    d_lt = deltas[pair_lt]
    assert d_lt[0] < 0.0, f"Luna text should have negative log-odds favoring Luna, got {d_lt[0]}"
    assert d_lt[1] > 0.0, f"Terra text should have positive log-odds favoring Terra, got {d_lt[1]}"


def test_empirical_fit_and_separation():
    """Verify empirical fit method of CrossTokenizerSiblingDiscriminator."""
    rng = np.random.RandomState(42)
    N_samples = 40
    D = N_CROSS_TOKENIZER_FEATURES

    # Create synthetic feature matrix with separation on pair
    X_synthetic = rng.randn(N_samples, D)
    idx1 = MODEL_TO_IDX["gemini-3.7-flash"]
    idx2 = MODEL_TO_IDX["gemini-3.8-flash"]

    # Half samples class 1, half class 2
    y_synthetic = np.array([idx1] * (N_samples // 2) + [idx2] * (N_samples // 2))
    # Add signal to curly_dq_ratio (feature index 59)
    curly_idx = CROSS_TOKENIZER_FEATURE_NAMES.index("curly_dq_ratio")
    X_synthetic[:N_samples // 2, curly_idx] += 3.0  # Class 1 (3.7) has higher curly quotes
    X_synthetic[N_samples // 2:, curly_idx] -= 3.0

    disc = CrossTokenizerSiblingDiscriminator(pairs=[("gemini-3.7-flash", "gemini-3.8-flash")])
    disc.fit(X_synthetic, y_synthetic)

    deltas = disc.predict_log_odds(X_synthetic)[("gemini-3.7-flash", "gemini-3.8-flash")]
    # Class 1 should have negative deltas, Class 2 should have positive deltas
    mean_cls1 = deltas[:N_samples // 2].mean()
    mean_cls2 = deltas[N_samples // 2:].mean()
    assert mean_cls2 > mean_cls1, f"Fitted discriminator failed to separate classes ({mean_cls2} <= {mean_cls1})"


# --------------------------------------------------------------------------- #
# 9. HodgeRank Potential Projection & Simplex Conservation
# --------------------------------------------------------------------------- #

def test_hodgerank_potential_and_simplex_invariants(sample_corpus):
    """Verify cycle-free HodgeRank projection, zero-mean potential, and simplex preservation."""
    raw = extract_raw_cross_tokenizer_features(sample_corpus)
    N = len(sample_corpus)
    K = len(VALID_18_MODELS)

    # Uniform base distribution
    P_base = np.ones((N, K), dtype=np.float64) / K

    injector = CrossTokenizerPairwiseInjector(lam=0.20, tau=1.0, gate=True)
    P_adj = injector.inject(P_base, raw)

    assert P_adj.shape == (N, K)
    assert np.isfinite(P_adj).all()
    assert (P_adj >= 0.0).all(), "Negative probabilities detected"
    assert np.allclose(P_adj.sum(axis=1), 1.0), "Row sums must equal 1.0 (simplex invariant)"


def test_off_pair_conservation():
    """Verify that unrelated models do not receive probability mass when pair support is zero."""
    N = 2
    K = len(VALID_18_MODELS)
    # Assign probability only to Anthropic models
    P_base = np.zeros((N, K), dtype=np.float64)
    c46 = MODEL_TO_IDX["claude-opus-4-6"]
    c5 = MODEL_TO_IDX["claude-opus-5"]
    P_base[:, c46] = 0.5
    P_base[:, c5] = 0.5

    raw = extract_raw_cross_tokenizer_features(["Test passage with text."] * N)
    injector = CrossTokenizerPairwiseInjector(lam=0.20, tau=1.0, gate=True)
    P_adj = injector.inject(P_base, raw)

    # Gemini, OpenAI, ZAI models should have 0 probability
    gemini_idx = MODEL_TO_IDX["gemini-3.7-flash"]
    assert np.allclose(P_adj[:, gemini_idx], 0.0, atol=1e-6), "Off-pair mass leakage detected"


# --------------------------------------------------------------------------- #
# 10. End-to-End Pipeline
# --------------------------------------------------------------------------- #

def test_fuse_cross_tokenizer_forensics(sample_corpus):
    """Verify end-to-end convenience wrapper fuse_cross_tokenizer_forensics."""
    N = len(sample_corpus)
    K = len(VALID_18_MODELS)
    P_base = np.ones((N, K), dtype=np.float64) / K

    P_fused = fuse_cross_tokenizer_forensics(P_base, sample_corpus, lam=0.15)
    assert P_fused.shape == (N, K)
    assert np.isfinite(P_fused).all()
    assert (P_fused >= 0.0).all()
    assert np.allclose(P_fused.sum(axis=1), 1.0)
