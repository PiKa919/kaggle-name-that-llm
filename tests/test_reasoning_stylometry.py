"""
Unit tests for src/reasoning_stylometry.py.
Verifies all mathematical invariants, edge cases, watermark proxies,
discourse Markov chains, and dedicated intra-family sibling discriminators.
Zero local training required.
"""

import numpy as np
import pytest

from src.config import MODEL_TO_IDX, VALID_18_MODELS
from src.reasoning_stylometry import (
    N_REASONING_FEATURES,
    REASONING_FEATURE_DOCS,
    REASONING_FEATURE_NAMES,
    TARGET_SIBLING_PAIRS,
    DedicatedSiblingDiscriminator,
    ReasoningPairwiseInjector,
    extract_raw_reasoning_features,
    extract_reasoning_features,
    fuse_reasoning_stylometry,
    ls_potential,
)


# --------------------------------------------------------------------------- #
# 1. Schema and Documentation Invariants
# --------------------------------------------------------------------------- #

def test_feature_schema_integrity():
    assert len(REASONING_FEATURE_NAMES) == N_REASONING_FEATURES
    assert len(set(REASONING_FEATURE_NAMES)) == N_REASONING_FEATURES
    assert len(REASONING_FEATURE_DOCS) == N_REASONING_FEATURES

    for name in REASONING_FEATURE_NAMES:
        doc = REASONING_FEATURE_DOCS[name]
        assert doc and len(doc.strip()) > 20, f"Feature {name} lacks substantive docstring"

    assert len(TARGET_SIBLING_PAIRS) >= 5
    for m1, m2 in TARGET_SIBLING_PAIRS:
        assert m1 in MODEL_TO_IDX, f"Model {m1} not in MODEL_TO_IDX"
        assert m2 in MODEL_TO_IDX, f"Model {m2} not in MODEL_TO_IDX"


# --------------------------------------------------------------------------- #
# 2. Degenerate and Boundary Inputs
# --------------------------------------------------------------------------- #

def test_degenerate_batches():
    # Empty list
    empty = extract_reasoning_features([])
    assert empty.shape == (0, N_REASONING_FEATURES)
    assert empty.dtype == np.float64

    # None input
    none_res = extract_reasoning_features(None)
    assert none_res.shape == (0, N_REASONING_FEATURES)

    # Empty string passage
    empty_str = extract_reasoning_features([""])
    assert empty_str.shape == (1, N_REASONING_FEATURES)
    assert np.isfinite(empty_str).all()

    # Whitespace and punctuation only
    punct_only = extract_raw_reasoning_features(["   \n\n  --- ... ??? !!!  "])
    assert punct_only.shape == (1, N_REASONING_FEATURES)
    assert np.isfinite(punct_only).all()

    # Single-word passage
    single_word = extract_raw_reasoning_features(["Incomprehensibility"])
    assert single_word.shape == (1, N_REASONING_FEATURES)
    assert np.isfinite(single_word).all()

    # Single-sample batch with robust_z scaling should safely normalize to 0
    single_scaled = extract_reasoning_features(["A normal complete sentence here."], method="robust_z")
    assert np.allclose(single_scaled, 0.0)

    # Constant batch
    constant = extract_reasoning_features(["Identical passage repeated."] * 5, method="robust_z")
    assert np.allclose(constant, 0.0)


def test_scaling_modes():
    texts = [
        "First short sentence with simple words.",
        "A second longer and more descriptive sentence containing complex vocabulary.",
        "Third sentence featuring numbers 1,234 and 5,678 for diversity.",
        "Fourth sentence introducing contrasting ideas; however, evidence remains clear.",
    ]
    raw = extract_raw_reasoning_features(texts)
    assert raw.shape == (4, N_REASONING_FEATURES)
    assert np.isfinite(raw).all()

    none_mode = extract_reasoning_features(texts, method="none")
    assert np.allclose(raw, none_mode)

    z_mode = extract_reasoning_features(texts, method="robust_z", clip=5.0)
    assert z_mode.shape == (4, N_REASONING_FEATURES)
    assert np.isfinite(z_mode).all()
    assert (z_mode >= -5.0).all() and (z_mode <= 5.0).all()

    q_mode = extract_reasoning_features(texts, method="quantile")
    assert q_mode.shape == (4, N_REASONING_FEATURES)
    assert np.isfinite(q_mode).all()
    assert (q_mode >= -1.0).all() and (q_mode <= 1.0).all()

    with pytest.raises(ValueError):
        extract_reasoning_features(texts, method="invalid_mode")


# --------------------------------------------------------------------------- #
# 3. Discourse Markov Transition Dynamics
# --------------------------------------------------------------------------- #

def test_discourse_markov_transitions():
    idx_map = {n: i for i, n in enumerate(REASONING_FEATURE_NAMES)}

    # Causal chain passage (therefore -> thus -> consequently)
    causal_text = (
        "The premise holds without question. "
        "Therefore, the initial hypothesis must be validated. "
        "Consequently, all subsequent derivations follow necessarily. "
        "Thus, the conclusion is firmly established."
    )
    raw_causal = extract_raw_reasoning_features([causal_text])
    assert raw_causal[0, idx_map["discourse_causal_share"]] > 0.50
    assert raw_causal[0, idx_map["trans_causal_to_causal"]] > 0.25

    # Contrastive to grounded passage (however -> for example)
    contrast_grounded_text = (
        "The standard theory posits absolute convergence. "
        "However, empirical experiments frequently deviate from theoretical predictions. "
        "For example, in clinical trials we observe widespread variance across cohorts."
    )
    raw_cg = extract_raw_reasoning_features([contrast_grounded_text])
    assert raw_cg[0, idx_map["discourse_contrastive_share"]] > 0.20
    assert raw_cg[0, idx_map["discourse_grounded_share"]] > 0.20
    assert raw_cg[0, idx_map["trans_contrast_to_grounded"]] > 0.20

    # Question followed by answer
    q_text = (
        "Why does the loss function oscillate so violently during training? "
        "In practice, the learning rate is set too high for the AdamW optimizer."
    )
    raw_q = extract_raw_reasoning_features([q_text])
    assert raw_q[0, idx_map["question_to_explanation_ratio"]] == 1.0


# --------------------------------------------------------------------------- #
# 4. Subword Boundary Fingerprints and Typography
# --------------------------------------------------------------------------- #

def test_subword_boundary_fingerprints():
    idx_map = {n: i for i, n in enumerate(REASONING_FEATURE_NAMES)}

    # Curly double quotes vs straight double quotes
    curly_text = "\u201cThis is an elegant quotation,\u201d she whispered. \u201cIndeed.\u201d"
    straight_text = '"This is an elegant quotation," she whispered. "Indeed."'

    res = extract_raw_reasoning_features([curly_text, straight_text])
    assert res[0, idx_map["curly_dq_ratio"]] > 0.95
    assert res[1, idx_map["curly_dq_ratio"]] == 0.0

    # Spaced em-dash vs unspaced em-dash
    spaced_dash = "The experimental hypothesis \u2014 although bold \u2014 was thoroughly verified."
    unspaced_dash = "The experimental hypothesis\u2014although bold\u2014was thoroughly verified."

    res_dash = extract_raw_reasoning_features([spaced_dash, unspaced_dash])
    assert res_dash[0, idx_map["spaced_em_dash_ratio"]] > 0.95
    assert res_dash[1, idx_map["spaced_em_dash_ratio"]] == 0.0
    assert res_dash[1, idx_map["unspaced_em_dash_rate"]] > 0.0

    # Contraction suppression (GLM formal style vs Claude natural style)
    formal_text = "We do not believe this claim. It is not plausible and we cannot agree."
    contracted_text = "We don't believe this claim. It's not plausible and we can't agree."

    res_contr = extract_raw_reasoning_features([formal_text, contracted_text])
    assert res_contr[0, idx_map["contraction_suppression_index"]] > 0.90
    assert res_contr[1, idx_map["contraction_suppression_index"]] < 0.20

    # Number formatting
    fmt_num_text = "The company recorded 1,250,000 active users and 3.1415 percent churn."
    raw_num_text = "The company recorded 1250000 active users and 3 percent churn."

    res_num = extract_raw_reasoning_features([fmt_num_text, raw_num_text])
    assert res_num[0, idx_map["formatted_number_ratio"]] >= 0.40
    assert res_num[1, idx_map["formatted_number_ratio"]] == 0.0


# --------------------------------------------------------------------------- #
# 5. Neural Watermark Entropy Proxies
# --------------------------------------------------------------------------- #

def test_watermark_entropy_proxies():
    idx_map = {n: i for i, n in enumerate(REASONING_FEATURE_NAMES)}

    passage = (
        "In artificial intelligence and machine learning, deep neural networks "
        "have revolutionized natural language processing, computer vision, and "
        "reinforcement learning. Large language models exhibit remarkable capabilities "
        "in reasoning, translation, and code generation across diverse benchmarks."
    )
    raw = extract_raw_reasoning_features([passage])

    # Zipfian slope should be positive and close to 1.0
    assert raw[0, idx_map["zipf_beta_slope"]] > 0.0
    assert raw[0, idx_map["zipf_residual_mse"]] >= 0.0

    # Determinism of Kirchenbauer pseudo-green score
    raw2 = extract_raw_reasoning_features([passage])
    assert raw[0, idx_map["kirchenbauer_green_zscore"]] == raw2[0, idx_map["kirchenbauer_green_zscore"]]
    assert np.isfinite(raw[0, idx_map["kirchenbauer_green_zscore"]])

    # Entropies are strictly non-negative
    assert raw[0, idx_map["token_entropy_unigram"]] > 0.0
    assert raw[0, idx_map["token_entropy_bigram"]] >= raw[0, idx_map["token_entropy_unigram"]]
    assert raw[0, idx_map["conditional_entropy_proxy"]] >= 0.0


# --------------------------------------------------------------------------- #
# 6. Dedicated Sibling Pair Discriminators
# --------------------------------------------------------------------------- #

def test_dedicated_sibling_discriminators_zero_shot():
    disc = DedicatedSiblingDiscriminator()

    # Pair 1: Gemini 3.7 Flash vs Gemini 3.8 Flash
    # 3.8 Flash has curly double quotes; 3.7 Flash uses straight double quotes
    text_gemini_38 = "\u201cNotice the profound implications of this theorem,\u201d the author noted."
    text_gemini_37 = '"Notice the profound implications of this theorem," the author noted.'

    deltas_38 = disc.predict_pair_log_odds([text_gemini_38])
    deltas_37 = disc.predict_pair_log_odds([text_gemini_37])

    pair_gemini = ("gemini-3.7-flash", "gemini-3.8-flash")
    # Positive indicates m2 (3.8-flash) favored; negative indicates m1 (3.7-flash) favored
    assert deltas_38[pair_gemini][0] > 0.0
    assert deltas_37[pair_gemini][0] < 0.0

    # Pair 2: Claude Opus 4-6 vs Claude Opus 5
    # Opus 4-6 has high spaced em-dash and reflective grounding
    text_opus_46 = (
        "The philosophical dilemma \u2014 profound as it appeared \u2014 was carefully evaluated. "
        "However, empirical observations showed subtle deviations. "
        "For example, human participants consistently resisted deterministic choices."
    )
    deltas_opus = disc.predict_pair_log_odds([text_opus_46])
    pair_opus = ("claude-opus-4-6", "claude-opus-5")
    # Negative indicates Opus 4-6 favored
    assert deltas_opus[pair_opus][0] < 0.0

    # Pair 3: GPT-5.6 Luna vs Sol vs Terra
    # Luna: Causal chain
    text_luna = (
        "Therefore, the gradient of the potential function vanishes at critical points. "
        "Consequently, the system remains in a stationary state. "
        "Thus, stability is guaranteed."
    )
    # Terra: Grounded empirical with formatted digits
    text_terra = (
        "In practice, the operational pipeline processed 2,450 records in 12.5 seconds. "
        "For example, during peak hours we observed 99.4 percent reliability."
    )
    deltas_luna = disc.predict_pair_log_odds([text_luna])
    deltas_terra = disc.predict_pair_log_odds([text_terra])

    pair_ls = ("gpt-5.6-luna", "gpt-5.6-sol")
    pair_st = ("gpt-5.6-sol", "gpt-5.6-terra")
    # For Luna text: negative favors Luna in (luna, sol)
    assert deltas_luna[pair_ls][0] < 0.0
    # For Terra text: positive favors Terra in (sol, terra)
    assert deltas_terra[pair_st][0] > 0.0

    # Pair 4: GLM-5.3 vs Claude Opus 4-6
    # GLM has unspaced em-dash and zero contractions
    text_glm = "The computational complexity\u2014measured accurately\u2014does not increase. We cannot accept approximations."
    text_claude = "The computational complexity \u2014 measured accurately \u2014 doesn't increase. We can't accept approximations."

    deltas_glm = disc.predict_pair_log_odds([text_glm])
    deltas_claude = disc.predict_pair_log_odds([text_claude])

    pair_glm_claude = ("glm-5.3", "claude-opus-4-6")
    assert deltas_glm[pair_glm_claude][0] < 0.0
    assert deltas_claude[pair_glm_claude][0] > 0.0


def test_dedicated_sibling_discriminators_supervised_fit():
    # Synthetic dataset for two classes
    N = 30
    texts = []
    y = []
    idx_37 = MODEL_TO_IDX["gemini-3.7-flash"]
    idx_38 = MODEL_TO_IDX["gemini-3.8-flash"]

    for i in range(N // 2):
        texts.append(f'Sample {i} using "straight quotes" and standard diction.')
        y.append(idx_37)
    for i in range(N // 2):
        texts.append(f"Sample {i} using \u201ccurly double quotes\u201d and reflective questions? In practice it works.")
        y.append(idx_38)

    y = np.array(y, dtype=int)
    disc = DedicatedSiblingDiscriminator()
    disc.fit(texts, y)

    pair = ("gemini-3.7-flash", "gemini-3.8-flash")
    assert pair in disc.classifiers_

    deltas = disc.predict_pair_log_odds(texts)
    assert pair in deltas
    # Predictions for 38 should be positive, for 37 should be negative
    assert deltas[pair][N // 2:].mean() > deltas[pair][:N // 2].mean()


# --------------------------------------------------------------------------- #
# 7. HodgeRank Potential and Probability Fusion
# --------------------------------------------------------------------------- #

def test_ls_potential_cycle_freedom():
    pairs = [
        ("gemini-3.7-flash", "gemini-3.8-flash"),
        ("claude-opus-4-6", "claude-opus-5"),
        ("gpt-5.6-luna", "gpt-5.6-sol"),
        ("gpt-5.6-sol", "gpt-5.6-terra"),
    ]
    N = 5
    terms = {
        ("gemini-3.7-flash", "gemini-3.8-flash"): np.array([1.5, -1.0, 0.5, 0.0, -2.0]),
        ("claude-opus-4-6", "claude-opus-5"): np.array([-1.2, 0.8, -0.4, 1.0, 0.0]),
        ("gpt-5.6-luna", "gpt-5.6-sol"): np.array([0.5, 0.5, -0.5, -0.5, 0.0]),
        ("gpt-5.6-sol", "gpt-5.6-terra"): np.array([1.0, -1.0, 1.0, -1.0, 0.0]),
    }

    v = ls_potential(pairs, terms, n_classes=18)
    assert v.shape == (N, 18)
    assert np.isfinite(v).all()

    # Zero mass transfer to untouched classes (e.g. qwen3.8-max or kimi-k3)
    idx_qwen = MODEL_TO_IDX["qwen3.8-max"]
    idx_kimi = MODEL_TO_IDX["kimi-k3"]
    assert np.allclose(v[:, idx_qwen], 0.0)
    assert np.allclose(v[:, idx_kimi], 0.0)


def test_fuse_reasoning_stylometry_properties():
    N = 4
    K = len(VALID_18_MODELS)
    P_base = np.full((N, K), 1.0 / K)

    texts = [
        "\u201cClear evidence,\u201d she said.",
        '"Clear evidence," she said.',
        "Therefore, consequently, thus it is proven.",
        "In practice, 1,000 items were processed successfully.",
    ]

    P_fused = fuse_reasoning_stylometry(P_base, texts, weight=0.20)
    assert P_fused.shape == (N, K)
    assert np.isfinite(P_fused).all()
    assert (P_fused >= 0.0).all()
    assert np.allclose(P_fused.sum(axis=1), 1.0)

    # For text 0 (curly quotes), Gemini 3.8 Flash probability should be boosted relative to Gemini 3.7 Flash
    idx_38 = MODEL_TO_IDX["gemini-3.8-flash"]
    idx_37 = MODEL_TO_IDX["gemini-3.7-flash"]
    assert P_fused[0, idx_38] > P_fused[0, idx_37]

    # For text 1 (straight quotes), Gemini 3.7 Flash probability should be boosted relative to Gemini 3.8 Flash
    assert P_fused[1, idx_37] > P_fused[1, idx_38]
