import numpy as np
import pytest
from src.config import VALID_18_MODELS, MODEL_TO_IDX
from src.eval_stats import length_stratified_balanced_accuracy
from src.hierarchical_classifier import HierarchicalAttributionHead, compute_typographic_priors


def test_length_stratified_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    texts = [
        "word " * 45,   # bucket <60
        "word " * 50,   # bucket <60
        "word " * 75,   # bucket 60-90
        "word " * 80,   # bucket 60-90
        "word " * 150,  # bucket 90-200
        "word " * 160,  # bucket 90-200
    ]
    res = length_stratified_balanced_accuracy(y_true, y_pred, texts, bins=[0, 60, 90, 200])
    assert "0-60" in res
    assert "60-90" in res
    assert "90-200" in res
    for bin_name, score in res.items():
        assert 0.0 <= score <= 1.0


def test_fuse_with_length_gate():
    head = HierarchicalAttributionHead()
    N = 2
    logits_lab = np.zeros((N, 6))
    logits_siblings = {lab: np.zeros((N, count)) for lab, count in {
        "anthropic": 5, "google": 4, "openai": 5, "qwen": 1, "zai": 2, "moonshot": 1
    }.items()}
    
    texts = [
        "Short text with spaced em — dash.",
        "This is a substantially longer text passage designed to test standard length passages."
    ]
    word_counts = np.array([6, 13])
    typo_priors = compute_typographic_priors(texts)
    
    # Test with threshold_words = 10:
    # First sample has 6 words (< 10) -> gets short_weight (0.50)
    # Second sample has 13 words (>= 10) -> gets standard_weight (0.20)
    P = head.fuse_with_length_gate(
        logits_lab,
        logits_siblings,
        typo_priors,
        word_counts,
        threshold_words=10,
        short_weight=0.50,
        standard_weight=0.20
    )
    assert P.shape == (N, 18)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    
    # First sample has spaced em-dash -> with high typo weight, claude-opus-4-6 must have high prob
    idx_opus46 = MODEL_TO_IDX["claude-opus-4-6"]
    idx_sonnet = MODEL_TO_IDX["claude-sonnet-5"]
    assert P[0, idx_opus46] > P[0, idx_sonnet]
