import numpy as np
import pytest
from src.config import VALID_18_MODELS, MODEL_TO_LAB

LAB_COUNTS = {
    "anthropic": 5,
    "google": 4,
    "openai": 5,
    "qwen": 1,
    "zai": 2,
    "moonshot": 1,
}

def test_hierarchical_probability_sums():
    from src.hierarchical_classifier import HierarchicalAttributionHead
    head = HierarchicalAttributionHead()
    N = 15
    
    # Dummy lab logits for 6 labs
    logits_lab = np.random.randn(N, 6)
    # Dummy sibling logits for each lab's candidate models
    logits_siblings = {lab: np.random.randn(N, count) for lab, count in LAB_COUNTS.items()}
    
    P = head.fuse_probabilities(logits_lab, logits_siblings)
    assert P.shape == (N, 18), f"Expected shape ({N}, 18), got {P.shape}"
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5), "Rows do not sum to 1.0"
    assert (P >= 0.0).all(), "Found negative probabilities"
