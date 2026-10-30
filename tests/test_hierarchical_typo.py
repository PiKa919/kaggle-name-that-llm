import numpy as np
import pytest
from src.config import VALID_18_MODELS, MODEL_TO_IDX
from src.hierarchical_classifier import (
    HierarchicalAttributionHead,
    compute_typographic_priors
)


def test_typographic_prior_bounds():
    texts = [
        "This is a sample text with straight quotes 'hello' and unspaced em-dash—like so.",
        "Here is text with curly quotes “hello” and no em dash at all.",
        "And here is Claude style text with spaced em — dash and straight quotes."
    ]
    priors = compute_typographic_priors(texts)
    assert priors.shape == (3, 18)
    # Probabilities must be strictly positive and sum to 1.0
    assert (priors > 0.0).all()
    assert np.allclose(priors.sum(axis=1), 1.0, atol=1e-5)


def test_hierarchical_fusion_with_typo_priors():
    head = HierarchicalAttributionHead()
    N = 2
    
    # 2 dummy samples
    logits_lab = np.zeros((N, 6)) # Equal lab logits
    
    # Lab counts: anthropic 5, google 4, openai 5, qwen 1, zai 2, moonshot 1
    logits_siblings = {
        "anthropic": np.zeros((N, 5)),
        "google": np.zeros((N, 4)),
        "openai": np.zeros((N, 5)),
        "qwen": np.zeros((N, 1)),
        "zai": np.zeros((N, 2)),
        "moonshot": np.zeros((N, 1)),
    }
    
    # Text 0: spaced em-dash (favors opus-4-6 and opus-5 within Anthropic)
    # Text 1: curly double quotes (favors gemini-3.8-flash within Google)
    texts = [
        "Passage one has spaced em — dash and 'straight' quotes.",
        "Passage two has “curly” double quotes and zero dashes."
    ]
    typo_priors = compute_typographic_priors(texts)
    
    P_fused = head.fuse_with_typo(logits_lab, logits_siblings, typo_priors, typo_weight=0.25)
    assert P_fused.shape == (N, 18)
    assert np.allclose(P_fused.sum(axis=1), 1.0, atol=1e-5)
    
    # Verify directional shift
    idx_opus46 = MODEL_TO_IDX["claude-opus-4-6"]
    idx_sonnet = MODEL_TO_IDX["claude-sonnet-5"]
    assert P_fused[0, idx_opus46] > P_fused[0, idx_sonnet], "Expected spaced em-dash to favor opus-4-6 over sonnet-5"
    
    idx_gem38 = MODEL_TO_IDX["gemini-3.8-flash"]
    idx_gem36 = MODEL_TO_IDX["gemini-3.6-flash"]
    assert P_fused[1, idx_gem38] > P_fused[1, idx_gem36], "Expected curly double quotes to favor gemini-3.8-flash over gemini-3.6-flash"
