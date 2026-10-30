import numpy as np
import pytest
from src.config import VALID_18_MODELS, MODEL_TO_IDX
from src.sibling_discriminators import PairwiseSiblingEnsemble, SIBLING_PAIRS, fuse_with_pairwise_corrections


def test_pairwise_sibling_ensemble():
    N = 100
    D = 17
    # Synthetic quotient embeddings
    Z = np.random.randn(N, D)
    y = np.random.choice(len(VALID_18_MODELS), size=N)
    
    pairs = [
        ("gemini-3.7-flash", "gemini-3.8-flash"),
        ("claude-opus-4-6", "claude-opus-5")
    ]
    ensemble = PairwiseSiblingEnsemble(pairs=pairs)
    ensemble.fit(Z, y)
    
    corrections = ensemble.predict_log_odds_corrections(Z)
    assert len(corrections) == len(pairs)
    for pair in pairs:
        assert pair in corrections
        assert corrections[pair].shape == (N,)
        assert not np.isnan(corrections[pair]).any()


def test_fuse_with_pairwise_corrections():
    N = 4
    K = 18
    P_base = np.full((N, K), 1.0 / K)
    
    # Correction favoring gemini-3.8-flash over gemini-3.7-flash
    corrections = {
        ("gemini-3.7-flash", "gemini-3.8-flash"): np.array([2.0, 2.0, -2.0, -2.0])
    }
    
    P_corrected = fuse_with_pairwise_corrections(P_base, corrections, weight=0.15)
    assert P_corrected.shape == (N, K)
    assert np.allclose(P_corrected.sum(axis=1), 1.0, atol=1e-5)
    
    idx_38 = MODEL_TO_IDX["gemini-3.8-flash"]
    idx_37 = MODEL_TO_IDX["gemini-3.7-flash"]
    
    # For samples 0 and 1 (positive correction), 3.8 should be favored over 3.7
    assert P_corrected[0, idx_38] > P_corrected[0, idx_37]
    assert P_corrected[1, idx_38] > P_corrected[1, idx_37]
    
    # For samples 2 and 3 (negative correction), 3.7 should be favored over 3.8
    assert P_corrected[2, idx_37] > P_corrected[2, idx_38]
    assert P_corrected[3, idx_37] > P_corrected[3, idx_38]
