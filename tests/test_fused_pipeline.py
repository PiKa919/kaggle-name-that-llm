"""
Unit tests for ANOVA centering and Sinkhorn-Knopp marginal balancing.
"""

import numpy as np
import pytest


def test_sinkhorn_knopp_convergence():
    from experiments.exp08_fused_quotient_stylometry import sinkhorn_knopp_balance
    K = 18
    N = 1000
    np.random.seed(42)
    P_raw = np.random.uniform(0.01, 1.0, (N, K))
    P_raw /= P_raw.sum(axis=1, keepdims=True)
    P_balanced = sinkhorn_knopp_balance(P_raw, iters=100)
    
    # Assert row sums equal 1
    assert np.allclose(P_balanced.sum(axis=1), 1.0, atol=1e-5), "Rows do not sum to 1"
    # Assert column sums are balanced (N / K)
    expected_col_sum = N / K
    assert np.allclose(P_balanced.sum(axis=0), expected_col_sum, atol=1e-2), "Columns are not balanced"


def test_anova_center_annihilation():
    from experiments.exp08_fused_quotient_stylometry import anova_center
    N = 180
    D = 10
    K = 18
    np.random.seed(42)
    prompt_ids = np.repeat(np.arange(10), K)
    
    # Construct synthetic data with huge prompt effect
    Z = np.random.randn(N, D)
    prompt_effect = np.repeat(np.random.randn(10, D) * 10.0, K, axis=0)
    Z_confounded = Z + prompt_effect
    
    Z_centered = anova_center(Z_confounded, prompt_ids)
    
    # Compute variance of prompt means after centering
    unique_prompts, prompt_inv = np.unique(prompt_ids, return_inverse=True)
    p_means = np.zeros((len(unique_prompts), D))
    np.add.at(p_means, prompt_inv, Z_centered)
    p_means /= np.bincount(prompt_inv)[:, None]
    
    assert np.allclose(p_means - Z_centered.mean(0), 0.0, atol=1e-5), "Prompt effect was not annihilated"


def test_loo_cavity_center():
    from src.cavity import loo_cavity_center
    N = 90
    D = 10
    np.random.seed(42)
    prompt_ids = np.repeat(np.arange(5), 18)
    Z = np.random.randn(N, D)
    prompt_effect = np.repeat(np.random.randn(5, D) * 15.0, 18, axis=0)
    Z_confounded = Z + prompt_effect
    
    Z_cavity = loo_cavity_center(Z_confounded, prompt_ids)
    
    # Assert self-subtraction bias is eliminated
    assert Z_cavity.shape == (N, D)
    # Check that prompt effect is removed
    unique_prompts, prompt_inv = np.unique(prompt_ids, return_inverse=True)
    p_means = np.zeros((len(unique_prompts), D))
    np.add.at(p_means, prompt_inv, Z_cavity)
    p_means /= np.bincount(prompt_inv)[:, None]
    assert np.allclose(p_means - Z_cavity.mean(0), 0.0, atol=1e-5)
