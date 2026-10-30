"""
Leave-One-Out (LOO) Cavity Centering & Prompt Confound Annihilation.
Implements the scale-invariant LOO residual operator:
    z_tilde_i = (m_g / (m_g - 1)) * (z_i - bar{z}_g) + bar{z}_grand
Eliminating self-subtraction bias across varying cluster sizes (m_g).
"""

import numpy as np


def loo_cavity_center(Z: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    """
    Apply leave-one-out cavity centering to eliminate prompt topic confound
    without self-subtraction scale distortion.
    
    For each sample i in prompt cluster g with size m_g:
        bar{z}_{-i} = (sum_{j in g, j != i} z_j) / (m_g - 1)
        z_tilde_i = z_i - bar{z}_{-i} + bar{z}_grand
                  = (m_g / (m_g - 1)) * (z_i - bar{z}_g) + bar{z}_grand
    """
    unique_prompts, prompt_inv = np.unique(prompt_ids, return_inverse=True)
    G = len(unique_prompts)
    N, D = Z.shape
    
    # Compute cluster sums and counts
    counts = np.bincount(prompt_inv, minlength=G)[:, None].astype(np.float32)
    prompt_sums = np.zeros((G, D), dtype=np.float32)
    np.add.at(prompt_sums, prompt_inv, Z)
    
    prompt_means = prompt_sums / np.maximum(counts, 1.0)
    grand_mean = Z.mean(axis=0, keepdims=True)
    
    # Sample-wise counts
    sample_counts = counts[prompt_inv]  # shape (N, 1)
    
    # Scale factor m_g / (m_g - 1) for m_g > 1
    factors = np.where(sample_counts > 1.0, sample_counts / (sample_counts - 1.0), 1.0)
    
    # Centered residual
    residuals = Z - prompt_means[prompt_inv]
    Z_cavity = factors * residuals + grand_mean
    return Z_cavity
