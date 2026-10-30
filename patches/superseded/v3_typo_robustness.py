"""VULNERABILITY 3 FIX: Robust typographic priors with Dirichlet concentration.
Replaces hard Bernoulli rates with Dirichlet concentration priors so single-symbol
obs don't dominate posterior on OOD prompts (code blocks, fiction dialogue).
"""
import numpy as np

def compute_typographic_priors_dirichlet(texts, dir_alpha=2.0):
    """
    Dirichlet-smoothed typographic priors. Each class gets concentration alpha on each
    margin. A symbol observation contributes alpha*log(p) with Dirichlet correction;
    non-observation contributes (alpha-1)*log(1-p).
    Returns (N, K) probability matrix via softmax of log-likelihoods.
    """
    from src.config import VALID_18_MODELS
    from src.typo_profiles import TYPO_PROFILES
    log_priors = np.zeros((N, K))

    for i, t in enumerate(texts):
        obs = [("—" in t), (" — " in t), ('"' in t or '"' in t), ("'" in t or "'" in t)]

        for k, m in enumerate(VALID_18_MODELS):
            # Smooth empirical rates
            p = np.clip(np.array(TYPO_PROFILES[m]), 0.01, 0.99)
            obs_arr = np.array(obs, dtype=float)

            # Dirichlet-smoothed log-likelihood:
            # Each observed symbol: dir_alpha * log(p)
            # Each unobserved symbol: (dir_alpha - 1) * log(1 - p)
            # dir_alpha > 1 prevents overconfident zeroing
            ll = np.sum(
                dir_alpha * obs_arr * np.log(p)
                + (dir_alpha - 1) * (1 - obs_arr) * np.log(1 - p)
            )
            log_priors[i, k] = ll

    def _softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        return np.exp(x - x_max) / np.sum(np.exp(x - x_max), axis=axis, keepdims=True)

    p_typo = _softmax(log_priors, axis=1)
    return p_typo