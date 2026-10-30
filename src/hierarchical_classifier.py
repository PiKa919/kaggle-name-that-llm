"""
Hierarchical Attribution Classifier:
Decomposes 18-way classification into coarse lab attribution (6 labs)
and fine-grained conditional sibling discrimination within each lab:
    P(model_k) = P(lab(model_k)) * P(model_k | lab(model_k))

Now extended with soft Typographic Prior Regularization (P_typo)
to collapse intra-family sibling confusion without hard threshold truncation.
"""

from typing import Dict, List, Optional
import numpy as np
from src.config import VALID_18_MODELS, MODEL_TO_IDX, MODEL_TO_LAB

LABS = ["anthropic", "google", "openai", "qwen", "zai", "moonshot"]
LAB_TO_IDX = {l: i for i, l in enumerate(LABS)}

# Group models by lab in deterministic alphabetical order
LAB_TO_MODELS: Dict[str, List[str]] = {
    lab: [m for m in VALID_18_MODELS if MODEL_TO_LAB[m] == lab]
    for lab in LABS
}

# Empirical Typographic Affinity Priors across 18 Models (smoothed probabilities)
# [p_has_em_dash, p_spaced_em_dash, p_curly_dq, p_curly_sq]
TYPO_PROFILES = {
    "claude-fable-5":        [0.83, 0.01, 0.01, 0.01],
    "claude-opus-4-6":       [0.75, 0.45, 0.01, 0.01],
    "claude-opus-4-8":       [0.63, 0.01, 0.01, 0.01],
    "claude-opus-5":         [0.81, 0.40, 0.01, 0.01],
    "claude-sonnet-5":       [0.73, 0.01, 0.01, 0.01],
    "gemini-3.1-pro-preview":[0.01, 0.01, 0.01, 0.35],
    "gemini-3.6-flash":      [0.01, 0.01, 0.01, 0.48],
    "gemini-3.7-flash":      [0.01, 0.01, 0.01, 0.68],
    "gemini-3.8-flash":      [0.01, 0.01, 0.75, 0.67],
    "glm-5.3":               [0.82, 0.05, 0.01, 0.01],
    "glm-5.3-flash":         [0.79, 0.04, 0.01, 0.01],
    "gpt-5.5":               [0.13, 0.01, 0.73, 0.82],
    "gpt-5.6-luna":          [0.37, 0.01, 0.95, 0.97],
    "gpt-5.6-sol":           [0.27, 0.01, 0.95, 0.98],
    "gpt-5.6-terra":         [0.23, 0.01, 0.95, 0.93],
    "gpt-6-astra":           [0.09, 0.01, 0.95, 0.80],
    "kimi-k3":               [0.52, 0.01, 0.85, 0.56],
    "qwen3.8-max":           [0.03, 0.01, 0.98, 0.74],
}


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_typographic_priors(
    texts: List[str],
    min_prob: float = 0.08,
    max_prob: float = 0.92,
    temperature: float = 0.50
) -> np.ndarray:
    """
    Computes a robust probabilistic prior matrix (N, 18) based on typographic
    and watermarking indicators (spaced em dashes, curly double quotes, curly single quotes, em dashes).
    Uses Dirichlet-Laplace bounded likelihood to prevent single OOD symbol observations
    from overwhelming deep representation likelihoods.
    """
    N = len(texts)
    K = len(VALID_18_MODELS)
    log_priors = np.zeros((N, K), dtype=np.float64)

    for i, t in enumerate(texts):
        has_em = float("—" in t)
        has_spaced_em = float(" — " in t)
        has_curly_dq = float("“" in t or "”" in t)
        has_curly_sq = float("‘" in t or "’" in t)

        obs = np.array([has_em, has_spaced_em, has_curly_dq, has_curly_sq])

        for k, m_name in enumerate(VALID_18_MODELS):
            probs = np.array(TYPO_PROFILES[m_name])
            # Bounded probability smoothing
            p_smooth = np.clip(probs, min_prob, max_prob)
            ll = obs * np.log(p_smooth) + (1.0 - obs) * np.log(1.0 - p_smooth)
            log_priors[i, k] = temperature * ll.sum()

    # Softmax over 18 classes to normalize each row into a probability distribution
    p_typo = softmax(log_priors, axis=1)
    return p_typo


def smooth_length_gate(
    word_counts: np.ndarray,
    threshold: float = 80.0,
    w_max: float = 0.45,
    w_min: float = 0.15,
    k: float = 0.10
) -> np.ndarray:
    """
    Generalized logistic (Richards) smooth transition:
        w(L) = w_min + (w_max - w_min) / (1 + exp(k * (L - L_0)))
    Provides a C^infinity smooth transition between short and long passages.
    Monotonically decreasing with word count; equals (w_max + w_min)/2 at threshold.
    """
    z = np.clip(k * (word_counts.astype(np.float64) - threshold), -25.0, 25.0)
    sig = 1.0 / (1.0 + np.exp(z))
    return w_min + (w_max - w_min) * sig


class HierarchicalAttributionHead:
    """
    Combines lab-level logits and within-lab sibling logits via chain rule:
        P(m) = P(lab) * P(m | lab)
    With optional typographic prior soft regularization.
    """
    def __init__(self):
        self.labs = LABS
        self.lab_to_models = LAB_TO_MODELS

    def fuse_probabilities(self, logits_lab: np.ndarray, logits_siblings: Dict[str, np.ndarray]) -> np.ndarray:
        N = logits_lab.shape[0]
        p_lab = softmax(logits_lab, axis=1)  # (N, 6)

        P_final = np.zeros((N, len(VALID_18_MODELS)), dtype=np.float64)

        for lab_idx, lab in enumerate(self.labs):
            models_in_lab = self.lab_to_models[lab]
            p_lab_val = p_lab[:, lab_idx]  # (N,)

            if len(models_in_lab) == 1:
                m_name = models_in_lab[0]
                m_idx = MODEL_TO_IDX[m_name]
                P_final[:, m_idx] = p_lab_val
            else:
                sub_logits = logits_siblings[lab]  # (N, len(models_in_lab))
                p_cond = softmax(sub_logits, axis=1)
                for sub_i, m_name in enumerate(models_in_lab):
                    m_idx = MODEL_TO_IDX[m_name]
                    P_final[:, m_idx] = p_lab_val * p_cond[:, sub_i]

        P_final /= np.maximum(P_final.sum(axis=1, keepdims=True), 1e-12)
        return P_final

    def fuse_with_typo(
        self,
        logits_lab: np.ndarray,
        logits_siblings: Dict[str, np.ndarray],
        typo_priors: np.ndarray,
        typo_weight: float = 0.25
    ) -> np.ndarray:
        """
        Fuses hierarchical model probabilities with typographic priors in log-probability space:
            log P_fused = (1 - w) * log P_hierarchical + w * log P_typo
        """
        P_base = self.fuse_probabilities(logits_lab, logits_siblings)
        log_base = np.log(np.maximum(P_base, 1e-12))
        log_typo = np.log(np.maximum(typo_priors, 1e-12))

        log_fused = (1.0 - typo_weight) * log_base + typo_weight * log_typo
        P_fused = softmax(log_fused, axis=1)
        return P_fused

    def fuse_with_length_gate(
        self,
        logits_lab: np.ndarray,
        logits_siblings: Dict[str, np.ndarray],
        typo_priors: np.ndarray,
        word_counts: np.ndarray,
        threshold_words: float = 80.0,
        short_weight: float = 0.45,
        standard_weight: float = 0.15,
        smoothness_k: float = 0.10
    ) -> np.ndarray:
        """
        Dynamically and smoothly weights typographic priors based on passage length:
        Short passages (< threshold_words) smoothly ramp up typographic weight (up to short_weight),
        since semantic embeddings lose discrimination on brief snippets while
        typographic artifacts remain invariant. Longer passages transition smoothly to standard_weight.
        """
        P_base = self.fuse_probabilities(logits_lab, logits_siblings)
        log_base = np.log(np.maximum(P_base, 1e-12))
        log_typo = np.log(np.maximum(typo_priors, 1e-12))

        # Compute smooth C^infinity weights: (N, 1)
        w = smooth_length_gate(
            word_counts,
            threshold=threshold_words,
            w_max=short_weight,
            w_min=standard_weight,
            k=smoothness_k
        )[:, None]

        log_fused = (1.0 - w) * log_base + w * log_typo
        P_fused = softmax(log_fused, axis=1)
        return P_fused

