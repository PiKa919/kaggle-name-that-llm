"""
Pairwise Sibling Discriminator Module on ANOVA Quotient Space.
Trains dedicated, calibrated linear discriminant classifiers on the 10 most
confused sibling pairs across Anthropic, Google, OpenAI, and ZAI to resolve
intra-family attribution ambiguities.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.config import VALID_18_MODELS, MODEL_TO_IDX

# Priority confused sibling pairs from empirical confusion matrix analysis
SIBLING_PAIRS: List[Tuple[str, str]] = [
    # Anthropic (Opus vs Sonnet vs Fable)
    ("claude-opus-4-6", "claude-opus-5"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("claude-fable-5", "claude-opus-4-8"),
    # Google (Gemini Flash trio + Pro)
    ("gemini-3.6-flash", "gemini-3.7-flash"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gemini-3.1-pro-preview", "gemini-3.7-flash"),
    # OpenAI (Reasoning vs Grounded variants)
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("gpt-5.5", "gpt-6-astra"),
    # ZAI (Full vs Flash)
    ("glm-5.3", "glm-5.3-flash"),
    # Cross-lab straight quote / em-dash collision (GLM vs Claude)
    ("glm-5.3", "claude-opus-4-6"),
]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PairwiseSiblingEnsemble:
    """
    Fits independent binary LDA discriminators on the ANOVA quotient space
    for each designated sibling pair.
    """
    def __init__(self, pairs: List[Tuple[str, str]] = SIBLING_PAIRS):
        self.pairs = pairs
        self.classifiers: Dict[Tuple[str, str], LinearDiscriminantAnalysis] = {}

    def fit(self, Z: np.ndarray, y: np.ndarray):
        """
        Args:
            Z: (N, D) representation matrix (e.g. 17-D ANOVA quotient space).
            y: (N,) integer class indices (0..17).
        """
        for m1, m2 in self.pairs:
            idx1 = MODEL_TO_IDX[m1]
            idx2 = MODEL_TO_IDX[m2]

            mask = (y == idx1) | (y == idx2)
            if mask.sum() < 4 or len(np.unique(y[mask])) < 2:
                # Insufficient samples for pair, skip
                continue

            Z_pair = Z[mask]
            y_pair = (y[mask] == idx2).astype(int)  # 0 for m1, 1 for m2

            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(Z_pair, y_pair)
            self.classifiers[(m1, m2)] = clf

    def predict_log_odds_corrections(self, Z: np.ndarray, clip_range: float = 3.0) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Computes log-odds corrections for each fitted pair:
            log_odds = log(P(m2) / P(m1))
        Positive indicates m2 favored, negative indicates m1 favored.
        """
        corrections = {}
        N = Z.shape[0]

        for (m1, m2), clf in self.classifiers.items():
            probs = clf.predict_proba(Z)[:, 1]  # P(m2)
            probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
            log_odds = np.log(probs / (1.0 - probs))
            corrections[(m1, m2)] = np.clip(log_odds, -clip_range, clip_range)

        return corrections


def project_hodgerank(
    corrections: Dict[Tuple[str, str], np.ndarray],
    n_classes: int = len(VALID_18_MODELS)
) -> np.ndarray:
    """
    HodgeRank Least-Squares Potential Projection:
    Projects independent pairwise log-odds differences onto a cycle-free, globally
    consistent potential vector V (shape (N, K) or (K,)) via Moore-Penrose pseudo-inverse
    of the graph Laplacian L^dagger:
        V = B @ L^dagger
    Guarantees:
    1. Zero Condorcet cycles: V[:, m2] - V[:, m1] is globally transitive.
    2. Conservative probability scaling: sum_k V[:, k] = 0.
    3. Handles multi-pair overlaps gracefully without artificial multi-counting.
    """
    if not corrections:
        return np.zeros((1, n_classes), dtype=np.float64)

    sample_key = next(iter(corrections))
    delta_sample = np.asarray(corrections[sample_key])
    is_scalar = (delta_sample.ndim == 0)
    N = 1 if is_scalar else delta_sample.shape[0]

    A = np.zeros((n_classes, n_classes), dtype=np.float64)
    for m1, m2 in corrections.keys():
        i1, i2 = MODEL_TO_IDX[m1], MODEL_TO_IDX[m2]
        A[i1, i2] = 1.0
        A[i2, i1] = 1.0

    D = np.diag(A.sum(axis=1))
    L = D - A
    L_pinv = np.linalg.pinv(L)

    B = np.zeros((N, n_classes), dtype=np.float64)
    for (m1, m2), delta in corrections.items():
        i1, i2 = MODEL_TO_IDX[m1], MODEL_TO_IDX[m2]
        d = np.asarray(delta, dtype=np.float64)
        if is_scalar:
            B[0, i2] += d
            B[0, i1] -= d
        else:
            B[:, i2] += d
            B[:, i1] -= d

    V = B @ L_pinv
    V -= np.mean(V, axis=1, keepdims=True)

    if is_scalar:
        return V[0]
    return V


def fuse_with_pairwise_corrections(
    P_base: np.ndarray,
    corrections: Dict[Tuple[str, str], np.ndarray],
    weight: float = 0.15,
    use_hodgerank: bool = True
) -> np.ndarray:
    """
    Applies pairwise sibling log-odds adjustments to the base probability matrix.
    When use_hodgerank=True (default), projects pairwise deltas into a cycle-free
    consistent potential vector using HodgeRank before adding to log-probabilities.
    """
    log_P = np.log(np.maximum(P_base, 1e-12))

    if use_hodgerank:
        potential = project_hodgerank(corrections, n_classes=P_base.shape[1])
        if potential.ndim == 1:
            log_P += weight * potential[None, :]
        else:
            log_P += weight * potential
    else:
        for (m1, m2), delta in corrections.items():
            idx1 = MODEL_TO_IDX[m1]
            idx2 = MODEL_TO_IDX[m2]
            log_P[:, idx2] += weight * delta
            log_P[:, idx1] -= weight * delta

    return softmax(log_P, axis=1)
