"""
Metric-aligned post-processing for 'Name That LLM'.

Three operators, in the order they must be applied:

  1. SLD-EM prior correction  -- estimate the hidden test class prior pi and
     invert the train/test mixture shift. Balanced accuracy is an unweighted
     mean of 18 recalls, so a class that is rare in test but common in train
     will be over-predicted and its recall destroyed. This is the graded
     version of "adaptive centering": centering in logit space with a learned
     scalar per class is exactly prior correction.

  2. Balanced-accuracy class multipliers -- 18 free parameters fitted by
     coordinate ascent directly on the competition metric, on OOF only.

  3. Within-prompt structural constraint -- Hungarian/Sinkhorn with dummy rows.
     Each prompt is answered at most once per model, so 18 rows of one cluster
     are a PARTIAL MATCHING, not 18 independent decisions. Faked as a soft,
     temperature-controlled constraint so that a wrong cluster cannot force a
     catastrophic permutation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score

__all__ = [
    "sld_em",
    "fit_class_multipliers",
    "apply_multipliers",
    "sinkhorn_matching_posterior",
    "hungarian_matching",
    "PostProcessor",
]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. Prior shift: Saerens-Latinne-Decaestecker EM
# --------------------------------------------------------------------------- #
def sld_em(
    probs: np.ndarray,
    source_priors: np.ndarray,
    n_iter: int = 200,
    tol: float = 1e-7,
    smooth: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate target priors and reweight posteriors.

    probs          : (n, K) posteriors from a model trained under source_priors
    source_priors  : (K,) empirical class prior of the training set
    smooth         : Dirichlet-style pull toward uniform, in [0, 1]. Use ~0.1
                     when clusters are small; EM on 18 rows is high variance.

    Returns (adjusted_probs, estimated_target_prior).

    Correctness property: for a CALIBRATED source model whose mean posterior
    over the data equals source_priors, the EM is a fixed point at step 0 and
    adjusted_probs == probs. Assert this in tests on simulated calibrated data;
    on the real competition model the mean posterior will differ from 1/18 and
    that difference IS the shift being corrected.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
    src = np.clip(np.asarray(source_priors, dtype=np.float64), EPS, 1.0)
    src = src / src.sum()
    K = p.shape[1]

    pi = np.full(K, 1.0 / K)
    for _ in range(n_iter):
        w = p * (pi / src)[None, :]
        w /= w.sum(axis=1, keepdims=True)
        new_pi = w.mean(axis=0)
        if smooth > 0:
            new_pi = (1.0 - smooth) * new_pi + smooth / K
        new_pi /= new_pi.sum()
        if np.abs(new_pi - pi).max() < tol:
            pi = new_pi
            break
        pi = new_pi

    adj = p * (pi / src)[None, :]
    adj /= adj.sum(axis=1, keepdims=True)
    return adj, pi


# --------------------------------------------------------------------------- #
# 2. Direct optimisation of balanced accuracy
# --------------------------------------------------------------------------- #
def _encode(y_true: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Map string labels to probability-column indices. Fails loudly on unknown."""
    lut = {lab: i for i, lab in enumerate(labels)}
    try:
        return np.array([lut[v] for v in y_true], dtype=int)
    except KeyError as exc:  # pragma: no cover - data integrity guard
        raise KeyError(f"label {exc.args[0]!r} is not in `labels`") from exc


def apply_multipliers(probs: np.ndarray, mult: np.ndarray) -> np.ndarray:
    s = np.clip(probs, EPS, 1.0) * mult[None, :]
    return s / s.sum(axis=1, keepdims=True)


def fit_class_multipliers(
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    n_passes: int = 3,
    grid: Optional[np.ndarray] = None,
    anchor: int = 0,
) -> Tuple[np.ndarray, float]:
    """Coordinate ascent on the 18-dim multiplier vector, maximising BAcc.

    BAcc is scale invariant, so `anchor` is pinned to 1.0 and only K-1 effective
    parameters are fitted. Fitted on OOF only. Never fit on the fold you report.

    `labels` must give the model-name for each probability column; without it,
    y_true must already be integer column indices.
    """
    y_true = np.asarray(y_true)
    if labels is not None:
        y_int = _encode(y_true, labels)
    else:
        y_int = y_true.astype(int)
    logp = np.log(np.clip(probs, EPS, 1.0))
    K = logp.shape[1]
    if grid is None:
        grid = np.exp(np.linspace(-2.5, 2.5, 21))

    mult = np.ones(K)
    best = balanced_accuracy_score(y_int, np.argmax(logp, axis=1))

    for _ in range(n_passes):
        improved = False
        for k in range(K):
            if k == anchor:
                continue
            cur = mult[k]
            for g in grid:
                if g == cur:
                    continue
                mult[k] = g
                score = balanced_accuracy_score(
                    y_int, np.argmax(logp + np.log(mult)[None, :], axis=1)
                )
                if score > best + 1e-12:
                    best, cur, improved = score, g, True
            mult[k] = cur
        if not improved:
            break
    return mult, float(best)


# --------------------------------------------------------------------------- #
# 3. Partial-matching constraint inside a prompt cluster
# --------------------------------------------------------------------------- #
def sinkhorn_matching_posterior(
    logp: np.ndarray,
    beta: float = 1.0,
    dummy_logit: float = -1.5,
    n_iter: int = 60,
) -> np.ndarray:
    """Soft at-most-one constraint over a prompt cluster.

    logp : (m, K) log posteriors for the m passages of one prompt cluster.
    Builds a K x K cost matrix: the m real rows plus (K-m) dummy rows carrying
    `dummy_logit` (the log-odds that a model is simply absent from the sample).
    Doubly-stochastic Sinkhorn, then marginalise the dummies away.

    beta -> 0  : no constraint (reduces to the input posteriors)
    beta -> inf: hard permutation, which is exactly what destroyed your LB
                 (0.353 -> 0.344) because test prompts are subsampled.
    """
    logp = np.asarray(logp, dtype=np.float64)
    m, K = logp.shape
    if m > K:
        # More rows than classes cannot be a partial matching; refuse to lie.
        return np.exp(logp - logsumexp(logp, axis=1, keepdims=True))

    M = np.full((K, K), dummy_logit, dtype=np.float64)
    M[:m] = beta * logp
    M[:m] -= logsumexp(M[:m], axis=1, keepdims=True)  # row-normalise real rows

    a = np.full(K, -np.log(K))
    u = np.zeros(K)
    v = np.zeros(K)
    for _ in range(n_iter):
        u = a - logsumexp(M + v[None, :], axis=1)
        v = a - logsumexp(M + u[:, None], axis=0)
    P = np.exp(M + u[:, None] + v[None, :])

    out = P[:m]
    out = out / np.clip(out.sum(axis=1, keepdims=True), EPS, None)
    return out


def hungarian_matching(logp: np.ndarray, dummy_logit: float = -1.5) -> np.ndarray:
    """Hard partial matching. Use only as an ablation, never as the default."""
    logp = np.asarray(logp, dtype=np.float64)
    m, K = logp.shape
    if m > K:
        return np.argmax(logp, axis=1)
    C = np.full((K, K), -dummy_logit, dtype=np.float64)
    C[:m] = -logp
    rows, cols = linear_sum_assignment(C)
    assign = np.full(m, -1, dtype=int)
    for r, c in zip(rows, cols):
        if r < m:
            assign[r] = c
    return assign


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class PostProcessor:
    """Fit on OOF, apply to test. Order is fixed and non-negotiable."""

    def __init__(self, n_classes: int, source_priors: Optional[np.ndarray] = None):
        self.K = n_classes
        self.source_priors = (
            np.full(n_classes, 1.0 / n_classes)
            if source_priors is None
            else np.asarray(source_priors, dtype=np.float64)
        )
        self.mult: Optional[np.ndarray] = None
        self.report: Dict[str, float] = {}

    def fit(
        self,
        y_true: np.ndarray,
        oof_probs: np.ndarray,
        labels: Sequence[str],
        use_prior_shift: bool = True,
        use_multipliers: bool = True,
    ) -> "PostProcessor":
        y_int = _encode(np.asarray(y_true), labels)
        base = float(balanced_accuracy_score(y_int, np.argmax(oof_probs, axis=1)))
        p = oof_probs
        if use_prior_shift:
            p, _ = sld_em(p, self.source_priors, smooth=0.1)
        self.report["bacc_raw"] = base
        self.report["bacc_after_prior"] = float(
            balanced_accuracy_score(y_int, np.argmax(p, axis=1))
        )
        if use_multipliers:
            self.mult, best = fit_class_multipliers(y_int, p)
            self.report["bacc_after_mult"] = best
        else:
            self.mult = np.ones(self.K)
            self.report["bacc_after_mult"] = self.report["bacc_after_prior"]
        return self

    def transform(self, probs: np.ndarray, use_prior_shift: bool = True) -> np.ndarray:
        p = np.asarray(probs, dtype=np.float64)
        if use_prior_shift:
            p, _ = sld_em(p, self.source_priors, smooth=0.1)
        if self.mult is not None:
            p = apply_multipliers(p, self.mult)
        return p
