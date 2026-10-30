"""
Typographic evidence as a calibrated, robust likelihood -- not a hardcoded table.

Why this file exists
--------------------
`compute_typographic_priors()` in `notebooks/master_sota/submission_sota.py`
hardcodes four Bernoulli rates per model and multiplies them into a softmax.
Three separate things are wrong with that, and all three are measurable on
`data/train.csv`:

1. The constants do not match the data.  The curly-double-quote rate for the
   whole GPT-5.6 family is hardcoded 0.95; the file's value is 0.107.  For
   qwen3.8-max it is hardcoded 0.98; the file's value is 0.007.  Averaged over
   each model's own passages, the hardcoded profile is 4.29 nats worse for
   qwen3.8-max than the empirical profile and 0.00 nats worse for all five
   Claude models.  The "prior" is a Claude-favouring bias, not a prior.

2. The four indicators are not independent.  P(spaced_em | em) = 0.082 while
   P(spaced_em) = 0.032, so the joint is 2.6x the product of marginals.  The
   factorised likelihood charges the same em-dash twice.

3. A Bernoulli floored at p = 0.01 charges log(0.01) = -4.6 nats per marker.
   One em-dash in an otherwise Gemini passage is enough to delete Gemini; one
   quote in a code block is enough to delete Claude.

This module replaces the table with:
  * an empirical 16-state categorical distribution per model (the joint
    distribution of the four indicators), fitted on the training fold only,
    Dirichlet-smoothed.  No independence assumption, no double counting.
  * a contamination mixture  P(bit | model) = (1 - rho) * Cat_model + rho * U
    where U is the pooled marginal.  rho is the probability that the prompt
    forces a marker regardless of author (a code block makes every model emit
    straight quotes; a dialogue prompt makes every model emit em-dashes).
  * winsorisation of the per-passage log-likelihood at `max_nats`, so the
    worst case a single out-of-distribution passage can suffer is bounded and
    known, instead of depending on which class happened to have a zero count.

Temperatures and fusion weights are deliberately NOT here; under a log-opinion
pool only the product of (1/T) and the pool weight is identifiable, so both are
fitted together in `src/fusion.py`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.config import IDX_TO_MODEL, MODEL_TO_IDX, VALID_18_MODELS

__all__ = [
    "typo_bits",
    "bit_state",
    "empirical_categorical",
    "empirical_bernoulli",
    "TypoPriorView",
    "calibration_report",
]

EM_DASH = "\u2014"
CURLY_DQ = ("\u201c", "\u201d")
CURLY_SQ = ("\u2018", "\u2019")

N_BITS = 4
N_STATES = 2 ** N_BITS


def typo_bits(texts: Sequence[str]) -> np.ndarray:
    """(N, 4) uint8 indicator matrix.

    Columns: em-dash, spaced em-dash, curly double quote, curly single quote.
    `spaced_em` implies `em` by construction (P(spaced|em) = 0.082 in train);
    the redundancy is harmless here because the joint state is modelled
    directly rather than factorised.
    """
    out = np.zeros((len(texts), N_BITS), dtype=np.uint8)
    for i, t in enumerate(texts):
        out[i, 0] = int(EM_DASH in t)
        out[i, 1] = int(" %s " % EM_DASH in t)
        out[i, 2] = int(any(c in t for c in CURLY_DQ))
        out[i, 3] = int(any(c in t for c in CURLY_SQ))
    return out


def bit_state(bits: np.ndarray) -> np.ndarray:
    """Map an (N, 4) indicator matrix to (N,) states in 0..15."""
    bits = np.asarray(bits, dtype=np.int64)
    weights = (1 << np.arange(bits.shape[1]))[::-1]
    return bits @ weights


def empirical_bernoulli(bits: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """(K, 4) per-class marginal rates. Diagnostic only -- do not factorise."""
    prof = np.full((n_classes, N_BITS), 0.5)
    for k in range(n_classes):
        m = y == k
        if m.any():
            prof[k] = bits[m].mean(axis=0)
    return prof


def empirical_categorical(
    bits: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """(K, 16) Dirichlet-smoothed joint distribution of the four indicators.

    Posterior mean of a Dirichlet(alpha) prior: (n_kc + alpha) / (n_k + 16 alpha).
    """
    states = bit_state(bits)
    counts = np.zeros((n_classes, N_STATES))
    for k in range(n_classes):
        m = y == k
        if m.any():
            counts[k] = np.bincount(states[m], minlength=N_STATES)
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + N_STATES * alpha)


class TypoPriorView:
    """Calibrated, contamination-robust typographic likelihood.

    Parameters
    ----------
    alpha : Dirichlet concentration for the 16-state table (1.0 = uniform prior
        with one pseudo-count per state).
    rho : contamination weight in [0, 1). Probability that the prompt compels
        the marker regardless of the author. 0 = trust the table completely
        (fragile), 0.25 = a quarter of the evidence budget is spendable by topic.
    max_nats : relative ceiling on the evidence a passage may contribute.  No
        class can be pushed more than `max_nats` below the most favoured class,
        because the floor is applied to `ll_k - max_k ll`.  The current code's
        effective floor is log(0.01) * 2 = -9.2 nats; `max_nats = 2.0` caps the
        likelihood ratio at e^-2 = 0.14 per passage.
    """

    def __init__(self, alpha: float = 1.0, rho: float = 0.25, max_nats: float = 2.0):
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.max_nats = float(max_nats)
        self.table_: np.ndarray | None = None
        self.marginal_: np.ndarray | None = None
        self.n_classes_ = len(VALID_18_MODELS)

    # -- fitting --------------------------------------------------------- #
    def fit(self, bits: np.ndarray, y: np.ndarray) -> "TypoPriorView":
        states = bit_state(bits)
        self.marginal_ = np.bincount(states, minlength=N_STATES) / len(states)
        cat = empirical_categorical(bits, y, self.n_classes_, self.alpha)
        self.table_ = (1.0 - self.rho) * cat + self.rho * self.marginal_[None, :]
        return self

    # -- scoring --------------------------------------------------------- #
    def log_prob(self, bits: np.ndarray) -> np.ndarray:
        """(N, K) log-likelihoods with a *relative* evidence ceiling.

        The floor is applied to each class's distance from the best class,

            ll'_k = max(ll_k - max_k ll, -max_nats)

        rather than to the absolute value.  Absolute winsorisation (ll_k >= -c)
        collapses every bad class onto the same number and destroys the ranking,
        which a smoke test showed costs more than it saves: the fitted view
        scored 0.099 balanced accuracy standalone versus 0.141 for the hardcoded
        table it was meant to improve.  The relative form bounds the damage a
        single out-of-distribution marker can do -- no class can be pushed more
        than `max_nats` below the most favoured one, i.e. a likelihood ratio of
        e^-max_nats -- while preserving the ordering inside that range.
        """
        if self.table_ is None:
            raise RuntimeError("TypoPriorView.fit() must be called before log_prob()")
        states = bit_state(bits)
        ll = np.log(self.table_[:, states]).T  # (N, K)
        ll = ll - ll.max(axis=1, keepdims=True)
        ll = np.maximum(ll, -self.max_nats)
        ll = ll - ll.mean(axis=1, keepdims=True)  # gauge fix: evidence, not a bias
        return ll

    def fit_predict_oof(
        self,
        bits: np.ndarray,
        y: np.ndarray,
        folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Out-of-fold log-likelihoods. The table must never see its own row."""
        out = np.zeros((len(y), self.n_classes_))
        for tr, va in folds:
            sub = TypoPriorView(self.alpha, self.rho, self.max_nats).fit(bits[tr], y[tr])
            out[va] = sub.log_prob(bits[va])
        return out


def calibration_report(
    bits: np.ndarray,
    y: np.ndarray,
    hardcoded_profiles: Dict[str, Sequence[float]],
) -> "object":
    """Side-by-side nats lost by the hardcoded table against the empirical one.

    Positive `nats_penalty_vs_truth` means the hardcoded profile scores the true
    author WORSE than the empirical profile does -- i.e. it is a systematic bias
    against that class, at every fusion weight above zero.
    """
    import pandas as pd

    hard = np.array([hardcoded_profiles[m] for m in VALID_18_MODELS], dtype=float)
    emp = empirical_bernoulli(bits, y, len(VALID_18_MODELS))
    p_hard = np.clip(hard, 0.01, 0.99)
    p_emp = np.clip(emp, 0.01, 0.99)

    rows: List[Dict[str, float]] = []
    for k, name in enumerate(VALID_18_MODELS):
        m = y == k
        if not m.any():
            continue
        ll_hard = (bits[m] @ np.log(p_hard[k]) + (1 - bits[m]) @ np.log(1 - p_hard[k])).mean()
        ll_emp = (bits[m] @ np.log(p_emp[k]) + (1 - bits[m]) @ np.log(1 - p_emp[k])).mean()
        rows.append(
            {
                "model": name,
                "emp_curly_dq": emp[k, 2],
                "hard_curly_dq": hard[k, 2],
                "emp_curly_sq": emp[k, 3],
                "hard_curly_sq": hard[k, 3],
                "nats_penalty_vs_truth": float(ll_emp - ll_hard),
            }
        )
    return pd.DataFrame(rows).sort_values("nats_penalty_vs_truth", ascending=False)
