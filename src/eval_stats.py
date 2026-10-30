"""
Proof instruments for 'Name That LLM'.

Why this file exists
--------------------
Balanced accuracy on 5 grouped folds of 150 prompts has ~30 passages per class
per fold. Per-class recall standard error is therefore ~sqrt(p(1-p)/30). Any
single-fold or single-seed CV delta below ~2 SE is noise being marketed as
progress. Nothing in this repository may be called an improvement without the
intervals produced here.

Units of resampling
-------------------
Passages from the same prompt are statistically dependent (shared topic, shared
prompt). The independent unit is the PROMPT, not the row. Every bootstrap and
every permutation test below resamples prompt CLUSTERS, never rows. Resampling
rows silently shrinks the CI by roughly sqrt(18) and is the single most common
way Kaggle CV gets inflated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

__all__ = [
    "repeated_grouped_folds",
    "balanced_accuracy_ci",
    "paired_bootstrap_delta",
    "per_class_recall_table",
    "length_stratified_balanced_accuracy",
    "summarize",
]


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def repeated_grouped_folds(
    df,
    n_splits: int = 5,
    n_repeats: int = 3,
    base_seed: int = 42,
    label_col: str = "model",
    group_col: str = "prompt_id",
) -> List[np.ndarray]:
    """Repeated StratifiedGroupKFold.

    Returns a list of fold-id vectors, one per repeat. A single seed on 150
    prompts is a coin flip; three repeats is the minimum defensible standard.
    """
    out: List[np.ndarray] = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=base_seed + r
        )
        folds = np.zeros(len(df), dtype=int)
        for fold, (_, val_idx) in enumerate(
            cv.split(df, df[label_col], groups=df[group_col])
        ):
            folds[val_idx] = fold
        out.append(folds)
    return out


# --------------------------------------------------------------------------- #
# Interval estimation over prompt clusters
# --------------------------------------------------------------------------- #
def _cluster_bacc(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def balanced_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Cluster bootstrap CI for balanced accuracy.

    Resamples prompt groups with replacement. Groups that end up absent from a
    bootstrap draw contribute a zero-recall class, which is the honest
    behaviour: the deployed metric is an unweighted mean over 18 recalls.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.where(inv == g)[0] for g in range(len(uniq))]

    point = _cluster_bacc(y_true, y_pred, labels)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([idx_by_group[g] for g in pick])
        stats[b] = _cluster_bacc(y_true[rows], y_pred[rows], labels)

    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(stats.std(ddof=1)),
        "n_groups": int(len(uniq)),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired cluster bootstrap for delta = bacc(a) - bacc(b).

    Reports the two-sided p-value as the fraction of resamples whose delta
    crosses zero. Paired resampling is mandatory: the two systems saw the same
    passages, so unpaired CIs overstate uncertainty and hide real gains.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.where(inv == g)[0] for g in range(len(uniq))]

    point = _cluster_bacc(y_true, pred_a, labels) - _cluster_bacc(y_true, pred_b, labels)

    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([idx_by_group[g] for g in pick])
        deltas[b] = _cluster_bacc(y_true[rows], pred_a[rows], labels) - _cluster_bacc(
            y_true[rows], pred_b[rows], labels
        )

    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p_two_sided = float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())))
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(deltas.std(ddof=1)),
        "p_two_sided": p_two_sided,
        "significant_at_95": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def per_class_recall_table(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> List[Dict[str, float]]:
    """Per-class recall plus the two failure modes that matter.

    `lost_to_lab`      : true lab was predicted, sibling was wrong  -> fine-grained
    `lost_to_other_lab`: lab was missed entirely                   -> representation
    """
    from src.config import MODEL_TO_LAB

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    rows: List[Dict[str, float]] = []
    for i, lab in enumerate(labels):
        support = int(cm[i].sum())
        correct = int(cm[i, i])
        sib = 0
        other = 0
        for j in range(len(labels)):
            if i == j or cm[i, j] == 0:
                continue
            if MODEL_TO_LAB.get(lab) == MODEL_TO_LAB.get(labels[j]):
                sib += int(cm[i, j])
            else:
                other += int(cm[i, j])
        rows.append(
            {
                "model": lab,
                "lab_name": MODEL_TO_LAB.get(lab, "?"),
                "support": support,
                "recall": correct / support if support else 0.0,
                "lost_to_sibling": sib,
                "lost_to_other_lab": other,
            }
        )
    return rows


def summarize(results: Dict[str, Dict[str, float]]) -> str:
    """Render a compact markdown verdict block."""
    lines = ["| System | BAcc | 95% CI | SE |", "| :--- | ---: | :--- | ---: |"]
    for name, r in results.items():
        lines.append(
            f"| `{name}` | {r['point']:.4f} | "
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['se']:.4f} |"
        )
    return "\n".join(lines)


def length_stratified_balanced_accuracy(
    y_true: Sequence[int | str],
    y_pred: Sequence[int | str],
    texts: Sequence[str],
    bins: Sequence[int] = (0, 60, 90, 120, 160, 250),
) -> Dict[str, float]:
    """
    Computes balanced accuracy across text length bins.
    Allows diagnosing short-passage performance cliffs.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    word_counts = np.array([len(t.split()) for t in texts])

    results = {}
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (word_counts >= low) & (word_counts < high)
        bin_label = f"{low}-{high}"
        if mask.sum() > 0:
            results[bin_label] = float(balanced_accuracy_score(y_t[mask], y_p[mask]))
    return results

