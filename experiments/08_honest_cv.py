"""
Experiment 08: Deployment-faithful CV.

THE FLAW THIS FILE EXISTS TO EXPOSE
-----------------------------------
Experiment 07 reports 0.3937 using `df["prompt_id"]` from train.csv. On test.csv
that column does not exist and the prompts are unseen. So exp07 measures a
pipeline that cannot be deployed: its centering needs ground-truth prompt
grouping, which the rerun never provides.

Every number produced here is therefore one of two things:
  ORACLE   -- clusters given by prompt_id (what exp07 did). Fiction.
  DEPLOYED -- clusters inferred from text only. Reality.

The gap between them is the single most important number in this repository.
If ARI is below ~0.95 the entire cavity/centering family is a liability, not an
asset, and must be gated by cluster confidence.

Run: python experiments/08_honest_cv.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, VALID_18_MODELS, MODEL_TO_IDX
from src.eval_stats import balanced_accuracy_ci, repeated_grouped_folds
from src.postproc import sinkhorn_matching_posterior


# --------------------------------------------------------------------------- #
# Cluster inference -- the part everyone skips
# --------------------------------------------------------------------------- #
def embed_for_clustering(texts: List[str], n_components: int = 128) -> np.ndarray:
    """Topic-level embedding, deliberately NOT style-level.

    For grouping we want what the 18 passages of a prompt SHARE: the subject.
    TF-IDF + SVD is a strong, cheap baseline. On Kaggle swap in a sentence
    encoder (bge-small / e5-small) and average with this -- but re-run the
    threshold calibration below, because the threshold is not transferable.
    """
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    X = vec.fit_transform(texts)
    Z = (
        TruncatedSVD(n_components=min(n_components, X.shape[1] - 1), random_state=42).fit_transform(X)
        if X.shape[1] > n_components
        else X.toarray()
    )
    return normalize(Z)


def calibrate_threshold(
    Z: np.ndarray, true_groups: np.ndarray, grid: np.ndarray | None = None
) -> Tuple[float, float, int]:
    """Pick the cosine distance threshold that maximises ARI against known groups.

    Calibrating on train is legitimate: train prompt_ids are given, and the
    clustering rule (threshold) transfers to test. The ARI achieved here is the
    CEILING for any cluster-dependent post-processing.
    """
    if grid is None:
        grid = np.linspace(0.15, 0.95, 33)
    best = (-1.0, 0.0, -1)
    for t in grid:
        pred = AgglomerativeClustering(
            n_clusters=None, distance_threshold=float(t), metric="cosine", linkage="average"
        ).fit_predict(Z)
        ari = adjusted_rand_score(true_groups, pred)
        if ari > best[0]:
            best = (float(ari), float(t), int(pred.max() + 1))
    return best[1], best[0], best[2]


def infer_clusters(Z: np.ndarray, threshold: float) -> np.ndarray:
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=float(threshold), metric="cosine", linkage="average"
    ).fit_predict(Z)


# --------------------------------------------------------------------------- #
# The two evaluations
# --------------------------------------------------------------------------- #
def within_cluster_center(Z: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Remove the cluster mean. With inferred groups this is approximate; hence
    the shrinkage toward the grand mean controlled by `lam`."""
    uniq, inv = np.unique(groups, return_inverse=True)
    means = np.zeros((len(uniq), Z.shape[1]), dtype=np.float32)
    np.add.at(means, inv, Z)
    means /= np.bincount(inv, minlength=len(uniq))[:, None]
    return Z - means[inv] + Z.mean(axis=0, keepdims=True)


def run(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    y = df["model"].values
    prompts = df["prompt_id"].values

    Z = embed_for_clustering(df["text"].fillna("").tolist())

    thr, ari, n_clusters = calibrate_threshold(Z, prompts)
    inferred = infer_clusters(Z, thr)
    print(f"[cluster] threshold={thr:.3f}  ARI={ari:.4f}  clusters={n_clusters} (truth=150)")
    sizes = pd.Series(inferred).value_counts()
    print(
        f"[cluster] size distribution: min={sizes.min()} p50={sizes.median():.0f} "
        f"max={sizes.max()}  frac_singletons={(sizes == 1).mean():.3f}"
    )
    # Conditioning on cluster size is mandatory: singletons carry zero
    # information and centering them deletes the row's identity entirely.
    print(
        "[cluster] WARNING: any centering step must SKIP singleton and size-2 "
        "clusters; centering a singleton subtracts the row from itself."
    )

    folds = repeated_grouped_folds(df, n_splits=5, n_repeats=1)[0]
    K = len(VALID_18_MODELS)

    results: Dict[str, Dict[str, float]] = {}
    for tag, groups in (("oracle_clusters", prompts), ("inferred_clusters", inferred)):
        centered = within_cluster_center(Z, groups)
        oof = np.zeros((len(df), K))
        pred = np.empty(len(df), dtype=object)
        for f in range(folds.max() + 1):
            trn, val = np.where(folds != f)[0], np.where(folds == f)[0]
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit(centered[trn], y[trn])
            oof[val] = clf.predict_proba(centered[val])
            pred[val] = clf.predict(centered[val])

        results[tag] = balanced_accuracy_ci(y, pred, prompts, VALID_18_MODELS, n_boot=500)

        # Structural constraint, applied per cluster, soft.
        constrained = pred.copy()
        for g in np.unique(groups):
            rows = np.where(groups == g)[0]
            if len(rows) < 2 or len(rows) > K:
                continue
            lp = np.log(np.clip(oof[rows], 1e-12, 1.0))
            sm = sinkhorn_matching_posterior(lp, beta=0.6, dummy_logit=-1.5)
            constrained[rows] = [VALID_18_MODELS[i] for i in sm.argmax(axis=1)]
        results[f"{tag}+sinkhorn"] = balanced_accuracy_ci(
            y, constrained, prompts, VALID_18_MODELS, n_boot=500
        )

    return results


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    res = run(df)
    print("\n| Setting | BAcc | 95% CI | SE |")
    print("| :--- | ---: | :--- | ---: |")
    for k, r in res.items():
        print(f"| `{k}` | {r['point']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['se']:.4f} |")
    gap = res["oracle_clusters"]["point"] - res["inferred_clusters"]["point"]
    print(f"\n[VERDICT] Deployment gap (oracle - inferred) = {gap:+.4f} BAcc")
    print("If this gap exceeds the SE, every previously reported cluster-dependent score is void.")


if __name__ == "__main__":
    main()
