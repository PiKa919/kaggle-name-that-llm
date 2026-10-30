"""
V6: marginal projection (Sinkhorn-Knopp OT) on recovered prompt clusters.

Drop-in for notebooks/master_sota/submission_sota.py.  Three additions, only
one of which is enabled by default, because that is what the measurements
support:

    operator                                    OOF balanced accuracy
    ----------------------------------------------------------------
    base (4-view TF-IDF + sibling dense)              0.3790
    + pairwise residual injector, lam=0.15            0.3730   NEGATIVE
    + OT on RECOVERED clusters, reg=.10 alpha=.5      0.4920   +0.113
      + 95% cluster-bootstrap CI                            [+0.095, +0.131]
    + short-passage gate, best of 12 configs          0.4912   NEGATIVE

Prompt-grouped 5-fold, seed 42, on data/train.csv.  The semantic (Qwen3-8B
LoRA) view is absent from this machine, so the base here is the n-gram view:
`AGENTS.md` forbids local model execution.  Every hyper-parameter below is a
property of the posterior geometry, but re-fit alpha on the real fused OOF if
you have it -- `experiments/27_end_to_end.py` prints the sweep.

DEPLOYMENT NOTE (important)
---------------------------
`src/` is not on the path inside a Kaggle kernel.  Upload this repo's `src/`
directory as a private dataset (e.g. `pikaaa/ntl-modules`) and add it to
`dataset_sources`; then `sinkhorn_knopp_transport` is used verbatim and
`extract_sibling_features` becomes available.  If the dataset is absent the
module falls back to the vendored solver below, which was validated to within
-0.011 balanced accuracy of the reviewed module at the operating point
(0.4857 vs 0.4968).  That is the whole cost of not uploading the dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

EPS = 1e-12
_CHECK_EVERY = 5

# --- measured operating point ------------------------------------------------
# reg: flat plateau 0.05-0.20 (0.4853 / 0.4916 / 0.4856 at alpha=0.5, recovered
#      clusters).  0.10 is the interior optimum.
# alpha: 0.35-0.50 (0.4920 / 0.4916).  1.0 is measurably worse (0.4871) because
#      a full projection assumes every cluster is a complete 18-model round,
#      which the hidden test is not.
OT_REG = 0.10
OT_ALPHA = 0.50
CLUSTER_DISTANCE = 0.35      # cosine; gives 174 clusters for 150 prompts


# --------------------------------------------------------------------------- #
# Optimal transport core: prefer the reviewed module, fall back to a vendored
# solver with the same signature and the same epsilon-scaling schedule.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on the Kaggle dataset being mounted
    import sys as _sys
    _sys.path.insert(0, "/kaggle/input/ntl-modules")
    from src.test_time_calibration import (
        sinkhorn_knopp_transport as _sinkhorn_module,
    )
    _HAVE_MODULE = True
except Exception:  # noqa: BLE001
    _sinkhorn_module = None
    _HAVE_MODULE = False


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(a - m).sum(axis=axis))


def _sinkhorn_fallback(P, reg=OT_REG, alpha=OT_ALPHA, max_iter=200, tol=1e-4):
    """Log-domain Sinkhorn with epsilon-scaling. Same contract as the module.

    The epsilon schedule is not cosmetic: at reg=0.1 the cost spans ~28 nats, so
    the kernel spans e^280 and plain alternating projections stall (0.445 vs
    0.497 measured).  Halve reg from the cost scale down to the target.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] == 0:
        return P
    n, K = P.shape
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0 or n < 2:
        return P
    P = np.clip(np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    s = P.sum(axis=1, keepdims=True)
    P = np.where(s > EPS, P / np.maximum(s, EPS), 1.0 / K)

    cost = -np.log(np.maximum(P, EPS))
    cost -= cost.max()
    log_r = np.full(n, -np.log(n))
    log_c = np.full(K, -np.log(K))

    span = float(cost.max() - cost.min())
    stages = [max(float(reg), span)]
    while stages[-1] > float(reg) * 1.0001 and len(stages) < 16:
        stages.append(max(stages[-1] / 2.0, float(reg)))

    f, g = np.zeros(n), np.zeros(K)
    prev = stages[0]
    for si, eps in enumerate(stages):
        if si > 0:
            f *= prev / eps
            g *= prev / eps
        prev = eps
        stage_tol = tol if si == len(stages) - 1 else max(tol, 1e-3)
        for it in range(int(max_iter)):
            f = log_r - _lse(g[None, :] - cost / eps, axis=1)
            g = log_c - _lse(f[:, None] - cost / eps, axis=0)
            if it % _CHECK_EVERY == 0:
                plan = np.exp(np.clip(f[:, None] + g[None, :] - cost / eps,
                                      -700.0, 700.0))
                if float(np.abs(plan.sum(axis=0) - np.exp(log_c)).max()) < stage_tol:
                    break
    plan = np.exp(np.clip(f[:, None] + g[None, :] - cost / float(reg),
                          -700.0, 700.0))
    if not np.isfinite(plan).all():
        safe = max(float(reg), span / 4.0, 0.5)
        f, g = np.zeros(n), np.zeros(K)
        for _ in range(int(max_iter)):
            f = log_r - _lse(g[None, :] - cost / safe, axis=1)
            g = log_c - _lse(f[:, None] - cost / safe, axis=0)
        plan = np.exp(np.clip(f[:, None] + g[None, :] - cost / safe, -700.0, 700.0))

    out = (1.0 - alpha) * P + alpha * (plan * n)
    out = np.clip(out, 0.0, None)
    return out / np.maximum(out.sum(axis=1, keepdims=True), EPS)


def sinkhorn_marginal_projection(P, reg=OT_REG, alpha=OT_ALPHA, **kw):
    """Project one prompt cluster's posteriors toward uniform class mass."""
    if _HAVE_MODULE:
        return _sinkhorn_module(P, reg=reg, alpha=alpha, **kw)
    return _sinkhorn_fallback(P, reg=reg, alpha=alpha, **kw)


# --------------------------------------------------------------------------- #
# Prompt-cluster recovery
# --------------------------------------------------------------------------- #
def recover_prompt_clusters(texts, distance_threshold=CLUSTER_DISTANCE,
                            svd_dim=128, max_features=80_000, seed=42):
    """Cluster passages by CONTENT, because `test.csv` has no prompt_id.

    Measured on train against the true prompt_id: 174 clusters for 150 prompts,
    purity 0.986, median size 18.  OT through these keeps 0.4920 of the 0.4968
    that oracle clusters give, so recovery costs 0.005 and not the 0.11 gain.

    Deliberately NOT the stylometric views: they are engineered to be
    topic-invariant, so clustering on them reaches purity 0.27 and OT through
    them lands at or below the unprojected posterior.
    """
    texts = [t or "" for t in texts]
    if len(texts) < 2:
        return np.zeros(len(texts), dtype=int)
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True, max_features=max_features,
                          dtype=np.float32)
    X = normalize(vec.fit_transform(texts))
    dim = int(min(svd_dim, max(2, X.shape[1] - 1)))
    F = normalize(TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X))
    return AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=float(distance_threshold)).fit_predict(F)


def apply_marginal_projection(p_final, texts, reg=OT_REG, alpha=OT_ALPHA,
                              min_cluster=4, min_clusters_ratio=1.0 / 60.0):
    """Project each recovered cluster, with two guards that earn their keep.

    min_cluster=4
        A 2-3 row cluster has no meaningful column marginal to impose.
    min_clusters_ratio
        If clustering degenerates into one giant blob, OT to uniform actively
        hurts: one-cluster OT measured 0.3712 against a 0.3790 unprojected
        base.  Bail out and return the input unchanged instead.
    """
    p = np.asarray(p_final, dtype=np.float64)
    labels = recover_prompt_clusters(texts)
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    groups = np.split(order, bounds)

    usable = [g for g in groups if len(g) >= min_cluster]
    if len(usable) < max(2, min_clusters_ratio * len(labels)):
        print(f"[OT] clustering degenerated ({len(usable)} usable clusters over "
              f"{len(labels)} rows) -- skipping projection")
        return p

    out = p.copy()
    for g in usable:
        out[g] = sinkhorn_marginal_projection(p[g], reg=reg, alpha=alpha)
    sizes = np.array([len(g) for g in usable])
    print(f"[OT] {len(groups)} clusters ({len(usable)} usable, median "
          f"{np.median(sizes):.0f} rows), reg={reg} alpha={alpha}, "
          f"module={'yes' if _HAVE_MODULE else 'vendored'}")
    return out


# --------------------------------------------------------------------------- #
# Optional sibling dense block (module 1).  Measured +0.0103 with 95% CI
# [-0.0037, +0.0263] -- not established.  Put it in the 18-way base, NOT in the
# 8 pairwise heads: there it changed mean pairwise accuracy by exactly 0.0000
# (0.6525 -> 0.6525) and made 3 of 8 pairs worse.
# --------------------------------------------------------------------------- #
def build_sibling_block(texts_train, texts_test, scale=0.1):
    """Return (Z_train, Z_test) or (None, None) if `src` is not mounted.

    The module normalises per BATCH, so train and test must be extracted in ONE
    call -- separate calls put the two matrices on different scales.
    """
    try:
        from src.sibling_features import extract_sibling_features
    except Exception:  # noqa: BLE001
        print("[sibling] src.sibling_features unavailable -- skipping block")
        return None, None
    n_tr = len(texts_train)
    Z = extract_sibling_features(list(texts_train) + list(texts_test))
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0) * float(scale)
    print(f"[sibling] {Z.shape[1]} features x {Z.shape[0]} rows")
    return Z[:n_tr], Z[n_tr:]
