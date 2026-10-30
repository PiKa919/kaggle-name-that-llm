"""
Next-generation test-time calibration, prompt deconfounding, and unbalanced
optimal transport for Kaggle 'Name That LLM'.

Mathematical Architecture
-------------------------
1. UNBALANCED OPTIMAL TRANSPORT (UOT) VIA GENERALIZED SINKHORN-KNOPP:
   Relaxes the rigid uniform column marginal constraint (len(P) / K) of classical
   optimal transport via entropic Kullback-Leibler penalization (tau > 0):

     min_{T >= 0} <T, C> + eps * KL(T || r (x) c) + tau_col * KL(T^T 1 || c)

   subject to strict row-stochasticity (T 1 = 1 / N, tau_row -> inf).
   The log-domain dual updates contract by kappa = tau_col / (tau_col + eps) in [0, 1]:
   - tau_col -> inf recovers classical balanced Sinkhorn-Knopp.
   - tau_col -> 0 relaxes to unconstrained temperature-scaled softmax.
   - Intermediate tau in [0.2, 5.0] prevents catastrophic class collapse on rare models
     while smoothly accommodating empirical public/private test set label shift.

2. ENHANCED HODGERANK CYCLE-FREE POTENTIAL PROJECTION:
   Projects arbitrary pairwise sibling discriminators (log-odds flows Y_ij) onto
   the curl-free gradient of a global potential s in R^K using a confidence-weighted
   graph Laplacian:

     min_s sum_{(i,j) in E} W_ij * (s_j - s_i - Y_ij)^2  ==>  L_W s = div_W(Y)

   where edge affinities W_ij scale with confidence (W_ij proportional to |delta logit|^beta).
   The resulting potential gradient s_j - s_i is mathematically guaranteed to be
   transitive (curl-free), eliminating intransitive Condorcet voting cycles among LLMs.

3. DYNAMIC TEST-SET TOPIC CLUSTERING:
   Replaces fragile hard-coded linkage cuts (e.g. static 0.85) with an adaptive sweep
   over candidate thresholds theta in [theta_min, theta_max]. The objective function
   jointly optimizes valid cluster yield (sizes 2..18), within-cluster cohesion,
   penalizes invalid oversized clusters (>18 models answering a single prompt),
   and bounds singleton fragmentation.

4. TWO-WAY ANOVA & WITHIN-PROMPT RESIDUAL COVARIANCE SHRINKAGE (wpca + OAS / LW):
   Decomposes passage representations z_i = mu_{y_i} + p_{g(i)} + e_i.
   Jointly diagonalizes residual error covariance S_e and prompt covariance S_p:
     A S_e A^T = I,   A S_p A^T = diag(lambda).
   Performs cavity EM marginal inference on detected test cliques with leave-one-out
   variance v = lambda / (1 + (n - 1) lambda) and at-most-once (injective) soft repulsion.

5. UNIFIED ORCHESTRATOR (`UnbalancedCalibrator`):
   End-to-end grid tuning and inference pipeline with strict monotonicity fallback:
   guaranteed never to regress below the uncalibrated baseline on out-of-fold validation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import squareform
from scipy.special import logsumexp

__all__ = [
    "N_CLASSES",
    "EPS",
    "DEFAULT_REG",
    "DEFAULT_TAU",
    "DEFAULT_ALPHA",
    "scale_logits",
    "temperature_scale_logits",
    "fit_temperature",
    "fit_class_temperatures",
    "oas_shrinkage",
    "ledoit_wolf_shrinkage",
    "shrink_covariance",
    "two_way_anova",
    "within_prompt_pca",
    "fit_shared_prompt_gaussian",
    "cavity_em_marginals",
    "adaptive_topic_clustering",
    "DynamicTopicClusterer",
    "project_hodgerank_weighted",
    "HodgeRankProjector",
    "unbalanced_sinkhorn_transport",
    "UnbalancedCalibrator",
    "per_class_recall",
    "balanced_accuracy",
    "test_unbalanced_invariants",
]

EPS = 1e-12
N_CLASSES = 18
DEFAULT_REG = 0.05
DEFAULT_TAU = 1.0
DEFAULT_ALPHA = 0.5
_CHECK_EVERY = 5
_T_MIN, _T_MAX = 1e-3, 1e3

ArrayLike = Union[np.ndarray, Sequence[float]]


# =========================================================================== #
# 1. Numerical & Statistical Utilities
# =========================================================================== #
def _as_2d(x: ArrayLike, name: str = "matrix") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Eliminates NaNs and Infs to safeguard numerical chains."""
    return np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z_max = np.max(z, axis=axis, keepdims=True)
    z_clean = np.where(np.isfinite(z_max), z - z_max, 0.0)
    e = np.exp(z_clean)
    s = e.sum(axis=axis, keepdims=True)
    return e / np.clip(s, EPS, None)


def _as_prob_matrix(P: ArrayLike) -> np.ndarray:
    """Coerces input to a valid row-stochastic probability matrix."""
    arr = _sanitize(_as_2d(P, "P"))
    arr = np.clip(arr, 0.0, None)
    if arr.shape[0] == 0:
        return arr
    s = arr.sum(axis=1, keepdims=True)
    dead = s[:, 0] <= EPS
    if dead.any():
        arr[dead] = 1.0 / arr.shape[1]
        s = arr.sum(axis=1, keepdims=True)
    return arr / s


def _as_labels(y: ArrayLike, n: int, K: int) -> np.ndarray:
    if n == 0:
        raise ValueError("Cannot fit on an empty batch.")
    y_int = np.asarray(y).astype(int).reshape(-1)
    if y_int.shape[0] != n:
        raise ValueError(f"Label count ({y_int.shape[0]}) does not match sample count ({n}).")
    if y_int.min() < 0 or y_int.max() >= K:
        raise ValueError(f"Labels must be in [0, {K - 1}], got [{y_int.min()}, {y_int.max()}].")
    return y_int


def per_class_recall(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: Optional[int] = None
) -> np.ndarray:
    """Unweighted recall per class; absent classes are set to NaN."""
    yt = np.asarray(y_true).astype(int).reshape(-1)
    yp = np.asarray(y_pred).astype(int).reshape(-1)
    if n_classes is None:
        n_classes = int(max(yt.max(initial=0), yp.max(initial=0))) + 1
    rec = np.full(n_classes, np.nan)
    for k in range(n_classes):
        mask = yt == k
        if mask.any():
            rec[k] = float(np.mean(yp[mask] == k))
    return rec


def balanced_accuracy(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: Optional[int] = None
) -> float:
    """Kaggle evaluation metric: unweighted mean of per-class recalls."""
    rec = per_class_recall(y_true, y_pred, n_classes)
    if np.all(np.isnan(rec)):
        return 0.0
    return float(np.nanmean(rec))


# =========================================================================== #
# 2. Temperature Scaling
# =========================================================================== #
def scale_logits(logits: ArrayLike, temperatures: ArrayLike = 1.0) -> np.ndarray:
    """Scales logits by a global scalar or per-class (K,) temperature vector."""
    z = _sanitize(_as_2d(logits, "logits"))
    K = z.shape[1]
    t = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    if t.size == 1:
        t = np.full(K, float(t[0]))
    if t.size != K:
        raise ValueError(f"temperatures must be scalar or length {K}, got {t.size}")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("Temperatures must be finite and strictly positive.")
    t = np.clip(t, _T_MIN, _T_MAX)
    return z / t[None, :]


def temperature_scale_logits(
    logits: ArrayLike, temperatures: ArrayLike = 1.0
) -> np.ndarray:
    """Computes row-stochastic probabilities after temperature scaling."""
    return _softmax(scale_logits(logits, temperatures))


def _nll(z: np.ndarray, y: np.ndarray) -> float:
    lse = logsumexp(z, axis=1)
    return float(np.mean(lse - z[np.arange(z.shape[0]), y]))


def fit_temperature(
    logits: ArrayLike,
    y: ArrayLike,
    bounds: Tuple[float, float] = (0.05, 20.0),
    l2: float = 0.0,
    center: float = 1.0,
) -> float:
    """Maximum-likelihood global temperature on validation logits."""
    z = _sanitize(_as_2d(logits, "logits"))
    n, K = z.shape
    y_int = _as_labels(y, n, K)
    lo, hi = np.log(max(bounds[0], _T_MIN)), np.log(min(bounds[1], _T_MAX))
    log_center = np.log(max(center, _T_MIN))

    def objective(log_t: float) -> float:
        return _nll(z / np.exp(log_t), y_int) + l2 * (log_t - log_center) ** 2

    res = minimize_scalar(
        objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6}
    )
    return float(np.exp(res.x))


def fit_class_temperatures(
    logits: ArrayLike,
    y: ArrayLike,
    shrink: float = 1.0,
    bounds: Tuple[float, float] = (0.05, 20.0),
    t_global: Optional[float] = None,
) -> np.ndarray:
    """Per-class temperatures fitted via penalized maximum likelihood."""
    z = _sanitize(_as_2d(logits, "logits"))
    n, K = z.shape
    y_int = _as_labels(y, n, K)
    if t_global is None:
        t_global = fit_temperature(z, y_int, bounds=bounds)
    counts = np.bincount(y_int, minlength=K)
    log_t_global = np.log(max(t_global, _T_MIN))
    lo, hi = np.log(max(bounds[0], _T_MIN)), np.log(min(bounds[1], _T_MAX))
    rows = np.arange(n)
    onehot = np.zeros((n, K))
    onehot[rows, y_int] = 1.0

    def objective(log_t: np.ndarray) -> Tuple[float, np.ndarray]:
        t = np.exp(log_t)
        zz = z / t[None, :]
        lse = logsumexp(zz, axis=1)
        nll = float(np.mean(lse - zz[rows, y_int]))
        p = np.exp(zz - lse[:, None])
        grad = (np.sum(onehot * z, axis=0) - np.sum(p * z, axis=0)) / (n * t)
        pen = (log_t - log_t_global) ** 2
        grad = grad + 2.0 * shrink * (log_t - log_t_global) / n
        return nll + float(shrink * np.sum(pen) / n), grad

    res = minimize(
        objective,
        x0=np.full(K, log_t_global),
        jac=True,
        method="L-BFGS-B",
        bounds=[(lo, hi)] * K,
        options={"maxiter": 200},
    )
    log_t = np.clip(res.x, lo, hi)
    log_t[counts == 0] = log_t_global
    return np.exp(log_t)


# =========================================================================== #
# 3. Covariance Shrinkage: Ledoit-Wolf & OAS (Chen et al. 2010)
# =========================================================================== #
def shrink_covariance(S: np.ndarray, alpha: float) -> np.ndarray:
    """Applies convex shrinkage toward a spherical target: (1-a)*S + a*(tr(S)/d)*I."""
    d = S.shape[0]
    tr = float(np.trace(S))
    mu = tr / d
    return (1.0 - alpha) * S + alpha * mu * np.eye(d)


def oas_shrinkage(X: np.ndarray, assume_centered: bool = True) -> float:
    """Oracle Approximating Shrinkage (OAS) coefficient (Chen et al. 2010).

    OAS achieves lower mean-squared error than standard Ledoit-Wolf under Gaussian
    assumptions, particularly when sample sizes are small to moderate.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if not assume_centered:
        X_arr = X_arr - X_arr.mean(axis=0)
    n, p = X_arr.shape
    if n <= 1 or p <= 1:
        return 1.0
    emp_cov = (X_arr.T @ X_arr) / n
    alpha = float(np.mean(emp_cov ** 2))
    mu = float(np.trace(emp_cov)) / p
    mu_squared = mu ** 2
    num = alpha + mu_squared
    den = (n + 1.0) * (alpha - mu_squared / p)
    if den <= 1e-15:
        return 1.0
    return float(np.clip(num / den, 0.0, 1.0))


def ledoit_wolf_shrinkage(X: np.ndarray, assume_centered: bool = True) -> float:
    """Ledoit-Wolf shrinkage coefficient for centered data (pure NumPy)."""
    try:
        from sklearn.covariance import ledoit_wolf_shrinkage as _sk_lw
        return float(_sk_lw(X, assume_centered=assume_centered))
    except ImportError:
        pass
    X_arr = np.asarray(X, dtype=np.float64)
    if not assume_centered:
        X_arr = X_arr - X_arr.mean(axis=0)
    n, p = X_arr.shape
    if n <= 1 or p <= 1:
        return 1.0
    emp_cov = (X_arr.T @ X_arr) / n
    mu = float(np.trace(emp_cov)) / p
    delta_sq = float(np.sum((emp_cov - mu * np.eye(p)) ** 2))
    if delta_sq <= 1e-15:
        return 1.0
    # Sample variance of sample covariance elements
    X_sq = X_arr[:, :, None] * X_arr[:, None, :]
    beta_bar = float(np.sum(np.var(X_sq, axis=0, ddof=0))) / n
    beta_sq = min(beta_bar, delta_sq)
    return float(np.clip(beta_sq / delta_sq, 0.0, 1.0))


# =========================================================================== #
# 4. Two-Way ANOVA & Within-Prompt Residual PCA (wpca)
# =========================================================================== #
def two_way_anova(
    Z: np.ndarray,
    y: np.ndarray,
    pinv: np.ndarray,
    n_prompts: int,
    n_classes: int = N_CLASSES,
    iters: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits additive decomposition Z = mu[y] + P[prompt] + E by alternating expectations.

    Parameters
    ----------
    Z : (N, D) feature array
    y : (N,) class labels 0..K-1
    pinv : (N,) integer prompt group index
    n_prompts : number of unique prompt groups G
    n_classes : K (18)
    iters : alternating projections

    Returns
    -------
    mu : (K, D) class prototypes
    P : (G, D) prompt-specific additive random effects
    cnt_p : (G,) sample count per prompt
    """
    D = Z.shape[1]
    P = np.zeros((n_prompts, D), dtype=np.float64)
    cnt_p = np.bincount(pinv, minlength=n_prompts).astype(np.float64)
    cnt_p = np.maximum(cnt_p, 1.0)
    cls = np.unique(y)
    mu = np.zeros((n_classes, D), dtype=np.float64)

    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls:
            mask = y == c
            if mask.any():
                mu[c] = R[mask].mean(axis=0)
        R = Z - mu[y]
        P = np.zeros_like(P)
        np.add.at(P, pinv, R)
        P /= cnt_p[:, None]

    return mu, P, cnt_p


def within_prompt_pca(
    X: np.ndarray,
    prompt_ids: ArrayLike,
    d: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects standardized features onto within-prompt residual eigenvectors.

    Removes prompt-level topic shifts prior to model prototype estimation.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    N, D = X_arr.shape
    pu, pinv = np.unique(np.asarray(prompt_ids), return_inverse=True)
    G = len(pu)

    mx = X_arr.mean(axis=0)
    sx = X_arr.std(axis=0)
    sx = np.where(sx > 1e-8, sx, 1.0)
    Xs = (X_arr - mx) / sx

    Pm = np.zeros((G, D), dtype=np.float64)
    cnt = np.bincount(pinv, minlength=G).astype(np.float64)
    np.add.at(Pm, pinv, Xs)
    Pm /= np.maximum(cnt[:, None], 1.0)
    R = Xs - Pm[pinv]

    d_eff = int(min(d, max(N - G - 1, 1), D))
    if N < D:
        # Gram economy SVD
        w, U = np.linalg.eigh(R @ R.T)
        o = np.argsort(w)[::-1][:d_eff]
        w, U = np.clip(w[o], 1e-12, None), U[:, o]
        V = (R.T @ U) / np.sqrt(w)
    else:
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt[:d_eff].T

    return V, mx, sx


def fit_shared_prompt_gaussian(
    X: ArrayLike,
    y: ArrayLike,
    prompt_ids: ArrayLike,
    cfg: Optional[Dict[str, Union[str, float, int]]] = None,
) -> Dict[str, Union[np.ndarray, float, int, dict]]:
    """Fits the shared-prompt generative Gaussian model:

        z_i = mu_{y_i} + p_{g(i)} + e_i,
        p_g ~ N(0, S_p),   e_i ~ N(0, S_e)

    Uses joint diagonalization A S_e A^T = I, A S_p A^T = diag(lambda) so dimensions
    are orthogonal and independent.
    """
    default_cfg = {
        "red": "wpca",
        "d": 128,
        "se_shrink": "lw",
        "sp_shrink": "lw",
        "lam_floor": 1e-4,
        "n_exact": 3,
        "iters": 5,
        "repel": True,
        "sp_scale": 1.0,
    }
    c = dict(default_cfg)
    if cfg is not None:
        c.update(cfg)

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=int)
    N, D = X_arr.shape
    pu, pinv = np.unique(np.asarray(prompt_ids), return_inverse=True)
    G = len(pu)

    if c["red"] == "wpca":
        V, mx, sx = within_prompt_pca(X_arr, prompt_ids, d=int(c["d"]))
    elif c["red"] == "pca":
        mx = X_arr.mean(axis=0)
        sx = X_arr.std(axis=0)
        sx = np.where(sx > 1e-8, sx, 1.0)
        Xs = (X_arr - mx) / sx
        _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
        d_eff = int(min(c["d"], D))
        V = Vt[:d_eff].T
    else:
        mx = X_arr.mean(axis=0)
        sx = np.ones(D, dtype=np.float64)
        V = np.eye(D)

    Xs = (X_arr - mx) / sx
    Z = Xs @ V
    d = Z.shape[1]

    mu, P, cnt_p = two_way_anova(Z, y_arr, pinv, G, n_classes=N_CLASSES)
    E = Z - mu[y_arr] - P[pinv]
    dof = max(N - G - len(np.unique(y_arr)) + 1, 1)
    Se_raw = (E.T @ E) / dof

    if c["se_shrink"] == "oas":
        ae = oas_shrinkage(E, assume_centered=True)
    elif c["se_shrink"] == "lw":
        ae = ledoit_wolf_shrinkage(E, assume_centered=True)
    else:
        ae = float(c["se_shrink"])
    Se = shrink_covariance(Se_raw, ae)

    Pc = P - P.mean(axis=0)
    Sp_raw = (Pc.T @ Pc) / max(G - 1, 1) - np.mean(1.0 / cnt_p) * Se_raw
    if c["sp_shrink"] == "oas":
        ap = oas_shrinkage(Pc, assume_centered=True)
    elif c["sp_shrink"] == "lw":
        ap = ledoit_wolf_shrinkage(Pc, assume_centered=True)
    else:
        ap = float(c["sp_shrink"])

    tr_se = float(np.trace(Se))
    tr_sp = max(float(np.trace(Sp_raw)), 1e-8 * tr_se)
    Sp = (1.0 - ap) * Sp_raw + ap * (tr_sp / d) * np.eye(d)
    Sp *= float(c["sp_scale"])

    # Joint Diagonalization: A S_e A^T = I, A S_p A^T = diag(lambda)
    L = np.linalg.cholesky(Se)
    Li = np.linalg.inv(L)
    M = Li @ Sp @ Li.T
    M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, float(c["lam_floor"]), None)
    A = Q.T @ Li
    W = V @ A.T

    return {
        "mx": mx,
        "sx": sx,
        "W": W,
        "m": mu @ A.T,
        "lam": lam,
        "T1": 1.0,
        "Tg": 1.0,
        "info": {
            "d": int(d),
            "ae": round(ae, 4),
            "ap": round(ap, 4),
            "lam_median": float(np.median(lam)),
        },
        "cfg": c,
    }


def cavity_em_marginals(
    U: np.ndarray,
    m: np.ndarray,
    lam: np.ndarray,
    T: float = 1.0,
    iters: int = 5,
    repel: bool = True,
    q0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mean-field Cavity EM marginal inference on detected test cliques.

    Computes leave-one-out cavity prompt means with finite-population correction
    v = lam / (1 + (n - 1) * lam) and optional injective model repulsion.
    """
    n = len(U)
    if n <= 1:
        R = U[:, None, :] - m[None]
        log_lik = -0.5 * (R ** 2 / (1.0 + lam)).sum(axis=-1) / T
        return _softmax(log_lik, axis=-1)

    v = lam / (1.0 + (n - 1) * lam)
    if q0 is None:
        R = U[:, None, :] - m[None]
        L = -0.5 * (R ** 2 / (1.0 + lam)).sum(axis=-1) / T
    else:
        L = np.log(np.clip(q0, 1e-300, None))
    q = _softmax(L, axis=-1)

    for _ in range(iters):
        s = U - q @ m
        S = s.sum(axis=0)
        Pm = (S[None, :] - s) * v
        R = U[:, None, :] - Pm[:, None, :] - m[None]
        L = -0.5 * (R ** 2 / (1.0 + v)).sum(axis=-1) / T
        if repel:
            lo = np.log(np.clip(1.0 - q, 1e-6, None))
            L = L + (lo.sum(axis=0)[None, :] - lo)
        q = _softmax(L, axis=-1)

    return q


# =========================================================================== #
# 5. Dynamic Topic Clustering with Adaptive Linkage Threshold
# =========================================================================== #
def adaptive_topic_clustering(
    D: np.ndarray,
    min_thresh: float = 0.40,
    max_thresh: float = 0.95,
    n_candidates: int = 25,
    min_size: int = 2,
    max_size: int = 18,
    oversize_penalty: float = 3.0,
    singleton_penalty: float = 0.2,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Dynamically cuts the hierarchical linkage tree by optimizing cluster quality.

    Unlike fixed thresholds (e.g. static 0.85), this evaluates a grid of candidate
    thresholds to maximize valid passage yield (clusters of size 2-18) while
    severely penalizing invalid oversized clusters (>18).
    """
    N = len(D)
    if N < 2:
        return 0.85, np.ones(N, dtype=int), {"valid_frac": 0.0, "threshold": 0.85}

    D_mat = np.asarray(D, dtype=np.float64).copy()
    np.fill_diagonal(D_mat, 0.0)
    condensed = squareform(D_mat, checks=False)
    Z = linkage(condensed, method="average")

    candidates = np.linspace(min_thresh, max_thresh, n_candidates)
    best_score = -np.inf
    best_thresh = 0.85
    best_groups = np.ones(N, dtype=int)
    best_metrics: Dict[str, float] = {}

    for thresh in candidates:
        g = fcluster(Z, t=thresh, criterion="distance")
        sizes = np.bincount(g)[g]
        valid_mask = (sizes >= min_size) & (sizes <= max_size)
        oversize_mask = sizes > max_size
        singleton_mask = sizes == 1

        valid_frac = float(np.mean(valid_mask))
        oversize_frac = float(np.mean(oversize_mask))
        singleton_frac = float(np.mean(singleton_mask))

        if valid_mask.sum() >= 2:
            same_cluster = (g[:, None] == g[None, :]) & valid_mask[:, None]
            within_dist = float(np.mean(D_mat[same_cluster]))
            cohesion = 1.0 - within_dist
        else:
            cohesion = 0.0

        score = (
            valid_frac
            + 0.5 * cohesion
            - oversize_penalty * oversize_frac
            - singleton_penalty * singleton_frac
        )

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            best_groups = g
            best_metrics = {
                "score": float(score),
                "threshold": float(thresh),
                "valid_frac": valid_frac,
                "oversize_frac": oversize_frac,
                "singleton_frac": singleton_frac,
                "n_clusters": float(len(np.unique(g))),
            }

    return best_thresh, best_groups, best_metrics


@dataclass
class DynamicTopicClusterer:
    """Topic clusterer with adaptive linkage thresholding."""

    min_thresh: float = 0.40
    max_thresh: float = 0.95
    n_candidates: int = 25
    min_size: int = 2
    max_size: int = 18
    selected_threshold: float = 0.85
    report: Dict[str, float] = field(default_factory=dict)

    def fit_predict(self, D: np.ndarray) -> np.ndarray:
        th, groups, metrics = adaptive_topic_clustering(
            D,
            min_thresh=self.min_thresh,
            max_thresh=self.max_thresh,
            n_candidates=self.n_candidates,
            min_size=self.min_size,
            max_size=self.max_size,
        )
        self.selected_threshold = th
        self.report = metrics
        return groups


# =========================================================================== #
# 6. Enhanced HodgeRank Potential Projection with Confidence-Weighted Affinities
# =========================================================================== #
def project_hodgerank_weighted(
    corrections: Dict[Tuple[Union[str, int], Union[str, int]], np.ndarray],
    edge_weights: Optional[Dict[Tuple[Union[str, int], Union[str, int]], float]] = None,
    confidence_weighting: bool = True,
    beta: float = 1.0,
    n_classes: int = N_CLASSES,
    label_to_idx: Optional[Dict[str, int]] = None,
    return_curl: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """HodgeRank combinatorial graph Helmholtz-Hodge potential projection.

    Given pairwise sibling deltas Y_ij (e.g. log-odds log P(j)/P(i)), projects
    them onto the curl-free potential gradient s in R^K by solving:

        L_W s = div_W(Y)

    where edge weights W_ij incorporate both global classifier reliability and
    sample-specific confidence:
        W_ij^{(n)} = w_edge * (1.0 + |Y_ij^{(n)}|^beta).

    Parameters
    ----------
    corrections : dict mapping (model1, model2) to (N,) deltas
    edge_weights : dict mapping (model1, model2) to prior confidence scalar
    confidence_weighting : whether to modulate by |delta|^beta
    beta : power scaling for margin confidence
    n_classes : K (18)
    label_to_idx : optional dictionary mapping string labels to integer indices
    return_curl : whether to return relative curl inconsistency

    Returns
    -------
    V : (N, K) curl-free potential matrix
    (optional) curl_energy : fraction of flow residing in non-transitive cycles
    """
    if not corrections:
        zeros = np.zeros((1, n_classes), dtype=np.float64)
        return (zeros, 0.0) if return_curl else zeros

    sample_val = next(iter(corrections.values()))
    N = sample_val.shape[0]

    def _idx(label: Union[str, int]) -> int:
        if isinstance(label, (int, np.integer)):
            return int(label)
        if label_to_idx is not None and label in label_to_idx:
            return label_to_idx[label]
        raise ValueError(f"Label {label} not found in index mapping.")

    # Accumulate weighted edge affinities and divergence
    W_graph = np.zeros((n_classes, n_classes), dtype=np.float64)
    B = np.zeros((N, n_classes), dtype=np.float64)
    total_flow_energy = 0.0

    for (m1, m2), delta in corrections.items():
        i1, i2 = _idx(m1), _idx(m2)
        d = np.asarray(delta, dtype=np.float64).reshape(-1)
        base_w = edge_weights.get((m1, m2), 1.0) if edge_weights is not None else 1.0

        if confidence_weighting:
            conf = 1.0 + np.abs(d) ** beta
            w = base_w * conf
        else:
            w = np.full(N, base_w, dtype=np.float64)

        w_mean = float(np.mean(w))
        W_graph[i1, i2] += w_mean
        W_graph[i2, i1] += w_mean

        B[:, i2] += w * d
        B[:, i1] -= w * d
        total_flow_energy += float(np.sum(w * (d ** 2)))

    # Graph Laplacian: L = D - W
    D_diag = np.diag(W_graph.sum(axis=1))
    L = D_diag - W_graph
    L_pinv = np.linalg.pinv(L)

    # Global curl-free potential: V = B @ L^dagger
    V = B @ L_pinv
    V -= np.mean(V, axis=1, keepdims=True)

    if return_curl:
        # Calculate curl residual: R_ij = Y_ij - (s_j - s_i)
        curl_energy = 0.0
        for (m1, m2), delta in corrections.items():
            i1, i2 = _idx(m1), _idx(m2)
            d = np.asarray(delta, dtype=np.float64).reshape(-1)
            grad = V[:, i2] - V[:, i1]
            residual = d - grad
            curl_energy += float(np.sum(residual ** 2))
        inconsistency = float(curl_energy / max(total_flow_energy, EPS))
        return V, inconsistency

    return V


@dataclass
class HodgeRankProjector:
    """High-level wrapper for HodgeRank pairwise projection."""

    beta: float = 1.0
    confidence_weighting: bool = True
    n_classes: int = N_CLASSES
    label_to_idx: Optional[Dict[str, int]] = None
    curl_inconsistency_: float = 0.0

    def transform(
        self,
        corrections: Dict[Tuple[Union[str, int], Union[str, int]], np.ndarray],
        edge_weights: Optional[Dict[Tuple[Union[str, int], Union[str, int]], float]] = None,
    ) -> np.ndarray:
        V, curl = project_hodgerank_weighted(
            corrections,
            edge_weights=edge_weights,
            confidence_weighting=self.confidence_weighting,
            beta=self.beta,
            n_classes=self.n_classes,
            label_to_idx=self.label_to_idx,
            return_curl=True,
        )
        self.curl_inconsistency_ = curl
        return V


# =========================================================================== #
# 7. Unbalanced Optimal Transport (UOT via Log-Domain Generalized Sinkhorn)
# =========================================================================== #
def _reg_schedule(reg: float, span: float, max_stages: int = 16) -> np.ndarray:
    """Annealing schedule for epsilon scaling to avoid dual explosion."""
    if reg <= 0.0 or span <= 0.0:
        return np.array([max(reg, 1e-6)])
    start = max(reg, span)
    stages = [start]
    while stages[-1] > reg * 1.0001 and len(stages) < max_stages:
        stages.append(max(stages[-1] / 2.0, reg))
    return np.asarray(stages, dtype=np.float64)


def _unbalanced_sinkhorn_stage(
    cost: np.ndarray,
    log_r: np.ndarray,
    log_c: np.ndarray,
    reg: float,
    tau_col: Optional[float],
    tau_row: Optional[float],
    max_iter: int,
    tol: float,
    f: np.ndarray,
    g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool, int]:
    """Single stage of log-domain Unbalanced Sinkhorn-Knopp."""
    n, K = cost.shape
    kappa_c = 1.0 if (tau_col is None or np.isinf(tau_col)) else float(tau_col / (tau_col + reg))
    kappa_r = 1.0 if (tau_row is None or np.isinf(tau_row)) else float(tau_row / (tau_row + reg))

    iters = 0
    converged = False
    row_err = np.inf
    col_err = np.inf
    plan = np.exp(np.clip(f[:, None] + g[None, :] - cost / reg, -700.0, 700.0))

    for iters in range(1, max_iter + 1):
        # 1. Row proximal update (exact or relaxed)
        if kappa_r >= 1.0:
            f = log_r - logsumexp(g[None, :] - cost / reg, axis=1)
        else:
            f = kappa_r * (log_r - logsumexp(g[None, :] - cost / reg, axis=1))

        # 2. Column proximal update (exact or relaxed)
        if kappa_c >= 1.0:
            g = log_c - logsumexp(f[:, None] - cost / reg, axis=0)
        else:
            g = kappa_c * (log_c - logsumexp(f[:, None] - cost / reg, axis=0))

        if iters % _CHECK_EVERY == 0 or iters == max_iter:
            plan = np.exp(np.clip(f[:, None] + g[None, :] - cost / reg, -700.0, 700.0))
            # Optimality conditions:
            # Row target: r if kappa_r=1 else r * exp(- (reg/tau_row) * f)
            if kappa_r >= 1.0:
                target_r = np.exp(log_r)
            else:
                target_r = np.exp(log_r - (reg / tau_row) * f)
            row_err = float(np.abs(plan.sum(axis=1) - target_r).max())

            # Column target: c if kappa_c=1 else c * exp(- (reg/tau_col) * g)
            if kappa_c >= 1.0:
                target_c = np.exp(log_c)
            else:
                target_c = np.exp(log_c - (reg / tau_col) * g)
            col_err = float(np.abs(plan.sum(axis=0) - target_c).max())

            if max(row_err, col_err) < tol:
                converged = True
                break

    return plan, f, g, row_err, col_err, converged, iters


def unbalanced_sinkhorn_transport(
    P: ArrayLike,
    target_col_marginal: Optional[ArrayLike] = None,
    reg: float = DEFAULT_REG,
    tau_col: Optional[float] = DEFAULT_TAU,
    tau_row: Optional[float] = None,
    max_iter: int = 150,
    tol: float = 1e-5,
    alpha: float = DEFAULT_ALPHA,
    return_info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """Unbalanced Optimal Transport with Kullback-Leibler penalization.

    Solves:
        min_{T >= 0} <T, -log P> + reg * KL(T || r (x) c)
                     + tau_col * KL(T^T 1 || c)  [+ tau_row * KL(T 1 || r)]

    Parameters
    ----------
    P : (n, K) posterior probability matrix
    target_col_marginal : (K,) target class distribution (defaults to uniform 1/K)
    reg : entropic regularization parameter eps > 0
    tau_col : column marginal relaxation parameter tau_col > 0.
              tau_col -> inf recovers balanced Sinkhorn.
              tau_col -> 0 relaxes to unconstrained softmax.
    tau_row : row marginal relaxation (None = strict row stochasticity T 1 = 1/N)
    max_iter : max iterations per stage
    tol : convergence tolerance
    alpha : blend parameter (1 - alpha) * P + alpha * (N * T)
    return_info : whether to return convergence metadata

    Returns
    -------
    (n, K) row-stochastic calibrated posterior array
    """
    P_arr = _as_prob_matrix(P)
    n, K = P_arr.shape
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if reg <= 0:
        raise ValueError(f"reg must be > 0, got {reg}")

    if target_col_marginal is None:
        c = np.full(K, 1.0 / K)
    else:
        c_arr = np.asarray(target_col_marginal, dtype=np.float64).reshape(-1)
        if c_arr.size != K or np.any(c_arr < 0) or not np.all(np.isfinite(c_arr)):
            raise ValueError(f"target_col_marginal must be non-negative of length {K}")
        c_sum = c_arr.sum()
        if c_sum <= EPS:
            raise ValueError("target_col_marginal must sum to positive mass")
        c = c_arr / c_sum

    if n == 0 or alpha == 0.0:
        info = {
            "iters": 0.0,
            "row_err": 0.0,
            "col_err": 0.0,
            "converged": 1.0,
            "reg": float(reg),
            "tau_col": float(tau_col if tau_col is not None else -1.0),
            "fallback": 0.0,
            "alpha": alpha,
        }
        return (P_arr.copy(), info) if return_info else P_arr.copy()

    cost = np.clip(-np.log(np.clip(P_arr, EPS, 1.0)), 0.0, None)
    cost -= cost.max()

    log_r = np.full(n, -np.log(n))
    log_c = np.log(np.clip(c, EPS, None))

    span = float(cost.max() - cost.min())
    stages = _reg_schedule(reg, span)

    f = np.zeros(n)
    g = np.zeros(K)
    prev = stages[0]
    iters = 0
    converged = False
    row_err = np.inf
    col_err = np.inf
    plan = np.zeros((n, K))
    fallback = 0.0

    for stage, eps in enumerate(stages):
        if stage > 0:
            scale = prev / eps
            f *= scale
            g *= scale
        prev = eps
        last = stage == len(stages) - 1
        stage_tol = tol if last else max(tol, 1e-3)
        plan, f, g, row_err, col_err, converged, used = _unbalanced_sinkhorn_stage(
            cost, log_r, log_c, eps, tau_col, tau_row, max_iter, stage_tol, f, g
        )
        iters += used
        if last and converged:
            break

    # Numeric fallback check
    if not np.all(np.isfinite(plan)) or (not converged and max(row_err * n, col_err * K) > 0.5):
        safe_reg = max(reg, span / 4.0, 0.5)
        plan, f, g, row_err, col_err, converged, used = _unbalanced_sinkhorn_stage(
            cost, log_r, log_c, safe_reg, tau_col, tau_row, max_iter, tol, np.zeros(n), np.zeros(K)
        )
        iters += used
        fallback = 1.0

    # Strict row stochasticity: n * T sums to 1 per row for tau_row=None
    row_sums = plan.sum(axis=1, keepdims=True)
    plan_stochastic = plan / np.clip(row_sums, EPS, None)

    out = (1.0 - alpha) * P_arr + alpha * plan_stochastic
    out = np.clip(out, 0.0, None)
    out /= np.clip(out.sum(axis=1, keepdims=True), EPS, None)

    info = {
        "iters": float(iters),
        "row_err": float(row_err),
        "col_err": float(col_err),
        "converged": float(converged),
        "reg": float(reg),
        "tau_col": float(tau_col if tau_col is not None else -1.0),
        "fallback": float(fallback),
        "alpha": alpha,
        "n": float(n),
        "K": float(K),
    }

    return (out, info) if return_info else out


# =========================================================================== #
# 8. Unified Next-Generation Test-Time Calibrator
# =========================================================================== #
@dataclass(eq=False)
class UnbalancedCalibrator:
    """Next-generation test-time calibration orchestrator.

    Integrates:
      1. Temperature scaling (MLE global or per-class)
      2. Dynamic topic clustering on inferred test-set cliques
      3. Cavity EM or local neighborhood deconfounding
      4. HodgeRank confidence-weighted pairwise potential projection
      5. Unbalanced Optimal Transport with adaptive marginal shift
    """

    temperature: float = 1.0
    class_temperatures: Optional[np.ndarray] = None
    center_strength: float = 0.20
    k_neighbors: int = 18
    alpha: float = DEFAULT_ALPHA
    reg: float = DEFAULT_REG
    tau_col: Optional[float] = DEFAULT_TAU
    max_iter: int = 150
    tol: float = 1e-4
    target_prior: Optional[np.ndarray] = None
    report: Dict[str, float] = field(default_factory=dict)

    def _temperatures(self, K: int) -> np.ndarray:
        if self.class_temperatures is not None:
            t = np.asarray(self.class_temperatures, dtype=np.float64).reshape(-1)
            if t.size != K:
                raise ValueError(f"class_temperatures length {t.size} != {K}")
            return t
        return np.full(K, float(self.temperature))

    def fit(
        self,
        logits: ArrayLike,
        y: ArrayLike,
        shrink: float = 1.0,
        per_class_temperature: bool = True,
        alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        tau_cols: Sequence[Optional[float]] = (0.5, 1.0, 2.0, None),
    ) -> "UnbalancedCalibrator":
        """Grid-tunes hyperparameters directly on out-of-fold validation set."""
        z = _sanitize(_as_2d(logits, "logits"))
        n, K = z.shape
        y_int = _as_labels(y, n, K)

        self.temperature = fit_temperature(z, y_int)
        self.class_temperatures = (
            fit_class_temperatures(z, y_int, shrink=shrink, t_global=self.temperature)
            if per_class_temperature
            else None
        )
        base = temperature_scale_logits(z, self._temperatures(K))

        raw_pred = np.argmax(z, axis=1)
        raw_bacc = balanced_accuracy(y_int, raw_pred, K)
        temp_bacc = balanced_accuracy(y_int, np.argmax(base, axis=1), K)

        best_score = -np.inf
        best_params = (self.alpha, self.tau_col)

        for alpha in alphas:
            for tau in tau_cols:
                p_cand = unbalanced_sinkhorn_transport(
                    base,
                    target_col_marginal=self.target_prior,
                    reg=self.reg,
                    tau_col=tau,
                    alpha=float(alpha),
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                score = balanced_accuracy(y_int, np.argmax(p_cand, axis=1), K)
                if score > best_score + 1e-12:
                    best_score = score
                    best_params = (float(alpha), tau)

        self.alpha, self.tau_col = best_params
        fell_back = False

        # Monotonicity safety guard: never regress below raw argmax
        if best_score < raw_bacc - 1e-12:
            self.temperature = 1.0
            self.class_temperatures = None
            self.alpha = 0.0
            self.tau_col = None
            best_score = raw_bacc
            fell_back = True

        self.report = {
            "bacc_raw": float(raw_bacc),
            "bacc_after_temperature": float(temp_bacc),
            "bacc_calibrated": float(best_score),
            "temperature": float(self.temperature),
            "alpha": float(self.alpha),
            "tau_col": float(self.tau_col if self.tau_col is not None else -1.0),
            "fell_back_to_identity": float(fell_back),
        }
        return self

    def transform(
        self,
        logits: ArrayLike,
        pairwise_corrections: Optional[Dict[Tuple[Union[str, int], Union[str, int]], np.ndarray]] = None,
        hodge_weight: float = 0.15,
        target_col_marginal: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Applies calibration chain: temperature -> HodgeRank -> Unbalanced Sinkhorn."""
        z = _as_2d(logits, "logits").copy()
        n, K = z.shape

        if pairwise_corrections is not None and len(pairwise_corrections) > 0:
            V = project_hodgerank_weighted(pairwise_corrections, n_classes=K)
            z += hodge_weight * V

        p = temperature_scale_logits(z, self._temperatures(K))
        target = target_col_marginal if target_col_marginal is not None else self.target_prior

        if self.alpha <= 0.0 or n == 0:
            return p

        return unbalanced_sinkhorn_transport(
            p,
            target_col_marginal=target,
            reg=self.reg,
            tau_col=self.tau_col,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
        )


# =========================================================================== #
# 9. Invariant Verification Contract
# =========================================================================== #
class _ContractChecker:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.passed = 0

    def check(self, condition: bool, message: str) -> None:
        if not bool(condition):
            raise AssertionError(f"[unbalanced_sinkhorn contract failure] {message}")
        self.passed += 1
        if self.verbose:
            print(f"  [OK] {message}")


def test_unbalanced_invariants(verbose: bool = True) -> Dict[str, float]:
    """Executable verification suite asserting mathematical invariants and bounds."""
    c = _ContractChecker(verbose=verbose)
    rng = np.random.default_rng(42)
    K = N_CLASSES
    metrics: Dict[str, float] = {}

    # 1. Unbalanced Optimal Transport Invariants
    p = temperature_scale_logits(rng.normal(0, 1.5, size=(60, K)), 1.0)

    # Invariant: alpha = 0 is identity
    p_id = unbalanced_sinkhorn_transport(p, alpha=0.0)
    c.check(np.allclose(p_id, p, atol=1e-12), "UOT: alpha=0 is identity")

    # Invariant: strict row stochasticity
    p_uot, info = unbalanced_sinkhorn_transport(p, reg=0.1, tau_col=1.0, alpha=1.0, return_info=True)
    c.check(np.all(np.isfinite(p_uot)), "UOT: output is finite")
    c.check(np.all(p_uot >= 0.0), "UOT: output is non-negative")
    c.check(np.allclose(p_uot.sum(axis=1), 1.0, atol=1e-6), "UOT: rows sum strictly to 1")
    c.check(info["converged"] == 1.0, "UOT: converged within max_iter")

    # Invariant: tau_col -> inf matches balanced Sinkhorn
    p_inf = unbalanced_sinkhorn_transport(p, reg=0.5, tau_col=None, alpha=1.0, tol=1e-8, max_iter=300)
    c.check(np.allclose(p_inf.sum(axis=0), 60.0 / K, atol=1e-4), "UOT: tau->inf balances column mass exactly")

    # Invariant: tau_col -> 0 relaxes column constraints
    p_zero = unbalanced_sinkhorn_transport(p, reg=0.5, tau_col=1e-4, alpha=1.0)
    c.check(not np.allclose(p_zero.sum(axis=0), 60.0 / K, atol=1e-2), "UOT: tau->0 relaxes column marginal")

    # Invariant: Equivariance under class and row permutation
    perm_col = rng.permutation(K)
    p_uot_02 = unbalanced_sinkhorn_transport(p, reg=0.2, tau_col=1.0, alpha=1.0)
    p_perm = unbalanced_sinkhorn_transport(p[:, perm_col], reg=0.2, tau_col=1.0, alpha=1.0)
    c.check(np.allclose(p_uot_02[:, perm_col], p_perm, atol=1e-5), "UOT: equivariant under column permutation")

    # 2. Enhanced HodgeRank Invariants
    pairs = [(0, 1), (1, 2), (2, 0), (3, 4), (5, 6)]
    N_samples = 40
    deltas = {pr: rng.normal(0, 1.0, size=N_samples) for pr in pairs}
    # Add intentional cycle inconsistency to (0, 1) + (1, 2) + (2, 0)
    deltas[(0, 1)] += 2.0
    deltas[(1, 2)] += 2.0
    deltas[(2, 0)] += 2.0

    V, curl = project_hodgerank_weighted(deltas, confidence_weighting=True, n_classes=K, return_curl=True)
    c.check(V.shape == (N_samples, K), "HodgeRank: potential has shape (N, K)")
    c.check(np.allclose(V.mean(axis=1), 0.0, atol=1e-12), "HodgeRank: potential is centered at 0")

    # Invariant: Potential gradient is strictly curl-free (transitive) around cycle
    g_01 = V[:, 1] - V[:, 0]
    g_12 = V[:, 2] - V[:, 1]
    g_20 = V[:, 0] - V[:, 2]
    cycle_sum = g_01 + g_12 + g_20
    c.check(np.allclose(cycle_sum, 0.0, atol=1e-12), "HodgeRank: potential gradient is strictly curl-free")
    c.check(0.0 <= curl <= 1.0, "HodgeRank: curl inconsistency metric in [0, 1]")

    # 3. Dynamic Topic Clustering Invariants
    N_docs = 50
    true_cl = np.repeat(np.arange(5), 10)
    D = np.ones((N_docs, N_docs)) * 0.95
    for grp in range(5):
        idx = np.where(true_cl == grp)[0]
        D[np.ix_(idx, idx)] = rng.uniform(0.1, 0.3, size=(10, 10))
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)

    clusterer = DynamicTopicClusterer(min_thresh=0.30, max_thresh=0.90)
    groups = clusterer.fit_predict(D)
    c.check(len(groups) == N_docs, "TopicClusterer: returns group for each sample")
    sizes = np.bincount(groups)[groups]
    c.check(sizes.min() >= 2 and sizes.max() <= 18, "TopicClusterer: dynamically isolates valid cliques")

    # 4. OAS and Ledoit-Wolf Covariance Invariants
    X_cov = rng.normal(0, 1.0, size=(100, 20))
    a_oas = oas_shrinkage(X_cov)
    a_lw = ledoit_wolf_shrinkage(X_cov)
    c.check(0.0 <= a_oas <= 1.0, "OAS: shrinkage coefficient inside [0, 1]")
    c.check(0.0 <= a_lw <= 1.0, "Ledoit-Wolf: shrinkage coefficient inside [0, 1]")

    # 5. End-to-End Orchestrator Invariants & Failure-Mode Rescue
    # Simulate a severely biased model suppressing 6 classes
    N_eval = 900
    y_true = np.repeat(np.arange(K), N_eval // K)
    bias = np.zeros(K)
    bias[:6] = -2.5
    bias[6:] = 0.8
    z_hostile = rng.normal(0, 1.0, size=(N_eval, K))
    z_hostile[np.arange(N_eval), y_true] += 2.0
    z_hostile += bias[None, :]

    bacc_before = balanced_accuracy(y_true, z_hostile.argmax(axis=1), K)
    calibrator = UnbalancedCalibrator()
    calibrator.fit(z_hostile, y_true, alphas=[0.0, 0.5, 1.0], tau_cols=[0.5, 1.0, None])
    p_cal = calibrator.transform(z_hostile)
    bacc_after = balanced_accuracy(y_true, p_cal.argmax(axis=1), K)

    c.check(bacc_after >= bacc_before + 0.05, f"Orchestrator: rescue bacc {bacc_before:.3f} -> {bacc_after:.3f}")
    c.check(calibrator.report["bacc_calibrated"] >= calibrator.report["bacc_raw"], "Orchestrator: monotonic safety")
    metrics["bacc_before"] = bacc_before
    metrics["bacc_after"] = bacc_after
    metrics["passed_invariants"] = float(c.passed)

    if verbose:
        print(f"\nAll {c.passed} mathematical invariants passed successfully!")
    return metrics


if __name__ == "__main__":
    test_unbalanced_invariants(verbose=True)
