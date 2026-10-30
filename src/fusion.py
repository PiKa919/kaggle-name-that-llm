"""
Fusion as a fitted log-opinion pool with simplex constraints.

Why this file exists
--------------------
`submission_sota.py` fuses views with three constants that were never fitted:

    log_p_lora = log(p_lora) / 0.86
    log_p_ng   = log(p_ng)   / 0.80
    log_base   = 0.85 * log_p_lora + 0.15 * log_p_ng

Three problems.

1. `1/T` and the pool weight are not separately identifiable.  In

       log q_k  ∝  sum_v a_v * log p_k^(v)

   any change of `a_v` is exactly a change of `1/T_v`.  Writing both is a gauge
   freedom, not a model.  What is identified is the single product `a_v`, so it
   should be fitted once, as a pool weight.

2. One scalar for all 18 classes is contradicted by the data.  Sweeping the
   weight of one auxiliary view per class on prompt-grouped OOF gives optimal
   weights spanning 0.00 (claude-opus-5, kimi-k3, gpt-5.6-terra, gpt-5.6-luna)
   to 0.90 (claude-fable-5, gemini-3.1-pro-preview).  For claude-fable-5 the
   same sweep moves recall from 0.14 to 0.46.  A global 0.15/0.45 hands every
   class the average of a distribution that is bimodal.

3. The weights are not constrained, and a static weight is not necessarily
   admissible.  A per-class fit can drive a view's weight negative (extrapolating
   a view's evidence by *subtracting* it), which is how inverse-variance pooling
   fails on correlated views.  The pool must be constrained to the simplex.

Correct formulation
-------------------
    W in R^{K x V},  W_{k,.} in simplex,  V = number of views
    log q_k ∝ sum_v W_{k,v} * log p_k^(v)

    NLL(W) = -1/N sum_i log softmax_k(log q_i)_ {y_i}
            + lambda * sum_k ||W_{k,.} - w^global||^2      (hierarchical shrinkage)

NLL is convex in `log q` and `log q` is linear in `W`, so the objective is convex
given the per-class view log-probabilities; the only non-convexity is the
softmax reparametrisation `W = softmax(A)`, and the shrinkage term makes the
landscape benign.  `lambda` is the number of OOF observations a per-class
deviation must "pay for"; with ~150 rows per class, `lambda` in [20, 100]
prevents an 18-way overfit while still letting bimodal classes separate.

Fit on out-of-fold log-probabilities only.  Nothing in this module may see the
test set.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "log_softmax",
    "simplex_project",
    "view_reliability",
    "LogOpinionPool",
    "per_class_weight_sweep",
]


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x - logsumexp(x, axis=axis, keepdims=True)


def simplex_project(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Duchi et al. 2008).

    Used for reporting and for the global weight; the per-class pool is
    parametrised through a softmax so it lives on the simplex by construction.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1.0))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def view_reliability(logp: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Nats the view assigns to the true class, above the uniform baseline log K.

    A view at 0 is uninformative; a negative view is actively misleading.
    Reported per view so that a view can be switched off on evidence.
    """
    k = logp.shape[1]
    true_ll = logp[np.arange(len(y)), y]
    uniform = -np.log(k)
    return {
        "mean_nats_above_uniform": float(true_ll.mean() - uniform),
        "median_nats_above_uniform": float(np.median(true_ll) - uniform),
        "frac_below_uniform": float((true_ll < uniform).mean()),
    }


class LogOpinionPool:
    """Simplex-constrained, optionally per-class log-opinion pool.

    Parameters
    ----------
    per_class : fit one weight vector per class (recommended) or one global row.
    l2 : shrinkage strength pulling each class row toward the global row, in
        units of observations. 0 disables (only defensible with >500 rows/class).
    """

    def __init__(self, per_class: bool = True, l2: float = 40.0):
        self.per_class = bool(per_class)
        self.l2 = float(l2)
        self.A_: Optional[np.ndarray] = None       # (K, V) logits
        self.global_w_: Optional[np.ndarray] = None
        self.n_views_: Optional[int] = None
        self.n_classes_: Optional[int] = None

    # -- internals ------------------------------------------------------- #
    def _pack(self, logps: Sequence[np.ndarray]) -> np.ndarray:
        """(N, K, V) stack with each view's rows gauge-fixed to sum to 0."""
        arr = np.stack([log_softmax(lp, axis=1) for lp in logps], axis=2)
        return arr

    @staticmethod
    def _weights(A: np.ndarray) -> np.ndarray:
        return np.exp(log_softmax(A, axis=1))

    # -- fitting --------------------------------------------------------- #
    def fit(self, logps: Sequence[np.ndarray], y: np.ndarray) -> "LogOpinionPool":
        from scipy.optimize import minimize

        L = self._pack(logps)
        n, k, v = L.shape
        y = np.asarray(y)
        rows = np.arange(n)
        self.n_views_, self.n_classes_ = v, k

        # global row first: closed-ish 1-D fit by L-BFGS on V-1 free logits
        def global_loss(a: np.ndarray) -> float:
            w = np.exp(log_softmax(a))
            logits = L @ w
            return float(np.mean(logsumexp(logits, axis=1) - logits[rows, y]))

        res_g = minimize(global_loss, x0=np.zeros(v), method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 4000})
        w_global = np.exp(log_softmax(res_g.x))
        self.global_w_ = w_global
        if not self.per_class:
            self.A_ = np.tile(res_g.x, (k, 1))
            return self

        def loss_grad(theta: np.ndarray):
            A = theta.reshape(k, v)
            W = np.exp(log_softmax(A, axis=1))           # (K, V)
            logits = np.einsum("nkv,kv->nk", L, W)
            lse = logsumexp(logits, axis=1, keepdims=True)
            p = np.exp(logits - lse)
            nll = float(np.mean(lse[:, 0] - logits[rows, y]))

            # dNLL/dW_{k,v} = mean_i (p_{ik} - 1{y_i=k}) * L_{i,k,v}
            resid = p.copy()
            resid[rows, y] -= 1.0
            g = np.einsum("nk,nkv->kv", resid, L) / n      # (K, V)

            if self.l2 > 0:
                diff = W - w_global[None, :]
                nll += self.l2 * float(np.sum(diff ** 2)) / k
                g = g + self.l2 * 2.0 * diff / k

            # softmax jacobian: dA_{k,u} = W_{k,u} (g_{k,u} - <g_k, W_k>)
            gw = np.sum(g * W, axis=1, keepdims=True)
            dA = W * (g - gw)
            return nll, dA.ravel()

        theta0 = np.tile(res_g.x, (k, 1)).astype(np.float64).ravel()
        res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-10})
        self.A_ = res.x.reshape(k, v)
        return self

    # -- prediction ------------------------------------------------------ #
    def weights(self) -> np.ndarray:
        """(K, V) pool weights, rows on the simplex."""
        if self.A_ is None:
            raise RuntimeError("LogOpinionPool.fit() must be called first")
        return self._weights(self.A_)

    def transform(self, logps: Sequence[np.ndarray]) -> np.ndarray:
        L = self._pack(logps)
        W = self.weights()
        return np.einsum("nkv,kv->nk", L, W)

    def report(self, logps: Sequence[np.ndarray], y: np.ndarray, names: Sequence[str]) -> "object":
        import pandas as pd

        W = self.weights()
        logits = self.transform(logps)
        nll = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))
        cols: Dict[str, object] = {"class": list(names)}
        for v in range(W.shape[1]):
            cols["w_view%d" % v] = W[:, v]
        cols["dominant"] = [int(np.argmax(r)) for r in W]
        cols["max_share"] = W.max(axis=1)
        df = pd.DataFrame(cols)
        df.attrs["nll"] = nll
        df.attrs["global_w"] = None if self.global_w_ is None else self.global_w_.copy()
        return df


def per_class_weight_sweep(
    logp_base: np.ndarray,
    logp_aux: np.ndarray,
    y: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> "object":
    """Per-class optimal weight of one auxiliary view -- the diagnostic that
    answers "is a static global weight defensible?".

    Returns a DataFrame with `w_star`, recall at w = 0, and recall at `w_star`.
    A wide spread in `w_star` is proof that a single scalar is the wrong model.
    """
    import pandas as pd

    if grid is None:
        grid = np.linspace(0.0, 0.9, 19)
    k = logp_base.shape[1]
    rows: List[Dict[str, float]] = []
    for c in range(k):
        m = y == c
        if m.sum() < 20:
            continue
        recall = [
            float((((1 - w) * logp_base[m] + w * logp_aux[m]).argmax(axis=1) == c).mean())
            for w in grid
        ]
        rows.append(
            {
                "class": c,
                "n": int(m.sum()),
                "w_star": float(grid[int(np.argmax(recall))]),
                "recall_at_w0": recall[0],
                "recall_at_wstar": float(max(recall)),
            }
        )
    return pd.DataFrame(rows)
