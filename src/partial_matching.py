"""
Exact transductive posterior for "one model per prompt" test clusters.

Why this file exists
--------------------
`src/postproc.py` approximates the within-prompt constraint with Sinkhorn on a
padded K x K matrix and then renormalises the real rows. Two defects:

  1. `dummy_logit` is a *uniform* constant added to every dummy row. After the
     column normalisation in Sinkhorn a column-constant shift cancels exactly,
     so the knob that is documented as "the log-odds that a model is absent"
     provably does nothing. (See test_dummy_logit_is_a_no_op.)
  2. Row renormalisation after marginalising the dummies destroys the column
     capacity constraint, so the output is a mild reweighting, not a matching.

This module computes the posterior of the correct generative model exactly.

Model
-----
A test prompt cluster is `m` passages answering the same prompt. Each passage
has exactly one author; each author answers a prompt at most once. So the
latent variable is an INJECTION A: {1..m} -> {1..K}, K = 18.

    P(A) ∝ prod_i W[i, A(i)] * prod_{k in A} r_k

with W[i, k] the calibrated likelihood of passage i under author k and
r_k = p_k / (1 - p_k) the presence odds of author k in a cluster. Authors that
are absent contribute prod_{k not in A} (1 - p_k), a constant given the model
parameters, so it is dropped.

Z = sum over injections of prod_i W[i, A(i)] * r_{A(i)} is a rectangular
permanent. It is computed by a subset DP over author masks, O(m * K * 2^K),
and the marginals P(i -> k) = d log Z / d log W[i,k] are obtained by reverse
mode differentiation through the same DP. No Sinkhorn iteration, no beta
temperature, no renormalisation hack.

Why the marginals are the thing you want
----------------------------------------
Balanced accuracy is the mean of 18 recalls. A wrong cluster cannot be allowed
to force a permutation, but a correct cluster must be allowed to veto a
duplicate. Exact marginals give the per-passage posterior under the constraint
*without* committing to a single global assignment, which is the only way to
keep rare-class recall alive when the cluster is ambiguous.

Complexity
----------
2^18 = 262,144 masks. One cluster costs ~0.1-0.4 s on CPU for m = 18 and is
much cheaper for the m = 3-10 clusters the hidden test set actually contains.
~220 clusters is therefore seconds to a couple of minutes, not hours.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.special import logsumexp

__all__ = [
    "partial_matching_posteriors",
    "fit_presence",
    "sinkhorn_to_uniform",
    "cluster_posterior_em",
    "balanced_log_odds",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Exact rectangular permanent and its marginals
# --------------------------------------------------------------------------- #
def partial_matching_posteriors(
    logw: np.ndarray,
    presence_logit: Optional[np.ndarray] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, np.ndarray]:
    """Exact marginals of the injection model.

    Parameters
    ----------
    logw : (m, K) array
        Log weights. Row i is passage i, column k is author k. Any monotone
        transform of the per-passage posterior is fine; only differences
        within a row matter for the assignment.
    presence_logit : (K,) array, optional
        Log presence odds r_k. Defaults to zeros (each author equally likely
        to appear in the cluster).

    Returns
    -------
    dict with
        marginals      : (m, K) P(passage i assigned to author k)
        usage          : (K,)  expected number of passages assigned to k
        logZ           : float log partition function
    """
    logw = np.asarray(logw, dtype=np.float64)
    m, K = logw.shape
    if m > K:
        raise ValueError(f"m={m} > K={K}: not an injection, refuse to lie")
    if presence_logit is None:
        presence_logit = np.zeros(K)
    presence_logit = np.asarray(presence_logit, dtype=np.float64)

    # Fold presence odds into the column weights. `lw` is a log weight, so the
    # DP multiplies by exp(lw) and d log Z / d lw is exactly the marginal.
    lw = torch.tensor(logw + presence_logit[None, :], dtype=dtype, requires_grad=True)
    W = torch.exp(lw)

    n_masks = 1 << K
    masks = np.arange(n_masks, dtype=np.int64)
    popcount = np.zeros(n_masks, dtype=np.int64)
    for k in range(K):
        popcount += (masks >> k) & 1
    masks_by_pc = [np.where(popcount == i)[0] for i in range(m + 1)]
    masks_by_pc_t = [torch.as_tensor(v) for v in masks_by_pc]

    # f[S] = weight of assigning rows 0..i-1 to the column set S.
    f = torch.zeros(n_masks, dtype=dtype)
    f[0] = 1.0
    for i in range(m):
        f = f.clone()
        src_all = masks_by_pc[i]
        for k in range(K):
            bit = 1 << k
            src = src_all[((src_all & bit) == 0)]
            if src.size == 0:
                continue
            tgt = src | bit
            contrib = f[torch.as_tensor(src)] * W[i, k]
            f = f.index_add(0, torch.as_tensor(tgt), contrib)

    # f holds linear weights, so Z is the sum over maximal-popcount masks.
    logZ = torch.log(f[masks_by_pc_t[m]].sum())
    logZ.backward()

    marginals = lw.grad.detach().numpy().copy()
    # Numerical guard: a true marginal lies in [0, 1].
    marginals = np.clip(marginals, 0.0, 1.0)
    usage = marginals.sum(axis=0)
    return {
        "marginals": marginals,
        "usage": usage,
        "logZ": float(logZ.detach()),
    }


# --------------------------------------------------------------------------- #
# Presence-odds EM by gradient ascent on the cluster likelihood
# --------------------------------------------------------------------------- #
def fit_presence(
    clusters: Sequence[np.ndarray],
    presence_logit: Optional[np.ndarray] = None,
    n_iter: int = 25,
    lr: float = 0.35,
    prior_strength: float = 2.0,
    prior_mean: float = 0.75,
    verbose: bool = False,
) -> np.ndarray:
    """Fit shared author-presence odds by maximising sum_c log Z_c.

    `clusters` is a sequence of (m_c, K) log-weight matrices. Because presence
    enters log Z additively per column, d log Z_c / d presence_logit[k] is just
    the expected usage of author k in cluster c, which
    `partial_matching_posteriors` already returns.

    A Beta-style prior pulls each presence probability toward `prior_mean`:
    the hidden test set is known to be subsampled, so p_k is well below 1 and
    an unregularised M-step drives it to the boundary on small clusters.
    """
    K = clusters[0].shape[1]
    if presence_logit is None:
        presence_logit = np.zeros(K)

    beta = np.asarray(presence_logit, dtype=np.float64).copy()
    for it in range(n_iter):
        grad = np.zeros(K)
        total = 0.0
        for lw in clusters:
            out = partial_matching_posteriors(lw, beta)
            grad += out["usage"]
            total += out["logZ"]
        # Prior gradient in log-odds space for a Beta(a, b) centered at
        # prior_mean with strength `prior_strength`.
        p = 1.0 / (1.0 + np.exp(-beta))
        a = prior_strength * prior_mean
        b = prior_strength * (1.0 - prior_mean)
        grad += (a - 1.0) * (1.0 - p) - (b - 1.0) * p
        beta += lr * grad / max(len(clusters), 1)
        beta = np.clip(beta, -6.0, 6.0)
        if verbose:
            print(f"  presence iter {it:02d} mean logZ/cluster {total/len(clusters):+.4f} "
                  f"p_mean={p.mean():.3f}")
    return beta


# --------------------------------------------------------------------------- #
# Margin correction: Sinkhorn-to-uniform is the balanced-accuracy-optimal rule
# --------------------------------------------------------------------------- #
def balanced_log_odds(logp: np.ndarray, n_iter: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Scale columns so predicted marginals are uniform.

    Balanced accuracy weights every class equally, so the decision problem is
    equivalent to a uniform class prior. The plug-in rule is therefore
    argmax_k p(x|y=k) = argmax_k p(y=k|x) / pi_test(k). Sinkhorn-to-uniform
    estimates pi_test implicitly (the column scale factors *are* the estimated
    marginals) and divides it out.

    This is the OPPOSITE of the SLD-EM prior shift in src/postproc.py, which
    multiplies by pi_test / pi_train and therefore pushes mass toward whatever
    is already over-predicted -- the exact direction that damages balanced
    accuracy when the test mixture is skewed.
    """
    lp = np.log(np.clip(np.asarray(logp, dtype=np.float64), _EPS, 1.0))
    N, K = lp.shape
    out = lp.copy()
    target = np.log(N / K)
    for _ in range(n_iter):
        # column marginal -> N/K, then row marginal -> 1, alternating.
        out -= logsumexp(out, axis=0, keepdims=True) - target
        out -= logsumexp(out, axis=1, keepdims=True)
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def cluster_posterior_em(
    logp: np.ndarray,
    cluster_ids: np.ndarray,
    fit_presence_odds: bool = True,
    min_cluster_size: int = 2,
    return_presence: bool = False,
):
    """Apply exact within-cluster matching, then global marginal balancing.

    Parameters
    ----------
    logp : (N, K) log posteriors from the upstream classifier, ALREADY
        temperature-calibrated. Scale within a row matters; the row offset
        does not.
    cluster_ids : (N,) recovered prompt cluster id. Singletons and clusters
        smaller than `min_cluster_size` are left untouched (a singleton
        cluster carries no constraint).

    Returns
    -------
    (N, K) balanced log posteriors, or (logpost, presence_logit) if
    `return_presence`.
    """
    logp = np.asarray(logp, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    N, K = logp.shape

    groups = [np.where(cluster_ids == c)[0] for c in np.unique(cluster_ids)]
    groups = [g for g in groups if len(g) >= min_cluster_size and len(g) <= K]

    presence = np.zeros(K)
    if fit_presence_odds and groups:
        presence = fit_presence([logp[g] for g in groups])

    out = logp.copy()
    for g in groups:
        res = partial_matching_posteriors(logp[g], presence)
        out[g] = np.log(np.clip(res["marginals"], _EPS, 1.0))

    out = balanced_log_odds(np.exp(out))
    if return_presence:
        return out, presence
    return out
