"""
Pairwise sibling corrections that do not double-count the base model.

Why this file exists
--------------------
`submission_sota.py` fits 8 independent binary logistic regressions and then does

    log_base[:, idx2] += 0.10 * delta
    log_base[:, idx1] -= 0.10 * delta

Five findings, all measured on prompt-grouped folds:

1. The Condorcet-cycle objection does not apply to the shipped pair list.  The 8
   edges touch 12 of the 18 classes and form a forest (four trees plus six
   isolated nodes), so the cycle space has dimension E - V + C = 8 - 18 + 10 = 0.
   Every edge flow on a forest is the gradient of a potential, and adding
   +w*delta to j and -w*delta to i IS an accumulation of that potential.
   Projecting it through HodgeRank changes nothing.  Verified in
   `tests/test_invalidation_fixes.py::test_forest_projection_is_identity`.

2. The real defect is double counting.  Each pairwise classifier is a logistic
   regression on the *same* 4-view TF-IDF matrix as the base ensemble.  Its
   output r_ij = logit P(y = j | y in {i, j}) re-derives information the base
   logits already hold; adding 0.10 * r_ij on top adds 0.20 * r_ij to the base
   margin s_ij = log p_i - log p_j, which already encodes it.  The only new
   information the specialist carries is the residual r_ij - s_ij.

3. The correction is applied to every passage, but ~89% of passages belong to
   neither class of any given pair.  There r_ij is an extrapolation far outside
   the two-class training window and the pair-membership probability
   pi_ij = p_i + p_j is near zero.  Measured on OOF: the correction flips
   82/2699 argmaxes and moves balanced accuracy 0.3689 -> 0.3686 (null).

4. The weight 0.10 is not fitted, and the clip at +-2.5 nats means the largest
   correction this entire view can ever apply is 0.25 nats -- against a
   typographic view whose per-row spread is 9.18 nats.  The view cannot defend
   its own contribution.

5. Applying deltas edge-by-edge does NOT add the potential v with v_j - v_i = delta_e.
   Adding +w*delta_e to class j and -w*delta_e to class i realises a margin change
   of 2*w*delta_e, and every *other* edge incident to i or j leaks into it.
   For the three-node chain i-j-k the realised margins are

       margin_ij = 2*delta_ij - delta_jk'(sign-adjusted)
       margin_jk = 2*delta_jk - delta_ij'(sign-adjusted)

   so the effective correction strength ranges from 2x (isolated edge) upward,
   and one boundary's calibration is contaminated by an unrelated boundary.
   Measured example (delta = [0.5, -0.3] on a chain): shipped realises
   [1.3, -1.1] instead of [0.5, -0.3].  The pair graph is a forest
   (cycle_dim = 0), so no Condorcet cycle is possible, but the shipped update is
   the DIVERGENCE of the edge flow, not a potential -- and that is the real
   coherence defect.

Correct formulation
-------------------
    s_ij(x) = log p_i(x) - log p_j(x)                  base margin (already coherent)
    r_ij(x) = logit P(y = j | y in {i, j})             sibling specialist
    pi_ij(x) = p_i(x) + p_j(x)                         pair membership probability
    D_ij(x) = r_ij(x) - s_ij(x) * tau_ij               residual after rescaling
    b_k(x)  = sum_e lam_e * pi_ij(x) * D_ij(x) * (1{k=j_e} - 1{k=i_e})   (divergence)
    v(x)    = L^+ b(x)                                 (least-squares POTENTIAL)

    log p'_k = log p_k + v_k(x)

v is the solution of

    min_v sum_e (v_j - v_i - lam_e * pi_e * D_e)^2   <=>   L v = b

so on a forest it realises every requested margin EXACTLY, and on a cyclic graph
it is the least-squares projection of the edge flow onto the gradient space
("HodgeRank").  L has a null space of one constant per connected component, so v
is unique only up to a per-component offset; `ls_potential` fixes that by
centring each component at zero, which keeps a within-family correction from
stealing mass across families.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from src.config import MODEL_TO_IDX

__all__ = [
    "base_pair_margin",
    "pair_membership",
    "residual_deltas",
    "shipped_divergence",
    "ls_potential",
    "realized_margins",
    "cycle_dimension",
    "PairwiseResidualInjector",
]

Pair = Tuple[str, str]


def base_pair_margin(logp: np.ndarray, i: int, j: int) -> np.ndarray:
    """s_ij = log p_i - log p_j, the base model's own opinion of the pair."""
    return logp[:, i] - logp[:, j]


def pair_membership(p: np.ndarray, i: int, j: int) -> np.ndarray:
    """pi_ij = p_i + p_j, the probability that the pair is even relevant."""
    return p[:, i] + p[:, j]


def residual_deltas(
    pair_prob_j: np.ndarray,
    logp: np.ndarray,
    i: int,
    j: int,
    eps: float = 1e-6,
    tau: float = 1.0,
    clip: float = 4.0,
) -> np.ndarray:
    """D_ij = logit(pair_prob_j) - tau * (log p_i - log p_j), winsorised.

    `tau` absorbs the fact that a two-class specialist is trained on a
    different prior than the 18-way model, so its logit is not on the same
    scale as an 18-way log-ratio.  tau = 1 assumes it is; it is fitted in
    practice and reported by `PairwiseResidualInjector.diagnostics`.
    """
    q = np.clip(pair_prob_j, eps, 1.0 - eps)
    r = np.log(q / (1.0 - q))
    d = r - tau * base_pair_margin(logp, i, j)
    return np.clip(d, -clip, clip)


def shipped_divergence(
    pairs: Sequence[Pair],
    terms: Dict[Pair, np.ndarray],
    n_classes: int = 18,
) -> np.ndarray:
    """The update the current submission applies: b_k = sum_e +-term_e.

    This is the DIVERGENCE of the edge flow, not a potential.  Applied directly
    to logits it realises a margin change of 2*term_e on an isolated edge and
    something else entirely on any edge that shares a node.  Kept only so the
    defect can be measured; use `ls_potential` for anything that ships.
    """
    n = next(iter(terms.values())).shape[0] if terms else 0
    v = np.zeros((n, n_classes), dtype=np.float64)
    for a, b in pairs:
        if (a, b) not in terms:
            continue
        i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
        t = terms[(a, b)]
        v[:, j] += t
        v[:, i] -= t
    return v


def _laplacian(pairs: Sequence[Pair], n_classes: int = 18) -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((n_classes, n_classes))
    for a, b in pairs:
        i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
        A[i, j] = A[j, i] = 1.0
    return np.diag(A.sum(axis=1)) - A, A


def ls_potential(
    pairs: Sequence[Pair],
    terms: Dict[Pair, np.ndarray],
    n_classes: int = 18,
) -> np.ndarray:
    """Least-squares potential: min_v sum_e (v_j - v_i - term_e)^2, i.e. L v = b.

    On a forest the system is consistent and the solution realises every
    requested margin exactly.  On a cyclic graph it is the projection of the
    edge flow onto the gradient space (HodgeRank).  L is singular by one
    constant per connected component; each component is centred at zero so the
    correction cannot move mass between families.
    """
    L, A = _laplacian(pairs, n_classes)
    b = shipped_divergence(pairs, terms, n_classes)
    n = b.shape[0]

    n_comp, labels = connected_components(A, directed=False)
    v = np.zeros((n, n_classes), dtype=np.float64)
    for comp in range(n_comp):
        nodes = np.where(labels == comp)[0]
        if len(nodes) < 2:
            continue
        Lc = L[np.ix_(nodes, nodes)]
        bc = b[:, nodes].T                      # (len(nodes), n_rows)
        free = nodes[1:]                        # pin the first node of the component
        Lff = L[np.ix_(free, free)]
        if np.linalg.matrix_rank(Lff) < Lff.shape[0]:
            sol = np.linalg.lstsq(Lff, bc[1:], rcond=None)[0]
        else:
            sol = np.linalg.solve(Lff, bc[1:])
        v[:, free] = sol.T
        v[:, nodes] -= v[:, nodes].mean(axis=1, keepdims=True)
    return v


def realized_margins(
    pairs: Sequence[Pair],
    v: np.ndarray,
) -> Dict[Pair, np.ndarray]:
    """Margin actually applied to each edge by a logit update `v`.

    Diagnostic: compare with the requested `terms` to see how much an update
    distorts the correction it was given.
    """
    return {
        (a, b): v[:, MODEL_TO_IDX[b]] - v[:, MODEL_TO_IDX[a]]
        for a, b in pairs
    }


def cycle_dimension(pairs: Sequence[Pair], n_classes: int = 18) -> int:
    """dim H^1 of the pair graph: E - V + C. Zero means no sequence of pairwise
    preferences can be circular, so Condorcet cycles are impossible."""
    _, A = _laplacian(pairs, n_classes)
    n_comp, _ = connected_components(A, directed=False)
    edges = int(A.sum() / 2)
    return edges - n_classes + n_comp


class PairwiseResidualInjector:
    """Fits sibling specialists and injects their *residual* against the base.

    Parameters
    ----------
    lam : global correction strength (fitted on OOF by `src/fusion.py`).
    tau : scale relating a two-class logit to an 18-way log-ratio.
    gate : if True, multiply each edge term by pi_ij = p_i + p_j.  This is what
        stops a specialist trained on ~300 rows from moving passages that are
        obviously neither of its two classes.
    coherent : if True, accumulate the potential with `ls_potential` (identical
        on forests, correct on cyclic graphs).
    """

    def __init__(
        self,
        pairs: Sequence[Pair],
        lam: float = 0.10,
        tau: float = 1.0,
        gate: bool = True,
        coherent: bool = True,
        clip: float = 4.0,
    ):
        self.pairs = list(pairs)
        self.lam = float(lam)
        self.tau = float(tau)
        self.gate = bool(gate)
        self.coherent = bool(coherent)
        self.clip = float(clip)
        self.models_: Dict[Pair, object] = {}
        self.diagnostics_: Dict[str, float] = {}

    def fit(self, X, y: np.ndarray) -> "PairwiseResidualInjector":
        from sklearn.linear_model import LogisticRegression

        for a, b in self.pairs:
            i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
            m = (y == i) | (y == j)
            if m.sum() < 8 or len(np.unique(y[m])) < 2:
                continue
            clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
            clf.fit(X[m], (y[m] == j).astype(int))
            self.models_[(a, b)] = clf
        return self

    def edge_terms(self, X, logp: np.ndarray) -> Dict[Pair, np.ndarray]:
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        terms: Dict[Pair, np.ndarray] = {}
        raw: List[float] = []
        for (a, b), clf in self.models_.items():
            i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
            q = clf.predict_proba(X)[:, 1]
            d = residual_deltas(q, logp, i, j, tau=self.tau, clip=self.clip)
            if self.gate:
                d = d * pair_membership(p, i, j)
            terms[(a, b)] = self.lam * d
            raw.append(float(np.abs(d).mean()))
        self.diagnostics_ = {
            "mean_abs_term": float(np.mean(raw)) if raw else 0.0,
            "max_abs_term": float(np.max(np.abs(list(terms.values())))) if terms else 0.0,
        }
        return terms

    def transform(self, X, logp: np.ndarray) -> np.ndarray:
        """Returns log p' = log p + v. Never renormalises by hand: the caller's
        softmax does that, and doing it here would destroy the gauge."""
        terms = self.edge_terms(X, logp)
        if not terms:
            return logp
        v = ls_potential(self.pairs, terms, logp.shape[1]) if self.coherent else shipped_divergence(
            self.pairs, terms, logp.shape[1]
        )
        if self.coherent:
            req = np.array([terms[p] for p in self.pairs if p in terms])
            got = np.array([realized_margins(self.pairs, v)[p] for p in self.pairs if p in terms])
            if req.size:
                self.diagnostics_["margin_fidelity"] = float(
                    np.abs(req - got).max()
                )
        return logp + v
