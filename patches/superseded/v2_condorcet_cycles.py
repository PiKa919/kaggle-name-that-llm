"""VULNERABILITY 2 FIX: HodgeRank projection for coherent potential vector.
Replaces unconstrained log-odds sum with least-squares projection onto gradient space.
"""
import numpy as np
from numpy.linalg import lstsq

from src.config import MODEL_TO_IDX


def build_pairwise_graph(pairs, n_classes):
    """Build Laplacian L = D - A for pairwise corrections. Nodes = 18 classes."""
    A = np.zeros((n_classes, n_classes))
    for m1, m2 in pairs:
        i1, i2 = MODEL_TO_IDX[m1], MODEL_TO_IDX[m2]
        A[i1, i2] = 1; A[i2, i1] = 1
    D = np.diag(A.sum(axis=1))
    return D - A, A.sum(axis=1)

def hodgerank_project(deltas_dict, pairs, n_classes):
    """
    Given independent pairwise delta_ij = log(P(j)/P(i)), find potential vector v
    such that v_j - v_i ≈ delta_ij. Solves L v ≈ b via least-squares with pinning.
    Pin v[0] = 0 to fix global scale.
    """
    # Build b from deltas
    L, d = build_pairwise_graph(pairs, n_classes)
    # Add pinning row: v_0 = 0
    L_pinned = np.vstack([np.hstack([L, np.zeros((n_classes,1))])])  # not needed; simpler:
    # Direct approach: solve L v = 0 + delta contributions
    # For simplicity, accumulate b as gradient of observed deltas
    b = np.zeros(n_classes)
    for (m1, m2), delta in deltas_dict.items():
        i1, i2 = MODEL_TO_IDX[m1], MODEL_TO_IDX[m2]
        b[i2] += delta
        b[i1] -= delta
    # Pinned least-squares: v = (L^T L)^{-1} L^T b with v[0]=0
    # Remove first row/col, solve reduced system
    L_red = L[1:, 1:]
    b_red = b[1:] - L[1:, 0] * 0.0  # v0 pinned at 0
    v_red, _, _, _ = lstsq(L_red, b_red, rcond=None)
    v = np.concatenate([[0.0], v_red])
    return v

# Drop-in for submission_sota.py pairwise section:
# Replace direct log_base[:, idx2] += 0.10*delta with:
#   potential = hodgerank_project(pairwise_corrections, SIBLING_PAIRS, 18)
#   log_base += 0.10 * potential[None, :]
# This guarantees no Condorcet cycles: v_j - v_i is globally consistent.
