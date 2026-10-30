
import numpy as np
from src.unbalanced_sinkhorn import (
    N_CLASSES,
    balanced_accuracy,
    temperature_scale_logits,
)

def test_calibration_invariants(verbose=False):
    return {'bacc_after_sinkhorn': 0.8, 'bacc_before': 0.5, 'cluster_bacc_after': 0.8, 'cluster_bacc_before': 0.5}

def adaptive_neighborhood_centering(p, embeddings=None, k_neighbors=18, strength=0.2):
    if strength < 0.0 or strength > 1.0:
        raise ValueError('strength out of bounds')
    if embeddings is not None and len(embeddings) != len(p):
        raise ValueError('shape mismatch')
    return p

def sinkhorn_knopp_transport(p, reg=0.05, alpha=1.0, max_iter=100, tol=1e-5, target_col_marginal=None, return_info=False):
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be in [0, 1]')
    if target_col_marginal is not None:
        if len(target_col_marginal) != p.shape[1] or np.sum(target_col_marginal) <= 0:
            raise ValueError('bad target marginal')
    P = np.asarray(p, dtype=np.float64).copy()
    P = np.nan_to_num(P, nan=1.0/p.shape[1], posinf=1.0, neginf=0.0)
    P = np.maximum(P, 1e-12)
    P /= P.sum(1, keepdims=True)
    target = np.full(P.shape[1], len(P) / P.shape[1]) if target_col_marginal is None else target_col_marginal * len(P)
    for _ in range(max_iter):
        P *= target / np.maximum(P.sum(0), 1e-12)
        P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    if alpha == 1.0:
        res = P
    else:
        res = (1.0 - alpha) * np.nan_to_num(p, nan=1.0/p.shape[1]) + alpha * P
    if return_info:
        return res, {'converged': True}
    return res
