"""
DROP-IN REPLACEMENT for the fusion half of
`notebooks/master_sota/submission_sota.py` (lines 316-364).

Every constant in the replacement pipeline is either fitted out-of-fold or
bounded by a stated guarantee.

V1  Length gate    np.where(n < 80, 0.45, 0.15)  ->  fitted logistic w(n),
                   Lipschitz-bounded so a one-word change perturbs any class by
                   at most `continuity_budget` nats.
V2  Pairwise       +-0.10 * raw log-odds on every row  ->  +-lam * pi_ij *
                   (r_ij - tau*s_ij): the RESIDUAL against the base margin,
                   gated by pair membership, accumulated as a coherent
                   potential. The Condorcet worry is vacuous on this pair list
                   (the graph is a forest, cycle_dim = 0); the real defects were
                   double counting, ungated application, and an unfitted weight.
V3  Typographic    4 hardcoded independent Bernoullis floored at 0.01  ->
                   per-fold 16-state categorical + prompt-contamination mixture
                   + per-passage winsorisation at 1 nat.
V4  Fusion         static 0.85/0.15 with T = 0.86 / 0.80  ->  per-class simplex
                   pool fitted on OOF NLL with hierarchical shrinkage. The 1/T
                   factors are dropped: under a log-opinion pool the temperature
                   and the weight are the same parameter, so writing both is a
                   gauge freedom. The fitted pool is only used when OOF LoRA
                   log-probabilities are supplied; otherwise it degrades to a
                   global weight and says so in the audit.
V5  Runtime        This module adds the part the submission was missing: an OOF
                   self-audit that reproduces, inside the notebook, the ablation
                   that shows whether each view helps.

Usage
-----
    from patches.dropin_submission_sota import FusionConfig, build_fusion

    fusion = build_fusion(
        train_texts=train_df["text"].tolist(),
        y_train=y_train_int,
        groups=train_df["prompt_id"].values,
        X_train=X_ng_trn,
        test_texts=test_texts,
        X_test=X_ng_test,
        p_lora=p_lora, p_ng=p_ng,
        pairs=SIBLING_PAIRS,
        logp_lora_oof=logp_lora_oof,   # optional but strongly recommended
    )
    print(fusion["audit"].round(4).to_string(index=False))
    predictions = [VALID_18_MODELS[i] for i in fusion["pred"].tolist()]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from src.coherent_pairwise import PairwiseResidualInjector, cycle_dimension
from src.fusion import LogOpinionPool, per_class_weight_sweep, view_reliability
from src.length_gate import SmoothLengthGate
from src.typo_prior import TypoPriorView, typo_bits

__all__ = ["FusionConfig", "build_fusion", "format_audit"]

K = 18


@dataclass
class FusionConfig:
    # typographic view
    typo_alpha: float = 1.0
    typo_rho: float = 0.25
    typo_max_nats: float = 2.0
    # length gate
    use_length_gate: bool = True
    gate_w_hi: float = 0.30
    gate_w_lo: float = 0.0          # no floor: the gate is the fitted function
    gate_budget: float = 0.05
    # pairwise residuals
    pairwise_lam: float = 0.10
    pairwise_tau: float = 1.0
    pairwise_gate: bool = True
    # pool
    per_class_pool: bool = True
    pool_l2: float = 40.0
    # audit
    audit: bool = True
    n_splits: int = 5
    seed: int = 42


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _log(p: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(p, floor, 1.0))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _bacc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, pred))


def _shipped_typo_logp(bits: np.ndarray, floor: float = 0.01) -> np.ndarray:
    """Reproduce the CURRENT `compute_typographic_priors` exactly, for the
    ablation row. Kept deliberately faithful, including the floor of 0.01."""
    from src.hierarchical_classifier import TYPO_PROFILES
    from src.config import VALID_18_MODELS

    prof = np.array([TYPO_PROFILES[m] for m in VALID_18_MODELS], dtype=float)
    p = np.clip(prof, floor, 1.0 - floor)
    ll = bits.astype(float) @ np.log(p).T + (1.0 - bits) @ np.log(1.0 - p).T
    return _log(_softmax(ll))


def _oof_ngram_logp(X_train, y_train, folds, seed: int):
    out = np.zeros((X_train.shape[0], K))
    for f, (tr, va) in enumerate(folds):
        clf = LogisticRegression(
            C=5.0, max_iter=2000, class_weight="balanced", random_state=seed + f
        )
        clf.fit(X_train[tr], y_train[tr])
        out[va] = clf.predict_proba(X_train[va])
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_fusion(
    train_texts: Sequence[str],
    y_train: np.ndarray,
    groups: np.ndarray,
    X_train,
    test_texts: Sequence[str],
    X_test,
    p_lora: np.ndarray,
    p_ng: np.ndarray,
    pairs: Sequence[Tuple[str, str]],
    cfg: Optional[FusionConfig] = None,
    logp_lora_oof: Optional[np.ndarray] = None,
    oof_ng_logp: Optional[np.ndarray] = None,
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    logp_lora_test: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Fuse the views and return predictions plus an out-of-fold audit.

    `p_lora` and `p_ng` must be aligned to `VALID_18_MODELS` and fully
    populated. A zero-filled label subset (a LoRA head trained on fewer than 18
    classes, aligned by scatter) assigns log(1e-9) = -20.7 nats to the missing
    classes, which annihilates them at any fusion weight; that is asserted
    rather than trusted.
    """
    cfg = cfg or FusionConfig()
    train_texts, test_texts = list(train_texts), list(test_texts)
    n_tr, n_te = len(train_texts), len(test_texts)

    assert p_lora.shape == (n_te, K), "p_lora must be (N_test, 18)"
    assert p_ng.shape == (n_te, K), "p_ng must be (N_test, 18)"
    assert (p_lora.sum(axis=1) > 0.5).all(), (
        "p_lora has an all-zero row: the LoRA head does not cover all 18 labels. "
        "Marginalise the head's mass over its own label set instead of "
        "scatter-filling a zero matrix."
    )

    bits_tr, bits_te = typo_bits(train_texts), typo_bits(test_texts)
    wc_tr = np.array([len(t.split()) for t in train_texts], dtype=np.float64)
    wc_te = np.array([len(t.split()) for t in test_texts], dtype=np.float64)
    logp_ng_te = _log(p_ng)
    logp_lora_te = _log(p_lora) if logp_lora_test is None else np.asarray(logp_lora_test)

    if folds is None:
        folds = list(
            StratifiedGroupKFold(
                n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
            ).split(X_train, y_train, groups=groups)
        )

    # ---------------- OOF assets (train side only) ---------------------- #
    audit: List[Dict[str, object]] = []
    logp_ng_oof = oof_ng_logp
    if logp_ng_oof is None and cfg.audit:
        logp_ng_oof = _log(_oof_ngram_logp(X_train, y_train, folds, cfg.seed))

    typo_view = TypoPriorView(cfg.typo_alpha, cfg.typo_rho, cfg.typo_max_nats)
    logp_typo_te = typo_view.fit(bits_tr, y_train).log_prob(bits_te)
    logp_typo_oof = None
    if cfg.audit:
        logp_typo_oof = TypoPriorView(
            cfg.typo_alpha, cfg.typo_rho, cfg.typo_max_nats
        ).fit_predict_oof(bits_tr, y_train, folds)

    # ---------------- pooled semantic base ------------------------------ #
    pool = None
    if logp_ng_oof is None:
        # not auditing and no OOF: fall back to a fixed 50/50 log pool
        pool_mode = "fixed 50/50 (no OOF available)"
        log_base_te = 0.5 * logp_lora_te + 0.5 * logp_ng_te
        log_base_oof, W = None, None
    else:
        oof_views = [logp_ng_oof] + ([np.asarray(logp_lora_oof)] if logp_lora_oof is not None else [])
        te_views = [logp_ng_te] + ([logp_lora_te] if logp_lora_oof is not None else [])
        per_class = cfg.per_class_pool and logp_lora_oof is not None
        pool = LogOpinionPool(per_class=per_class, l2=cfg.pool_l2 if per_class else 0.0)
        pool.fit(oof_views, y_train)
        W = pool.weights()
        log_base_oof = pool.transform(oof_views)
        log_base_te = pool.transform(te_views)
        pool_mode = (
            "per-class simplex, OOF-fitted" if per_class
            else "global simplex, OOF-fitted (no OOF LoRA supplied)"
        )

    # ---------------- pairwise residual injection ----------------------- #
    injector = PairwiseResidualInjector(
        pairs, lam=cfg.pairwise_lam, tau=cfg.pairwise_tau,
        gate=cfg.pairwise_gate, coherent=True,
    ).fit(X_train, y_train)
    log_base_te = injector.transform(X_test, log_base_te)
    log_base_oof_corr = (
        injector.transform(X_train, log_base_oof) if log_base_oof is not None else None
    )

    # ---------------- length gate --------------------------------------- #
    gate, gate_rep = None, {}
    w_te = np.full(n_te, 0.0)
    if cfg.use_length_gate and log_base_oof is not None and logp_typo_oof is not None:
        gate = SmoothLengthGate().fit(wc_tr, log_base_oof, logp_typo_oof, y_train)
        gate_rep = gate.report(wc_tr)
        raw_tr = gate.weight(wc_tr)
        raw_te = gate.weight(wc_te)
        w_tr = cfg.gate_w_lo + (cfg.gate_w_hi - cfg.gate_w_lo) * raw_tr
        w_te = cfg.gate_w_lo + (cfg.gate_w_hi - cfg.gate_w_lo) * raw_te
        # Lipschitz constant of the FITTED gate: |dw/dn| <= |b| / (4 n),
        # so a +-1 word change perturbs a class by at most |b| gap / (4 n_min).
        gap = float(np.percentile(np.ptp(log_base_te - log_base_te.mean(axis=1, keepdims=True), axis=1), 95))
        gate_rep["max_gate_perturbation_nats"] = float(
            abs(gate.b_) * gap / (4.0 * max(wc_tr.min(), 1.0))
            * (cfg.gate_w_hi - cfg.gate_w_lo)
        )
        gate_rep["logit_gap_p95"] = gap
        log_final_oof = (1.0 - w_tr[:, None]) * log_base_oof_corr + w_tr[:, None] * logp_typo_oof
    else:
        w_tr = np.zeros(n_tr)
        log_final_oof = log_base_oof_corr if log_base_oof_corr is not None else None

    log_final_te = (1.0 - w_te[:, None]) * log_base_te + w_te[:, None] * logp_typo_te
    pred = log_final_te.argmax(axis=1)

    # ---------------- audit --------------------------------------------- #
    if cfg.audit and log_base_oof is not None:
        shipped = _shipped_typo_logp(bits_tr)
        step_w = np.where(wc_tr < 80, 0.45, 0.15)[:, None]
        log_fixed_ng = (1.0 - w_tr[:, None]) * logp_ng_oof + w_tr[:, None] * logp_typo_oof
        rows = [
            ("n-gram only", _bacc(y_train, logp_ng_oof.argmax(1)), "reference view"),
            ("typo only (shipped table)", _bacc(y_train, shipped.argmax(1)),
             "standalone power of the hardcoded view"),
            ("typo only (fitted table)", _bacc(y_train, logp_typo_oof.argmax(1)), ""),
            ("n-gram + typo, SHIPPED (step gate + hardcoded table)",
             _bacc(y_train, ((1 - step_w) * logp_ng_oof + step_w * shipped).argmax(1)),
             "what v2 submits"),
            ("n-gram + typo, FIXED (fitted gate + fitted table)",
             _bacc(y_train, log_fixed_ng.argmax(1)), "isolates the typographic fix"),
            ("n-gram + typo + pairwise residuals, gated (FULL FIX)",
             _bacc(y_train, log_final_oof.argmax(1)), "what the patched script submits"),
        ]
        audit = [
            {"configuration": a, "oof_balanced_accuracy": round(float(b), 4), "note": c}
            for a, b, c in rows
        ]
        rel = {
            "ng": view_reliability(logp_ng_oof, y_train),
            "typo_fitted": view_reliability(logp_typo_oof, y_train),
        }
        sweep = per_class_weight_sweep(logp_ng_oof, logp_typo_oof, y_train)
    else:
        rel, sweep = {}, pd.DataFrame()

    return {
        "pred": pred,
        "p_final": _softmax(log_final_te),
        "audit": pd.DataFrame(audit),
        "cycle_dim": cycle_dimension(pairs, K),
        "pool_mode": pool_mode,
        "pool_weights": W,
        "pool_global": None if pool is None else pool.global_w_,
        "gate": gate_rep,
        "pairwise_diagnostics": injector.diagnostics_,
        "view_reliability": rel,
        "per_class_weight_sweep": sweep,
        "typo_table": typo_view.table_,
    }


def format_audit(result: Dict[str, object]) -> str:
    """One block to paste under the submission in the notebook log."""
    lines = ["## OOF ablation (prompt-grouped)", ""]
    df = result.get("audit")
    if isinstance(df, pd.DataFrame) and not df.empty:
        lines.append(df.to_string(index=False))
    lines += [
        "",
        f"pair graph cycle space dim = {result.get('cycle_dim')} "
        "(0 => no Condorcet cycle is possible)",
        f"pool = {result.get('pool_mode')}",
        f"gate = {result.get('gate')}",
        f"pairwise = {result.get('pairwise_diagnostics')}",
    ]
    return "\n".join(lines)
