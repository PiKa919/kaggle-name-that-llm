"""
Experiment 23: measured integration gate for the three new src/ modules.

Question we are answering with numbers, not opinion:
  A. Does Sinkhorn-Knopp OT help on prompt-grouped OOF, and at which
     (alpha, reg)?  Does its placement relative to the HodgeRank potential
     projection matter?
  B. Do the 101 sibling stylometric features add anything to the 8 pairwise
     sibling classifiers over the 4-view TF-IDF they already use?
  C. Is the short-passage gate threshold (L0) worth moving off 80, and what
     is the entropy ceiling doing?

Honest scope: the LoRA (Qwen3-8B readout) view cannot be produced on this
machine.  p_semantic is therefore the 4-view TF-IDF view, exactly as in
INVALIDATION_v2.md.  Every calibration hyper-parameter below must be re-fitted
on the real fused OOF inside the notebook; the script prints the sweep so the
notebook can do that in one line.

CPU only.  Writes experiments/integration_gate.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import balanced_accuracy_score  # noqa: E402


def balanced_accuracy(y_true, y_pred_or_scores):
    """Balanced accuracy. Accepts labels or a (n, K) score/posterior matrix."""
    y_true = np.asarray(y_true)
    a = np.asarray(y_pred_or_scores)
    y_pred = a.argmax(axis=1) if a.ndim == 2 else a
    return float(balanced_accuracy_score(y_true, y_pred))
from src.sibling_features import (  # noqa: E402
    FEATURE_NAMES,
    SIBLING_PAIRS,
    extract_sibling_features,
    get_pair_feature_indices,
)
from src.test_time_calibration import (  # noqa: E402
    adaptive_neighborhood_centering,
    sinkhorn_knopp_transport,
)

RNG = 42
N_CLASSES = 18
TRAIN = "data/train.csv"
OUT = "experiments/integration_gate.json"

VALID_18_MODELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
]
M2I = {m: i for i, m in enumerate(VALID_18_MODELS)}

# The 8 pairs actually shipped in notebooks/master_sota/submission_sota.py
SHIPPED_PAIRS = [
    ("claude-opus-4-6", "claude-opus-5"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("gemini-3.6-flash", "gemini-3.7-flash"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("glm-5.3", "glm-5.3-flash"),
    ("glm-5.3", "claude-opus-4-6"),
]

FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom "
    "whose this these those it its they them their theirs he him his she her hers "
    "we us our ours you your yours i me my mine in on at to from with without for "
    "of by over under between through into about is are was were be been being have "
    "has had do does did can could will would shall should may might must not no nor "
    "all any some each every both either neither other another such only also just "
    "very too more most less least here there when where why how what whether then "
    "now still yet even much many few little".split()
)


def shape_transform(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_transform(text: str) -> str:
    return " ".join(
        w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
        for w in re.findall(r"\w+|[^\w\s]", text)
    )


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_views(df):
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s),
        ("word", 1, 2, 120_000, 0.8, lambda s: s),
        ("shape", 2, 5, 40_000, 0.5, shape_transform),
        ("function", 1, 2, 30_000, 0.5, function_transform),
    ]
    mats = []
    for vname, lo, hi, max_f, w, prep in views:
        analyzer = "char" if vname in ("char", "shape") else "word"
        tok = r"(?u)\b\w+\b|[^\w\s]" if analyzer == "word" else None
        vec = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(lo, hi), max_features=max_f,
            lowercase=False, sublinear_tf=True, min_df=2, token_pattern=tok,
            dtype=np.float32,
        )
        mats.append(vec.fit_transform([prep(t) for t in df["text"]]) * w)
    return hstack(mats, format="csr")


def main():
    t0 = time.time()
    df = pd.read_csv(TRAIN)
    df["text"] = df["text"].fillna("")
    y = np.array([M2I[m] for m in df["model"]])
    groups = df["prompt_id"].values
    wc = df["text"].str.split().str.len().values
    n = len(df)
    print(f"train {n} rows, {len(set(groups))} prompts, {N_CLASSES} classes")

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
                 .split(df, y, groups=groups))
    res = {}

    # ---- view 1: 4-view TF-IDF, 5-fold OOF -------------------------------- #
    print("\n[1] 4-view TF-IDF OOF ...")
    t = time.time()
    X = build_views(df)
    print(f"    X {X.shape} nnz={X.nnz:,} {time.time()-t:.1f}s")
    lp_ng = np.zeros((n, N_CLASSES))
    for k, (tr, va) in enumerate(folds):
        clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced",
                                 random_state=RNG + k, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        lp_ng[va] = np.log(np.clip(clf.predict_proba(X[va]), 1e-9, 1))
        print(f"    fold {k} done {time.time()-t:.0f}s")
    bacc_ng = balanced_accuracy(y, lp_ng.argmax(1))
    print(f"    n-gram OOF bacc = {bacc_ng:.4f}")
    res["bacc_ngram"] = bacc_ng

    # Pairwise specialists: TF-IDF only vs TF-IDF + pair-routed dense vs dense only
    print("\n[2] sibling_features on the 8 shipped pairs ...")
    t = time.time()
    S = extract_sibling_features(df["text"].tolist())
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S = StandardScaler().fit_transform(S)          # unit variance, no labels
    print(f"    sibling matrix {S.shape} median|z|={np.median(np.abs(S)):.2f} "
          f"{time.time()-t:.1f}s")
    res["sibling_shape"] = list(S.shape)
    res["sibling_extract_seconds_2699"] = round(time.time() - t, 2)

    # ---- does the dense view help the 18-way base at all? ----------------- #
    print("\n[2a] 18-way LR with/without the dense sibling block ...")
    dense_views = {}
    for tag, scale in (("dense_only", 1.0), ("sparse+dense0.1", 0.1),
                       ("sparse+dense0.5", 0.5)):
        lp = np.zeros((n, N_CLASSES))
        for k, (tr, va) in enumerate(folds):
            Z = csr_matrix(S * scale)
            Xin = Z if tag == "dense_only" else hstack([X, Z], format="csr")
            clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced",
                                     random_state=RNG + k)
            clf.fit(Xin[tr], y[tr])
            lp[va] = np.log(np.clip(clf.predict_proba(Xin[va]), 1e-9, 1))
        dense_views[tag] = (lp, balanced_accuracy(y, lp.argmax(1)))
        print(f"    {tag:16s} OOF bacc = {dense_views[tag][1]:.4f}")
    res["dense_18way"] = {k: v[1] for k, v in dense_views.items()}

    pair_rows = []
    pair_logodds = {}          # (m1,m2) -> OOF log-odds, fold-fit routed dense
    pair_logodds_tfidf = {}    # (m1,m2) -> OOF log-odds, TF-IDF only
    PAIR_K = 12

    def select_pair_features(S_tr, yp_tr, k=PAIR_K):
        """Fold-internal Cohen's d ranking. Leak-free and independent of the
        hand-authored routing table (which is missing one shipped pair)."""
        a, b = S_tr[yp_tr == 0], S_tr[yp_tr == 1]
        if len(a) < 3 or len(b) < 3:
            return np.arange(S_tr.shape[1])
        d = np.abs(b.mean(0) - a.mean(0)) / (b.std(0) + a.std(0) + 1e-9)
        return np.argsort(-d)[:k]

    for m1, m2 in SHIPPED_PAIRS:
        try:
            key = (m1, m2) if (m1, m2) in SIBLING_PAIRS else (m2, m1)
            routed = get_pair_feature_indices(key)
        except KeyError:
            routed = None
        mask = (y == M2I[m1]) | (y == M2I[m2])
        yp = (y[mask] == M2I[m2]).astype(int)
        Xt = X[mask]
        St = S[mask]
        oof_tf = np.zeros(mask.sum())
        oof_dense = np.zeros(mask.sum())
        oof_both = np.zeros(mask.sum())
        row_pos = np.where(mask)[0]
        for k, (tr, va) in enumerate(folds):
            tr_m = np.isin(row_pos, tr)
            va_m = np.isin(row_pos, va)
            if yp[tr_m].sum() < 3 or (1 - yp[tr_m]).sum() < 3 or va_m.sum() == 0:
                continue
            a = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced",
                                   random_state=RNG)
            a.fit(Xt[tr_m], yp[tr_m])
            oof_tf[va_m] = a.predict_proba(Xt[va_m])[:, 1]

            sel = select_pair_features(St[tr_m], yp[tr_m])
            Sd = csr_matrix(St[:, sel])
            b = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced",
                                   random_state=RNG)
            b.fit(Sd[tr_m], yp[tr_m])
            oof_dense[va_m] = b.predict_proba(Sd[va_m])[:, 1]
            c = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced",
                                   random_state=RNG)
            c.fit(hstack([Xt[tr_m], Sd[tr_m] * 0.1], format="csr"), yp[tr_m])
            oof_both[va_m] = c.predict_proba(
                hstack([Xt[va_m], Sd[va_m] * 0.1], format="csr"))[:, 1]

        def acc(p):
            return float(((p > 0.5).astype(int) == yp).mean())

        pair_rows.append({
            "pair": f"{m1}|{m2}", "n": int(mask.sum()),
            "routed": routed is not None,
            "acc_tfidf": acc(oof_tf), "acc_dense": acc(oof_dense),
            "acc_both": acc(oof_both),
        })
        for store, p in ((pair_logodds_tfidf, oof_tf), (pair_logodds, oof_both)):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            store[(m1, m2)] = (np.log(p / (1 - p)), row_pos)

    for r in pair_rows:
        print(f"    {r['pair']:36s} n={r['n']:4d} routed={str(r['routed']):5s} "
              f"tfidf={r['acc_tfidf']:.3f} dense={r['acc_dense']:.3f} "
              f"both={r['acc_both']:.3f}")
    res["pairwise"] = pair_rows
    res["pair_mean_acc_tfidf"] = float(np.mean([r["acc_tfidf"] for r in pair_rows]))
    res["pair_mean_acc_dense"] = float(np.mean([r["acc_dense"] for r in pair_rows]))
    res["pair_mean_acc_both"] = float(np.mean([r["acc_both"] for r in pair_rows]))

    # ---- HodgeRank / potential injection -------------------------------- #
    def inject(lp_base, logodds, weight, gate=None):
        """Least-squares potential injection, per-row. Mirrors
        src/coherent_pairwise.py semantics (residual-free here: we inject the
        pair log-odds gated by pair membership)."""
        out = lp_base.copy()
        p = softmax(lp_base)
        A = np.zeros((N_CLASSES, N_CLASSES))
        for (i, j) in [(M2I[a], M2I[b]) for a, b in SHIPPED_PAIRS]:
            A[i, j] = A[j, i] = 1.0
        L = np.diag(A.sum(1)) - A
        Lp = np.linalg.pinv(L)
        B = np.zeros_like(lp_base)
        for (m1, m2), (d, rows) in logodds.items():
            i, j = M2I[m1], M2I[m2]
            # margin already implied by the base model
            s = lp_base[rows, j] - lp_base[rows, i]
            r = d - 0.0 * s
            g = (p[rows, i] + p[rows, j]) if gate is None else gate
            v = weight * g * r
            B[rows, j] += v
            B[rows, i] -= v
        V = B @ Lp
        V -= V.mean(axis=1, keepdims=True)
        return out + V

    print("\n[3] pairwise injection (potential, membership-gated) ...")
    inj = {}
    inj_by_source = {}
    for src_name, src in (("tfidf_only", pair_logodds_tfidf),
                          ("tfidf+dense", pair_logodds)):
        for w in (0.0, 0.05, 0.10, 0.20):
            lp = inject(lp_ng, src, w)
            b = balanced_accuracy(y, lp.argmax(1))
            inj_by_source[f"{src_name}|{w:.2f}"] = b
            if src_name == "tfidf+dense":
                inj[w] = b
            print(f"    {src_name:12s} w={w:.2f}  bacc={b:.4f}  (base {bacc_ng:.4f})")
    res["injection_sweep"] = inj_by_source
    best_w = max(inj, key=inj.get)
    lp_inj = inject(lp_ng, pair_logodds, best_w)
    res["best_injection_weight"] = best_w
    # also: does replacing the base with the dense-augmented 18-way view help?
    lp_dense_aug, b_dense_aug = dense_views["sparse+dense0.1"]
    lp_best_base = lp_dense_aug if b_dense_aug > bacc_ng else lp_ng
    res["bacc_base_chosen"] = max(bacc_ng, b_dense_aug)
    print(f"    chosen base view bacc = {max(bacc_ng, b_dense_aug):.4f}")

    # ---- Sinkhorn sweep, per prompt cluster ------------------------------ #
    print("\n[4] Sinkhorn-Knopp sweep (per prompt cluster, OOF) ...")
    clusters = {}
    for pid, idxs in df.groupby("prompt_id").indices.items():
        clusters[pid] = np.asarray(idxs)

    def apply_ot(lp, alpha, reg, max_iter=200):
        """Per-cluster entropic OT. alpha is the module's own blend: the
        returned matrix is already (1-a)P + a*(n*T), so we must NOT blend
        again downstream."""
        out = softmax(lp)
        for idxs in clusters.values():
            p = softmax(lp[idxs])
            out[idxs] = sinkhorn_knopp_transport(
                p, reg=reg, alpha=alpha, max_iter=max_iter)
        return out

    grid = []
    t = time.time()
    for reg in (0.02, 0.05, 0.10, 0.20, 0.50):
        for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
            q = apply_ot(lp_inj, alpha, reg)
            b = balanced_accuracy(y, q.argmax(1))
            grid.append({"reg": reg, "alpha": alpha, "bacc": b})
    print(f"    {len(grid)} configs in {time.time()-t:.1f}s")
    grid.sort(key=lambda r: -r["bacc"])
    for r in grid[:10]:
        print(f"    reg={r['reg']:.2f} alpha={r['alpha']:.2f} bacc={r['bacc']:.4f}")
    res["sinkhorn_grid"] = grid
    best = grid[0]
    res["sinkhorn_best"] = best

    # placement test: OT then injection vs injection then OT
    print("\n[5] placement: OT-before-potential vs OT-after-potential ...")
    # (a) OT applied to the raw n-gram posterior, then potential injected
    q_raw = apply_ot(lp_ng, best["alpha"], best["reg"])
    lp_ot_then_inj = inject(np.log(np.clip(q_raw, 1e-12, 1)), pair_logodds, best_w)
    b_ot_then_inj = balanced_accuracy(y, lp_ot_then_inj.argmax(1))
    # (b) potential first, then OT
    b_inj_then_ot = balanced_accuracy(y, apply_ot(lp_inj, best["alpha"], best["reg"]).argmax(1))
    print(f"    OT -> potential : {b_ot_then_inj:.4f}")
    print(f"    potential -> OT : {b_inj_then_ot:.4f}")
    res["placement"] = {"ot_then_potential": b_ot_then_inj,
                        "potential_then_ot": b_inj_then_ot}

    # ---- neighborhood centering sweep ------------------------------------ #
    print("\n[6] adaptive neighborhood centering sweep ...")
    cen = []
    p_inj = softmax(lp_inj)
    logp_inj = np.log(np.clip(p_inj, 1e-12, 1))
    # oracle: true prompt-cluster mean removed (upper bound, not available at
    # test time because test.csv has no prompt_id)
    for strength in (0.0, 0.1, 0.2, 0.3, 0.5):
        for k in (8, 12, 24):
            try:
                q = adaptive_neighborhood_centering(
                    p_inj, strength=strength, k_neighbors=k)
            except TypeError:
                q = adaptive_neighborhood_centering(p_inj, strength=strength)
            q = apply_ot(np.log(np.clip(q, 1e-12, 1)), best["alpha"], best["reg"])
            cen.append({"strength": strength, "k": k, "mode": "posterior-knn",
                        "bacc": balanced_accuracy(y, q.argmax(1))})
    # oracle variant: subtract the true prompt-cluster mean log-posterior
    for strength in (0.0, 0.1, 0.2, 0.3, 0.5):
        z = logp_inj.copy()
        for idxs in clusters.values():
            z[idxs] -= strength * logp_inj[idxs].mean(axis=0, keepdims=True)
        q = softmax(z)
        q = apply_ot(np.log(np.clip(q, 1e-12, 1)), best["alpha"], best["reg"])
        cen.append({"strength": strength, "k": 18, "mode": "oracle-prompt",
                    "bacc": balanced_accuracy(y, q.argmax(1))})
    cen.sort(key=lambda r: -r["bacc"])
    for r in cen[:8]:
        print(f"    {r['mode']:14s} strength={r['strength']:.1f} k={r['k']:2d} "
              f"bacc={r['bacc']:.4f}")
    res["centering_grid"] = cen

    res["runtime_seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT} in {res['runtime_seconds']}s")
    print(f"BEST: ngram {bacc_ng:.4f} -> injected {inj[best_w]:.4f} -> "
          f"+OT {best['bacc']:.4f}")


if __name__ == "__main__":
    main()
