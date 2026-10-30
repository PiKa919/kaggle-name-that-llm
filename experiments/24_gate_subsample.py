"""
Experiment 24: the three questions the brief asks, answered against the
subsampling regime the hidden test actually lives in.

  Q1  placement of Sinkhorn-Knopp OT relative to the pairwise potential
  Q2  alpha / reg for OT, and the short-passage gate location L0
  Q3  cost of sibling_features on the 8 pairwise heads (and whether it helps)

Why a second experiment: experiment 23 measured +0.12 balanced accuracy from
per-prompt OT on prompt-grouped OOF.  That number is not trustworthy on its
face, because every OOF cluster is a COMPLETE 18-model round -- exactly the
balanced regime in which forcing uniform column marginals is trivially right.
The hidden test is subsampled (AGENTS.md 4.4: the hard Hungarian version of
this same idea gained +0.10 CV and lost 0.009 LB).  So we re-measure with
clusters deterministically subsampled to a retention fraction rho, and report
the OT gain as a function of rho.  The rho at which the gain dies is the
number that matters.

CPU only.  Caches the OOF matrices to experiments/cache_oof.npz.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sibling_features import extract_sibling_features  # noqa: E402
from src.short_passage_engine import (  # noqa: E402
    ShortPassageFallbackModel,
    entropy_regularized_short_gate,
)
from src.test_time_calibration import sinkhorn_knopp_transport  # noqa: E402

RNG = 42
K = 18
CACHE = "experiments/cache_oof.npz"
OUT = "experiments/integration_gate2.json"

VALID_18_MODELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
]
M2I = {m: i for i, m in enumerate(VALID_18_MODELS)}

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


def shape_transform(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_transform(t):
    return " ".join(
        w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
        for w in re.findall(r"\w+|[^\w\s]", t))


def bacc(y, p):
    a = np.asarray(p)
    return float(balanced_accuracy_score(y, a.argmax(1) if a.ndim == 2 else a))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def build_views(texts):
    specs = [("char", 3, 5, 200_000, 1.0, lambda s: s),
             ("word", 1, 2, 120_000, 0.8, lambda s: s),
             ("shape", 2, 5, 40_000, 0.5, shape_transform),
             ("function", 1, 2, 30_000, 0.5, function_transform)]
    mats = []
    for name, lo, hi, mf, w, prep in specs:
        an = "char" if name in ("char", "shape") else "word"
        tok = r"(?u)\b\w+\b|[^\w\s]" if an == "word" else None
        v = TfidfVectorizer(analyzer=an, ngram_range=(lo, hi), max_features=mf,
                            lowercase=False, sublinear_tf=True, min_df=2,
                            token_pattern=tok, dtype=np.float32)
        mats.append(v.fit_transform([prep(t) for t in texts]) * w)
    return hstack(mats, format="csr")


def main():
    t0 = time.time()
    rows = list(csv.DictReader(open("data/train.csv")))
    texts = [r["text"] or "" for r in rows]
    y = np.array([M2I[r["model"]] for r in rows])
    groups = np.array([r["prompt_id"] for r in rows])
    wc = np.array([len(t.split()) for t in texts])
    n = len(rows)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=RNG)
                 .split(np.zeros(n), y, groups=groups))
    clusters = [np.asarray(ix) for ix in
                (np.where(groups == g)[0] for g in dict.fromkeys(groups))]
    res = {}

    if os.path.exists(CACHE):
        z = np.load(CACHE)
        lp_ng, lp_dense, S = z["lp_ng"], z["lp_dense"], z["S"]
        print(f"loaded cache {CACHE}")
    else:
        print("[1] building views ...")
        X = build_views(texts)
        print(f"    X {X.shape} in {time.time()-t0:.0f}s")
        S = StandardScaler().fit_transform(
            np.nan_to_num(extract_sibling_features(texts)))
        lp_ng = np.zeros((n, K))
        lp_dense = np.zeros((n, K))
        for k, (tr, va) in enumerate(folds):
            a = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced",
                                   random_state=RNG + k)
            a.fit(X[tr], y[tr])
            lp_ng[va] = np.log(np.clip(a.predict_proba(X[va]), 1e-9, 1))
            b = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced",
                                   random_state=RNG + k)
            b.fit(hstack([X[tr], csr_matrix(S[tr] * 0.1)], format="csr"), y[tr])
            lp_dense[va] = np.log(np.clip(
                b.predict_proba(hstack([X[va], csr_matrix(S[va] * 0.1)],
                                       format="csr")), 1e-9, 1))
            print(f"    fold {k} {time.time()-t0:.0f}s")
        np.savez_compressed(CACHE, lp_ng=lp_ng, lp_dense=lp_dense, S=S)

    b_ng, b_dense = bacc(y, lp_ng), bacc(y, lp_dense)
    print(f"\nbase: ngram {b_ng:.4f}  ngram+sibling-dense {b_dense:.4f}")
    res["base_ngram"] = b_ng
    res["base_dense"] = b_dense

    # paired bootstrap on the dense delta (cluster bootstrap over prompts)
    rng = np.random.default_rng(RNG)
    uniq = np.unique(groups)
    deltas = []
    for _ in range(400):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        deltas.append(bacc(y[idx], lp_dense[idx]) - bacc(y[idx], lp_ng[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"    dense delta {np.mean(deltas):+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")
    res["dense_delta_ci"] = [float(lo), float(hi)]
    res["dense_delta_mean"] = float(np.mean(deltas))

    base = lp_dense if b_dense > b_ng else lp_ng
    b_base = max(b_ng, b_dense)

    # ---- Q1/Q2: OT under subsampling ------------------------------------- #
    print("\n[2] OT on subsampled clusters (the hidden-test regime) ...")
    grid = []
    rng = np.random.default_rng(RNG)
    for rho in (1.0, 0.9, 0.75, 0.5):
        keep = []
        for ix in clusters:
            m = max(2, int(round(rho * len(ix))))
            keep.append(np.sort(rng.choice(ix, size=m, replace=False)))
        mask = np.zeros(n, bool)
        for ix in keep:
            mask[ix] = True
        b_raw = bacc(y[mask], base[mask])
        print(f"    rho={rho:.2f} n={mask.sum():4d} raw bacc={b_raw:.4f}")
        for reg in (0.02, 0.05, 0.10, 0.20, 0.50):
            for alpha in (0.1, 0.3, 0.5, 0.7, 1.0):
                q = softmax(base)
                for ix in keep:
                    q[ix] = sinkhorn_knopp_transport(
                        softmax(base[ix]), reg=reg, alpha=alpha, max_iter=200)
                grid.append({"rho": rho, "reg": reg, "alpha": alpha,
                             "bacc": bacc(y[mask], q[mask]),
                             "delta": bacc(y[mask], q[mask]) - b_raw})
        best = max((g for g in grid if g["rho"] == rho), key=lambda g: g["bacc"])
        print(f"      best: reg={best['reg']} alpha={best['alpha']} "
              f"bacc={best['bacc']:.4f} (delta {best['delta']:+.4f})")
    res["ot_subsample"] = grid

    print("\n    best alpha/reg by rho")
    for rho in (1.0, 0.9, 0.75, 0.5):
        sub = [g for g in grid if g["rho"] == rho]
        b = max(sub, key=lambda g: g["bacc"])
        pos = [g for g in sub if g["delta"] > 0]
        print(f"      rho={rho:.2f} best reg={b['reg']:<5} alpha={b['alpha']:<4} "
              f"bacc={b['bacc']:.4f} | configs still positive: {len(pos)}/{len(sub)}")
        res[f"ot_best_rho_{rho}"] = b
        res[f"ot_positive_rho_{rho}"] = [len(pos), len(sub)]

    # alpha sensitivity at the reg that wins most often
    print("\n[3] alpha sensitivity at reg=0.10, rho=0.75 ...")
    regs = {}
    keep = []
    rng2 = np.random.default_rng(RNG)
    for ix in clusters:
        m = max(2, int(round(0.75 * len(ix))))
        keep.append(np.sort(rng2.choice(ix, size=m, replace=False)))
    mask = np.zeros(n, bool)
    for ix in keep:
        mask[ix] = True
    for alpha in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        q = softmax(base)
        for ix in keep:
            q[ix] = sinkhorn_knopp_transport(softmax(base[ix]), reg=0.10,
                                             alpha=alpha, max_iter=200)
        regs[alpha] = bacc(y[mask], q[mask])
        print(f"    alpha={alpha:.2f} bacc={regs[alpha]:.4f}")
    res["alpha_curve_reg010_rho075"] = {str(k): v for k, v in regs.items()}

    # ---- Q3: short-passage engine + gate --------------------------------- #
    print("\n[4] short-passage engine, 5-fold OOF ...")
    t = time.time()
    p_short = np.zeros((n, K))
    for k, (tr, va) in enumerate(folds):
        m = ShortPassageFallbackModel(class_order=VALID_18_MODELS,
                                      random_state=RNG + k)
        m.fit([texts[i] for i in tr], [VALID_18_MODELS[y[i]] for i in tr],
              wc[tr])
        p_short[va] = m.predict_proba([texts[i] for i in va])
        print(f"    fold {k} {time.time()-t:.0f}s  "
              f"short-only bacc={bacc(y[va], p_short[va]):.4f}")
    b_short = bacc(y, p_short)
    print(f"    OOF short-view bacc = {b_short:.4f}")
    res["short_view_bacc"] = b_short
    res["short_fit_seconds"] = round(time.time() - t, 1)

    print("\n[5] gate sweep L0 x tau (fused with OT applied last) ...")
    gate = []
    p_base = softmax(base)
    # one OT pass at the chosen setting so the gate is measured on the real
    # operating point, not on the raw posterior
    p_ot = p_base.copy()
    for ix in clusters:
        p_ot[ix] = sinkhorn_knopp_transport(p_base[ix], reg=0.10, alpha=0.5,
                                            max_iter=200)
    for L0 in (40, 50, 60, 70, 80, 90, 100, 120):
        for tau in (5.0, 10.0, 20.0):
            for w_max in (0.2, 0.35, 0.5):
                w = w_max / (1.0 + np.exp((wc - L0) / tau))
                p = (1 - w)[:, None] * p_ot + w[:, None] * p_short
                gate.append({"L0": L0, "tau": tau, "w_max": w_max,
                             "bacc": bacc(y, p)})
    gate.sort(key=lambda g: -g["bacc"])
    for g in gate[:10]:
        print(f"    L0={g['L0']:4d} tau={g['tau']:4.1f} w_max={g['w_max']:.2f} "
              f"bacc={g['bacc']:.4f}")
    base_bacc = bacc(y, p_ot)
    print(f"    no short view: {base_bacc:.4f}")
    res["gate_sweep"] = gate
    res["gate_best"] = gate[0]
    res["gate_no_short"] = base_bacc

    # the module's own gate, for comparison
    p_mod = entropy_regularized_short_gate(p_ot, p_short, wc)
    res["module_gate_bacc"] = bacc(y, p_mod)
    print(f"    module's entropy_regularized_short_gate: "
          f"{res['module_gate_bacc']:.4f}")

    res["runtime_seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT} in {res['runtime_seconds']}s")


if __name__ == "__main__":
    main()
