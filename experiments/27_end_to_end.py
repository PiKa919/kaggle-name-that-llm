"""
Experiment 27: the end-to-end number, measured the way the notebook will
actually run, plus the two remaining hyper-parameters.

Pipeline emulated exactly:

    base = 4-view TF-IDF  (+/- 101/93-dim sibling dense block)
    -> PairwiseResidualInjector (fold-fit, gated residual, LS potential)
    -> Sinkhorn-Knopp OT per RECOVERED content cluster
    -> optional short-passage gate

Deliberate choices:
  * clusters are recovered from content vocabulary, never from prompt_id
    (experiment 26: purity 0.986, 174 clusters vs 150 true)
  * clustering the full 2699-row batch mirrors the 4000-row test batch;
    it uses no labels, so it cannot leak
  * the injector is refit inside each fold for the CV estimate
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coherent_pairwise import PairwiseResidualInjector  # noqa: E402
from src.test_time_calibration import sinkhorn_knopp_transport  # noqa: E402

RNG = 42
K = 18
CACHE = "experiments/cache_oof.npz"
OUT = "experiments/integration_gate5.json"

VALID_18_MODELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
]
M2I = {m: i for i, m in enumerate(VALID_18_MODELS)}

PAIRS = [
    ("claude-opus-4-6", "claude-opus-5"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("gemini-3.6-flash", "gemini-3.7-flash"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("glm-5.3", "glm-5.3-flash"),
    ("glm-5.3", "claude-opus-4-6"),
]


def bacc(y, p):
    a = np.asarray(p)
    return float(balanced_accuracy_score(y, a.argmax(1) if a.ndim == 2 else a))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


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
    res = {}

    z = np.load(CACHE)
    lp_ng, lp_dense, S = z["lp_ng"], z["lp_dense"], z["S"]
    b_ng, b_dense = bacc(y, lp_ng), bacc(y, lp_dense)
    base = lp_dense if b_dense > b_ng else lp_ng
    b_base = max(b_ng, b_dense)
    print(f"base: sparse {b_ng:.4f} | sparse+sibling {b_dense:.4f} -> {b_base:.4f}")
    res["base_sparse"] = b_ng
    res["base_dense"] = b_dense

    # ---- injection ------------------------------------------------------- #
    print("\n[1] PairwiseResidualInjector, fold-fit ...")
    lp_inj = np.zeros_like(base)
    diag = {}
    for k, (tr, va) in enumerate(folds):
        inj = PairwiseResidualInjector(PAIRS, lam=0.15, tau=1.0, gate=True,
                                       coherent=True)
        inj.fit(S[tr], y[tr])
        lp_inj[va] = inj.transform(S[va], base[va])
        diag = dict(inj.diagnostics_)
    b_inj = bacc(y, lp_inj)
    print(f"    bacc {b_inj:.4f}  ({b_inj - b_base:+.4f})  "
          f"fidelity={diag.get('margin_fidelity')}")

    # sweep lam on the same OOF (cheap, no refit needed: lam is a scale)
    lam_rows = []
    for lam in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40):
        lp = np.zeros_like(base)
        for k, (tr, va) in enumerate(folds):
            inj = PairwiseResidualInjector(PAIRS, lam=lam, tau=1.0, gate=True,
                                           coherent=True)
            inj.fit(S[tr], y[tr])
            lp[va] = inj.transform(S[va], base[va])
        lam_rows.append({"lam": lam, "bacc": bacc(y, lp)})
        print(f"    lam={lam:.2f} bacc={lam_rows[-1]['bacc']:.4f}")
    res["lam_sweep"] = lam_rows
    best_lam = max(lam_rows, key=lambda r: r["bacc"])
    res["best_lam"] = best_lam["lam"]
    if best_lam["lam"] != 0.15:
        lp_inj = np.zeros_like(base)
        for k, (tr, va) in enumerate(folds):
            inj = PairwiseResidualInjector(PAIRS, lam=best_lam["lam"], tau=1.0,
                                           gate=True, coherent=True)
            inj.fit(S[tr], y[tr])
            lp_inj[va] = inj.transform(S[va], base[va])
        b_inj = bacc(y, lp_inj)
    res["bacc_after_injection"] = b_inj

    # ---- recovered clusters ---------------------------------------------- #
    print("\n[2] recovering clusters from content vocabulary ...")
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True, max_features=80_000,
                          dtype=np.float32)
    F = normalize(TruncatedSVD(n_components=128, random_state=RNG)
                  .fit_transform(normalize(vec.fit_transform(texts))))
    pred = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                   linkage="average",
                                   distance_threshold=0.35).fit_predict(F)
    cl = [np.where(pred == c)[0] for c in np.unique(pred)]
    sizes = np.array([len(c) for c in cl])
    print(f"    {len(cl)} clusters, median {np.median(sizes):.0f}, "
          f"mean {sizes.mean():.1f}")
    res["n_clusters"] = int(len(cl))
    res["median_cluster_size"] = float(np.median(sizes))

    p_inj = softmax(lp_inj)
    print("\n[3] reg x alpha on recovered clusters ...")
    grid = []
    for reg in (0.02, 0.05, 0.10, 0.20, 0.50):
        for alpha in (0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
            q = p_inj.copy()
            for ix in cl:
                if len(ix) < 2:
                    continue
                q[ix] = sinkhorn_knopp_transport(p_inj[ix], reg=reg, alpha=alpha,
                                                 max_iter=200)
            grid.append({"reg": reg, "alpha": alpha, "bacc": bacc(y, q)})
    grid.sort(key=lambda r: -r["bacc"])
    for r in grid[:8]:
        print(f"    reg={r['reg']:.2f} alpha={r['alpha']:.2f} bacc={r['bacc']:.4f}")
    res["ot_grid_recovered"] = grid
    best = grid[0]
    res["ot_best_recovered"] = best

    print("\n    reg profile at alpha=0.5 (robustness margin):")
    for reg in (0.02, 0.05, 0.10, 0.20, 0.50):
        r = next(x for x in grid if x["reg"] == reg and x["alpha"] == 0.5)
        print(f"      reg={reg:.2f} bacc={r['bacc']:.4f}")
        res.setdefault("reg_profile_alpha05", {})[str(reg)] = r["bacc"]

    q_best = p_inj.copy()
    for ix in cl:
        if len(ix) >= 2:
            q_best[ix] = sinkhorn_knopp_transport(p_inj[ix], reg=best["reg"],
                                                  alpha=best["alpha"], max_iter=200)
    b_final = bacc(y, q_best)
    res["bacc_final"] = b_final

    # cluster bootstrap CI on the OT delta
    rng = np.random.default_rng(RNG)
    uniq = np.unique(groups)
    deltas = []
    for _ in range(300):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in pick])
        deltas.append(bacc(y[idx], q_best[idx]) - bacc(y[idx], p_inj[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"\n[4] OT delta {np.mean(deltas):+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")
    res["ot_delta_mean"] = float(np.mean(deltas))
    res["ot_delta_ci"] = [float(lo), float(hi)]

    # per-class recall
    cm0 = confusion_matrix(y, p_inj.argmax(1), labels=range(K))
    cm1 = confusion_matrix(y, q_best.argmax(1), labels=range(K))
    r0 = np.diag(cm0) / np.clip(cm0.sum(1), 1, None)
    r1 = np.diag(cm1) / np.clip(cm1.sum(1), 1, None)
    print("\n[5] per-class recall before -> after OT")
    tbl = []
    for i, m in enumerate(VALID_18_MODELS):
        tbl.append({"model": m, "before": float(r0[i]), "after": float(r1[i]),
                    "delta": float(r1[i] - r0[i])})
        print(f"    {m:26s} {r0[i]:.3f} -> {r1[i]:.3f} ({r1[i]-r0[i]:+.3f})")
    res["per_class_recall"] = tbl
    res["worst_class_before"] = float(min(r0))
    res["worst_class_after"] = float(min(r1))
    res["n_classes_improved"] = int((r1 > r0).sum())
    res["n_classes_hurt"] = int((r1 < r0).sum())

    # ---- short gate on top ------------------------------------------------ #
    print("\n[6] short gate on top of the final posterior ...")
    p_short = np.load("experiments/cache_short.npz")["p_short"]
    gate = []
    for L0 in (60, 80, 100, 120):
        for w_max in (0.1, 0.2, 0.35):
            w = w_max / (1.0 + np.exp((wc - L0) / 10.0))
            p = (1 - w)[:, None] * q_best + w[:, None] * p_short
            gate.append({"L0": L0, "w_max": w_max, "bacc": bacc(y, p)})
    gate.sort(key=lambda r: -r["bacc"])
    for r in gate[:5]:
        print(f"    L0={r['L0']:3d} w_max={r['w_max']:.2f} bacc={r['bacc']:.4f}")
    res["gate_on_final"] = gate
    res["gate_best_delta_vs_final"] = gate[0]["bacc"] - b_final
    print(f"    best gate delta vs final: {res['gate_best_delta_vs_final']:+.4f}")

    res["runtime_seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT} in {res['runtime_seconds']}s")
    print(f"\nEND TO END: {b_base:.4f} -> inject {b_inj:.4f} -> +OT {b_final:.4f}")


if __name__ == "__main__":
    main()
