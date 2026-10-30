"""
Experiment 25: the two feasibility questions experiment 24 could not answer.

  F1  Every OT number in experiment 24 used the TRUE prompt_id as the cluster.
      test.csv has no prompt_id, so at submission time clusters must be
      RECOVERED.  This measures how much of the +0.12 survives recovery, using
      the cheapest available representation (TF-IDF -> SVD -> cosine
      agglomerative), which is what the notebook can afford without new models.

  F2  short_passage_engine's premise is that it is BETTER than the base view on
      short passages, therefore a length gate should help below some L0.  That
      premise is testable per length bin and experiment 24 did not test it.
      If the short view loses in the bin it was built for, the gate is a
      no-op and no L0 can save it.

CPU only.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.short_passage_engine import ShortPassageFallbackModel  # noqa: E402
from src.test_time_calibration import sinkhorn_knopp_transport  # noqa: E402

RNG = 42
K = 18
CACHE = "experiments/cache_oof.npz"
CACHE_SHORT = "experiments/cache_short.npz"
OUT = "experiments/integration_gate3.json"

VALID_18_MODELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
]
M2I = {m: i for i, m in enumerate(VALID_18_MODELS)}


def bacc(y, p):
    a = np.asarray(p)
    return float(balanced_accuracy_score(y, a.argmax(1) if a.ndim == 2 else a))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def cluster_purity(pred, truth):
    """Fraction of rows whose cluster is dominated by their true prompt."""
    tot = 0
    for c in np.unique(pred):
        idx = np.where(pred == c)[0]
        vals, counts = np.unique(truth[idx], return_counts=True)
        tot += counts.max()
    return tot / len(truth)


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
    lp_ng, lp_dense = z["lp_ng"], z["lp_dense"]
    base = lp_dense if bacc(y, lp_dense) > bacc(y, lp_ng) else lp_ng
    print(f"base OOF bacc {bacc(y, base):.4f}")

    # ---- F2: short view per length bin ----------------------------------- #
    if os.path.exists(CACHE_SHORT):
        p_short = np.load(CACHE_SHORT)["p_short"]
        print("loaded short cache")
    else:
        print("\n[F2] fitting short-passage engine, 5-fold OOF ...")
        p_short = np.zeros((n, K))
        for k, (tr, va) in enumerate(folds):
            m = ShortPassageFallbackModel(class_order=VALID_18_MODELS,
                                          random_state=RNG + k)
            m.fit([texts[i] for i in tr],
                  [VALID_18_MODELS[y[i]] for i in tr], wc[tr])
            p_short[va] = m.predict_proba([texts[i] for i in va])
        np.savez_compressed(CACHE_SHORT, p_short=p_short)
    print(f"short view OOF bacc {bacc(y, p_short):.4f}")

    print("\n    per-length-bin: does the short view win where it claims to?")
    bins = [(0, 60), (60, 80), (80, 120), (120, 160), (160, 300)]
    bin_rows = []
    for lo, hi in bins:
        sel = (wc >= lo) & (wc < hi)
        if sel.sum() < 20:
            continue
        b_base = bacc(y[sel], base[sel])
        b_short = bacc(y[sel], p_short[sel])
        r = {"bin": f"{lo}-{hi}", "n": int(sel.sum()),
             "base": b_base, "short": b_short, "delta": b_short - b_base}
        bin_rows.append(r)
        print(f"      {lo:3d}-{hi:3d} n={sel.sum():4d}  base={b_base:.4f}  "
              f"short={b_short:.4f}  delta={r['delta']:+.4f}")
    res["length_bins"] = bin_rows

    p_base = softmax(base)
    for ix in [np.where(groups == g)[0] for g in dict.fromkeys(groups)]:
        p_base[ix] = sinkhorn_knopp_transport(softmax(base[ix]), reg=0.10,
                                              alpha=0.5, max_iter=200)
    b_ot = bacc(y, p_base)
    print(f"\n    with oracle-cluster OT: {b_ot:.4f}")
    for r in bin_rows:
        lo, hi = [int(v) for v in r["bin"].split("-")]
        sel = (wc >= lo) & (wc < hi)
        for w in (0.0, 0.2, 0.35, 0.5):
            p = (1 - w) * p_base[sel] + w * p_short[sel]
            if w in (0.0, 0.35):
                print(f"      gate w={w:.2f} on {r['bin']:8s}: "
                      f"{bacc(y[sel], p):.4f}")

    # ---- F1: recover clusters instead of reading prompt_id --------------- #
    print("\n[F1] recovering prompt clusters without prompt_id ...")
    # cheap representation the notebook already has: the 4-view TF-IDF is not
    # cached, so use the dense sibling block + posterior, both cached.
    feats = {}
    S = z["S"]
    feats["sibling_dense"] = normalize(S)
    lp = np.log(np.clip(softmax(base), 1e-9, 1))
    feats["posterior_sqrt"] = normalize(np.sqrt(softmax(base)))
    feats["dense+posterior"] = normalize(np.hstack([normalize(S),
                                                    normalize(np.sqrt(softmax(base)))]))

    out = []
    for name, F in feats.items():
        Fs = TruncatedSVD(n_components=min(64, F.shape[1] - 1),
                          random_state=RNG).fit_transform(F)
        Fs = normalize(Fs)
        for thresh in (0.6, 0.8, 0.95, 1.1):
            cl = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                         linkage="average",
                                         distance_threshold=thresh)
            pred = cl.fit_predict(Fs)
            pur = cluster_purity(pred, groups)
            sizes = np.bincount(pred)
            # OT within recovered clusters
            q = softmax(base)
            for c in np.unique(pred):
                ix = np.where(pred == c)[0]
                if len(ix) < 2:
                    continue
                q[ix] = sinkhorn_knopp_transport(softmax(base[ix]), reg=0.10,
                                                 alpha=0.5, max_iter=200)
            b = bacc(y, q)
            r = {"repr": name, "threshold": thresh, "n_clusters": int(len(sizes)),
                 "purity": float(pur), "median_size": float(np.median(sizes)),
                 "bacc": b, "delta_vs_oracle_cluster": b - b_ot}
            out.append(r)
            print(f"    {name:16s} th={thresh:.2f} clusters={len(sizes):4d} "
                  f"purity={pur:.3f} med_size={np.median(sizes):5.1f} "
                  f"bacc={b:.4f} ({b - b_ot:+.4f} vs oracle)")
    res["cluster_recovery"] = out
    res["bacc_oracle_cluster_ot"] = b_ot
    res["bacc_no_ot"] = bacc(y, base)

    res["runtime_seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT} in {res['runtime_seconds']}s")


if __name__ == "__main__":
    main()
