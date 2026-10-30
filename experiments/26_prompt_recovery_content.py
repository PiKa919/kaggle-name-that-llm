"""
Experiment 26: the linchpin test.

Experiment 25 recovered prompt clusters from stylometric features and failed
(best purity 0.27).  That result is uninformative, because every stylometric
feature in this repo is deliberately topic-INVARIANT: it measures the author,
not the prompt.  Prompt identity lives in CONTENT vocabulary.

So this re-tests with a content representation (word 1-2 grams, no authorial
signal), and also with a soft alternative that avoids hard clustering
entirely: per-row kNN OT, where every passage gets its own transport problem
over its k nearest content neighbours (overlapping windows allowed).

If neither recovers the oracle-cluster gain, tree-free OT cannot be shipped.
If one does, the notebook has a concrete recipe.
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
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.test_time_calibration import sinkhorn_knopp_transport  # noqa: E402

RNG = 42
CACHE = "experiments/cache_oof.npz"
OUT = "experiments/integration_gate4.json"

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


def purity(pred, truth):
    return sum(np.unique(truth[pred == c], return_counts=True)[1].max()
               for c in np.unique(pred)) / len(truth)


def main():
    t0 = time.time()
    rows = list(csv.DictReader(open("data/train.csv")))
    texts = [r["text"] or "" for r in rows]
    y = np.array([M2I[r["model"]] for r in rows])
    groups = np.array([r["prompt_id"] for r in rows])
    n = len(rows)
    res = {}

    z = np.load(CACHE)
    lp_dense, lp_ng = z["lp_dense"], z["lp_ng"]
    base = lp_dense if bacc(y, lp_dense) > bacc(y, lp_ng) else lp_ng
    p_base = softmax(base)
    oracle = [np.where(groups == g)[0] for g in dict.fromkeys(groups)]
    p_or = p_base.copy()
    for ix in oracle:
        p_or[ix] = sinkhorn_knopp_transport(p_base[ix], reg=0.10, alpha=0.5,
                                            max_iter=200)
    b_oracle = bacc(y, p_or)
    print(f"no OT {bacc(y, p_base):.4f} | oracle-cluster OT {b_oracle:.4f}")

    # content representation: word 1-2 grams, stopwords removed, no authorial
    # signal at all
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True, max_features=80_000,
                          dtype=np.float32)
    Xc = vec.fit_transform(texts)
    Xc = normalize(Xc)
    print(f"content TF-IDF {Xc.shape}")
    F = normalize(TruncatedSVD(n_components=128, random_state=RNG)
                  .fit_transform(Xc))
    print(f"svd done {time.time()-t0:.0f}s")

    # ---- hard clustering -------------------------------------------------- #
    print("\n[A] agglomerative on content SVD ...")
    rowsA = []
    for th in (0.15, 0.25, 0.35, 0.5, 0.7, 0.9):
        pred = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                       linkage="average",
                                       distance_threshold=th).fit_predict(F)
        sizes = np.bincount(pred)
        q = p_base.copy()
        for c in np.unique(pred):
            ix = np.where(pred == c)[0]
            if len(ix) < 2:
                continue
            q[ix] = sinkhorn_knopp_transport(p_base[ix], reg=0.10, alpha=0.5,
                                             max_iter=200)
        r = {"threshold": th, "n_clusters": int(len(sizes)),
             "purity": float(purity(pred, groups)),
             "median_size": float(np.median(sizes)), "bacc": bacc(y, q)}
        rowsA.append(r)
        print(f"    th={th:.2f} clusters={r['n_clusters']:4d} "
              f"purity={r['purity']:.3f} med={r['median_size']:5.1f} "
              f"bacc={r['bacc']:.4f}")
    res["hard_clustering"] = rowsA

    # ---- soft per-row kNN OT --------------------------------------------- #
    print("\n[B] per-row kNN OT (overlapping neighbourhoods) ...")
    rowsB = []
    sim = cosine_similarity(F)
    np.fill_diagonal(sim, -1.0)
    order = np.argsort(-sim, axis=1)
    for k in (17, 35, 53):
        for alpha in (0.25, 0.5):
            q = p_base.copy()
            for i in range(n):
                nb = np.concatenate([[i], order[i, :k]])
                q[i] = sinkhorn_knopp_transport(p_base[nb], reg=0.10,
                                                alpha=alpha, max_iter=200)[0]
            rowsB.append({"k": k, "alpha": alpha, "bacc": bacc(y, q)})
            print(f"    k={k:3d} alpha={alpha:.2f} bacc={rowsB[-1]['bacc']:.4f}")
    res["knn_ot"] = rowsB

    # ---- oracle neighbourhood size control ------------------------------- #
    print("\n[C] what does OT need? oracle neighbourhoods of size k ...")
    rowsC = []
    rng = np.random.default_rng(RNG)
    for k in (9, 17, 35, 71):
        for alpha in (0.2, 0.5, 1.0):
            q = p_base.copy()
            for ix in oracle:
                for i in ix:
                    nb = rng.choice(ix, size=min(k + 1, len(ix)), replace=False)
                    if i not in nb:
                        nb[0] = i
                    pos = np.where(nb == i)[0][0]
                    q[i] = sinkhorn_knopp_transport(p_base[nb], reg=0.10,
                                                    alpha=alpha,
                                                    max_iter=200)[pos]
            rowsC.append({"k": k, "alpha": alpha, "bacc": bacc(y, q)})
            print(f"    k={k:3d} alpha={alpha:.2f} bacc={rowsC[-1]['bacc']:.4f}")
    res["oracle_partial_cluster"] = rowsC

    res["bacc_no_ot"] = bacc(y, p_base)
    res["bacc_oracle_full_cluster"] = b_oracle
    res["runtime_seconds"] = round(time.time() - t0, 1)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT} in {res['runtime_seconds']}s")


if __name__ == "__main__":
    main()
