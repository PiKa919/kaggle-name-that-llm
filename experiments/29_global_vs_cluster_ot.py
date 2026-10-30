"""
Experiment 29: the operator actually in the notebook at 03:25, versus the one
the measurement supports.

Current notebooks/master_sota/submission_sota.py (section 8):

    P = softmax(LOGIT)
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)

That is one global Sinkhorn scaling to a GLOBAL uniform class marginal, at full
strength (no alpha blend).  Two properties make it dangerous and, unlike the
per-cluster operator, impossible to validate on train:

  * train.csv is exactly 150 prompts x 18 models, so the global class marginal
    of any OOF matrix is uniform BY CONSTRUCTION.  A global-uniform projection
    therefore cannot be falsified in CV: it will look fine or good, always.
    AGENTS.md 1.2 says the hidden test mixture is explicitly NOT equal and
    differs between the public and private splits.
  * per-cluster projection is falsifiable, because clusters smaller than 18
    make the uniform column marginal a real (and sometimes wrong) assumption.

So this simulates a non-uniform test mixture -- class-dependent retention,
which is the mechanism AGENTS.md describes -- and scores the candidates on the
rows that survive.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.test_time_calibration import sinkhorn_knopp_transport  # noqa: E402

OUT = "experiments/integration_gate6.json"
K = 18
VALID = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
]


def bacc(y, p):
    return float(balanced_accuracy_score(y, np.asarray(p).argmax(1)))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def global_uniform_balancing(P, iters=100, alpha=1.0):
    """The notebook's operator, with an alpha blend bolted on so the two
    versions are comparable."""
    Q = P.copy()
    target = np.full(P.shape[1], P.shape[0] / P.shape[1])
    for _ in range(iters):
        Q *= target / np.maximum(Q.sum(0), 1e-12)
        Q /= np.maximum(Q.sum(1, keepdims=True), 1e-12)
    R = (1 - alpha) * P + alpha * Q
    return R / np.maximum(R.sum(1, keepdims=True), 1e-12)


def main():
    rows = list(csv.DictReader(open("data/train.csv")))
    M = {m: i for i, m in enumerate(VALID)}
    y = np.array([M[r["model"]] for r in rows])
    g = np.array([r["prompt_id"] for r in rows])
    n = len(rows)
    d = np.load("experiments/cache_oof.npz")
    p_all = softmax(d["lp_dense"])
    clusters = [np.where(g == x)[0] for x in dict.fromkeys(g)]

    # class-dependent retention: the mixture mechanism AGENTS.md 1.2 describes
    schemes = {
        "uniform (train-like)": np.ones(K),
        "mild skew":            np.array([1.0] * 5 + [0.8] * 4 + [0.85, 0.8] +
                                         [1.0] * 5 + [0.85, 0.8]),
        "strong skew":          np.array([1.0] * 5 + [0.55] * 4 + [0.5, 0.45] +
                                         [1.0] * 5 + [0.55, 0.5]),
    }
    out = {}
    for name, w in schemes.items():
        rng = np.random.default_rng(7)
        accs = {"raw": [], "global a=1": [], "global a=0.5": [],
                "cluster a=1": [], "cluster a=0.5": [], "cluster a=0.35": []}
        for _ in range(5):
            keep = []
            for ix in clusters:
                p_keep = np.clip(w[y[ix]], 0.05, 1.0)
                sel = ix[rng.random(len(ix)) < p_keep]
                if len(sel) < 2:
                    sel = rng.choice(ix, size=2, replace=False)
                keep.append(np.sort(sel))
            mask = np.zeros(n, bool)
            for ix in keep:
                mask[ix] = True
            P, yy = p_all[mask], y[mask]

            accs["raw"].append(bacc(yy, P))
            for a in (1.0, 0.5):
                accs[f"global a={a:g}"].append(bacc(yy, global_uniform_balancing(P, alpha=a)))
            for a in (1.0, 0.5, 0.35):
                Q = P.copy()
                orig = np.flatnonzero(mask)      # original index -> row of P
                for ix in keep:
                    pos = np.searchsorted(orig, ix)
                    Q[pos] = sinkhorn_knopp_transport(
                        P[pos], reg=0.10, alpha=a, max_iter=200)
                accs[f"cluster a={a:g}"].append(bacc(yy, Q))
        out[name] = {k: float(np.mean(v)) for k, v in accs.items()}
        print(f"\n{name}  (n_kept={mask.sum()})")
        for k, v in out[name].items():
            print(f"    {k:14s} {v:.4f}")

    # how much does the global operator shift the predicted distribution?
    print("\nclass distribution of the test predictions, uniform scheme:")
    ytrue = np.bincount(y, minlength=K) / n
    print("    true  ", np.round(ytrue, 3))
    print("    argmax", np.round(np.bincount(
        global_uniform_balancing(p_all, alpha=1.0).argmax(1), minlength=K) / n, 3))

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
