"""
Does the 245k open pool make the *prompt-centred* member (the "twin") strong?

Why this is the interesting lever
---------------------------------
Every public pipeline trains its centred twin head on the 2,699 official rows
(150 prompts).  The open pool has 24,240 prompt groups with >=2 passages
(179,591 rows), i.e. 66x more *grouped* training data for exactly the member type
that exploits within-prompt contrast -- which is the most reliable signal in this
competition (twins, Gaussian group heads and adaptive centring are the only
test-time rules that keep surviving on the LB).

Nobody publishes this: the public notebooks use the pool for the plain SVC, the
TF-IDF member and the Gaussian, never for the centred member.

Protocol
--------
StratifiedGroupKFold(3) grouped on prompt_id over the official rows.
Per fold, four fits on the same four TF-IDF views:
  A  plain SVC      official_train                     (what a no-pool baseline has)
  B  plain SVC      official_train + pool              (svc4, weight 1.5 in the 0.660 blend)
  C  centred SVC    centred official_train             (the twin the public pipeline has)
  D  centred SVC    centred official_train + centred pool   (the proposal)
Scored on the validation prompts, both raw and centred inside their true prompt
(the second is the optimistic bound: it assumes the test clusters are recovered
perfectly; the real pipeline infers them from content).

Usage: python experiments/31_pool_centred_twin.py [--pool_rows N] [--folds K]
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}

FUNCTION_WORDS = frozenset(
    "a an the and or but if while although because as so than that which who whom whose this these those it its "
    "they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from "
    "with without for of by over under between through into about is are was were be been being have has had do "
    "does did can could will would shall should may might must not no nor all any some each every both either "
    "neither other another such only also just very too more most less least here there when where why how what "
    "whether then now still yet even much many few little".split())
TOKEN_RE = re.compile(r"\w+|[^\w\s]")

VIEWS = [("char", str, (3, 5), 150_000, 1.0),
         ("word", str, (1, 3), 100_000, 0.7),
         ("shape", lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t), (2, 6), 40_000, 0.5),
         ("function", lambda t: " ".join(w.lower() if (w.lower() in FUNCTION_WORDS or not w.isalnum()) else "CONTENT"
                                         for w in TOKEN_RE.findall(t)), (1, 3), 30_000, 0.5)]


def logits(model, X):
    d = model.decision_function(X)
    d = d - d.max(1, keepdims=True)
    return d - np.log(np.exp(d).sum(1, keepdims=True))


def centre(M, groups):
    """Subtract each group's mean row (dense-free: sparse row scaling)."""
    M = M.tocsr()
    uniq, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    keep = cnt >= 2
    if not keep.any():
        return M.copy()
    col = np.full(len(cnt), -1)
    col[keep] = np.arange(int(keep.sum()))
    rows = np.flatnonzero(col[inv] >= 0)
    from scipy.sparse import csr_matrix, diags
    Msel = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(M.shape[0], int(keep.sum())))
    means = diags(1.0 / cnt[keep]) @ (Msel.T @ M)
    return (M - Msel @ means).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_rows", type=int, default=80_000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", default="experiments/31_pool_centred_twin.json")
    a = ap.parse_args()

    tr = pd.read_csv("data/train.csv")
    tr["nw"] = tr.text.str.split().str.len()
    y = tr.model.map(L2I).to_numpy()

    pool = pd.read_parquet("/tmp/pool/ntl_open_pool.parquet",
                           columns=["text", "model", "prompt_group", "n_words",
                                    "flag_near_train_prompt", "flag_train_prompt_regen",
                                    "flag_gemini37_routing_risk"])
    clean = pool[~pool.flag_near_train_prompt & ~pool.flag_train_prompt_regen
                 & ~pool.flag_gemini37_routing_risk & pool.model.isin(LABELS)]
    clean = clean.drop_duplicates(subset=["text"])
    # keep only groups of 2..18 so centring is well defined and matches the test-cluster regime
    gs = clean.groupby("prompt_group").size()
    clean = clean[clean.prompt_group.isin(gs[(gs >= 2) & (gs <= 18)].index)]
    if a.pool_rows < len(clean):
        n_all = len(clean)
        parts = [d.sample(min(len(d), max(1, int(a.pool_rows * len(d) / n_all))), random_state=0)
                 for _, d in clean.groupby("model")]
        clean = pd.concat(parts)
    clean = clean.reset_index(drop=True)
    print(f"pool: {len(clean)} rows, {clean.prompt_group.nunique()} groups "
          f"({clean.n_words.median():.0f} median words)", flush=True)

    print("vectorising ...", flush=True)
    t0 = time.time()
    pool_texts = clean.text.tolist()
    off_texts = tr.text.tolist()
    n_pool = len(pool_texts)
    A, B = [], []
    for view, prep, ngr, maxf, w in VIEWS:
        analyzer = "char" if view == "shape" else "word" if view == "function" else view
        opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True,
                    min_df=2, max_features=maxf, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        tv = time.time()
        prepped = [prep(t) for t in pool_texts] + [prep(t) for t in off_texts]
        X = TfidfVectorizer(**opts).fit_transform(prepped)
        A.append(X[:n_pool] * w)
        B.append(X[n_pool:] * w)
        print(f"  view {view:9s} {X.shape[1]:>7d} feats  {time.time() - tv:6.0f}s", flush=True)
        del X, prepped
    Xp = hstack(A, format="csr")
    Xt = hstack(B, format="csr")
    del A, B
    print(f"  pool {Xp.shape}  official {Xt.shape}", flush=True)

    py = clean.model.map(L2I).to_numpy()
    pgroup = clean.prompt_group.to_numpy()
    Xp_c = centre(Xp, pgroup)
    print(f"  centred pool nnz {Xp_c.nnz}", flush=True)

    sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=42)
    folds = list(sgkf.split(tr, y, groups=tr.prompt_id.to_numpy()))
    res = []

    for k, (itr, iva) in enumerate(folds):
        print(f"\n=== fold {k + 1}/{a.folds}", flush=True)
        ytr, yva = y[itr], y[iva]
        Xo_tr, Xo_va = Xt[itr], Xt[iva]
        Xo_tr_c = centre(Xo_tr, tr.prompt_id.to_numpy()[itr])
        Xo_va_c = centre(Xo_va, tr.prompt_id.to_numpy()[iva])
        sw_o = np.ones(len(itr), dtype=np.float32)
        sw_p = np.full(len(py), 0.35, dtype=np.float32)

        def go(tag, X, yy, sw, Xev):
            t1 = time.time()
            svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
            svc.fit(X, yy, sample_weight=sw)
            P = logits(svc, Xev)
            ba = balanced_accuracy_score(yva, P.argmax(1))
            print(f"  {tag:34s} rows {X.shape[0]:>7d}  BA {ba:.4f}  ({time.time() - t1:.0f}s)", flush=True)
            res.append(dict(fold=k, tag=tag, rows=int(X.shape[0]), ba=float(ba)))
            return P, svc

        # A plain, official only
        go("A plain | official", Xo_tr, ytr, sw_o, Xo_va)
        # B plain, official + pool
        P_plain, _ = go("B plain | official+pool", vstack([Xo_tr, Xp], format="csr"),
                        np.concatenate([ytr, py]), np.concatenate([sw_o, sw_p]), Xo_va)
        # C centred twin trained on official only (what the public pipeline has)
        Pc_off, svc_c = go("C centred | official (current twin)", Xo_tr_c, ytr, sw_o, Xo_va_c)
        # D centred twin trained on official + centred pool (the proposal)
        Pc_pool, _ = go("D centred | official+centred pool", vstack([Xo_tr_c, Xp_c], format="csr"),
                        np.concatenate([ytr, py]), np.concatenate([sw_o, sw_p]), Xo_va_c)

        # blend of the plain pool member and the strong twin, on clustered rows
        for wgt in (0.25, 0.5, 1.0):
            Pb = (1 - wgt) * P_plain + wgt * Pc_pool
            ba = balanced_accuracy_score(yva, Pb.argmax(1))
            print(f"  E blend plain+centredpool w={wgt:<4}          BA {ba:.4f}", flush=True)
            res.append(dict(fold=k, tag=f"E blend w={wgt}", rows=int(len(yva)), ba=float(ba)))

    df = pd.DataFrame(res)
    print("\n==== MEAN BALANCED ACCURACY ====")
    print(df.groupby("tag").ba.agg(["mean", "std"]).round(4).to_string())
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
