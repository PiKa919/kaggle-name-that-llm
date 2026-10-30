"""
Honest, prompt-grouped measurement of two candidate levers for v20.

Everything here runs on CPU, no GPU, no Kaggle.  The point is to replace guesses
with numbers before spending an 8.3 h submission run.

Lever A -- the 245k open pool.  Does training the 4-view SVC (the `svc4` member,
weight 1.5 in the 0.660 blend) on the public 254k pool beat training it on the
2,699 official rows only, measured on competition prompts the fit never saw?

Lever B -- a length specialist.  Short passages are the documented weak spot
(<=60 words ~0.27 accuracy vs ~0.46 for >=160 words).  Does a second 4-view SVC
trained only on short pool rows beat the general member on short held-out rows,
and does a log-probability blend of the two beat both?

Protocol
--------
StratifiedGroupKFold(3, shuffle, seed 42) grouped on prompt_id over the 2,699
official rows.  The pool never contains official rows (the pool builder removed
train copies and near-copies; we also drop its own leakage flags).

The TF-IDF vectorizers are fitted ONCE on pool + official text (idf over ~250k
documents); per fold only the SVC is refitted.  This is the same idf-sharing the
public notebooks do, and it cannot leak a label.

Usage:  python experiments/30_pool_and_length_measure.py [--pool_rows N]
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


def shape_view(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_view(t):
    return " ".join(w.lower() if (w.lower() in FUNCTION_WORDS or not w.isalnum()) else "CONTENT"
                    for w in TOKEN_RE.findall(t))


VIEWS = [("char", str, (3, 5), 150_000, 1.0),
         ("word", str, (1, 3), 100_000, 0.7),
         ("shape", shape_view, (2, 6), 40_000, 0.5),
         ("function", function_view, (1, 3), 30_000, 0.5)]


def build_views(fit_texts, other_texts):
    """Fit the four TF-IDF views on fit_texts; return (fit matrix, other matrix)."""
    A, B = [], []
    for analyzer, prep, ngr, maxf, w in VIEWS:
        opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True,
                    min_df=2, max_features=maxf, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        v = TfidfVectorizer(**opts)
        t0 = time.time()
        A.append(v.fit_transform([prep(t) for t in fit_texts]) * w)
        B.append(v.transform([prep(t) for t in other_texts]) * w)
        print(f"    view {analyzer:8s} fit {A[-1].shape[1]:>7d} feats  {time.time() - t0:6.1f}s", flush=True)
    return hstack(A, format="csr"), hstack(B, format="csr")


def logits(model, X):
    d = model.decision_function(X)
    d = d - d.max(1, keepdims=True)
    return d - np.log(np.exp(d).sum(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_rows", type=int, default=245_000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", default="experiments/30_pool_and_length_measure.json")
    a = ap.parse_args()

    tr = pd.read_csv("data/train.csv")
    tr["nw"] = tr.text.str.split().str.len()
    y = tr.model.map(L2I).to_numpy()

    pool = pd.read_parquet("/tmp/pool/ntl_open_pool.parquet",
                           columns=["text", "model", "n_words", "flag_near_train_prompt",
                                    "flag_train_prompt_regen", "flag_gemini37_routing_risk"])
    clean = pool[~pool.flag_near_train_prompt & ~pool.flag_train_prompt_regen
                 & ~pool.flag_gemini37_routing_risk]
    clean = clean[clean.model.isin(LABELS)]
    clean = clean.drop_duplicates(subset=["text"])
    if a.pool_rows < len(clean):
        n_all = len(clean)
        parts = [d.sample(min(len(d), max(1, int(a.pool_rows * len(d) / n_all))), random_state=0)
                 for _, d in clean.groupby("model")]
        clean = pd.concat(parts)
    print(f"pool rows used: {len(clean)}  ({clean.n_words.median():.0f} median words)", flush=True)

    # ---- vectorizers fitted once on pool + official
    print("vectorising pool + official ...", flush=True)
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
    print(f"  final matrices pool {Xp.shape} official {Xt.shape}  ({time.time() - t0:.0f}s)", flush=True)

    py = clean.model.map(L2I).to_numpy()
    short_pool = clean.n_words.to_numpy() <= 60

    sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=42)
    folds = list(sgkf.split(tr, y, groups=tr.prompt_id.to_numpy()))

    res = []
    for k, (itr, iva) in enumerate(folds):
        print(f"\n=== fold {k + 1}/{len(folds)}  train {len(itr)}  val {len(iva)}", flush=True)
        Xtr_o, Xva_o = Xt[itr], Xt[iva]
        ytr, yva = y[itr], y[iva]
        sw_tr = np.ones(len(itr), dtype=np.float32)

        def fit_eval(use_pool, short_only=False, tag=""):
            t1 = time.time()
            if use_pool:
                m = short_pool if short_only else np.ones(len(py), bool)
                X = vstack([Xp[m], Xtr_o], format="csr")
                yy = np.concatenate([py[m], ytr])
                sw = np.concatenate([np.full(int(m.sum()), 0.35, dtype=np.float32), sw_tr])
            else:
                X, yy, sw = Xtr_o, ytr, sw_tr
            svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
            svc.fit(X, yy, sample_weight=sw)
            P = logits(svc, Xva_o)
            pred = P.argmax(1)
            ba = balanced_accuracy_score(yva, pred)
            print(f"  {tag:28s} rows {X.shape[0]:>7d}  BA {ba:.4f}  ({time.time() - t1:.0f}s)", flush=True)
            return dict(tag=tag, rows=int(X.shape[0]), ba=float(ba), P=P, pred=pred)

        r_off = fit_eval(False, tag="svc4 official only")
        r_pool = fit_eval(True, tag="svc4 official+pool")
        r_short = fit_eval(True, short_only=True, tag="svc4 official+shortpool")
        r_short_off = fit_eval(False, short_only=True, tag="svc4 official only (short)")

        # ---- routed comparison on short validation rows
        is_short = tr.nw.to_numpy()[iva] <= 60
        if is_short.sum() > 0:
            ys = yva[is_short]
            for name, P in [("general", r_pool["P"]), ("short-specialist", r_short["P"])]:
                ba = balanced_accuracy_score(ys, P[is_short].argmax(1))
                print(f"  SHORT rows n={int(is_short.sum()):4d}  {name:16s} BA {ba:.4f}", flush=True)
                res.append(dict(fold=k, subset="short", model=name, n=int(is_short.sum()), ba=float(ba)))
            for wgt in (0.5, 1.0):
                P = r_pool["P"].copy()
                P[is_short] = (1 - wgt) * r_pool["P"][is_short] + wgt * r_short["P"][is_short]
                ba = balanced_accuracy_score(ys, P[is_short].argmax(1))
                print(f"  SHORT rows blend w={wgt}  BA {ba:.4f}", flush=True)
                res.append(dict(fold=k, subset="short", model=f"blend{wgt}", n=int(is_short.sum()), ba=float(ba)))
        long_mask = tr.nw.to_numpy()[iva] >= 160
        if long_mask.sum() > 0:
            ba = balanced_accuracy_score(yva[long_mask], r_pool["P"][long_mask].argmax(1))
            print(f"  LONG rows n={int(long_mask.sum()):4d}  general BA {ba:.4f}", flush=True)
            res.append(dict(fold=k, subset="long", model="general", n=int(long_mask.sum()), ba=float(ba)))

        res.append(dict(fold=k, subset="all", model="official only", ba=r_off["ba"]))
        res.append(dict(fold=k, subset="all", model="official+pool", ba=r_pool["ba"]))
        res.append(dict(fold=k, subset="all", model="official+shortpool", ba=r_short["ba"]))
        res.append(dict(fold=k, subset="all", model="official only (short-fit)", ba=r_short_off["ba"]))

    df = pd.DataFrame(res)
    piv = df.pivot_table(index="model", columns="subset", values="ba", aggfunc="mean")
    print("\n==== MEAN BALANCED ACCURACY OVER FOLDS ====")
    print(piv.round(4).to_string())
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
