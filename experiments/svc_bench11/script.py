"""Transductive-vocabulary stylegram SVC: each CV fold's validation texts join the
TF-IDF fit (exactly how the hidden test texts can join it at submission time).
Pool rows are down-weighted to 0.25; sweep the weight again because transductive
IDF may shift the optimum."""
import gc
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

T0 = time.time()
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
L2I = {label: i for i, label in enumerate(LABELS)}
FW = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t):
    return " ".join(w.lower() if w.lower() in FW or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, pattern
    return hits[0]


data_dir = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"]
                if os.path.exists(p + "/train.csv"))
train = pd.read_csv(data_dir + "/train.csv")
y = train.model.map(L2I).values
groups = train.prompt_id.values
texts = train.text.astype(str).tolist()

pool = pd.read_parquet(one("curated_pool.parquet"))
if "split" in pool:
    _drop = int((pool.split == "train_regen").sum())
    pool = pool[pool.split != "train_regen"]
    print(f"dropped train_regen rows (official train prompts): {_drop}", flush=True)
if "register_ok" in pool:
    pool = pool[pool.register_ok.astype(bool)]
pool = pool[["text", "model"]].copy()
pool["text"] = pool["text"].astype(str)
pool = pool[pool.model.isin(L2I)]
pool["n"] = pool.text.str.split().str.len()
pool = pool[(pool.n >= 30) & (pool.n <= 300)].drop_duplicates("text")
po_texts = pool.text.tolist()
po_y = pool.model.map(L2I).values
print(f"pool {len(po_texts)} rows", flush=True)
print("pool class counts", np.bincount(po_y, minlength=18).tolist(), flush=True)

VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]
CONFIGS = [("C_pw0.10", 0.10), ("C_pw0.25", 0.25), ("C_pw0.40", 0.40)]
SPLITS = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups))
oof = {name: np.zeros(len(y), dtype=int) for name, _ in CONFIGS}


def block(fn, an, lo, hi, maxf, weight, fit_docs, val_docs):
    opts = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True,
                min_df=2, max_features=maxf, dtype=np.float32)
    if an == "word":
        opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    vec = TfidfVectorizer(**opts)
    X = vec.fit_transform([fn(t) for t in fit_docs + val_docs])
    return X[:len(fit_docs)] * weight, X[len(fit_docs):] * weight


for f, (ti, vi) in enumerate(SPLITS):
    fit_docs = [texts[i] for i in ti] + po_texts
    val_docs = [texts[i] for i in vi]
    yy = np.r_[y[ti], po_y]
    A, B = [], []
    for view, lo, hi, maxf, w in VIEWS:
        fn = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
        an = "char" if view == "shape" else "word" if view == "function" else view
        a, b = block(fn, an, lo, hi, maxf, w, fit_docs, val_docs)
        A.append(a)
        B.append(b)
        print(f"  fold{f} {view}: {time.time() - T0:.0f}s", flush=True)
    A, B = hstack(A, format="csr"), hstack(B, format="csr")
    for name, pw in CONFIGS:
        sw = np.r_[np.ones(len(ti)), np.full(len(po_texts), pw)]
        clf = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight="balanced").fit(A, yy, sample_weight=sw)
        oof[name][vi] = clf.predict(B)
        print(f"  {name} fold{f}: {balanced_accuracy_score(y[vi], oof[name][vi]):.4f} {time.time() - T0:.0f}s", flush=True)
    del A, B
    gc.collect()

results = []
for name, _ in CONFIGS:
    ba = float(balanced_accuracy_score(y, oof[name]))
    rec = recall_score(y, oof[name], average=None)
    print(f"RESULT {name}: BA {ba:.5f} minrec {rec.min():.3f} medianrec {np.median(rec):.3f}", flush=True)
    results.append({"name": name, "ba": ba, "min_recall": float(rec.min()), "median_recall": float(np.median(rec))})
with open("/kaggle/working/svc_bench11_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
print(f"done {time.time() - T0:.0f}s", flush=True)
