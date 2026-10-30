"""Transductive stylegram SVC with pool_v1 + a small mixed-pool share.
Bench5 (non-transductive) showed v1 pw0.25 + mix pw0.10 = 0.4434 vs v1 0.4390,
with worst-class recall 0.073 -> 0.127. Here the same mix is added on top of the
transductive vectorizer (bench6, 0.4538 for v1 alone)."""
import gc
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
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


def load_pool(path, name):
    d = pd.read_parquet(path)
    if "model" not in d:
        d = d.rename(columns={d.columns[-1]: "model"})
    d = d[["text", "model"]].copy()
    d["text"] = d["text"].astype(str)
    d = d[d.model.isin(L2I)]
    d["n"] = d.text.str.split().str.len()
    d = d[(d.n >= 30) & (d.n <= 300)].drop_duplicates("text")
    d["pool"] = name
    print(f"pool {name}: {len(d)} rows", flush=True)
    return d


data_dir = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"]
                if os.path.exists(p + "/train.csv"))
train = pd.read_csv(data_dir + "/train.csv")
y = train.model.map(L2I).values
groups = train.prompt_id.values
texts = train.text.astype(str).tolist()

v1 = load_pool(one("pool_v1.parquet"), "v1")
mix = pd.concat([
    load_pool(one("ntl_pool_v4.parquet"), "v4"),
    load_pool(one("nlm_matched_clean.parquet"), "mc"),
    load_pool(one("ntl_mega_corpus.parquet"), "mega"),
], ignore_index=True).drop_duplicates("text")
mix = mix.sample(n=min(len(mix), len(v1)), random_state=42)
pool = pd.concat([v1, mix], ignore_index=True)
po_texts = pool.text.tolist()
po_y = pool.model.map(L2I).values
n1 = len(v1)
print(f"combined pool {len(po_texts)} rows ({n1} v1 + {len(mix)} mix)", flush=True)

VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]
CONFIGS = [("TM_mix0.10", 0.10), ("TM_mix0.05", 0.05)]
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
    for name, mixw in CONFIGS:
        sw = np.r_[np.ones(len(ti)), np.full(n1, 0.25), np.full(len(mix), mixw)]
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
with open("/kaggle/working/svc_bench8_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
print(f"done {time.time() - T0:.0f}s", flush=True)
