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

pool = pd.read_parquet(one("pool_v1.parquet"))
pool = pool.rename(columns={"model": "model"})[["text", "model"]].copy()
pool["text"] = pool["text"].astype(str)
pool = pool[pool.model.isin(L2I)]
pool["n"] = pool.text.str.split().str.len()
pool = pool[(pool.n >= 30) & (pool.n <= 300)].drop_duplicates("text")
print(f"pool v1: {len(pool)} rows", flush=True)

VIEWS_A = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]
VIEWS_B = [("char", 1, 5, 400000, 1.0), ("word", 1, 3, 300000, 0.7), ("shape", 2, 6, 80000, 0.5), ("function", 1, 4, 100000, 0.6)]


def run(name, views, min_df, pool_weight, folds=3):
    tr_texts = train.text.astype(str).tolist()
    po_texts = pool.text.tolist()
    blocks = []
    for view, lo, hi, maxf, w in views:
        prep = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "function" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True,
                    min_df=min_df, max_features=maxf, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts).fit(tr_texts + po_texts)
        blocks.append(vstack([vec.transform(tr_texts) * w, vec.transform(po_texts) * w], format="csr"))
        print(f"  {name} {view}: {len(vec.vocabulary_)} feats {time.time() - T0:.0f}s", flush=True)
    X = hstack(blocks, format="csr")
    del blocks
    gc.collect()
    yy = np.r_[y, pool.model.map(L2I).values]
    sw = np.r_[np.ones(len(y)), np.full(len(pool), pool_weight)]
    oof = np.zeros(len(y), dtype=int)
    for f, (ti, vi) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=42).split(X[:len(y)], y, groups)):
        idx = np.r_[ti, np.arange(len(y), len(y) + len(pool))]
        clf = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight="balanced").fit(X[idx], yy[idx], sample_weight=sw[idx])
        oof[vi] = clf.predict(X[vi])
        print(f"  {name} fold {f}: {balanced_accuracy_score(y[vi], oof[vi]):.4f} {time.time() - T0:.0f}s", flush=True)
    ba = float(balanced_accuracy_score(y, oof))
    rec = recall_score(y, oof, average=None)
    print(f"RESULT {name}: BA {ba:.5f} minrec {rec.min():.3f} medianrec {np.median(rec):.3f} {time.time() - T0:.0f}s", flush=True)
    return {"name": name, "ba": ba, "min_recall": float(rec.min()), "median_recall": float(np.median(rec))}


results = []
results.append(run("A_pw0.10", VIEWS_A, 2, 0.10))
results.append(run("A_pw0.05", VIEWS_A, 2, 0.05))
results.append(run("B_pw0.25", VIEWS_B, 3, 0.25))
results.append(run("A_pw0", VIEWS_A, 2, 1e-9))
print(json.dumps(results, indent=1))
with open("/kaggle/working/svc_bench3_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"done {time.time() - T0:.0f}s", flush=True)
