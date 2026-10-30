import gc
import glob
import json
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
                if __import__("os").path.exists(p + "/train.csv"))
train = pd.read_csv(data_dir + "/train.csv")
y = train.model.map(L2I).values
groups = train.prompt_id.values

A = load_pool(one("pool_v1.parquet"), "v1")
B = pd.concat([
    load_pool(one("ntl_pool_v4.parquet"), "v4"),
    load_pool(one("nlm_matched_clean.parquet"), "mc"),
    load_pool(one("ntl_mega_corpus.parquet"), "mega"),
], ignore_index=True).drop_duplicates("text")
B = B.sample(n=min(len(B), len(A)), random_state=42)
print("benchmark pools", len(A), len(B), flush=True)

VIEWS_A = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]
VIEWS_B = [("char", 1, 5, 400000, 1.0), ("word", 1, 3, 300000, 0.7), ("shape", 2, 6, 80000, 0.5), ("function", 1, 4, 100000, 0.6)]


def run(name, pool, views, min_df, folds=3):
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
    oof = np.zeros(len(y), dtype=int)
    for f, (ti, vi) in enumerate(StratifiedGroupKFold(folds, shuffle=True, random_state=42).split(X[:len(y)], y, groups)):
        idx = np.r_[ti, np.arange(len(y), len(y) + len(pool))]
        clf = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight="balanced").fit(X[idx], yy[idx])
        oof[vi] = clf.predict(X[vi])
        print(f"  {name} fold {f}: {balanced_accuracy_score(y[vi], oof[vi]):.4f} {time.time() - T0:.0f}s", flush=True)
    ba = float(balanced_accuracy_score(y, oof))
    rec = recall_score(y, oof, average=None)
    print(f"RESULT {name}: BA {ba:.5f} minrec {rec.min():.3f} medianrec {np.median(rec):.3f} {time.time() - T0:.0f}s", flush=True)
    return {"name": name, "ba": ba, "min_recall": float(rec.min()), "median_recall": float(np.median(rec))}


results = []
results.append(run("A_v1", A, VIEWS_A, 2))
results.append(run("A_mix", B, VIEWS_A, 2))
results.append(run("B_mix", B, VIEWS_B, 3))
print(json.dumps(results, indent=1))
with open("/kaggle/working/svc_bench_results.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(f"done {time.time() - T0:.0f}s", flush=True)
