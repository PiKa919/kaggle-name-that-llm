"""Fit the ModernBERT / UltraBERT / Pierre blend weights against the final SVC member.

Runs the bench8 transductive stylegram (pool_v1 0.25 + mixed pool 0.10, the best
member found so far: 0.4605) with 3 prompt-grouped folds, but keeps the OOF
decision scores instead of only argmax. Then loads the small public OOF bundle and
measures the marginal gain of each external member over the SVC, with a coarse
weight grid and per-class recalls.

CPU only. No test data is touched; the output is a weight table for v12.
"""
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


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


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
    log(f"pool {name}: {len(d)} rows")
    return d


data_dir = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"]
                if os.path.exists(p + "/train.csv"))
train = pd.read_csv(data_dir + "/train.csv")
y = train.model.map(L2I).values
groups = train.prompt_id.values
texts = train.text.astype(str).tolist()
ids = train.id.astype(str).values

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
log(f"combined pool {len(po_texts)} rows ({n1} v1 + {len(mix)} mix)")

VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7),
         ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]
SPLITS = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups))
scores_oof = np.zeros((len(y), 18), dtype=np.float32)
P_MIXW = 0.10


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
        log(f"fold{f} {view}")
    A, B = hstack(A, format="csr"), hstack(B, format="csr")
    sw = np.r_[np.ones(len(ti)), np.full(n1, 0.25), np.full(len(mix), P_MIXW)]
    clf = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight="balanced").fit(A, yy, sample_weight=sw)
    scores_oof[vi[:, None], clf.classes_[None, :]] = clf.decision_function(B)
    log(f"fold{f} BA {balanced_accuracy_score(y[vi], scores_oof[vi].argmax(1)):.4f}")
    del A, B, clf
    gc.collect()

svc_ba = float(balanced_accuracy_score(y, scores_oof.argmax(1)))
log(f"SVC OOF BA {svc_ba:.5f} minrec {recall_score(y, scores_oof.argmax(1), average=None).min():.3f}")
np.savez("/kaggle/working/svc_oof_scores.npz", scores=scores_oof, y=y, ids=ids, labels=np.array(LABELS))


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def load_oof(path, name):
    d = pd.read_csv(path).set_index("id").loc[ids]
    missing = d.isna().any(axis=1).sum()
    assert missing == 0, (name, missing)
    cols = [c for c in d.columns if c != "fold"]
    # The public files use their own label order (claude-fable-5 sits first among
    # the Claude rows there); select by name into this benchmark's order.
    assert set(cols) == set(LABELS), (name, sorted(set(cols) ^ set(LABELS)))
    log(f"oof {name}: {len(d)} rows, fold col {('fold' in d.columns)}")
    return d[LABELS].values


oof_files = {
    "modA": "oof_modernbert_a.csv", "modB": "oof_modernbert_b.csv", "ultra": "oof_ultrabert.csv",
    "p_linear": "oof_pierre_linear.csv", "p_q08": "oof_pierre_qwen08.csv", "p_q4b": "oof_pierre_qwen4b.csv"}
O = {k: load_oof(one(v), k) for k, v in oof_files.items()}
P = O   # the public files already hold probabilities
MOD = 0.5 * (P["modA"] + P["modB"])
ULTRA = P["ultra"]
PIERRE = 0.30 * P["p_linear"] + 0.25 * P["p_q08"] + 0.45 * P["p_q4b"]

base = softmax(scores_oof)
log(f"member BA: svc {svc_ba:.4f} modA {balanced_accuracy_score(y, P['modA'].argmax(1)):.4f} "
    f"modB {balanced_accuracy_score(y, P['modB'].argmax(1)):.4f} ultra {balanced_accuracy_score(y, ULTRA.argmax(1)):.4f} "
    f"pierre {balanced_accuracy_score(y, PIERRE.argmax(1)):.4f}")


def score(prob):
    pred = np.array(LABELS)[prob.argmax(1)]
    return float(balanced_accuracy_score(y, pred)), float(recall_score(y, pred, average=None).min())


def best_weight(member, lo=0.0, hi=1.5, step=0.1, base_prob=None):
    """Add w * log(member) to log(base), exactly like the production blend."""
    b = base if base_prob is None else base_prob
    lb, lm = np.log(np.clip(b, 1e-12, 1)), np.log(np.clip(member, 1e-12, 1))
    best = (-1, None, None)
    for w in np.arange(lo, hi + 1e-9, step):
        ba, mn = score(softmax(lb + w * lm))
        if ba > best[0]:
            best = (ba, w, mn)
    return dict(ba=round(best[0], 5), w=round(best[1], 3), min_recall=round(best[2], 4))


results = {"svc_ba": round(svc_ba, 5), "members": {}}
for name, M in (("modavg", MOD), ("ultra", ULTRA), ("pierre", PIERRE)):
    results["members"][name] = best_weight(M)
    log(f"OVER svc: {name} best {results['members'][name]}")
# SVC + modavg + ultra joint coarse grid (log space, same as production)
lb, lmod, lult = (np.log(np.clip(X, 1e-12, 1)) for X in (base, MOD, ULTRA))
best_joint = (-1, None)
for wm in np.arange(0.0, 1.01, 0.1):
    for wu in np.arange(0.0, 1.01, 0.1):
        ba, _ = score(softmax(lb + wm * lmod + wu * lult))
        if ba > best_joint[0]:
            best_joint = (ba, (round(wm, 2), round(wu, 2)))
results["joint_svc_mod_ultra"] = {"ba": round(best_joint[0], 5), "weights": best_joint[1]}
log(f"JOINT svc + mod + ultra: {results['joint_svc_mod_ultra']}")

with open("/kaggle/working/svc_oof_blend.json", "w") as fh:
    json.dump(results, fh, indent=1)
log("saved svc_oof_scores.npz and svc_oof_blend.json")
log("done")
