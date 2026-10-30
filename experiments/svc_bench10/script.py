"""Family-gate experiment on the transductive SVC member.

Full 18-class SVC first. Rows whose argmax falls in a family (Claude/GPT/Gemini/
GLM) are re-ranked by a specialist trained only on that family's rows (train +
pool, same weights and features). The gate can only change a prediction inside
the predicted family, so non-family classes are never touched.

Reported on 3-fold grouped OOF: base BA, gated BA, and per-family within-BA.
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
FAMILIES = {
    "claude": ["claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5"],
    "gpt": ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"],
    "gemini": ["gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "glm": ["glm-5.3", "glm-5.3-flash"],
}
FAM_IDX = {f: np.array([L2I[l] for l in labs]) for f, labs in FAMILIES.items()}


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

pool = pd.read_parquet(one("pool_v1.parquet"))
pool = pool[["text", "model"]].copy()
pool["text"] = pool["text"].astype(str)
pool = pool[pool.model.isin(L2I)]
pool["n"] = pool.text.str.split().str.len()
pool = pool[(pool.n >= 30) & (pool.n <= 300)].drop_duplicates("text")
po_texts = pool.text.tolist()
po_y = pool.model.map(L2I).values
print(f"pool {len(po_texts)} rows", flush=True)

VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]
PW = 0.25
SPLITS = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups))
base_oof = np.full(len(y), -1, dtype=int)
gate_oof = np.full(len(y), -1, dtype=int)
fam_rep = {f: {"n": 0, "changed": 0, "correct_base": 0, "correct_gate": 0} for f in FAMILIES}


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
    A, B = hstack(A, format="csr"), hstack(B, format="csr")
    sw = np.r_[np.ones(len(ti)), np.full(len(po_texts), PW)]
    full = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight="balanced").fit(A, yy, sample_weight=sw)
    pred = full.predict(B)
    base_oof[vi] = pred
    gated = pred.copy()
    for fam, idx in FAM_IDX.items():
        idx = np.asarray(idx)
        mask_fit = np.isin(yy, idx)
        mask_val = np.isin(pred, idx)
        if mask_fit.sum() == 0 or mask_val.sum() == 0:
            continue
        local = {c: j for j, c in enumerate(idx)}
        y_loc = np.array([local.get(v, -1) for v in yy[mask_fit]])
        keep = y_loc >= 0
        spec = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight="balanced").fit(
            A[mask_fit][keep], y_loc[keep], sample_weight=sw[mask_fit][keep])
        sp = idx[spec.predict(B[mask_val])]
        gated[mask_val] = sp
        fam_rep[fam]["n"] += int(mask_val.sum())
        fam_rep[fam]["changed"] += int((sp != pred[mask_val]).sum())
        fam_rep[fam]["correct_base"] += int((pred[mask_val] == y[vi][mask_val]).sum())
        fam_rep[fam]["correct_gate"] += int((sp == y[vi][mask_val]).sum())
        print(f"  fold{f} gate[{fam}]: rows {int(mask_val.sum())} changed {fam_rep[fam]['changed']} {time.time()-T0:.0f}s", flush=True)
    gate_oof[vi] = gated
    print(f"  fold{f} base {balanced_accuracy_score(y[vi], pred):.4f} gated {balanced_accuracy_score(y[vi], gated):.4f}", flush=True)
    del A, B
    gc.collect()

base_ba = float(balanced_accuracy_score(y, base_oof))
gate_ba = float(balanced_accuracy_score(y, gate_oof))
print(f"BASE BA {base_ba:.5f}  GATED BA {gate_ba:.5f}  delta {gate_ba-base_ba:+.5f}", flush=True)
for fam, r in fam_rep.items():
    print(f"  {fam}: rows {r['n']} changed {r['changed']} base_correct {r['correct_base']} gate_correct {r['correct_gate']}", flush=True)
for name, oof in [("base", base_oof), ("gated", gate_oof)]:
    rec = recall_score(y, oof, average=None)
    print(f"{name} min {rec.min():.3f} median {np.median(rec):.3f}", flush=True)
res = {"base_ba": base_ba, "gate_ba": gate_ba, "delta": gate_ba - base_ba, "families": fam_rep}
with open("/kaggle/working/svc_bench10_results.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
print(f"done {time.time() - T0:.0f}s", flush=True)
