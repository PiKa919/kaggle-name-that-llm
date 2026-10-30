"""
Experiment 21: honest out-of-fold auditors.

E6  Pairwise corrections, fit per-fold (no leakage): does +0.10*delta help or hurt?
E7  Empirical optimal typo weight w*(n) by length bin (bacc and NLL).
E8  NLL-optimal temperatures T* for the n-gram view (check the sign of 0.80 / 0.86).
E9  Per-class heterogeneity of the optimal view weight (is a static 85/15 defensible?).
E10 Transform-time measurement at 4000 rows.
"""
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from src.config import MODEL_TO_IDX, MODEL_TO_LAB, IDX_TO_MODEL, VALID_18_MODELS

K = len(VALID_18_MODELS)
OUT = {}


def log(*a):
    print(*a, flush=True)


def bacc(y, p):
    return float(balanced_accuracy_score(y, p))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


FUNCTION_WORDS = set((
    "a an the and or but if while although because as so than that which who whom whose this these "
    "those it its they them their theirs he him his she her hers we us our ours you your yours i me my "
    "mine in on at to from with without for of by over under between through into about is are was "
    "were be been being have has had do does did can could will would shall should may might must not "
    "no nor all any some each every both either neither other another such only also just very too more "
    "most less least here there when where why how what whether then now still yet even much many few little"
).split())


def shape_transform(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_transform(t):
    return " ".join(
        w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
        for w in re.findall(r"\w+|[^\w\s]", t)
    )


df = pd.read_csv("data/train.csv")
df["text"] = df["text"].fillna("")
y = np.array([MODEL_TO_IDX[m] for m in df["model"]])
groups = df["prompt_id"].values
wc = df["text"].str.split().str.len().values
lab = np.array([MODEL_TO_LAB[m] for m in df["model"]])
N = len(df)

try:
    X_all = load_npz("experiments/cache_X_all.npz")
    views = None
    log("[cache] loaded X_all")
except Exception:
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s, "char"),
        ("word", 1, 2, 120_000, 0.8, lambda s: s, "word"),
        ("shape", 2, 5, 40_000, 0.5, shape_transform, "char"),
        ("function", 1, 2, 30_000, 0.5, function_transform, "word"),
    ]
    mats, fitted = [], []
    for vname, lo, hi, mf, w, prep, an in views:
        pat = r"(?u)\b\w+\b|[^\w\s]" if an == "word" else None
        vec = TfidfVectorizer(analyzer=an, ngram_range=(lo, hi), max_features=mf, lowercase=False,
                              sublinear_tf=True, min_df=2, token_pattern=pat)
        mats.append(vec.fit_transform([prep(s) for s in df["text"]]) * w)
        fitted.append((vec, prep))
    X_all = hstack(mats, format="csr")
    save_npz("experiments/cache_X_all.npz", X_all)
    # E10: transform cost for 4000 docs
    fake = df["text"].tolist()
    fake = (fake * 2)[:4000]
    t = time.time()
    for vec, prep in fitted:
        _ = vec.transform([prep(s) for s in fake])
    OUT["e10_transform_4000_s"] = time.time() - t
    log(f"[E10] 4-view transform on 4000 docs: {OUT['e10_transform_4000_s']:.2f}s")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(df, y, groups=groups))
fold_id = np.zeros(N, dtype=int)
P_ng = np.zeros((N, K))
try:
    P_ng = np.load("experiments/cache_P_ng.npy")
    log("[cache] loaded P_ng")
except Exception:
    for f, (tr, va) in enumerate(folds):
        fold_id[va] = f
        clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=42 + f)
        clf.fit(X_all[tr], y[tr])
        P_ng[va] = clf.predict_proba(X_all[va])
    np.save("experiments/cache_P_ng.npy", P_ng)
log(f"[base] OOF bacc ng = {bacc(y, P_ng.argmax(1)):.4f}")

OBS = np.stack([
    df["text"].str.contains("—", regex=False).values,
    df["text"].str.contains(" — ", regex=False).values,
    df["text"].str.contains("[“”]", regex=True).values,
    df["text"].str.contains("[‘’]", regex=True).values,
], axis=1).astype(float)

# empirical typo profiles fitted per fold (no test leakage)
PAIRS = [
    ("claude-opus-4-6", "claude-opus-5"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("gemini-3.6-flash", "gemini-3.7-flash"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("glm-5.3", "glm-5.3-flash"),
    ("glm-5.3", "claude-opus-4-6"),
]

TYPO_ALPHA, TYPO_BETA = 1.0, 1.0  # Dirichlet/Beta posterior smoothing


def fit_typo_profiles(idx_train):
    """Empirical Bernoulli rates with Beta(1,1) posterior mean smoothing."""
    prof = np.zeros((K, 4))
    for k in range(K):
        m = idx_train[y[idx_train] == k]
        if len(m) == 0:
            prof[k] = 0.5
        else:
            s = OBS[m].sum(0)
            n = len(m)
            prof[k] = (s + TYPO_ALPHA) / (n + TYPO_ALPHA + TYPO_BETA)
    return prof


def typo_loglik(obs, prof, floor=0.02):
    """Bernoulli log-likelihood with a likelihood floor (robust to OOD typography)."""
    p = np.clip(prof, floor, 1 - floor)
    return obs @ np.log(p).T + (1 - obs) @ np.log(1 - p).T


log_p_ng = np.log(np.clip(P_ng, 1e-9, 1.0)) / 0.80

# ---------------- E6: honest per-fold pairwise corrections ----------------
log("\n=== E6: pairwise correction, fit per fold (honest) ===")
delta_oof = {p: np.zeros(N) for p in PAIRS}
for f, (tr, va) in enumerate(folds):
    for a, b in PAIRS:
        i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
        m = tr[(y[tr] == i) | (y[tr] == j)]
        yp = (y[m] == j).astype(int)
        c = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
        c.fit(X_all[m], yp)
        pr = np.clip(c.predict_proba(X_all[va])[:, 1], 1e-4, 1 - 1e-4)
        delta_oof[(a, b)][va] = np.clip(np.log(pr / (1 - pr)), -2.5, 2.5)

P_ng_base = softmax(log_p_ng, 1)

# corrected typo view using per-fold profiles
typo_loglik_oof = np.zeros((N, K))
for f, (tr, va) in enumerate(folds):
    prof = fit_typo_profiles(tr)
    typo_loglik_oof[va] = typo_loglik(OBS[va], prof)
log_p_typo_oof = np.log(np.clip(softmax(typo_loglik_oof, 1), 1e-9, 1.0))

res = []
for w in [0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45]:
    lf = (1 - w) * log_p_ng + w * log_p_typo_oof
    res.append({"w_typo": w, "bacc": bacc(y, lf.argmax(1))})
e7 = pd.DataFrame(res)
log("typo weight sweep (per-fold empirical profiles, no length gate):")
log(e7.round(4).to_string(index=False))
OUT["e7_weight_sweep"] = e7.round(4).to_dict("records")

for w_pair in [0.0, 0.05, 0.10, 0.20]:
    lf = log_p_ng.copy()
    for (a, b), d in delta_oof.items():
        lf[:, MODEL_TO_IDX[b]] += w_pair * d
        lf[:, MODEL_TO_IDX[a]] -= w_pair * d
    log(f"pairwise weight {w_pair:.2f}: OOF bacc {bacc(y, lf.argmax(1)):.4f} "
        f"argmax-changed {int((lf.argmax(1) != log_p_ng.argmax(1)).sum())}")
    OUT.setdefault("e6", []).append({
        "w_pair": w_pair,
        "bacc": bacc(y, lf.argmax(1)),
        "n_changed": int((lf.argmax(1) != log_p_ng.argmax(1)).sum()),
    })

# ---------------- E8: NLL-optimal temperature ----------------
log("\n=== E8: temperature sign check ===")


def nll_at(T):
    p = softmax(np.log(np.clip(P_ng, 1e-12, 1.0)) / T, 1)
    return log_loss(y, p, labels=list(range(K)))


grid = np.linspace(0.3, 3.0, 55)
nlls = [nll_at(t) for t in grid]
Tstar = grid[int(np.argmin(nlls))]
log(f"T=0.80 (as used in submission): NLL {nll_at(0.80):.4f}, bacc {bacc(y, softmax(log_p_ng,1).argmax(1)):.4f}")
log(f"T* by NLL minimization        : {Tstar:.2f} (NLL {min(nlls):.4f})")
log(f"T=1.00                        : NLL {nll_at(1.0):.4f}")
OUT["e8"] = {"T_used": 0.80, "T_star": float(Tstar), "nll_T080": nll_at(0.80), "nll_Tstar": float(min(nlls)),
             "nll_T1": nll_at(1.0)}
log("If T* > 1 the probabilities are UNDER-confident and 0.80 is the wrong sign; "
    "if T* < 1 they are over-confident and 0.80 is defensible but un-fitted.")

# ---------------- E7b: empirical w*(n) by length bin ----------------
log("\n=== E7b: empirical optimal typo weight by length bin ===")
bins = [(0, 60), (60, 80), (80, 100), (100, 140), (140, 300)]
rows = []
ws = np.linspace(0, 0.9, 19)
for lo, hi in bins:
    m = (wc >= lo) & (wc < hi)
    if m.sum() < 60:
        continue
    bb = [bacc(y[m], ((1 - w) * log_p_ng[m] + w * log_p_typo_oof[m]).argmax(1)) for w in ws]
    rows.append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "w_star": float(ws[int(np.argmax(bb))]),
                 "bacc_w0": float(bb[0]), "bacc_w_star": float(max(bb)),
                 "bacc_w045": float(bb[np.argmin(np.abs(ws - 0.45))]),
                 "bacc_w015": float(bb[np.argmin(np.abs(ws - 0.15))])})
e7b = pd.DataFrame(rows)
log(e7b.round(4).to_string(index=False))
OUT["e7b"] = e7b.round(4).to_dict("records")

# ---------------- E9: per-class heterogeneity ----------------
log("\n=== E9: per-class optimal w (static 0.15/0.45 defensible?) ===")
rows = []
for k in range(K):
    m = y == k
    if m.sum() < 30:
        continue
    bb = [float((((1 - w) * log_p_ng[m] + w * log_p_typo_oof[m]).argmax(1) == k).mean()) for w in ws]
    rows.append({"model": VALID_18_MODELS[k], "n": int(m.sum()),
                 "w_star": float(ws[int(np.argmax(bb))]), "recall_at_w0": float(bb[0]),
                 "recall_at_w_star": float(max(bb))})
e9 = pd.DataFrame(rows).sort_values("w_star")
log(e9.round(4).to_string(index=False))
OUT["e9"] = e9.round(4).to_dict("records")
log(f"\nw_star spread across classes: min {e9.w_star.min():.2f} max {e9.w_star.max():.2f} "
    f"- a single global scalar cannot serve all 18 classes.")

with open("experiments/invalidation_evidence2.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
log("\nwrote experiments/invalidation_evidence2.json")
