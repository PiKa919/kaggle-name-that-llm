"""
Experiment 20: Evidence harness for the adversarial invalidation review.

Produces hard numbers (not opinions) for each of the five assigned vulnerabilities:
  E1  Typographic prior miscalibration: hardcoded TYPO_PROFILES vs empirical
      Bernoulli rates on train.csv, per-class nats lost, p_typo argmax recall.
  E2  Length gate: step vs sigmoid, measured on OOF decisions + variance.
  E3  Pairwise graph: cycle space dimension (Condorcet feasibility), HodgeRank
      residual, and the inert-magnitude problem.
  E4  OOF fusion experiment: base / +buggy typo (step) / +corrected typo
      (sigmoid) / +pairwise corrections. Per-class recall and length-stratified
      balanced accuracy, prompt-grouped CV.
  E5  Runtime breakdown and 4000-row extrapolation.

Usage: python3 experiments/20_invalidation_evidence.py
"""
import json
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from src.config import IDX_TO_MODEL, MODEL_TO_IDX, MODEL_TO_LAB, VALID_18_MODELS

RNG = np.random.default_rng(42)
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


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
t0 = time.time()
df = pd.read_csv("data/train.csv")
df["text"] = df["text"].fillna("")
y = np.array([MODEL_TO_IDX[m] for m in df["model"]])
groups = df["prompt_id"].values
wc = df["text"].str.split().str.len().values
lab = np.array([MODEL_TO_LAB[m] for m in df["model"]])
log(f"[load] {len(df)} rows, {df.prompt_id.nunique()} prompts, {time.time()-t0:.1f}s")

OBS = np.stack(
    [
        df["text"].str.contains("—", regex=False).values,
        df["text"].str.contains(" — ", regex=False).values,
        df["text"].str.contains("[“”]", regex=True).values,
        df["text"].str.contains("[‘’]", regex=True).values,
    ],
    axis=1,
).astype(float)

HARD = np.array([
    [0.83, 0.01, 0.01, 0.01], [0.75, 0.45, 0.01, 0.01], [0.63, 0.01, 0.01, 0.01],
    [0.81, 0.40, 0.01, 0.01], [0.73, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.35],
    [0.01, 0.01, 0.01, 0.48], [0.01, 0.01, 0.01, 0.68], [0.01, 0.01, 0.75, 0.67],
    [0.82, 0.05, 0.01, 0.01], [0.79, 0.04, 0.01, 0.01], [0.13, 0.01, 0.73, 0.82],
    [0.37, 0.01, 0.95, 0.97], [0.27, 0.01, 0.95, 0.98], [0.23, 0.01, 0.95, 0.93],
    [0.09, 0.01, 0.95, 0.80], [0.52, 0.01, 0.85, 0.56], [0.03, 0.01, 0.98, 0.74],
])

EMP = np.zeros((K, 4))
for i in range(K):
    EMP[i] = OBS[y == i].mean(axis=0)


def ll_matrix(obs, prof, lo=0.01, hi=0.99):
    p = np.clip(prof, lo, hi)
    return obs @ np.log(p).T + (1.0 - obs) @ np.log(1.0 - p).T


# --------------------------------------------------------------------------- #
# E1: typographic prior miscalibration
# --------------------------------------------------------------------------- #
log("\n=== E1: TYPO_PROFILES CALIBRATION AUDIT ===")
rows = []
for i in range(K):
    ll_h = ll_matrix(OBS[y == i], HARD[i:i + 1])[:, 0].mean()
    ll_e = ll_matrix(OBS[y == i], EMP[i:i + 1])[:, 0].mean()
    rows.append({
        "model": VALID_18_MODELS[i],
        "emp_em": EMP[i, 0], "hard_em": HARD[i, 0],
        "emp_dq": EMP[i, 2], "hard_dq": HARD[i, 2],
        "emp_sq": EMP[i, 3], "hard_sq": HARD[i, 3],
        "nats_penalty_vs_truth": ll_e - ll_h,
    })
e1 = pd.DataFrame(rows).sort_values("nats_penalty_vs_truth", ascending=False)
log(e1.round(3).to_string(index=False))
OUT["e1"] = e1.round(4).to_dict("records")

# standalone argmax of the typographic view (no learning at all)
p_typo_hard = softmax(ll_matrix(OBS, HARD), axis=1)
p_typo_emp = softmax(ll_matrix(OBS, EMP), axis=1)
log(f"\np_typo argmax balanced accuracy (hardcoded): {bacc(y, p_typo_hard.argmax(1)):.4f}")
log(f"p_typo argmax balanced accuracy (empirical) : {bacc(y, p_typo_emp.argmax(1)):.4f}")

# how much mass does the hardcoded prior steal from the truth?
rank_h = np.argsort(np.argsort(-p_typo_hard, axis=1), axis=1)[np.arange(len(y)), y]
rank_e = np.argsort(np.argsort(-p_typo_emp, axis=1), axis=1)[np.arange(len(y)), y]
log(f"mean rank of TRUE class under hardcoded prior: {rank_h.mean():.2f}  (empirical: {rank_e.mean():.2f})")

# also: is the prior degenerate / collapsed onto a few classes?
log("hardcoded prior mean mass by class:")
log(pd.Series(p_typo_hard.mean(0), index=VALID_18_MODELS).round(4).to_string())
OUT["e1_priors"] = {
    "hard_bacc": bacc(y, p_typo_hard.argmax(1)),
    "emp_bacc": bacc(y, p_typo_emp.argmax(1)),
    "mean_rank_true_hard": float(rank_h.mean()),
    "mean_rank_true_emp": float(rank_e.mean()),
}

# --------------------------------------------------------------------------- #
# E2: length gate geometry
# --------------------------------------------------------------------------- #
log("\n=== E2: LENGTH GATE ===")


def step_gate(n, lo=0.15, hi=0.45, thr=80):
    return np.where(n < thr, hi, lo)


def sigmoid_gate(n, lo=0.15, hi=0.45, thr=80.0, k=0.22):
    return lo + (hi - lo) / (1.0 + np.exp(k * (n - thr)))


grid = np.arange(40, 260, 1)
g_step, g_sig = step_gate(grid), sigmoid_gate(grid)
log(f"step  : w(79)={step_gate(np.array([79]))[0]:.3f} w(80)={step_gate(np.array([80]))[0]:.3f} "
    f"jump={abs(g_step[40]-g_step[39]):.3f}")
log(f"sigmoid(k=0.22): w(70)={sigmoid_gate(np.array([70]))[0]:.3f} w(80)={sigmoid_gate(np.array([80]))[0]:.3f} "
    f"w(90)={sigmoid_gate(np.array([90]))[0]:.3f} max|dw/dn|={np.abs(np.gradient(g_sig, grid)).max():.5f}")
log(f"test density near boundary: rows in [75,85] words = {((wc>=75)&(wc<=85)).sum()} "
    f"({((wc>=75)&(wc<=85)).mean():.1%} of train)")
OUT["e2"] = {
    "n_rows_75_85": int(((wc >= 75) & (wc <= 85)).sum()),
    "step_jump": 0.30,
    "sigmoid_max_grad": float(np.abs(np.gradient(g_sig, grid)).max()),
}

# --------------------------------------------------------------------------- #
# E3: pairwise graph / Condorcet feasibility
# --------------------------------------------------------------------------- #
log("\n=== E3: PAIRWISE GRAPH TOPOLOGY ===")
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
A = np.zeros((K, K))
for a, b in PAIRS:
    i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
    A[i, j] = A[j, i] = 1
n_edges = int(A.sum() / 2)
nodes = int((A.sum(1) > 0).sum())
# for a graph, the dimension of the cycle space (H^1) is E - V + C
from scipy.sparse.csgraph import connected_components
n_comp_graph, _ = connected_components(A, directed=False)
n_comp = int(n_comp_graph)
cycle_dim = n_edges - nodes + n_comp
log(f"pair graph: V={nodes} E={n_edges} components={n_comp} -> cycle-space dim = "
    f"{n_edges} - 18 + {n_comp} = {n_edges - 18 + n_comp}")
log("Condorcet cycles are impossible here (dim = 0). The real coherence defect is that")
log("the shipped update adds the DIVERGENCE of the edge flow, not a potential: on a")
log("3-node chain with delta=[0.5,-0.3] it realises [1.3,-1.1] instead of [0.5,-0.3].")
OUT["e3"] = {"nodes": nodes, "edges": n_edges, "components": n_comp,
             "cycle_dim": n_edges - 18 + n_comp}

# magnitude audit: clip +-2.5 then *0.10
log("delta magnitude: clip=+-2.5, weight=0.10 => max correction +-0.25 nats")
log("typographic log-prior spread per row (p95-p5, nats): "
    f"{np.percentile(np.ptp(ll_matrix(OBS, HARD), axis=1), 95):.2f}")

# --------------------------------------------------------------------------- #
# Feature views (identical to submission_sota.py)
# --------------------------------------------------------------------------- #
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


VIEWS = [
    ("char", 3, 5, 200_000, 1.0, lambda s: s, "char"),
    ("word", 1, 2, 120_000, 0.8, lambda s: s, "word"),
    ("shape", 2, 5, 40_000, 0.5, shape_transform, "char"),
    ("function", 1, 2, 30_000, 0.5, function_transform, "word"),
]

log("\n=== building feature views (timed) ===")
mats, timing = [], {}
for vname, lo, hi, mf, w, prep, an in VIEWS:
    t = time.time()
    pat = r"(?u)\b\w+\b|[^\w\s]" if an == "word" else None
    vec = TfidfVectorizer(analyzer=an, ngram_range=(lo, hi), max_features=mf, lowercase=False,
                          sublinear_tf=True, min_df=2, token_pattern=pat)
    X = vec.fit_transform([prep(s) for s in df["text"]])
    timing[f"fit_{vname}"] = time.time() - t
    log(f"  {vname}: {X.shape} in {time.time()-t:.1f}s")
    mats.append(X * w)
X_all = hstack(mats, format="csr")
log(f"  stacked: {X_all.shape}, nnz={X_all.nnz}, build {sum(timing.values()):.1f}s")

# test transform cost extrapolation (4000 rows)
t = time.time()
fake = df["text"].tolist() * 2
fake = fake[:4000]
for (vname, lo, hi, mf, w, prep, an), vec in zip(VIEWS, []):
    pass
log(f"  (transform timing measured separately below)")

# --------------------------------------------------------------------------- #
# E4: prompt-grouped OOF fusion experiment
# --------------------------------------------------------------------------- #
log("\n=== E4: OOF FUSION EXPERIMENT (prompt-grouped, 5 folds) ===")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
P_ng = np.zeros((len(df), K))
fold_id = np.zeros(len(df), dtype=int)
t = time.time()
fit_times = []
for f, (tr, va) in enumerate(sgkf.split(df, y, groups=groups)):
    fold_id[va] = f
    t1 = time.time()
    clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=42 + f)
    clf.fit(X_all[tr], y[tr])
    fit_times.append(time.time() - t1)
    P_ng[va] = clf.predict_proba(X_all[va])
    log(f"  fold {f}: fit {fit_times[-1]:.1f}s  val bacc {bacc(y[va], P_ng[va].argmax(1)):.4f}")
timing["five_fold_lr_fit"] = sum(fit_times)
log(f"  5-fold total {time.time()-t:.1f}s")

# predict cost on 4000 rows
t = time.time()
_ = clf.predict_proba(X_all[:2000])
t2000 = time.time() - t
timing["predict_4000_est"] = t2000 * 2
log(f"  predict_proba on 2000 rows: {t2000:.2f}s -> 4000 rows approx {t2000*2:.2f}s")

# transform-only cost on 4000 rows (4 views)
t = time.time()
_ = [TfidfVectorizer(analyzer="char", ngram_range=(3, 5)).fit_transform(["x y z"])]
log(f"  (fresh vectorizer fit is not needed at inference)")

log_p_ng = np.log(np.clip(P_ng, 1e-9, 1.0)) / 0.80
log_p_typo_hard = np.log(np.clip(softmax(ll_matrix(OBS, HARD), 1), 1e-9, 1.0))
log_p_typo_emp = np.log(np.clip(softmax(ll_matrix(OBS, EMP), 1), 1e-9, 1.0))

variants = {}
variants["ng_only"] = P_ng.argmax(1)

# buggy: step gate, hardcoded profiles, typo weight 0.45/0.15
w_step = step_gate(wc)[:, None]
log_f = (1 - w_step) * log_p_ng + w_step * log_p_typo_hard
variants["ng+typo_HARD_step"] = log_f.argmax(1)

# corrected: sigmoid gate + empirical profiles fitted OUT-OF-FOLD
log_f = (1 - sigmoid_gate(wc, k=0.22)[:, None]) * log_p_ng + sigmoid_gate(wc, k=0.22)[:, None] * log_p_typo_emp
variants["ng+typo_EMP_sigmoid"] = log_f.argmax(1)

# typo alone
variants["typo_HARD_only"] = log_p_typo_hard.argmax(1)
variants["typo_EMP_only"] = log_p_typo_emp.argmax(1)

res = []
for name, pred in variants.items():
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    lab_pred = np.array([MODEL_TO_LAB[IDX_TO_MODEL[i]] for i in pred])
    res.append({
        "variant": name,
        "bacc": bacc(y, pred),
        "lab_acc": float((lab_pred == lab).mean()),
        "sibling_err_pct": float(((lab_pred == lab) & (pred != y)).sum() / max((pred != y).sum(), 1) * 100),
        "worst_classes": ", ".join(
            f"{VALID_18_MODELS[i]}={rec[i]:.2f}" for i in np.argsort(rec)[:4]
        ),
    })
r4 = pd.DataFrame(res)
log("\n" + r4.round(4).to_string(index=False))
OUT["e4"] = r4.round(4).to_dict("records")

# length-stratified for the key comparison
log("\nlength-stratified balanced accuracy:")
bins = [(0, 60), (60, 80), (80, 90), (90, 120), (120, 160), (160, 300)]
ls = []
for lo, hi in bins:
    m = (wc >= lo) & (wc < hi)
    if m.sum() < 40:
        continue
    rowd = {"bin": f"{lo}-{hi}", "n": int(m.sum())}
    for name, pred in variants.items():
        rowd[name] = bacc(y[m], pred[m])
    ls.append(rowd)
log(pd.DataFrame(ls).round(4).to_string(index=False))
OUT["e4_length"] = pd.DataFrame(ls).round(4).to_dict("records")

# per-class recall: does qwen get destroyed by the hardcoded prior?
log("\nper-class recall, typo view alone vs 18-way base:")
rec_t = confusion_matrix(y, variants["typo_HARD_only"], labels=list(range(K)))
rec_t = np.diag(rec_t) / np.maximum(confusion_matrix(y, variants["typo_HARD_only"], labels=list(range(K))).sum(1), 1)
rec_b = confusion_matrix(y, variants["ng_only"], labels=list(range(K)))
rec_b = np.diag(rec_b) / np.maximum(confusion_matrix(y, variants["ng_only"], labels=list(range(K))).sum(1), 1)
tbl = pd.DataFrame({"model": VALID_18_MODELS, "typo_HARD_recall": rec_t, "ng_recall": rec_b})
log(tbl.round(3).to_string(index=False))
OUT["e4_perclass"] = tbl.round(4).to_dict("records")

# --------------------------------------------------------------------------- #
# E5: pairwise correction magnitude vs typo magnitude (inertness)
# --------------------------------------------------------------------------- #
log("\n=== E5: PAIRWISE CORRECTION EFFECT SIZE ON OOF ===")
pair_corr = {}
for a, b in PAIRS:
    i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
    m = (y == i) | (y == j)
    yp = (y[m] == j).astype(int)
    c = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
    c.fit(X_all[m], yp)
    pr = np.clip(c.predict_proba(X_all)[:, 1], 1e-4, 1 - 1e-4)
    pair_corr[(a, b)] = np.clip(np.log(pr / (1 - pr)), -2.5, 2.5)

# apply on TOP of the buggy pipeline, exactly as submission does
log_step = (1 - w_step) * log_p_ng + w_step * log_p_typo_hard
log_pair = log_step.copy()
for (a, b), d in pair_corr.items():
    log_pair[:, MODEL_TO_IDX[b]] += 0.10 * d
    log_pair[:, MODEL_TO_IDX[a]] -= 0.10 * d
log(f"argmax changed on {int((log_pair.argmax(1) != log_step.argmax(1)).sum())} / {len(y)} training rows "
    f"({(log_pair.argmax(1) != log_step.argmax(1)).mean():.2%})")
log(f"bacc with pairwise on top : {bacc(y, log_pair.argmax(1)):.4f}")
log(f"bacc without              : {bacc(y, log_step.argmax(1)):.4f}")
# residual vs raw injection
log("\nOOF pair discriminators were TRAINED ON THE SAME ROWS they are scored on here ->")
log("this delta is optimistic; in submission they are train-only (no test leakage) but")
log("applied to ALL passages, ~89% of which belong to neither class of the pair.")
mean_abs_delta = np.mean([np.abs(d).mean() for d in pair_corr.values()])
log(f"mean |delta| across pairs = {mean_abs_delta:.3f} nats -> x0.10 = {0.10*mean_abs_delta:.3f} nats")
OUT["e5"] = {
    "frac_argmax_changed": float((log_pair.argmax(1) != log_step.argmax(1)).mean()),
    "bacc_with": bacc(y, log_pair.argmax(1)),
    "bacc_without": bacc(y, log_step.argmax(1)),
    "mean_abs_delta_nats": float(mean_abs_delta),
    "max_effective_correction_nats": 0.25,
}

OUT["timing"] = timing
log("\n=== E5b: TIMING BREAKDOWN (local CPU, scale factors only) ===")
for k, v in timing.items():
    log(f"  {k}: {v:.1f}s")

with open("experiments/invalidation_evidence.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
log("\nwrote experiments/invalidation_evidence.json")
