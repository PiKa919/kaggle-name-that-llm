"""
Smoke test: run the drop-in fusion end to end on the real corpus.

Plumbing test, not a score claim.  There are no LoRA weights locally, so the
LoRA view is simulated, and the "test" set is the training set.  What is checked
is that every stage executes, that the audit table is populated, and that the
diagnostics say what the review claims.

    python3 experiments/22_dropin_smoke.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, ".")
from src.config import MODEL_TO_IDX, VALID_18_MODELS
from patches.dropin_submission_sota import FusionConfig, build_fusion, format_audit

K = 18
rng = np.random.default_rng(42)

df = pd.read_csv("data/train.csv")
df["text"] = df["text"].fillna("")
y = np.array([MODEL_TO_IDX[m] for m in df["model"]])
groups = df["prompt_id"].values
texts = df["text"].tolist()
X = load_npz("experiments/cache_X_all.npz")
P_ng = np.load("experiments/cache_P_ng.npy")
logp_ng_oof = np.log(np.clip(P_ng, 1e-9, 1.0))
logp_ng_te = logp_ng_oof.copy()   # same rows: plumbing only

# Simulated LoRA view: the OOF n-gram probabilities, sharpened and perturbed.
# It exists only to exercise the three-view code path.
logp_sim = logp_ng_oof * 1.7 + rng.normal(scale=0.35, size=P_ng.shape)
logp_sim -= logp_sim.max(axis=1, keepdims=True)
p_sim = np.exp(logp_sim)
p_sim /= p_sim.sum(axis=1, keepdims=True)

folds = list(
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups=groups)
)

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

res = build_fusion(
    train_texts=texts,
    y_train=y,
    groups=groups,
    X_train=X,
    test_texts=texts,
    X_test=X,
    p_lora=p_sim,
    p_ng=np.exp(logp_ng_te),
    pairs=PAIRS,
    cfg=FusionConfig(audit=True),
    logp_lora_oof=logp_sim,
    oof_ng_logp=logp_ng_oof,
    folds=folds,
)

print(format_audit(res))
print("\ngate report:", {k: round(v, 3) for k, v in res["gate"].items()})
print("pool weights shape:", None if res["pool_weights"] is None else res["pool_weights"].shape)
if res["pool_weights"] is not None:
    W = res["pool_weights"]
    print("  mean share of the simulated LoRA view: %.3f" % W[:, 0].mean())
    print("  per-class share range: %.3f .. %.3f" % (W[:, 0].min(), W[:, 0].max()))
print("view reliability:", res["view_reliability"])
print("prediction length:", len(res["pred"]), "distinct classes:", len(set(res["pred"].tolist())))
print("top predicted classes:")
print(pd.Series([VALID_18_MODELS[i] for i in res["pred"]]).value_counts().head(5).to_string())
print("\nper-class optimal typo weight (first 6 rows):")
print(res["per_class_weight_sweep"].head(6).round(3).to_string(index=False))
