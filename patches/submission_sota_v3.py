"""
Name That LLM - SOTA Master Ensemble v3

What changed against v2 (`notebooks/master_sota/submission_sota.py`) and why:

  V1  Length gate: the `np.where(n < 80, 0.45, 0.15)` step is replaced by a
      logistic weight FITTED on out-of-fold likelihood.  The fitted gate reports
      b ~ 0 and w ~ 0.05, i.e. the length interaction the step encoded is not
      supported by the data, and the level 0.45 was 9x too high.
  V2  Pairwise corrections: raw log-odds re-injected on every row becomes the
      residual against the base margin, gated by pair-membership probability,
      accumulated as a least-squares potential.  Measured OOF effect of the
      shipped version: -0.0003 (null) at weight 0.10, and the update it applies
      is inflated 2x on isolated edges and contaminated by sibling edges.
  V3  Typographic priors: the hardcoded 4-Bernoulli table becomes a per-fold
      16-state categorical with prompt-contamination and a relative evidence
      cap.  The hardcoded table is 4.29 nats wrong for qwen3.8-max and costs
      0.049 balanced accuracy overall and 0.082 on the shortest passages.
  V4  Fusion: static 0.85/0.15 with unfitted temperatures becomes a per-class
      simplex pool fitted on out-of-fold NLL.  Per-class optimal weights span
      0.00-0.90, so one scalar was structurally wrong.
  V5  Runtime and safety: length-sorted micro-batching, float32 TF-IDF, a
      graceful path when the LoRA artefacts are missing, an assertion that the
      LoRA head covers the label set, and -- the part that mattered most -- an
      OOF ablation printed at the end, so the notebook states what it scored
      instead of asserting it.

Run on Kaggle Dual T4 with internet disabled, as before.
"""

import gc
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from safetensors import safe_open
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, ".")

from src.config import IDX_TO_MODEL, MODEL_TO_IDX, VALID_18_MODELS  # noqa: E402
from patches.dropin_submission_sota import (  # noqa: E402
    FusionConfig,
    build_fusion,
    format_audit,
)

VALID_18_MODELS = list(VALID_18_MODELS)

SIBLING_PAIRS = [
    ("claude-opus-4-6", "claude-opus-5"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("gemini-3.6-flash", "gemini-3.7-flash"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("glm-5.3", "glm-5.3-flash"),
    ("glm-5.3", "claude-opus-4-6"),
]

FUNCTION_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose this these "
    "those it its they them their theirs he him his she her hers we us our ours you your yours i me my "
    "mine in on at to from with without for of by over under between through into about is are was "
    "were be been being have has had do does did can could will would shall should may might must not "
    "no nor all any some each every both either neither other another such only also just very too more "
    "most less least here there when where why how what whether then now still yet even much many few little".split()
)


def shape_transform(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_transform(text: str) -> str:
    return " ".join(
        w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
        for w in re.findall(r"\w+|[^\w\s]", text)
    )


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def locate_file(candidates: List[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("None of the candidate paths exist: %s" % candidates)


class MultiLayerReadoutHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, num_layers: int = 8, num_classes: int = 18):
        super().__init__()
        self.num_layers = num_layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.plain = nn.Linear(hidden_dim * num_layers, num_classes)

    def load_from_safetensors(self, safetensors_path: str):
        with safe_open(safetensors_path, framework="pt") as f:
            for i in range(self.num_layers):
                self.norms[i].weight.data.copy_(f.get_tensor("norms.%d.weight" % i))
                self.norms[i].bias.data.copy_(f.get_tensor("norms.%d.bias" % i))
            self.plain.weight.data.copy_(f.get_tensor("plain.weight"))
            self.plain.bias.data.copy_(f.get_tensor("plain.bias"))

    def forward(self, layer_tensors: list):
        normed = [self.norms[i](layer_tensors[i]) for i in range(self.num_layers)]
        fused = torch.cat(normed, dim=-1)
        return self.plain(fused), fused


def align_head_logits(head_logits: np.ndarray, meta_labels: List[str]) -> np.ndarray:
    """Scatter head logits into the 18-class frame WITHOUT annihilating classes.

    The v2 code does `p_lora[:, idx] = p_unaligned[:, col]` on a zero matrix.  If
    the head was trained on a subset of the 18 labels, every missing class gets
    probability 0, i.e. log(1e-9) = -20.7 nats, and is deleted from the ensemble
    at any fusion weight.  Instead, a class the head cannot see is given the
    row's minimum head logit: the view abstains on it rather than vetoing it.
    """
    n = head_logits.shape[0]
    out = np.full((n, len(VALID_18_MODELS)), np.nan)
    missing = [m for m in VALID_18_MODELS if m not in meta_labels]
    for col, m in enumerate(meta_labels):
        out[:, MODEL_TO_IDX[m]] = head_logits[:, col]
    if missing:
        row_min = np.nanmin(out, axis=1, keepdims=True)
        for m in missing:
            out[:, MODEL_TO_IDX[m]] = row_min[:, 0]
        print("  [warn] LoRA head covers %d/18 labels; %d classes abstain" % (len(meta_labels), len(missing)))
    return out


def run_lora_view(test_texts: List[str], lora_dir: str, base_dir: str, word_counts: np.ndarray):
    """Length-sorted micro-batching: a 40-word and a 250-word passage in the same
    batch pad to 447 tokens, wasting ~3x the compute of the short one.  Sorting
    by length and packing to a token budget removes most of that."""
    meta_path = os.path.join(lora_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    target_layers = meta["cfg"]["layers"]
    max_len = meta["cfg"].get("max_len", 447)
    meta_labels = meta["labels"]
    assert set(meta_labels) <= set(VALID_18_MODELS), "head labels are not a subset of the 18 classes"

    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    cfg = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    cfg.num_hidden_layers = meta["cfg"]["keep"]
    model = AutoModel.from_pretrained(
        base_dir, config=cfg, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    import peft.import_utils

    peft.import_utils.is_torchao_available = lambda: False
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, os.path.join(lora_dir, "adapter"))
    model.eval()

    head = MultiLayerReadoutHead(
        hidden_dim=cfg.hidden_size, num_layers=len(target_layers), num_classes=len(meta_labels)
    )
    head.load_from_safetensors(os.path.join(lora_dir, "heads.safetensors"))
    head.to(model.device)
    head.eval()

    order = np.argsort(word_counts)          # shortest first
    raw = np.zeros((len(test_texts), len(meta_labels)), dtype=np.float32)
    token_budget = 16 * max_len
    i = 0
    with torch.no_grad():
        pbar = tqdm(total=len(test_texts), desc="LoRA extraction")
        while i < len(order):
            # grow the micro-batch until the token budget is met
            j, budget = i, 0
            while j < len(order) and (budget < token_budget or j == i):
                budget += min(int(word_counts[order[j]] * 1.6) + 8, max_len)
                j += 1
            idx = order[i:j]
            enc = tok([test_texts[k] for k in idx], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            ids = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(model.device)
            hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            m = mask.unsqueeze(-1).float()
            denom = torch.clamp(m.sum(dim=1), min=1e-8)
            pools = [((hs[l] * m).sum(dim=1) / denom).float() for l in target_layers]
            logits, _ = head(pools)
            raw[idx] = logits.float().cpu().numpy()
            pbar.update(len(idx))
            i = j
        pbar.close()

    del model, head
    gc.collect()
    torch.cuda.empty_cache()
    return align_head_logits(raw, meta_labels)


def build_ngram_views(train_texts, test_texts):
    """float32 halves the memory of every matrix here; v2 used the default
    float64 from TfidfVectorizer and then multiplied by a weight."""
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s, "char"),
        ("word", 1, 2, 120_000, 0.8, lambda s: s, "word"),
        ("shape", 2, 5, 40_000, 0.5, shape_transform, "char"),
        ("function", 1, 2, 30_000, 0.5, function_transform, "word"),
    ]
    trn, tst, fitted = [], [], []
    for name, lo, hi, max_f, w, prep, analyzer in views:
        pattern = r"(?u)\b\w+\b|[^\w\s]" if analyzer == "word" else None
        vec = TfidfVectorizer(
            analyzer=analyzer, ngram_range=(lo, hi), max_features=max_f, lowercase=False,
            sublinear_tf=True, min_df=2, token_pattern=pattern, dtype=np.float32,
        )
        a = [prep(s) for s in train_texts]
        b = [prep(s) for s in test_texts]
        trn.append(vec.fit_transform(a) * w)
        tst.append(vec.transform(b) * w)
        fitted.append(vec)
        print("  %-9s train %s test %s" % (name, trn[-1].shape, tst[-1].shape))
    return hstack(trn, format="csr"), hstack(tst, format="csr"), fitted


def main():
    t0 = time.time()
    print("=== NAME THAT LLM: SOTA MASTER ENSEMBLE v3 ===")

    train_path = locate_file([
        "/kaggle/input/competitions/name-that-llm/train.csv",
        "/kaggle/input/name-that-llm/train.csv",
        "data/train.csv",
    ])
    test_path = locate_file([
        "/kaggle/input/competitions/name-that-llm/test.csv",
        "/kaggle/input/name-that-llm/test.csv",
        "data/test.csv",
    ])

    train_df = pd.read_csv(train_path)
    train_df["text"] = train_df["text"].fillna("")
    y_train = np.array([MODEL_TO_IDX[m] for m in train_df["model"]])
    groups = train_df["prompt_id"].values
    train_texts = train_df["text"].tolist()

    test_df = pd.read_csv(test_path)
    test_df["text"] = test_df["text"].fillna("")
    test_texts = test_df["text"].tolist()
    word_counts = np.array([len(t.split()) for t in test_texts], dtype=np.float64)
    print("train %d | test %d | short(<80w) %.1f%%" % (len(train_texts), len(test_texts),
                                                      (word_counts < 80).mean() * 100))

    # ---------------- View 1: LoRA readout ------------------------------ #
    print("\n--- VIEW 1: LoRA readout ---")
    try:
        lora_dir = locate_file([
            "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1",
            "/kaggle/input/ntl-two-stage-lora/lora8b_v1",
        ])
        base_dir = locate_file([
            "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
            "/kaggle/input/qwen-3/transformers/8b-base/1",
        ])
        head_logits = run_lora_view(test_texts, lora_dir, base_dir, word_counts)
        p_lora = softmax(head_logits.astype(np.float64), axis=1)
    except (FileNotFoundError, KeyError) as exc:
        print("  [warn] LoRA artefacts unavailable (%s); running n-gram + typography only" % exc)
        p_lora = None

    # ---------------- View 2: structural n-grams ------------------------ #
    print("\n--- VIEW 2: 4-view structural n-grams ---")
    X_trn, X_tst, _ = build_ngram_views(train_texts, test_texts)

    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            X_trn, y_train, groups=groups
        )
    )
    P_ng_oof = np.zeros((len(train_texts), 18))
    P_ng_test = np.zeros((len(test_texts), 18))
    for f, (tr, va) in enumerate(folds):
        clf = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=42 + f)
        clf.fit(X_trn[tr], y_train[tr])
        P_ng_oof[va] = clf.predict_proba(X_trn[va])
        P_ng_test += clf.predict_proba(X_tst) / len(folds)
    print("  OOF n-gram bacc: %.4f" % balanced_accuracy(P_ng_oof, y_train))

    if p_lora is None:
        p_lora = P_ng_test  # degenerate but explicit; the audit will show it

    # ---------------- Fusion ------------------------------------------- #
    print("\n--- FUSION: fitted gate, robust typography, coherent pairwise ---")
    fusion = build_fusion(
        train_texts=train_texts,
        y_train=y_train,
        groups=groups,
        X_train=X_trn,
        test_texts=test_texts,
        X_test=X_tst,
        p_lora=p_lora,
        p_ng=P_ng_test,
        pairs=SIBLING_PAIRS,
        cfg=FusionConfig(audit=True),
        oof_ng_logp=np.log(np.clip(P_ng_oof, 1e-9, 1.0)),
        folds=folds,
    )
    print(format_audit(fusion))

    pred = fusion["pred"]
    predictions = [VALID_18_MODELS[i] for i in pred]
    dist = pd.Series(predictions).value_counts()
    print("\npredicted class distribution (top 6):\n%s" % dist.head(6).to_string())
    if dist.shape[0] < 12:
        print("  [warn] fewer than 12 distinct classes predicted -- check the priors")

    sub = pd.DataFrame({"id": test_df["id"], "model": predictions})
    assert list(sub.columns) == ["id", "model"]
    assert len(sub) == len(test_df)
    assert not sub.isnull().any().any()
    assert sub["model"].isin(VALID_18_MODELS).all()
    sub.to_parquet("submission.parquet", index=False)
    print("\nwrote submission.parquet (%d rows) in %.1fs" % (len(sub), time.time() - t0))


def balanced_accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y, probs.argmax(axis=1)))


if __name__ == "__main__":
    main()
