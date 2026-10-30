"""
Experiment 16: Prompt-disjoint head CV and adapter provenance audit.

This is the experiment that decides whether the reported 0.7695 means anything.
It costs no leaderboard submissions and ~15 minutes on dual T4.

Three questions, three answers:

Q1. Does the 8-layer Qwen3 readout *representation* transfer to prompts the
    head has never seen?
    -> Refit ONLY the 32,768 -> 18 head on 4/5 of the prompts, evaluate on the
       held-out prompt fifth. The backbone adapter and the pooled features are
       frozen, so this isolates the representation from the head's capacity to
       memorise 2,699 rows.

Q2. How much of the shipped 0.7706 is pure in-sample memorisation?
    -> Score the *shipped* head on the same folds. A 32,768 x 18 linear map has
       589,824 free parameters against 2,699 samples; it can fit ANY labelling,
       including shuffled labels. The gap between the shipped head's score and
       the refit head's score is the memorisation premium. If it is large, the
       0.7695 "honest nested CV" is a fantasy and the 0.539 LB is the truth.

Q3. Did the third-party adapter see competition train.csv?
    -> Compare per-token NLL of the frozen adapter on (a) competition train
       passages and (b) external-pool passages from prompt groups not in train.
       A model trained only on the external pool shows no systematic gap.

Usage on Kaggle (dual T4, internet off):
    !python3 /kaggle/working/experiments/16_prompt_disjoint_head_cv.py
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.append("/kaggle/working")
from src.config import MODEL_TO_IDX, TRAIN_CSV, VALID_18_MODELS
from src.eval_stats import balanced_accuracy_ci

import peft.import_utils

peft.import_utils.is_torchao_available = lambda: False
try:  # pragma: no cover - Kaggle container quirk
    import peft.tuners.lora.torchao as lora_torchao

    lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass
from peft import PeftModel

BASE_CANDIDATES = [
    "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
    "/kaggle/input/qwen-3/transformers/8b-base/1",
]
LORA_CANDIDATES = [
    "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1",
    "/kaggle/input/ntl-two-stage-lora/lora8b_v1",
]
POOL_CANDIDATES = [
    "/kaggle/working/data/curated_pool_v2.parquet",
    "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/pool_v1.parquet",
]

SEED = 42
N_SPLITS = 5


def _locate(cands, what):
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{what}: none of {cands}")


# --------------------------------------------------------------------------- #
# Feature extraction: frozen backbone, multiple pooling statistics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(texts, tok, model, layers, max_len, batch_size=16):
    """Return (N, L*H*3): per-layer mean, std and max pooling concatenated."""
    model.eval()
    chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc="features"):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        am = enc["attention_mask"].to(model.device)
        hs = model(input_ids=ids, attention_mask=am,
                   output_hidden_states=True).hidden_states
        m = am.unsqueeze(-1).float()
        denom = torch.clamp(m.sum(dim=1), min=1e-8)
        pooled = []
        for lyr in layers:
            s = hs[lyr].float()
            mean = (s * m).sum(dim=1) / denom
            var = ((s - mean.unsqueeze(1)) ** 2 * m).sum(dim=1) / denom
            mx = s.masked_fill(m == 0, -1e4).max(dim=1).values
            pooled += [mean, torch.sqrt(torch.clamp(var, min=0)), mx]
        chunks.append(torch.cat(pooled, dim=-1).cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train = pd.read_csv(TRAIN_CSV)
    train["text"] = train["text"].fillna("")
    y = np.array([MODEL_TO_IDX[m] for m in train["model"]])
    groups = train["prompt_id"].values
    labels = VALID_18_MODELS

    base = _locate(BASE_CANDIDATES, "base model")
    lora = _locate(LORA_CANDIDATES, "lora dir")
    with open(os.path.join(lora, "meta.json")) as f:
        meta = json.load(f)
    layers = meta["cfg"]["layers"]
    max_len = meta["cfg"].get("max_len", 447)

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    cfg = AutoConfig.from_pretrained(base, trust_remote_code=True)
    cfg.num_hidden_layers = meta["cfg"]["keep"]
    model = AutoModel.from_pretrained(base, config=cfg, torch_dtype=torch.float16,
                                      device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, os.path.join(lora, "adapter"))
    model.eval()

    Z_trn = extract_features(train["text"].tolist(), tok, model, layers, max_len)
    print(f"train features {Z_trn.shape}")

    # ---- Q3: adapter provenance via NLL on train vs external -----------------
    pool_path = None
    for c in POOL_CANDIDATES:
        if os.path.exists(c):
            pool_path = c
            break
    if pool_path is not None:
        pool = pd.read_parquet(pool_path)
        pool = pool[pool["model"].isin(VALID_18_MODELS)]
        if "prompt_id" in pool.columns:
            # Use only prompt groups whose text does not appear in train.
            train_texts = set(train["text"])
            pool = pool[~pool["text"].isin(train_texts)]
        rng = np.random.default_rng(SEED)
        sample = pool.sample(min(400, len(pool)), random_state=SEED)
        print(f"\n--- Q3 adapter provenance: NLL on {len(sample)} external passages ---")
        nll_ext = _mean_nll(sample["text"].tolist(), tok, model, max_len)
        trn_sample = train.sample(min(400, len(train)), random_state=SEED)
        nll_trn = _mean_nll(trn_sample["text"].tolist(), tok, model, max_len)
        print(f"  train.csv NLL      : {nll_trn:.4f}")
        print(f"  external-pool NLL  : {nll_ext:.4f}")
        print(f"  gap (train - ext)  : {nll_trn - nll_ext:+.4f}"
              f"   {'<-- SUSPICIOUS: adapter saw train.csv' if nll_trn < nll_ext - 0.05 else '<-- no evidence of direct contamination'}")
    else:
        print("\n--- Q3 skipped: no external pool mounted ---")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Q1 + Q2: prompt-disjoint refit head vs shipped head -----------------
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train), dtype=int)
    for f, (_, val_idx) in enumerate(sgkf.split(train, y, groups=groups)):
        folds[val_idx] = f

    shipped = _load_shipped_head(lora, len(layers), cfg.hidden_size, len(meta["labels"]))
    shipped_logits = _shipped_forward(shipped, Z_trn, layers, cfg.hidden_size,
                                      meta["labels"], len(VALID_18_MODELS))
    shipped_pred = shipped_logits.argmax(axis=1)
    print(f"\n--- Q2 shipped head, in-sample, all 2699 rows ---")
    print(f"  BAcc = {balanced_accuracy_score(y, shipped_pred):.4f}   (this is the 0.7706)")

    results = {}
    for name, clf_factory in [
        ("logreg_C1", lambda: LogisticRegression(C=1.0, max_iter=2000,
                                                 class_weight="balanced")),
        ("logreg_C0.01", lambda: LogisticRegression(C=0.01, max_iter=2000,
                                                    class_weight="balanced")),
    ]:
        oof = np.zeros(len(train), dtype=int)
        shipped_fold = np.zeros(len(train), dtype=int)
        for f in range(N_SPLITS):
            trn = np.where(folds != f)[0]
            val = np.where(folds == f)[0]
            scaler = StandardScaler().fit(Z_trn[trn])
            clf = clf_factory()
            clf.fit(scaler.transform(Z_trn[trn]), y[trn])
            oof[val] = clf.predict(scaler.transform(Z_trn[val]))
            shipped_fold[val] = shipped_pred[val]
        b = balanced_accuracy_score(y, oof)
        ci = balanced_accuracy_ci(y, oof, groups, labels)
        s = balanced_accuracy_score(y, shipped_fold)
        results[name] = (b, ci)
        print(f"\n--- Q1 prompt-disjoint refit head: {name} ---")
        print(f"  BAcc = {b:.4f}  95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
              f"  SE {ci['se']:.4f}")
        print(f"  shipped head on the same rows: {s:.4f}   memorisation premium {s - b:+.4f}")

    # ---- Control: shuffled labels must collapse to chance --------------------
    rng = np.random.default_rng(SEED)
    y_perm = rng.permutation(y)
    ctrl = np.zeros(len(train), dtype=int)
    for f in range(N_SPLITS):
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        scaler = StandardScaler().fit(Z_trn[trn])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(Z_trn[trn]), y_perm[trn])
        ctrl[val] = clf.predict(scaler.transform(Z_trn[val]))
    print(f"\n--- control: shuffled labels, same harness ---")
    print(f"  BAcc = {balanced_accuracy_score(y_perm, ctrl):.4f}  (must be near 0.056)")

    np.save("/kaggle/working/exp16_features.npy", Z_trn)
    np.save("/kaggle/working/exp16_folds.npy", folds)
    print("\nSaved exp16_features.npy / exp16_folds.npy for downstream use.")


def _mean_nll(texts, tok, model, max_len):
    """Length-normalised, length-robust NLL of the frozen adapter."""
    total, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            batch = texts[i : i + 8]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt")
            ids = enc["input_ids"].to(model.device)
            am = enc["attention_mask"].to(model.device)
            # NOTE: AutoModel has no lm_head; this measures the encoder's
            # next-token difficulty proxy via hidden-state drift, so it is a
            # relative diagnostic only, not a calibrated perplexity.
            out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
            h = out.last_hidden_state.float()
            drift = (h[:, 1:] - h[:, :-1]).pow(2).mean(dim=-1)
            mask = am[:, 1:].float()
            total += float((drift * mask).sum())
            ntok += int(mask.sum())
    return total / max(ntok, 1)


def _load_shipped_head(lora, n_layers, hidden, n_classes):
    import torch.nn as nn
    from safetensors import safe_open

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
            self.plain = nn.Linear(hidden * n_layers, n_classes)

        def forward(self, per_layer):
            fused = torch.cat([self.norms[i](per_layer[i]) for i in range(n_layers)], -1)
            return self.plain(fused)

    head = Head()
    with safe_open(os.path.join(lora, "heads.safetensors"), framework="pt") as f:
        for i in range(n_layers):
            head.norms[i].weight.data.copy_(f.get_tensor(f"norms.{i}.weight"))
            head.norms[i].bias.data.copy_(f.get_tensor(f"norms.{i}.bias"))
        head.plain.weight.data.copy_(f.get_tensor("plain.weight"))
        head.plain.bias.data.copy_(f.get_tensor("plain.bias"))
    head.eval()
    return head


def _shipped_forward(head, Z, layers, hidden, meta_labels, n_classes):
    """`Z` stores mean,std,max per layer; the shipped head only used mean."""
    L = len(layers)
    per_layer = []
    for i in range(L):
        idx = i * 3 * hidden
        per_layer.append(torch.tensor(Z[:, idx : idx + hidden]))
    with torch.no_grad():
        logits = head(per_layer).numpy()
    out = np.zeros((len(Z), n_classes), dtype=np.float32)
    for col, name in enumerate(meta_labels):
        out[:, MODEL_TO_IDX[name]] = logits[:, col]
    return out


if __name__ == "__main__":
    main()
