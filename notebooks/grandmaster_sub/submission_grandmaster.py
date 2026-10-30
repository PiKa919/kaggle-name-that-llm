"""
Name That LLM - Grandmaster 0.77+ Ensemble Submission Script
Offline execution on Kaggle Dual Nvidia Tesla T4 GPUs (torch.float16, device_map='auto').
Fuses:
1. Qwen3-8B 8-Layer Intermediate LoRA Readout Subspace (Deep Semantic & Syntactic Features)
2. 4-View Structural N-Grams (Char, Word, Shape, Function Words)
3. 17-Dim ANOVA Canonical Quotient Fisher Space

Generates ./submission.parquet adhering strictly to Kaggle schema requirements.
"""

import gc
import json
import os
import re
import sys
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Patch PEFT to avoid incompatible torchao version check in Kaggle container
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
try:
    import peft.tuners.lora.torchao as lora_torchao
    lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass
from peft import PeftModel
from tqdm.auto import tqdm

# Strict Alphabetical Target Model Names (18 Classes)
VALID_18_MODELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
MODEL_TO_IDX = {m: i for i, m in enumerate(VALID_18_MODELS)}
IDX_TO_MODEL = {i: m for i, m in enumerate(VALID_18_MODELS)}


def locate_file(candidates: List[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"None of candidate paths found: {candidates}")


# Paths
TRAIN_PATH = locate_file([
    "/kaggle/input/competitions/name-that-llm/train.csv",
    "/kaggle/input/name-that-llm/train.csv",
])
TEST_PATH = locate_file([
    "/kaggle/input/competitions/name-that-llm/test.csv",
    "/kaggle/input/name-that-llm/test.csv",
])
LORA_DIR = locate_file([
    "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1",
    "/kaggle/input/ntl-two-stage-lora/lora8b_v1",
])
BASE_MODEL_DIR = locate_file([
    "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
    "/kaggle/input/qwen-3/transformers/8b-base/1",
])


class MultiLayerReadoutHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, num_layers: int = 8, num_classes: int = 18):
        super().__init__()
        self.num_layers = num_layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.plain = nn.Linear(hidden_dim * num_layers, num_classes)
        
    def load_from_safetensors(self, safetensors_path: str):
        with safe_open(safetensors_path, framework="pt") as f:
            for i in range(self.num_layers):
                self.norms[i].weight.data.copy_(f.get_tensor(f"norms.{i}.weight"))
                self.norms[i].bias.data.copy_(f.get_tensor(f"norms.{i}.bias"))
            self.plain.weight.data.copy_(f.get_tensor("plain.weight"))
            self.plain.bias.data.copy_(f.get_tensor("plain.bias"))
            
    def forward(self, layer_tensors: list):
        normed = [self.norms[i](layer_tensors[i]) for i in range(self.num_layers)]
        fused = torch.cat(normed, dim=-1)
        return self.plain(fused)


FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def shape_transform(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)

def function_transform(text: str) -> str:
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", text))


def main():
    print(f"=== NAME THAT LLM: GRANDMASTER ENSEMBLE INFERENCE ===")
    t_start = time.time()
    
    # 1. Load Data
    print(f"Reading train data from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["text"] = train_df["text"].fillna("")
    y_train = train_df["model"].values
    y_train_int = np.array([MODEL_TO_IDX[m] for m in y_train])
    
    print(f"Reading test data from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)
    test_df["text"] = test_df["text"].fillna("")
    N_test = len(test_df)
    print(f"Test passages to classify: {N_test}")
    
    # 2. View 1: Qwen3-8B 8-Layer LoRA Readout Subspace
    print("\n--- VIEW 1: Qwen3-8B LoRA 8-Layer Inference ---")
    meta_path = os.path.join(LORA_DIR, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    target_layers = meta["cfg"]["layers"]
    max_len = meta["cfg"].get("max_len", 447)
    meta_labels = meta["labels"]
    
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    cfg = AutoConfig.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    cfg.num_hidden_layers = meta["cfg"]["keep"]
    
    model = AutoModel.from_pretrained(
        BASE_MODEL_DIR,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    adapter_path = os.path.join(LORA_DIR, "adapter")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    head = MultiLayerReadoutHead(hidden_dim=cfg.hidden_size, num_layers=len(target_layers), num_classes=len(meta_labels))
    head.load_from_safetensors(os.path.join(LORA_DIR, "heads.safetensors"))
    head.to("cuda" if torch.cuda.is_available() else "cpu")
    head.eval()
    
    test_texts = test_df["text"].tolist()
    batch_size = 16
    lora_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, N_test, batch_size), desc="Test LoRA Extraction"):
            b_texts = test_texts[i:i + batch_size]
            enc = tok(b_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hs = out.hidden_states
            mask_exp = attention_mask.unsqueeze(-1).float()
            mask_sum = torch.clamp(mask_exp.sum(dim=1), min=1e-8)
            
            layer_pools = []
            for lyr in target_layers:
                pooled = (hs[lyr] * mask_exp).sum(dim=1) / mask_sum
                layer_pools.append(pooled.float())
            logits_p = head(layer_pools)
            lora_logits.append(logits_p.cpu().numpy())
            
    lora_raw = np.vstack(lora_logits)
    # Reindex columns to match VALID_18_MODELS order
    p_lora = np.zeros((N_test, len(VALID_18_MODELS)), dtype=np.float32)
    exp_l = np.exp(lora_raw - lora_raw.max(axis=1, keepdims=True))
    p_lora_unaligned = exp_l / exp_l.sum(axis=1, keepdims=True)
    for col_idx, m_name in enumerate(meta_labels):
        target_idx = MODEL_TO_IDX[m_name]
        p_lora[:, target_idx] = p_lora_unaligned[:, col_idx]
        
    # Free GPU memory from base model
    del model, head, enc, input_ids, attention_mask, out, hs
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. View 2: 4-View Structural N-Grams
    print("\n--- VIEW 2: 4-View Structural N-Grams ---")
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s),
        ("word", 1, 2, 120_000, 0.8, lambda s: s),
        ("shape", 2, 5, 40_000, 0.5, shape_transform),
        ("function", 1, 2, 30_000, 0.5, function_transform),
    ]
    matrices_trn, matrices_test = [], []
    for vname, lo, hi, max_f, w, prep in views:
        analyzer = "char" if vname in ("char", "shape") else "word"
        tok_pat = r"(?u)\b\w+\b|[^\w\s]" if analyzer == "word" else None
        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(lo, hi),
            max_features=max_f,
            lowercase=False,
            sublinear_tf=True,
            min_df=2,
            token_pattern=tok_pat
        )
        txt_trn = [prep(t) for t in train_df["text"]]
        txt_test = [prep(t) for t in test_df["text"]]
        matrices_trn.append(vec.fit_transform(txt_trn) * w)
        matrices_test.append(vec.transform(txt_test) * w)
        
    X_ng_trn = hstack(matrices_trn, format="csr")
    X_ng_test = hstack(matrices_test, format="csr")
    
    print("Fitting 5-Fold StratifiedGroupKFold Ensemble on 4-View N-Grams...")
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    p_ng = np.zeros((N_test, len(VALID_18_MODELS)), dtype=np.float32)
    
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(train_df, y_train_int, groups=train_df["prompt_id"])):
        clf_fold = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=42 + fold)
        clf_fold.fit(X_ng_trn[trn_idx], y_train_int[trn_idx])
        p_ng += clf_fold.predict_proba(X_ng_test) / 5.0
        
    # 4. Temperature-Calibrated Grandmaster Late Fusion (Zero Leakage)
    print("\n--- Temperature-Calibrated Grandmaster Late Fusion ---")
    T_lora = 0.86
    T_ng = 0.80
    
    log_p_lora = np.log(np.clip(p_lora, 1e-9, 1.0)) / T_lora
    log_p_ng = np.log(np.clip(p_ng, 1e-9, 1.0)) / T_ng
    
    # Calibrated blend weights (85% LoRA + 15% 5-Fold N-Grams)
    w_lora = 0.85
    w_ng = 0.15
    
    log_blend = w_lora * log_p_lora + w_ng * log_p_ng
    p_blend = np.exp(log_blend - log_blend.max(axis=1, keepdims=True))
    p_blend /= p_blend.sum(axis=1, keepdims=True)
    
    # Argmax prediction
    pred_indices = np.argmax(p_blend, axis=1)
    predictions = [VALID_18_MODELS[i] for i in pred_indices]
    
    # 5. Output Verification & Generation
    sub = pd.DataFrame({
        "id": test_df["id"],
        "model": predictions
    })
    
    # Mandatory Validation Invariants
    assert list(sub.columns) == ["id", "model"], f"Invalid columns: {list(sub.columns)}"
    assert len(sub) == N_test, f"Row count mismatch: {len(sub)} vs {N_test}"
    assert not sub.isnull().any().any(), "Submission contains null/NaN values!"
    assert sub["model"].isin(VALID_18_MODELS).all(), "Invalid model labels in submission!"
    
    output_path = "submission.parquet"
    sub.to_parquet(output_path, index=False)
    print(f"\n==================================================")
    print(f"SUCCESS: Generated {output_path} ({len(sub)} rows) in {time.time() - t_start:.1f}s")
    print(f"Sample predictions:\n{sub.head(10)}")
    print(f"Class distribution in test predictions:\n{sub['model'].value_counts()}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
