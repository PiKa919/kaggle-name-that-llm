"""
Experiment 06: Frozen Qwen3 Intermediate Representations + Concat(Mean, Max) Pooling + Ridge/Logistic Probing
Prescribed by DeepSeek Brutal Teacher.
Runs on Kaggle Dual T4 interactive GPU session.
"""

import gc
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.metrics import evaluate_predictions, format_evaluation_summary


def get_model_path():
    candidates = [
        "/kaggle/input/models/qwen-lm/qwen-3/transformers/4b-base/1",
        "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
        "/kaggle/input/qwen-3/transformers/4b-base/1",
        "/kaggle/input/qwen-3/transformers/8b-base/1",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    import kagglehub
    print("Downloading qwen-3/4b-base via kagglehub...")
    return kagglehub.model_download("qwen-lm/qwen-3/transformers/4b-base")


def extract_intermediate_embeddings(texts: list, model_path: str, layers: list = None, batch_size: int = 16, max_len: int = 256):
    print(f"\n[1/3] Loading tokenizer and model from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    n_layers = cfg.num_hidden_layers
    hidden_dim = cfg.hidden_size
    print(f"Model has {n_layers} layers with hidden dimension {hidden_dim}")
    
    if layers is None:
        # Select 3 evenly spaced intermediate layers (e.g. ~35%, ~55%, ~75% depth)
        layers = [n_layers // 3, (2 * n_layers) // 3, (5 * n_layers) // 6]
    print(f"Target extraction layers: {layers}")
    
    # Dual T4 memory mapping
    if torch.cuda.device_count() > 1:
        max_mem = {i: "14GiB" for i in range(torch.cuda.device_count())}
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True
        ).eval()
    else:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True
        ).eval()
        
    print(f"[2/3] Tokenizing {len(texts)} samples (max_len={max_len})...")
    encoded = tok(
        texts,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_attention_mask=True
    )
    input_ids = encoded["input_ids"]
    
    # Sort by length to minimize dynamic padding overhead
    lengths = [len(x) for x in input_ids]
    sort_order = np.argsort(lengths)
    inv_order = np.empty_like(sort_order)
    inv_order[sort_order] = np.arange(len(sort_order))
    
    feature_dim = len(layers) * hidden_dim * 2  # concat(mean, max) per layer
    all_features = np.zeros((len(texts), feature_dim), dtype=np.float32)
    
    print(f"[3/3] Running forward pass on Dual T4 (batch_size={batch_size})...")
    t0 = time.time()
    
    device = next(model.parameters()).device
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_indices = sort_order[i:i + batch_size]
        batch_ids = [input_ids[j] for j in batch_indices]
        
        batch = tok.pad({"input_ids": batch_ids}, return_tensors="pt")
        b_input_ids = batch["input_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hs = outputs.hidden_states
            
            # Mask out token 0 (attention sink) as DeepSeek prescribed
            b_mask_active = b_mask.clone().float()
            b_mask_active[:, 0] = 0.0
            mask_expanded = b_mask_active.unsqueeze(-1)  # (B, T, 1)
            token_counts = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
            
            layer_pools = []
            for lyr in layers:
                h = hs[lyr].to(device).float()  # (B, T, D)
                
                # 1. Masked Mean Pooling
                h_mean = (h * mask_expanded).sum(dim=1) / token_counts  # (B, D)
                
                # 2. Masked Max Pooling (captures rare stylometric tokens)
                h_masked = h.clone()
                h_masked[mask_expanded.expand_as(h) == 0] = -1e4
                h_max = h_masked.max(dim=1)[0]  # (B, D)
                
                layer_pools.extend([h_mean, h_max])
                
            batch_repr = torch.cat(layer_pools, dim=-1).cpu().numpy()
            all_features[batch_indices] = batch_repr
            
    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Extraction complete in {time.time() - t0:.1f}s. Extracted matrix: {all_features.shape}")
    return all_features


def main():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    
    folds = split_folds(df)
    
    model_path = get_model_path()
    
    feat_cache = "/kaggle/working/qwen_probe_features.npy"
    if os.path.exists(feat_cache):
        print(f"Loading cached features from {feat_cache}...")
        X = np.load(feat_cache)
    else:
        X = extract_intermediate_embeddings(df["text"].tolist(), model_path, batch_size=16)
        np.save(feat_cache, X)
        print(f"Saved extracted features to {feat_cache}")
        
    # L2 normalize feature rows
    print("\nL2 Normalizing features...")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms
    
    print("\n--- Running 5-Fold StratifiedGroupKFold on Extracted Representations ---")
    oof_ridge_preds = np.empty(len(df), dtype=object)
    oof_lr_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    
    for fold in range(5):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # A. Ridge Classifier
        ridge = RidgeClassifier(alpha=100.0, class_weight="balanced")
        ridge.fit(X_norm[trn_idx], y_raw[trn_idx])
        pred_ridge = ridge.predict(X_norm[val_idx])
        oof_ridge_preds[val_idx] = pred_ridge
        bacc_ridge = balanced_accuracy_score(y_raw[val_idx], pred_ridge)
        
        # B. Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs")
        lr.fit(X_norm[trn_idx], y_raw[trn_idx])
        probs = lr.predict_proba(X_norm[val_idx])
        oof_lr_probs[val_idx] = probs
        pred_lr = [VALID_18_MODELS[i] for i in np.argmax(probs, axis=1)]
        bacc_lr = balanced_accuracy_score(y_raw[val_idx], pred_lr)
        
        print(f"Fold {fold}: Ridge BAcc = {bacc_ridge:.4f} | LogisticRegression BAcc = {bacc_lr:.4f}")
        
    pred_lr_all = np.array([VALID_18_MODELS[i] for i in np.argmax(oof_lr_probs, axis=1)])
    rep_ridge = evaluate_predictions(y_raw, oof_ridge_preds)
    rep_lr = evaluate_predictions(y_raw, pred_lr_all)
    
    print("\n" + "="*50)
    print("FROZEN QWEN3 RIDGE CLASSIFIER REPORT:")
    print(format_evaluation_summary(rep_ridge))
    
    print("\n" + "="*50)
    print("FROZEN QWEN3 LOGISTIC REGRESSION REPORT:")
    print(format_evaluation_summary(rep_lr))
    
    # Compare with our baseline of 0.3501 - 0.3689
    delta = rep_lr["balanced_accuracy"] - 0.3689
    print(f"\n[VERDICT] Delta vs Previous Best Baseline (0.3689): {delta:+.4f} Balanced Accuracy")
    if delta > 0.05:
        print("[SUCCESS] Major breakthrough achieved! Sibling attribution significantly improved.")
        

if __name__ == "__main__":
    main()
