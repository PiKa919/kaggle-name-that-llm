"""
Experiment 08: Fused ANOVA Canonical Quotient Space + Dense Stylometry + 4-View Structural N-Grams + Sinkhorn Balancing
Engineered to break the 0.50+ Balanced Accuracy barrier on Kaggle Dual T4.
"""

import os
import re
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.features import extract_stylometric_features
from src.metrics import evaluate_predictions, format_evaluation_summary


def anova_center(Z: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    """Within-prompt centering annihilates the rank-150 prompt topic subspace exactly."""
    unique_prompts, prompt_inv = np.unique(prompt_ids, return_inverse=True)
    G = len(unique_prompts)
    
    prompt_means = np.zeros((G, Z.shape[1]), dtype=np.float32)
    counts = np.bincount(prompt_inv, minlength=G)[:, None]
    np.add.at(prompt_means, prompt_inv, Z)
    prompt_means /= counts
    
    grand_mean = Z.mean(axis=0, keepdims=True)
    Z_centered = Z - prompt_means[prompt_inv] + grand_mean
    return Z_centered


def sinkhorn_knopp_balance(P: np.ndarray, iters: int = 100) -> np.ndarray:
    """
    Sinkhorn-Knopp matrix scaling to enforce uniform marginal class predictions (N / K),
    optimizing the unweighted Balanced Accuracy evaluation metric.
    """
    P_out = P.copy()
    N, K = P_out.shape
    target = np.full(K, N / K, dtype=np.float64)
    
    for _ in range(iters):
        # Scale columns to match target class marginals
        col_sums = P_out.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums > 1e-12, col_sums, 1.0)
        P_out *= target / col_sums
        
        # Scale rows to be valid probabilities
        row_sums = P_out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        P_out /= row_sums
        
    return P_out


FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def shape_transform(text: str) -> str:
    """Map uppercase letters to X and lowercase to x to isolate capitalization rhythm."""
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)

def function_transform(text: str) -> str:
    """Isolate function words and punctuation, replacing content words with 'CONTENT'."""
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", text))


def run_experiment_08():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    prompts = df["prompt_id"].values
    
    # 1. Load cached Qwen3 intermediate representations
    feat_cache = "/kaggle/working/qwen_probe_features.npy"
    if not os.path.exists(feat_cache):
        raise FileNotFoundError(f"Missing {feat_cache}! Run exp 06 first.")
    print(f"Loading cached Qwen3 embeddings from {feat_cache}...")
    Z = np.load(feat_cache)
    print(f"Loaded ambient Qwen3 matrix: {Z.shape}")
    
    # 2. Extract dense stylometry
    print("Extracting 74 dense stylometric features...")
    t0 = time.time()
    sty_df = extract_stylometric_features(df["text"].tolist())
    X_sty = sty_df.values.astype(np.float32)
    print(f"Extracted stylometric features in {time.time() - t0:.1f}s.")
    
    # 3. Crossed Two-Way ANOVA Centering on Qwen3
    print("Applying Crossed Two-Way ANOVA Centering to purge prompt topic nuisance...")
    Zc = anova_center(Z, prompts)
    
    # 4. Fit 4-View Structural N-Grams
    print("Configuring 4-View Structural N-Grams...")
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s),
        ("word", 1, 2, 120_000, 0.8, lambda s: s),
        ("shape", 2, 5, 40_000, 0.5, shape_transform),
        ("function", 1, 2, 30_000, 0.5, function_transform),
    ]
    
    # 5-fold StratifiedGroupKFold on prompt_id
    folds = split_folds(df)
    
    oof_p_quotient = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_p_ngrams = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_p_fused = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    
    print("\n--- Running 5-Fold StratifiedGroupKFold (Quotient Space + Stylometry + 4-Views) ---")
    
    for fold in range(5):
        t_fold = time.time()
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # A. Canonical 17-Dim Quotient Space + Dense Stylometry Head
        _, _, Vt = randomized_svd(Zc[trn_idx], n_components=256, random_state=42)
        Y_trn = Zc[trn_idx] @ Vt.T
        Y_val = Zc[val_idx] @ Vt.T
        
        D_svd = Y_trn.shape[1]
        Sb = np.zeros((D_svd, D_svd), dtype=np.float64)
        Sw = np.zeros((D_svd, D_svd), dtype=np.float64)
        mu_trn = Y_trn.mean(axis=0)
        
        for k in range(len(VALID_18_MODELS)):
            Yk = Y_trn[y_int[trn_idx] == k]
            if len(Yk) == 0:
                continue
            mk = Yk.mean(axis=0)
            diff = mk - mu_trn
            Sb += len(Yk) * np.outer(diff, diff)
            within_diff = Yk - mk
            Sw += within_diff.T @ within_diff
            
        Sw += 1e-2 * (np.trace(Sw) / D_svd) * np.eye(D_svd)
        ev, V = sla.eigh(Sb, Sw)
        idx_desc = np.argsort(ev)[::-1]
        V17 = V[:, idx_desc][:, :17]
        
        F_trn = Y_trn @ V17
        F_val = Y_val @ V17
        
        # Scale stylometric features
        scaler = StandardScaler()
        X_sty_trn = scaler.fit_transform(X_sty[trn_idx])
        X_sty_val = scaler.transform(X_sty[val_idx])
        
        # Fused Quotient + Stylometry Representation
        X_fused_trn = np.hstack([F_trn, X_sty_trn])
        X_fused_val = np.hstack([F_val, X_sty_val])
        
        clf_quotient = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        clf_quotient.fit(X_fused_trn, y_raw[trn_idx])
        oof_p_quotient[val_idx] = clf_quotient.predict_proba(X_fused_val)
        
        # B. 4-View Structural N-Grams with LinearSVC
        matrices_trn = []
        matrices_val = []
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
            txt_trn = [prep(t) for t in df["text"].iloc[trn_idx]]
            txt_val = [prep(t) for t in df["text"].iloc[val_idx]]
            matrices_trn.append(vec.fit_transform(txt_trn) * w)
            matrices_val.append(vec.transform(txt_val) * w)
            
        X_ng_trn = hstack(matrices_trn, format="csr")
        X_ng_val = hstack(matrices_val, format="csr")
        
        clf_ng = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=42)
        clf_ng.fit(X_ng_trn, y_raw[trn_idx])
        oof_p_ngrams[val_idx] = clf_ng.predict_proba(X_ng_val)
        
        # C. Geometric Log-Probability Fusion
        # Fused = 0.65 * Quotient/Stylometry + 0.35 * 4-View N-Grams
        log_p = 0.65 * np.log(np.clip(oof_p_quotient[val_idx], 1e-9, 1.0)) + \
                0.35 * np.log(np.clip(oof_p_ngrams[val_idx], 1e-9, 1.0))
        p_blend = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        oof_p_fused[val_idx] = p_blend
        
        bacc_q = balanced_accuracy_score(y_raw[val_idx], [VALID_18_MODELS[i] for i in np.argmax(oof_p_quotient[val_idx], axis=1)])
        bacc_ng = balanced_accuracy_score(y_raw[val_idx], [VALID_18_MODELS[i] for i in np.argmax(oof_p_ngrams[val_idx], axis=1)])
        bacc_fused = balanced_accuracy_score(y_raw[val_idx], [VALID_18_MODELS[i] for i in np.argmax(p_blend, axis=1)])
        
        print(f"Fold {fold}: Quotient/Sty = {bacc_q:.4f} | 4-Views = {bacc_ng:.4f} | Fused = {bacc_fused:.4f} ({time.time() - t_fold:.1f}s)")
        
    # Evaluate OOF before Sinkhorn
    pred_raw_fused = [VALID_18_MODELS[i] for i in np.argmax(oof_p_fused, axis=1)]
    rep_raw = evaluate_predictions(y_raw, np.array(pred_raw_fused))
    print("\n" + "="*50)
    print("FUSED PIPELINE (BEFORE SINKHORN):")
    print(format_evaluation_summary(rep_raw))
    
    # Apply Sinkhorn-Knopp Marginal Balancing
    print("\nApplying 100-Iteration Sinkhorn-Knopp Marginal Balancing...")
    oof_p_balanced = sinkhorn_knopp_balance(oof_p_fused, iters=100)
    pred_balanced = [VALID_18_MODELS[i] for i in np.argmax(oof_p_balanced, axis=1)]
    rep_balanced = evaluate_predictions(y_raw, np.array(pred_balanced))
    
    print("\n" + "="*50)
    print("FINAL FUSED PIPELINE (WITH SINKHORN BALANCING):")
    print(format_evaluation_summary(rep_balanced))
    
    print(f"\n[SCORE COMPARISON]")
    print(f"Previous Best Baseline: 0.3689 (CV) | 0.355 (LB)")
    print(f"New Fused Pipeline:     {rep_balanced['balanced_accuracy']:.4f} (CV)")
    print(f"Gain:                  {rep_balanced['balanced_accuracy'] - 0.3689:+.4f} Balanced Accuracy")
    
    # Save OOF artifacts
    np.save("/kaggle/working/oof_exp08_probs.npy", oof_p_balanced)
    np.save("/kaggle/working/oof_exp08_preds.npy", pred_balanced)


if __name__ == "__main__":
    run_experiment_08()
