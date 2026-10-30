"""
Experiment 07: Crossed Two-Way ANOVA Centering + Canonical LDA (17-Dim Quotient Space)
Prescribed by DeepSeek Brutal Teacher.
Runs in seconds on cached Qwen3 embeddings.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.linalg as sla
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, VALID_18_MODELS, MODEL_TO_IDX
from src.cv import split_folds
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


def compute_eta_squared_gate(Z: np.ndarray, prompt_ids: np.ndarray, model_labels: np.ndarray):
    """DeepSeek Proof Gate: calculate variance explained by prompt vs model."""
    print("\n--- Running DeepSeek Proof Gate (η² Analysis) ---")
    grand_mean = Z.mean(0)
    ss_tot = ((Z - grand_mean) ** 2).sum(0)
    ss_tot = np.where(ss_tot > 1e-8, ss_tot, 1.0)
    
    # Prompt eta^2
    _, p_inv = np.unique(prompt_ids, return_inverse=True)
    p_means = np.zeros((len(np.unique(prompt_ids)), Z.shape[1]))
    np.add.at(p_means, p_inv, Z)
    p_means /= np.bincount(p_inv)[:, None]
    ss_prompt = np.sum(np.bincount(p_inv)[:, None] * ((p_means - grand_mean) ** 2), axis=0)
    eta2_prompt = ss_prompt / ss_tot
    
    # Model eta^2
    _, m_inv = np.unique(model_labels, return_inverse=True)
    m_means = np.zeros((len(np.unique(model_labels)), Z.shape[1]))
    np.add.at(m_means, m_inv, Z)
    m_means /= np.bincount(m_inv)[:, None]
    ss_model = np.sum(np.bincount(m_inv)[:, None] * ((m_means - grand_mean) ** 2), axis=0)
    eta2_model = ss_model / ss_tot
    
    print(f"Mean Prompt η² (Topic Confound): {eta2_prompt.mean():.4f}")
    print(f"Mean Model η² (Author Signal):  {eta2_model.mean():.4f}")
    print(f"Fraction of dims where Topic > Author: {(eta2_prompt > eta2_model).mean() * 100:.1f}%")
    print(f"Ratio (Topic / Author): {eta2_prompt.mean() / max(eta2_model.mean(), 1e-6):.1f}x dominance")


def run_experiment_07():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    prompts = df["prompt_id"].values
    
    feat_cache = "/kaggle/working/qwen_probe_features.npy"
    if not os.path.exists(feat_cache):
        raise FileNotFoundError(f"Missing {feat_cache}! Run exp 06 first.")
        
    print(f"Loading cached features from {feat_cache}...")
    Z = np.load(feat_cache)
    print(f"Raw ambient matrix: {Z.shape}")
    
    # Run the Proof Gate
    compute_eta_squared_gate(Z, prompts, y_raw)
    
    print("\n--- Applying Crossed Two-Way ANOVA Centering ---")
    t0 = time.time()
    Zc = anova_center(Z, prompts)
    print(f"ANOVA centered in {time.time() - t0:.2f}s. Centered shape: {Zc.shape}")
    
    # Verify topic is annihilated
    compute_eta_squared_gate(Zc, prompts, y_raw)
    
    # 5-fold StratifiedGroupKFold
    folds = split_folds(df)
    oof_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_preds = np.empty(len(df), dtype=object)
    
    print("\n--- Running 5-Fold Canonical LDA in 17-Dim Quotient Space ---")
    eigenvalues_list = []
    
    for fold in range(5):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # 1. Project to 256 SVD subspace on training set
        _, _, Vt = randomized_svd(Zc[trn_idx], n_components=256, random_state=42)
        Y_trn = Zc[trn_idx] @ Vt.T
        Y_val = Zc[val_idx] @ Vt.T
        
        # 2. Canonical LDA Eigenproblem: Sb @ v = lambda * Sw @ v
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
            
        # Ledoit-Wolf-like shrinkage for Sw stability
        Sw += 1e-2 * (np.trace(Sw) / D_svd) * np.eye(D_svd)
        
        # Solve generalized eigenvalue problem
        ev, V = sla.eigh(Sb, Sw)
        # Sort descending
        idx_desc = np.argsort(ev)[::-1]
        ev = ev[idx_desc]
        V = V[:, idx_desc]
        
        # Keep top 17 canonical variates (K-1 = 17)
        V17 = V[:, :17]
        eigenvalues_list.append(ev[:17])
        
        # Project into 17-dimensional pure author identity space
        F_trn = Y_trn @ V17
        F_val = Y_val @ V17
        
        # Train well-posed calibrated classifier in 17-dim space
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(F_trn, y_raw[trn_idx])
        
        oof_probs[val_idx] = clf.predict_proba(F_val)
        pred_fold = clf.predict(F_val)
        oof_preds[val_idx] = pred_fold
        
        bacc_fold = balanced_accuracy_score(y_raw[val_idx], pred_fold)
        print(f"Fold {fold}: 17-Dim Canonical LDA BAcc = {bacc_fold:.4f} (Top eigenval: {ev[0]:.2f}, 17th: {ev[16]:.2f})")
        
    mean_ev = np.mean(eigenvalues_list, axis=0)
    print("\nMean Canonical Eigenvalue Spectrum (17 Canonical Dimensions):")
    print(np.round(mean_ev, 3))
    
    rep_lda = evaluate_predictions(y_raw, oof_preds)
    print("\n" + "="*50)
    print("CANONICAL 17-DIM QUOTIENT LDA REPORT:")
    print(format_evaluation_summary(rep_lda))
    
    delta = rep_lda["balanced_accuracy"] - 0.1874
    print(f"\n[VERDICT] Delta vs Raw Ambient Embeddings (0.1874): {delta:+.4f} Balanced Accuracy")
    print(f"Delta vs Baseline TF-IDF (0.3689): {rep_lda['balanced_accuracy'] - 0.3689:+.4f}")


if __name__ == "__main__":
    run_experiment_07()
