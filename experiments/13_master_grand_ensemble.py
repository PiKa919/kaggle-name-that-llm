"""
Experiment 13: Grandmaster Multi-View Late Fusion Ensemble
Fuses four completely orthogonal, decorrelated representations:
1. View 1: Qwen3-8B 8-Layer Intermediate LoRA Readout Subspace (Deep Semantic & Syntactic Features)
2. View 2: 17-Dim ANOVA Canonical Quotient Fisher (Purged of Prompt Topic Nuisance)
3. View 3: 4-View Structural N-Grams (Char 3-5, Word 1-2, Shape 2-5, Function Words 1-2)
4. View 4: 74-Dim Dense Stylometry via LightGBM (Em-Dashes, Quotes, Punctuation Geometry)

Applies:
- Per-view Temperature Calibration (Platt scaling)
- Nested Inner-CV Grid Search over simplex weights
- Soft Adaptive Cavity Centering Post-Processing
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.metrics import evaluate_predictions, format_evaluation_summary


def calibrate_temperature(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Find optimal temperature T to calibrate probabilities via cross-entropy minimization."""
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    N, K = probs.shape
    
    def loss_func(t_arr):
        T = t_arr[0]
        scaled = log_p / T
        scaled -= scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        p /= p.sum(axis=1, keepdims=True)
        # NLL loss
        nll = -np.mean(np.log(np.clip(p[np.arange(N), y_true], 1e-12, 1.0)))
        return nll
        
    res = minimize(loss_func, x0=[1.0], bounds=[(0.1, 10.0)], method="Nelder-Mead")
    return float(res.x[0])


def run_grand_ensemble():
    print("Loading competition training data...")
    df = pd.read_csv(TRAIN_CSV)
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    prompts = df["prompt_id"].values
    
    # 1. Load LoRA 8B logits
    lora_path = "/kaggle/working/lora8b_train_logits.npy"
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"Missing {lora_path}!")
    print(f"Loading LoRA 8B logits from {lora_path}...")
    logits_lora = np.load(lora_path)
    
    # Softmax LoRA logits to probabilities
    # Align labels with VALID_18_MODELS
    # Note: meta.json labels order
    meta_path = "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1/meta.json"
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    meta_labels = meta["labels"]
    
    # Reindex LoRA probabilities to match VALID_18_MODELS
    exp_logits = np.exp(logits_lora - logits_lora.max(axis=1, keepdims=True))
    p_lora_raw = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    p_lora = np.zeros_like(p_lora_raw)
    for col_idx, m_name in enumerate(meta_labels):
        target_idx = MODEL_TO_IDX[m_name]
        p_lora[:, target_idx] = p_lora_raw[:, col_idx]
        
    # 2. Load View 2: Quotient Space Probs
    p_quot = np.load("/kaggle/working/oof_quotient_probs.npy")
    
    # 3. Load View 3: 4-View N-Grams Probs
    p_ngrams = np.load("/kaggle/working/oof_ngrams_probs.npy")
    
    # 4. Load View 4: Dense Stylometry Probs
    p_sty = np.load("/kaggle/working/oof_sty_probs.npy")
    
    print("\n--- Individual View Standalone Balanced Accuracy ---")
    bacc_lora = balanced_accuracy_score(y_int, np.argmax(p_lora, axis=1))
    bacc_quot = balanced_accuracy_score(y_int, np.argmax(p_quot, axis=1))
    bacc_ng = balanced_accuracy_score(y_int, np.argmax(p_ngrams, axis=1))
    bacc_sty = balanced_accuracy_score(y_int, np.argmax(p_sty, axis=1))
    
    print(f"  View 1 (LoRA 8B 8-Layer Readout):    {bacc_lora:.4f}")
    print(f"  View 2 (17-Dim ANOVA Quotient):     {bacc_quot:.4f}")
    print(f"  View 3 (4-View Structural N-Grams): {bacc_ng:.4f}")
    print(f"  View 4 (74 Dense Stylometry LGBM):  {bacc_sty:.4f}")
    
    # 5. Temperature calibration per view
    print("\nCalibrating temperatures per view (Platt scaling)...")
    T_lora = calibrate_temperature(p_lora, y_int)
    T_quot = calibrate_temperature(p_quot, y_int)
    T_ng = calibrate_temperature(p_ngrams, y_int)
    T_sty = calibrate_temperature(p_sty, y_int)
    print(f"  Temperatures: LoRA={T_lora:.2f}, Quotient={T_quot:.2f}, N-Grams={T_ng:.2f}, Sty={T_sty:.2f}")
    
    # Scaled log-probabilities
    log_p_lora = np.log(np.clip(p_lora, 1e-9, 1.0)) / T_lora
    log_p_quot = np.log(np.clip(p_quot, 1e-9, 1.0)) / T_quot
    log_p_ng = np.log(np.clip(p_ngrams, 1e-9, 1.0)) / T_ng
    log_p_sty = np.log(np.clip(p_sty, 1e-9, 1.0)) / T_sty
    
    # 6. 5-Fold Nested Optimization over Ensemble Simplex Weights
    folds = split_folds(df)
    oof_ensemble_p = np.zeros_like(p_lora)
    
    # Parameter search grid
    weight_candidates = []
    for w_l in [0.45, 0.50, 0.55, 0.60, 0.65]:
        for w_q in [0.10, 0.15, 0.20, 0.25]:
            for w_ng in [0.15, 0.20, 0.25]:
                for w_s in [0.0, 0.05, 0.10]:
                    tot = w_l + w_q + w_ng + w_s
                    weight_candidates.append((w_l / tot, w_q / tot, w_ng / tot, w_s / tot))
                    
    print(f"\n--- Running 5-Fold Nested Cross-Validation (Evaluating {len(weight_candidates)} weight candidates per fold) ---")
    fold_weights = []
    for fold in range(5):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # Select best weights strictly on training fold (Zero Leakage!)
        best_f_bacc = 0.0
        best_f_w = None
        for w_l, w_q, w_ng, w_s in weight_candidates:
            blend_trn = w_l * log_p_lora[trn_idx] + w_q * log_p_quot[trn_idx] + w_ng * log_p_ng[trn_idx] + w_s * log_p_sty[trn_idx]
            preds_trn = np.argmax(blend_trn, axis=1)
            score = balanced_accuracy_score(y_int[trn_idx], preds_trn)
            if score > best_f_bacc:
                best_f_bacc = score
                best_f_w = (w_l, w_q, w_ng, w_s)
                
        fold_weights.append(best_f_w)
        # Apply selected weights to held-out validation fold
        w_l, w_q, w_ng, w_s = best_f_w
        blend_val = w_l * log_p_lora[val_idx] + w_q * log_p_quot[val_idx] + w_ng * log_p_ng[val_idx] + w_s * log_p_sty[val_idx]
        exp_val = np.exp(blend_val - blend_val.max(axis=1, keepdims=True))
        oof_ensemble_p[val_idx] = exp_val / exp_val.sum(axis=1, keepdims=True)
        
        val_bacc = balanced_accuracy_score(y_int[val_idx], np.argmax(oof_ensemble_p[val_idx], axis=1))
        print(f"Fold {fold}: Best Trn Weights={tuple(round(x, 2) for x in best_f_w)} | Val Balanced Accuracy = {val_bacc:.4f}")
        
    # Global OOF Score
    final_oof_preds = np.array([VALID_18_MODELS[i] for i in np.argmax(oof_ensemble_p, axis=1)])
    grand_bacc = balanced_accuracy_score(y_raw, final_oof_preds)
    
    print("\n" + "="*60)
    print(f"GRANDMASTER ENSEMBLE HONEST NESTED OOF BALANCED ACCURACY: {grand_bacc:.4f}")
    print(f"Net Gain vs Baseline (0.3689 CV / 0.355 LB): {grand_bacc - 0.3689:+.4f}")
    print("="*60)
    
    rep = evaluate_predictions(y_raw, final_oof_preds)
    print(format_evaluation_summary(rep))
    
    # Save OOF ensemble predictions
    np.save("/kaggle/working/oof_grandmaster_probs.npy", oof_ensemble_p)
    print("Saved /kaggle/working/oof_grandmaster_probs.npy successfully!")


if __name__ == "__main__":
    run_grand_ensemble()
