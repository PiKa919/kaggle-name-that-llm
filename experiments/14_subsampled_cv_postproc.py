"""
Experiment 14: Subsampled CV Harness & Fold-Wise Metric-Aligned Post-Processing
Prescribed by DeepSeek Brutal Teacher.

Evaluates:
1. Subsampled prompt-group robustness across p in {0.60, 0.75, 0.85, 1.00}.
2. Fold-wise class multipliers (coordinate ascent on Balanced Accuracy fitted strictly on training folds).
3. Soft partial matching via sinkhorn_matching_posterior with dummy rows.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.postproc import fit_class_multipliers, apply_multipliers, sinkhorn_matching_posterior


def evaluate_subsampled_cv(p_keep: float = 0.85, seed: int = 42):
    df = pd.read_csv(TRAIN_CSV)
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    prompts = df["prompt_id"].values
    
    # Load Grandmaster OOF probabilities
    oof_p = np.load("/kaggle/working/oof_grandmaster_probs.npy")
    folds = split_folds(df)
    
    rng = np.random.RandomState(seed)
    
    # Generate subsampling mask: for each prompt, keep each row with probability p_keep (minimum 3 rows)
    keep_mask = np.zeros(len(df), dtype=bool)
    for pid in np.unique(prompts):
        pid_indices = np.where(prompts == pid)[0]
        if p_keep >= 1.0:
            keep_mask[pid_indices] = True
        else:
            n_keep = max(int(np.round(len(pid_indices) * p_keep)), 3)
            chosen = rng.choice(pid_indices, size=n_keep, replace=False)
            keep_mask[chosen] = True
            
    print(f"\n--- Subsampled CV Evaluation (p={p_keep:.2f}, Retained {keep_mask.sum()}/{len(df)} passages) ---")
    
    # Baseline raw score on subsampled set
    sub_raw_preds = np.argmax(oof_p[keep_mask], axis=1)
    raw_bacc = balanced_accuracy_score(y_int[keep_mask], sub_raw_preds)
    print(f"Raw Blend Balanced Accuracy (p={p_keep:.2f}): {raw_bacc:.4f}")
    
    # Fold-wise Class Multipliers (strictly fit on training folds)
    oof_mult_preds = np.zeros(len(df), dtype=int)
    
    for fold in range(5):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where((folds == fold) & keep_mask)[0]
        
        # Fit multipliers strictly on training fold (un-subsampled or subsampled trn)
        mult, best_trn_score = fit_class_multipliers(
            y_true=y_int[trn_idx],
            probs=oof_p[trn_idx],
            n_passes=2
        )
        
        # Apply multipliers to held-out validation fold
        p_val_adj = apply_multipliers(oof_p[val_idx], mult)
        oof_mult_preds[val_idx] = np.argmax(p_val_adj, axis=1)
        
    mult_bacc = balanced_accuracy_score(y_int[keep_mask], oof_mult_preds[keep_mask])
    print(f"Fold-Wise Multipliers Balanced Accuracy (p={p_keep:.2f}): {mult_bacc:.4f} ({mult_bacc - raw_bacc:+.4f})")
    
    # Soft Sinkhorn with Dummy Rows per prompt cluster
    oof_sinkhorn_preds = np.zeros(len(df), dtype=int)
    
    for fold in range(5):
        trn_idx = np.where(folds != fold)[0]
        mult, _ = fit_class_multipliers(y_true=y_int[trn_idx], probs=oof_p[trn_idx], n_passes=2)
        
        val_prompts = np.unique(prompts[folds == fold])
        for pid in val_prompts:
            cluster_val_idx = np.where((prompts == pid) & keep_mask)[0]
            if len(cluster_val_idx) == 0:
                continue
                
            cluster_probs = oof_p[cluster_val_idx]
            # Apply class multipliers first
            cluster_adj = apply_multipliers(cluster_probs, mult)
            
            # Apply soft partial matching
            if len(cluster_val_idx) >= 4:
                cluster_logp = np.log(np.clip(cluster_adj, 1e-9, 1.0))
                # Soft Sinkhorn with beta=0.4, dummy_logit=-1.5
                p_matched = sinkhorn_matching_posterior(cluster_logp, beta=0.4, dummy_logit=-1.5)
                oof_sinkhorn_preds[cluster_val_idx] = np.argmax(p_matched, axis=1)
            else:
                oof_sinkhorn_preds[cluster_val_idx] = np.argmax(cluster_adj, axis=1)
                
    sk_bacc = balanced_accuracy_score(y_int[keep_mask], oof_sinkhorn_preds[keep_mask])
    print(f"Soft Sinkhorn Matching Balanced Accuracy (p={p_keep:.2f}): {sk_bacc:.4f} ({sk_bacc - raw_bacc:+.4f})")
    
    return raw_bacc, mult_bacc, sk_bacc


def run_experiment_14():
    print("=== EXPERIMENT 14: SUBSAMPLED CV & METRIC-ALIGNED POST-PROCESSING ===")
    results = {}
    for p in [1.00, 0.85, 0.75, 0.60]:
        results[p] = evaluate_subsampled_cv(p_keep=p, seed=42)
        
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL SUBSAMPLING REGIMES:")
    print("  p   | Raw Blend | Fold-Wise Multipliers | Soft Sinkhorn Matching")
    print("------+-----------+-----------------------+------------------------")
    for p, (r, m, s) in results.items():
        print(f" {p:.2f} |  {r:.4f}   |        {m:.4f}         |        {s:.4f}")
    print("="*60)


if __name__ == "__main__":
    run_experiment_14()
