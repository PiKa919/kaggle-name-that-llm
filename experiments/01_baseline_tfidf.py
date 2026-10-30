"""
Experiment 01: Fast Baseline Comparison (Starter vs Optimized Case-Preserving TF-IDF)
Executed on Kaggle Dual T4 environment.
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# Ensure src is importable
sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_LAB
from src.cv import split_folds
from src.metrics import evaluate_predictions, format_evaluation_summary


def run_starter_baseline(df: pd.DataFrame, folds: np.ndarray) -> np.ndarray:
    """Run baseline matching starter notebook (lowercasing, small vocab)."""
    print("\n--- Running Starter Baseline (lowercase=True, 25k word / 35k char) ---")
    oof = np.empty(len(df), dtype=object)
    
    for fold in range(5):
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=35000, sublinear_tf=True)
        
        Xtr = hstack([word_vec.fit_transform(df["text"].iloc[trn_idx]), char_vec.fit_transform(df["text"].iloc[trn_idx])]).tocsr()
        Xva = hstack([word_vec.transform(df["text"].iloc[val_idx]), char_vec.transform(df["text"].iloc[val_idx])]).tocsr()
        
        clf = LogisticRegression(C=2.0, max_iter=1000)
        clf.fit(Xtr, df["model"].iloc[trn_idx])
        oof[val_idx] = clf.predict(Xva)
        
    return oof


def run_optimized_baseline(df: pd.DataFrame, folds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run optimized TF-IDF baseline (lowercase=False, 200k word / 300k char, punctuation preserved)."""
    print("\n--- Running Optimized Baseline (lowercase=False, 200k word / 300k char) ---")
    oof_preds = np.empty(len(df), dtype=object)
    oof_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    
    cfgs = [
        dict(analyzer="char", ngram_range=(1, 5), max_features=300_000),
        dict(analyzer="word", ngram_range=(1, 2), max_features=200_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
    ]
    
    models_fitted = []
    vectorizers_fitted = []
    
    for fold in range(5):
        t0 = time.time()
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **c) for c in cfgs]
        Xtr = hstack([v.fit_transform(df["text"].iloc[trn_idx]) for v in vs]).tocsr()
        Xva = hstack([v.transform(df["text"].iloc[val_idx]) for v in vs]).tocsr()
        
        clf = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
        clf.fit(Xtr, df["model"].iloc[trn_idx])
        
        oof_preds[val_idx] = clf.predict(Xva)
        oof_probs[val_idx] = clf.predict_proba(Xva)
        
        fold_bacc = balanced_accuracy_score(df["model"].iloc[val_idx], oof_preds[val_idx])
        print(f"Fold {fold} Balanced Accuracy: {fold_bacc:.4f} ({time.time() - t0:.1f}s)")
        
    return oof_preds, oof_probs


def main():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    print(f"Train samples: {len(df)} across {df['prompt_id'].nunique()} prompt groups.")
    
    folds = split_folds(df)
    
    # 1. Starter Baseline
    starter_oof = run_starter_baseline(df, folds)
    starter_report = evaluate_predictions(df["model"].values, starter_oof)
    print("\n" + "="*50)
    print("STARTER BASELINE REPORT:")
    print(format_evaluation_summary(starter_report))
    
    # 2. Optimized Baseline
    opt_oof, opt_probs = run_optimized_baseline(df, folds)
    opt_report = evaluate_predictions(df["model"].values, opt_oof)
    print("\n" + "="*50)
    print("OPTIMIZED BASELINE REPORT:")
    print(format_evaluation_summary(opt_report))
    
    # Save OOF artifacts
    os.makedirs("/kaggle/working/experiments", exist_ok=True)
    np.save("/kaggle/working/experiments/oof_exp01_preds.npy", opt_oof)
    np.save("/kaggle/working/experiments/oof_exp01_probs.npy", opt_probs)
    
    # Test submission generation check
    test_df = pd.read_csv(TEST_CSV)
    test_df["text"] = test_df["text"].fillna("")
    
    # Train full model on 100% data
    cfgs = [
        dict(analyzer="char", ngram_range=(1, 5), max_features=300_000),
        dict(analyzer="word", ngram_range=(1, 2), max_features=200_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
    ]
    vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **c) for c in cfgs]
    X_full = hstack([v.fit_transform(df["text"]) for v in vs]).tocsr()
    X_test = hstack([v.transform(test_df["text"]) for v in vs]).tocsr()
    
    full_clf = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
    full_clf.fit(X_full, df["model"])
    test_preds = full_clf.predict(X_test)
    
    sub = pd.DataFrame({"id": test_df["id"], "model": test_preds})
    sub.to_parquet("/kaggle/working/submission.parquet", index=False)
    
    # Validation assertions
    assert sub.shape == (len(test_df), 2), f"Expected shape {(len(test_df), 2)}, got {sub.shape}"
    assert list(sub.columns) == ["id", "model"]
    assert sub["model"].isin(VALID_18_MODELS).all(), "Invalid model labels in submission!"
    assert not sub.isnull().any().any(), "NaN values found in submission!"
    print(f"\n[OK] Submission verified and saved: {len(sub)} rows in /kaggle/working/submission.parquet")


if __name__ == "__main__":
    main()
