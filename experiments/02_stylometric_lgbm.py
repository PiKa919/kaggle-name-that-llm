"""
Experiment 02: Stylometric Feature Engineering + LightGBM + TF-IDF Probability Blend
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
import lightgbm as lgb

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.features import extract_stylometric_features
from src.metrics import evaluate_predictions, format_evaluation_summary


def run_experiment_02():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    
    folds = split_folds(df)
    
    # 1. Extract Stylometric Dense Features
    print("Extracting handcrafted stylometric & typographic features...")
    t0 = time.time()
    sty_df = extract_stylometric_features(df["text"].tolist())
    sty_feat_names = list(sty_df.columns)
    X_sty = sty_df.values.astype(np.float32)
    print(f"Extracted {len(sty_feat_names)} features in {time.time() - t0:.1f}s.")
    
    # 2. Extract TF-IDF Sparse Features
    cfgs = [
        dict(analyzer="char", ngram_range=(1, 5), max_features=250_000),
        dict(analyzer="word", ngram_range=(1, 2), max_features=150_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
    ]
    
    oof_tfidf_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_lgbm_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    
    print("\n--- Running 5-Fold StratifiedGroupKFold on Kaggle 2xT4 ---")
    for fold in range(5):
        t_fold = time.time()
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # A. TF-IDF + Logistic Regression
        vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **c) for c in cfgs]
        Xtr_tf = hstack([v.fit_transform(df["text"].iloc[trn_idx]) for v in vs]).tocsr()
        Xva_tf = hstack([v.transform(df["text"].iloc[val_idx]) for v in vs]).tocsr()
        
        clf_tf = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
        clf_tf.fit(Xtr_tf, y_raw[trn_idx])
        oof_tfidf_probs[val_idx] = clf_tf.predict_proba(Xva_tf)
        
        # B. LightGBM on Stylometric Dense Features
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
            verbose=-1,
        )
        clf_lgb.fit(X_sty[trn_idx], y_int[trn_idx])
        oof_lgbm_probs[val_idx] = clf_lgb.predict_proba(X_sty[val_idx])
        
        # Fold evaluation
        pred_tf_fold = [VALID_18_MODELS[i] for i in np.argmax(oof_tfidf_probs[val_idx], axis=1)]
        pred_lgb_fold = [VALID_18_MODELS[i] for i in np.argmax(oof_lgbm_probs[val_idx], axis=1)]
        bacc_tf = balanced_accuracy_score(y_raw[val_idx], pred_tf_fold)
        bacc_lgb = balanced_accuracy_score(y_raw[val_idx], pred_lgb_fold)
        
        print(f"Fold {fold}: TF-IDF BAcc = {bacc_tf:.4f} | Stylometric LightGBM BAcc = {bacc_lgb:.4f} ({time.time() - t_fold:.1f}s)")
        
    # Evaluate pure TF-IDF OOF
    pred_tfidf = [VALID_18_MODELS[i] for i in np.argmax(oof_tfidf_probs, axis=1)]
    rep_tfidf = evaluate_predictions(y_raw, np.array(pred_tfidf))
    print("\n" + "="*50)
    print("PURE TF-IDF REPORT:")
    print(format_evaluation_summary(rep_tfidf))
    
    # Evaluate pure LightGBM OOF
    pred_lgbm = [VALID_18_MODELS[i] for i in np.argmax(oof_lgbm_probs, axis=1)]
    rep_lgbm = evaluate_predictions(y_raw, np.array(pred_lgbm))
    print("\n" + "="*50)
    print("PURE STYLOMETRIC LIGHTGBM REPORT:")
    print(format_evaluation_summary(rep_lgbm))
    
    # Grid search optimal blend weight alpha: alpha * TFIDF + (1 - alpha) * LGBM
    print("\n--- Searching Optimal Ensemble Blend Weights ---")
    best_alpha = 0.5
    best_blend_bacc = 0.0
    for alpha in np.linspace(0.0, 1.0, 21):
        blend_probs = alpha * oof_tfidf_probs + (1.0 - alpha) * oof_lgbm_probs
        pred_blend = [VALID_18_MODELS[i] for i in np.argmax(blend_probs, axis=1)]
        bacc = balanced_accuracy_score(y_raw, pred_blend)
        if bacc > best_blend_bacc:
            best_blend_bacc = bacc
            best_alpha = alpha
        print(f"Weight Alpha={alpha:.2f} (TF-IDF) + {1.0-alpha:.2f} (LGBM) -> Balanced Accuracy: {bacc:.4f}")
        
    print(f"\nBest Ensemble Alpha: {best_alpha:.2f} with Balanced Accuracy: {best_blend_bacc:.4f}")
    
    # Final blend report
    final_probs = best_alpha * oof_tfidf_probs + (1.0 - best_alpha) * oof_lgbm_probs
    pred_final = np.array([VALID_18_MODELS[i] for i in np.argmax(final_probs, axis=1)])
    rep_final = evaluate_predictions(y_raw, pred_final)
    print("\n" + "="*50)
    print("FINAL BLENDED ENSEMBLE REPORT:")
    print(format_evaluation_summary(rep_final))
    
    # Feature importance from LightGBM (trained on full dataset)
    clf_lgb_full = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    clf_lgb_full.fit(X_sty, y_int)
    importances = pd.Series(clf_lgb_full.feature_importances_, index=sty_feat_names).sort_values(ascending=False)
    print("\nTop 15 Stylometric Features by Split Importance:")
    print(importances.head(15))
    
    # Save artifacts
    np.save("/kaggle/working/experiments/oof_exp02_probs.npy", final_probs)
    np.save("/kaggle/working/experiments/oof_exp02_preds.npy", pred_final)
    
    # Generate test submission
    test_df = pd.read_csv(TEST_CSV)
    test_df["text"] = test_df["text"].fillna("")
    test_sty = extract_stylometric_features(test_df["text"].tolist()).values.astype(np.float32)
    
    vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **c) for c in cfgs]
    X_full_tf = hstack([v.fit_transform(df["text"]) for v in vs]).tocsr()
    X_test_tf = hstack([v.transform(test_df["text"]) for v in vs]).tocsr()
    
    clf_tf_full = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
    clf_tf_full.fit(X_full_tf, y_raw)
    
    test_p_tf = clf_tf_full.predict_proba(X_test_tf)
    test_p_lgb = clf_lgb_full.predict_proba(test_sty)
    test_p_blend = best_alpha * test_p_tf + (1.0 - best_alpha) * test_p_lgb
    test_preds = [VALID_18_MODELS[i] for i in np.argmax(test_p_blend, axis=1)]
    
    sub = pd.DataFrame({"id": test_df["id"], "model": test_preds})
    sub.to_parquet("/kaggle/working/submission.parquet", index=False)
    print(f"\n[OK] Submission verified and updated in /kaggle/working/submission.parquet ({len(sub)} rows)")


if __name__ == "__main__":
    run_experiment_02()
