"""
Experiment 03: Extended Stylometrics + Multi-Model Stacking Meta-Learner
Integrates recommendations from GLM-Stylometry-Guide and GLM-Deeplearning-Guide.
Executed on Kaggle Dual T4.
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.features import extract_stylometric_features
from src.metrics import evaluate_predictions, format_evaluation_summary


def run_experiment_03():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    
    folds = split_folds(df)
    
    # 1. Extract Extended Stylometric Dense Features
    print("Extracting extended stylometric, typographic & syntactic features...")
    t0 = time.time()
    sty_df = extract_stylometric_features(df["text"].tolist())
    sty_feat_names = list(sty_df.columns)
    X_sty_raw = sty_df.values.astype(np.float32)
    scaler = StandardScaler()
    X_sty = scaler.fit_transform(X_sty_raw)
    print(f"Extracted {len(sty_feat_names)} extended features in {time.time() - t0:.1f}s.")
    
    # 2. Extract Multi-Ngram Sparse TF-IDF Features
    cfgs = [
        dict(analyzer="char", ngram_range=(1, 5), max_features=300_000),
        dict(analyzer="word", ngram_range=(1, 2), max_features=200_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
    ]
    
    oof_tfidf_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_lgb_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_stacked_probs = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_stacked_preds = np.empty(len(df), dtype=object)
    
    print("\n--- Running 5-Fold StratifiedGroupKFold on Kaggle 2xT4 ---")
    for fold in range(5):
        t_fold = time.time()
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # A. Base Model 1: Multi-Ngram TF-IDF + LogisticRegression
        vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **c) for c in cfgs]
        Xtr_tf = hstack([v.fit_transform(df["text"].iloc[trn_idx]) for v in vs]).tocsr()
        Xva_tf = hstack([v.transform(df["text"].iloc[val_idx]) for v in vs]).tocsr()
        
        clf_tf = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
        clf_tf.fit(Xtr_tf, y_raw[trn_idx])
        oof_tfidf_probs[val_idx] = clf_tf.predict_proba(Xva_tf)
        
        # B. Base Model 2: LightGBM on Extended Stylometric Features
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
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
        oof_lgb_probs[val_idx] = clf_lgb.predict_proba(X_sty[val_idx])
        
        # C. Meta-Learner: Nested Stacking on Out-of-Fold representations
        # Concatenate TF-IDF probabilities, LightGBM probabilities, and key stylometric features
        X_meta_tr = np.hstack([
            clf_tf.predict_proba(Xtr_tf),
            clf_lgb.predict_proba(X_sty[trn_idx]),
            X_sty[trn_idx][:, :20],  # Top 20 typography signals
        ])
        X_meta_va = np.hstack([
            oof_tfidf_probs[val_idx],
            oof_lgb_probs[val_idx],
            X_sty[val_idx][:, :20],
        ])
        
        meta_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        meta_clf.fit(X_meta_tr, y_raw[trn_idx])
        
        oof_stacked_probs[val_idx] = meta_clf.predict_proba(X_meta_va)
        oof_stacked_preds[val_idx] = meta_clf.predict(X_meta_va)
        
        bacc_tf = balanced_accuracy_score(y_raw[val_idx], [VALID_18_MODELS[i] for i in np.argmax(oof_tfidf_probs[val_idx], axis=1)])
        bacc_lgb = balanced_accuracy_score(y_raw[val_idx], [VALID_18_MODELS[i] for i in np.argmax(oof_lgb_probs[val_idx], axis=1)])
        bacc_stack = balanced_accuracy_score(y_raw[val_idx], oof_stacked_preds[val_idx])
        
        print(f"Fold {fold}: TF-IDF BAcc = {bacc_tf:.4f} | LGBM BAcc = {bacc_lgb:.4f} | Stacking Meta BAcc = {bacc_stack:.4f} ({time.time() - t_fold:.1f}s)")
        
    # Global Stacked Meta-Learner Evaluation
    rep_stacked = evaluate_predictions(y_raw, oof_stacked_preds)
    print("\n" + "="*50)
    print("STACKED META-LEARNER ENSEMBLE REPORT:")
    print(format_evaluation_summary(rep_stacked))
    
    # Save artifacts
    np.save("/kaggle/working/experiments/oof_exp03_probs.npy", oof_stacked_probs)
    np.save("/kaggle/working/experiments/oof_exp03_preds.npy", oof_stacked_preds)
    
    # Train full meta-ensemble for test submission
    test_df = pd.read_csv(TEST_CSV)
    test_df["text"] = test_df["text"].fillna("")
    test_sty_raw = extract_stylometric_features(test_df["text"].tolist()).values.astype(np.float32)
    test_sty = scaler.transform(test_sty_raw)
    
    vs = [TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **c) for c in cfgs]
    X_full_tf = hstack([v.fit_transform(df["text"]) for v in vs]).tocsr()
    X_test_tf = hstack([v.transform(test_df["text"]) for v in vs]).tocsr()
    
    clf_tf_full = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")
    clf_tf_full.fit(X_full_tf, y_raw)
    
    clf_lgb_full = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
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
    
    X_meta_full = np.hstack([
        clf_tf_full.predict_proba(X_full_tf),
        clf_lgb_full.predict_proba(X_sty),
        X_sty[:, :20],
    ])
    X_meta_test = np.hstack([
        clf_tf_full.predict_proba(X_test_tf),
        clf_lgb_full.predict_proba(test_sty),
        test_sty[:, :20],
    ])
    
    meta_clf_full = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    meta_clf_full.fit(X_meta_full, y_raw)
    test_preds = meta_clf_full.predict(X_meta_test)
    
    sub = pd.DataFrame({"id": test_df["id"], "model": test_preds})
    sub.to_parquet("/kaggle/working/submission.parquet", index=False)
    assert sub.shape == (len(test_df), 2)
    assert list(sub.columns) == ["id", "model"]
    assert sub["model"].isin(VALID_18_MODELS).all()
    assert not sub.isnull().any().any()
    print(f"\n[OK] Submission verified and updated in /kaggle/working/submission.parquet ({len(sub)} rows)")


if __name__ == "__main__":
    run_experiment_03()
