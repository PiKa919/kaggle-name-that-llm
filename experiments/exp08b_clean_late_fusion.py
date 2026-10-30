"""
Experiment 08b: Clean Late Fusion of Three Pristine Decorrelated Views
1. View 1: 17-Dim ANOVA Canonical Quotient Space (0.394 CV alone)
2. View 2: 74-Dim Handcrafted Dense Stylometry via LightGBM (0.255 CV alone)
3. View 3: 4-View Structural N-Grams (Char, Word, Shape, Function) via LinearSVC (0.391 CV alone)
Followed by 100-iteration Sinkhorn-Knopp marginal balancing.
"""

import os
import re
import sys
import time
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.extmath import randomized_svd
import lightgbm as lgb
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.features import extract_stylometric_features
from src.metrics import evaluate_predictions, format_evaluation_summary


def anova_center(Z: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    unique_prompts, prompt_inv = np.unique(prompt_ids, return_inverse=True)
    G = len(unique_prompts)
    prompt_means = np.zeros((G, Z.shape[1]), dtype=np.float32)
    counts = np.bincount(prompt_inv, minlength=G)[:, None]
    np.add.at(prompt_means, prompt_inv, Z)
    prompt_means /= counts
    grand_mean = Z.mean(axis=0, keepdims=True)
    return Z - prompt_means[prompt_inv] + grand_mean


def sinkhorn_knopp_balance(P: np.ndarray, iters: int = 100) -> np.ndarray:
    P_out = P.copy()
    N, K = P_out.shape
    target = np.full(K, N / K, dtype=np.float64)
    for _ in range(iters):
        col_sums = P_out.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums > 1e-12, col_sums, 1.0)
        P_out *= target / col_sums
        row_sums = P_out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
        P_out /= row_sums
    return P_out


FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def shape_transform(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)

def function_transform(text: str) -> str:
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", text))


def run_experiment_08b():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    prompts = df["prompt_id"].values
    
    # 1. Load cached Qwen3 embeddings
    feat_cache = "/kaggle/working/qwen_probe_features.npy"
    if not os.path.exists(feat_cache):
        raise FileNotFoundError(f"Missing {feat_cache}!")
    print(f"Loading cached Qwen3 embeddings from {feat_cache}...")
    Z = np.load(feat_cache)
    
    # 2. Extract dense stylometry
    print("Extracting 74 dense stylometric features...")
    sty_df = extract_stylometric_features(df["text"].tolist())
    X_sty = sty_df.values.astype(np.float32)
    
    # 3. Crossed Two-Way ANOVA Centering on Qwen3
    print("Applying Crossed Two-Way ANOVA Centering...")
    Zc = anova_center(Z, prompts)
    
    # 4. 4-View specifications
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s),
        ("word", 1, 2, 120_000, 0.8, lambda s: s),
        ("shape", 2, 5, 40_000, 0.5, shape_transform),
        ("function", 1, 2, 30_000, 0.5, function_transform),
    ]
    
    folds = split_folds(df)
    oof_p_quotient = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_p_sty = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_p_ngrams = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    
    print("\n--- Running 5-Fold StratifiedGroupKFold on 3 Independent Views ---")
    for fold in range(5):
        t_fold = time.time()
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # View 1: Pristine 17-Dim Canonical Quotient Space (No concatenation!)
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
        
        clf_q = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        clf_q.fit(F_trn, y_int[trn_idx])
        oof_p_quotient[val_idx] = clf_q.predict_proba(F_val)
        
        # View 2: LightGBM on 74 Dense Stylometric Features (breaks GLM vs Claude!)
        clf_lgb = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(VALID_18_MODELS),
            n_estimators=150,
            learning_rate=0.05,
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
        oof_p_sty[val_idx] = clf_lgb.predict_proba(X_sty[val_idx])
        
        # View 3: 4-View Structural N-Grams
        matrices_trn, matrices_val = [], []
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
        clf_ng.fit(X_ng_trn, y_int[trn_idx])
        oof_p_ngrams[val_idx] = clf_ng.predict_proba(X_ng_val)
        
        bacc_q = balanced_accuracy_score(y_int[val_idx], np.argmax(oof_p_quotient[val_idx], axis=1))
        bacc_sty = balanced_accuracy_score(y_int[val_idx], np.argmax(oof_p_sty[val_idx], axis=1))
        bacc_ng = balanced_accuracy_score(y_int[val_idx], np.argmax(oof_p_ngrams[val_idx], axis=1))
        print(f"Fold {fold}: Quotient = {bacc_q:.4f} | Stylometry = {bacc_sty:.4f} | 4-Views = {bacc_ng:.4f} ({time.time() - t_fold:.1f}s)")
        
    print("\n" + "="*50)
    print("INDIVIDUAL VIEW PERFORMANCES (OOF Balanced Accuracy):")
    print(f"View 1 (ANOVA 17-Dim Quotient): {balanced_accuracy_score(y_int, np.argmax(oof_p_quotient, axis=1)):.4f}")
    print(f"View 2 (74 Dense Stylometry):   {balanced_accuracy_score(y_int, np.argmax(oof_p_sty, axis=1)):.4f}")
    print(f"View 3 (4-View N-Grams):        {balanced_accuracy_score(y_int, np.argmax(oof_p_ngrams, axis=1)):.4f}")
    
    # Grid search optimal late logit blend weights
    best_bacc = 0.0
    best_weights = None
    
    for w_q in [0.40, 0.45, 0.50, 0.55]:
        for w_sty in [0.10, 0.15, 0.20]:
            w_ng = round(1.0 - w_q - w_sty, 2)
            if w_ng < 0.20:
                continue
            log_p = w_q * np.log(np.clip(oof_p_quotient, 1e-9, 1.0)) + \
                    w_sty * np.log(np.clip(oof_p_sty, 1e-9, 1.0)) + \
                    w_ng * np.log(np.clip(oof_p_ngrams, 1e-9, 1.0))
            p_blend = np.exp(log_p - log_p.max(axis=1, keepdims=True))
            p_blend /= p_blend.sum(axis=1, keepdims=True)
            
            p_balanced = sinkhorn_knopp_balance(p_blend, iters=100)
            preds = [VALID_18_MODELS[i] for i in np.argmax(p_balanced, axis=1)]
            score = balanced_accuracy_score(y_raw, preds)
            if score > best_bacc:
                best_bacc = score
                best_weights = (w_q, w_sty, w_ng)
                
    print(f"\n[OPTIMAL LATE FUSION] Best Weights: Quotient={best_weights[0]}, Sty={best_weights[1]}, N-Grams={best_weights[2]}")
    print(f"Ensemble + Sinkhorn Balanced Accuracy: {best_bacc:.4f} (+{best_bacc - 0.3689:+.4f} vs 0.3689 baseline)")
    
    # Final full report
    w_q, w_sty, w_ng = best_weights
    final_log_p = w_q * np.log(np.clip(oof_p_quotient, 1e-9, 1.0)) + \
                  w_sty * np.log(np.clip(oof_p_sty, 1e-9, 1.0)) + \
                  w_ng * np.log(np.clip(oof_p_ngrams, 1e-9, 1.0))
    final_p = np.exp(final_log_p - final_log_p.max(axis=1, keepdims=True))
    final_p /= final_p.sum(axis=1, keepdims=True)
    final_balanced = sinkhorn_knopp_balance(final_p, iters=100)
    final_preds = np.array([VALID_18_MODELS[i] for i in np.argmax(final_balanced, axis=1)])
    
    # Save OOF predictions to disk
    np.save("/kaggle/working/oof_quotient_probs.npy", oof_p_quotient)
    np.save("/kaggle/working/oof_sty_probs.npy", oof_p_sty)
    np.save("/kaggle/working/oof_ngrams_probs.npy", oof_p_ngrams)
    np.save("/kaggle/working/oof_fused_balanced_probs.npy", final_balanced)
    print("Saved all OOF probability matrices to /kaggle/working/!")
    
    rep = evaluate_predictions(y_raw, final_preds)
    print("\n" + "="*50)
    print("FINAL GRAND REPORT:")
    print(format_evaluation_summary(rep))


if __name__ == "__main__":
    run_experiment_08b()
