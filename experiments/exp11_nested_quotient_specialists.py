"""
Experiment 11: Nested Quotient Fisher Hierarchy + LOO Cavity Centering + Unnormalized Verbosity View
Prescribed by DeepSeek Brutal Teacher.

Architecture:
1. LOO Cavity Centering: Scale-invariant prompt-topic annihilation without self-subtraction distortion.
2. Unnormalized Verbosity View: Raw sentence lengths, word counts, and punctuation ratios (breaks compression pairs).
3. Level 1: Lab-Level Fisher (18 -> 6 labs, 5 canonical dimensions).
4. Level 2: Per-Lab Conditional Specialist Micro-Heads:
   - Google Head (4 models -> 3 Fisher dims + verbosity)
   - OpenAI Head (5 models -> 4 Fisher dims + verbosity)
   - Anthropic Head (5 models -> 4 Fisher dims + typography)
   - ZAI Head (2 models -> 1 Fisher dim + compression features)
5. Chain-Rule Composition: P(model) = P(lab) * P(model | lab).
"""

import os
import sys
import time
import re
import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.extmath import randomized_svd
import lightgbm as lgb
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL, MODEL_TO_LAB
from src.cv import split_folds
from src.cavity import loo_cavity_center
from src.features import extract_stylometric_features
from src.metrics import evaluate_predictions, format_evaluation_summary


# Mapping 18 models to 6 labs
LABS = ["anthropic", "google", "openai", "qwen", "zai", "moonshot"]
LAB_TO_IDX = {l: i for i, l in enumerate(LABS)}
MODEL_TO_LAB_IDX = {m: LAB_TO_IDX[MODEL_TO_LAB[m]] for m in VALID_18_MODELS}

# Sibling groups per lab
LAB_MODELS = {lab: [m for m in VALID_18_MODELS if MODEL_TO_LAB[m] == lab] for lab in LABS}


def extract_unnormalized_verbosity_features(texts: list) -> np.ndarray:
    """Extract raw length, sentence length variance, and structural ratios."""
    feats = []
    for t in texts:
        words = t.split()
        n_words = len(words)
        n_chars = len(t)
        sentences = re.split(r"[.!?]+", t)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        n_sentences = max(len(sentences), 1)
        sent_lens = [len(s.split()) for s in sentences]
        mean_sent_len = np.mean(sent_lens) if sent_lens else 0.0
        std_sent_len = np.std(sent_lens) if sent_lens else 0.0
        
        # Punctuation counts
        c_em_dash = t.count("—")
        c_en_dash = t.count("–")
        c_colon = t.count(":")
        c_semicolon = t.count(";")
        c_comma = t.count(",")
        c_curly_sq = t.count("’") + t.count("‘")
        c_straight_sq = t.count("'")
        c_curly_dq = t.count("“") + t.count("”")
        c_straight_dq = t.count('"')
        c_digits = sum(c.isdigit() for c in t)
        
        row = [
            n_words,
            n_chars,
            n_sentences,
            mean_sent_len,
            std_sent_len,
            c_comma / max(n_words, 1),
            c_em_dash,
            c_en_dash,
            c_colon,
            c_semicolon,
            c_curly_sq,
            c_straight_sq,
            c_curly_dq,
            c_straight_dq,
            c_digits / max(n_chars, 1),
        ]
        feats.append(row)
    return np.array(feats, dtype=np.float32)


def run_experiment_11():
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    y_lab = np.array([MODEL_TO_LAB_IDX[m] for m in y_raw])
    prompts = df["prompt_id"].values
    
    # 1. Load cached Qwen3 embeddings
    feat_cache = "/kaggle/working/qwen_probe_features.npy"
    if not os.path.exists(feat_cache):
        raise FileNotFoundError(f"Missing {feat_cache}!")
    print(f"Loading cached Qwen3 embeddings from {feat_cache}...")
    Z = np.load(feat_cache)
    
    # 2. Extract dense stylometry and unnormalized verbosity
    print("Extracting 74 dense stylometric features...")
    sty_df = extract_stylometric_features(df["text"].tolist())
    X_sty = sty_df.values.astype(np.float32)
    
    print("Extracting unnormalized verbosity features...")
    X_verb = extract_unnormalized_verbosity_features(df["text"].tolist())
    
    # 3. Apply LOO Cavity Centering (Scale-Invariant!)
    print("Applying Scale-Invariant LOO Cavity Centering...")
    t0 = time.time()
    Z_cav = loo_cavity_center(Z, prompts)
    print(f"LOO Cavity Centered in {time.time() - t0:.2f}s")
    
    # 4. 4-View specifications
    FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
    def shape_transform(text: str) -> str:
        return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)
    def function_transform(text: str) -> str:
        return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                        for w in re.findall(r"\w+|[^\w\s]", text))
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s),
        ("word", 1, 2, 120_000, 0.8, lambda s: s),
        ("shape", 2, 5, 40_000, 0.5, shape_transform),
        ("function", 1, 2, 30_000, 0.5, function_transform),
    ]
    
    folds = split_folds(df)
    
    oof_p_hierarchical = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_p_global_quot = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_p_ngrams = np.zeros((len(df), len(VALID_18_MODELS)), dtype=np.float32)
    oof_p_lab = np.zeros((len(df), len(LABS)), dtype=np.float32)
    
    print("\n--- Running 5-Fold StratifiedGroupKFold on Nested Quotient Architecture ---")
    for fold in range(5):
        t_fold = time.time()
        trn_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        
        # SVD basis on training cavity embeddings
        _, _, Vt = randomized_svd(Z_cav[trn_idx], n_components=256, random_state=42)
        Y_trn = Z_cav[trn_idx] @ Vt.T
        Y_val = Z_cav[val_idx] @ Vt.T
        D_svd = Y_trn.shape[1]
        
        # --- LEVEL 1: Lab-Level Fisher Discriminant (18 -> 6 Labs) ---
        Sb_lab = np.zeros((D_svd, D_svd), dtype=np.float64)
        Sw_lab = np.zeros((D_svd, D_svd), dtype=np.float64)
        mu_trn = Y_trn.mean(axis=0)
        
        for l_idx in range(len(LABS)):
            Yl = Y_trn[y_lab[trn_idx] == l_idx]
            if len(Yl) == 0:
                continue
            ml = Yl.mean(axis=0)
            diff = ml - mu_trn
            Sb_lab += len(Yl) * np.outer(diff, diff)
            within_diff = Yl - ml
            Sw_lab += within_diff.T @ within_diff
            
        Sw_lab += 1e-2 * (np.trace(Sw_lab) / D_svd) * np.eye(D_svd)
        ev_l, V_l = sla.eigh(Sb_lab, Sw_lab)
        V_lab5 = V_l[:, np.argsort(ev_l)[::-1]][:, :5]
        
        F_lab_trn = np.hstack([Y_trn @ V_lab5, X_verb[trn_idx]])
        F_lab_val = np.hstack([Y_val @ V_lab5, X_verb[val_idx]])
        
        clf_lab = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
        clf_lab.fit(F_lab_trn, y_lab[trn_idx])
        oof_p_lab[val_idx] = clf_lab.predict_proba(F_lab_val)
        
        # --- Global 17-Dim Model Fisher (for baseline comparison) ---
        Sb_m = np.zeros((D_svd, D_svd), dtype=np.float64)
        Sw_m = np.zeros((D_svd, D_svd), dtype=np.float64)
        for k in range(len(VALID_18_MODELS)):
            Yk = Y_trn[y_int[trn_idx] == k]
            if len(Yk) == 0:
                continue
            mk = Yk.mean(axis=0)
            diff = mk - mu_trn
            Sb_m += len(Yk) * np.outer(diff, diff)
            within_diff = Yk - mk
            Sw_m += within_diff.T @ within_diff
        Sw_m += 1e-2 * (np.trace(Sw_m) / D_svd) * np.eye(D_svd)
        ev_m, V_m = sla.eigh(Sb_m, Sw_m)
        V_m17 = V_m[:, np.argsort(ev_m)[::-1]][:, :17]
        
        clf_qm = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        clf_qm.fit(Y_trn @ V_m17, y_int[trn_idx])
        oof_p_global_quot[val_idx] = clf_qm.predict_proba(Y_val @ V_m17)
        
        # --- LEVEL 2: Sibling Specialist Micro-Heads Per Lab ---
        # Initialize conditional probabilities P(model | lab)
        p_cond = np.zeros((len(val_idx), len(VALID_18_MODELS)), dtype=np.float32)
        
        for lab_name, sib_models in LAB_MODELS.items():
            l_idx = LAB_TO_IDX[lab_name]
            if len(sib_models) == 1:
                # Deterministic conditional probability = 1.0 for singletons (Qwen, Moonshot)
                m_idx = MODEL_TO_IDX[sib_models[0]]
                p_cond[:, m_idx] = 1.0
                continue
                
            # Filter training rows belonging strictly to this lab
            lab_mask_trn = (y_lab[trn_idx] == l_idx)
            sub_trn_idx = trn_idx[lab_mask_trn]
            y_sub_trn = y_int[sub_trn_idx]
            
            # Sub-model mapping
            sub_models_idx = [MODEL_TO_IDX[m] for m in sib_models]
            sub_label_map = {orig_k: sub_k for sub_k, orig_k in enumerate(sub_models_idx)}
            y_sub_local = np.array([sub_label_map[y] for y in y_sub_trn])
            
            # Compute within-lab Fisher contrast:
            Y_sub_trn = Y_trn[lab_mask_trn]
            mu_lab_trn = Y_sub_trn.mean(axis=0)
            D_sub = Y_sub_trn.shape[1]
            
            Sb_sub = np.zeros((D_sub, D_sub), dtype=np.float64)
            Sw_sub = np.zeros((D_sub, D_sub), dtype=np.float64)
            for sub_k, orig_k in enumerate(sub_models_idx):
                Y_k = Y_sub_trn[y_sub_local == sub_k]
                if len(Y_k) == 0:
                    continue
                mk = Y_k.mean(axis=0)
                diff = mk - mu_lab_trn
                Sb_sub += len(Y_k) * np.outer(diff, diff)
                w_diff = Y_k - mk
                Sw_sub += w_diff.T @ w_diff
                
            Sw_sub += 1e-2 * (np.trace(Sw_sub) / D_sub) * np.eye(D_sub)
            ev_sub, V_sub = sla.eigh(Sb_sub, Sw_sub)
            k_dims = len(sib_models) - 1
            V_sub_k = V_sub[:, np.argsort(ev_sub)[::-1]][:, :k_dims]
            
            # Feature representation for specialist head: Fisher canonical + unnormalized verbosity
            F_sub_trn = np.hstack([Y_sub_trn @ V_sub_k, X_verb[sub_trn_idx]])
            F_sub_val = np.hstack([Y_val @ V_sub_k, X_verb[val_idx]])
            
            clf_sub = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=42)
            clf_sub.fit(F_sub_trn, y_sub_local)
            sub_probs = clf_sub.predict_proba(F_sub_val)  # shape (val_len, len(sib_models))
            
            for sub_k, orig_k in enumerate(sub_models_idx):
                p_cond[:, orig_k] = sub_probs[:, sub_k]
                
        # --- LEVEL 3: Chain-Rule Hierarchical Composition ---
        # P(model) = P(lab) * P(model | lab)
        for m_idx, m_name in enumerate(VALID_18_MODELS):
            l_idx = MODEL_TO_LAB_IDX[m_name]
            oof_p_hierarchical[val_idx, m_idx] = oof_p_lab[val_idx, l_idx] * p_cond[:, m_idx]
            
        # Normalize rows
        oof_p_hierarchical[val_idx] /= oof_p_hierarchical[val_idx].sum(axis=1, keepdims=True)
        
        # --- View 3: 4-View N-Grams ---
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
        
        bacc_lab = balanced_accuracy_score(y_lab[val_idx], np.argmax(oof_p_lab[val_idx], axis=1))
        bacc_hier = balanced_accuracy_score(y_int[val_idx], np.argmax(oof_p_hierarchical[val_idx], axis=1))
        bacc_glob = balanced_accuracy_score(y_int[val_idx], np.argmax(oof_p_global_quot[val_idx], axis=1))
        bacc_ng = balanced_accuracy_score(y_int[val_idx], np.argmax(oof_p_ngrams[val_idx], axis=1))
        print(f"Fold {fold}: Lab Acc = {bacc_lab:.4f} | Global Fisher = {bacc_glob:.4f} | Hierarchical Sibling = {bacc_hier:.4f} | 4-Views = {bacc_ng:.4f} ({time.time() - t_fold:.1f}s)")
        
    print("\n" + "="*50)
    print("SUMMARY OF ARCHITECTURAL METRICS (OOF):")
    lab_acc = balanced_accuracy_score(y_lab, np.argmax(oof_p_lab, axis=1))
    glob_acc = balanced_accuracy_score(y_int, np.argmax(oof_p_global_quot, axis=1))
    hier_acc = balanced_accuracy_score(y_int, np.argmax(oof_p_hierarchical, axis=1))
    ng_acc = balanced_accuracy_score(y_int, np.argmax(oof_p_ngrams, axis=1))
    
    print(f"Lab Attribution Accuracy (6 Labs):             {lab_acc:.4f}")
    print(f"Global 17-Dim Quotient Space:                  {glob_acc:.4f}")
    print(f"Nested Hierarchical Specialist Model alone:   {hier_acc:.4f} (+{hier_acc - glob_acc:+.4f} vs Global)")
    print(f"4-View Structural N-Grams alone:               {ng_acc:.4f}")
    
    # Blending Hierarchical Specialists with 4-View N-Grams
    print("\n--- Evaluating Hierarchical + N-Gram Fusion (Nested Grid) ---")
    best_bacc = 0.0
    best_w = None
    for w_hier in np.linspace(0.40, 0.70, 7):
        w_ng = round(1.0 - w_hier, 2)
        log_p = w_hier * np.log(np.clip(oof_p_hierarchical, 1e-9, 1.0)) + w_ng * np.log(np.clip(oof_p_ngrams, 1e-9, 1.0))
        p_blend = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        preds = np.argmax(p_blend, axis=1)
        score = balanced_accuracy_score(y_int, preds)
        if score > best_bacc:
            best_bacc = score
            best_w = (w_hier, w_ng)
            
    print(f"Optimal Fusion: Hierarchical={best_w[0]:.2f}, N-Grams={best_w[1]:.2f} -> Balanced Accuracy: {best_bacc:.4f} (+{best_bacc - 0.3689:+.4f} vs 0.3689 baseline)")
    
    # Full report on best blend
    final_log_p = best_w[0] * np.log(np.clip(oof_p_hierarchical, 1e-9, 1.0)) + best_w[1] * np.log(np.clip(oof_p_ngrams, 1e-9, 1.0))
    final_p = np.exp(final_log_p - final_log_p.max(axis=1, keepdims=True))
    final_p /= final_p.sum(axis=1, keepdims=True)
    final_preds = np.array([VALID_18_MODELS[i] for i in np.argmax(final_p, axis=1)])
    
    rep = evaluate_predictions(y_raw, final_preds)
    print("\n" + "="*50)
    print("FINAL GRAND REPORT (EXPERIMENT 11):")
    print(format_evaluation_summary(rep))


if __name__ == "__main__":
    run_experiment_11()
