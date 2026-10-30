"""
Name That LLM - 4-View Structural N-Grams & Dense Stylometry Late Fusion Submission
Runs entirely on CPU in under 60 seconds without GPU queue dependencies.
Outputs valid ./submission.parquet adhering strictly to Kaggle schema requirements.
"""

import math
import os
import re
import sys
import time
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from tqdm.auto import tqdm

# Strict Alphabetical Target Model Names (18 Classes)
VALID_18_MODELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
MODEL_TO_IDX = {m: i for i, m in enumerate(VALID_18_MODELS)}
IDX_TO_MODEL = {i: m for i, m in enumerate(VALID_18_MODELS)}

# Punctuation & Typographic constants
CURLY_SINGLE = set(["‘", "’"])
CURLY_DOUBLE = set(["“", "”"])
STRAIGHT_QUOTES = set(["'", '"'])
EM_DASH = "—"
EN_DASH = "–"

CONTRACTION_STRAIGHT = re.compile(r"\b[a-zA-Z]+'[a-zA-Z]+\b")
CONTRACTION_CURLY = re.compile(r"\b[a-zA-Z]+’[a-zA-Z]+\b")
EM_DASH_SPACED = re.compile(r"\s—\s|\s—\w|\w—\s")
EM_DASH_UNSPACED = re.compile(r"\w—\w")

INITIAL_CONJUNCTIONS = set(["and", "but", "yet", "so", "or"])
CLAUDE_MARKERS = set(["perhaps", "nuance", "underlying", "tapestry", "interplay", "woven", "profound", "notion"])
REASONING_MARKERS = set(["therefore", "thus", "hence", "step", "implies", "consequently", "assume", "furthermore"])
SUBORDINATORS = set(["because", "although", "whereas", "while", "since", "unless", "provided", "despite"])

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())


def shape_transform(text: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_transform(text: str) -> str:
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", text))


def extract_features_single(text: str) -> Dict[str, float]:
    feats = {}
    n_chars = len(text)
    words = text.split()
    n_words = len(words)
    w_safe = max(n_words, 1)
    c_safe = max(n_chars, 1)

    feats["char_count"] = n_chars
    feats["word_count"] = n_words

    c_single_curly = sum(text.count(c) for c in CURLY_SINGLE)
    c_double_curly = sum(text.count(c) for c in CURLY_DOUBLE)
    c_straight = sum(text.count(c) for c in STRAIGHT_QUOTES)

    feats["count_single_curly"] = c_single_curly
    feats["count_double_curly"] = c_double_curly
    feats["count_curly_total"] = c_single_curly + c_double_curly
    feats["count_straight_quotes"] = c_straight
    feats["rate_curly_single"] = c_single_curly / w_safe
    feats["rate_curly_double"] = c_double_curly / w_safe
    feats["rate_straight_quotes"] = c_straight / w_safe
    feats["curly_ratio"] = (c_single_curly + c_double_curly) / max(c_single_curly + c_double_curly + c_straight, 1)

    c_em_dash = text.count(EM_DASH)
    c_en_dash = text.count(EN_DASH)
    feats["count_em_dash"] = c_em_dash
    feats["count_en_dash"] = c_en_dash
    feats["rate_em_dash"] = c_em_dash / w_safe
    feats["has_em_dash"] = 1.0 if c_em_dash > 0 else 0.0

    feats["count_em_dash_spaced"] = len(EM_DASH_SPACED.findall(text))
    feats["count_em_dash_unspaced"] = len(EM_DASH_UNSPACED.findall(text))

    feats["count_semicolon"] = text.count(";")
    feats["count_colon"] = text.count(":")
    feats["count_comma"] = text.count(",")
    feats["rate_comma"] = text.count(",") / w_safe

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    n_sentences = max(len(sentences), 1)
    feats["sentence_count"] = n_sentences
    sent_lens = [len(s.split()) for s in sentences]
    feats["mean_sentence_length"] = np.mean(sent_lens) if sent_lens else 0.0
    feats["std_sentence_length"] = np.std(sent_lens) if sent_lens else 0.0

    lower_words = [w.lower() for w in words]
    vocab = set(lower_words)
    v_len = len(vocab)
    feats["ttr"] = v_len / w_safe

    return feats


def extract_features_df(texts: List[str]) -> pd.DataFrame:
    rows = [extract_features_single(t) for t in texts]
    return pd.DataFrame(rows)


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


def locate_file(candidates: List[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"None of candidate paths found: {candidates}")


def main():
    print("=== NAME THAT LLM: FUSED 4-VIEW N-GRAM + STYLOMETRY SUBMISSION ===")
    t_start = time.time()
    
    train_path = locate_file([
        "/kaggle/input/competitions/name-that-llm/train.csv",
        "/kaggle/input/name-that-llm/train.csv",
    ])
    test_path = locate_file([
        "/kaggle/input/competitions/name-that-llm/test.csv",
        "/kaggle/input/name-that-llm/test.csv",
    ])
    
    print(f"Reading train data from {train_path}...")
    train_df = pd.read_csv(train_path)
    train_df["text"] = train_df["text"].fillna("")
    y_raw = train_df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    
    print(f"Reading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    test_df["text"] = test_df["text"].fillna("")
    N_test = len(test_df)
    print(f"Test passages to classify: {N_test}")
    
    # 1. Dense Stylometric Features
    print("Extracting dense stylometric features...")
    X_sty_train = extract_features_df(train_df["text"].tolist()).values.astype(np.float32)
    X_sty_test = extract_features_df(test_df["text"].tolist()).values.astype(np.float32)
    
    print("Training LightGBM on stylometric features...")
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
        n_jobs=-1,
        verbose=-1,
    )
    clf_lgb.fit(X_sty_train, y_int)
    p_sty = clf_lgb.predict_proba(X_sty_test)
    
    # 2. 4-View Structural N-Grams
    print("Configuring and extracting 4-View Structural N-Grams...")
    views = [
        ("char", 3, 5, 200_000, 1.0, lambda s: s),
        ("word", 1, 2, 120_000, 0.8, lambda s: s),
        ("shape", 2, 5, 40_000, 0.5, shape_transform),
        ("function", 1, 2, 30_000, 0.5, function_transform),
    ]
    matrices_trn, matrices_test = [], []
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
        txt_trn = [prep(t) for t in train_df["text"]]
        txt_test = [prep(t) for t in test_df["text"]]
        matrices_trn.append(vec.fit_transform(txt_trn) * w)
        matrices_test.append(vec.transform(txt_test) * w)
        
    X_ng_trn = hstack(matrices_trn, format="csr")
    X_ng_test = hstack(matrices_test, format="csr")
    
    print("Fitting Logistic Regression on 4-View N-Grams...")
    clf_ng = LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_ng.fit(X_ng_trn, y_int)
    p_ng = clf_ng.predict_proba(X_ng_test)
    
    # 3. Late Logit Fusion & Sinkhorn Balancing
    print("Blending views in logit space...")
    w_ng = 0.70
    w_sty = 0.30
    log_p = w_ng * np.log(np.clip(p_ng, 1e-9, 1.0)) + w_sty * np.log(np.clip(p_sty, 1e-9, 1.0))
    p_blend = np.exp(log_p - log_p.max(axis=1, keepdims=True))
    p_blend /= p_blend.sum(axis=1, keepdims=True)
    
    p_final = sinkhorn_knopp_balance(p_blend, iters=50) if N_test > 10 else p_blend
    
    pred_indices = np.argmax(p_final, axis=1)
    predictions = [VALID_18_MODELS[i] for i in pred_indices]
    
    # 4. Submission Parquet & Assertions
    sub = pd.DataFrame({
        "id": test_df["id"],
        "model": predictions
    })
    
    assert list(sub.columns) == ["id", "model"]
    assert len(sub) == N_test
    assert not sub.isnull().any().any()
    assert sub["model"].isin(VALID_18_MODELS).all()
    
    output_path = "submission.parquet"
    sub.to_parquet(output_path, index=False)
    print(f"\nSUCCESS: Created {output_path} ({len(sub)} rows) in {time.time() - t_start:.1f}s")
    print(sub.head())


if __name__ == "__main__":
    main()
