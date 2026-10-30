"""
Name That LLM - Baseline Stylometric & TF-IDF Blend Submission
Self-contained Kaggle submission script.
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

# Valid 18 competition target classes in strict alphabetical order
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

DISCOURSE_MARKERS = [
    "furthermore", "moreover", "however", "in contrast", "specifically",
    "ultimately", "notably", "conversely", "indeed", "consequently",
    "in summary", "essentially", "crucially", "paradoxically", "fundamentally"
]


def extract_stylometric_features(texts: List[str]) -> pd.DataFrame:
    """Extract dense DataFrame of stylometric, typographic, syntactic, and rhythm features."""
    records = []
    
    for text in tqdm(texts, desc="Extracting stylometric features"):
        feat = {}
        t_len = len(text)
        words = text.split()
        n_words = max(len(words), 1)
        
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        n_sent = max(len(sentences), 1)
        
        # 1. Punctuation & Typography Counts & Rates
        feat["char_count"] = t_len
        feat["word_count"] = n_words
        feat["sentence_count"] = n_sent
        feat["mean_word_length"] = np.mean([len(w) for w in words]) if words else 0.0
        feat["std_word_length"] = np.std([len(w) for w in words]) if words else 0.0
        
        # Dashes & Em-Dash Spacing Geometry (GLM vs Claude)
        c_em = text.count(EM_DASH)
        feat["count_em_dash"] = c_em
        feat["has_em_dash"] = float(c_em > 0)
        feat["rate_em_dash"] = c_em / n_words
        feat["count_en_dash"] = text.count(EN_DASH)
        feat["count_hyphen"] = text.count("-")
        feat["dash_ratio"] = (c_em + 1) / (feat["count_hyphen"] + 1)
        
        c_spaced = len(EM_DASH_SPACED.findall(text))
        c_unspaced = len(EM_DASH_UNSPACED.findall(text))
        feat["count_em_dash_spaced"] = c_spaced
        feat["count_em_dash_unspaced"] = c_unspaced
        feat["em_dash_spacing_ratio"] = c_spaced / max(c_spaced + c_unspaced, 1)
        
        # Contractions & Typographic Quote Selection (Gemini Flash & Claude vs GPT)
        c_straight = len(CONTRACTION_STRAIGHT.findall(text))
        c_curly = len(CONTRACTION_CURLY.findall(text))
        feat["count_straight_contractions"] = c_straight
        feat["count_curly_contractions"] = c_curly
        feat["total_contractions"] = c_straight + c_curly
        feat["contraction_rate"] = feat["total_contractions"] / n_words
        feat["curly_contraction_ratio"] = c_curly / max(feat["total_contractions"], 1)
        
        # Quotes: Straight vs Curly
        straight_s = text.count("'")
        straight_d = text.count('"')
        feat["count_straight_quotes"] = straight_s + straight_d
        feat["has_straight_quotes"] = float(feat["count_straight_quotes"] > 0)
        
        curly_s = sum(text.count(c) for c in CURLY_SINGLE)
        curly_d = sum(text.count(c) for c in CURLY_DOUBLE)
        feat["count_curly_quotes"] = curly_s + curly_d
        feat["count_curly_apostrophe"] = text.count("’")
        feat["has_curly_quotes"] = float(feat["count_curly_quotes"] > 0)
        
        total_quotes = feat["count_straight_quotes"] + feat["count_curly_quotes"]
        feat["curly_quote_ratio"] = feat["count_curly_quotes"] / max(total_quotes, 1)
        
        # Semicolons, Colons, Commas & Clausal Balancing
        feat["count_semicolon"] = text.count(";")
        feat["rate_semicolon"] = feat["count_semicolon"] / n_words
        feat["count_colon"] = text.count(":")
        feat["rate_colon"] = feat["count_colon"] / n_words
        feat["count_comma"] = text.count(",")
        feat["rate_comma"] = feat["count_comma"] / n_words
        feat["semicolon_to_comma_ratio"] = (feat["count_semicolon"] + 1.0) / (feat["count_comma"] + 1.0)
        feat["comma_per_sentence"] = feat["count_comma"] / n_sent
        feat["colon_per_sentence"] = feat["count_colon"] / n_sent
        feat["count_exclamation"] = text.count("!")
        feat["count_question"] = text.count("?")
        feat["count_ellipsis"] = text.count("…") + text.count("...")
        feat["count_parentheses"] = text.count("(") + text.count(")")
        feat["count_brackets"] = text.count("[") + text.count("]")
        
        # 2. Capitalization & Acronyms
        upper_chars = sum(1 for c in text if c.isupper())
        feat["rate_uppercase_chars"] = upper_chars / max(t_len, 1)
        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        feat["rate_acronyms"] = upper_words / n_words
        title_words = sum(1 for w in words[1:] if w.istitle())
        feat["rate_mid_sentence_title"] = title_words / max(n_words - 1, 1)
        digits = sum(1 for c in text if c.isdigit())
        feat["rate_digits"] = digits / max(t_len, 1)
        long_words = sum(1 for w in words if len(w) > 6)
        feat["rate_long_words"] = long_words / n_words
        
        # 3. Sentence Length Rhythm & Dynamics
        sent_lens = [len(s.split()) for s in sentences] if sentences else [n_words]
        feat["words_per_sentence_mean"] = float(np.mean(sent_lens))
        feat["words_per_sentence_std"] = float(np.std(sent_lens))
        feat["words_per_sentence_max"] = float(max(sent_lens))
        feat["words_per_sentence_min"] = float(min(sent_lens))
        feat["sentence_len_cv"] = feat["words_per_sentence_std"] / max(feat["words_per_sentence_mean"], 1e-4)
        
        # Sentence-initial coordination
        init_conj_count = sum(1 for s in sentences if s.split() and s.split()[0].lower() in INITIAL_CONJUNCTIONS)
        feat["rate_initial_conjunctions"] = init_conj_count / n_sent
        
        # 4. Readability: Automated Readability Index (ARI)
        feat["ari_score"] = 4.71 * (t_len / n_words) + 0.5 * (n_words / n_sent) - 21.43
        
        # 5. Lexical Diversity & Richness
        lower_words = [w.lower().strip(".,!?;:\"'()[]") for w in words]
        unique_words = set(lower_words)
        v = len(unique_words)
        feat["ttr"] = v / n_words
        feat["root_ttr"] = v / math.sqrt(n_words)
        
        freqs = pd.Series(lower_words).value_counts().values
        feat["hapax_legomena_ratio"] = sum(1 for f in freqs if f == 1) / max(v, 1)
        feat["dis_legomena_ratio"] = sum(1 for f in freqs if f == 2) / max(v, 1)
        s1 = n_words
        s2 = sum(f * f for f in freqs)
        feat["yules_k"] = (10000 * (s2 - s1) / (s1 * s1)) if s1 > 1 else 0.0
        
        # 6. Lexical & Epistemic Markers
        feat["claude_marker_density"] = sum(1 for w in lower_words if w in CLAUDE_MARKERS) / n_words
        feat["reasoning_marker_density"] = sum(1 for w in lower_words if w in REASONING_MARKERS) / n_words
        feat["subordinator_density"] = sum(1 for w in lower_words if w in SUBORDINATORS) / n_words
        
        lower_text = text.lower()
        for marker in DISCOURSE_MARKERS:
            feat[f"has_{marker.replace(' ', '_')}"] = float(marker in lower_text)
            
        records.append(feat)
        
    return pd.DataFrame(records)


def main():
    print("[1/6] Locating dataset paths...")
    # Support both /kaggle/input and /kaggle/input/competitions/ paths
    train_candidates = [
        "/kaggle/input/competitions/name-that-llm/train.csv",
        "/kaggle/input/name-that-llm/train.csv",
    ]
    test_candidates = [
        "/kaggle/input/competitions/name-that-llm/test.csv",
        "/kaggle/input/name-that-llm/test.csv",
    ]
    
    train_path = next((p for p in train_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)
    
    if not train_path or not test_path:
        raise FileNotFoundError(f"Cannot find dataset! Checked: {train_candidates}")
        
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train["text"] = df_train["text"].fillna("")
    df_test["text"] = df_test["text"].fillna("")
    
    y_raw = df_train["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    print(f"Train samples: {len(df_train)} | Test samples: {len(df_test)}")
    
    print("\n[2/6] Extracting stylometric features...")
    X_sty_train = extract_stylometric_features(df_train["text"].tolist()).values.astype(np.float32)
    X_sty_test = extract_stylometric_features(df_test["text"].tolist()).values.astype(np.float32)
    
    print("\n[3/6] Fitting TF-IDF Vectorizers...")
    cfgs = [
        dict(analyzer="char", ngram_range=(1, 5), max_features=250_000),
        dict(analyzer="word", ngram_range=(1, 2), max_features=150_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
    ]
    
    vs = []
    for c in tqdm(cfgs, desc="Fitting TF-IDF Vectorizers"):
        v = TfidfVectorizer(sublinear_tf=True, lowercase=False, min_df=2, **c)
        vs.append(v)
        
    X_tf_train = hstack([v.fit_transform(df_train["text"]) for v in vs]).tocsr()
    X_tf_test = hstack([v.transform(df_test["text"]) for v in vs]).tocsr()
    print(f"TF-IDF sparse matrix shape: {X_tf_train.shape}")
    
    print("\n[4/6] Training Logistic Regression on TF-IDF...")
    clf_tf = LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced", random_state=42)
    clf_tf.fit(X_tf_train, y_raw)
    p_tf_test = clf_tf.predict_proba(X_tf_test)
    
    print("\n[5/6] Training LightGBM on Stylometric features...")
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
    clf_lgb.fit(X_sty_train, y_int)
    p_lgb_test = clf_lgb.predict_proba(X_sty_test)
    
    print("\n[6/6] Blending predictions and formatting submission...")
    # Blend with optimal CV weight alpha = 0.90
    alpha = 0.90
    p_blend_test = alpha * p_tf_test + (1.0 - alpha) * p_lgb_test
    pred_indices = np.argmax(p_blend_test, axis=1)
    test_preds = [VALID_18_MODELS[i] for i in pred_indices]
    
    sub = pd.DataFrame({"id": df_test["id"], "model": test_preds})
    
    # Validation checks
    assert sub.shape == (len(df_test), 2), f"Shape mismatch: {sub.shape} vs ({len(df_test)}, 2)"
    assert list(sub.columns) == ["id", "model"], f"Columns mismatch: {list(sub.columns)}"
    assert sub["model"].isin(VALID_18_MODELS).all(), "Invalid model class predicted!"
    assert not sub.isnull().any().any(), "Submission contains null/NaN values!"
    
    sub_path = "submission.parquet"
    sub.to_parquet(sub_path, index=False)
    print(f"\n[SUCCESS] Successfully written {len(sub)} rows to {sub_path}!")
    print("Class prediction distribution in test sample:")
    print(sub["model"].value_counts())


if __name__ == "__main__":
    main()
