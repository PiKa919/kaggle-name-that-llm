"""
Watermark & Hidden Tokenization Artifact Probe
Empirically tests all 18 candidate models in train.csv for:
1. Hidden Unicode characters (zero-width spaces, joiners, non-breaking spaces, variation selectors).
2. Typographic watermarks (straight vs curly quotes, apostrophes, dashes, ellipses).
3. Bigram transition predictability & entropy (Kirchenbauer watermark signal).
4. Short-passage length distributions.
"""

import math
import os
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

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


def analyze_watermarks_and_signatures(df: pd.DataFrame):
    print("=" * 80)
    print("=== WATERMARK & STATISTICAL SIGNATURE PROBE ACROSS 18 MODELS ===")
    print("=" * 80)
    
    # 1. Hidden Unicode Codepoints Probe
    hidden_codepoints = {
        "zero_width_space (\\u200b)": "\u200b",
        "zero_width_joiner (\\u200d)": "\u200d",
        "zero_width_non_joiner (\\u200c)": "\u200c",
        "byte_order_mark (\\ufeff)": "\ufeff",
        "non_breaking_space (\\u00a0)": "\u00a0",
        "thin_space (\\u2009)": "\u2009",
        "hair_space (\\u200a)": "\u200a",
        "soft_hyphen (\\u00ad)": "\u00ad",
    }
    
    print("\n--- 1. HIDDEN UNICODE CODEPOINT ANALYSIS ---")
    for name, char in hidden_codepoints.items():
        counts_per_model = defaultdict(int)
        total_occurrences = 0
        for _, row in df.iterrows():
            c = row["text"].count(char)
            if c > 0:
                counts_per_model[row["model"]] += c
                total_occurrences += c
        if total_occurrences > 0:
            print(f"  [FOUND] {name}: {total_occurrences} total occurrences!")
            for m, count in sorted(counts_per_model.items(), key=lambda x: -x[1]):
                print(f"    - {m}: {count}")
        else:
            print(f"  [CLEAR] {name}: 0 occurrences across all 2,699 passages.")
            
    # 2. Typographic Deterministic Signatures
    print("\n--- 2. TYPOGRAPHIC WATERMARK / CONVENTIONAL SIGNATURES ---")
    metrics = []
    for model_name in VALID_18_MODELS:
        m_df = df[df["model"] == model_name]
        n_passages = len(m_df)
        
        # Quotes and apostrophes
        straight_single = sum(t.count("'") for t in m_df["text"]) / n_passages
        curly_single = sum(t.count("‘") + t.count("’") for t in m_df["text"]) / n_passages
        straight_double = sum(t.count('"') for t in m_df["text"]) / n_passages
        curly_double = sum(t.count("“") + t.count("”") for t in m_df["text"]) / n_passages
        
        # Dashes
        em_dash_passages = sum(1 for t in m_df["text"] if "—" in t) / n_passages * 100
        en_dash_passages = sum(1 for t in m_df["text"] if "–" in t) / n_passages * 100
        hyphen_per_passage = sum(t.count("-") for t in m_df["text"]) / n_passages
        
        # Spaced vs unspaced em-dash
        spaced_em = sum(t.count(" — ") for t in m_df["text"]) / n_passages
        unspaced_em = sum(len(re.findall(r"\S—\S", t)) for t in m_df["text"]) / n_passages
        
        # Word counts
        word_lens = [len(t.split()) for t in m_df["text"]]
        avg_words = np.mean(word_lens)
        std_words = np.std(word_lens)
        
        metrics.append({
            "model": model_name,
            "curly_sq_rate": f"{curly_single / (straight_single + curly_single + 1e-6):.2%}",
            "curly_dq_rate": f"{curly_double / (straight_double + curly_double + 1e-6):.2%}",
            "em_dash_%": f"{em_dash_passages:.1f}%",
            "spaced_em": f"{spaced_em:.2f}",
            "unspaced_em": f"{unspaced_em:.2f}",
            "avg_words": f"{avg_words:.1f}±{std_words:.1f}",
        })
        
    met_df = pd.DataFrame(metrics)
    print(met_df.to_string(index=False))
    
    # 3. Bigram Transition Entropy & Repetition Bias (Watermarking Signal)
    print("\n--- 3. CHARACTER BIGRAM CONDITIONAL ENTROPY ---")
    entropies = []
    for model_name in VALID_18_MODELS:
        m_df = df[df["model"] == model_name]
        all_text = " ".join(m_df["text"])
        
        bigram_counts = defaultdict(Counter)
        for i in range(len(all_text) - 1):
            c1, c2 = all_text[i], all_text[i+1]
            bigram_counts[c1][c2] += 1
            
        # Compute conditional entropy H(C2 | C1)
        total_transitions = sum(sum(ctr.values()) for ctr in bigram_counts.values())
        h_cond = 0.0
        for c1, ctr in bigram_counts.items():
            c1_total = sum(ctr.values())
            p_c1 = c1_total / total_transitions
            h_c1 = -sum((cnt / c1_total) * math.log2(cnt / c1_total) for cnt in ctr.values())
            h_cond += p_c1 * h_c1
            
        entropies.append({"model": model_name, "H_cond_bigram": f"{h_cond:.4f} bits"})
        
    print(pd.DataFrame(entropies).to_string(index=False))


if __name__ == "__main__":
    train_path = "data/train.csv"
    if not os.path.exists(train_path):
        train_path = "/kaggle/input/competitions/name-that-llm/train.csv"
    df = pd.read_csv(train_path)
    analyze_watermarks_and_signatures(df)
