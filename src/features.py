"""
Stylometric and Typographic Feature Extraction Engine for 'Name That LLM'.
Extracts fine-grained punctuation, typography, lexical richness, and structural rhythm features.
"""

import math
import re
from typing import Dict, List
import numpy as np
import pandas as pd


# Punctuation constants
CURLY_SINGLE = set(["‘", "’"])
CURLY_DOUBLE = set(["“", "”"])
STRAIGHT_QUOTES = set(["'", '"'])
EM_DASH = "—"
EN_DASH = "–"
ELLIPSIS = set(["…", "..."])

# Contraction patterns
CONTRACTION_STRAIGHT = re.compile(r"\b[a-zA-Z]+'[a-zA-Z]+\b")
CONTRACTION_CURLY = re.compile(r"\b[a-zA-Z]+’[a-zA-Z]+\b")

# Em-dash spacing geometry (distinguishes Claude unspaced from GLM spaced)
EM_DASH_SPACED = re.compile(r"\s—\s|\s—\w|\w—\s")
EM_DASH_UNSPACED = re.compile(r"\w—\w")

# Sentence-initial conjunctions (Anthropic cadence)
INITIAL_CONJUNCTIONS = set(["and", "but", "yet", "so", "or"])

# Model-specific lexical marker sets
CLAUDE_MARKERS = set(["perhaps", "nuance", "underlying", "tapestry", "interplay", "woven", "profound", "notion"])
REASONING_MARKERS = set(["therefore", "thus", "hence", "step", "implies", "consequently", "assume", "furthermore"])
SUBORDINATORS = set(["because", "although", "whereas", "while", "since", "unless", "provided", "despite"])

DISCOURSE_MARKERS = [
    "furthermore", "moreover", "however", "in contrast", "specifically",
    "ultimately", "notably", "conversely", "indeed", "consequently",
    "in summary", "essentially", "crucially", "paradoxically", "fundamentally"
]


def extract_stylometric_features(texts: List[str]) -> pd.DataFrame:
    """
    Extract a comprehensive dense DataFrame of stylometric, typographic,
    syntactic, and rhythm features from a list of strings.
    """
    records = []
    
    for text in texts:
        feat = {}
        t_len = len(text)
        words = text.split()
        n_words = max(len(words), 1)
        
        # Sentences
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
