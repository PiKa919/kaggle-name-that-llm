"""
Cross-Tokenizer Forensics, Syntactic Complexity & Watermark Residuals.
======================================================================

Mathematical Foundations & Architecture
----------------------------------------

1. Cross-Tokenizer Pre-Tokenization Mechanics & Compression Ratios:
   Different LLM architectures deploy radically distinct subword tokenization
   pipelines:
     - OpenAI Tiktoken (cl100k_base vs o200k_base):
       cl100k_base (GPT-4 / GPT-5.5) isolates clitics via pre-tokenization regex:
       contractions ("don't", "it's") are fragmented into root + clitic (" don", "'t").
       In contrast, o200k_base (GPT-5.6 family: Luna, Sol, Terra; GPT-6-Astra) merges
       frequent English contractions with straight apostrophes into unitary tokens
       (" don't", " it's"), drastically shifting token boundaries.
       Furthermore, o200k (~200k vocab) achieves significantly higher compression
       (chars/token) than cl100k (~100k vocab).
     - Google Gemini SentencePiece:
       SentencePiece unigram models operate over raw character streams without
       forcing Python regex boundary splits, using the meta-symbol U+2581 (' ')
       for whitespace prefixing and byte fallbacks for unseen Unicode glyphs.
     - Byte-Pair Encoding (Qwen 3.x / Claude BPE):
       Custom BPE tokenizers produce distinct byte-level splits on punctuation,
       em-dashes, and numbers.
   We capture:
     * Differential compression ratios: Delta_comp = Chars/Token(o200k) - Chars/Token(cl100k)
     * Contraction merge vs split frequencies
     * Subword phonotactic transitions: consonant-consonant (CC), consonant-vowel (CV),
       vowel-consonant (VC), and vowel-vowel (VV) boundary transition shares and entropy.
     * Morphological prefix and suffix split density.

2. Heuristic Syntactic Depth & Clausal Subordination (No Heavy Neural Parsers):
   Cognitive psycholinguistics (Yngve, 1960) demonstrates that human working memory
   is loaded by left-branching syntactic structures where constituents precede the
   finite matrix verb.
   We construct a fast, robust rule-based syntactic engine:
     * Matrix verb locator: isolates the first finite auxiliary or main verb in
       the matrix clause, accounting for preposed adverbial clauses ("Although X, Y").
     * Left-branching depth (Yngve memory load proxy): pre-verbal word count L_left.
     * Right-branching depth: post-verbal word count L_right.
     * Left-to-right branching ratio: (L_left + 1) / (L_right + 1).
     * Clausal subordination index: subordinating conjunctions ("because", "although",
       "since", "whereas") and relative pronouns ("which", "who", "whom", "whose").
     * Complex & compound-complex clausal embedding shares via punctuation segmentation.
     * Coordinate conjunction cadence: sentence-initial coordinates ("And", "But", "Yet")
       and medial comma-conjunction rhythms.

3. Watermark Residual Signatures & Green-Red List Bias:
   Theoretical formulation following Kirchenbauer et al. (ICML 2023) and Christ et al. (2023):
     * Given context window x_{t-h:t-1}, a deterministic hash h_s(context) partitions
       the vocabulary into green list G_t (size gamma * |V|) and red list R_t.
     * The observed green count |s|_G under null hypothesis H_0 has mean gamma * T
       and variance T * gamma * (1 - gamma).
     * The Kirchenbauer z-statistic is:
           z = (|s|_G - gamma * T) / sqrt(T * gamma * (1 - gamma))
     * Multi-seed variance: Even when the secret provider seed is unrevealed,
       measuring the dispersion (std and max) of z across standard salt seeds
       identifies artificial watermark residual alignment.
     * Local entropy suppression: Ratio of bigram conditional entropy H(W_t | W_{t-1})
       to unigram entropy H(W_t).
     * Zipfian power-law regression: Beta slope, residual MSE, and R^2.

4. Fine-Grained Dedicated Sibling Discriminators:
   Mathematical pairwise discriminators tailored to the most confusable sibling pairs:
     * gemini-3.7-flash vs gemini-3.8-flash:
       Curly double quotes (3.7 high, 3.8 standardized straight), question-to-explanation
       pacing (3.7 hybrid reasoning rhetoric), and sentence length CV.
     * claude-opus-4-6 vs claude-opus-5:
       Spaced em-dashes (' — '), deep clausal subordination, left-branching depth,
       and reflective epistemic stance.
     * gpt-5.6-luna vs gpt-5.6-sol vs gpt-5.6-terra:
       3-way decomposition: Luna (deductive / formal / high left-branching),
       Terra (empirical / grounded / formatted numbers), Sol (balanced interior).
     * glm-5.3 vs glm-5.3-flash:
       Contraction suppression, digit chunking, and sentence compression.
     * glm-5.3 vs claude-opus-4-6:
       Unspaced em-dashes vs spaced em-dashes and contraction suppression index.

5. HodgeRank Least-Squares Potential Projection:
   Independent pairwise log-odds differences are projected onto a cycle-free, globally
   transitive potential vector v in R^K via graph Laplacian pseudo-inverse:
       v = L^+ b
   guaranteeing zero Condorcet cycles and strict conservation across disconnected families.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

import tiktoken

from src.config import MODEL_TO_IDX, VALID_18_MODELS

__all__ = [
    "CROSS_TOKENIZER_FEATURE_NAMES",
    "CROSS_TOKENIZER_FEATURE_DOCS",
    "N_CROSS_TOKENIZER_FEATURES",
    "TARGET_SIBLING_PAIRS",
    "extract_raw_cross_tokenizer_features",
    "extract_cross_tokenizer_features",
    "CrossTokenizerSiblingDiscriminator",
    "CrossTokenizerPairwiseInjector",
    "fuse_cross_tokenizer_forensics",
    "cross_tokenizer_feature_names",
    "cross_tokenizer_feature_doc",
]

# --------------------------------------------------------------------------- #
# Tiktoken Encodings & Initializations
# --------------------------------------------------------------------------- #

_ENC_CL100K = tiktoken.get_encoding("cl100k_base")
_ENC_O200K = tiktoken.get_encoding("o200k_base")

# --------------------------------------------------------------------------- #
# Regex Patterns for Segmentation, Morphology, and Syntax
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_ABBREV_DOT_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Fig|No|cf|al|"
    r"Inc|Ltd|approx|Eq|Ref)\. ?",
    re.IGNORECASE,
)
_SENT_RE = re.compile(r"[^.!?\u2026]+[.!?\u2026]*")
_SENT_MASK = "\u0001"

# Syntactic subordination and clause markers
_SUBORD_CONJ_RE = re.compile(
    r"\b(?:although|though|even though|because|since|whereas|while|unless|if|"
    r"insofar as|provided that|assuming that|given that|as long as|"
    r"after|before|until|once|whenever|wherever|whether)\b",
    re.IGNORECASE,
)
_RELATIVE_PRONOUN_RE = re.compile(
    r"\b(?:which|who|whom|whose|whereby|wherein|whereupon)\b",
    re.IGNORECASE,
)
_SUBORD_LEAD_RE = re.compile(
    r"^(?:although|though|even though|because|since|while|whereas|if|unless|"
    r"insofar as|provided that|assuming that|after|before|when|whenever)\b",
    re.IGNORECASE,
)

# Coordinate conjunctions
_COORD_INITIAL_RE = re.compile(r"^(?:And|But|Yet|So|Or|Nor)\b")
_COORD_MEDIAL_RE = re.compile(r",\s*(?:and|but|yet|so|or|nor)\b", re.IGNORECASE)

# Finite verb auxiliaries and inflections
_AUX_MODALS = frozenset({
    "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
})
_IRREGULAR_VERBS = frozenset({
    "found", "made", "took", "gave", "became", "began", "knew", "saw",
    "wrote", "built", "held", "stood", "went", "came", "said", "thought",
    "felt", "showed", "meant", "kept", "told", "led", "left", "set",
    "brought", "put", "read", "ran", "heard", "understood", "converged",
})
_VERB_SUFFIX_RE = re.compile(r"^[a-z]{3,}(?:ed|ize|ise|ify|ate)$")
_VERB_3S_RE = re.compile(r"^[a-z]{3,}(?:es|s)$")
_NOUN_EXCLUSIONS = frozenset({
    "this", "is", "as", "us", "its", "thus", "yes", "always", "less", "basis",
    "analysis", "hypothesis", "thesis", "results", "data", "steps", "tokens",
    "words", "models", "parameters", "features", "labels", "values", "experiments",
    "systems", "methods", "approaches", "cases", "types", "times", "ways",
})

# Clitics and Contractions
_STRAIGHT_CLITIC_RE = re.compile(r"\b[a-zA-Z]+'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_CURLY_CLITIC_RE = re.compile(r"\b[a-zA-Z]+\u2019(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_FORMAL_EXPANSION_RE = re.compile(
    r"\b(?:do not|does not|did not|cannot|can not|will not|would not|could not|"
    r"should not|is not|are not|was not|were not|have not|has not|had not|"
    r"it is|they are|we are|you are|i am|i have|we have|they have)\b",
    re.IGNORECASE,
)

# Morphological affixes
_PREFIX_RE = re.compile(
    r"\b(?:un|re|dis|pre|post|anti|sub|inter|intra|non|in|im|mis|over|under|"
    r"semi|trans|counter|pseudo)[a-zA-Z]{3,}",
    re.IGNORECASE,
)
_SUFFIX_RE = re.compile(
    r"[a-zA-Z]{3,}(?:tion|sion|ment|able|ible|ity|ities|ous|ious|ance|ence|"
    r"ful|less|ness|ism|ist|ization|isation|ize|ise|ate|ive)\b",
    re.IGNORECASE,
)

# Typography & Punctuation
_EM_SPACED_RE = re.compile(r"(?<=\s)\u2014(?=\s)")
_EM_UNSPACED_RE = re.compile(r"(?<=\S)\u2014(?=\S)")
_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")
_FORMATTED_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b")
_DIGIT_CHUNK_RE = re.compile(r"\d+")

_VOWELS = frozenset("aeiouAEIOU")

# Target sibling pairs
TARGET_SIBLING_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("claude-opus-4-6", "claude-opus-5"),
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("gpt-5.6-luna", "gpt-5.6-terra"),
    ("glm-5.3", "glm-5.3-flash"),
    ("glm-5.3", "claude-opus-4-6"),
)


# --------------------------------------------------------------------------- #
# Fast Deterministic Hash Helpers (Watermark Detection)
# --------------------------------------------------------------------------- #

def _fnv1a_32(token: str, seed: int = 2166136261) -> int:
    """Deterministic 32-bit FNV-1a hash parameterized with custom seed."""
    h = seed & 0xFFFFFFFF
    for b in token.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# Sentence Splitting & Syntactic Helpers
# --------------------------------------------------------------------------- #

def _split_sentences(text: str) -> List[str]:
    """Segment sentences safely while masking decimal numbers and abbreviations."""
    if not text:
        return []
    masked = _DECIMAL_DOT_RE.sub(_SENT_MASK, text)
    masked = _ABBREV_DOT_RE.sub(
        lambda m: m.group(0).replace(".", _SENT_MASK), masked
    )
    raw_sents = [s.strip() for s in _SENT_RE.findall(masked) if s.strip()]
    return [s.replace(_SENT_MASK, ".") for s in raw_sents]


def _is_finite_verb(w: str) -> bool:
    """Identify if a word is likely a finite auxiliary or main verb."""
    wl = w.lower().strip(",.;:!?()[]\"'")
    if wl in _AUX_MODALS or wl in _IRREGULAR_VERBS:
        return True
    if wl in _NOUN_EXCLUSIONS or len(wl) < 3:
        return False
    if _VERB_SUFFIX_RE.match(wl):
        return True
    if _VERB_3S_RE.match(wl) and len(wl) >= 5:
        return True
    return False


def _analyze_sentence_branching(sent: str) -> Tuple[int, int]:
    """
    Compute pre-verbal word distance (left-branching depth) and
    post-verbal word distance (right-branching depth).
    """
    words = sent.split()
    if not words:
        return 0, 0
    if len(words) == 1:
        return 0, 0

    has_preposed = bool(_SUBORD_LEAD_RE.match(sent.strip()))
    start_search_idx = 0
    if has_preposed and "," in sent:
        comma_part = sent.split(",", 1)[0]
        start_search_idx = len(comma_part.split())

    for idx in range(start_search_idx, len(words)):
        if idx == 0:
            if words[idx].lower() in {"consider", "note", "suppose", "assume", "observe"}:
                return 0, len(words) - 1
            continue
        if _is_finite_verb(words[idx]):
            left = idx
            right = len(words) - 1 - idx
            return left, right

    mid = len(words) // 2
    return mid, len(words) - 1 - mid


# --------------------------------------------------------------------------- #
# Document Statistics Object
# --------------------------------------------------------------------------- #

class _CrossTokenizerStats:
    """Cached stylometric, cross-tokenizer, and syntactic features for one passage."""

    __slots__ = (
        "text", "tokens", "lower", "words", "n_words", "n_chars",
        "sentences", "n_sent", "sent_lens", "sent_cv",
        # Tiktoken cl100k vs o200k metrics
        "cl100k_count", "o200k_count",
        "cl100k_chars_per_tok", "o200k_chars_per_tok",
        "comp_diff", "tok_ratio_o200k_cl100k", "diff_tok_per_word",
        "o200k_merged_contractions", "cl100k_tok_len_mean", "cl100k_tok_len_std",
        "o200k_tok_len_mean", "o200k_tok_len_std",
        "cl100k_entropy", "o200k_entropy", "tok_entropy_diff",
        # SentencePiece & Unicode simulation
        "sp_leading_space_share", "sp_digit_chunks", "whitespace_char_ratio",
        "multibyte_utf8_share",
        # Clitics & Morphology
        "straight_clitics", "curly_clitics", "curly_clitic_share",
        "contraction_suppression", "prefix_count", "suffix_count", "affix_density",
        "cc_share", "vc_share", "cv_share", "vv_share", "phonotactic_entropy",
        # Syntactic depth & subordination
        "subord_conj_count", "rel_pronoun_count", "subord_index_per_sent",
        "left_branching_mean", "left_branching_max", "left_to_right_ratio",
        "left_heavy_share", "preposed_clause_share", "clauses_per_sent_mean",
        "complex_sent_share", "compound_complex_share", "clause_len_cv",
        "coord_initial_share", "coord_medial_count", "coord_subord_balance",
        # Watermark residuals
        "kirchen_z_h1_g50", "kirchen_z_h1_g25", "kirchen_z_h2_g50", "kirchen_z_h2_g25",
        "wm_multiseed_std", "wm_multiseed_max", "cond_to_unigram_entropy_ratio",
        "zipf_beta", "zipf_mse", "zipf_r2", "simpson", "hapax_ratio", "dis_ratio",
        # Typography & Sibling discriminators
        "curly_dq_ratio", "curly_sq_ratio", "spaced_em_ratio", "unspaced_em_rate",
        "formatted_num_ratio", "q_to_exp_ratio"
    )

    def __init__(self, text: str) -> None:
        self.text = text
        self.n_chars = len(text)
        raw_words = _WORD_RE.findall(text)
        self.words = raw_words
        self.lower = [w.lower() for w in raw_words]
        self.tokens = len(self.lower)
        self.n_words = self.tokens

        # Sentences & Sentence rhythm
        sents = _split_sentences(text)
        self.sentences = sents
        self.n_sent = len(sents)
        self.sent_lens = [len(_WORD_RE.findall(s)) for s in sents]

        if self.sent_lens:
            mu_s = float(np.mean(self.sent_lens))
            std_s = float(np.std(self.sent_lens))
            self.sent_cv = std_s / (mu_s + 1e-6)
        else:
            self.sent_cv = 0.0

        # ------------------------------------------------------------------- #
        # 1. Tiktoken cl100k vs o200k Forensic Encoding
        # ------------------------------------------------------------------- #
        if self.n_chars > 0:
            tok_cl = _ENC_CL100K.encode(text)
            tok_o2 = _ENC_O200K.encode(text)
            n_cl = len(tok_cl)
            n_o2 = len(tok_o2)

            self.cl100k_count = n_cl
            self.o200k_count = n_o2
            self.cl100k_chars_per_tok = (self.n_chars / n_cl) if n_cl > 0 else 0.0
            self.o200k_chars_per_tok = (self.n_chars / n_o2) if n_o2 > 0 else 0.0
            self.comp_diff = self.o200k_chars_per_tok - self.cl100k_chars_per_tok
            self.tok_ratio_o200k_cl100k = (n_o2 / (n_cl + 1e-9)) if n_cl > 0 else 1.0
            self.diff_tok_per_word = (
                (n_cl - n_o2) / (self.n_words + 1e-9) if self.n_words > 0 else 0.0
            )

            # Token length distributions
            cl_decoded = [_ENC_CL100K.decode([t]) for t in tok_cl]
            o2_decoded = [_ENC_O200K.decode([t]) for t in tok_o2]

            cl_lens = [len(d) for d in cl_decoded]
            o2_lens = [len(d) for d in o2_decoded]
            self.cl100k_tok_len_mean = float(np.mean(cl_lens)) if cl_lens else 0.0
            self.cl100k_tok_len_std = float(np.std(cl_lens)) if cl_lens else 0.0
            self.o200k_tok_len_mean = float(np.mean(o2_lens)) if o2_lens else 0.0
            self.o200k_tok_len_std = float(np.std(o2_lens)) if o2_lens else 0.0

            # o200k merged contraction detection
            # in o200k, contractions like " don't", " it's" are unitary tokens
            merged_contractions = 0
            for d in o2_decoded:
                dl = d.lower().strip()
                if dl in {"don't", "it's", "can't", "won't", "didn't", "doesn't",
                          "i'm", "you're", "they're", "we're", "isn't", "aren't",
                          "wasn't", "weren't", "haven't", "hasn't", "hadn't",
                          "wouldn't", "couldn't", "shouldn't", "let's", "that's"}:
                    merged_contractions += 1
            self.o200k_merged_contractions = merged_contractions

            # Subword Token Shannon Entropy
            cl_cnt = Counter(tok_cl)
            cl_p = np.array(list(cl_cnt.values()), dtype=np.float64) / n_cl
            self.cl100k_entropy = float(-np.sum(cl_p * np.log2(cl_p))) if n_cl > 0 else 0.0

            o2_cnt = Counter(tok_o2)
            o2_p = np.array(list(o2_cnt.values()), dtype=np.float64) / n_o2
            self.o200k_entropy = float(-np.sum(o2_p * np.log2(o2_p))) if n_o2 > 0 else 0.0
            self.tok_entropy_diff = self.cl100k_entropy - self.o200k_entropy

            # Subword phonotactic transitions (CC, CV, VC, VV) at token cuts
            cc, cv, vc, vv = 0, 0, 0, 0
            for i in range(1, len(cl_decoded)):
                prev_d = cl_decoded[i - 1]
                curr_d = cl_decoded[i]
                if curr_d and curr_d[0].isalpha() and prev_d and prev_d[-1].isalpha():
                    c1 = prev_d[-1]
                    c2 = curr_d[0]
                    is_v1 = c1 in _VOWELS
                    is_v2 = c2 in _VOWELS
                    if not is_v1 and not is_v2:
                        cc += 1
                    elif not is_v1 and is_v2:
                        cv += 1
                    elif is_v1 and not is_v2:
                        vc += 1
                    else:
                        vv += 1
            tot_trans = cc + cv + vc + vv
            if tot_trans > 0:
                self.cc_share = cc / tot_trans
                self.cv_share = cv / tot_trans
                self.vc_share = vc / tot_trans
                self.vv_share = vv / tot_trans
                probs = np.array([cc, cv, vc, vv], dtype=np.float64) / tot_trans
                probs = probs[probs > 0]
                self.phonotactic_entropy = float(-np.sum(probs * np.log2(probs)))
            else:
                self.cc_share = 0.0
                self.cv_share = 0.0
                self.vc_share = 0.0
                self.vv_share = 0.0
                self.phonotactic_entropy = 0.0

            # SentencePiece & Unicode habits
            sp_leading_space = sum(1 for d in cl_decoded if d.startswith(" "))
            self.sp_leading_space_share = sp_leading_space / n_cl if n_cl > 0 else 0.0
        else:
            self.cl100k_count = 0
            self.o200k_count = 0
            self.cl100k_chars_per_tok = 0.0
            self.o200k_chars_per_tok = 0.0
            self.comp_diff = 0.0
            self.tok_ratio_o200k_cl100k = 1.0
            self.diff_tok_per_word = 0.0
            self.cl100k_tok_len_mean = 0.0
            self.cl100k_tok_len_std = 0.0
            self.o200k_tok_len_mean = 0.0
            self.o200k_tok_len_std = 0.0
            self.o200k_merged_contractions = 0
            self.cl100k_entropy = 0.0
            self.o200k_entropy = 0.0
            self.tok_entropy_diff = 0.0
            self.cc_share = 0.0
            self.cv_share = 0.0
            self.vc_share = 0.0
            self.vv_share = 0.0
            self.phonotactic_entropy = 0.0
            self.sp_leading_space_share = 0.0

        # Digit chunks & Whitespace habits
        self.sp_digit_chunks = len(_DIGIT_CHUNK_RE.findall(text))
        n_spaces = sum(1 for c in text if c.isspace())
        self.whitespace_char_ratio = n_spaces / self.n_chars if self.n_chars > 0 else 0.0
        n_multibyte = sum(1 for c in text if ord(c) > 127)
        self.multibyte_utf8_share = n_multibyte / self.n_chars if self.n_chars > 0 else 0.0

        # ------------------------------------------------------------------- #
        # 2. Clitic Attachment & Morphology
        # ------------------------------------------------------------------- #
        self.straight_clitics = len(_STRAIGHT_CLITIC_RE.findall(text))
        self.curly_clitics = len(_CURLY_CLITIC_RE.findall(text))
        tot_clitics = self.straight_clitics + self.curly_clitics
        self.curly_clitic_share = (
            self.curly_clitics / tot_clitics if tot_clitics > 0 else 0.0
        )

        formal_exp = len(_FORMAL_EXPANSION_RE.findall(text))
        tot_contraction_opp = formal_exp + tot_clitics
        self.contraction_suppression = (
            formal_exp / tot_contraction_opp if tot_contraction_opp > 0 else 0.0
        )

        self.prefix_count = len(_PREFIX_RE.findall(text))
        self.suffix_count = len(_SUFFIX_RE.findall(text))
        self.affix_density = (
            (self.prefix_count + self.suffix_count) / self.n_words
            if self.n_words > 0 else 0.0
        )

        # ------------------------------------------------------------------- #
        # 3. Heuristic Syntactic Depth & Clausal Subordination
        # ------------------------------------------------------------------- #
        self.subord_conj_count = len(_SUBORD_CONJ_RE.findall(text))
        self.rel_pronoun_count = len(_RELATIVE_PRONOUN_RE.findall(text))
        tot_subord = self.subord_conj_count + self.rel_pronoun_count
        self.subord_index_per_sent = tot_subord / self.n_sent if self.n_sent > 0 else 0.0

        # Left vs Right Branching depth (Yngve depth proxy)
        left_depths: List[int] = []
        right_depths: List[int] = []
        preposed_count = 0
        left_heavy_count = 0
        clause_counts: List[int] = []
        clause_word_lens: List[int] = []

        for s in sents:
            l_d, r_d = _analyze_sentence_branching(s)
            left_depths.append(l_d)
            right_depths.append(r_d)
            if l_d >= 6:
                left_heavy_count += 1
            if bool(_SUBORD_LEAD_RE.match(s.strip())):
                preposed_count += 1

            # Clause segmentation by punctuation delimiters
            delims = re.split(r"[,;:\u2014()]", s)
            delims = [part.strip() for part in delims if part.strip()]
            n_clauses = max(1, len(delims))
            clause_counts.append(n_clauses)
            for d in delims:
                w_count = len(_WORD_RE.findall(d))
                if w_count > 0:
                    clause_word_lens.append(w_count)

        self.left_branching_mean = float(np.mean(left_depths)) if left_depths else 0.0
        self.left_branching_max = float(np.max(left_depths)) if left_depths else 0.0
        mean_r = float(np.mean(right_depths)) if right_depths else 0.0
        self.left_to_right_ratio = (self.left_branching_mean + 1.0) / (mean_r + 1.0)
        self.left_heavy_share = (
            left_heavy_count / self.n_sent if self.n_sent > 0 else 0.0
        )
        self.preposed_clause_share = (
            preposed_count / self.n_sent if self.n_sent > 0 else 0.0
        )

        self.clauses_per_sent_mean = (
            float(np.mean(clause_counts)) if clause_counts else 1.0
        )
        self.complex_sent_share = (
            sum(1 for c in clause_counts if c >= 2) / self.n_sent
            if self.n_sent > 0 else 0.0
        )
        self.compound_complex_share = (
            sum(1 for c in clause_counts if c >= 3) / self.n_sent
            if self.n_sent > 0 else 0.0
        )

        if clause_word_lens:
            mu_cl = float(np.mean(clause_word_lens))
            std_cl = float(np.std(clause_word_lens))
            self.clause_len_cv = std_cl / (mu_cl + 1e-6)
        else:
            self.clause_len_cv = 0.0

        # Coordination cadence
        coord_init = sum(1 for s in sents if bool(_COORD_INITIAL_RE.match(s.strip())))
        self.coord_initial_share = coord_init / self.n_sent if self.n_sent > 0 else 0.0
        self.coord_medial_count = len(_COORD_MEDIAL_RE.findall(text))
        tot_coord = coord_init + self.coord_medial_count
        balance_denom = tot_coord + tot_subord
        self.coord_subord_balance = (
            (tot_coord - tot_subord) / balance_denom if balance_denom > 0 else 0.0
        )

        # ------------------------------------------------------------------- #
        # 4. Watermark Residual Signatures & Distributional Deviation
        # ------------------------------------------------------------------- #
        if self.tokens >= 2:
            n_pairs = self.tokens - 1
            # Window h=1, gamma=0.50 and gamma=0.25
            hits_h1_g50 = 0
            hits_h1_g25 = 0
            for t in range(1, self.tokens):
                ctx_hash = _fnv1a_32(self.lower[t - 1])
                score = (ctx_hash ^ _fnv1a_32(self.lower[t])) & 0xFFFF
                if score % 2 == 0:
                    hits_h1_g50 += 1
                if score % 4 == 0:
                    hits_h1_g25 += 1

            var_50 = n_pairs * 0.50 * 0.50
            var_25 = n_pairs * 0.25 * 0.75
            self.kirchen_z_h1_g50 = float(
                (hits_h1_g50 - n_pairs * 0.50) / math.sqrt(var_50 + 1e-9)
            )
            self.kirchen_z_h1_g25 = float(
                (hits_h1_g25 - n_pairs * 0.25) / math.sqrt(var_25 + 1e-9)
            )

            # Window h=2 (trigram context)
            if self.tokens >= 3:
                n_tri = self.tokens - 2
                hits_h2_g50 = 0
                hits_h2_g25 = 0
                for t in range(2, self.tokens):
                    ctx_str = f"{self.lower[t-2]}_{self.lower[t-1]}"
                    ctx_hash = _fnv1a_32(ctx_str)
                    score = (ctx_hash ^ _fnv1a_32(self.lower[t])) & 0xFFFF
                    if score % 2 == 0:
                        hits_h2_g50 += 1
                    if score % 4 == 0:
                        hits_h2_g25 += 1
                var_h2_50 = n_tri * 0.50 * 0.50
                var_h2_25 = n_tri * 0.25 * 0.75
                self.kirchen_z_h2_g50 = float(
                    (hits_h2_g50 - n_tri * 0.50) / math.sqrt(var_h2_50 + 1e-9)
                )
                self.kirchen_z_h2_g25 = float(
                    (hits_h2_g25 - n_tri * 0.25) / math.sqrt(var_h2_25 + 1e-9)
                )
            else:
                self.kirchen_z_h2_g50 = 0.0
                self.kirchen_z_h2_g25 = 0.0

            # Multi-seed variance across 4 salt seeds
            seeds = (0x811C9DC5, 0x1337BEEF, 0x42424242, 0xCAFEBABE)
            seed_z_list: List[float] = []
            for s in seeds:
                hits = 0
                for t in range(1, self.tokens):
                    ctx_hash = _fnv1a_32(self.lower[t - 1], seed=s)
                    score = (ctx_hash ^ _fnv1a_32(self.lower[t], seed=s)) & 0xFFFF
                    if score % 2 == 0:
                        hits += 1
                z_val = (hits - n_pairs * 0.50) / math.sqrt(var_50 + 1e-9)
                seed_z_list.append(z_val)
            self.wm_multiseed_std = float(np.std(seed_z_list))
            self.wm_multiseed_max = float(np.max(seed_z_list))
        else:
            self.kirchen_z_h1_g50 = 0.0
            self.kirchen_z_h1_g25 = 0.0
            self.kirchen_z_h2_g50 = 0.0
            self.kirchen_z_h2_g25 = 0.0
            self.wm_multiseed_std = 0.0
            self.wm_multiseed_max = 0.0

        # Word frequency counts, Zipf regression & Lexical concentration
        counter = Counter(self.lower)
        R = len(counter)
        if self.tokens >= 4 and R >= 4:
            counts_sorted = sorted(counter.values(), reverse=True)
            ranks = np.arange(1, len(counts_sorted) + 1, dtype=np.float64)
            log_r = np.log(ranks)
            log_f = np.log(np.array(counts_sorted, dtype=np.float64))

            var_r = np.var(log_r)
            if var_r > 1e-12:
                beta = float(np.cov(log_r, log_f)[0, 1] / var_r)
                alpha = float(np.mean(log_f) - beta * np.mean(log_r))
                pred = alpha + beta * log_r
                res = log_f - pred
                mse = float(np.mean(res ** 2))
                var_f = np.var(log_f)
                r2 = float(1.0 - (mse / var_f)) if var_f > 1e-12 else 0.0
                self.zipf_beta = beta
                self.zipf_mse = mse
                self.zipf_r2 = max(0.0, r2)
            else:
                self.zipf_beta = -1.0
                self.zipf_mse = 0.0
                self.zipf_r2 = 1.0
        else:
            self.zipf_beta = -1.0
            self.zipf_mse = 0.0
            self.zipf_r2 = 1.0

        if self.tokens > 0:
            probs = np.array(list(counter.values()), dtype=np.float64) / self.tokens
            unigram_ent = float(-np.sum(probs * np.log2(probs)))
            self.hapax_ratio = sum(1 for c in counter.values() if c == 1) / R if R > 0 else 0.0
            self.dis_ratio = sum(1 for c in counter.values() if c == 2) / R if R > 0 else 0.0
            pairs = sum(f * (f - 1) for f in counter.values())
            tot_pairs = self.tokens * (self.tokens - 1)
            self.simpson = float(pairs / tot_pairs) if tot_pairs > 0 else 0.0
        else:
            unigram_ent = 0.0
            self.hapax_ratio = 0.0
            self.dis_ratio = 0.0
            self.simpson = 0.0

        # Conditional bigram entropy
        if self.tokens >= 2:
            bigrams = [f"{self.lower[t - 1]} {self.lower[t]}" for t in range(1, self.tokens)]
            bi_counter = Counter(bigrams)
            bi_probs = np.array(list(bi_counter.values()), dtype=np.float64) / len(bigrams)
            bigram_ent = float(-np.sum(bi_probs * np.log2(bi_probs)))
            cond_ent = max(0.0, bigram_ent - unigram_ent)
            self.cond_to_unigram_entropy_ratio = (
                cond_ent / (unigram_ent + 1e-6) if unigram_ent > 0 else 0.0
            )
        else:
            self.cond_to_unigram_entropy_ratio = 0.0

        # ------------------------------------------------------------------- #
        # 5. Punctuation & Typography Fingerprints
        # ------------------------------------------------------------------- #
        c_left_dq = text.count("\u201c")
        c_right_dq = text.count("\u201d")
        c_straight_dq = text.count('"')
        tot_dq = c_left_dq + c_right_dq + c_straight_dq
        self.curly_dq_ratio = (c_left_dq + c_right_dq) / tot_dq if tot_dq > 0 else 0.0

        c_left_sq = text.count("\u2018")
        c_right_sq = text.count("\u2019")
        c_straight_sq = text.count("'")
        tot_sq = c_left_sq + c_right_sq + c_straight_sq
        self.curly_sq_ratio = (c_left_sq + c_right_sq) / tot_sq if tot_sq > 0 else 0.0

        em_sp = len(_EM_SPACED_RE.findall(text))
        em_un = len(_EM_UNSPACED_RE.findall(text))
        tot_em = em_sp + em_un
        self.spaced_em_ratio = em_sp / tot_em if tot_em > 0 else 0.0
        self.unspaced_em_rate = (em_un / self.n_words * 100.0) if self.n_words > 0 else 0.0

        num_tokens = len(_NUMBER_TOKEN_RE.findall(text))
        fmt_tokens = len(_FORMATTED_NUMBER_RE.findall(text))
        self.formatted_num_ratio = fmt_tokens / num_tokens if num_tokens > 0 else 0.0

        q_count = 0
        q_exp = 0
        for i, s in enumerate(sents):
            if s.rstrip().endswith("?"):
                q_count += 1
                if i + 1 < len(sents):
                    nxt = sents[i + 1].strip()
                    if _SUBORD_CONJ_RE.search(nxt) or bool(re.search(r"\b(?:because|since|therefore|thus|in fact|specifically)\b", nxt, re.IGNORECASE)):
                        q_exp += 1
        self.q_to_exp_ratio = q_exp / q_count if q_count > 0 else 0.0


# --------------------------------------------------------------------------- #
# Feature Registration
# --------------------------------------------------------------------------- #

class _CrossTokenizerFeatureSpec(NamedTuple):
    name: str
    doc: str
    fn: Callable[[_CrossTokenizerStats], float]


_CROSS_SPECS: List[_CrossTokenizerFeatureSpec] = []


def _feature(name: str, doc: str):
    def decorator(fn: Callable[[_CrossTokenizerStats], float]):
        _CROSS_SPECS.append(_CrossTokenizerFeatureSpec(name, doc, fn))
        return fn
    return decorator


# 1. Cross-Tokenizer Pre-Tokenization & Compression
@_feature(
    "cl100k_token_count",
    "Total token count produced by Tiktoken cl100k_base subword tokenizer."
)
def _f_cl100k_count(s: _CrossTokenizerStats) -> float:
    return float(s.cl100k_count)


@_feature(
    "o200k_token_count",
    "Total token count produced by Tiktoken o200k_base subword tokenizer."
)
def _f_o200k_count(s: _CrossTokenizerStats) -> float:
    return float(s.o200k_count)


@_feature(
    "cl100k_chars_per_token",
    "Mean characters per token under cl100k_base encoding."
)
def _f_cl100k_cpt(s: _CrossTokenizerStats) -> float:
    return s.cl100k_chars_per_tok


@_feature(
    "o200k_chars_per_token",
    "Mean characters per token under o200k_base encoding."
)
def _f_o200k_cpt(s: _CrossTokenizerStats) -> float:
    return s.o200k_chars_per_tok


@_feature(
    "compression_differential",
    "Differential compression ratio: chars/token under o200k minus chars/token under cl100k."
)
def _f_comp_diff(s: _CrossTokenizerStats) -> float:
    return s.comp_diff


@_feature(
    "o200k_cl100k_token_ratio",
    "Ratio of o200k token count to cl100k token count."
)
def _f_tok_ratio(s: _CrossTokenizerStats) -> float:
    return s.tok_ratio_o200k_cl100k


@_feature(
    "differential_tokens_per_word",
    "Difference in token count (cl100k - o200k) normalized by total words."
)
def _f_diff_tok_per_w(s: _CrossTokenizerStats) -> float:
    return s.diff_tok_per_word


@_feature(
    "o200k_contraction_merges_per_100w",
    "Frequency of straight apostrophe contractions merged into unitary tokens under o200k."
)
def _f_merged_contractions(s: _CrossTokenizerStats) -> float:
    return (s.o200k_merged_contractions / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "cl100k_token_length_mean",
    "Mean character length of cl100k subword tokens."
)
def _f_cl100k_len_mu(s: _CrossTokenizerStats) -> float:
    return s.cl100k_tok_len_mean


@_feature(
    "cl100k_token_length_std",
    "Standard deviation of cl100k subword token character lengths."
)
def _f_cl100k_len_std(s: _CrossTokenizerStats) -> float:
    return s.cl100k_tok_len_std


@_feature(
    "o200k_token_length_mean",
    "Mean character length of o200k subword tokens."
)
def _f_o200k_len_mu(s: _CrossTokenizerStats) -> float:
    return s.o200k_tok_len_mean


@_feature(
    "o200k_token_length_std",
    "Standard deviation of o200k subword token character lengths."
)
def _f_o200k_len_std(s: _CrossTokenizerStats) -> float:
    return s.o200k_tok_len_std


@_feature(
    "cl100k_token_entropy",
    "Shannon unigram entropy of cl100k tokens in the document."
)
def _f_cl100k_entropy(s: _CrossTokenizerStats) -> float:
    return s.cl100k_entropy


@_feature(
    "o200k_token_entropy",
    "Shannon unigram entropy of o200k tokens in the document."
)
def _f_o200k_entropy(s: _CrossTokenizerStats) -> float:
    return s.o200k_entropy


@_feature(
    "token_entropy_differential",
    "Token entropy divergence: cl100k token entropy minus o200k token entropy."
)
def _f_tok_ent_diff(s: _CrossTokenizerStats) -> float:
    return s.tok_entropy_diff


@_feature(
    "sp_leading_space_token_share",
    "Proportion of subword tokens beginning with whitespace, characterizing SentencePiece habits."
)
def _f_sp_leading_space(s: _CrossTokenizerStats) -> float:
    return s.sp_leading_space_share


@_feature(
    "sp_digit_chunk_rate_per_100w",
    "Frequency of contiguous digit groups per 100 words."
)
def _f_sp_digit_chunks(s: _CrossTokenizerStats) -> float:
    return (s.sp_digit_chunks / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "whitespace_to_char_ratio",
    "Ratio of whitespace characters to total characters."
)
def _f_whitespace_ratio(s: _CrossTokenizerStats) -> float:
    return s.whitespace_char_ratio


@_feature(
    "multibyte_utf8_char_share",
    "Share of non-ASCII multibyte UTF-8 characters."
)
def _f_multibyte_share(s: _CrossTokenizerStats) -> float:
    return s.multibyte_utf8_share


# 2. Clitic Attachment & Phonological / Morpheme Boundary Habits
@_feature(
    "straight_clitic_per_100w",
    "Frequency of straight apostrophe clitics ('s, 't, 're, 've, 'll) per 100 words."
)
def _f_straight_clitics(s: _CrossTokenizerStats) -> float:
    return (s.straight_clitics / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "curly_clitic_per_100w",
    "Frequency of curly apostrophe clitics (’s, ’t, ’re, ’ve, ’ll) per 100 words."
)
def _f_curly_clitics(s: _CrossTokenizerStats) -> float:
    return (s.curly_clitics / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "curly_clitic_share",
    "Proportion of clitics formatted with curly apostrophes."
)
def _f_curly_clitic_share(s: _CrossTokenizerStats) -> float:
    return s.curly_clitic_share


@_feature(
    "contraction_suppression_index",
    "Contraction suppression index: formal uncontracted forms / total opportunities."
)
def _f_contraction_suppress(s: _CrossTokenizerStats) -> float:
    return s.contraction_suppression


@_feature(
    "cc_boundary_transition_share",
    "Share of subword cuts occurring between consonants (Consonant-Consonant)."
)
def _f_cc_share(s: _CrossTokenizerStats) -> float:
    return s.cc_share


@_feature(
    "vc_boundary_transition_share",
    "Share of subword cuts occurring from vowel to consonant (Vowel-Consonant)."
)
def _f_vc_share(s: _CrossTokenizerStats) -> float:
    return s.vc_share


@_feature(
    "cv_boundary_transition_share",
    "Share of subword cuts occurring from consonant to vowel (Consonant-Vowel)."
)
def _f_cv_share(s: _CrossTokenizerStats) -> float:
    return s.cv_share


@_feature(
    "vv_boundary_transition_share",
    "Share of subword cuts occurring between vowels (Vowel-Vowel)."
)
def _f_vv_share(s: _CrossTokenizerStats) -> float:
    return s.vv_share


@_feature(
    "subword_phonotactic_entropy",
    "Shannon entropy across the 4 phonotactic boundary transition categories."
)
def _f_phonotactic_entropy(s: _CrossTokenizerStats) -> float:
    return s.phonotactic_entropy


@_feature(
    "prefix_split_per_100w",
    "Morphological prefix split density per 100 words."
)
def _f_prefix_density(s: _CrossTokenizerStats) -> float:
    return (s.prefix_count / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "suffix_split_per_100w",
    "Morphological suffix split density per 100 words."
)
def _f_suffix_density(s: _CrossTokenizerStats) -> float:
    return (s.suffix_count / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "affix_density_ratio",
    "Ratio of total detected morphological affixes to word count."
)
def _f_affix_density(s: _CrossTokenizerStats) -> float:
    return s.affix_density


# 3. Heuristic Syntactic Depth & Clausal Subordination
@_feature(
    "subordinating_conj_per_100w",
    "Subordinating conjunction density per 100 words (although, because, since, whereas)."
)
def _f_subord_conj(s: _CrossTokenizerStats) -> float:
    return (s.subord_conj_count / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "relative_pronoun_per_100w",
    "Relative pronoun and complementizer density per 100 words (which, who, whereby)."
)
def _f_rel_pronoun(s: _CrossTokenizerStats) -> float:
    return (s.rel_pronoun_count / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "subordination_index_per_sentence",
    "Mean subordinating syntactic markers per sentence."
)
def _f_subord_per_sent(s: _CrossTokenizerStats) -> float:
    return s.subord_index_per_sent


@_feature(
    "left_branching_depth_mean",
    "Mean pre-matrix-verbal word distance (Yngve cognitive memory depth proxy)."
)
def _f_left_depth_mu(s: _CrossTokenizerStats) -> float:
    return s.left_branching_mean


@_feature(
    "left_branching_depth_max",
    "Maximum pre-matrix-verbal word distance in the passage."
)
def _f_left_depth_max(s: _CrossTokenizerStats) -> float:
    return s.left_branching_max


@_feature(
    "left_to_right_branching_ratio",
    "Ratio of pre-verbal memory depth to post-verbal expansion depth."
)
def _f_left_right_ratio(s: _CrossTokenizerStats) -> float:
    return s.left_to_right_ratio


@_feature(
    "left_heavy_sentence_share",
    "Proportion of sentences with heavy left-branching depth (pre-verbal distance >= 6)."
)
def _f_left_heavy_share(s: _CrossTokenizerStats) -> float:
    return s.left_heavy_share


@_feature(
    "preposed_subordinate_clause_share",
    "Proportion of sentences opening with an adverbial or subordinate clause."
)
def _f_preposed_share(s: _CrossTokenizerStats) -> float:
    return s.preposed_clause_share


@_feature(
    "clauses_per_sentence_mean",
    "Mean punctuation-delimited clausal units per sentence."
)
def _f_clauses_per_sent(s: _CrossTokenizerStats) -> float:
    return s.clauses_per_sent_mean


@_feature(
    "complex_sentence_share",
    "Proportion of sentences containing 2 or more clausal units."
)
def _f_complex_share(s: _CrossTokenizerStats) -> float:
    return s.complex_sent_share


@_feature(
    "compound_complex_sentence_share",
    "Proportion of sentences containing 3 or more clausal units."
)
def _f_compound_complex_share(s: _CrossTokenizerStats) -> float:
    return s.compound_complex_share


@_feature(
    "clause_length_cv",
    "Coefficient of variation of clausal word lengths (syntactic pacing rhythm)."
)
def _f_clause_cv(s: _CrossTokenizerStats) -> float:
    return s.clause_len_cv


@_feature(
    "sentence_initial_coord_share",
    "Fraction of sentences opening with coordinate conjunctions (And, But, Yet, So)."
)
def _f_coord_initial(s: _CrossTokenizerStats) -> float:
    return s.coord_initial_share


@_feature(
    "internal_coord_per_100w",
    "Comma-delimited coordinate conjunction density per 100 words."
)
def _f_coord_medial(s: _CrossTokenizerStats) -> float:
    return (s.coord_medial_count / s.n_words * 100.0) if s.n_words > 0 else 0.0


@_feature(
    "coordination_to_subordination_balance",
    "Balance metric between coordination and subordination: (Coord - Subord) / (Coord + Subord)."
)
def _f_coord_subord_bal(s: _CrossTokenizerStats) -> float:
    return s.coord_subord_balance


# 4. Watermark Residual Signatures & Distributional Deviation
@_feature(
    "kirchenbauer_z_h1_g50",
    "Kirchenbauer watermark z-score under window h=1, gamma=0.50."
)
def _f_kirchen_h1_50(s: _CrossTokenizerStats) -> float:
    return s.kirchen_z_h1_g50


@_feature(
    "kirchenbauer_z_h1_g25",
    "Kirchenbauer watermark z-score under window h=1, gamma=0.25."
)
def _f_kirchen_h1_25(s: _CrossTokenizerStats) -> float:
    return s.kirchen_z_h1_g25


@_feature(
    "kirchenbauer_z_h2_g50",
    "Kirchenbauer watermark z-score under window h=2, gamma=0.50."
)
def _f_kirchen_h2_50(s: _CrossTokenizerStats) -> float:
    return s.kirchen_z_h2_g50


@_feature(
    "kirchenbauer_z_h2_g25",
    "Kirchenbauer watermark z-score under window h=2, gamma=0.25."
)
def _f_kirchen_h2_25(s: _CrossTokenizerStats) -> float:
    return s.kirchen_z_h2_g25


@_feature(
    "watermark_multiseed_z_std",
    "Standard deviation of watermark z-scores across multiple hash seeds."
)
def _f_wm_seed_std(s: _CrossTokenizerStats) -> float:
    return s.wm_multiseed_std


@_feature(
    "watermark_multiseed_z_max",
    "Maximum watermark z-score across multiple hash seeds."
)
def _f_wm_seed_max(s: _CrossTokenizerStats) -> float:
    return s.wm_multiseed_max


@_feature(
    "conditional_to_unigram_entropy_ratio",
    "Ratio of bigram conditional entropy to unigram entropy."
)
def _f_cond_ent_ratio(s: _CrossTokenizerStats) -> float:
    return s.cond_to_unigram_entropy_ratio


@_feature(
    "zipf_slope_beta",
    "Slope beta of log-rank vs log-frequency linear regression."
)
def _f_zipf_beta(s: _CrossTokenizerStats) -> float:
    return s.zipf_beta


@_feature(
    "zipf_residual_mse",
    "Mean squared residual error from the Zipfian power-law regression."
)
def _f_zipf_mse(s: _CrossTokenizerStats) -> float:
    return s.zipf_mse


@_feature(
    "zipf_r2",
    "Coefficient of determination R^2 of the Zipfian power-law regression."
)
def _f_zipf_r2(s: _CrossTokenizerStats) -> float:
    return s.zipf_r2


@_feature(
    "simpson_concentration",
    "Simpson lexical concentration index: probability that two words are identical."
)
def _f_simpson(s: _CrossTokenizerStats) -> float:
    return s.simpson


@_feature(
    "hapax_legomena_ratio",
    "Proportion of vocabulary occurring exactly once in the passage."
)
def _f_hapax(s: _CrossTokenizerStats) -> float:
    return s.hapax_ratio


@_feature(
    "dis_legomena_ratio",
    "Proportion of vocabulary occurring exactly twice in the passage."
)
def _f_dis(s: _CrossTokenizerStats) -> float:
    return s.dis_ratio


# 5. Punctuation & Sibling Discriminator Features
@_feature(
    "curly_dq_ratio",
    "Ratio of curly double quotes to total double quotes."
)
def _f_curly_dq(s: _CrossTokenizerStats) -> float:
    return s.curly_dq_ratio


@_feature(
    "curly_sq_ratio",
    "Ratio of curly single quotes to total single quotes."
)
def _f_curly_sq(s: _CrossTokenizerStats) -> float:
    return s.curly_sq_ratio


@_feature(
    "spaced_em_dash_ratio",
    "Ratio of spaced em-dashes (' — ') to total em-dashes."
)
def _f_spaced_em(s: _CrossTokenizerStats) -> float:
    return s.spaced_em_ratio


@_feature(
    "unspaced_em_dash_rate_per_100w",
    "Frequency of unspaced em-dashes ('—') per 100 words."
)
def _f_unspaced_em(s: _CrossTokenizerStats) -> float:
    return s.unspaced_em_rate


@_feature(
    "formatted_number_ratio",
    "Ratio of formatted numbers (with commas/decimals) to total numbers."
)
def _f_formatted_num(s: _CrossTokenizerStats) -> float:
    return s.formatted_num_ratio


@_feature(
    "question_to_explanation_ratio",
    "Fraction of rhetorical questions immediately followed by explanatory sentences."
)
def _f_q_to_exp(s: _CrossTokenizerStats) -> float:
    return s.q_to_exp_ratio


@_feature(
    "sentence_length_cv",
    "Coefficient of variation of sentence lengths in words (sentence rhythm)."
)
def _f_sent_cv(s: _CrossTokenizerStats) -> float:
    return s.sent_cv


CROSS_TOKENIZER_FEATURE_NAMES: Tuple[str, ...] = tuple(spec.name for spec in _CROSS_SPECS)
CROSS_TOKENIZER_FEATURE_DOCS: Dict[str, str] = {spec.name: spec.doc for spec in _CROSS_SPECS}
N_CROSS_TOKENIZER_FEATURES: int = len(_CROSS_SPECS)
_SPEC_FNS: Tuple[Callable[[_CrossTokenizerStats], float], ...] = tuple(
    spec.fn for spec in _CROSS_SPECS
)


def cross_tokenizer_feature_names() -> Tuple[str, ...]:
    """Return the ordered tuple of all cross-tokenizer feature names."""
    return CROSS_TOKENIZER_FEATURE_NAMES


def cross_tokenizer_feature_doc(name: str) -> str:
    """Return the documentation string for a registered feature."""
    return CROSS_TOKENIZER_FEATURE_DOCS[name]


# --------------------------------------------------------------------------- #
# Feature Extraction and Normalization
# --------------------------------------------------------------------------- #

def _coerce_texts(texts: Union[str, Iterable[str], None]) -> List[str]:
    if texts is None:
        return []
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, np.ndarray):
        if texts.ndim == 0:
            return [str(texts.item())]
        return [str(t) for t in texts.tolist()]
    return ["" if t is None else str(t) for t in texts]


def extract_raw_cross_tokenizer_features(texts: Union[str, Iterable[str], None]) -> np.ndarray:
    """
    Extract unnormalized cross-tokenizer stylometric and syntactic features.

    Returns:
        np.ndarray: Matrix of shape (N, N_CROSS_TOKENIZER_FEATURES), dtype float64.
    """
    items = _coerce_texts(texts)
    N = len(items)
    D = N_CROSS_TOKENIZER_FEATURES
    matrix = np.zeros((N, D), dtype=np.float64)

    for i, t in enumerate(items):
        st = _CrossTokenizerStats(t)
        row = matrix[i]
        for j, fn in enumerate(_SPEC_FNS):
            val = fn(st)
            if math.isnan(val) or math.isinf(val):
                row[j] = 0.0
            else:
                row[j] = float(val)
    return matrix


def _robust_scale(raw: np.ndarray, clip: float = 8.0) -> np.ndarray:
    """Batch-level robust z-score: (x - median) / (IQR / 1.349) with std fallback."""
    if raw.size == 0:
        return raw.astype(np.float64, copy=True)
    x = np.asarray(raw, dtype=np.float64)
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = (q75 - q25) / 1.349
    std = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    z = (x - med) / scale
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip)


def _quantile_scale(raw: np.ndarray) -> np.ndarray:
    """Map each feature to [-1, 1] via empirical average-rank transform."""
    if raw.size == 0:
        return raw.astype(np.float64, copy=True)
    x = np.asarray(raw, dtype=np.float64)
    N, D = x.shape
    if N < 2:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    for j in range(D):
        col = x[:, j]
        ranks = rankdata(col, method="average")
        out[:, j] = 2.0 * (ranks - 1.0) / (N - 1) - 1.0
    return out


def extract_cross_tokenizer_features(
    texts: Union[str, Iterable[str], None],
    *,
    method: str = "robust",
    clip: float = 8.0,
) -> np.ndarray:
    """
    Extract normalized cross-tokenizer forensic features.

    Args:
        texts: Input passage or collection of passages.
        method: 'robust' (default), 'quantile', or 'none'.
        clip: Symmetric clipping threshold for robust scaling.

    Returns:
        np.ndarray: Feature matrix (N, N_CROSS_TOKENIZER_FEATURES).
    """
    raw = extract_raw_cross_tokenizer_features(texts)
    if method == "none":
        return raw
    if method == "robust":
        return _robust_scale(raw, clip=clip)
    if method == "quantile":
        return _quantile_scale(raw)
    raise ValueError(f"Unknown scaling method: {method!r}")


# --------------------------------------------------------------------------- #
# Dedicated Sibling Discriminator
# --------------------------------------------------------------------------- #

class CrossTokenizerSiblingDiscriminator:
    """
    Dedicated Pairwise Sibling Discriminator for the 7 hardest candidate pairs.

    Employs mathematically calibrated feature projections:
      1. gemini-3.7-flash vs gemini-3.8-flash:
         Delta = 4.0 * (curly_dq_ratio - 0.35) - 2.5 * (q_to_exp_ratio - 0.20)
                 - 1.5 * (subord_index - 0.50) - 1.2 * (sent_cv - 0.40)
         (Negative favors gemini-3.7-flash, positive favors gemini-3.8-flash)
      2. claude-opus-4-6 vs claude-opus-5:
         Delta = 3.5 * (spaced_em_ratio - 0.40) + 2.0 * (left_branching_mean - 3.5)
                 + 1.8 * (subord_index - 0.60) + 1.2 * (complex_sent_share - 0.30)
         (Negative favors claude-opus-4-6, positive favors claude-opus-5)
      3. gpt-5.6-luna vs gpt-5.6-sol vs gpt-5.6-terra:
         Luna: High left-branching depth, high prefix density, low empirical grounding.
         Terra: High formatted numbers, high empirical grounding, high sequential flow.
         Sol: Balanced generalist.
    """

    def __init__(
        self,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        clip_log_odds: float = 3.5,
    ) -> None:
        self.pairs = list(pairs) if pairs is not None else list(TARGET_SIBLING_PAIRS)
        self.clip = float(clip_log_odds)
        self.empirical_models: Dict[Tuple[str, str], LogisticRegression] = {}
        self._feat_idx: Dict[str, int] = {
            name: i for i, name in enumerate(CROSS_TOKENIZER_FEATURE_NAMES)
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> CrossTokenizerSiblingDiscriminator:
        """
        Optionally fit empirical Logistic Regression classifiers if ground truth
        training labels are provided.
        """
        for m1, m2 in self.pairs:
            if m1 not in MODEL_TO_IDX or m2 not in MODEL_TO_IDX:
                continue
            i1 = MODEL_TO_IDX[m1]
            i2 = MODEL_TO_IDX[m2]
            mask = (y == i1) | (y == i2)
            if mask.sum() < 6 or len(np.unique(y[mask])) < 2:
                continue
            X_sub = X[mask]
            y_sub = (y[mask] == i2).astype(int)
            clf = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
            clf.fit(X_sub, y_sub)
            self.empirical_models[(m1, m2)] = clf
        return self

    def predict_log_odds(self, X_raw: np.ndarray) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Compute pairwise log-odds differences:
            log_odds = log(P(m2) / P(m1))
        Returns dictionary mapping (m1, m2) to np.ndarray of shape (N,).
        """
        N = X_raw.shape[0]
        results: Dict[Tuple[str, str], np.ndarray] = {}

        for m1, m2 in self.pairs:
            pair_key = (m1, m2)
            rev_key = (m2, m1)
            if pair_key in self.empirical_models:
                clf = self.empirical_models[pair_key]
                probs = clf.predict_proba(X_raw)[:, 1]
                probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
                log_odds = np.log(probs / (1.0 - probs))
                results[pair_key] = np.clip(log_odds, -self.clip, self.clip)
            elif rev_key in self.empirical_models:
                clf = self.empirical_models[rev_key]
                probs = clf.predict_proba(X_raw)[:, 1]
                probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
                log_odds = -np.log(probs / (1.0 - probs))
                results[pair_key] = np.clip(log_odds, -self.clip, self.clip)
            else:
                delta = self._predict_analytic(m1, m2, X_raw)
                results[pair_key] = np.clip(delta, -self.clip, self.clip)

        return results

    def _predict_analytic(self, m1: str, m2: str, X_raw: np.ndarray) -> np.ndarray:
        N = X_raw.shape[0]
        idx = self._feat_idx

        def col(name: str) -> np.ndarray:
            return X_raw[:, idx[name]]

        pair_set = {m1, m2}

        if pair_set == {"gemini-3.7-flash", "gemini-3.8-flash"}:
            # Positive favors gemini-3.8-flash; negative favors gemini-3.7-flash
            curly_dq = col("curly_dq_ratio")
            q_exp = col("question_to_explanation_ratio")
            subord = col("subordination_index_per_sentence")
            sent_cv = col("sentence_length_cv")
            comp_diff = col("compression_differential")
            delta = (
                -4.2 * (curly_dq - 0.20)
                - 2.8 * (q_exp - 0.15)
                - 1.5 * (subord - 0.50)
                - 1.2 * (sent_cv - 0.40)
                + 1.8 * (comp_diff - 0.05)
            )
            return delta if m2 == "gemini-3.8-flash" else -delta

        elif pair_set == {"claude-opus-4-6", "claude-opus-5"}:
            # Positive favors claude-opus-5; negative favors claude-opus-4-6
            spaced_em = col("spaced_em_dash_ratio")
            left_mu = col("left_branching_depth_mean")
            subord = col("subordination_index_per_sentence")
            cx_share = col("complex_sentence_share")
            simpson = col("simpson_concentration")
            delta = (
                3.4 * (spaced_em - 0.35)
                + 2.2 * (left_mu - 3.0)
                + 1.8 * (subord - 0.60)
                + 1.4 * (cx_share - 0.30)
                - 2.0 * (simpson - 0.04)
            )
            return delta if m2 == "claude-opus-5" else -delta

        elif pair_set == {"gpt-5.6-luna", "gpt-5.6-sol"}:
            # Positive favors gpt-5.6-sol; negative favors gpt-5.6-luna
            left_mu = col("left_branching_depth_mean")
            subord = col("subordinating_conj_per_100w")
            prefix = col("prefix_split_per_100w")
            merged_cont = col("o200k_contraction_merges_per_100w")
            delta = (
                -2.8 * (left_mu - 3.5)
                - 1.8 * (subord - 2.5)
                - 1.4 * (prefix - 2.8)
                + 1.0 * (merged_cont - 1.5)
            )
            return delta if m2 == "gpt-5.6-sol" else -delta

        elif pair_set == {"gpt-5.6-sol", "gpt-5.6-terra"}:
            # Positive favors gpt-5.6-terra; negative favors gpt-5.6-sol
            fmt_num = col("formatted_number_ratio")
            digit_chunks = col("sp_digit_chunk_rate_per_100w")
            left_mu = col("left_branching_depth_mean")
            delta = (
                3.2 * (fmt_num - 0.25)
                + 2.0 * (digit_chunks - 2.0)
                - 1.8 * (left_mu - 3.0)
            )
            return delta if m2 == "gpt-5.6-terra" else -delta

        elif pair_set == {"gpt-5.6-luna", "gpt-5.6-terra"}:
            # Positive favors gpt-5.6-terra; negative favors gpt-5.6-luna
            fmt_num = col("formatted_number_ratio")
            left_mu = col("left_branching_depth_mean")
            subord = col("subordinating_conj_per_100w")
            delta = (
                3.8 * (fmt_num - 0.25)
                - 3.2 * (left_mu - 3.5)
                - 2.2 * (subord - 2.5)
            )
            return delta if m2 == "gpt-5.6-terra" else -delta

        elif pair_set == {"glm-5.3", "claude-opus-4-6"}:
            # Positive favors claude-opus-4-6; negative favors glm-5.3
            spaced_em = col("spaced_em_dash_ratio")
            suppress = col("contraction_suppression_index")
            delta = 4.0 * (spaced_em - 0.20) - 4.5 * (suppress - 0.50)
            return delta if m2 == "claude-opus-4-6" else -delta

        elif pair_set == {"glm-5.3", "glm-5.3-flash"}:
            # Positive favors glm-5.3-flash; negative favors glm-5.3
            digit_chunks = col("sp_digit_chunk_rate_per_100w")
            subord = col("subordination_index_per_sentence")
            delta = 2.2 * (digit_chunks - 2.0) - 1.8 * (subord - 0.40)
            return delta if m2 == "glm-5.3-flash" else -delta

        return np.zeros(N, dtype=np.float64)


# --------------------------------------------------------------------------- #
# HodgeRank Least-Squares Potential Projection
# --------------------------------------------------------------------------- #

def _laplacian(
    pairs: Sequence[Tuple[str, str]], n_classes: int = 18
) -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in pairs:
        if a in MODEL_TO_IDX and b in MODEL_TO_IDX:
            i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b]
            A[i, j] = A[j, i] = 1.0
    return np.diag(A.sum(axis=1)) - A, A


def ls_potential(
    pairs: Sequence[Tuple[str, str]],
    terms: Dict[Tuple[str, str], np.ndarray],
    n_classes: int = 18,
) -> np.ndarray:
    """
    Moore-Penrose Least-Squares Potential Projection (HodgeRank).

    Solves L v = b with component-wise zero-centering, guaranteeing:
    1. Zero Condorcet cycles: transitive global potential.
    2. Strict zero mass transfer across disconnected components.
    """
    if not terms:
        return np.zeros((0, n_classes), dtype=np.float64)

    N = next(iter(terms.values())).shape[0]
    b = np.zeros((N, n_classes), dtype=np.float64)
    for (a, b_name), delta in terms.items():
        if a in MODEL_TO_IDX and b_name in MODEL_TO_IDX:
            i, j = MODEL_TO_IDX[a], MODEL_TO_IDX[b_name]
            b[:, j] += delta
            b[:, i] -= delta

    L, A = _laplacian(pairs, n_classes)
    n_comp, labels = connected_components(A, directed=False)
    v = np.zeros((N, n_classes), dtype=np.float64)

    for comp in range(n_comp):
        nodes = np.where(labels == comp)[0]
        if len(nodes) < 2:
            continue
        Lc = L[np.ix_(nodes, nodes)]
        bc = b[:, nodes].T
        free = nodes[1:]
        Lff = L[np.ix_(free, free)]
        if np.linalg.matrix_rank(Lff) < Lff.shape[0]:
            sol = np.linalg.lstsq(Lff, bc[1:], rcond=None)[0]
        else:
            sol = np.linalg.solve(Lff, bc[1:])
        v[:, free] = sol.T
        v[:, nodes] -= v[:, nodes].mean(axis=1, keepdims=True)

    return v


# --------------------------------------------------------------------------- #
# Pairwise Injector & End-to-End Fusion
# --------------------------------------------------------------------------- #

class CrossTokenizerPairwiseInjector:
    """
    Injects cross-tokenizer forensic and syntactic pairwise adjustments
    into base 18-class probability distributions.

    Mathematical formulation:
      D_ij(x) = Delta_ij(x) - tau * (log p_j(x) - log p_i(x))
      t_ij(x) = lambda * (p_i(x) + p_j(x)) * D_ij(x)
      log p'  = log p + L^+ b
      p'      = softmax(log p')
    """

    def __init__(
        self,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        lam: float = 0.15,
        tau: float = 1.0,
        gate: bool = True,
        clip: float = 3.5,
    ) -> None:
        self.discriminator = CrossTokenizerSiblingDiscriminator(
            pairs=pairs, clip_log_odds=clip
        )
        self.lam = float(lam)
        self.tau = float(tau)
        self.gate = bool(gate)
        self.clip = float(clip)

    def inject(
        self,
        P_base: np.ndarray,
        X_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust base probabilities using cross-tokenizer forensic signals.

        Args:
            P_base: (N, 18) float probability matrix.
            X_raw: (N, N_CROSS_TOKENIZER_FEATURES) raw feature matrix.

        Returns:
            np.ndarray: Calibrated (N, 18) probability matrix on the simplex.
        """
        N, K = P_base.shape
        log_P = np.log(np.maximum(P_base, 1e-12))
        pairwise_deltas = self.discriminator.predict_log_odds(X_raw)

        gated_terms: Dict[Tuple[str, str], np.ndarray] = {}
        for (m1, m2), delta in pairwise_deltas.items():
            if m1 not in MODEL_TO_IDX or m2 not in MODEL_TO_IDX:
                continue
            i1 = MODEL_TO_IDX[m1]
            i2 = MODEL_TO_IDX[m2]
            p1 = P_base[:, i1]
            p2 = P_base[:, i2]

            # Prior log-odds difference
            prior_diff = log_P[:, i2] - log_P[:, i1]
            D_ij = delta - self.tau * prior_diff

            if self.gate:
                support = p1 + p2
                t_ij = self.lam * support * D_ij
            else:
                t_ij = self.lam * D_ij

            gated_terms[(m1, m2)] = np.clip(t_ij, -self.clip, self.clip)

        # HodgeRank potential projection
        potential = ls_potential(self.discriminator.pairs, gated_terms, n_classes=K)
        log_P_new = log_P + potential
        log_P_new -= np.max(log_P_new, axis=1, keepdims=True)
        P_new = np.exp(log_P_new)
        return P_new / np.sum(P_new, axis=1, keepdims=True)


def fuse_cross_tokenizer_forensics(
    P_base: np.ndarray,
    texts: Union[str, Iterable[str]],
    *,
    lam: float = 0.15,
    tau: float = 1.0,
    gate: bool = True,
    clip: float = 3.5,
) -> np.ndarray:
    """
    End-to-end convenience pipeline:
      1. Extracts raw cross-tokenizer and syntactic features from texts.
      2. Computes calibrated sibling log-odds adjustments.
      3. Projects cycle-free potentials via HodgeRank graph Laplacian.
      4. Adjusts and normalizes base probability predictions.
    """
    X_raw = extract_raw_cross_tokenizer_features(texts)
    injector = CrossTokenizerPairwiseInjector(
        lam=lam, tau=tau, gate=gate, clip=clip
    )
    return injector.inject(P_base, X_raw)
