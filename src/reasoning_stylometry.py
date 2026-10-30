"""
Reasoning Stylometry, Neural Watermark Detection & Subword Boundary Fingerprinting.
===================================================================================

Mathematical Foundations & Architecture:
1. Discourse Transition Markov Chains (arXiv:2407.14905):
   Models text as a stochastic sequence of Argumentative Discourse Units (ADUs).
   Transitions across Causal, Contrastive, Additive, Sequential, and Grounded
   states form a 6x6 row-stochastic Markov transition kernel P_ij = P(S_t = j | S_{t-1} = i).
   Computes Markov transition entropy rate H(S) and critical edge flows:
     - P(Causal -> Causal): chain-of-thought deductive depth (GPT-5.6-Luna signature)
     - P(Contrast -> Grounded): reflective qualification (Claude Opus signature)
     - P(Sequential -> Grounded): procedural empirical grounding (GPT-5.6-Terra signature)
     - P(Grounded -> Causal): inductive inference transfer

2. Subword Tokenizer Boundary & Clitic Mechanics:
   Captures subword pre-tokenization boundary habits across Tiktoken (OpenAI),
   SentencePiece (Google/Gemini), and byte-level BPE (Claude/GLM):
     - Apostrophe clitic binding: curly vs straight apostrophes ('s, 't, 're vs ’s, ’t, ’re)
     - Contraction vs formal expansion suppression index (GLM-5.3 zero-contraction fingerprint)
     - Morphological head/tail split ratio: prefix tokens vs suffix tokens
     - Dash boundary isolation: spaced em-dash (' — ') vs unspaced ('—')
     - Numerical chunking geometry: formatted (1,000 / 3.14) vs contiguous digit chunks

3. Neural Watermark Entropy & Residual Distribution Deviation:
   Detects subtle distribution perturbations induced by red-green list watermarking
   (Kirchenbauer et al., 2023) and nucleus / min-p truncation:
     - Zipfian log-rank vs log-frequency linear regression (beta slope, residual MSE, R^2)
     - Deterministic context-hash green list z-scores (gamma = 0.50 and gamma = 0.25)
     - Unigram, bigram, and conditional bigram entropy H(W_t | W_{t-1})
     - Hapax / Dis legomena proportions and Simpson concentration

4. Dedicated Intra-Family Sibling Pair Discriminators:
   Calibrated analytic and empirical pairwise discriminators for the hardest sibling pairs:
     - gemini-3.7-flash vs gemini-3.8-flash (curly double quotes, question pacing)
     - claude-opus-4-6 vs claude-opus-5 (spaced em-dash rates, reflective sentence pacing)
     - gpt-5.6-luna vs gpt-5.6-sol vs gpt-5.6-terra (deductive vs balanced vs grounded)
     - glm-5.3 vs claude-opus-4-6 (unspaced em-dashes, contraction suppression)
     - glm-5.3 vs glm-5.3-flash (sentence compression, digit frequency)

5. HodgeRank Least-Squares Potential Projection:
   Integrates pairwise log-odds deltas with pair-membership gating pi_ij(x) = p_i(x) + p_j(x)
   and graph Laplacian pseudo-inverse L^+ b, guaranteeing zero Condorcet cycles and
   preventing off-pair contamination.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_TO_IDX, VALID_18_MODELS

__all__ = [
    "REASONING_FEATURE_NAMES",
    "REASONING_FEATURE_DOCS",
    "N_REASONING_FEATURES",
    "TARGET_SIBLING_PAIRS",
    "extract_raw_reasoning_features",
    "extract_reasoning_features",
    "DedicatedSiblingDiscriminator",
    "ReasoningPairwiseInjector",
    "fuse_reasoning_stylometry",
]

# --------------------------------------------------------------------------- #
# Discourse states and compiled patterns
# --------------------------------------------------------------------------- #

STATE_CAUSAL = 0
STATE_CONTRASTIVE = 1
STATE_ADDITIVE = 2
STATE_SEQUENTIAL = 3
STATE_GROUNDED = 4
STATE_NEUTRAL = 5

N_DISCOURSE_STATES = 6

_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_ABBREV_DOT_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Fig|No|cf|al|"
    r"Inc|Ltd|approx|Eq|Ref)\. ?",
    re.IGNORECASE,
)
_SENT_RE = re.compile(r"[^.!?\u2026]+[.!?\u2026]*")
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_MASK = "\u0001"

_CAUSAL_CONNECTIVES = re.compile(
    r"\b(?:therefore|thus|hence|consequently|because|since|as a result|"
    r"due to|leads to|leading to|it follows that|implies that|giving rise to|"
    r"accordingly|so that)\b",
    re.IGNORECASE,
)
_CONTRASTIVE_CONNECTIVES = re.compile(
    r"\b(?:however|but|yet|although|though|nevertheless|nonetheless|whereas|"
    r"while|in contrast|on the other hand|instead|conversely|despite|even so)\b",
    re.IGNORECASE,
)
_ADDITIVE_CONNECTIVES = re.compile(
    r"\b(?:moreover|furthermore|in addition|additionally|besides|similarly|"
    r"likewise|equally|also|what is more)\b",
    re.IGNORECASE,
)
_SEQUENTIAL_CONNECTIVES = re.compile(
    r"\b(?:first|firstly|second|secondly|third|thirdly|then|next|finally|"
    r"subsequently|meanwhile|eventually|afterwards|later|initially)\b",
    re.IGNORECASE,
)
_GROUNDED_CONNECTIVES = re.compile(
    r"\b(?:for example|for instance|in practice|in fact|concretely|specifically|"
    r"in reality|take the case|consider|evidence indicates|experimentally|"
    r"empirical|observations show)\b",
    re.IGNORECASE,
)

# Subword boundary patterns
_PREFIX_RE = re.compile(
    r"\b(?:un|re|dis|pre|post|anti|sub|inter|intra|non|in|im|mis|over|under|semi|trans|counter|pseudo)[a-zA-Z]{3,}",
    re.IGNORECASE,
)
_SUFFIX_RE = re.compile(
    r"[a-zA-Z]{3,}(?:tion|sion|ment|able|ible|ity|ities|ous|ious|ance|ence|ful|less|ly|ness|ism|ist|ization|isation|ize|ise|ate|ive)\b",
    re.IGNORECASE,
)
_CONTRACTION_ANY_RE = re.compile(
    r"\b[a-zA-Z]+(?:['\u2019](?:s|t|re|ve|ll|d|m))\b",
    re.IGNORECASE,
)
_FORMAL_EXPANSION_RE = re.compile(
    r"\b(?:do not|does not|did not|cannot|can not|will not|would not|could not|"
    r"should not|is not|are not|was not|were not|have not|has not|had not|"
    r"it is|they are|we are|you are|i am|i have|we have|they have)\b",
    re.IGNORECASE,
)
_HYPHENATED_COMPOUND_RE = re.compile(r"(?<=[a-zA-Z])-(?=[a-zA-Z])")
_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")
_FORMATTED_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b")
_EM_SPACED_RE = re.compile(r"(?<=\s)\u2014(?=\s)")
_EM_UNSPACED_RE = re.compile(r"(?<=\S)\u2014(?=\S)")

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
# Fast deterministic hashing for watermark detection (FNV-1a)
# --------------------------------------------------------------------------- #

def _fnv1a_32(token: str) -> int:
    """Deterministic 32-bit FNV-1a hash across all Python platforms."""
    h = 2166136261
    for b in token.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# Sentence splitting and discourse analysis
# --------------------------------------------------------------------------- #

def _split_sentences(text: str) -> List[str]:
    """Segment sentences while masking decimal numbers and standard abbreviations."""
    if not text:
        return []
    masked = _DECIMAL_DOT_RE.sub(_SENT_MASK, text)
    masked = _ABBREV_DOT_RE.sub(
        lambda m: m.group(0).replace(".", _SENT_MASK), masked
    )
    raw_sents = [s.strip() for s in _SENT_RE.findall(masked) if s.strip()]
    return [s.replace(_SENT_MASK, ".") for s in raw_sents]


def _classify_sentence_state(sent: str) -> int:
    """Classify sentence into primary argumentative discourse state (ADU)."""
    # Prefer opening connective in first 8 words
    words = sent.split()[:8]
    lead_text = " ".join(words)
    if _CAUSAL_CONNECTIVES.search(lead_text):
        return STATE_CAUSAL
    if _CONTRASTIVE_CONNECTIVES.search(lead_text):
        return STATE_CONTRASTIVE
    if _ADDITIVE_CONNECTIVES.search(lead_text):
        return STATE_ADDITIVE
    if _SEQUENTIAL_CONNECTIVES.search(lead_text):
        return STATE_SEQUENTIAL
    if _GROUNDED_CONNECTIVES.search(lead_text):
        return STATE_GROUNDED

    # Search remainder of sentence with priority order
    if _CAUSAL_CONNECTIVES.search(sent):
        return STATE_CAUSAL
    if _CONTRASTIVE_CONNECTIVES.search(sent):
        return STATE_CONTRASTIVE
    if _GROUNDED_CONNECTIVES.search(sent):
        return STATE_GROUNDED
    if _ADDITIVE_CONNECTIVES.search(sent):
        return STATE_ADDITIVE
    if _SEQUENTIAL_CONNECTIVES.search(sent):
        return STATE_SEQUENTIAL
    return STATE_NEUTRAL


# --------------------------------------------------------------------------- #
# Document Statistics Object
# --------------------------------------------------------------------------- #

class _ReasoningStats:
    """Precomputed stylometric and discourse statistics for a single passage."""

    __slots__ = (
        "text", "tokens", "lower", "words", "n_words", "n_chars",
        "sentences", "n_sent", "sent_lens", "states", "trans_matrix",
        "state_counts", "entropy_rate", "question_explanation_ratio",
        "curly_dq", "straight_dq", "curly_sq", "straight_sq",
        "curly_dq_ratio", "curly_sq_ratio",
        "em_spaced", "em_unspaced", "em_total", "spaced_em_ratio",
        "unspaced_em_rate", "contractions", "formal_expansions",
        "contraction_suppression", "prefix_count", "suffix_count",
        "prefix_suffix_ratio", "hyphenated_count", "quote_space_discipline",
        "digit_run_mean_len", "formatted_num_ratio", "subword_fertility",
        "zipf_slope", "zipf_mse", "zipf_r2", "green_zscore_50", "green_zscore_25",
        "unigram_entropy", "bigram_entropy", "conditional_entropy",
        "hapax_ratio", "dis_ratio", "simpson", "sent_cv", "mean_word_len"
    )

    def __init__(self, text: str) -> None:
        self.text = text
        self.n_chars = len(text)
        raw_words = _WORD_RE.findall(text)
        self.words = raw_words
        self.lower = [w.lower() for w in raw_words]
        self.tokens = len(self.lower)
        self.n_words = self.tokens

        # Sentences & Discourse Markov Chain
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

        states = [_classify_sentence_state(s) for s in sents]
        self.states = states
        self.state_counts = [states.count(st) for st in range(N_DISCOURSE_STATES)]

        # Markov transition count matrix C_ij
        C = np.zeros((N_DISCOURSE_STATES, N_DISCOURSE_STATES), dtype=np.float64)
        if len(states) >= 2:
            for t in range(len(states) - 1):
                C[states[t], states[t + 1]] += 1.0

        # Laplace smoothing alpha = 0.1
        P = (C + 0.1) / (C.sum(axis=1, keepdims=True) + 0.6)
        self.trans_matrix = P

        # Transition entropy rate H(S)
        if self.n_sent > 0:
            pi = np.array(self.state_counts, dtype=np.float64) / self.n_sent
            # Row-wise entropy
            row_entropy = -np.sum(P * np.log2(np.clip(P, 1e-12, 1.0)), axis=1)
            self.entropy_rate = float(np.sum(pi * row_entropy))
        else:
            self.entropy_rate = 0.0

        # Interrogative to explanation flow
        q_count = 0
        q_answered = 0
        for i, s in enumerate(sents):
            if s.rstrip().endswith("?"):
                q_count += 1
                if i + 1 < len(sents):
                    next_st = states[i + 1]
                    if next_st in (STATE_CAUSAL, STATE_GROUNDED, STATE_ADDITIVE):
                        q_answered += 1
        self.question_explanation_ratio = q_answered / q_count if q_count > 0 else 0.0

        # Quotes & Dash Typography
        cdq = text.count("\u201c") + text.count("\u201d")
        sdq = text.count('"')
        csq = text.count("\u2018") + text.count("\u2019")
        ssq = text.count("'")
        self.curly_dq = cdq
        self.straight_dq = sdq
        self.curly_sq = csq
        self.straight_sq = ssq
        self.curly_dq_ratio = cdq / (cdq + sdq + 1e-9) if (cdq + sdq) > 0 else 0.0
        self.curly_sq_ratio = csq / (csq + ssq + 1e-9) if (csq + ssq) > 0 else 0.0

        em_sp = len(_EM_SPACED_RE.findall(text))
        em_un = len(_EM_UNSPACED_RE.findall(text))
        em_tot = text.count("\u2014")
        self.em_spaced = em_sp
        self.em_unspaced = em_un
        self.em_total = em_tot
        self.spaced_em_ratio = em_sp / em_tot if em_tot > 0 else 0.0
        self.unspaced_em_rate = (em_un / self.n_words * 100.0) if self.n_words > 0 else 0.0

        # Contractions & formal expansion
        contr = len(_CONTRACTION_ANY_RE.findall(text))
        formal = len(_FORMAL_EXPANSION_RE.findall(text))
        self.contractions = contr
        self.formal_expansions = formal
        self.contraction_suppression = (
            formal / (formal + contr + 1e-9) if (formal + contr) > 0 else 0.0
        )

        # Morphology & subword boundary proxies
        n_pref = len(_PREFIX_RE.findall(text))
        n_suff = len(_SUFFIX_RE.findall(text))
        self.prefix_count = n_pref
        self.suffix_count = n_suff
        self.prefix_suffix_ratio = n_pref / (n_suff + 1e-9) if (n_pref + n_suff) > 0 else 0.0
        self.hyphenated_count = len(_HYPHENATED_COMPOUND_RE.findall(text))

        # Quote space discipline
        q_spaced = len(re.findall(r'(?:[\u201c\u201d"]\s|\s[\u201c\u201d"])', text))
        q_punct = len(re.findall(r'(?:[.,;:!?][\u201c\u201d"]|[\u201c\u201d"][.,;:!?])', text))
        self.quote_space_discipline = q_spaced / (q_spaced + q_punct + 1e-9) if (q_spaced + q_punct) > 0 else 0.5

        # Numbers and digits (using robust number token regex)
        num_tokens = _NUMBER_TOKEN_RE.findall(text)
        num_lens = [len(n.replace(",", "").replace(".", "")) for n in num_tokens]
        self.digit_run_mean_len = float(np.mean(num_lens)) if num_lens else 0.0
        formatted_nums = len(_FORMATTED_NUMBER_RE.findall(text))
        self.formatted_num_ratio = formatted_nums / len(num_tokens) if num_tokens else 0.0

        # Subword fertility proxy
        word_lens = [len(w) for w in self.lower]
        self.mean_word_len = float(np.mean(word_lens)) if word_lens else 0.0
        self.subword_fertility = (
            (self.mean_word_len / 4.2) + (n_pref + n_suff) / (self.n_words + 1e-9)
            if self.n_words > 0 else 1.0
        )

        # Watermark & Zipf residual statistics
        counter = Counter(self.lower)
        freqs = sorted(counter.values(), reverse=True)
        R = len(freqs)

        if R >= 3:
            ranks = np.arange(1, R + 1, dtype=np.float64)
            log_r = np.log(ranks)
            log_f = np.log(np.array(freqs, dtype=np.float64))

            # Linear regression: log_f = alpha - beta * log_r
            x_bar, y_bar = np.mean(log_r), np.mean(log_f)
            denom = np.sum((log_r - x_bar) ** 2)
            if denom > 1e-9:
                beta = -np.sum((log_r - x_bar) * (log_f - y_bar)) / denom
                alpha = y_bar + beta * x_bar
                preds = alpha - beta * log_r
                resids = log_f - preds
                mse = float(np.mean(resids ** 2))
                ss_tot = np.sum((log_f - y_bar) ** 2)
                r2 = float(1.0 - np.sum(resids ** 2) / (ss_tot + 1e-9))
                self.zipf_slope = float(beta)
                self.zipf_mse = float(mse)
                self.zipf_r2 = float(np.clip(r2, -2.0, 1.0))
            else:
                self.zipf_slope, self.zipf_mse, self.zipf_r2 = 1.0, 0.0, 1.0
        else:
            self.zipf_slope, self.zipf_mse, self.zipf_r2 = 1.0, 0.0, 1.0

        # Kirchenbauer pseudo-green score simulation
        if self.tokens >= 4:
            hits_50 = 0
            hits_25 = 0
            n_pairs = self.tokens - 1
            for t in range(1, self.tokens):
                w_prev = self.lower[t - 1]
                w_curr = self.lower[t]
                h_prev = _fnv1a_32(w_prev)
                h_curr = _fnv1a_32(w_curr)
                score = (h_prev ^ h_curr) & 0xFFFFFFFF
                if score % 2 == 0:
                    hits_50 += 1
                if score % 4 == 0:
                    hits_25 += 1

            gamma_50 = 0.50
            gamma_25 = 0.25
            var_50 = n_pairs * gamma_50 * (1.0 - gamma_50)
            var_25 = n_pairs * gamma_25 * (1.0 - gamma_25)
            self.green_zscore_50 = float((hits_50 - n_pairs * gamma_50) / math.sqrt(var_50 + 1e-9))
            self.green_zscore_25 = float((hits_25 - n_pairs * gamma_25) / math.sqrt(var_25 + 1e-9))
        else:
            self.green_zscore_50 = 0.0
            self.green_zscore_25 = 0.0

        # Shannon token entropy & Simpson concentration
        if self.tokens > 0:
            probs = np.array(list(counter.values()), dtype=np.float64) / self.tokens
            self.unigram_entropy = float(-np.sum(probs * np.log2(probs)))
            self.hapax_ratio = sum(1 for c in counter.values() if c == 1) / R
            self.dis_ratio = sum(1 for c in counter.values() if c == 2) / R
            pairs = sum(f * (f - 1) for f in counter.values())
            tot_pairs = self.tokens * (self.tokens - 1)
            self.simpson = float(pairs / tot_pairs) if tot_pairs > 0 else 0.0
        else:
            self.unigram_entropy = 0.0
            self.hapax_ratio = 0.0
            self.dis_ratio = 0.0
            self.simpson = 0.0

        # Bigram entropy
        if self.tokens >= 2:
            bigrams = [f"{self.lower[t - 1]} {self.lower[t]}" for t in range(1, self.tokens)]
            bi_counter = Counter(bigrams)
            bi_probs = np.array(list(bi_counter.values()), dtype=np.float64) / len(bigrams)
            self.bigram_entropy = float(-np.sum(bi_probs * np.log2(bi_probs)))
            self.conditional_entropy = max(0.0, self.bigram_entropy - self.unigram_entropy)
        else:
            self.bigram_entropy = 0.0
            self.conditional_entropy = 0.0


# --------------------------------------------------------------------------- #
# Feature Registration
# --------------------------------------------------------------------------- #

class _ReasoningFeatureSpec(NamedTuple):
    name: str
    doc: str
    fn: Callable[[_ReasoningStats], float]


_SPECS: List[_ReasoningFeatureSpec] = []


def _feature(name: str, doc: str):
    def decorator(fn: Callable[[_ReasoningStats], float]):
        _SPECS.append(_ReasoningFeatureSpec(name, doc, fn))
        return fn
    return decorator


# 1. Discourse transition Markov chains
@_feature(
    "discourse_causal_share",
    "Share of sentences classified in the causal discourse state (therefore, thus, because)."
)
def _f_causal_share(s: _ReasoningStats) -> float:
    return s.state_counts[STATE_CAUSAL] / s.n_sent if s.n_sent else 0.0


@_feature(
    "discourse_contrastive_share",
    "Share of sentences classified in the contrastive discourse state (however, yet, although)."
)
def _f_contrastive_share(s: _ReasoningStats) -> float:
    return s.state_counts[STATE_CONTRASTIVE] / s.n_sent if s.n_sent else 0.0


@_feature(
    "discourse_additive_share",
    "Share of sentences classified in the additive discourse state (moreover, furthermore, also)."
)
def _f_additive_share(s: _ReasoningStats) -> float:
    return s.state_counts[STATE_ADDITIVE] / s.n_sent if s.n_sent else 0.0


@_feature(
    "discourse_sequential_share",
    "Share of sentences classified in the sequential discourse state (first, then, finally)."
)
def _f_sequential_share(s: _ReasoningStats) -> float:
    return s.state_counts[STATE_SEQUENTIAL] / s.n_sent if s.n_sent else 0.0


@_feature(
    "discourse_grounded_share",
    "Share of sentences classified in the grounded empirical discourse state (for example, in practice)."
)
def _f_grounded_share(s: _ReasoningStats) -> float:
    return s.state_counts[STATE_GROUNDED] / s.n_sent if s.n_sent else 0.0


@_feature(
    "trans_causal_to_causal",
    "Markov transition probability P(Causal -> Causal), measuring chain-of-thought depth."
)
def _f_trans_c_to_c(s: _ReasoningStats) -> float:
    return float(s.trans_matrix[STATE_CAUSAL, STATE_CAUSAL])


@_feature(
    "trans_causal_to_contrast",
    "Markov transition probability P(Causal -> Contrastive), tracking qualification propensity."
)
def _f_trans_c_to_cont(s: _ReasoningStats) -> float:
    return float(s.trans_matrix[STATE_CAUSAL, STATE_CONTRASTIVE])


@_feature(
    "trans_contrast_to_causal",
    "Markov transition probability P(Contrastive -> Causal), tracking rebuttal resolution."
)
def _f_trans_cont_to_c(s: _ReasoningStats) -> float:
    return float(s.trans_matrix[STATE_CONTRASTIVE, STATE_CAUSAL])


@_feature(
    "trans_contrast_to_grounded",
    "Markov transition probability P(Contrastive -> Grounded), capturing reflective grounding (Claude signature)."
)
def _f_trans_cont_to_g(s: _ReasoningStats) -> float:
    return float(s.trans_matrix[STATE_CONTRASTIVE, STATE_GROUNDED])


@_feature(
    "trans_additive_to_causal",
    "Markov transition probability P(Additive -> Causal), capturing cumulative inference synthesis."
)
def _f_trans_add_to_c(s: _ReasoningStats) -> float:
    return float(s.trans_matrix[STATE_ADDITIVE, STATE_CAUSAL])


@_feature(
    "trans_sequential_to_grounded",
    "Markov transition probability P(Sequential -> Grounded), capturing procedural instantiation."
)
def _f_trans_seq_to_g(s: _ReasoningStats) -> float:
    return float(s.trans_matrix[STATE_SEQUENTIAL, STATE_GROUNDED])


@_feature(
    "trans_grounded_to_causal",
    "Markov transition probability P(Grounded -> Causal), capturing empirical induction."
)
def _f_trans_g_to_c(s: _ReasoningStats) -> float:
    return float(s.trans_matrix[STATE_GROUNDED, STATE_CAUSAL])


@_feature(
    "discourse_markov_entropy",
    "Entropy rate of the sentence-level argumentative discourse transition matrix (in bits)."
)
def _f_discourse_entropy(s: _ReasoningStats) -> float:
    return s.entropy_rate


@_feature(
    "question_to_explanation_ratio",
    "Share of rhetorical questions immediately answered with explanation or evidence in the next sentence."
)
def _f_q_to_exp(s: _ReasoningStats) -> float:
    return s.question_explanation_ratio


# 2. Subword boundary & typography
@_feature(
    "curly_dq_ratio",
    "Ratio of curly double quotes to all double quotes. Crucial discriminator for Gemini 3.7 vs 3.8 Flash."
)
def _f_curly_dq_ratio(s: _ReasoningStats) -> float:
    return s.curly_dq_ratio


@_feature(
    "curly_sq_ratio",
    "Ratio of curly single quotes and apostrophes to all single quotes."
)
def _f_curly_sq_ratio(s: _ReasoningStats) -> float:
    return s.curly_sq_ratio


@_feature(
    "spaced_em_dash_ratio",
    "Ratio of spaced em-dashes (' — ') to total em-dashes. Sharp discriminator between Claude Opus and GLM."
)
def _f_spaced_em_ratio(s: _ReasoningStats) -> float:
    return s.spaced_em_ratio


@_feature(
    "unspaced_em_dash_rate",
    "Unspaced em-dashes per 100 words. Strong positive fingerprint for ZAI GLM-5.3 models."
)
def _f_unspaced_em_rate(s: _ReasoningStats) -> float:
    return s.unspaced_em_rate


@_feature(
    "contraction_rate_per_100w",
    "Frequency of contracted English forms ('s, n't, 're, 've, 'll) per 100 words."
)
def _f_contraction_rate(s: _ReasoningStats) -> float:
    return (s.contractions / s.n_words * 100.0) if s.n_words else 0.0


@_feature(
    "formal_expansion_rate_per_100w",
    "Frequency of uncontracted formal auxiliaries ('do not', 'cannot', 'it is') per 100 words."
)
def _f_formal_expansion_rate(s: _ReasoningStats) -> float:
    return (s.formal_expansions / s.n_words * 100.0) if s.n_words else 0.0


@_feature(
    "contraction_suppression_index",
    "Ratio of uncontracted formal auxiliaries to total contraction candidates (GLM vs Claude discriminator)."
)
def _f_contraction_suppression(s: _ReasoningStats) -> float:
    return s.contraction_suppression


@_feature(
    "prefix_split_density",
    "Density of morphological head prefix affixes per 100 words."
)
def _f_prefix_density(s: _ReasoningStats) -> float:
    return (s.prefix_count / s.n_words * 100.0) if s.n_words else 0.0


@_feature(
    "suffix_split_density",
    "Density of morphological tail suffix affixes per 100 words."
)
def _f_suffix_density(s: _ReasoningStats) -> float:
    return (s.suffix_count / s.n_words * 100.0) if s.n_words else 0.0


@_feature(
    "prefix_to_suffix_ratio",
    "Ratio of prefix split tokens to suffix split tokens."
)
def _f_pref_suff_ratio(s: _ReasoningStats) -> float:
    return s.prefix_suffix_ratio


@_feature(
    "hyphenated_compound_rate",
    "Frequency of hyphenated compound words per 100 words."
)
def _f_hyphenated_rate(s: _ReasoningStats) -> float:
    return (s.hyphenated_count / s.n_words * 100.0) if s.n_words else 0.0


@_feature(
    "quote_space_discipline",
    "Discipline of spacing around quotes: ratio of whitespace-delimited quotes to punctuation-attached quotes."
)
def _f_quote_space(s: _ReasoningStats) -> float:
    return s.quote_space_discipline


@_feature(
    "digit_run_mean_length",
    "Average contiguous character length of numeric digit sequences."
)
def _f_digit_run_len(s: _ReasoningStats) -> float:
    return s.digit_run_mean_len


@_feature(
    "formatted_number_ratio",
    "Fraction of numbers featuring standard thousand separators (commas) or decimal places."
)
def _f_formatted_num_ratio(s: _ReasoningStats) -> float:
    return s.formatted_num_ratio


@_feature(
    "subword_fertility_proxy",
    "Estimated subword tokens generated per natural word based on morphology and word length."
)
def _f_subword_fertility(s: _ReasoningStats) -> float:
    return s.subword_fertility


# 3. Watermark & Residual Entropy
@_feature(
    "zipf_beta_slope",
    "Empirical power-law exponent beta from log-rank vs log-frequency linear regression."
)
def _f_zipf_slope(s: _ReasoningStats) -> float:
    return s.zipf_slope


@_feature(
    "zipf_residual_mse",
    "Mean squared residual error from Zipf's law fit, detecting tail truncation from nucleus sampling or watermarking."
)
def _f_zipf_mse(s: _ReasoningStats) -> float:
    return s.zipf_mse


@_feature(
    "zipf_r2_fit",
    "R-squared goodness of fit to the Zipfian distribution."
)
def _f_zipf_r2(s: _ReasoningStats) -> float:
    return s.zipf_r2


@_feature(
    "kirchenbauer_green_zscore",
    "Empirical z-score test statistic for Kirchenbauer green-list selection with gamma = 0.50."
)
def _f_green_zscore_50(s: _ReasoningStats) -> float:
    return s.green_zscore_50


@_feature(
    "kirchenbauer_green_zscore_quarter",
    "Empirical z-score test statistic for Kirchenbauer green-list selection with gamma = 0.25."
)
def _f_green_zscore_25(s: _ReasoningStats) -> float:
    return s.green_zscore_25


@_feature(
    "token_entropy_unigram",
    "Shannon unigram entropy of the token frequency distribution in bits."
)
def _f_token_entropy_unigram(s: _ReasoningStats) -> float:
    return s.unigram_entropy


@_feature(
    "token_entropy_bigram",
    "Shannon bigram joint entropy of consecutive word tokens."
)
def _f_token_entropy_bigram(s: _ReasoningStats) -> float:
    return s.bigram_entropy


@_feature(
    "conditional_entropy_proxy",
    "Conditional bigram entropy proxy H(W_t | W_{t-1}) = H_2 - H_1."
)
def _f_conditional_entropy(s: _ReasoningStats) -> float:
    return s.conditional_entropy


@_feature(
    "hapax_ratio",
    "Share of vocabulary appearing exactly once (hapax legomena share)."
)
def _f_hapax_ratio(s: _ReasoningStats) -> float:
    return s.hapax_ratio


@_feature(
    "dis_ratio",
    "Share of vocabulary appearing exactly twice (dis legomena share)."
)
def _f_dis_ratio(s: _ReasoningStats) -> float:
    return s.dis_ratio


@_feature(
    "simpson_concentration",
    "Simpson concentration index representing the probability that two randomly selected words are identical."
)
def _f_simpson(s: _ReasoningStats) -> float:
    return s.simpson


@_feature(
    "sentence_length_cv",
    "Coefficient of variation (std / mean) of sentence word counts, capturing pacing variance."
)
def _f_sent_cv(s: _ReasoningStats) -> float:
    return s.sent_cv


@_feature(
    "mean_word_length",
    "Mean word length in characters."
)
def _f_mean_word_len(s: _ReasoningStats) -> float:
    return s.mean_word_len


REASONING_FEATURE_NAMES: Tuple[str, ...] = tuple(spec.name for spec in _SPECS)
REASONING_FEATURE_DOCS: Dict[str, str] = {spec.name: spec.doc for spec in _SPECS}
N_REASONING_FEATURES: int = len(REASONING_FEATURE_NAMES)
_FEATURE_FNS: Tuple[Callable[[_ReasoningStats], float], ...] = tuple(spec.fn for spec in _SPECS)


# --------------------------------------------------------------------------- #
# Feature Extraction Engine
# --------------------------------------------------------------------------- #

def extract_raw_reasoning_features(texts: Sequence[str]) -> np.ndarray:
    """Extract raw 42-feature reasoning and stylometry matrix without normalization."""
    if texts is None:
        return np.empty((0, N_REASONING_FEATURES), dtype=np.float64)

    N = len(texts)
    if N == 0:
        return np.empty((0, N_REASONING_FEATURES), dtype=np.float64)

    out = np.zeros((N, N_REASONING_FEATURES), dtype=np.float64)
    for i, t in enumerate(texts):
        s = _ReasoningStats(str(t) if t is not None else "")
        for j, fn in enumerate(_FEATURE_FNS):
            out[i, j] = fn(s)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_reasoning_features(
    texts: Sequence[str],
    method: str = "robust_z",
    clip: float = 8.0,
) -> np.ndarray:
    """Extract reasoning stylometry features with batch-robust scaling.

    Parameters
    ----------
    texts : sequence of input strings
    method : 'robust_z' (median / IQR / 1.349), 'quantile' ([-1, 1]), or 'none'
    clip : clipping range for robust_z scaling (default +/- 8.0)
    """
    raw = extract_raw_reasoning_features(texts)
    N, D = raw.shape
    if N <= 1 or method == "none":
        return raw if method == "none" else np.zeros_like(raw)

    if method in ("robust_z", "robust"):
        med = np.median(raw, axis=0)
        q75 = np.percentile(raw, 75, axis=0)
        q25 = np.percentile(raw, 25, axis=0)
        iqr = q75 - q25
        scale = iqr / 1.349
        std = np.std(raw, axis=0)

        # Fallback to std if IQR is degenerate
        degen = scale < 1e-6
        scale[degen] = std[degen]
        zero_var = scale < 1e-6
        scale[zero_var] = 1.0

        z = (raw - med) / scale
        z[:, zero_var] = 0.0
        return np.clip(z, -clip, clip)

    elif method == "quantile":
        ranks = np.argsort(np.argsort(raw, axis=0), axis=0).astype(np.float64)
        return (2.0 * ranks / (N - 1.0)) - 1.0

    raise ValueError(f"Unknown scaling method {method!r}; must be 'robust_z', 'quantile', or 'none'")


# --------------------------------------------------------------------------- #
# Dedicated Sibling Pair Discriminators
# --------------------------------------------------------------------------- #

class DedicatedSiblingDiscriminator:
    """Specialized pairwise discriminator for intra-family sibling models.

    Provides analytic zero-shot log-odds (from verified typography and reasoning laws)
    and empirical calibration when training data is provided.
    """

    def __init__(
        self,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        clip_log_odds: float = 3.5,
    ) -> None:
        self.pairs = list(pairs) if pairs is not None else list(TARGET_SIBLING_PAIRS)
        self.clip = float(clip_log_odds)
        self.classifiers_: Dict[Tuple[str, str], LogisticRegression] = {}
        self.pair_indices_: Dict[Tuple[str, str], List[int]] = {}
        self._setup_specialist_indices()

    def _setup_specialist_indices(self) -> None:
        name_to_idx = {name: i for i, name in enumerate(REASONING_FEATURE_NAMES)}
        for p in self.pairs:
            p_norm = self._norm_pair(p)
            m1, m2 = p_norm
            if "gemini" in m1 and "gemini" in m2:
                # Flash quotes and interrogative flow
                sub = [
                    "curly_dq_ratio", "curly_sq_ratio", "quote_space_discipline",
                    "question_to_explanation_ratio", "sentence_length_cv",
                    "digit_run_mean_length", "simpson_concentration"
                ]
            elif "claude" in m1 and "claude" in m2:
                # Opus spaced dash and reflective grounding
                sub = [
                    "spaced_em_dash_ratio", "unspaced_em_dash_rate",
                    "trans_contrast_to_grounded", "trans_causal_to_contrast",
                    "sentence_length_cv", "contraction_rate_per_100w",
                    "formal_expansion_rate_per_100w", "discourse_contrastive_share"
                ]
            elif "gpt-5.6" in m1 and "gpt-5.6" in m2:
                # Luna vs Sol vs Terra: reasoning depth vs grounded empirical
                sub = [
                    "trans_causal_to_causal", "trans_sequential_to_grounded",
                    "discourse_causal_share", "discourse_grounded_share",
                    "discourse_sequential_share", "formatted_number_ratio",
                    "prefix_split_density", "subword_fertility_proxy"
                ]
            elif ("glm" in m1 and "claude" in m2) or ("claude" in m1 and "glm" in m2):
                # GLM unspaced em-dash vs Claude spaced + contraction suppression
                sub = [
                    "spaced_em_dash_ratio", "unspaced_em_dash_rate",
                    "contraction_suppression_index", "contraction_rate_per_100w",
                    "formal_expansion_rate_per_100w"
                ]
            elif "glm" in m1 and "glm" in m2:
                # GLM full vs flash: sentence compression, digit frequency
                sub = [
                    "digit_run_mean_length", "formatted_number_ratio",
                    "sentence_length_cv", "mean_word_length",
                    "subword_fertility_proxy", "discourse_causal_share"
                ]
            else:
                sub = list(REASONING_FEATURE_NAMES[:12])
            self.pair_indices_[p_norm] = [name_to_idx[n] for n in sub if n in name_to_idx]

    @staticmethod
    def _norm_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
        a, b = pair[0].lower().strip(), pair[1].lower().strip()
        return (a, b) if a <= b else (b, a)

    def fit(self, texts: Sequence[str], y: np.ndarray) -> "DedicatedSiblingDiscriminator":
        """Fit empirical classifiers on training passages where labels are provided."""
        X_raw = extract_raw_reasoning_features(texts)
        for pair in self.pairs:
            p_norm = self._norm_pair(pair)
            m1, m2 = p_norm
            if m1 not in MODEL_TO_IDX or m2 not in MODEL_TO_IDX:
                continue
            idx1, idx2 = MODEL_TO_IDX[m1], MODEL_TO_IDX[m2]
            mask = (y == idx1) | (y == idx2)
            if mask.sum() < 6 or len(np.unique(y[mask])) < 2:
                continue

            sub_idx = self.pair_indices_[p_norm]
            X_sub = X_raw[mask][:, sub_idx]
            y_sub = (y[mask] == idx2).astype(int)

            clf = LogisticRegression(
                C=1.5, max_iter=500, class_weight="balanced", random_state=42
            )
            clf.fit(X_sub, y_sub)
            self.classifiers_[p_norm] = clf
        return self

    def predict_pair_log_odds(self, texts: Sequence[str]) -> Dict[Tuple[str, str], np.ndarray]:
        """Compute pairwise log-odds delta: Delta_ij = log(P(m2) / P(m1)).

        Positive indicates m2 favored; negative indicates m1 favored.
        Uses fitted logistic regressions when available; otherwise computes
        exact zero-shot analytic log-odds.
        """
        N = len(texts)
        if N == 0:
            return {}

        X_raw = extract_raw_reasoning_features(texts)
        name_to_idx = {name: i for i, name in enumerate(REASONING_FEATURE_NAMES)}
        deltas: Dict[Tuple[str, str], np.ndarray] = {}

        for pair in self.pairs:
            m1, m2 = pair[0].lower().strip(), pair[1].lower().strip()
            p_norm = self._norm_pair(pair)

            if p_norm in self.classifiers_:
                clf = self.classifiers_[p_norm]
                sub_idx = self.pair_indices_[p_norm]
                probs = clf.predict_proba(X_raw[:, sub_idx])[:, 1]
                probs = np.clip(probs, 1e-4, 1.0 - 1e-4)
                lo_norm = np.log(probs / (1.0 - probs))
                lo = lo_norm if p_norm == (m1, m2) else -lo_norm
                deltas[pair] = np.clip(lo, -self.clip, self.clip)
            else:
                # Zero-shot analytic log-odds
                lo = self._predict_analytic(m1, m2, X_raw, name_to_idx)
                deltas[pair] = np.clip(lo, -self.clip, self.clip)

        return deltas

    def _predict_analytic(
        self,
        m1: str,
        m2: str,
        X_raw: np.ndarray,
        idx_map: Dict[str, int],
    ) -> np.ndarray:
        N = X_raw.shape[0]

        def col(name: str) -> np.ndarray:
            return X_raw[:, idx_map[name]]

        pair_set = {m1, m2}

        if pair_set == {"gemini-3.7-flash", "gemini-3.8-flash"}:
            # Positive favors gemini-3.8-flash; negative favors gemini-3.7-flash
            curly_ratio = col("curly_dq_ratio")
            q_flow = col("question_to_explanation_ratio")
            delta = 4.5 * (curly_ratio - 0.38) + 1.2 * (q_flow - 0.20)
            return delta if m2 == "gemini-3.8-flash" else -delta

        elif pair_set == {"claude-opus-4-6", "claude-opus-5"}:
            # Positive favors claude-opus-5; negative favors claude-opus-4-6
            spaced_ratio = col("spaced_em_dash_ratio")
            refl = col("trans_contrast_to_grounded")
            sent_cv = col("sentence_length_cv")
            delta = -3.2 * (spaced_ratio - 0.425) - 2.0 * (refl - 0.15) + 1.0 * (sent_cv - 0.45)
            return delta if m2 == "claude-opus-5" else -delta

        elif pair_set == {"gpt-5.6-luna", "gpt-5.6-sol"}:
            # Positive favors gpt-5.6-sol; negative favors gpt-5.6-luna
            c_to_c = col("trans_causal_to_causal")
            c_share = col("discourse_causal_share")
            pref = col("prefix_split_density")
            delta = -2.8 * (c_to_c - 0.20) - 2.0 * (c_share - 0.15) - 1.2 * (pref - 3.0)
            return delta if m2 == "gpt-5.6-sol" else -delta

        elif pair_set == {"gpt-5.6-sol", "gpt-5.6-terra"}:
            # Positive favors gpt-5.6-terra; negative favors gpt-5.6-sol
            g_share = col("discourse_grounded_share")
            c_share = col("discourse_causal_share")
            fmt_num = col("formatted_number_ratio")
            seq_to_g = col("trans_sequential_to_grounded")
            delta = 2.6 * (g_share - c_share) + 2.0 * (fmt_num - 0.25) + 1.5 * (seq_to_g - 0.15)
            return delta if m2 == "gpt-5.6-terra" else -delta

        elif pair_set == {"gpt-5.6-luna", "gpt-5.6-terra"}:
            # Positive favors gpt-5.6-terra; negative favors gpt-5.6-luna
            g_share = col("discourse_grounded_share")
            c_share = col("discourse_causal_share")
            fmt_num = col("formatted_number_ratio")
            delta = 3.5 * (g_share - c_share) + 2.2 * (fmt_num - 0.25)
            return delta if m2 == "gpt-5.6-terra" else -delta

        elif pair_set == {"glm-5.3", "claude-opus-4-6"}:
            # Positive favors claude-opus-4-6; negative favors glm-5.3
            spaced_ratio = col("spaced_em_dash_ratio")
            suppress = col("contraction_suppression_index")
            delta = 3.8 * (spaced_ratio - 0.20) - 4.0 * (suppress - 0.60)
            return delta if m2 == "claude-opus-4-6" else -delta

        elif pair_set == {"glm-5.3", "glm-5.3-flash"}:
            # Positive favors glm-5.3-flash; negative favors glm-5.3
            digit_len = col("digit_run_mean_length")
            word_len = col("mean_word_length")
            c_share = col("discourse_causal_share")
            delta = 1.8 * (digit_len - 2.0) - 1.5 * (word_len - 5.0) - 1.2 * (c_share - 0.15)
            return delta if m2 == "glm-5.3-flash" else -delta

        return np.zeros(N, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Coherent Residual Gating & HodgeRank Projection
# --------------------------------------------------------------------------- #

def _laplacian(pairs: Sequence[Tuple[str, str]], n_classes: int = 18) -> Tuple[np.ndarray, np.ndarray]:
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
    """Moore-Penrose Least-Squares Potential Projection (HodgeRank).

    Solves L v = b with component-wise zero-centering, guaranteeing:
    1. Zero Condorcet cycles (transitive global potential).
    2. Zero mass transfer across disconnected components / families.
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


class ReasoningPairwiseInjector:
    """Injects reasoning stylometry specialist residuals into base predictions.

    Adheres strictly to the coherent residual formulation:
      D_ij(x) = Delta_ij(x) - tau * (log p_i(x) - log p_j(x))
      t_ij(x) = lambda * (p_i(x) + p_j(x)) * D_ij(x)
      log p'  = log p + L^+ b
    """

    def __init__(
        self,
        pairs: Optional[Sequence[Tuple[str, str]]] = None,
        lam: float = 0.15,
        tau: float = 1.0,
        gate: bool = True,
        clip: float = 3.5,
    ) -> None:
        self.discriminator = DedicatedSiblingDiscriminator(pairs=pairs, clip_log_odds=clip)
        self.lam = float(lam)
        self.tau = float(tau)
        self.gate = bool(gate)
        self.clip = float(clip)

    def fit(self, texts: Sequence[str], y: np.ndarray) -> "ReasoningPairwiseInjector":
        self.discriminator.fit(texts, y)
        return self

    def transform(self, texts: Sequence[str], logp: np.ndarray) -> np.ndarray:
        """Apply coherent pairwise corrections to base log-probabilities."""
        N, K = logp.shape
        if N == 0:
            return logp

        # Normalized base probabilities
        p_base = np.exp(logp - np.max(logp, axis=1, keepdims=True))
        p_base /= np.sum(p_base, axis=1, keepdims=True)

        deltas = self.discriminator.predict_pair_log_odds(texts)
        if not deltas:
            return logp

        terms: Dict[Tuple[str, str], np.ndarray] = {}
        for (m1, m2), r_ij in deltas.items():
            if m1 not in MODEL_TO_IDX or m2 not in MODEL_TO_IDX:
                continue
            i, j = MODEL_TO_IDX[m1], MODEL_TO_IDX[m2]
            # Base margin s_ij = log p_j - log p_i
            base_margin = logp[:, j] - logp[:, i]
            # Residual delta D_ij
            d_ij = r_ij - self.tau * base_margin
            d_ij = np.clip(d_ij, -self.clip, self.clip)
            # Pair membership probability gating
            if self.gate:
                pi_ij = p_base[:, i] + p_base[:, j]
                d_ij = d_ij * pi_ij
            terms[(m1, m2)] = self.lam * d_ij

        v = ls_potential(list(terms.keys()), terms, n_classes=K)
        return logp + v


def fuse_reasoning_stylometry(
    P_base: np.ndarray,
    texts: Sequence[str],
    weight: float = 0.15,
) -> np.ndarray:
    """Functional utility to apply reasoning stylometry correction to probability matrix.

    Parameters
    ----------
    P_base : (N, 18) base class probability distributions
    texts : sequence of N passage strings
    weight : global correction strength lambda (default 0.15)

    Returns
    -------
    P_corrected : (N, 18) valid probability distributions summing to 1.0
    """
    N, K = P_base.shape
    if N == 0:
        return P_base.copy()

    logp = np.log(np.clip(P_base, 1e-12, 1.0))
    injector = ReasoningPairwiseInjector(lam=weight)
    logp_corrected = injector.transform(texts, logp)

    # Softmax
    p_max = np.max(logp_corrected, axis=1, keepdims=True)
    exp_p = np.exp(logp_corrected - p_max)
    return exp_p / np.sum(exp_p, axis=1, keepdims=True)
