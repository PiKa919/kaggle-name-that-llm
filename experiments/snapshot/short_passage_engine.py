"""
Short-passage specialisation and adversarial robustness for "Name That LLM".

Why this file exists
--------------------
On prompt-grouped OOF, per-class recall collapses with passage length:

    < 60 words   0.27 balanced recall
    60-160       0.36
    160+         0.46

21.5% of train sits below 60 words and the test set is drawn from the same
length distribution, so closing the short-bin gap is worth up to ~0.04-0.06 of
the global balanced-accuracy score.  Nothing in a 7B-14B decoder's final hidden
state fixes this: with 40-60 words, self-attention has too few context tokens
for the semantic layer to resolve a persona, and the layer that *does* help --
early tokenisation-adjacent representations -- is exactly the layer a classifier
head trained on 250-word passages learns to ignore.

Two observations drive the design here.

1. Length is the only thing that breaks.  Surface authorial fingerprints are
   invariant under truncation: nobody's em-dash rate, curly-quote habit,
   contraction rate or affix morphology changes when a paragraph is cut in
   half.  Only the *estimator* of those rates gets noisier, by ~1/sqrt(n).
   A model trained and validated on short inputs therefore does not have to
   learn anything new; it has to stop being surprised by short inputs.

2. The right fix is a soft hand-off, not a hard switch.  `length_gate.py`
   already shows what a hard `is_short` branch costs: a one-word difference
   near the threshold moves the fused logit by up to 2.8 nats, which is larger
   than the margin between many sibling pairs.  The gate below is a logistic
   in word count with a finite bandwidth tau, so it is C-infinity and its
   induced logit perturbation is bounded by `|dlogit| * |dw/dn|`, which for
   tau = 10 is at most 0.025 * |dlogit| per word.

What is here
------------
`extract_short_passage_features`
    A dense, fixed-width (591 dims) view built only from signals that survive
    truncation: hashed character 1-4 grams (shingled and length-normalised, so
    the vector is a relative-frequency profile and is therefore invariant to
    text length), hashed 10-character prefixes/suffixes (how a completion
    *opens* is often a boilerplate marker -- "Certainly!", "In ", "The " --
    and models differ sharply there), word-shape histograms ("Xx", "ALL",
    "camelCase", ...), a fixed contraction panel, clause-boundary punctuation
    rates, and 59 normalised structural statistics.

`ShortPassageFallbackModel`
    A three-view linear ensemble (word n-grams with punctuation preserved,
    char_wb 2-5 grams, dense stylometry), each member strongly regularised and
    class-balanced, fitted only on rows that are already short plus word-level
    truncations of long rows.  Ensemble weights and a temperature are fitted on
    an internal holdout, so the readout is calibrated rather than assumed.

`entropy_regularized_short_gate`
    w = sigmoid(-(n - L0) / tau), P = (1-w) Psem + w Pshort, then a smooth
    entropy ceiling that pulls rows whose normalised entropy exceeds `h_lo`
    back toward the balanced prior.  This is the piece that stops a 45-word
    passage from being assigned to a random sibling on the strength of one
    lucky 4-gram: a flat posterior is not evidence, and the balanced-accuracy
    metric punishes a confident wrong sibling exactly as hard as a confident
    wrong lab.

Design constraints honoured
--------------------------
* numpy / scipy / scikit-learn only, no torch, no network, no downloads.
* Deterministic: no `hash()` (PYTHONHASHSEED-dependent) and no unseeded RNG.
* No lowercasing, no quote or dash normalisation anywhere in the features.
  The one exception is contraction *matching*, which folds curly apostrophes to
  straight ones so that "can't" and "can’t" land in the same panel slot; the raw
  punctuation counters keep the distinction.
* Every returned array is finite, non-negative and row-stochastic.

Measured behaviour (prompt-grouped 5-fold OOF on the 2,699 competition rows,
semantic baseline = word 1-2 + char_wb 3-5 TF-IDF, logistic, balanced)
--------------------------------------------------------------------
    bin            n     semantic   short-only
    0-40          18      0.159      0.000   (18 rows, ignore)
    40-60        537      0.258      0.237
    60-80        442      0.304      0.254
    80-100       325      0.275      0.286
    100-160      723      0.348      0.304
    160+         654      0.375      0.329
    all         2699      0.323      0.288

    gate L0=65, tau=10 (the specified default):  0.319 all, 0.235 short
    gate L0=45, tau=10 (OOF-tuned):              0.329 all, 0.264 short
    gate L0=40, tau=12:                          0.329 all, 0.258 short

Read this honestly: against a *TF-IDF* semantic view the fallback is not a
rescue, because that view already is surface stylometry -- the two models are
correlated and the gate is worth roughly +0.006 overall at L0=45 and -0.005 at
L0=65, both inside the noise band of 2,699 rows.  Its purpose is different: it
is a length-specialised, decorrelated surface view for pipelines whose semantic
model is a neural readout (LLM embeddings, which are the component that actually
collapses to 0.27 below 60 words).  Always tune L0 on your own OOF; 45-50 is
what this data supports.  The specified default of 65 pushes the fallback's
influence out to ~80 words, which is precisely where it is weakest relative to
the semantic view, so prefer 45-50 when you have OOF to tune on.

Typical use
-----------
    from src.short_passage_engine import (
        ShortPassageFallbackModel, entropy_regularized_short_gate,
    )

    short = ShortPassageFallbackModel(class_order=VALID_18_MODELS)
    short.fit(train.text, train.model, train.text.str.split().str.len())
    P_short = short.predict_proba(test.text)

    P_final = entropy_regularized_short_gate(P_semantic, P_short, n_words_test)
"""

from __future__ import annotations

import re
import string
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__all__ = [
    "extract_short_passage_features",
    "feature_names",
    "feature_index",
    "count_words",
    "ShortPassageFallbackModel",
    "entropy_regularized_short_gate",
    "short_gate_weight",
    "test_short_passage_invariants",
]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_N_CHAR_BUCKETS = 96
_CHAR_NGRAM_MIN = 1
_CHAR_NGRAM_MAX = 4

_N_AFFIX_BUCKETS = 24
_AFFIX_SLOTS = ("prefix10", "suffix10", "prefix4", "suffix4")

_SHAPE_CLASSES = (
    "x",       # single lowercase character ("a")
    "xx",      # all lowercase ("word")
    "Xx",      # capitalised ("Word")
    "Xxx",     # capitalised, length >= 4 ("Word")
    "X",       # single uppercase character ("I")
    "ALL",     # all caps ("NASA")
    "camel",   # upper inside, upper start ("CamelCase")
    "xX",      # lower start, upper inside ("iPhone")
    "xd",      # letters and digits ("GPT4")
    "mixed",   # anything else alphabetic
    "num",     # pure digits ("2024")
    "punct",   # non-alphanumeric token
)

_CONTRACTIONS: Tuple[str, ...] = (
    "can't", "cannot", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "isn't", "aren't", "wasn't", "weren't", "it's", "that's", "there's",
    "what's", "who's", "let's", "here's", "i'm", "i've", "i'll", "i'd",
    "you're", "you've", "you'll", "we're", "we've", "we'll", "they're",
    "they've", "he's", "she's", "couldn't", "shouldn't", "hasn't", "haven't",
    "hadn't", "ain't", "gonna", "wanna",
)

# Deterministic, platform-independent hashing: FNV-1a's prime used as a
# polynomial base modulo 2**64.  numpy's uint64 array arithmetic wraps
# silently, which is exactly the ring arithmetic we want here.
_HASH_BASE = np.uint64(1099511628211)

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

_CHAR_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"char:{n}gram:{b:03d}"
    for n in range(_CHAR_NGRAM_MIN, _CHAR_NGRAM_MAX + 1)
    for b in range(_N_CHAR_BUCKETS)
)

_AFFIX_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"affix:{slot}:{b:02d}"
    for slot in _AFFIX_SLOTS
    for b in range(_N_AFFIX_BUCKETS)
)

_SHAPE_FEATURE_NAMES: Tuple[str, ...] = tuple(f"shape:{s}" for s in _SHAPE_CLASSES)

_CONTRACTION_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"contract:{c}" for c in _CONTRACTIONS
)

_NUMERIC_FEATURE_NAMES: Tuple[str, ...] = (
    # length / shape of the input (kept, because short-text models must know
    # how much evidence they are looking at -- the gate uses it explicitly)
    "len:log1p_words",
    "len:mean_sentence_words",
    "len:std_sentence_words",
    "len:mean_word_chars",
    "len:std_word_chars",
    "len:unique_char_rate",
    # lexical richness
    "lex:type_token_ratio",
    "lex:hapax_ratio",
    "lex:yule_k_log",
    "lex:cap_word_rate",
    "lex:all_caps_rate",
    "lex:title_word_rate",
    "lex:digit_rate",
    "lex:punct_rate",
    "lex:space_rate",
    "lex:marked_word_rate",
    # clause boundaries
    "punct:comma_per_100w",
    "punct:semicolon_per_100w",
    "punct:colon_per_100w",
    "punct:period_per_100w",
    "punct:exclam_per_100w",
    "punct:question_per_100w",
    "punct:em_dash_per_100w",
    "punct:en_dash_per_100w",
    "punct:hyphen_per_100w",
    "punct:ellipsis_per_100w",
    "punct:paren_per_100w",
    "punct:bracket_per_100w",
    "punct:slash_per_100w",
    "punct:amp_per_100w",
    "punct:at_per_100w",
    "punct:hash_per_100w",
    "punct:star_per_100w",
    "punct:backtick_per_100w",
    "punct:percent_per_100w",
    "punct:dollar_per_100w",
    # quote typography (straight vs curly is the single strongest lab prior)
    "typ:straight_double_per_100w",
    "typ:straight_single_per_100w",
    "typ:curly_double_per_100w",
    "typ:curly_single_per_100w",
    "typ:straight_share_of_quotes",
    "typ:curly_share_of_quotes",
    # layout / markdown residue
    "layout:newline_rate",
    "layout:double_newline",
    "layout:bullet_line",
    "layout:numbered_line",
    "layout:bold_markers",
    # boundary habits
    "bound:starts_upper",
    "bound:starts_lower",
    "bound:starts_quote",
    "bound:ends_period",
    "bound:ends_question",
    "bound:ends_exclam",
    "bound:ends_colon",
    # discourse / person
    "disc:first_person_per_100w",
    "disc:first_plural_per_100w",
    "disc:second_person_per_100w",
    "disc:negation_per_100w",
    "disc:demonstrative_per_100w",
)

FEATURE_NAMES: Tuple[str, ...] = (
    _CHAR_FEATURE_NAMES
    + _AFFIX_FEATURE_NAMES
    + _SHAPE_FEATURE_NAMES
    + _CONTRACTION_FEATURE_NAMES
    + _NUMERIC_FEATURE_NAMES
)

_FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

if len(_FEATURE_INDEX) != len(FEATURE_NAMES):  # pragma: no cover - import guard
    raise RuntimeError("duplicate feature names in short-passage layout")

_N_FEATURES = len(FEATURE_NAMES)
_CHAR_SLICE = slice(0, len(_CHAR_FEATURE_NAMES))
_AFFIX_SLICE = slice(_CHAR_SLICE.stop, _CHAR_SLICE.stop + len(_AFFIX_FEATURE_NAMES))

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_TOKEN_RE = re.compile(r"\S+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+[\"'”’\)\]]*\s+")
_NUMBERED_RE = re.compile(r"(?m)^\s*\d+[.)]\s")
_BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s")

_STRAIGHT_QUOTES = ("'", '"')
_CURLY_SINGLE = ("\u2018", "\u2019")
_CURLY_DOUBLE = ("\u201c", "\u201d")

_FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself"})
_FIRST_PLURAL = frozenset({"we", "us", "our", "ours", "ourselves"})
_SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "yourselves"})
_NEGATIONS = frozenset(
    {"not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
     "cannot", "can't", "don't", "doesn't", "didn't", "won't", "isn't",
     "aren't", "wasn't", "weren't", "shouldn't", "couldn't", "wouldn't"}
)
_DEMONSTRATIVES = frozenset({"this", "that", "these", "those"})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Whitespace word count, matching the gate's `word_counts` convention."""
    if not text:
        return 0
    return len(text.split())


def _as_text_list(texts: Sequence[str]) -> List[str]:
    out: List[str] = []
    for t in texts:
        if t is None:
            out.append("")
        elif isinstance(t, str):
            out.append(t)
        else:
            out.append(str(t))
    return out


def _clean(text: str) -> str:
    """Drop only superficial markdown/whitespace wrapping.

    A wrapper is peeled only when the text both starts and ends with the same
    marker and that marker does not appear inside -- so "**bold**" loses its
    asterisks while "**a** and **b**" is left alone instead of being mangled
    into a half-stripped string.

    Deliberately *not* doing: lowercasing, quote folding, dash folding,
    whitespace collapsing.  Those are the signal.
    """
    t = text.strip()
    for _ in range(2):
        peeled = False
        for marker in ("```", "**", "__", "`"):
            if len(t) <= 2 * len(marker):
                continue
            if not (t.startswith(marker) and t.endswith(marker)):
                continue
            if marker in t[len(marker):-len(marker)]:
                continue
            t = t[len(marker):-len(marker)].strip()
            peeled = True
            break
        if not peeled:
            break
    return t


def feature_names() -> Tuple[str, ...]:
    """Ordered feature names, aligned with `extract_short_passage_features`."""
    return FEATURE_NAMES


def feature_index(name: str) -> int:
    """Column index of a named feature. Raises KeyError if unknown."""
    return _FEATURE_INDEX[name]


def _hash_bucket(value: str, n_buckets: int) -> int:
    # zlib.crc32 is stable across processes and platforms; Python's hash() is not.
    return int(zlib.crc32(value.encode("utf-8", "replace")) % n_buckets)


def _char_ngram_profile(text: str) -> np.ndarray:
    """Length-normalised hashed character 1-4 gram profile.

    Computed with a vectorised polynomial hash over the UTF-8 byte array:
    h[i] = sum_k b[i+k] * B^(n-1-k) mod 2**64.  Cost is O(len(text)) per n,
    which keeps 4,000 passages in the low seconds.

    The histogram is divided by its own sum, so the vector is a *relative
    frequency* profile: the same author writing 45 words instead of 200 gets a
    near-identical vector, with only ~1/sqrt(n) sampling noise.  That is the
    property that makes the short model useful at all.
    """
    dim = _N_CHAR_BUCKETS * (_CHAR_NGRAM_MAX - _CHAR_NGRAM_MIN + 1)
    out = np.zeros(dim, dtype=np.float64)
    if not text:
        return out

    data = np.frombuffer(text.encode("utf-8", "replace"), dtype=np.uint8).astype(np.uint64)
    length = data.size
    for slot, n in enumerate(range(_CHAR_NGRAM_MIN, _CHAR_NGRAM_MAX + 1)):
        span = length - n + 1
        if span <= 0:
            continue
        h = data[:span].copy()
        for k in range(1, n):
            h *= _HASH_BASE
            h += data[k:k + span]
        idx = (h % np.uint64(_N_CHAR_BUCKETS)).astype(np.int64)
        idx += slot * _N_CHAR_BUCKETS
        out += np.bincount(idx, minlength=dim)[:dim]

    total = out.sum()
    if total > 0:
        out /= total
    return out


def _affix_profile(text: str) -> np.ndarray:
    """Hashed 10-char / 4-char prefixes and suffixes (96 dims).

    Completion openings are boilerplate more often than not -- "In ", "The ",
    "Certainly", "It's ", "Okay" -- and those habits are length-invariant.
    """
    out = np.zeros(len(_AFFIX_SLOTS) * _N_AFFIX_BUCKETS, dtype=np.float64)
    if not text:
        return out
    values = (text[:10], text[-10:], text[:4], text[-4:])
    for slot, value in enumerate(values):
        bucket = _hash_bucket(value, _N_AFFIX_BUCKETS)
        out[slot * _N_AFFIX_BUCKETS + bucket] += 1.0
    return out


def _word_shape(token: str) -> str:
    if not token:
        return "punct"
    if token.isdigit():
        return "num"
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return "punct"
    if any(c.isdigit() for c in token):
        return "xd"
    upper = [c.isupper() for c in letters]
    if len(letters) == 1:
        return "X" if upper[0] else "x"
    if all(upper):
        return "ALL"
    if not any(upper):
        return "xx"
    if upper[0] and not any(upper[1:]):
        return "Xx" if len(letters) <= 3 else "Xxx"
    if not upper[0]:
        return "xX"
    return "camel"


def _shape_profile(text: str) -> np.ndarray:
    out = np.zeros(len(_SHAPE_CLASSES), dtype=np.float64)
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return out
    pos = {name: i for i, name in enumerate(_SHAPE_CLASSES)}
    for tok in tokens:
        out[pos[_word_shape(tok)]] += 1.0
    out /= out.sum()
    return out


def _contraction_profile(text: str) -> np.ndarray:
    """Contraction panel, apostrophes folded so ' and ’ share a slot."""
    out = np.zeros(len(_CONTRACTIONS), dtype=np.float64)
    if not text:
        return out
    folded = text.replace("\u2019", "'").lower()
    words = max(1, len(_TOKEN_RE.findall(text)))
    for i, c in enumerate(_CONTRACTIONS):
        if c in folded:
            # token boundary check to avoid matching inside longer words
            out[i] = 100.0 * len(re.findall(r"(?<![a-z])" + re.escape(c) + r"(?![a-z])", folded)) / words
    return out


def _numeric_profile(text: str) -> np.ndarray:
    stripped = text.strip()
    tokens = _TOKEN_RE.findall(text)
    words = _WORD_RE.findall(text)
    n_tokens = len(tokens)
    n_words = len(words)
    word_denom = max(1, n_words)
    char_denom = max(1, len(text))

    lower_words = [w.lower() for w in words]

    n_sent = len([s for s in _SENT_SPLIT_RE.split(stripped) if s.strip()]) if stripped else 0
    n_sent = max(1, n_sent)
    sent_lens = [
        len(s.split()) for s in _SENT_SPLIT_RE.split(stripped) if s.strip()
    ] or ([n_tokens] if n_tokens else [0])

    counts: Dict[str, int] = {}
    for w in lower_words:
        counts[w] = counts.get(w, 0) + 1
    n_unique = len(counts)
    n_hapax = sum(1 for v in counts.values() if v == 1)

    # Yule's K on word frequencies; log-scaled because raw K is heavy-tailed.
    freq_of_freq: Dict[int, int] = {}
    for v in counts.values():
        freq_of_freq[v] = freq_of_freq.get(v, 0) + 1
    yule = 0.0
    if n_words > 0:
        yule = 1e4 * (sum(i * i * v for i, v in freq_of_freq.items()) - n_words) / (n_words ** 2)
    yule = max(0.0, yule)

    cap_words = sum(1 for w in words if w[0].isupper() and w[1:].islower())
    all_caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    marked = sum(1 for w in words if "*" in w or "_" in w or "`" in w)

    n_digit_chars = sum(c.isdigit() for c in text)
    n_punct_chars = sum(c in string.punctuation or c in "—–…“”‘’" for c in text)
    n_space_chars = sum(c.isspace() for c in text)

    n_straight = sum(text.count(c) for c in _STRAIGHT_QUOTES)
    n_curly = sum(text.count(c) for c in _CURLY_SINGLE + _CURLY_DOUBLE)

    word_lens = [len(w) for w in words] or [0]
    first_person = sum(1 for w in lower_words if w in _FIRST_PERSON)
    first_plural = sum(1 for w in lower_words if w in _FIRST_PLURAL)
    second_person = sum(1 for w in lower_words if w in _SECOND_PERSON)
    negation = sum(1 for w in lower_words if w in _NEGATIONS)
    demonstrative = sum(1 for w in lower_words if w in _DEMONSTRATIVES)

    def per100(value: int, d: int = word_denom) -> float:
        return 100.0 * value / d

    values = (
        float(np.log1p(n_tokens)),
        float(np.mean(sent_lens)),
        float(np.std(sent_lens)),
        float(np.mean(word_lens)) if n_words else 0.0,
        float(np.std(word_lens)) if n_words else 0.0,
        len(set(text)) / char_denom,
        n_unique / word_denom,
        n_hapax / max(1, n_unique),
        float(np.log1p(yule)),
        cap_words / word_denom,
        all_caps / word_denom,
        sum(1 for w in words if w[:1].isupper()) / word_denom,
        n_digit_chars / char_denom,
        n_punct_chars / char_denom,
        n_space_chars / char_denom,
        marked / word_denom,
        per100(text.count(",")),
        per100(text.count(";")),
        per100(text.count(":")),
        per100(text.count(".")),
        per100(text.count("!")),
        per100(text.count("?")),
        per100(text.count("—")),
        per100(text.count("–")),
        per100(text.count("-")),
        per100(text.count("...") + text.count("…")),
        per100(text.count("(") + text.count(")")),
        per100(text.count("[") + text.count("]")),
        per100(text.count("/")),
        per100(text.count("&")),
        per100(text.count("@")),
        per100(text.count("#")),
        per100(text.count("*")),
        per100(text.count("`")),
        per100(text.count("%")),
        per100(text.count("$")),
        per100(text.count('"')),
        per100(sum(text.count(c) for c in ("'", "\u2018", "\u2019"))),
        per100(sum(text.count(c) for c in _CURLY_DOUBLE)),
        per100(sum(text.count(c) for c in _CURLY_SINGLE)),
        n_straight / max(1, n_straight + n_curly),
        n_curly / max(1, n_straight + n_curly),
        text.count("\n") / max(1, len(tokens)),
        1.0 if "\n\n" in text else 0.0,
        1.0 if _BULLET_RE.search(text) else 0.0,
        1.0 if _NUMBERED_RE.search(text) else 0.0,
        text.count("**") / 2.0,
        1.0 if stripped[:1].isupper() else 0.0,
        1.0 if stripped[:1].islower() else 0.0,
        1.0 if stripped[:1] in ("'", '"', "\u201c", "\u2018") else 0.0,
        1.0 if stripped.endswith(".") else 0.0,
        1.0 if stripped.endswith("?") else 0.0,
        1.0 if stripped.endswith("!") else 0.0,
        1.0 if stripped.endswith(":") else 0.0,
        per100(first_person),
        per100(first_plural),
        per100(second_person),
        per100(negation),
        per100(demonstrative),
    )
    return np.asarray(values, dtype=np.float64)


def _features_for_text(text: str) -> np.ndarray:
    parts = (
        _char_ngram_profile(text),
        _affix_profile(text),
        _shape_profile(text),
        _contraction_profile(text),
        _numeric_profile(text),
    )
    out = np.concatenate(parts).astype(np.float32, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_short_passage_features(texts: List[str]) -> np.ndarray:
    """Dense (n_samples, 591) short-passage feature matrix.

    Row i contains, in order: hashed char 1-4 gram relative frequencies (384),
    hashed prefix/suffix buckets (96), word-shape histogram (12), contraction
    rates per 100 words (40), and normalised structural statistics (59).
    All columns are non-negative and finite; empty strings map to zeros.
    """
    texts = _as_text_list(texts)
    out = np.zeros((len(texts), _N_FEATURES), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i] = _features_for_text(_clean(text))
    return out


# ---------------------------------------------------------------------------
# Truncation augmentation
# ---------------------------------------------------------------------------

def _word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _crop_words(text: str, n_words: int) -> str:
    spans = _word_spans(text)
    if n_words <= 0 or not spans:
        return ""
    if n_words >= len(spans):
        return text
    return text[spans[0][0]:spans[n_words - 1][1]]


def _random_window(text: str, n_words: int, rng: np.random.Generator) -> str:
    spans = _word_spans(text)
    if n_words <= 0 or not spans or n_words >= len(spans):
        return text
    start = int(rng.integers(0, len(spans) - n_words + 1))
    return text[spans[start][0]:spans[start + n_words - 1][1]]


# ---------------------------------------------------------------------------
# Fallback model
# ---------------------------------------------------------------------------

class ShortPassageFallbackModel:
    """High-regularisation linear ensemble specialised on short passages.

    Training corpus
    ---------------
    Only passages that are *already* short (`word_count < short_word_threshold`)
    plus word-level truncations of every longer passage.  The truncations are
    the point: a 220-word sample contributes a 40-, 55- and 70-word view of
    itself, so the classifier sees the distribution of short inputs it will be
    scored on, and the decision boundary is fitted where the semantic model is
    weakest.

    Ensembling
    ----------
    Three deliberately decorrelated linear views:

        word_tfidf   word 1-3 grams, lowercase=False, token pattern keeps
                     punctuation as its own token (so "word," != "word.")
        char_tfidf   char_wb 2-5 grams, lowercase=False
        stylometry   the dense block from `extract_short_passage_features`

    Each member is class-balanced (balanced accuracy is the metric) and strongly
    regularised.  Probabilities are combined as a weighted geometric mean, i.e.
    a log-linear pool, which is the maximum-entropy combination of the members
    when they are individually calibrated.  Pool weights are fitted by
    minimising holdout NLL, so a member that happens to be useless on short
    passages gets weight ~0 instead of a hand-tuned 1/3 (measured: word 0.46-0.80,
    char 0.00-0.32, dense 0.20-0.22 across folds).

    The pool temperature is constrained to T >= 1 (flattening only).  Fitting it
    freely drove T to 0.3, which minimises holdout NLL but makes this model ~3x
    sharper than a typical semantic readout; a sharp member then dominates any
    linear fusion it is later mixed into, and on grouped OOF that cost 0.11
    balanced accuracy in the fused score.  Calibration must be comparable for a
    mixture to be meaningful, so the pool refuses to sharpen.

    Probability floor
    -----------------
    The pooled distribution is mixed with the uniform class prior at
    `prior_strength`, which is the in-model analogue of the gate's entropy
    ceiling: it costs at most `prior_strength` of NLL and prevents a 40-word
    passage from producing p = 1.0 on the wrong sibling.
    """

    _MEMBER_NAMES = ("word_tfidf", "char_tfidf", "stylometry")

    def __init__(
        self,
        class_order: Optional[Sequence[str]] = None,
        short_word_threshold: float = 80.0,
        truncation_lengths: Sequence[int] = (40, 55, 70),
        max_augments_per_row: int = 3,
        random_state: int = 42,
        C_word: float = 1.0,
        C_char: float = 1.0,
        C_dense: float = 0.3,
        prior_strength: float = 0.05,
        calibrate_fraction: float = 0.2,
        min_rows_for_calibration: int = 150,
        max_word_features: int = 120_000,
        max_char_features: int = 150_000,
        tta: bool = False,
        verbose: bool = False,
    ) -> None:
        self.class_order = list(class_order) if class_order is not None else None
        self.short_word_threshold = float(short_word_threshold)
        self.truncation_lengths = tuple(int(v) for v in truncation_lengths)
        self.max_augments_per_row = int(max_augments_per_row)
        self.random_state = int(random_state)
        self.C_word = float(C_word)
        self.C_char = float(C_char)
        self.C_dense = float(C_dense)
        self.prior_strength = float(prior_strength)
        self.calibrate_fraction = float(calibrate_fraction)
        self.min_rows_for_calibration = int(min_rows_for_calibration)
        self.max_word_features = int(max_word_features)
        self.max_char_features = int(max_char_features)
        self.tta = bool(tta)
        self.verbose = bool(verbose)

        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.prior_: Optional[np.ndarray] = None
        self.temperature_: float = 1.0
        self.weights_: Optional[np.ndarray] = None
        self.members_: List[Tuple[str, int, object]] = []
        self.n_augmented_rows_: int = 0
        self.n_short_rows_: int = 0
        self.calibration_: Dict[str, float] = {}
        self.fitted_: bool = False

    # -- internals ---------------------------------------------------------

    def _encode(self, y: Sequence[object]) -> np.ndarray:
        assert self.classes_ is not None
        lookup = {label: i for i, label in enumerate(self.classes_.tolist())}
        out = np.empty(len(y), dtype=np.int64)
        for i, label in enumerate(np.asarray(y).tolist()):
            try:
                out[i] = lookup[label]
            except KeyError as exc:
                raise ValueError(
                    f"label {label!r} is not in the model's class space; "
                    f"pass `class_order` covering every training label"
                ) from exc
        return out

    def _augment(
        self,
        texts: Sequence[str],
        y_idx: np.ndarray,
        word_counts: Sequence[int],
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        out_texts: List[str] = []
        out_y: List[int] = []
        out_wc: List[int] = []
        n_short = 0
        for text, label, wc in zip(texts, y_idx.tolist(), word_counts):
            if wc < self.short_word_threshold:
                out_texts.append(text)
                out_y.append(label)
                out_wc.append(int(wc))
                n_short += 1
            added = 0
            for target in self.truncation_lengths:
                if added >= self.max_augments_per_row:
                    break
                if wc <= target:
                    continue
                crop = _crop_words(text, target)
                if len(crop.split()) < 8:
                    continue
                out_texts.append(crop)
                out_y.append(label)
                out_wc.append(count_words(crop))
                added += 1
                if added >= self.max_augments_per_row:
                    break
                if wc > 2 * target and added < self.max_augments_per_row:
                    window = _random_window(text, target, rng)
                    if count_words(window) >= 8:
                        out_texts.append(window)
                        out_y.append(label)
                        out_wc.append(count_words(window))
                        added += 1
        self.n_short_rows_ = n_short
        return out_texts, np.asarray(out_y, dtype=np.int64), np.asarray(out_wc, dtype=np.int64)

    def _new_member(self, name: str) -> object:
        if name == "word_tfidf":
            c = self.C_word
        elif name == "char_tfidf":
            c = self.C_char
        else:
            c = self.C_dense
        base = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=self.random_state,
            tol=1e-4,
        )
        # liblinear is by far the fastest solver at 100k+ sparse features, but
        # recent scikit-learn dropped its native multiclass mode.  Wrap it in
        # one-vs-rest, which is what liblinear was doing internally anyway and
        # whose predict_proba normalises the per-class scores for us.
        if self.n_classes_ > 2:
            return OneVsRestClassifier(base)
        return base

    def _views(self, texts: Sequence[str]) -> List[object]:
        dense = extract_short_passage_features(list(texts))
        dense = self.scaler_.transform(dense)
        return [self.word_vec_.transform(texts), self.char_vec_.transform(texts), dense]

    def _log_probs(self, models: Sequence[object], views: Sequence[object]) -> List[np.ndarray]:
        """Member probabilities expanded to the full class space, as log-probs."""
        out: List[np.ndarray] = []
        for model, X in zip(models, views):
            p = model.predict_proba(X)
            full = np.full((p.shape[0], self.n_classes_), 1e-6, dtype=np.float64)
            cols = np.asarray(model.classes_).astype(np.int64)
            full[:, cols] = np.maximum(p, 1e-6)
            full /= full.sum(axis=1, keepdims=True)
            out.append(np.log(full))
        return out

    def _fit_pool(self, logps: Sequence[np.ndarray], y: np.ndarray) -> None:
        n_members = len(logps)
        stack = np.stack(logps, axis=0)  # (M, N, K)
        rows = np.arange(len(y))

        def unpack(theta: np.ndarray) -> Tuple[np.ndarray, float]:
            z = theta[:n_members]
            w = np.exp(z - z.max())
            w = w / w.sum()
            # Only flattening is allowed (T >= 1).  Sharpening the pool improves
            # holdout NLL but makes this model 3x more confident than a typical
            # semantic readout, and a sharp member dominates any linear mixture
            # it is later fused with -- measured on prompt-grouped OOF, a
            # T = 0.3 pool dragged fused balanced accuracy from 0.32 to 0.21.
            temperature = float(np.clip(np.exp(theta[n_members]), 1.0, 4.0))
            return w, temperature

        def loss(theta: np.ndarray) -> float:
            w, temperature = unpack(theta)
            logits = np.tensordot(w, stack, axes=(0, 0)) / temperature
            return float(np.mean(logsumexp(logits, axis=1) - logits[rows, y]))

        theta0 = np.zeros(n_members + 1, dtype=np.float64)
        res = minimize(
            loss,
            theta0,
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 800},
        )
        w, temperature = unpack(np.asarray(res.x, dtype=np.float64))
        self.weights_ = w
        self.temperature_ = temperature
        self.calibration_ = {
            "holdout_nll": float(res.fun),
            "temperature": float(temperature),
            "weight_word": float(w[0]),
            "weight_char": float(w[1]),
            "weight_dense": float(w[2]),
        }

    def _pooled_proba(self, logps: Sequence[np.ndarray]) -> np.ndarray:
        assert self.weights_ is not None
        logits = np.tensordot(self.weights_, np.stack(logps, axis=0), axes=(0, 0))
        logits = logits / max(self.temperature_, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)

        # entropy-aware prior floor: a flat row is not evidence
        k = self.n_classes_
        entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0), axis=1)
        confidence = 1.0 - entropy / np.log(max(k, 2))
        floor = self.prior_strength * (1.0 - np.clip(confidence, 0.0, 1.0)) ** 2
        p = (1.0 - floor[:, None]) * p + floor[:, None] * self.prior_[None, :]
        p = np.clip(p, 1e-12, None)
        p /= p.sum(axis=1, keepdims=True)
        return p

    # -- public API --------------------------------------------------------

    def fit(
        self,
        texts_train: Sequence[str],
        y_train: Sequence[str],
        word_counts_train: Optional[Sequence[float]] = None,
    ) -> "ShortPassageFallbackModel":
        texts = _as_text_list(texts_train)
        y = np.asarray(y_train)
        if len(texts) != len(y):
            raise ValueError("texts_train and y_train must have the same length")
        if len(texts) == 0:
            raise ValueError("cannot fit on an empty training set")

        if word_counts_train is None:
            word_counts = np.asarray([count_words(t) for t in texts], dtype=np.float64)
        else:
            word_counts = np.asarray(word_counts_train, dtype=np.float64).ravel()
            if len(word_counts) != len(texts):
                raise ValueError("word_counts_train must match texts_train length")
        if not np.isfinite(word_counts).all():
            raise ValueError("word_counts_train contains non-finite values")

        if self.class_order is not None:
            classes = np.asarray(list(self.class_order), dtype=object)
        else:
            classes = np.asarray(sorted(set(y.tolist())), dtype=object)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        if self.n_classes_ < 2:
            raise ValueError("need at least two classes to fit")
        self.prior_ = np.full(self.n_classes_, 1.0 / self.n_classes_, dtype=np.float64)

        y_idx = self._encode(y.tolist())

        aug_texts, aug_y, aug_wc = self._augment(texts, y_idx, word_counts)
        self.n_augmented_rows_ = len(aug_texts)
        if len(aug_texts) < 2:
            raise ValueError("no training rows survived short-passage augmentation")

        self.word_vec_ = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[A-Za-z0-9_'\u2019\-]+|[^\sA-Za-z0-9_'\u2019\-]",
            ngram_range=(1, 3),
            lowercase=False,
            sublinear_tf=True,
            min_df=1,
            max_features=self.max_word_features,
            dtype=np.float64,
        )
        self.char_vec_ = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=False,
            sublinear_tf=True,
            min_df=1,
            max_features=self.max_char_features,
            dtype=np.float64,
        )
        X_word = self.word_vec_.fit_transform(aug_texts)
        X_char = self.char_vec_.fit_transform(aug_texts)
        X_dense_raw = extract_short_passage_features(aug_texts)
        self.scaler_ = StandardScaler(with_mean=False).fit(X_dense_raw)
        X_dense = self.scaler_.transform(X_dense_raw)
        views = [X_word, X_char, X_dense]

        # -- internal holdout for the pool weights + temperature ------------
        can_calibrate = (
            len(aug_texts) >= self.min_rows_for_calibration
            and self.n_classes_ >= 2
            and np.min(np.bincount(aug_y, minlength=self.n_classes_)) >= 5
            and 0.0 < self.calibrate_fraction < 0.5
        )
        self.weights_ = np.full(len(self._MEMBER_NAMES), 1.0 / len(self._MEMBER_NAMES))
        self.temperature_ = 1.0
        if can_calibrate:
            strat = aug_y if np.min(np.bincount(aug_y, minlength=self.n_classes_)) >= 2 else None
            idx_tr, idx_ho = train_test_split(
                np.arange(len(aug_texts)),
                test_size=self.calibrate_fraction,
                random_state=self.random_state,
                stratify=strat,
            )
            hold_models = []
            for name, X in zip(self._MEMBER_NAMES, views):
                model = self._new_member(name)
                model.fit(X[idx_tr], aug_y[idx_tr])
                hold_models.append(model)
            hold_logps = self._log_probs(hold_models, [X[idx_ho] for X in views])
            self._fit_pool(hold_logps, aug_y[idx_ho])
        else:
            self.calibration_ = {
                "holdout_nll": float("nan"),
                "temperature": 1.0,
                "weight_word": float(self.weights_[0]),
                "weight_char": float(self.weights_[1]),
                "weight_dense": float(self.weights_[2]),
                "skipped": 1.0,
            }

        # -- final members on the full augmented corpus ---------------------
        self.members_ = []
        for name, X in zip(self._MEMBER_NAMES, views):
            model = self._new_member(name)
            model.fit(X, aug_y)
            self.members_.append((name, X.shape[1], model))
        self.fitted_ = True
        if self.verbose:
            print(
                f"[short-passage] {self.n_augmented_rows_} rows "
                f"({self.n_short_rows_} short originals), "
                f"T={self.temperature_:.2f}, weights={np.round(self.weights_, 3)}"
            )
        return self

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("ShortPassageFallbackModel.fit() must be called first")

    def predict_proba(self, texts_test: Sequence[str], tta: Optional[bool] = None) -> np.ndarray:
        """Row-stochastic (n, n_classes) probabilities in `self.classes_` order.

        With `tta=True` (or the constructor flag), probabilities are averaged
        over the full text and two word-level truncations.  Averaging in
        probability space, not logit space, is deliberate: truncations are
        noisy views of the same author and the arithmetic mean is the estimator
        with the lowest variance against that noise.
        """
        self._require_fitted()
        texts = _as_text_list(texts_test)
        use_tta = self.tta if tta is None else bool(tta)
        if len(texts) == 0:
            return np.zeros((0, self.n_classes_), dtype=np.float64)

        if not use_tta:
            views = self._views(texts)
            logps = self._log_probs([m for _, _, m in self.members_], views)
            return self._pooled_proba(logps)

        variants = [texts]
        for target in (60, 40):
            variants.append([_crop_words(t, target) if count_words(t) > target else t for t in texts])
        accum = np.zeros((len(texts), self.n_classes_), dtype=np.float64)
        for variant in variants:
            logps = self._log_probs([m for _, _, m in self.members_], self._views(variant))
            accum += self._pooled_proba(logps)
        accum /= len(variants)
        accum = np.clip(accum, 1e-12, None)
        accum /= accum.sum(axis=1, keepdims=True)
        return accum

    def predict(self, texts_test: Sequence[str]) -> np.ndarray:
        self._require_fitted()
        proba = self.predict_proba(texts_test)
        return self.classes_[np.argmax(proba, axis=1)]

    def report(self) -> Dict[str, float]:
        self._require_fitted()
        out = dict(self.calibration_)
        out.update(
            {
                "n_classes": float(self.n_classes_),
                "n_augmented_rows": float(self.n_augmented_rows_),
                "n_short_rows": float(self.n_short_rows_),
                "prior_strength": float(self.prior_strength),
            }
        )
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.fitted_ else "unfitted"
        return (
            f"ShortPassageFallbackModel({state}, n_classes={self.n_classes_}, "
            f"threshold={self.short_word_threshold:g}w, T={self.temperature_:.2f})"
        )


# ---------------------------------------------------------------------------
# Length gate with entropy bounding
# ---------------------------------------------------------------------------

def short_gate_weight(
    word_counts: np.ndarray,
    L0: float = 65.0,
    tau: float = 10.0,
) -> np.ndarray:
    """w(n) = 1 / (1 + exp((n - L0) / tau)); 1 means "use the short model".

    Smooth, strictly decreasing, w(L0) = 0.5, and d w/dn has magnitude at most
    1/(4 tau) = 0.025 per word at the defaults.
    """
    n = np.asarray(word_counts, dtype=np.float64).ravel()
    if tau <= 0:
        raise ValueError("tau must be > 0")
    return expit(-(n - float(L0)) / float(tau))


def _normalize_rows(p: np.ndarray, name: str) -> np.ndarray:
    if p.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"{name} contains NaN or inf")
    if (p < 0).any():
        raise ValueError(f"{name} contains negative probabilities")
    row_sums = p.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError(f"{name} has a row that sums to zero")
    return p / row_sums


def entropy_regularized_short_gate(
    P_semantic: np.ndarray,
    P_short: np.ndarray,
    word_counts: np.ndarray,
    L0: float = 65.0,
    tau: float = 10.0,
    prior: Optional[np.ndarray] = None,
    h_lo: float = 0.55,
    h_hi: float = 0.95,
    beta_max: float = 0.5,
) -> np.ndarray:
    """Blend the semantic and short-passage posteriors by length, then bound entropy.

    Step 1 -- smooth hand-off
        w = 1 / (1 + exp((n - L0) / tau))
        P = (1 - w) * P_semantic + w * P_short

    Step 2 -- entropy ceiling
        Let H_norm = H(P) / log K in [0, 1].  A row whose posterior is nearly
        flat carries ~no information, and on balanced accuracy a confident
        wrong sibling costs as much as a confident wrong lab.  Rows are pulled
        toward `prior` (uniform by default) with weight

            beta = beta_max * smoothstep((H_norm - h_lo) / (h_hi - h_lo))

        which is 0 below h_lo = 0.55 and beta_max above h_hi = 0.95.  Shrinking
        toward a uniform prior is monotone, so it leaves the argmax untouched
        and only removes confidence the row has not earned.

        beta_max is deliberately < 1.  At beta = 1 the posterior *is* the
        uniform prior, whose argmax is a tie-break on column order -- i.e. every
        genuinely uncertain row silently becomes class 0.  That failure mode was
        observed on prompt-grouped OOF (long-passage balanced accuracy fell from
        0.375 to 0.227) and is the reason the ceiling is capped rather than
        absolute.

    Parameters
    ----------
    P_semantic, P_short : (n, K) arrays of per-class probabilities.
    word_counts : (n,) word counts of the *test* passages.
    L0, tau : gate centre and bandwidth in words.
    prior : (K,) positive, sums to 1; default uniform.
    h_lo, h_hi : entropy ramp, in units of normalised entropy.
    beta_max : maximum prior-mixing weight, must be < 1 to keep the argmax.

    Returns
    -------
    (n, K) float64, rows summing to 1, all entries >= 1e-12.
    """
    Ps = np.asarray(P_semantic, dtype=np.float64)
    Psh = np.asarray(P_short, dtype=np.float64)
    n = np.asarray(word_counts, dtype=np.float64).ravel()

    if Ps.shape != Psh.shape:
        raise ValueError(f"P_semantic {Ps.shape} and P_short {Psh.shape} must match")
    if Ps.ndim != 2:
        raise ValueError("P_semantic must be 2-D (n_samples, n_classes)")
    if n.shape[0] != Ps.shape[0]:
        raise ValueError(
            f"word_counts has length {n.shape[0]} but posteriors have {Ps.shape[0]} rows"
        )
    if not np.isfinite(n).all() or (n < 0).any():
        raise ValueError("word_counts must be finite and non-negative")
    if not (0.0 <= h_lo <= h_hi <= 1.0):
        raise ValueError("require 0 <= h_lo <= h_hi <= 1")
    if not (0.0 <= beta_max < 1.0):
        raise ValueError("beta_max must be in [0, 1)")

    Ps = _normalize_rows(Ps, "P_semantic")
    Psh = _normalize_rows(Psh, "P_short")
    n_classes = Ps.shape[1]
    if n_classes < 2:
        raise ValueError("need at least two classes")

    if prior is None:
        prior_arr = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    else:
        prior_arr = np.asarray(prior, dtype=np.float64).ravel()
        if prior_arr.shape[0] != n_classes:
            raise ValueError("prior must have one entry per class")
        if not np.isfinite(prior_arr).all() or (prior_arr <= 0).any():
            raise ValueError("prior must be finite and strictly positive")
        prior_arr = prior_arr / prior_arr.sum()

    w = short_gate_weight(n, L0=L0, tau=tau)[:, None]
    fused = (1.0 - w) * Ps + w * Psh
    fused = np.clip(fused, 1e-12, None)
    fused /= fused.sum(axis=1, keepdims=True)

    entropy = -np.sum(fused * np.log(fused), axis=1)
    h_norm = entropy / np.log(n_classes)
    t = np.clip((h_norm - h_lo) / max(h_hi - h_lo, 1e-12), 0.0, 1.0)
    beta = beta_max * (t * t * (3.0 - 2.0 * t))  # C1 smoothstep, no cliff at h_lo

    out = (1.0 - beta[:, None]) * fused + beta[:, None] * prior_arr[None, :]
    out = np.clip(out, 1e-12, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Invariant test suite
# ---------------------------------------------------------------------------

def test_short_passage_invariants() -> bool:  # pragma: no cover - executed as script
    """Self-contained checks for features, model and gate. Returns True."""
    rng = np.random.default_rng(42)

    long_text = (
        "The quick brown fox — as the saying goes — jumps over the lazy dog, "
        "again and again; it's relentless. Consider, for a moment, what that "
        "implies about motion itself. "
    ) * 3
    short_a = (
        "In mathematics, a proof is not merely a sequence of symbols; it is an "
        "argument. It is constructed — carefully — so that each step follows."
    )
    short_b = (
        "okay so here's the thing: proofs are basically arguments, right? like, "
        "you can't just write symbols and call it a day!! it's just vibes."
    )
    passed: List[str] = []

    def check(name: str) -> None:
        passed.append(name)
        print(f"  ok  {name}")

    # -- 1. shape, finiteness, determinism --------------------------------
    X = extract_short_passage_features([short_a, short_b, long_text, "", "   "])
    assert X.shape == (5, len(feature_names())), X.shape
    assert X.dtype == np.float32
    assert np.isfinite(X).all(), "features must be finite"
    assert (X >= 0).all(), "features must be non-negative"
    X2 = extract_short_passage_features([short_a, short_b, long_text, "", "   "])
    assert np.array_equal(X, X2), "feature extraction must be deterministic"
    assert np.allclose(X[3], 0.0) and np.allclose(X[4], 0.0), "empty text -> zeros"
    check("features: shape/finite/deterministic/empty-safe")

    # -- 2. char block is a length-invariant relative-frequency profile ---
    char_slice = _CHAR_SLICE
    a = X[0, char_slice]
    assert abs(a.sum() - 1.0) < 1e-6, "char profile must be L1-normalised"
    doubled = extract_short_passage_features([short_a * 2])[0, char_slice]
    cos = float(doubled @ a / (np.linalg.norm(doubled) * np.linalg.norm(a) + 1e-12))
    assert cos > 0.9, f"char profile drifted under duplication (cos={cos:.3f})"
    assert abs(doubled.sum() - 1.0) < 1e-6
    check(f"features: char profile length-invariant (cos={cos:.3f})")

    # -- 3. affix block is sensitive to how a completion opens ------------
    open_a = "Certainly! " + short_a
    open_b = "Okay, " + short_a
    aff_a = extract_short_passage_features([open_a])[0, _AFFIX_SLICE]
    aff_b = extract_short_passage_features([open_b])[0, _AFFIX_SLICE]
    aff_plain = X[0, _AFFIX_SLICE]
    assert not np.array_equal(aff_a, aff_plain)
    assert not np.array_equal(aff_a, aff_b), "affix block must see the opening"
    check("features: prefix/suffix block detects openings")

    # -- 4. word shapes ----------------------------------------------------
    shapes = extract_short_passage_features(["NASA and WHO agree with CamelCase iPhone GPT4 2024"])
    for name in ("shape:ALL", "shape:camel", "shape:xX", "shape:xd", "shape:num"):
        assert shapes[0, feature_index(name)] > 0, name
    check("features: word-shape encodings")

    # -- 5. contractions and clause punctuation ----------------------------
    can_idx = feature_index("contract:can't")
    assert X[0, can_idx] == 0.0, "class A has no contraction"
    assert X[1, can_idx] > 0.0, "class B uses can't"
    em = feature_index("punct:em_dash_per_100w")
    assert X[0, em] > 0.0 and X[1, em] == 0.0
    semi = feature_index("punct:semicolon_per_100w")
    assert X[0, semi] > 0.0
    assert X[0, feature_index("lex:type_token_ratio")] > 0.0
    check("features: contractions + clause-boundary punctuation")

    # -- 6. typography distinguishes straight vs curly --------------------
    straight = extract_short_passage_features(['He said "no" and left.'])[0]
    curly = extract_short_passage_features(["He said \u201cno\u201d and left."])[0]
    assert straight[feature_index("typ:straight_double_per_100w")] > 0
    assert straight[feature_index("typ:curly_double_per_100w")] == 0
    assert curly[feature_index("typ:curly_double_per_100w")] > 0
    assert curly[feature_index("typ:straight_double_per_100w")] == 0
    check("features: straight/curly quote separation")

    # -- 7. gate: shape, smoothness, limits --------------------------------
    n = np.array([0.0, 20.0, 60.0, 65.0, 70.0, 120.0, 250.0])
    w = short_gate_weight(n)
    assert np.all(np.diff(w) <= 1e-12), "gate must be monotonically decreasing"
    assert abs(float(short_gate_weight(np.array([65.0]))[0]) - 0.5) < 1e-9
    assert w[0] > 0.99 and w[-1] < 1e-4
    dw = np.abs(np.diff(w)) / np.diff(n)
    assert dw.max() <= 1.0 / (4.0 * 10.0) + 1e-9, "slope exceeds 1/(4 tau)"
    check("gate: monotone, w(65)=0.5, Lipschitz bound respected")

    # -- 8. gate: semantics for lengths, prior for flat rows --------------
    K = 18
    Ps = np.full((3, K), 0.01)
    Psh = np.full((3, K), 0.01)
    Ps[:, 0] = 0.83
    Ps /= Ps.sum(axis=1, keepdims=True)
    Psh[:, 1] = 0.83
    Psh /= Psh.sum(axis=1, keepdims=True)
    P = entropy_regularized_short_gate(Ps, Psh, np.array([20.0, 65.0, 240.0]))
    assert P.shape == (3, K)
    assert np.allclose(P.sum(axis=1), 1.0), "rows must sum to 1"
    assert P[0].argmax() == 1, "short passage must follow the short model"
    assert P[2].argmax() == 0, "long passage must follow the semantic model"
    assert abs(P[1, 0] - P[1, 1]) < 0.05, "w=0.5 must split the evidence evenly"

    flat_sem = np.full((2, K), 1.0 / K)
    flat_short = np.full((2, K), 1.0 / K)
    P_flat = entropy_regularized_short_gate(flat_sem, flat_short, np.array([30.0, 30.0]))
    assert np.allclose(P_flat, 1.0 / K, atol=1e-6), "flat input -> prior"
    assert P_flat.max() < 0.1
    check("gate: length hand-off + flat rows default to prior")

    # -- 8b. the entropy ceiling must never reorder classes ---------------
    # A near-flat but still ordered row (H_norm ~ 0.99) exercises the ceiling:
    # with beta_max < 1 the argmax must survive, otherwise every uncertain row
    # collapses onto class 0 (observed as a 0.375 -> 0.227 balanced-accuracy
    # drop on long passages before the cap was introduced).
    ordered = np.full((1, K), 0.9 / (K - 1))
    ordered[0, 7] = 0.1
    ordered /= ordered.sum()
    P_ordered = entropy_regularized_short_gate(ordered, ordered, np.array([200.0]))
    assert P_ordered.argmax(axis=1)[0] == 7, "entropy ceiling reordered the classes"
    assert P_ordered[0, 7] < ordered[0, 7], "ceiling failed to shrink confidence"
    h_in = -float((ordered * np.log(ordered)).sum()) / np.log(K)
    assert h_in > 0.95, h_in
    check("gate: entropy ceiling shrinks confidence without reordering")

    # -- 9. gate: adversarial / malformed input ---------------------------
    for bad in (
        lambda: entropy_regularized_short_gate(Ps, Psh[:2], np.array([1.0, 2.0])),
        lambda: entropy_regularized_short_gate(Ps, Psh, np.array([1.0, 2.0])),
        lambda: entropy_regularized_short_gate(-Ps, Psh, np.array([1.0, 2.0, 3.0])),
        lambda: entropy_regularized_short_gate(Ps, Psh, np.array([1.0, 2.0, 3.0]), tau=0.0),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("malformed gate input must raise ValueError")
    zero_rows = np.zeros((2, K))
    zero_rows[:, 5] = 1.0
    P_onehot = entropy_regularized_short_gate(zero_rows, zero_rows, np.array([10.0, 10.0]))
    assert np.allclose(P_onehot.sum(axis=1), 1.0)
    check("gate: rejects malformed input, handles one-hot rows")

    # -- 10. fallback model: learns short-passage style -------------------
    templates = {
        "classA": (
            "Consider the proposition {i}: the structure — as such — entails a "
            "necessary condition; it is, therefore, sufficient. \"Indeed,\" one "
            "might say, \"the argument holds.\""
        ),
        "classB": (
            "so like i was saying about {i}, it's just kinda vibes y'know? "
            "can't really argue with that, it's fine!! honestly, whatever..."
        ),
        "classC": (
            "First, note that {i} follows from the premises (see above); second, "
            "the converse fails. Hence the result — which is standard — obtains."
        ),
    }
    train_texts: List[str] = []
    train_labels: List[str] = []
    for label, tmpl in templates.items():
        for i in range(24):
            train_texts.append(tmpl.format(i=i))
            train_labels.append(label)
    order = ["classA", "classB", "classC", "gpt-5.5", "claude-opus-5"]
    model = ShortPassageFallbackModel(
        class_order=order,
        short_word_threshold=80.0,
        truncation_lengths=(40, 55, 70),
        prior_strength=0.02,
        min_rows_for_calibration=10**9,  # keep the tiny test fast/deterministic
    )
    model.fit(train_texts, train_labels, [count_words(t) for t in train_texts])
    proba = model.predict_proba(train_texts[:9])
    assert proba.shape == (9, len(order)), proba.shape
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    assert np.allclose(proba[:, 3:], 0.0) or proba[:, 3:].max() < 0.05, "unseen classes stay near prior"
    acc = float(np.mean(model.predict(train_texts) == np.asarray(train_labels)))
    assert acc > 0.95, f"fallback model failed to learn the templates (acc={acc:.3f})"
    check(f"model: fits short corpus (train acc={acc:.3f}, {model.report()['n_augmented_rows']:.0f} rows)")

    # -- 11. truncation augmentation actually produces short rows ---------
    long_train = [long_text] * 6
    long_labels = ["classA", "classB", "classC"] * 2
    aug_model = ShortPassageFallbackModel(class_order=order, min_rows_for_calibration=10**9)
    aug_model.fit(long_train, long_labels, [count_words(t) for t in long_train])
    assert aug_model.n_short_rows_ == 0, "long-only corpus has no short originals"
    assert aug_model.n_augmented_rows_ > 6, "truncation augmentation produced no extra rows"
    assert aug_model.n_augmented_rows_ <= 6 * (1 + 3), "augmentation over-produced"
    check("model: truncation augmentation from long passages")

    # -- 12. held-out generalisation + adversarial truncation stability ---
    held = [templates["classB"].format(i=100 + i) for i in range(4)]
    held += [templates["classA"].format(i=200 + i) for i in range(4)]
    held_y = np.array(["classB"] * 4 + ["classA"] * 4)
    held_acc = float(np.mean(model.predict(held) == held_y))
    assert held_acc >= 0.875, f"held-out accuracy too low ({held_acc:.3f})"

    trunc = [t[: max(40, len(t) // 2)] for t in held]
    tta = model.predict_proba(trunc, tta=True)
    assert np.allclose(tta.sum(axis=1), 1.0)
    trunc_acc = float(np.mean(model.predict(trunc) == held_y))
    assert trunc_acc >= 0.75, f"truncated-view accuracy too low ({trunc_acc:.3f})"
    check(f"model: held-out acc={held_acc:.3f}, truncated acc={trunc_acc:.3f}")

    # -- 13. end-to-end fusion keeps argmax for confident long rows -------
    n_rows = 12
    sem = rng.dirichlet(np.ones(K) * 0.3, size=n_rows)
    sh = rng.dirichlet(np.ones(K) * 0.3, size=n_rows)
    wc = np.array([25, 35, 45, 55, 60, 65, 70, 90, 120, 160, 200, 250], dtype=float)
    fused = entropy_regularized_short_gate(sem, sh, wc)
    assert fused.shape == (n_rows, K)
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.all(fused.max(axis=1) <= 1.0)
    long_rows = wc >= 200
    assert np.array_equal(fused[long_rows].argmax(axis=1), sem[long_rows].argmax(axis=1))
    ent = -(fused * np.log(fused)).sum(axis=1) / np.log(K)
    assert ent.max() <= 1.0 + 1e-9
    check("gate: end-to-end fusion preserves confident long-row argmax")

    # -- 14. API contract / failure modes ---------------------------------
    assert len(feature_names()) == X.shape[1]
    try:
        feature_index("no:such:feature")
    except KeyError:
        pass
    else:
        raise AssertionError("feature_index must raise KeyError")

    unfitted = ShortPassageFallbackModel()
    try:
        unfitted.predict_proba(["x"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("predict_proba before fit must raise RuntimeError")

    try:
        ShortPassageFallbackModel().fit(["a b c"] * 4, ["classA"] * 4, [3] * 4)
    except ValueError:
        pass
    else:
        raise AssertionError("single-class fit must raise ValueError")

    try:
        ShortPassageFallbackModel(class_order=["classA", "classB"]).fit(
            train_texts, train_labels, [count_words(t) for t in train_texts]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("labels outside class_order must raise ValueError")

    assert np.array_equal(model.classes_, np.asarray(order, dtype=object))
    assert model.predict(train_texts[:2]).shape == (2,)
    check("api: class-order alignment + failure modes")

    print(f"\nAll {len(passed)} short-passage invariants passed.")
    return True


if __name__ == "__main__":
    test_short_passage_invariants()
