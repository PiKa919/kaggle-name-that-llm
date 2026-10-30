"""
Sibling-level stylometric feature engine for the Kaggle "Name That LLM" contest.
================================================================================

Why this module exists
----------------------
Lab-level attribution (OpenAI / Google / Anthropic / Qwen / ZAI / Moonshot) is
comparatively easy: quote typography, em-dash geometry and markdown habits carry
most of the signal.  Roughly 55-58% of the remaining errors are *sibling*
errors -- the lab is right, the family member is wrong (``glm-5.3`` vs
``glm-5.3-flash``, or five near-identical Claude variants).  This module
isolates the fine-grained authorial fingerprints that separate the ten most
confusable sibling pairs, as ten routed sub-schemas.

Design rules
------------
1. **Never sanitise the text.**  Case, straight vs curly quotes, spaced vs
   unspaced em dashes and digit habits are the signal.  Regexes are case
   sensitive unless the feature is explicitly lexical.  The only destructive
   step is sentence/paragraph segmentation, which masks decimal points and
   common abbreviations so ``3.14`` and ``e.g.`` do not split sentences.
2. **numpy + standard library only.**  One compiled word regex, one compiled
   sentence regex; every per-document statistic is computed once in
   ``_TextStats.__init__`` and cached on a ``__slots__`` object.
3. **Traceable schema.**  Features register through ``@_feature(name, doc)``;
   registration order *is* column order, and every column carries a written
   stylometric rationale in ``FEATURE_DOCS``.
4. **Batch-robust normalisation.**  Raw rates are heavy tailed (Yule's K, digit
   share, em dashes on 40-word passages).  The default is a robust z-score
   ``(x - median) / (IQR / 1.349)`` with a standard-deviation fallback for
   degenerate columns, clipped to ``+/- 8``.  ``method="quantile"`` maps each
   column to ``[-1, 1]`` via average ranks.
5. **Pair routing.**  ``get_pair_feature_indices(pair)`` returns the ordered,
   most-discriminative columns for one sibling pair, so a stacker can train a
   pair-specialised head without re-deriving the schema.

Usage
-----
>>> X = extract_sibling_features(train_texts)                  # (N, 93)
>>> idx = get_pair_feature_indices(("glm-5.3", "glm-5.3-flash"))
>>> X_pair = X[:, idx]
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, NamedTuple, Sequence, Tuple

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "FEATURE_DOCS",
    "N_FEATURES",
    "SIBLING_PAIRS",
    "PAIR_FEATURE_NAMES",
    "extract_sibling_features",
    "extract_raw_sibling_features",
    "get_pair_feature_indices",
    "get_pair_feature_names",
    "feature_names",
    "feature_doc",
    "test_sibling_features_invariants",
]

# --------------------------------------------------------------------------- #
# Compiled patterns and lexical inventories
# --------------------------------------------------------------------------- #

# Words: letters, optionally joined by an internal straight/curly apostrophe.
# Digits are excluded so "3.14" does not pollute lexical richness; digit habits
# are measured by dedicated features instead.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")
_DIGIT_RE = re.compile(r"\d")

_SENT_RE = re.compile(r"[^.!?\u2026]+[.!?\u2026]*")
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_ABBREV_DOT_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Fig|No|cf|al|"
    r"Inc|Ltd|approx|Eq|Ref)\. ?",
    re.IGNORECASE,
)
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_MASK = "\u0001"

# Punctuation geometry.
_EM_SPACED_RE = re.compile(r"(?<=\s)\u2014(?=\s)")
_HYPHENATED_RE = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
_DOUBLE_HYPHEN_RE = re.compile(r"--")
_CONTRACTION_RE = re.compile(r"['\u2019](?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_ELLIPSIS_DOTS_RE = re.compile(r"\.\.\.+")

# Structural / markdown markers (line anchored).
_HEADER_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S", re.MULTILINE)
_LIST_RE = re.compile(r"^[ \t]*(?:[-*+\u2022]|\d{1,2}[.)])[ \t]+\S", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
_CODE_RE = re.compile(r"`[^`\n]+`")

# Morphology.
_LATINATE_RE = re.compile(
    r"(?:tion|sion|ment|ity|ities|ance|ence|ency|ancy|ous|ious|eous|al|ial|"
    r"ive|ative|ate|ify|ise|ize|ism|ist|ary|ory|ical|ic|fy)$"
)
_ABSTRACT_RE = re.compile(r"(?:tion|sion|ment|ity|ness|ism|ance|ence|ship|hood)$")

# Discourse inventories.  These are deliberately small, high-precision lists:
# a wrong marker dilutes a column far more than a missing one.
_REASONING_RE = re.compile(
    r"\b(?:therefore|consequently|thus|hence|specifically|in other words|"
    r"moreover|furthermore|it follows that|this implies|as a result|"
    r"accordingly|which means that|in turn)\b",
    re.IGNORECASE,
)
_GROUNDED_RE = re.compile(
    r"\b(?:in practice|for example|for instance|in fact|the fact that|"
    r"consider|suppose|imagine|take the case|concretely|in reality|here|"
    r"that is|this means|in short|in essence)\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:perhaps|maybe|might|could|seems?|appears?|likely|possibly|"
    r"arguably|somewhat|rather|fairly|in some sense|tends? to|may)\b",
    re.IGNORECASE,
)
_BOOSTER_RE = re.compile(
    r"\b(?:clearly|certainly|indeed|obviously|undoubtedly|always|never|must|"
    r"absolutely|fundamentally|crucially|remarkably|precisely|exactly)\b",
    re.IGNORECASE,
)
_CONTRASTIVE_RE = re.compile(
    r"\b(?:however|but|yet|although|though|nevertheless|nonetheless|whereas|"
    r"while|despite|in contrast|on the other hand|instead|conversely)\b",
    re.IGNORECASE,
)
_ADDITIVE_RE = re.compile(
    r"\b(?:and|also|moreover|furthermore|in addition|additionally|besides|"
    r"similarly|likewise|equally)\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(?:because|since|therefore|thus|hence|consequently|as a result|"
    r"due to|owing to|leads? to|causes?|gives? rise to)\b",
    re.IGNORECASE,
)
_DEFINITIONAL_RE = re.compile(
    r"\b(?:refers to|is defined as|can be defined as|denotes|consists of|"
    r"is a kind of|is a type of|is called|we define)\b",
    re.IGNORECASE,
)
_TECHNICAL_RE = re.compile(
    r"\b(?:algorithms?|entropy|gradients?|theorems?|functions?|parameters?|"
    r"equations?|systems?|energy|particles?|quantum|neurons?|proteins?|cells?|"
    r"models?|data|probability|distributions?|structures?|mechanisms?|"
    r"processes|hypothes[ei]s|experiments?|evidence|theor(?:y|ies)|"
    r"variables?|vectors?|matrices|matrix|tensors?|optimization)\b",
    re.IGNORECASE,
)
_MATH_RE = re.compile(r"[=+\u00d7\u00f7\u2264\u2265\u2260\u2248\u221e\u2211\u222b\u221a^%]")

# Narrative / sensory inventories.
_DIALOGUE_TAG_RE = re.compile(
    r"\b(?:said|asked|replied|whispered|murmured|shouted|answered|cried|"
    r"laughed|sighed|muttered|exclaimed|added|continued|explained|told|"
    r"called|repeated|insisted)\b",
    re.IGNORECASE,
)
_NARRATIVE_RE = re.compile(
    r"\b(?:then|later|before|after|once|suddenly|finally|meanwhile|"
    r"eventually|at last|that night|the next day|years later|for a moment|"
    r"afterwards|soon)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(?:ran|walked|turned|opened|closed|looked|stood|sat|grabbed|pulled|"
    r"pushed|threw|reached|stepped|moved|lifted|dropped|held|watched|"
    r"listened|felt|smiled|nodded|knelt|leaned|glanced)\b",
    re.IGNORECASE,
)
_SENSORY_RE = re.compile(
    r"\b(?:soft|rough|warm|cold|bright|dark|sweet|bitter|sharp|smooth|"
    r"heavy|damp|dry|fragrant|metallic|salty|crisp|faint|loud|quiet|"
    r"silence|scent|smell|taste|texture|touch)\b",
    re.IGNORECASE,
)
_COLOR_RE = re.compile(
    r"\b(?:red|blue|green|yellow|gold|golden|silver|white|black|grey|gray|"
    r"brown|crimson|amber|violet|orange|pink|purple)\b",
    re.IGNORECASE,
)
_SOUND_RE = re.compile(
    r"\b(?:sound|noise|echo|hum|whisper|rustle|creak|thud|silence|"
    r"murmur|clatter|ring|roar|sigh)\w*\b",
    re.IGNORECASE,
)
_TEXTURE_RE = re.compile(
    r"\b(?:smooth|rough|coarse|silky|grain|texture|velvet|jagged|polished|"
    r"sticky|slick|brittle|damp)\w*\b",
    re.IGNORECASE,
)
_LIGHT_RE = re.compile(
    r"\b(?:light|dark|shadow|glow|gleam|shine|bright|dim|flicker|lamp|"
    r"sunlight|moonlight|twilight|dawn|dusk)\w*\b",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(
    r"\b(?:warm|cold|heat|chill|freezing|hot|tepid|icy|frozen|burning)\w*\b",
    re.IGNORECASE,
)

_PUNCT_CHARS = frozenset(
    ".,;:!?\"'()[]{}<>/\\|@#$%^&*_+=~`\u2014\u2013-\u2026"
    "\u201c\u201d\u2018\u2019\u00ab\u00bb"
)
_VOWELS = frozenset("aeiou")

_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with without into onto over under about above below between among
    is are was were be been being am do does did doing have has had having
    i you he she it we they me him her us them my your his its our their
    as not no nor so such very can could may might must shall should will
    would there here where when while who whom whose which what why how
    all any both each few more most other some own same just only also too
    up down out off again further once
    """.split()
)

# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs)) / len(xs) if xs else 0.0


def _pstdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _binary_entropy(p: float) -> float:
    """Entropy in bits of a Bernoulli variable; 0 for degenerate columns."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    count = len(groups)
    if count > 1 and word.lower().endswith("e") and not word.lower().endswith(
        ("le", "ee", "ye")
    ):
        count -= 1
    return max(1, count)


def _is_latinate(word: str) -> bool:
    return len(word) >= 4 and bool(_LATINATE_RE.search(word))


def _split_sentences(block: str) -> List[str]:
    """Split on terminal punctuation without breaking decimals/abbreviations."""
    masked = _DECIMAL_DOT_RE.sub(_SENT_MASK, block)
    masked = _ABBREV_DOT_RE.sub(
        lambda m: m.group(0).replace(".", _SENT_MASK), masked
    )
    out: List[str] = []
    for part in _SENT_RE.findall(masked):
        restored = part.replace(_SENT_MASK, ".").strip()
        if restored and _WORD_RE.search(restored):
            out.append(restored)
    return out


# --------------------------------------------------------------------------- #
# Per-document statistics (computed once)
# --------------------------------------------------------------------------- #


class _TextStats:
    """All scalar summaries of one passage, computed in a single pass."""

    __slots__ = (
        "text", "n_words", "n_chars", "words", "lower", "types", "tokens",
        "n_sent", "sentences", "sent_lens", "n_para", "para_lens", "lines",
        "line_lens",
        "em_total", "em_spaced", "em_unspaced", "en_dash", "dbl_hyphen",
        "hyphenated", "semicolon", "colon", "paren_open", "paren_close",
        "ellipsis", "dq_straight", "dq_curly", "sq_straight", "sq_curly",
        "apos_straight", "apos_curly", "qmark", "excl", "comma", "period",
        "digits", "numbers", "punct", "ttr", "root_ttr", "yule_k", "simpson",
        "hapax_share", "mean_word_len", "std_word_len", "chars_per_word",
        "long_word", "short_word", "poly", "latinate", "abstract", "stop",
        "fps", "fpp", "second", "third", "contractions", "upper_words",
        "caps_words", "hedge", "booster", "reasoning", "grounded",
        "contrastive", "additive", "causal", "definitional", "technical",
        "math", "dialogue_tag", "narrative", "action", "sensory", "color",
        "sound", "texture", "light", "temperature", "headers", "bold", "code",
        "list_items", "list_lines", "sent_short", "sent_long",
        "sent_single_para", "sent_per_para",
    )

    def __init__(self, text: str) -> None:
        self.text = text
        self.n_chars = len(text)

        raw_words = _WORD_RE.findall(text)
        self.words = raw_words
        self.lower = [w.lower() for w in raw_words]
        self.tokens = len(self.lower)
        self.n_words = self.tokens
        counter = Counter(self.lower)
        self.types = len(counter)

        # --- sentence and paragraph geometry ------------------------------- #
        sentences = _split_sentences(text)
        self.sentences = sentences
        self.n_sent = len(sentences)
        self.sent_lens = [len(_WORD_RE.findall(s)) for s in sentences]

        blocks = [b.strip() for b in _PARA_RE.split(text) if b.strip()]
        if not blocks and text.strip():
            blocks = [text.strip()]
        self.n_para = len(blocks)
        self.para_lens = [len(_WORD_RE.findall(b)) for b in blocks]
        self.lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        self.line_lens = [len(_WORD_RE.findall(ln)) for ln in self.lines]

        self.sent_short = sum(1 for n in self.sent_lens if 0 < n <= 12)
        self.sent_long = sum(1 for n in self.sent_lens if n >= 30)
        single = sum(
            1 for b in blocks if len(_split_sentences(b)) == 1
        )
        self.sent_single_para = single / self.n_para if self.n_para else 0.0
        self.sent_per_para = (
            self.n_sent / self.n_para if self.n_para else 0.0
        )

        # --- punctuation counts (never normalised away) -------------------- #
        n = text
        self.em_total = n.count("\u2014")
        self.em_spaced = len(_EM_SPACED_RE.findall(n))
        self.em_unspaced = max(0, self.em_total - self.em_spaced)
        self.en_dash = n.count("\u2013")
        self.dbl_hyphen = len(_DOUBLE_HYPHEN_RE.findall(n))
        self.hyphenated = len(_HYPHENATED_RE.findall(n))
        self.semicolon = n.count(";")
        self.colon = n.count(":")
        self.paren_open = n.count("(")
        self.paren_close = n.count(")")
        self.ellipsis = len(_ELLIPSIS_DOTS_RE.findall(n)) + n.count("\u2026")
        self.dq_straight = n.count('"')
        self.dq_curly = n.count("\u201c") + n.count("\u201d")
        self.sq_straight = n.count("'")
        self.sq_curly = n.count("\u2018") + n.count("\u2019")
        self.apos_straight = n.count("'")
        self.apos_curly = n.count("\u2019")
        self.qmark = n.count("?")
        self.excl = n.count("!")
        self.comma = n.count(",")
        self.period = n.count(".")
        self.digits = len(_DIGIT_RE.findall(n))
        self.numbers = len(_NUMBER_RE.findall(n))
        self.punct = sum(1 for ch in n if ch in _PUNCT_CHARS)
        self.contractions = len(_CONTRACTION_RE.findall(n))
        self.upper_words = sum(1 for w in raw_words if w[:1].isupper())
        self.caps_words = sum(
            1 for w in raw_words if len(w) > 1 and w.isupper()
        )

        # --- lexical richness ---------------------------------------------- #
        self.ttr = self.types / self.tokens if self.tokens else 0.0
        self.root_ttr = (
            self.types / math.sqrt(self.tokens) if self.tokens else 0.0
        )
        if self.tokens:
            sq = sum(f * f for f in counter.values())
            self.yule_k = max(0.0, 1e4 * (sq - self.tokens) / (self.tokens ** 2))
            pairs = sum(f * (f - 1) for f in counter.values())
            denom = self.tokens * (self.tokens - 1)
            self.simpson = pairs / denom if denom else 0.0
            self.hapax_share = (
                sum(1 for f in counter.values() if f == 1) / self.types
            )
        else:
            self.yule_k = 0.0
            self.simpson = 0.0
            self.hapax_share = 0.0

        lengths = [len(w) for w in self.lower]
        self.mean_word_len = _mean(lengths)
        self.std_word_len = _pstdev(lengths)
        self.chars_per_word = (
            sum(len(w) for w in raw_words) / self.tokens if self.tokens else 0.0
        )
        self.long_word = sum(1 for w in self.lower if len(w) >= 9)
        self.short_word = sum(1 for w in self.lower if len(w) <= 3)
        self.poly = sum(1 for w in self.lower if _syllables(w) >= 3)
        self.latinate = sum(1 for w in self.lower if _is_latinate(w))
        self.abstract = sum(
            1 for w in self.lower if len(w) >= 5 and _ABSTRACT_RE.search(w)
        )
        self.stop = sum(1 for w in self.lower if w in _STOPWORDS)

        # --- pronouns / stance --------------------------------------------- #
        first = {"i", "me", "my", "mine", "myself"}
        first_pl = {"we", "us", "our", "ours", "ourselves"}
        second = {"you", "your", "yours", "yourself", "yourselves"}
        third = {"he", "she", "it", "they", "him", "her", "them", "his",
                 "hers", "its", "their", "theirs"}
        self.fps = sum(1 for w in self.lower if w in first)
        self.fpp = sum(1 for w in self.lower if w in first_pl)
        self.second = sum(1 for w in self.lower if w in second)
        self.third = sum(1 for w in self.lower if w in third)

        def count(pattern: "re.Pattern[str]") -> int:
            return len(pattern.findall(text))

        self.hedge = count(_HEDGE_RE)
        self.booster = count(_BOOSTER_RE)
        self.reasoning = count(_REASONING_RE)
        self.grounded = count(_GROUNDED_RE)
        self.contrastive = count(_CONTRASTIVE_RE)
        self.additive = count(_ADDITIVE_RE)
        self.causal = count(_CAUSAL_RE)
        self.definitional = count(_DEFINITIONAL_RE)
        self.technical = count(_TECHNICAL_RE)
        self.math = count(_MATH_RE)
        self.dialogue_tag = count(_DIALOGUE_TAG_RE)
        self.narrative = count(_NARRATIVE_RE)
        self.action = count(_ACTION_RE)
        self.sensory = count(_SENSORY_RE)
        self.color = count(_COLOR_RE)
        self.sound = count(_SOUND_RE)
        self.texture = count(_TEXTURE_RE)
        self.light = count(_LIGHT_RE)
        self.temperature = count(_TEMPERATURE_RE)

        # --- markdown / structure ------------------------------------------ #
        self.headers = len(_HEADER_RE.findall(text))
        self.bold = len(_BOLD_RE.findall(text))
        self.code = len(_CODE_RE.findall(text))
        self.list_items = len(_LIST_RE.findall(text))
        self.list_lines = sum(
            1 for ln in self.lines
            if re.match(r"^(?:[-*+\u2022]|\d{1,2}[.)])[ \t]+\S", ln)
        )


# --------------------------------------------------------------------------- #
# Feature registry
# --------------------------------------------------------------------------- #


class _FeatureSpec(NamedTuple):
    name: str
    doc: str
    fn: Callable[[_TextStats], float]


_SPECS: List[_FeatureSpec] = []


def _feature(name: str, doc: str) -> Callable[[Callable], Callable]:
    """Register one feature column with its stylometric rationale."""

    def decorator(fn: Callable[[_TextStats], float]) -> Callable:
        _SPECS.append(_FeatureSpec(name, doc, fn))
        return fn

    return decorator


def _p100(s: _TextStats, count: float) -> float:
    return 100.0 * count / s.n_words if s.n_words else 0.0


def _safe(value: float) -> float:
    return value if math.isfinite(value) else 0.0


# ------------------------------- typography -------------------------------- #


@_feature(
    "em_dash_total_per_100w",
    "Em dashes per 100 words. GLM (79-82%) and Claude Opus (63-83%) lean on "
    "the em dash far more than competition Gemini, so the rate is a lab prior "
    "and, inside a lab, a sibling discriminator.",
)
def _em_dash_total(s: _TextStats) -> float:
    return _p100(s, s.em_total)


@_feature(
    "em_dash_spaced_per_100w",
    "Spaced em dashes (' \u2014 ') per 100 words. Spacing style is a stronger "
    "sibling cue than raw frequency: Opus 4.6 and Opus 5 differ mainly in how "
    "habitually they isolate the dash with spaces.",
)
def _em_dash_spaced(s: _TextStats) -> float:
    return _p100(s, s.em_spaced)


@_feature(
    "em_dash_spaced_share",
    "Share of em dashes that are space-delimited. Captures the typographic "
    "habit independently of passage length, which matters because short "
    "passages rarely contain more than one dash.",
)
def _em_dash_spaced_share(s: _TextStats) -> float:
    return s.em_spaced / s.em_total if s.em_total else 0.0


@_feature(
    "em_dash_unspaced_per_100w",
    "Unspaced em dashes per 100 words. The complementary style; some siblings "
    "prefer the tight 'word\u2014word' form even at the same total rate.",
)
def _em_dash_unspaced(s: _TextStats) -> float:
    return _p100(s, s.em_unspaced)


@_feature(
    "em_dash_style_entropy",
    "Binary entropy (bits) of the spaced/unspaced dash mixture. A sibling that "
    "mixes both styles is typographically inconsistent; a single-style writer "
    "scores 0, which separates disciplined from exploratory variants.",
)
def _em_dash_entropy(s: _TextStats) -> float:
    return _binary_entropy(_em_dash_spaced_share(s))


@_feature(
    "em_dash_per_sentence",
    "Em dashes per sentence. Normalising by sentences rather than words "
    "exposes clause-joining habits in long-sentence writers such as Opus 5.",
)
def _em_dash_per_sentence(s: _TextStats) -> float:
    return s.em_total / s.n_sent if s.n_sent else 0.0


@_feature(
    "en_dash_per_100w",
    "En dashes per 100 words. En dashes appear in numeric ranges and some "
    "house styles; their presence is a low-volume but clean typographic tell.",
)
def _en_dash(s: _TextStats) -> float:
    return _p100(s, s.en_dash)


@_feature(
    "double_hyphen_per_100w",
    "Double hyphens ('--') per 100 words. A markdown-era substitute for the em "
    "dash; common in some model families and almost absent in others.",
)
def _double_hyphen(s: _TextStats) -> float:
    return _p100(s, s.dbl_hyphen)


@_feature(
    "hyphenated_word_per_100w",
    "Hyphenated compounds per 100 words. Compound density reflects vocabulary "
    "formation habits and rises in technical exposition.",
)
def _hyphenated(s: _TextStats) -> float:
    return _p100(s, s.hyphenated)


@_feature(
    "semicolon_per_100w",
    "Semicolons per 100 words. The semicolon is a deliberate, formal clause "
    "joiner; Opus 4.6 uses it more than Opus 5, and flash variants suppress it.",
)
def _semicolon(s: _TextStats) -> float:
    return _p100(s, s.semicolon)


@_feature(
    "colon_per_100w",
    "Colons per 100 words. Colons introduce elaboration and lists, a marker of "
    "structured expository planning rather than conversational flow.",
)
def _colon(s: _TextStats) -> float:
    return _p100(s, s.colon)


@_feature(
    "paren_open_per_100w",
    "Opening parentheses per 100 words. Parenthetical asides signal a "
    "qualifying, self-interrupting voice; density differs between Opus 4.6 "
    "and the more linear Opus 5.",
)
def _paren_open(s: _TextStats) -> float:
    return _p100(s, s.paren_open)


@_feature(
    "paren_pair_per_sentence",
    "Balanced parenthesis pairs per sentence. Unbalanced or absent parentheses "
    "separate writers who annotate their prose from those who do not.",
)
def _paren_pair(s: _TextStats) -> float:
    pairs = min(s.paren_open, s.paren_close)
    return pairs / s.n_sent if s.n_sent else 0.0


@_feature(
    "ellipsis_per_100w",
    "Ellipses per 100 words. The ellipsis is a narrative/trailing-off device, "
    "rare in formal reasoning and common in creative registers.",
)
def _ellipsis(s: _TextStats) -> float:
    return _p100(s, s.ellipsis)


@_feature(
    "curly_dq_share",
    "Share of double quotes that are typographic curly quotes. The single "
    "strongest lab separator: OpenAI and Gemini emit curly quotes, Claude and "
    "GLM emit straight ones. Gemini 3.8 flash shifts sharply toward curly.",
)
def _curly_dq_share(s: _TextStats) -> float:
    total = s.dq_curly + s.dq_straight
    return s.dq_curly / total if total else 0.0


@_feature(
    "straight_dq_share",
    "Share of double quotes that are plain ASCII. Complements the curly share "
    "and keeps the column informative for batches with zero quotes.",
)
def _straight_dq_share(s: _TextStats) -> float:
    total = s.dq_curly + s.dq_straight
    return s.dq_straight / total if total else 0.0


@_feature(
    "curly_sq_share",
    "Share of single quotes that are curly. Tracks the same typographic engine "
    "as double quotes but is noisier, so it is a secondary confirmation signal.",
)
def _curly_sq_share(s: _TextStats) -> float:
    total = s.sq_curly + s.sq_straight
    return s.sq_curly / total if total else 0.0


@_feature(
    "curly_apostrophe_share",
    "Share of apostrophes that are curly. Apostrophes occur in nearly every "
    "passage, making this a denser sample of the quote engine than quote marks "
    "and useful for GPT 5.5 vs 6-astra where quote counts are low.",
)
def _curly_apostrophe_share(s: _TextStats) -> float:
    total = s.apos_curly + s.apos_straight
    return s.apos_curly / total if total else 0.0


@_feature(
    "quote_mark_per_100w",
    "Double-quote marks per 100 words. Quote density measures how much of the "
    "completion is dramatised speech; 3.8 flash quotes more freely than 3.7, "
    "independently of which glyph it chooses.",
)
def _quote_mark_per_100w(s: _TextStats) -> float:
    return _p100(s, s.dq_curly + s.dq_straight)


@_feature(
    "question_mark_per_sentence",
    "Question marks per sentence. Interrogative framing is a discourse habit "
    "that separates Gemini 3.8 flash (question-heavy) from 3.7.",
)
def _question_per_sentence(s: _TextStats) -> float:
    return s.qmark / s.n_sent if s.n_sent else 0.0


@_feature(
    "question_sentence_share",
    "Share of sentences ending in a question mark. Normalises by sentence "
    "count and is robust to passages that quote several questions at once.",
)
def _question_sentence_share(s: _TextStats) -> float:
    if not s.n_sent:
        return 0.0
    ends = sum(1 for sent in s.sentences if sent.rstrip().endswith("?"))
    return ends / s.n_sent


@_feature(
    "exclamation_per_100w",
    "Exclamation marks per 100 words. Marks an emphatic, conversational voice "
    "that formal reasoning variants suppress almost entirely.",
)
def _exclamation(s: _TextStats) -> float:
    return _p100(s, s.excl)


@_feature(
    "comma_per_100w",
    "Commas per 100 words. Comma density proxies clause complexity and the "
    "willingness to embed subordinate material; it drifts between siblings of "
    "the same family.",
)
def _comma(s: _TextStats) -> float:
    return _p100(s, s.comma)


@_feature(
    "digit_char_share",
    "Digits as a share of all characters. GLM 5.3 flash is measurably more "
    "numeric than full 5.3; digits also reveal enumeration and date habits.",
)
def _digit_char_share(s: _TextStats) -> float:
    return s.digits / s.n_chars if s.n_chars else 0.0


@_feature(
    "number_token_per_100w",
    "Standalone numeric tokens per 100 words. Distinguishes genuine quantitative "
    "content from incidental digits embedded in identifiers.",
)
def _number_token(s: _TextStats) -> float:
    return _p100(s, s.numbers)


@_feature(
    "punct_per_100w",
    "Punctuation characters per 100 words. A global measure of how heavily a "
    "passage is punctuated; flash models compress this along with sentence "
    "length.",
)
def _punct_per_100w(s: _TextStats) -> float:
    return _p100(s, s.punct)


@_feature(
    "uppercase_word_share",
    "Share of words beginning with a capital letter. Capitalisation habits "
    "(title case, sentence-case discipline, acronyms) survive tokenisation and "
    "are destroyed by lowercasing, so they are preserved here deliberately.",
)
def _uppercase_word_share(s: _TextStats) -> float:
    return s.upper_words / s.n_words if s.n_words else 0.0


@_feature(
    "all_caps_word_share",
    "Share of multi-letter all-caps words. Measures acronym and emphasis "
    "density, a small but stable technical-register marker.",
)
def _all_caps_word_share(s: _TextStats) -> float:
    return s.caps_words / s.n_words if s.n_words else 0.0


# --------------------------- structure and rhythm -------------------------- #


@_feature(
    "log_word_count",
    "Natural log of the word count. Length is a nuisance variable for most "
    "features but a genuine signal for families whose flash variants answer "
    "more briefly; the log keeps the scale well behaved.",
)
def _log_word_count(s: _TextStats) -> float:
    return math.log1p(s.n_words)


@_feature(
    "mean_sentence_words",
    "Mean words per sentence. The core rhythm statistic and the primary "
    "compression axis for GLM 5.3 vs 5.3-flash and Gemini 3.6 vs 3.7.",
)
def _mean_sentence_words(s: _TextStats) -> float:
    return _mean(s.sent_lens)


@_feature(
    "std_sentence_words",
    "Standard deviation of sentence length. Captures rhythmic variety: Opus 4.8 "
    "alternates long and short sentences more than the uniform Sonnet 5.",
)
def _std_sentence_words(s: _TextStats) -> float:
    return _pstdev(s.sent_lens)


@_feature(
    "cv_sentence_words",
    "Coefficient of variation of sentence length (std/mean). A length-"
    "normalised burstiness measure, safer than raw std when passages differ in "
    "mean sentence length.",
)
def _cv_sentence_words(s: _TextStats) -> float:
    mu = _mean(s.sent_lens)
    return (_pstdev(s.sent_lens) / mu) if mu else 0.0


@_feature(
    "median_sentence_words",
    "Median words per sentence. Robust to a single very long sentence, which "
    "makes it a cleaner compression signal for the GLM siblings.",
)
def _median_sentence_words(s: _TextStats) -> float:
    return _median(s.sent_lens)


@_feature(
    "short_sentence_share",
    "Share of sentences with at most 12 words. Clipped sentences are the "
    "signature of narrative pacing and of flash-model concision.",
)
def _short_sentence_share(s: _TextStats) -> float:
    return s.sent_short / s.n_sent if s.n_sent else 0.0


@_feature(
    "long_sentence_share",
    "Share of sentences with at least 30 words. Long periodic sentences mark "
    "expository depth and are depleted in flash variants.",
)
def _long_sentence_share(s: _TextStats) -> float:
    return s.sent_long / s.n_sent if s.n_sent else 0.0


@_feature(
    "sentence_count_log",
    "Log sentence count. Controls for passage length when comparing rhythm "
    "features and flags fragment-heavy writers.",
)
def _sentence_count_log(s: _TextStats) -> float:
    return math.log1p(s.n_sent)


@_feature(
    "paragraph_count_log",
    "Log paragraph count. Paragraphing is a planning habit: Pro models often "
    "produce multi-paragraph answers where flash models emit one block.",
)
def _paragraph_count_log(s: _TextStats) -> float:
    return math.log1p(s.n_para)


@_feature(
    "mean_paragraph_words",
    "Mean words per paragraph. Measures structural compactness; Gemini 3.6 "
    "flash writes shorter blocks than 3.7.",
)
def _mean_paragraph_words(s: _TextStats) -> float:
    return _mean(s.para_lens)


@_feature(
    "std_paragraph_words",
    "Standard deviation of paragraph length. Even block sizes indicate "
    "template-driven structure; uneven sizes indicate organic prose.",
)
def _std_paragraph_words(s: _TextStats) -> float:
    return _pstdev(s.para_lens)


@_feature(
    "single_sentence_paragraph_share",
    "Share of paragraphs containing exactly one sentence. A strong layout tell: "
    "list-like or punchy writers produce many one-sentence blocks.",
)
def _single_sentence_paragraph_share(s: _TextStats) -> float:
    return s.sent_single_para


@_feature(
    "sentences_per_paragraph",
    "Mean sentences per paragraph. Complements paragraph length with a purely "
    "structural ratio independent of word counts.",
)
def _sentences_per_paragraph(s: _TextStats) -> float:
    return s.sent_per_para


@_feature(
    "list_item_per_100w",
    "Markdown list items per 100 words. List usage is a planning fingerprint "
    "that separates Gemini 3.6 from 3.7 and Pro from flash.",
)
def _list_item_per_100w(s: _TextStats) -> float:
    return _p100(s, s.list_items)


@_feature(
    "list_line_share",
    "Share of non-empty lines that are list items. Normalises by layout size "
    "and is robust when passages mix prose and bullets.",
)
def _list_line_share(s: _TextStats) -> float:
    return s.list_lines / len(s.lines) if s.lines else 0.0


@_feature(
    "markdown_header_per_100w",
    "Markdown headers per 100 words. Header usage signals an outline-first "
    "generation style characteristic of Pro/preview models.",
)
def _header_per_100w(s: _TextStats) -> float:
    return _p100(s, s.headers)


@_feature(
    "markdown_header_present",
    "Binary indicator of any markdown header. Presence/absence is often more "
    "reliable than a rate when passages are short.",
)
def _header_present(s: _TextStats) -> float:
    return 1.0 if s.headers else 0.0


@_feature(
    "bold_span_per_100w",
    "Bold spans per 100 words. Emphasis markup is a formatting habit that "
    "varies sharply between chat-tuned siblings.",
)
def _bold_per_100w(s: _TextStats) -> float:
    return _p100(s, s.bold)


@_feature(
    "code_span_per_100w",
    "Inline code spans per 100 words. Technical answers that quote identifiers "
    "use backticks; prose-only siblings do not.",
)
def _code_per_100w(s: _TextStats) -> float:
    return _p100(s, s.code)


@_feature(
    "mean_line_words",
    "Mean words per non-empty line. A layout measure: hard-wrapped, list-like "
    "output has short lines, flowing prose has long ones.",
)
def _mean_line_words(s: _TextStats) -> float:
    return _mean(s.line_lens)


@_feature(
    "short_line_share",
    "Share of lines with at most 6 words. Detects bulleted or stanza-like "
    "layout without relying on explicit list markers.",
)
def _short_line_share(s: _TextStats) -> float:
    if not s.line_lens:
        return 0.0
    return sum(1 for n in s.line_lens if 0 < n <= 6) / len(s.line_lens)


# ----------------------------- lexical richness ---------------------------- #


@_feature(
    "ttr",
    "Type-token ratio. The classic lexical-diversity measure requested for the "
    "Opus 4.8 vs Sonnet 5 pair; higher values indicate more varied vocabulary.",
)
def _ttr(s: _TextStats) -> float:
    return s.ttr


@_feature(
    "root_ttr",
    "Guiraud's root TTR = types / sqrt(tokens). Length-robust diversity index, "
    "preferable to raw TTR when passages range from 40 to 250 words.",
)
def _root_ttr(s: _TextStats) -> float:
    return s.root_ttr


@_feature(
    "yule_k_log",
    "log1p of Yule's K. K measures repetition concentration and is one of the "
    "standard authorship-attribution statistics; the log tames its heavy tail.",
)
def _yule_k_log(s: _TextStats) -> float:
    return math.log1p(s.yule_k)


@_feature(
    "simpson_d",
    "Simpson's index of word repetition: the probability that two random tokens "
    "share a type. A bounded complement to Yule's K.",
)
def _simpson_d(s: _TextStats) -> float:
    return s.simpson


@_feature(
    "hapax_share",
    "Share of vocabulary types occurring exactly once. Hapax density is "
    "sensitive to one-off Latinate coinages, a marker of ornate Claude prose.",
)
def _hapax_share(s: _TextStats) -> float:
    return s.hapax_share


@_feature(
    "mean_word_length",
    "Mean letters per word. A core readability and formality proxy; GLM flash "
    "compresses word length relative to full 5.3.",
)
def _mean_word_length(s: _TextStats) -> float:
    return s.mean_word_len


@_feature(
    "std_word_length",
    "Standard deviation of word length. Wide distributions mix short function "
    "words with long technical terms, typical of expository Pro prose.",
)
def _std_word_length(s: _TextStats) -> float:
    return s.std_word_len


@_feature(
    "chars_per_word",
    "Non-space characters per word, including punctuation. The compression "
    "measure requested for the GLM pair: it blends word length with punctuation "
    "density, both of which flash variants reduce.",
)
def _chars_per_word(s: _TextStats) -> float:
    return s.chars_per_word


@_feature(
    "long_word_share",
    "Share of words with at least 9 letters. The simplest surface measure of a "
    "Latinate, academic register.",
)
def _long_word_share(s: _TextStats) -> float:
    return s.long_word / s.n_words if s.n_words else 0.0


@_feature(
    "short_word_share",
    "Share of words with at most 3 letters. The conversational tail; narrative "
    "and chatty registers use more short words.",
)
def _short_word_share(s: _TextStats) -> float:
    return s.short_word / s.n_words if s.n_words else 0.0


@_feature(
    "polysyllable_share",
    "Share of words with three or more estimated syllables. A readability "
    "dimension that tracks technical exposition depth.",
)
def _polysyllable_share(s: _TextStats) -> float:
    return s.poly / s.n_words if s.n_words else 0.0


@_feature(
    "latinate_share",
    "Share of words carrying a Latinate/Greco-Latinate suffix. The requested "
    "register discriminator for Opus 4.8 vs Sonnet 5 and a formality proxy for "
    "the GPT siblings.",
)
def _latinate_share(s: _TextStats) -> float:
    return s.latinate / s.n_words if s.n_words else 0.0


@_feature(
    "abstract_noun_share",
    "Share of abstract-noun morphology (-tion, -ity, -ness, -ism, -ment). "
    "Nominalisation density separates philosophical abstraction from concrete "
    "narrative prose.",
)
def _abstract_noun_share(s: _TextStats) -> float:
    return s.abstract / s.n_words if s.n_words else 0.0


@_feature(
    "stopword_share",
    "Share of tokens that are function words. High rates indicate grammatical, "
    "connective-heavy prose; low rates indicate dense content-word packing.",
)
def _stopword_share(s: _TextStats) -> float:
    return s.stop / s.n_words if s.n_words else 0.0


@_feature(
    "contraction_per_100w",
    "Contractions per 100 words. The strongest single informality cue, "
    "suppressed in formal reasoning variants and elevated in chatty ones.",
)
def _contraction(s: _TextStats) -> float:
    return _p100(s, s.contractions)


# ------------------------------ persona / tone ----------------------------- #


@_feature(
    "first_person_singular_per_100w",
    "First-person singular pronouns per 100 words. Authorial 'I' is far more "
    "frequent in reflective/narrative Claude variants than in impersonal "
    "technical siblings.",
)
def _fps(s: _TextStats) -> float:
    return _p100(s, s.fps)


@_feature(
    "first_person_plural_per_100w",
    "First-person plural pronouns per 100 words. 'We' frames a shared inquiry "
    "and is a didactic register marker used unevenly across labs.",
)
def _fpp(s: _TextStats) -> float:
    return _p100(s, s.fpp)


@_feature(
    "second_person_per_100w",
    "Second-person pronouns per 100 words. Direct address is a conversational "
    "habit that separates chatty flash models from formal Pros.",
)
def _second_person(s: _TextStats) -> float:
    return _p100(s, s.second)


@_feature(
    "third_person_per_100w",
    "Third-person pronouns per 100 words. Pronoun-heavy prose signals narrative "
    "continuity rather than abstract exposition.",
)
def _third_person(s: _TextStats) -> float:
    return _p100(s, s.third)


@_feature(
    "hedge_per_100w",
    "Hedges ('perhaps', 'might', 'seems', 'arguably') per 100 words. Hedging "
    "marks epistemic caution and is unevenly distributed across sibling "
    "temperaments.",
)
def _hedge(s: _TextStats) -> float:
    return _p100(s, s.hedge)


@_feature(
    "booster_per_100w",
    "Boosters ('clearly', 'indeed', 'must', 'precisely') per 100 words. "
    "Boosting marks assertive stance, the opposite pole from hedging.",
)
def _booster(s: _TextStats) -> float:
    return _p100(s, s.booster)


@_feature(
    "reasoning_marker_per_100w",
    "Explicit reasoning connectives ('therefore', 'consequently', "
    "'specifically', 'in other words') per 100 words. The requested "
    "discourse-marker axis for GPT 5.6 sol vs terra.",
)
def _reasoning_marker(s: _TextStats) -> float:
    return _p100(s, s.reasoning)


@_feature(
    "grounded_marker_per_100w",
    "Grounded/illustrative markers ('in practice', 'for example', 'consider') "
    "per 100 words. Terra prefers concrete instantiation over abstract "
    "derivation, which this column captures.",
)
def _grounded_marker(s: _TextStats) -> float:
    return _p100(s, s.grounded)


@_feature(
    "contrastive_connective_per_100w",
    "Contrastive connectives ('however', 'although', 'whereas') per 100 words. "
    "Measures argumentative counterpointing rather than linear addition.",
)
def _contrastive(s: _TextStats) -> float:
    return _p100(s, s.contrastive)


@_feature(
    "additive_connective_per_100w",
    "Additive connectives ('also', 'moreover', 'in addition') per 100 words. "
    "Tracks list-like accumulation of points, common in structured answers.",
)
def _additive(s: _TextStats) -> float:
    return _p100(s, s.additive)


@_feature(
    "causal_connective_per_100w",
    "Causal connectives ('because', 'thus', 'leads to') per 100 words. Captures "
    "explanatory chaining, the backbone of reasoning-style completions.",
)
def _causal(s: _TextStats) -> float:
    return _p100(s, s.causal)


@_feature(
    "connective_per_100w",
    "All discourse connectives per 100 words (contrastive + additive + causal). "
    "A compactness-versus-cohesion axis for the Gemini flash siblings.",
)
def _connective(s: _TextStats) -> float:
    return _p100(s, s.contrastive + s.additive + s.causal)


@_feature(
    "definitional_per_100w",
    "Definitional constructions ('refers to', 'is defined as', 'consists of') "
    "per 100 words. Textbook-definition framing is a strong Pro/preview habit.",
)
def _definitional(s: _TextStats) -> float:
    return _p100(s, s.definitional)


@_feature(
    "technical_term_per_100w",
    "Domain-technical terms per 100 words (algorithm, entropy, gradient, "
    "theorem, protein, ...). Measures technical exposition depth independently "
    "of general vocabulary richness.",
)
def _technical(s: _TextStats) -> float:
    return _p100(s, s.technical)


@_feature(
    "math_symbol_per_100w",
    "Mathematical operator symbols per 100 words. Formal notation is a narrow "
    "but high-precision marker of quantitative reasoning answers.",
)
def _math_symbol(s: _TextStats) -> float:
    return _p100(s, s.math)


# -------------------------- narrative / sensory ---------------------------- #


@_feature(
    "dialogue_tag_per_100w",
    "Speech verbs ('said', 'whispered', 'replied') per 100 words. Dialogue tags "
    "are the clearest single signal of fictional narration in the Fable 5 vs "
    "Opus 4.8 pair.",
)
def _dialogue_tag(s: _TextStats) -> float:
    return _p100(s, s.dialogue_tag)


@_feature(
    "dialogue_sentence_share",
    "Share of sentences containing a quotation mark. Measures how much of the "
    "passage is dramatised speech rather than exposition.",
)
def _dialogue_sentence_share(s: _TextStats) -> float:
    if not s.n_sent:
        return 0.0
    quoted = sum(
        1 for sent in s.sentences
        if any(ch in sent for ch in "\"\u201c\u201d\u2018\u2019'")
    )
    return quoted / s.n_sent


@_feature(
    "narrative_marker_per_100w",
    "Temporal/narrative markers ('then', 'later', 'suddenly', 'meanwhile') per "
    "100 words. Explicit time sequencing is the skeleton of story-shaped prose.",
)
def _narrative_marker(s: _TextStats) -> float:
    return _p100(s, s.narrative)


@_feature(
    "action_verb_per_100w",
    "Concrete action verbs ('walked', 'opened', 'turned') per 100 words. "
    "Physical agency distinguishes narrated scenes from abstract reflection.",
)
def _action_verb(s: _TextStats) -> float:
    return _p100(s, s.action)


@_feature(
    "sensory_descriptor_per_100w",
    "Sensory adjectives and nouns (soft, bitter, scent, silence) per 100 words. "
    "Sensory density is a hallmark of the creative Claude variant.",
)
def _sensory(s: _TextStats) -> float:
    return _p100(s, s.sensory)


@_feature(
    "color_term_per_100w",
    "Colour terms per 100 words. Deliberate colour naming is a narrow, "
    "high-precision imagery habit in narrative completions.",
)
def _color_term(s: _TextStats) -> float:
    return _p100(s, s.color)


@_feature(
    "sound_term_per_100w",
    "Sound terms per 100 words. Auditory imagery tracks scene-setting prose.",
)
def _sound_term(s: _TextStats) -> float:
    return _p100(s, s.sound)


@_feature(
    "texture_term_per_100w",
    "Tactile/texture terms per 100 words. Touch imagery is rarer than visual "
    "imagery and therefore carries high discriminative value when present.",
)
def _texture_term(s: _TextStats) -> float:
    return _p100(s, s.texture)


@_feature(
    "light_term_per_100w",
    "Light/darkness terms per 100 words. Lighting imagery is a recurring "
    "stylistic tic in creative completions.",
)
def _light_term(s: _TextStats) -> float:
    return _p100(s, s.light)


@_feature(
    "temperature_term_per_100w",
    "Temperature terms per 100 words. Thermal imagery is a narrow but precise "
    "sensory signal.",
)
def _temperature_term(s: _TextStats) -> float:
    return _p100(s, s.temperature)


@_feature(
    "narrative_pacing_index",
    "Short-sentence share amplified by narrative-marker rate. Pacing is not one "
    "statistic but the interaction of clipped sentences and explicit time "
    "sequencing; this composite captures it in a single column.",
)
def _narrative_pacing(s: _TextStats) -> float:
    return s.sent_short / s.n_sent * (
        1.0 + _p100(s, s.narrative) / 5.0
    ) if s.n_sent else 0.0


# --------------------------- token-boundary proxies ------------------------ #


@_feature(
    "clitic_share",
    "Share of words carrying a clitic apostrophe ('s, n't, 're, 'll). Clitic "
    "splitting is a tokenizer-visible habit: some families write 'it is' where "
    "others write 'it's', which changes the subword stream.",
)
def _clitic_share(s: _TextStats) -> float:
    return s.contractions / s.n_words if s.n_words else 0.0


@_feature(
    "spaces_per_word",
    "Spaces per word. A crude token-boundary fingerprint: spacing discipline "
    "around punctuation and dashes shifts this ratio between siblings.",
)
def _spaces_per_word(s: _TextStats) -> float:
    return s.text.count(" ") / s.n_words if s.n_words else 0.0


@_feature(
    "chars_per_sentence",
    "Mean characters per sentence. Blends words per sentence with word length "
    "and thus approximates the tokenizer's view of segment size.",
)
def _chars_per_sentence(s: _TextStats) -> float:
    return s.n_chars / s.n_sent if s.n_sent else 0.0


# --------------------------------------------------------------------------- #
# Public schema
# --------------------------------------------------------------------------- #

FEATURE_NAMES: Tuple[str, ...] = tuple(spec.name for spec in _SPECS)
FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
FEATURE_DOCS: Dict[str, str] = {spec.name: spec.doc for spec in _SPECS}
N_FEATURES: int = len(FEATURE_NAMES)
_SPEC_FNS: Tuple[Callable[[_TextStats], float], ...] = tuple(
    spec.fn for spec in _SPECS
)

if len(FEATURE_INDEX) != N_FEATURES:  # pragma: no cover - import-time guard
    raise RuntimeError("sibling feature names must be unique")

SIBLING_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("claude-opus-4-6", "claude-opus-5"),
    ("claude-opus-4-8", "claude-sonnet-5"),
    ("claude-fable-5", "claude-opus-4-8"),
    ("gemini-3.6-flash", "gemini-3.7-flash"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gemini-3.1-pro-preview", "gemini-3.7-flash"),
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("gpt-5.5", "gpt-6-astra"),
    ("glm-5.3", "glm-5.3-flash"),
)


def _normalize_pair(pair: Tuple[str, str]) -> Tuple[str, str]:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise TypeError("pair must be a 2-tuple of model names, got %r" % (pair,))
    a, b = str(pair[0]).strip().lower(), str(pair[1]).strip().lower()
    if not a or not b:
        raise ValueError("pair entries must be non-empty model names: %r" % (pair,))
    return (a, b) if a <= b else (b, a)


_PAIR_FEATURE_NAMES_RAW: Tuple[Tuple[Tuple[str, str], Tuple[str, ...]], ...] = (
    (
        # Spaced-dash geometry, semicolon rate and parenthetical density.
        ("claude-opus-4-6", "claude-opus-5"),
        (
            "em_dash_spaced_per_100w", "em_dash_spaced_share",
            "em_dash_style_entropy", "em_dash_per_sentence",
            "semicolon_per_100w", "paren_open_per_100w",
            "paren_pair_per_sentence", "colon_per_100w",
            "contrastive_connective_per_100w", "comma_per_100w",
        ),
    ),
    (
        # Lexical diversity, rhythm variance and Latinate register.
        ("claude-opus-4-8", "claude-sonnet-5"),
        (
            "ttr", "root_ttr", "yule_k_log", "simpson_d", "hapax_share",
            "std_sentence_words", "cv_sentence_words", "latinate_share",
            "polysyllable_share", "mean_word_length", "long_word_share",
            "abstract_noun_share", "short_sentence_share", "stopword_share",
            "mean_sentence_words", "paren_open_per_100w",
        ),
    ),
    (
        # Narrative pacing, dialogue and sensory density.
        ("claude-fable-5", "claude-opus-4-8"),
        (
            "narrative_pacing_index", "dialogue_tag_per_100w",
            "dialogue_sentence_share", "narrative_marker_per_100w",
            "action_verb_per_100w", "sensory_descriptor_per_100w",
            "color_term_per_100w", "sound_term_per_100w",
            "texture_term_per_100w", "light_term_per_100w",
            "first_person_singular_per_100w", "short_sentence_share",
            "cv_sentence_words", "paren_pair_per_sentence",
            "temperature_term_per_100w", "short_word_share",
        ),
    ),
    (
        # Paragraph compactness, list habit, connective density.
        ("gemini-3.6-flash", "gemini-3.7-flash"),
        (
            "mean_paragraph_words", "sentences_per_paragraph",
            "std_paragraph_words", "single_sentence_paragraph_share",
            "list_item_per_100w", "list_line_share", "connective_per_100w",
            "mean_line_words", "short_line_share", "log_word_count",
            "chars_per_sentence", "punct_per_100w", "yule_k_log",
            "dialogue_sentence_share", "comma_per_100w",
            "mean_sentence_words",
        ),
    ),
    (
        # Curly-quote variance and interrogative frequency.
        ("gemini-3.7-flash", "gemini-3.8-flash"),
        (
            "curly_dq_share", "straight_dq_share", "quote_mark_per_100w",
            "curly_sq_share", "curly_apostrophe_share",
            "question_mark_per_sentence", "question_sentence_share",
            "stopword_share", "simpson_d", "comma_per_100w",
            "semicolon_per_100w", "mean_sentence_words", "cv_sentence_words",
        ),
    ),
    (
        # Header usage, exposition depth, sentence complexity.
        ("gemini-3.1-pro-preview", "gemini-3.7-flash"),
        (
            "technical_term_per_100w", "mean_sentence_words",
            "std_sentence_words", "polysyllable_share", "latinate_share",
            "definitional_per_100w", "markdown_header_per_100w",
            "markdown_header_present", "bold_span_per_100w",
            "list_item_per_100w", "math_symbol_per_100w",
            "chars_per_sentence", "comma_per_100w",
        ),
    ),
    (
        # Em-dash variance and vocabulary formality.
        ("gpt-5.6-luna", "gpt-5.6-sol"),
        (
            "em_dash_total_per_100w", "em_dash_per_sentence",
            "em_dash_spaced_share", "yule_k_log", "latinate_share",
            "polysyllable_share", "booster_per_100w", "hedge_per_100w",
            "mean_word_length", "reasoning_marker_per_100w",
            "number_token_per_100w", "semicolon_per_100w",
            "std_sentence_words", "short_word_share",
        ),
    ),
    (
        # Reasoning discourse markers vs grounded declarative phrasing.
        ("gpt-5.6-sol", "gpt-5.6-terra"),
        (
            "reasoning_marker_per_100w", "grounded_marker_per_100w",
            "causal_connective_per_100w", "definitional_per_100w",
            "hedge_per_100w", "booster_per_100w", "std_sentence_words",
            "digit_char_share", "number_token_per_100w",
            "mean_sentence_words", "sentences_per_paragraph", "root_ttr",
            "long_word_share", "mean_word_length",
        ),
    ),
    (
        # Curly-quote and em-dash variance across the OpenAI pair.
        ("gpt-5.5", "gpt-6-astra"),
        (
            "curly_dq_share", "curly_apostrophe_share", "quote_mark_per_100w",
            "straight_dq_share", "em_dash_total_per_100w",
            "em_dash_unspaced_per_100w", "comma_per_100w",
            "semicolon_per_100w", "ttr", "mean_sentence_words",
            "chars_per_sentence",
        ),
    ),
    (
        # Sentence compression, character load per word, digit frequency.
        ("glm-5.3", "glm-5.3-flash"),
        (
            "mean_sentence_words", "median_sentence_words",
            "short_sentence_share", "chars_per_word", "mean_word_length",
            "digit_char_share", "number_token_per_100w", "semicolon_per_100w",
            "em_dash_total_per_100w", "std_sentence_words",
            "punct_per_100w", "yule_k_log", "exclamation_per_100w",
            "long_word_share",
        ),
    ),
)

PAIR_FEATURE_NAMES: Dict[Tuple[str, str], Tuple[str, ...]] = {
    _normalize_pair(pair): names for pair, names in _PAIR_FEATURE_NAMES_RAW
}

if len(PAIR_FEATURE_NAMES) != len(SIBLING_PAIRS):  # pragma: no cover
    raise RuntimeError("sibling pair routes must cover exactly the ten pairs")

for _pair_key, _pair_names in PAIR_FEATURE_NAMES.items():  # pragma: no cover
    _missing = [n for n in _pair_names if n not in FEATURE_INDEX]
    if _missing:
        raise RuntimeError(
            "pair %r routes unknown features: %r" % (_pair_key, _missing)
        )
    if len(set(_pair_names)) != len(_pair_names):
        raise RuntimeError("pair %r routes duplicate features" % (_pair_key,))


# --------------------------------------------------------------------------- #
# Extraction and normalisation
# --------------------------------------------------------------------------- #


def _coerce_texts(texts: Iterable[str]) -> List[str]:
    if texts is None:
        return []
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, np.ndarray):
        if texts.ndim == 0:
            return [str(texts.item())]
        return [str(t) for t in texts.tolist()]
    return ["" if t is None else str(t) for t in texts]


def extract_raw_sibling_features(texts) -> np.ndarray:
    """
    Extract unnormalised sibling features as an ``(N, N_FEATURES)`` float64 array.

    Raw columns are interpretable on their own (rates per 100 words, shares in
    ``[0, 1]``, log counts), which makes them suitable for tree models such as
    LightGBM that do not need scaling.  ``NaN``/``inf`` can never escape: every
    column falls back to ``0.0`` if a document is empty or degenerate.
    """
    items = _coerce_texts(texts)
    matrix = np.zeros((len(items), N_FEATURES), dtype=np.float64)
    for row_index, text in enumerate(items):
        stats = _TextStats(text)
        row = matrix[row_index]
        for col_index, fn in enumerate(_SPEC_FNS):
            row[col_index] = _safe(float(fn(stats)))
    return matrix


def _robust_zscore(raw: np.ndarray, clip: float = 8.0) -> np.ndarray:
    """Per-column ``(x - median) / (IQR/1.349)`` with a std fallback."""
    if raw.size == 0:
        return raw.astype(np.float64, copy=True)
    x = np.asarray(raw, dtype=np.float64)
    median = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = (q75 - q25) / 1.349
    std = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    z = (x - median) / scale
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip)


def _quantile_scale(raw: np.ndarray) -> np.ndarray:
    """Per-column average-rank transform onto ``[-1, 1]`` (ties share a rank)."""
    if raw.size == 0:
        return raw.astype(np.float64, copy=True)
    x = np.asarray(raw, dtype=np.float64)
    out = np.zeros_like(x)
    n = x.shape[0]
    if n < 2:
        return out
    for j in range(x.shape[1]):
        col = x[:, j]
        uniq, inverse, counts = np.unique(
            col, return_inverse=True, return_counts=True
        )
        ranks = np.empty(n, dtype=np.float64)
        ranks[inverse] = np.arange(n, dtype=np.float64)
        sums = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
        out[:, j] = 2.0 * ranks / (n - 1) - 1.0
    return out


def extract_sibling_features(
    texts: Iterable[str],
    *,
    method: str = "robust",
    clip: float = 8.0,
) -> np.ndarray:
    """
    Extract sibling-discriminative stylometric features.

    Parameters
    ----------
    texts:
        Iterable of raw passage strings.  A single string is treated as a
        one-element batch.  ``None`` yields an empty ``(0, N_FEATURES)`` matrix.
    method:
        ``"robust"`` (default) applies ``(x - median) / (IQR / 1.349)`` per
        column with a standard-deviation fallback; ``"quantile"`` maps each
        column to ``[-1, 1]`` via average ranks; ``"none"`` returns raw values.
    clip:
        Symmetric clip applied after robust scaling (default 8.0 sigmas).

    Returns
    -------
    numpy.ndarray
        ``(N, N_FEATURES)`` float64 matrix; rows follow the input order.

    Notes
    -----
    Normalisation is *batch-level*: a row depends on the other rows in the
    call.  This is intentional for leaderboard inference (the hidden test set
    is scored as one batch) and for CV, where the scaler must be fit inside
    each fold to avoid leakage.
    """
    raw = extract_raw_sibling_features(texts)
    if method == "none":
        return raw
    if method == "robust":
        return _robust_zscore(raw, clip=float(clip))
    if method == "quantile":
        return _quantile_scale(raw)
    raise ValueError(
        "method must be one of {'robust', 'quantile', 'none'}, got %r" % (method,)
    )


def get_pair_feature_names(pair: Tuple[str, str]) -> Tuple[str, ...]:
    """Return the routed feature names for a sibling pair (order-insensitive)."""
    key = _normalize_pair(pair)
    try:
        return PAIR_FEATURE_NAMES[key]
    except KeyError:
        known = ", ".join("%s/%s" % p for p in sorted(PAIR_FEATURE_NAMES))
        raise KeyError(
            "unknown sibling pair %r; known pairs: %s" % (pair, known)
        ) from None


def get_pair_feature_indices(pair: Tuple[str, str]) -> List[int]:
    """
    Return the most discriminative column indices for a sibling pair.

    The order encodes prior importance (most discriminative first), so a
    specialised head can consume ``X[:, idx[:k]]`` for any prefix ``k``.
    Passing the pair in either order is equivalent.

    Raises
    ------
    KeyError
        If the pair is not one of the ten registered sibling pairs.
    TypeError
        If the argument is not a 2-tuple of model names.
    """
    return [FEATURE_INDEX[name] for name in get_pair_feature_names(pair)]


def feature_names() -> Tuple[str, ...]:
    """Return the full feature schema in column order."""
    return FEATURE_NAMES


def feature_doc(name: str) -> str:
    """Return the stylometric rationale for one feature."""
    return FEATURE_DOCS[name]


# --------------------------------------------------------------------------- #
# Invariant test suite
# --------------------------------------------------------------------------- #


def test_sibling_features_invariants() -> bool:
    """
    Self-contained invariant tests (no external data, no pytest required).

    Covered invariants
    ------------------
    1. Shape, dtype, finiteness and clipping of both normalisation modes.
    2. Empty batch, single document and constant batch degenerate safely.
    3. Permutation equivariance of the batch-robust transform and determinism.
    4. Schema integrity: unique names, substantive rationales, ten valid pair
       routes, order-insensitive lookup, rejection of unknown/malformed pairs.
    5. Stylometric sanity: curly quotes, spaced em dashes, questions, lists,
       digits and word length move the expected features in the expected
       direction.

    Returns
    -------
    bool
        ``True`` if every invariant holds; raises ``AssertionError`` otherwise.
    """
    texts = [
        "The lamp \u2014 a small, stubborn thing \u2014 flickered twice before "
        "dying. She set down the cup; the room went quiet. Later, she would "
        "remember the smell of rain on the sill, and the way the light fell.",
        "\u201cWhy does the algorithm converge?\u201d he asked. \u201cBecause "
        "the gradient vanishes,\u201d she replied. \u201cSpecifically, the "
        "Hessian is positive definite, and therefore the step is bounded.\u201d",
        "## Thermodynamics\n\nEntropy is not disorder; it is a count of "
        "microstates. Therefore, when a system exchanges heat, the number of "
        "accessible configurations changes. Consequently, the second law "
        "follows from statistics, not from a separate postulate.",
        "1. First, measure the baseline.\n2. Second, perturb the parameter.\n"
        "3. Finally, compare the outcomes.\n\nThis protocol is compact and "
        "reproducible.",
        "cat dog run sun map top red bed\n\nhat pin cup box",
        "",
        "   ",
        "...!!??",
        "Multi\nline\ntext with no\nblank lines at all.",
    ]

    # 1. Basic contract -----------------------------------------------------
    matrix = extract_sibling_features(texts)
    assert matrix.shape == (len(texts), N_FEATURES), matrix.shape
    assert matrix.dtype == np.float64, matrix.dtype
    assert np.isfinite(matrix).all(), "non-finite values in robust matrix"
    assert matrix.min() >= -8.0 - 1e-12 and matrix.max() <= 8.0 + 1e-12

    raw = extract_raw_sibling_features(texts)
    assert raw.shape == (len(texts), N_FEATURES), raw.shape
    assert raw.dtype == np.float64, raw.dtype
    assert np.isfinite(raw).all(), "non-finite values in raw matrix"
    non_negative = (
        "em_dash_total_per_100w", "semicolon_per_100w",
        "question_mark_per_sentence", "list_item_per_100w",
        "digit_char_share", "number_token_per_100w", "log_word_count",
    )
    assert (raw[:, [FEATURE_INDEX[n] for n in non_negative]] >= 0.0).all()
    bounded = (
        "em_dash_spaced_share", "curly_dq_share", "straight_dq_share",
        "question_sentence_share", "ttr", "hapax_share", "simpson_d",
        "stopword_share", "markdown_header_present",
    )
    cols = raw[:, [FEATURE_INDEX[n] for n in bounded]]
    assert (cols >= -1e-12).all() and (cols <= 1.0 + 1e-12).all()

    # 2. Degenerate batches --------------------------------------------------
    empty = extract_sibling_features([])
    assert empty.shape == (0, N_FEATURES), empty.shape
    assert empty.dtype == np.float64
    assert extract_raw_sibling_features([]).shape == (0, N_FEATURES)
    assert extract_sibling_features(None).shape == (0, N_FEATURES)

    single = extract_sibling_features(["A single lonely passage."])
    assert single.shape == (1, N_FEATURES)
    assert np.isfinite(single).all()
    assert np.allclose(single, 0.0), "N=1 has no batch variance and must be 0"

    constant = extract_sibling_features(["identical text about nothing"] * 4)
    assert np.isfinite(constant).all()
    assert np.allclose(constant, 0.0), "constant batch must normalise to 0"

    # 3. Permutation equivariance and determinism ---------------------------
    perm = [3, 0, 7, 2, 8, 1, 6, 4, 5]
    assert sorted(perm) == list(range(len(texts)))
    permuted = extract_sibling_features([texts[i] for i in perm])
    inverse = np.argsort(perm)
    assert np.allclose(permuted[inverse], matrix, atol=1e-9), (
        "robust scaling must be permutation-equivariant"
    )
    assert np.allclose(extract_sibling_features(texts), matrix, atol=0.0), (
        "extraction must be deterministic"
    )

    # 4. Quantile / raw modes ------------------------------------------------
    quantile = extract_sibling_features(texts, method="quantile")
    assert quantile.shape == matrix.shape
    assert quantile.dtype == np.float64
    assert np.isfinite(quantile).all()
    assert quantile.min() >= -1.0 - 1e-12 and quantile.max() <= 1.0 + 1e-12
    assert np.allclose(extract_sibling_features(texts, method="none"), raw)
    assert np.allclose(
        extract_sibling_features(texts, method="quantile", clip=2.0),
        quantile,
        atol=1e-12,
    ), "clip must not affect the bounded quantile transform"
    try:
        extract_sibling_features(texts, method="nope")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown normalisation method must raise ValueError")

    # 5. Schema integrity ----------------------------------------------------
    assert len(set(FEATURE_NAMES)) == N_FEATURES
    assert FEATURE_NAMES == tuple(spec.name for spec in _SPECS)
    assert all(FEATURE_DOCS[name].strip() for name in FEATURE_NAMES)
    assert all(len(FEATURE_DOCS[name]) > 40 for name in FEATURE_NAMES), (
        "every feature needs a substantive stylometric rationale"
    )
    assert len(PAIR_FEATURE_NAMES) == len(SIBLING_PAIRS) == 10
    assert set(PAIR_FEATURE_NAMES) == {_normalize_pair(p) for p in SIBLING_PAIRS}

    for pair in SIBLING_PAIRS:
        indices = get_pair_feature_indices(pair)
        assert indices, "pair route must not be empty: %r" % (pair,)
        assert len(set(indices)) == len(indices), "duplicate indices: %r" % (pair,)
        assert all(0 <= i < N_FEATURES for i in indices), pair
        reversed_indices = get_pair_feature_indices((pair[1], pair[0]))
        assert indices == reversed_indices, "pair lookup must be order-insensitive"
        assert get_pair_feature_names(pair) == get_pair_feature_names(
            (pair[1].upper(), pair[0])
        )

    try:
        get_pair_feature_indices(("gpt-5.5", "not-a-model"))
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown pairs must raise KeyError")

    try:
        get_pair_feature_indices(("only-one",))  # type: ignore[arg-type]
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("malformed pairs must raise TypeError")

    # 6. Stylometric sanity --------------------------------------------------
    curly = "\u201cA quiet room,\u201d she said. \u201cNothing more.\u201d"
    straight = '"A quiet room," she said. "Nothing more."'
    quote_raw = extract_raw_sibling_features([curly, straight])
    assert quote_raw[0, FEATURE_INDEX["curly_dq_share"]] > 0.99
    assert quote_raw[1, FEATURE_INDEX["curly_dq_share"]] == 0.0
    assert quote_raw[1, FEATURE_INDEX["straight_dq_share"]] == 1.0

    dash_raw = extract_raw_sibling_features(
        ["The plan \u2014 flawed as it was \u2014 worked.", "The plan worked."]
    )
    assert dash_raw[0, FEATURE_INDEX["em_dash_total_per_100w"]] > 0.0
    assert dash_raw[1, FEATURE_INDEX["em_dash_total_per_100w"]] == 0.0
    assert dash_raw[0, FEATURE_INDEX["em_dash_spaced_share"]] == 1.0
    assert dash_raw[0, FEATURE_INDEX["em_dash_style_entropy"]] == 0.0

    tight = extract_raw_sibling_features(["The plan\u2014flawed\u2014worked."])
    assert tight[0, FEATURE_INDEX["em_dash_spaced_share"]] == 0.0
    assert tight[0, FEATURE_INDEX["em_dash_unspaced_per_100w"]] > 0.0

    question_raw = extract_raw_sibling_features(["Why? How? When? Where?"])
    assert question_raw[0, FEATURE_INDEX["question_sentence_share"]] == 1.0

    list_raw = extract_raw_sibling_features(
        ["- alpha\n- beta\n- gamma\n\nA short closing line."]
    )
    assert list_raw[0, FEATURE_INDEX["list_line_share"]] == 0.75
    assert list_raw[0, FEATURE_INDEX["list_item_per_100w"]] > 0.0

    header_raw = extract_raw_sibling_features(["## Title\n\nBody text here."])
    assert header_raw[0, FEATURE_INDEX["markdown_header_present"]] == 1.0
    assert header_raw[0, FEATURE_INDEX["markdown_header_per_100w"]] > 0.0

    digit_raw = extract_raw_sibling_features(
        ["In 2024 the model scored 0.87 on 18 classes.", "No numbers here at all."]
    )
    assert digit_raw[0, FEATURE_INDEX["digit_char_share"]] > 0.0
    assert digit_raw[1, FEATURE_INDEX["digit_char_share"]] == 0.0
    assert digit_raw[0, FEATURE_INDEX["number_token_per_100w"]] > 0.0

    length_raw = extract_raw_sibling_features(
        ["Extraordinary incomprehensibility antidisestablishmentarianism.",
         "Cat dog sun map."]
    )
    assert (
        length_raw[0, FEATURE_INDEX["chars_per_word"]]
        > length_raw[1, FEATURE_INDEX["chars_per_word"]]
    )

    pacing_raw = extract_raw_sibling_features(
        ["He ran. Then he stopped. Later, he turned back. Suddenly, rain.",
         "The theoretical framework, which had been developed over several "
         "decades of careful scholarship, ultimately proved insufficient."]
    )
    assert (
        pacing_raw[0, FEATURE_INDEX["narrative_pacing_index"]]
        > pacing_raw[1, FEATURE_INDEX["narrative_pacing_index"]]
    )

    return True


if __name__ == "__main__":  # pragma: no cover
    test_sibling_features_invariants()
    print(
        "sibling_features invariants OK: %d features, %d sibling pairs"
        % (N_FEATURES, len(SIBLING_PAIRS))
    )
