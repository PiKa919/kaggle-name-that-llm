"""
Sibling-level stylometric feature engine for the Kaggle "Name That LLM" competition.
==============================================================================

Competition context
-------------------
The global lab-level decision (OpenAI vs Google vs Anthropic vs Qwen vs ZAI vs
Moonshot) is comparatively easy: quote typography, em-dash geometry and markdown
habits carry most of the signal.  Roughly 55-58% of the remaining errors are
*sibling* errors: the lab is right but the wrong member of the family is chosen
(e.g. ``glm-5.3`` vs ``glm-5.3-flash``, or five near-identical Claude variants).
This module isolates the fine-grained authorial fingerprints that separate the
ten most confusable sibling pairs.

Design principles
-----------------
1. **Never sanitize the text.**  Case, straight vs curly quotes, spaced vs
   unspaced em dashes and digit habits are the signal, not noise.  Every regex
   here is case-sensitive unless the feature is explicitly lexical.  The only
   irreversible operation performed on the raw string is *sentence/paragraph
   segmentation*, which uses a masked-dot trick so that ``3.14`` and ``e.g.``
   are not mistaken for sentence boundaries.
2. **Pure numpy + standard library.**  No pandas, no scikit-learn, no nltk.
   Tokenisation is a single compiled regex; all per-document statistics are
   computed in one pass and stored on a small ``__slots__`` object.
3. **Feature groups are traceable.**  Every feature is registered through the
   ``@_feature`` decorator with a docstring stating its stylometric rationale.
   Registration order defines the column order, so the matrix schema can never
   drift from the documented schema.
4. **Batch-robust normalisation.**  Raw rates are heavy-tailed (Yule's K, digit
   share, em-dash counts on very short passages).  The default output is a
   *robust z-score*: ``(x - median) / (IQR / 1.349)`` with a standard-deviation
   fallback for degenerate columns, clipped to ``+/- clip``.  Quantile scaling to
   ``[-1, 1]`` is available for models that prefer bounded inputs.
5. **Pair routing.**  ``get_pair_feature_indices(pair)`` returns the ordered
   subset of columns that previous CV experiments found most discriminative for
   a given sibling pair, so down-stream stackers can train pair-specialised
   heads without re-deriving the schema.

Typical use
-----------
>>> X = extract_sibling_features(train_texts)                 # (N, 101)
>>> idx = get_pair_feature_indices(("glm-5.3", "glm-5.3-flash"))
>>> X_pair = X[:, idx]
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Callable, Dict, List, NamedTuple, Sequence, Tuple

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
# Regexes and lexical inventories
# --------------------------------------------------------------------------- #

# A word: letters, optionally joined by an internal straight/curly apostrophe.
# Digits are deliberately excluded so that "3.14" and "2024" do not pollute
# lexical richness estimates; they are measured by dedicated digit features.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)
_DIGIT_RE = re.compile(r"\d")
_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w.])")

# Sentence segmentation: keep the terminator, mask decimal points first.
_SENT_RE = re.compile(r"[^.!?\u2026]+[.!?\u2026]*")
_SENT_MASK = "\u0001"
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
_ABBREV_DOT_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Fig|No|cf|al|"
    r"Inc|Ltd|approx|Eq|Ref)\. ?",
    re.IGNORECASE,
)
_PARA_RE = re.compile(r"\n\s*\n")

# Punctuation geometry.
_EM_SPACED_RE = re.compile(r"(?<=\s)\u2014(?=\s)")
_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
_DOUBLE_HYPHEN_RE = re.compile(r"--")
_STRAIGHT_APOS_RE = re.compile(r"(?<=[^\W\d_])'(?=[^\W\d_])")
_CURLY_APOS_RE = re.compile(r"(?<=[^\W\d_])\u2019(?=[^\W\d_])")
_ELLIPSIS_DOTS_RE = re.compile(r"\.\.\.+")
_CONTRACTION_RE = re.compile(r"['\u2019](?:s|t|re|ve|ll|d|m)$")

# Markdown / structural markers (line-anchored).
_HEADER_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S", re.MULTILINE)
_LIST_RE = re.compile(r"^[ \t]*(?:[-*+\u2022]|\d{1,2}[.)])[ \t]+\S", re.MULTILINE)
_TABLE_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
_CODE_RE = re.compile(r"`[^`\n]+`")

# Latinate / Greco-Latinate morphology (a proxy for register and formality).
_LATINATE_RE = re.compile(
    r"(?:tion|sion|ment|ity|ities|ance|ence|ency|ancy|ous|ious|eous|al|ial|"
    r"ive|ative|ate|ify|ise|ize|ism|ist|ary|ory|ical|ic|fy)$"
)
_ABSTRACT_RE = re.compile(r"(?:tion|sion|ment|ity|ness|ism|ance|ence|ship|hood)$")

_PUNCT_CHARS = ".,;:!?\"'()[]{}<>/\\|@#$%^&*_+=~`\u2014\u2013-\u2026\u201c\u201d\u2018\u2019\u00ab\u00bb"
_VOWELS = frozenset("aeiou")

_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be
    because been before being below between both but by can cannot could couldn't
    did didn't do does doesn't doing don't down during each few for from further
    had hadn't has hasn't have haven't having he her here hers herself him himself
    his how however i if in into is isn't it its itself just me more most mustn't
    my myself no nor not of off on once only or other ought our ours ourselves out
    over own same shan't she should shouldn't so some such than that the their
    theirs them themselves then there these they this those through to too under
    until up very was wasn't we were weren't what when where which while who whom
    why will with won't would wouldn't you your yours yourself yourselves
    """.split()
)

_FIRST_SINGULAR = frozenset({"i", "me", "my", "mine", "myself"})
_FIRST_PLURAL = frozenset({"we", "us", "our", "ours", "ourselves"})
_SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "yourselves"})
_THIRD_PERSON = frozenset(
    {"he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
     "their", "theirs", "themselves"}
)


def _phrase_regex(phrases: Sequence[str]) -> "re.Pattern[str]":
    """Compile a whole-word alternation for a phrase inventory."""
    ordered = sorted({p.lower() for p in phrases}, key=len, reverse=True)
    body = "|".join(r"\s+".join(re.escape(w) for w in p.split()) for p in ordered)
    return re.compile(r"\b(?:%s)\b" % body, re.IGNORECASE)


def _stem_regex(phrases: Sequence[str]) -> "re.Pattern[str]":
    """Compile a prefix-anchored alternation (for technical stems/plurals)."""
    ordered = sorted({p.lower() for p in phrases}, key=len, reverse=True)
    body = "|".join(re.escape(p) for p in ordered)
    return re.compile(r"\b(?:%s)" % body, re.IGNORECASE)


_HEDGES = (
    "perhaps", "maybe", "possibly", "probably", "might", "could", "may",
    "seems", "seem", "appears", "appear", "arguably", "presumably", "somewhat",
    "rather", "fairly", "relatively", "roughly", "approximately", "likely",
    "unlikely", "suggests", "suggest", "tends to", "tend to", "in some sense",
    "more or less", "if anything", "it seems", "one might", "to some extent",
)
_BOOSTERS = (
    "clearly", "obviously", "certainly", "undoubtedly", "definitely",
    "absolutely", "indeed", "of course", "always", "never", "must", "crucial",
    "essential", "critical", "fundamentally", "profoundly", "remarkably",
    "extremely", "highly", "very", "truly", "precisely",
)
_NEGATIONS = (
    "not", "no", "never", "none", "nothing", "neither", "nor", "cannot",
    "without", "rarely", "hardly", "barely", "n't",
)
_MODALS = ("can", "could", "may", "might", "must", "shall", "should", "will",
           "would", "ought")

_REASONING_MARKERS = (
    "therefore", "thus", "hence", "consequently", "accordingly", "as a result",
    "it follows", "this implies", "which means", "this means", "this shows",
    "so we can", "given that", "because", "since",
)
_CONCLUSION_MARKERS = (
    "in conclusion", "to conclude", "in summary", "to sum up", "overall",
    "ultimately", "taken together", "in short", "in essence", "on balance",
)
_GROUNDED_MARKERS = (
    "for example", "for instance", "such as", "according to", "in practice",
    "in fact", "typically", "generally", "usually", "in most cases",
    "it is worth noting", "notably", "research shows", "studies show",
)
_EXAMPLE_MARKERS = ("for example", "for instance", "such as", "e.g", "imagine",
                    "consider the case", "take the case")
_EVIDENCE_MARKERS = (
    "research", "study", "studies", "data", "evidence", "empirical",
    "experiment", "experiments", "measured", "observed", "observation",
    "statistically", "sample", "dataset", "peer reviewed",
)
_CONNECTIVES = (
    "however", "moreover", "furthermore", "additionally", "in addition",
    "nevertheless", "nonetheless", "meanwhile", "therefore", "thus",
    "consequently", "similarly", "likewise", "in contrast",
    "on the other hand", "by contrast", "first", "second", "third", "finally",
    "ultimately", "overall", "in conclusion", "to conclude", "in summary",
    "for example", "for instance", "also", "besides", "instead", "rather",
    "yet", "still",
)
_CONTRASTIVE = (
    "however", "nevertheless", "nonetheless", "in contrast", "on the other hand",
    "by contrast", "although", "though", "whereas", "while", "yet", "but",
    "instead", "conversely", "even so", "that said",
)
_TECHNICAL = (
    "theorem", "lemma", "proof", "equation", "algorithm", "complexity",
    "derivative", "integral", "gradient", "probability", "distribution",
    "entropy", "quantum", "relativity", "particle", "momentum", "wavelength",
    "photon", "neuron", "neural", "network", "parameter", "weights", "training",
    "inference", "optimization", "function", "variable", "matrix", "vector",
    "tensor", "dimension", "topology", "geometry", "symmetry", "conservation",
    "thermodynamic", "molecular", "protein", "enzyme", "cell", "evolution",
    "selection", "genome", "organism", "ecosystem", "cognition",
    "consciousness", "epistemology", "ontology", "ethics", "aesthetic",
    "semantics", "syntax", "recursion", "abstraction", "implementation",
    "protocol", "architecture", "compiler", "database", "latency", "throughput",
)
_NARRATIVE_MARKERS = (
    "then", "suddenly", "later", "once", "one day", "that night", "at last",
    "finally", "afterward", "afterwards", "before long", "in the morning",
    "by the time", "for a moment", "a moment later", "from then on", "as if",
    "seemed to",
)
_ACTION_VERBS = (
    "walked", "ran", "turned", "looked", "opened", "closed", "grabbed",
    "pulled", "pushed", "stepped", "moved", "reached", "stood", "sat", "fell",
    "rose", "lifted", "touched", "held", "dropped", "whispered", "shouted",
    "laughed", "cried", "smiled", "frowned", "nodded", "shook", "breathed",
    "listened", "watched", "waited", "knocked", "climbed", "slipped", "caught",
    "threw", "wrote", "read", "slept", "woke",
)
_DIALOGUE_TAGS = (
    "said", "says", "asked", "replied", "answered", "whispered", "murmured",
    "shouted", "called", "exclaimed", "muttered", "sighed", "laughed", "cried",
    "added", "continued", "began", "explained", "told", "responded",
)
_COLORS = (
    "red", "blue", "green", "gold", "golden", "silver", "grey", "gray",
    "black", "white", "crimson", "azure", "amber", "violet", "scarlet",
    "emerald", "indigo", "ochre", "copper", "bronze", "pale", "ivory",
    "turquoise", "maroon", "beige", "rust",
)
_SOUNDS = (
    "whisper", "whispered", "murmur", "murmured", "hum", "hummed", "echo",
    "echoed", "rustle", "rustled", "crackle", "crackled", "roar", "roared",
    "sigh", "sighed", "silence", "silent", "quiet", "loud", "clatter",
    "chime", "ring", "rang", "buzz", "hiss", "thud", "creak",
)
_TEXTURES = (
    "rough", "smooth", "soft", "hard", "damp", "sharp", "velvet", "grit",
    "silk", "coarse", "slick", "sticky", "brittle", "worn", "polished",
    "gravel", "stone", "wood", "glass", "metal",
)
_LIGHT = (
    "light", "shadow", "shadows", "glow", "glowed", "gleam", "gleamed",
    "shimmer", "shimmered", "dark", "darkness", "bright", "dusk", "dawn",
    "flicker", "flickered", "lamp", "sunlight", "moonlight", "twilight",
    "radiant", "dim",
)
_TEMPERATURE = (
    "cold", "warm", "hot", "chill", "freeze", "freezing", "heat", "icy",
    "frigid", "tepid", "burning", "cool", "frost", "steam",
)
_SENSORY = tuple(
    dict.fromkeys(_COLORS + _SOUNDS + _TEXTURES + _LIGHT + _TEMPERATURE)
)

_HEDGE_RE = _phrase_regex(_HEDGES)
_BOOSTER_RE = _phrase_regex(_BOOSTERS)
_NEGATION_RE = re.compile(
    r"\b(?:%s)\b|n't"
    % "|".join(re.escape(w) for w in _NEGATIONS if "'" not in w),
    re.IGNORECASE,
)
_MODAL_RE = _phrase_regex(_MODALS)
_REASONING_RE = _phrase_regex(_REASONING_MARKERS)
_CONCLUSION_RE = _phrase_regex(_CONCLUSION_MARKERS)
_GROUNDED_RE = _phrase_regex(_GROUNDED_MARKERS)
_EXAMPLE_RE = _phrase_regex(_EXAMPLE_MARKERS)
_EVIDENCE_RE = _stem_regex(_EVIDENCE_MARKERS)
_CONNECTIVE_RE = _phrase_regex(_CONNECTIVES)
_CONTRASTIVE_RE = _phrase_regex(_CONTRASTIVE)
_TECHNICAL_RE = _stem_regex(_TECHNICAL)
_NARRATIVE_RE = _phrase_regex(_NARRATIVE_MARKERS)
_ACTION_RE = _phrase_regex(_ACTION_VERBS)
_DIALOGUE_TAG_RE = _phrase_regex(_DIALOGUE_TAGS)
_COLOR_RE = _phrase_regex(_COLORS)
_SOUND_RE = _phrase_regex(_SOUNDS)
_TEXTURE_RE = _phrase_regex(_TEXTURES)
_LIGHT_RE = _phrase_regex(_LIGHT)
_TEMPERATURE_RE = _phrase_regex(_TEMPERATURE)
_SENSORY_RE = _phrase_regex(_SENSORY)


# --------------------------------------------------------------------------- #
# Small numeric helpers (no numpy per document: cheaper for short strings)
# --------------------------------------------------------------------------- #

def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs)) / len(xs) if xs else 0.0


def _pstdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


def _pctl(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return float(s[0])
    pos = (n - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _p100(s: "_TextStats", count: float) -> float:
    """Rate per 100 words; guarded so empty documents yield 0.0."""
    return (100.0 * count / s.n_words) if s.n_words else 0.0


@lru_cache(maxsize=1 << 17)
def _syllables(word: str) -> int:
    """Vowel-group syllable estimate, enough for polysyllable ratios."""
    w = word.lower()
    groups = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in _VOWELS
        if is_v and not prev_vowel:
            groups += 1
        prev_vowel = is_v
    if groups > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        groups -= 1
    return max(groups, 1)


@lru_cache(maxsize=1 << 17)
def _is_latinate(word: str) -> bool:
    return len(word) >= 5 and bool(_LATINATE_RE.search(word.lower()))


@lru_cache(maxsize=1 << 17)
def _is_abstract_noun(word: str) -> bool:
    return len(word) >= 6 and bool(_ABSTRACT_RE.search(word.lower()))


def _split_sentences(block: str) -> List[str]:
    """
    Segment one paragraph into sentences, preserving terminal punctuation.

    Decimal points and common abbreviations are replaced by an out-of-band
    marker before splitting so that "3.14" or "e.g." cannot create a false
    boundary; the marker is restored on the emitted sentences.
    """
    flat = re.sub(r"[ \t]*\n[ \t]*", " ", block).strip()
    if not flat:
        return []
    flat = _DECIMAL_DOT_RE.sub(_SENT_MASK, flat)

    def _mask_abbrev(match: "re.Match[str]") -> str:
        chunk = match.group(0)
        core = chunk.rstrip()
        return core[:-1] + _SENT_MASK + chunk[len(core):]

    flat = _ABBREV_DOT_RE.sub(_mask_abbrev, flat)
    out: List[str] = []
    for match in _SENT_RE.finditer(flat):
        sentence = match.group(0).strip().replace(_SENT_MASK, ".")
        if sentence and _WORD_RE.search(sentence):
            out.append(sentence)
    return out


# --------------------------------------------------------------------------- #
# Per-document statistics (single pass, __slots__, no intermediate DataFrames)
# --------------------------------------------------------------------------- #

class _TextStats:
    """All raw counts and aggregates needed by the feature functions."""

    __slots__ = (
        "text", "n_chars", "n_nonspace",
        "raw_tokens", "tokens", "n_words", "counter", "n_types", "word_lengths",
        "mean_word_length", "std_word_length", "word_length_iqr", "chars_per_word",
        "long_word_count", "short_word_count", "polysyllable_count",
        "ttr", "root_ttr", "herdan_c", "hapax_count", "hapax_share",
        "yule_k_log", "simpson_d",
        "latinate_count", "abstract_noun_count", "stopword_count",
        "contraction_count",
        "sentences", "sent_word_counts", "n_sentences", "questions",
        "dialogue_sentences",
        "paragraphs", "para_word_counts", "para_sent_counts", "n_paragraphs",
        "lines", "line_word_counts", "n_lines",
        "sent_mean", "sent_std", "sent_median", "sent_p10", "sent_p90",
        "sent_short_share", "sent_long_share",
        "para_mean", "para_std", "single_sent_para_share",
        "sentences_per_paragraph", "line_mean", "short_line_share",
        "em_dash", "em_dash_spaced", "en_dash", "hyphen", "double_hyphen",
        "semicolon", "colon", "paren_open", "paren_close", "square_bracket",
        "ellipsis",
        "curly_dq", "straight_dq", "curly_sq", "straight_sq",
        "curly_apostrophe", "straight_apostrophe",
        "comma", "period", "digit_chars", "number_tokens", "question_marks",
        "exclamations", "punct_chars",
        "uppercase_words", "allcaps_words", "titlecase_words",
        "header_count", "list_count", "bold_count", "code_count", "table_count",
        "fps_count", "fpp_count", "second_count", "third_count",
    )

    def __init__(self, text: str) -> None:
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        self.text = text
        self.n_chars = len(text)
        self.n_nonspace = sum(1 for ch in text if not ch.isspace())

        # ---- lexical tokens -------------------------------------------------
        raw_tokens = _WORD_RE.findall(text)
        tokens = [t.lower() for t in raw_tokens]
        self.raw_tokens = raw_tokens
        self.tokens = tokens
        n_words = len(tokens)
        self.n_words = n_words

        counter = Counter(tokens)
        self.counter = counter
        self.n_types = len(counter)

        lengths = [len(t.replace("'", "").replace("\u2019", "")) for t in tokens]
        self.word_lengths = lengths
        self.mean_word_length = _mean(lengths)
        self.std_word_length = _pstdev(lengths)
        self.word_length_iqr = _pctl(lengths, 75) - _pctl(lengths, 25)
        # Characters per word *including punctuation*: a compression signal that
        # mixes word length with punctuation density (GLM flash compresses both).
        self.chars_per_word = (self.n_nonspace / n_words) if n_words else 0.0
        self.long_word_count = sum(1 for x in lengths if x >= 9)
        self.short_word_count = sum(1 for x in lengths if 0 < x <= 3)
        self.polysyllable_count = sum(1 for t in tokens if _syllables(t) >= 3)

        # ---- lexical richness ----------------------------------------------
        self.ttr = (self.n_types / n_words) if n_words else 0.0
        self.root_ttr = (self.n_types / math.sqrt(n_words)) if n_words else 0.0
        if n_words > 1 and self.n_types > 1:
            self.herdan_c = math.log(self.n_types) / math.log(n_words)
        else:
            self.herdan_c = 0.0
        self.hapax_count = sum(1 for f in counter.values() if f == 1)
        self.hapax_share = (self.hapax_count / self.n_types) if self.n_types else 0.0
        if n_words >= 2:
            m2 = sum(f * f for f in counter.values())
            yule_k = 1e4 * (m2 - n_words) / (n_words * n_words)
            self.yule_k_log = math.log1p(max(yule_k, 0.0))
            simpson = sum(f * (f - 1) for f in counter.values())
            self.simpson_d = simpson / (n_words * (n_words - 1))
        else:
            self.yule_k_log = 0.0
            self.simpson_d = 0.0

        self.latinate_count = sum(1 for t in tokens if _is_latinate(t))
        self.abstract_noun_count = sum(1 for t in tokens if _is_abstract_noun(t))
        self.stopword_count = sum(1 for t in tokens if t in _STOPWORDS)
        self.contraction_count = sum(1 for t in tokens if _CONTRACTION_RE.search(t))

        # ---- sentence / paragraph / line structure --------------------------
        paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]
        self.paragraphs = paragraphs
        self.n_paragraphs = len(paragraphs)

        sentences: List[str] = []
        para_sent_counts: List[int] = []
        for para in paragraphs:
            sents = _split_sentences(para)
            para_sent_counts.append(len(sents))
            sentences.extend(sents)
        if not sentences and text.strip():
            sentences = [text.strip()]
            para_sent_counts = [1]
        self.sentences = sentences
        self.para_sent_counts = para_sent_counts
        self.n_sentences = len(sentences)
        self.sent_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
        self.questions = sum(1 for s in sentences if s.rstrip().endswith("?"))
        self.dialogue_sentences = sum(
            1 for s in sentences if ('"' in s or "\u201c" in s or "\u201d" in s)
        )

        self.para_word_counts = [
            len(_WORD_RE.findall(p)) for p in paragraphs
        ]
        self.lines = [ln for ln in text.splitlines() if ln.strip()]
        self.n_lines = len(self.lines)
        self.line_word_counts = [len(_WORD_RE.findall(ln)) for ln in self.lines]

        sent_counts = [c for c in self.sent_word_counts if c > 0]
        self.sent_mean = _mean(sent_counts)
        self.sent_std = _pstdev(sent_counts)
        self.sent_median = _median(sent_counts)
        self.sent_p10 = _pctl(sent_counts, 10)
        self.sent_p90 = _pctl(sent_counts, 90)
        if sent_counts:
            self.sent_short_share = sum(1 for c in sent_counts if c <= 8) / len(sent_counts)
            self.sent_long_share = sum(1 for c in sent_counts if c >= 30) / len(sent_counts)
        else:
            self.sent_short_share = 0.0
            self.sent_long_share = 0.0

        para_counts = [c for c in self.para_word_counts if c > 0]
        self.para_mean = _mean(para_counts)
        self.para_std = _pstdev(para_counts)
        if paragraphs:
            self.single_sent_para_share = (
                sum(1 for c in para_sent_counts if c == 1) / len(paragraphs)
            )
            self.sentences_per_paragraph = self.n_sentences / len(paragraphs)
        else:
            self.single_sent_para_share = 0.0
            self.sentences_per_paragraph = 0.0

        line_counts = [c for c in self.line_word_counts if c > 0]
        self.line_mean = _mean(line_counts)
        self.short_line_share = (
            sum(1 for c in line_counts if c <= 5) / len(line_counts)
            if line_counts else 0.0
        )

        # ---- punctuation geometry -------------------------------------------
        self.em_dash = text.count("\u2014")
        self.em_dash_spaced = len(_EM_SPACED_RE.findall(text))
        self.en_dash = text.count("\u2013")
        self.hyphen = len(_HYPHEN_RE.findall(text))
        self.double_hyphen = len(_DOUBLE_HYPHEN_RE.findall(text))
        self.semicolon = text.count(";")
        self.colon = text.count(":")
        self.paren_open = text.count("(")
        self.paren_close = text.count(")")
        self.square_bracket = text.count("[") + text.count("]")
        self.ellipsis = text.count("\u2026") + len(_ELLIPSIS_DOTS_RE.findall(text))

        self.straight_apostrophe = len(_STRAIGHT_APOS_RE.findall(text))
        self.curly_apostrophe = len(_CURLY_APOS_RE.findall(text))
        self.curly_dq = text.count("\u201c") + text.count("\u201d")
        self.straight_dq = text.count('"')
        self.curly_sq = max(
            0, text.count("\u2019") - self.curly_apostrophe + text.count("\u2018")
        )
        self.straight_sq = max(0, text.count("'") - self.straight_apostrophe)

        self.comma = text.count(",")
        self.period = text.count(".")
        self.digit_chars = len(_DIGIT_RE.findall(text))
        self.number_tokens = len(_NUMBER_RE.findall(text))
        self.question_marks = text.count("?")
        self.exclamations = text.count("!")
        self.punct_chars = sum(text.count(c) for c in _PUNCT_CHARS)

        # ---- orthographic and markdown habits --------------------------------
        self.uppercase_words = sum(1 for t in raw_tokens if t[:1].isupper())
        self.allcaps_words = sum(
            1 for t in raw_tokens if len(t) > 1 and t.isupper()
        )
        self.titlecase_words = sum(
            1 for t in raw_tokens
            if len(t) > 1 and t[0].isupper() and t[1:].islower()
        )
        self.header_count = len(_HEADER_RE.findall(text))
        self.list_count = len(_LIST_RE.findall(text))
        self.table_count = len(_TABLE_RE.findall(text))
        self.bold_count = len(_BOLD_RE.findall(text))
        self.code_count = len(_CODE_RE.findall(text))

        # ---- pronoun persona --------------------------------------------------
        self.fps_count = sum(1 for t in tokens if t in _FIRST_SINGULAR)
        self.fpp_count = sum(1 for t in tokens if t in _FIRST_PLURAL)
        self.second_count = sum(1 for t in tokens if t in _SECOND_PERSON)
        self.third_count = sum(1 for t in tokens if t in _THIRD_PERSON)


# --------------------------------------------------------------------------- #
# Feature registry
# --------------------------------------------------------------------------- #

_FeatureFn = Callable[[_TextStats], float]


class _FeatureSpec(NamedTuple):
    name: str
    fn: _FeatureFn
    doc: str


_SPECS: List[_FeatureSpec] = []


def _feature(name: str, doc: str) -> Callable[[_FeatureFn], _FeatureFn]:
    """Register a feature function together with its stylometric rationale."""

    def _register(fn: _FeatureFn) -> _FeatureFn:
        if any(spec.name == name for spec in _SPECS):
            raise RuntimeError("duplicate sibling feature name: %s" % name)
        fn.__doc__ = doc
        fn.__name__ = name
        _SPECS.append(_FeatureSpec(name=name, fn=fn, doc=doc))
        return fn

    return _register


# ---------------------------- typography ---------------------------------- #

@_feature(
    "em_dash_total_per_100w",
    "All em dashes (U+2014) per 100 words. Rationale: em-dash rate is the single "
    "strongest lab-level and sibling-level divider (GLM 5.3 79-82%, Claude Opus "
    "4.6 63-83%, Gemini flashes near 0%, GPT-6 Astra only ~9%).",
)
def _em_dash_total(s: _TextStats) -> float:
    return _p100(s, s.em_dash)


@_feature(
    "em_dash_spaced_per_100w",
    "Spaced em dashes (' \u2014 ') per 100 words. Rationale: spaced dashes mark "
    "parenthetical asides in Claude Opus 4.6 and GLM 5.3, while several siblings "
    "emit the same glyph without surrounding whitespace; spacing is a tokenizer "
    "habit that survives prompting.",
)
def _em_dash_spaced(s: _TextStats) -> float:
    return _p100(s, s.em_dash_spaced)


@_feature(
    "em_dash_unspaced_per_100w",
    "Em dashes glued to both neighbours ('word\u2014word') per 100 words. "
    "Rationale: the unspaced form is the mirror image of the spaced form and "
    "separates typographic conventions that a single total-rate feature would mix.",
)
def _em_dash_unspaced(s: _TextStats) -> float:
    return _p100(s, max(0, s.em_dash - s.em_dash_spaced))


@_feature(
    "em_dash_spaced_share",
    "Fraction of em dashes that are spaced. Rationale: normalises out dash "
    "frequency, isolating pure spacing style; 0.0 means 'never spaced' and 1.0 "
    "means 'always spaced', which is the stable signature of a given decoder.",
)
def _em_dash_spaced_share(s: _TextStats) -> float:
    return (s.em_dash_spaced / s.em_dash) if s.em_dash else 0.0


@_feature(
    "em_dash_style_entropy",
    "Binary entropy (bits) of the spaced-vs-unspaced em-dash choice. Rationale: "
    "some variants mix both conventions inside one paragraph; entropy captures "
    "that inconsistency better than a share alone and is bounded in [0, 1].",
)
def _em_dash_style_entropy(s: _TextStats) -> float:
    return _binary_entropy(_em_dash_spaced_share(s))


@_feature(
    "em_dash_per_sentence",
    "Em dashes per sentence. Rationale: normalising by sentences rather than "
    "words decouples dash habit from sentence length, which itself differs "
    "between verbose Pro and terse Flash siblings.",
)
def _em_dash_per_sentence(s: _TextStats) -> float:
    return (s.em_dash / s.n_sentences) if s.n_sentences else 0.0


@_feature(
    "double_hyphen_per_100w",
    "ASCII double hyphens ('--') per 100 words. Rationale: a fallback em-dash "
    "spelling used by some decoding stacks and older model variants when the "
    "typographic glyph is suppressed.",
)
def _double_hyphen(s: _TextStats) -> float:
    return _p100(s, s.double_hyphen)


@_feature(
    "en_dash_per_100w",
    "En dashes per 100 words. Rationale: en dashes appear in numeric ranges and "
    "British typography; their ratio to em dashes is a stable house style marker.",
)
def _en_dash(s: _TextStats) -> float:
    return _p100(s, s.en_dash)


@_feature(
    "hyphen_per_100w",
    "Intra-word hyphens per 100 words. Rationale: compound-adjective density is "
    "a register signal (formal exposition vs conversational prose) and differs "
    "between full and flash variants of the same family.",
)
def _hyphen(s: _TextStats) -> float:
    return _p100(s, s.hyphen)


@_feature(
    "semicolon_per_100w",
    "Semicolons per 100 words. Rationale: semicolons encode clause chaining; "
    "Opus 4.6 and Opus 5 differ in how often they fuse two independent clauses "
    "instead of starting a new sentence.",
)
def _semicolon(s: _TextStats) -> float:
    return _p100(s, s.semicolon)


@_feature(
    "colon_per_100w",
    "Colons per 100 words. Rationale: colons introduce lists and explanations; "
    "their rate tracks expository 'setup -> payoff' structure used heavily by "
    "Pro-tier models.",
)
def _colon(s: _TextStats) -> float:
    return _p100(s, s.colon)


@_feature(
    "paren_open_per_100w",
    "Opening parentheses per 100 words. Rationale: parenthetical density is the "
    "cleanest probe for the 'aside' habit that distinguishes Claude Opus 4.6 "
    "from the more linear Opus 5.",
)
def _paren_open(s: _TextStats) -> float:
    return _p100(s, s.paren_open)


@_feature(
    "paren_pair_per_sentence",
    "Balanced parenthesis pairs per sentence. Rationale: counts completed asides "
    "rather than stray brackets and normalises by sentence count, so it measures "
    "how often a writer interrupts their own syntax.",
)
def _paren_pair(s: _TextStats) -> float:
    pairs = min(s.paren_open, s.paren_close)
    return (pairs / s.n_sentences) if s.n_sentences else 0.0


@_feature(
    "square_bracket_per_100w",
    "Square brackets per 100 words. Rationale: editorial/technical bracketing is "
    "rare in prose but common in technical exposition, separating Pro-style "
    "answers from Flash-style summaries.",
)
def _square_bracket(s: _TextStats) -> float:
    return _p100(s, s.square_bracket)


@_feature(
    "ellipsis_per_100w",
    "Ellipses ('...' or U+2026) per 100 words. Rationale: trailing-off rhetoric "
    "is a narrative/creative habit (Claude Fable) and is nearly absent from "
    "technical siblings.",
)
def _ellipsis(s: _TextStats) -> float:
    return _p100(s, s.ellipsis)


@_feature(
    "curly_dq_share",
    "Share of double quotes that are typographic curly quotes. Rationale: the "
    "sharpest quote-style split in the corpus (Gemini 3.8 ~75% curly vs Gemini "
    "3.7 ~1%); a share is robust to how many quotations a passage contains.",
)
def _curly_dq_share(s: _TextStats) -> float:
    denom = s.curly_dq + s.straight_dq
    return (s.curly_dq / denom) if denom else 0.0


@_feature(
    "curly_sq_share",
    "Share of single quotes that are curly. Rationale: single-quote typography "
    "varies independently of double quotes across checkpoints, so it adds a "
    "second, partially independent style axis.",
)
def _curly_sq_share(s: _TextStats) -> float:
    denom = s.curly_sq + s.straight_sq
    return (s.curly_sq / denom) if denom else 0.0


@_feature(
    "curly_apostrophe_share",
    "Share of contraction apostrophes that are curly (U+2019). Rationale: "
    "apostrophe glyph choice is emitted by the tokenizer and is highly stable "
    "per checkpoint, independent of how much dialogue a passage contains.",
)
def _curly_apostrophe_share(s: _TextStats) -> float:
    denom = s.curly_apostrophe + s.straight_apostrophe
    return (s.curly_apostrophe / denom) if denom else 0.0


@_feature(
    "dq_style_entropy",
    "Binary entropy of the curly-vs-straight double-quote choice. Rationale: "
    "mixed quote styles inside one passage indicate a model that does not commit "
    "to a single typographic convention, which is itself discriminative.",
)
def _dq_style_entropy(s: _TextStats) -> float:
    return _binary_entropy(_curly_dq_share(s))


@_feature(
    "curly_quote_total_per_100w",
    "All curly quote glyphs per 100 words. Rationale: measures how much of the "
    "passage is quoted at all, which contextualises the curly *share* features.",
)
def _curly_quote_total(s: _TextStats) -> float:
    return _p100(s, s.curly_dq + s.curly_sq + s.curly_apostrophe)


@_feature(
    "straight_quote_total_per_100w",
    "All straight quote glyphs per 100 words. Rationale: the straight-quote "
    "counterpart of the previous feature; Claude and GLM are heavy straight-quote "
    "emitters, OpenAI and Gemini are not.",
)
def _straight_quote_total(s: _TextStats) -> float:
    return _p100(s, s.straight_dq + s.straight_sq + s.straight_apostrophe)


@_feature(
    "question_mark_per_sentence",
    "Question marks per sentence. Rationale: interrogative framing is a "
    "discourse strategy; Flash variants ask rhetorical questions far more often "
    "than Pro variants.",
)
def _question_mark_per_sentence(s: _TextStats) -> float:
    return (s.question_marks / s.n_sentences) if s.n_sentences else 0.0


@_feature(
    "question_sentence_share",
    "Fraction of sentences ending in a question mark. Rationale: normalises the "
    "previous feature by sentence count so that a single long question does not "
    "look like an interrogative style.",
)
def _question_sentence_share(s: _TextStats) -> float:
    return (s.questions / s.n_sentences) if s.n_sentences else 0.0


@_feature(
    "exclamation_per_100w",
    "Exclamation marks per 100 words. Rationale: emotional amplification is "
    "concentrated in creative/persona-heavy siblings and is almost absent from "
    "formal reasoning variants.",
)
def _exclamation(s: _TextStats) -> float:
    return _p100(s, s.exclamations)


@_feature(
    "digit_char_share",
    "Fraction of characters that are digits. Rationale: numeric density tracks "
    "grounded, example-heavy answers and separates GLM 5.3 from its compressed "
    "flash sibling.",
)
def _digit_char_share(s: _TextStats) -> float:
    return (s.digit_chars / s.n_chars) if s.n_chars else 0.0


@_feature(
    "number_token_per_100w",
    "Standalone number tokens per 100 words. Rationale: complements raw digit "
    "share by counting quantities rather than digits, so '1,000,000' counts once.",
)
def _number_token(s: _TextStats) -> float:
    return _p100(s, s.number_tokens)


@_feature(
    "comma_per_100w",
    "Commas per 100 words. Rationale: comma density is a syntactic-complexity "
    "proxy and one of the cheapest signals for clause-heavy Pro prose.",
)
def _comma(s: _TextStats) -> float:
    return _p100(s, s.comma)


@_feature(
    "period_per_100w",
    "Periods per 100 words. Rationale: sentence-termination density is the "
    "inverse of sentence length and flags the staccato style of flash models.",
)
def _period(s: _TextStats) -> float:
    return _p100(s, s.period)


@_feature(
    "punctuation_per_100w",
    "All punctuation characters per 100 words. Rationale: global punctuation "
    "loading separates heavily typeset models (Claude, GLM) from plain-prose "
    "models (some Gemini flashes).",
)
def _punctuation_per_100w(s: _TextStats) -> float:
    return _p100(s, s.punct_chars)


@_feature(
    "punctuation_per_sentence",
    "Punctuation characters per sentence. Rationale: measures intra-sentence "
    "ornamentation independently of sentence length.",
)
def _punctuation_per_sentence(s: _TextStats) -> float:
    return (s.punct_chars / s.n_sentences) if s.n_sentences else 0.0


# ------------------------- structure / rhythm ------------------------------ #

@_feature(
    "log_word_count",
    "Natural log of word count. Rationale: length is a genuine style prior (Pro "
    "models write longer answers than flash variants) and is also needed to "
    "contextualise every rate feature.",
)
def _log_word_count(s: _TextStats) -> float:
    return math.log1p(s.n_words)


@_feature(
    "mean_sentence_words",
    "Mean words per sentence. Rationale: the primary compression axis; GLM 5.3 "
    "Flash and Gemini Flash systematically shorten sentences relative to their "
    "full siblings.",
)
def _mean_sentence_words(s: _TextStats) -> float:
    return s.sent_mean


@_feature(
    "std_sentence_words",
    "Population standard deviation of sentence length. Rationale: rhythm "
    "variability separates Opus 4.8 from Sonnet 5, which keeps a much tighter "
    "sentence-length distribution.",
)
def _std_sentence_words(s: _TextStats) -> float:
    return s.sent_std


@_feature(
    "cv_sentence_words",
    "Coefficient of variation of sentence length (std / mean). Rationale: "
    "scale-free rhythm irregularity, comparable across short and long passages.",
)
def _cv_sentence_words(s: _TextStats) -> float:
    return (s.sent_std / s.sent_mean) if s.sent_mean > 0 else 0.0


@_feature(
    "median_sentence_words",
    "Median words per sentence. Rationale: robust to one very long or very short "
    "outlier sentence, giving a cleaner compression signal than the mean.",
)
def _median_sentence_words(s: _TextStats) -> float:
    return s.sent_median


@_feature(
    "sentence_p90_p10_range",
    "90th minus 10th percentile of sentence length. Rationale: interquantile "
    "range captures burstiness (a short punchline after a long clause) without "
    "being dominated by a single extreme sentence.",
)
def _sentence_range(s: _TextStats) -> float:
    return s.sent_p90 - s.sent_p10


@_feature(
    "short_sentence_share",
    "Fraction of sentences with at most 8 words. Rationale: short-sentence "
    "density is the fingerprint of punchy flash prose and of narrative pacing.",
)
def _short_sentence_share(s: _TextStats) -> float:
    return s.sent_short_share


@_feature(
    "long_sentence_share",
    "Fraction of sentences with at least 30 words. Rationale: the opposite tail; "
    "Pro-tier models produce more multi-clause sentences than flash siblings.",
)
def _long_sentence_share(s: _TextStats) -> float:
    return s.sent_long_share


@_feature(
    "sentence_count_log",
    "Log of sentence count. Rationale: absolute sentence count interacts with "
    "paragraph structure and is a useful control variable for the rate features.",
)
def _sentence_count_log(s: _TextStats) -> float:
    return math.log1p(s.n_sentences)


@_feature(
    "paragraph_count_log",
    "Log of paragraph count. Rationale: multi-paragraph answers are a structural "
    "choice; Gemini Pro and GPT reasoning variants split more often than flashes.",
)
def _paragraph_count_log(s: _TextStats) -> float:
    return math.log1p(s.n_paragraphs)


@_feature(
    "mean_paragraph_words",
    "Mean words per paragraph. Rationale: paragraph compactness; flash models "
    "produce shorter, more numerous blocks than Pro models.",
)
def _mean_paragraph_words(s: _TextStats) -> float:
    return s.para_mean


@_feature(
    "std_paragraph_words",
    "Standard deviation of paragraph length. Rationale: uneven paragraphing "
    "signals list-plus-prose mixing, common in structured technical answers.",
)
def _std_paragraph_words(s: _TextStats) -> float:
    return s.para_std


@_feature(
    "single_sentence_paragraph_share",
    "Fraction of paragraphs containing exactly one sentence. Rationale: the "
    "one-line paragraph is a distinct rhetorical habit (emphatic beats, list-like "
    "exposition) that differs across Gemini checkpoints.",
)
def _single_sentence_paragraph_share(s: _TextStats) -> float:
    return s.single_sent_para_share


@_feature(
    "sentences_per_paragraph",
    "Sentences per paragraph. Rationale: structural compactness in one number; "
    "high values mean dense blocks, low values mean fragmented layout.",
)
def _sentences_per_paragraph(s: _TextStats) -> float:
    return s.sentences_per_paragraph


@_feature(
    "mean_line_words",
    "Mean words per non-empty line. Rationale: line length is a Markdown layout "
    "habit and separates list-heavy flash output from continuous prose.",
)
def _mean_line_words(s: _TextStats) -> float:
    return s.line_mean


@_feature(
    "short_line_share",
    "Fraction of non-empty lines with at most 5 words. Rationale: detects bullet "
    "and header scaffolding even when list markers were stripped upstream.",
)
def _short_line_share(s: _TextStats) -> float:
    return s.short_line_share


@_feature(
    "list_item_per_100w",
    "Markdown/numbered list items per 100 words. Rationale: list frequency is a "
    "strong behavioural marker of flash models answering 'how many / what are' "
    "prompts.",
)
def _list_item_per_100w(s: _TextStats) -> float:
    return _p100(s, s.list_count)


@_feature(
    "list_line_share",
    "Fraction of non-empty lines that are list items. Rationale: normalises list "
    "usage by layout size, avoiding a bias toward long documents.",
)
def _list_line_share(s: _TextStats) -> float:
    return (s.list_count / s.n_lines) if s.n_lines else 0.0


@_feature(
    "markdown_header_per_100w",
    "Markdown headers per 100 words. Rationale: explicit sectioning is used by "
    "Pro/reasoning variants and rarely by flash variants in the same family.",
)
def _markdown_header_per_100w(s: _TextStats) -> float:
    return _p100(s, s.header_count)


@_feature(
    "markdown_header_present",
    "Binary indicator that at least one Markdown header exists. Rationale: the "
    "decision to structure at all is closer to binary than to a rate; keeps the "
    "feature robust for very short passages.",
)
def _markdown_header_present(s: _TextStats) -> float:
    return 1.0 if s.header_count else 0.0


@_feature(
    "bold_span_per_100w",
    "Bold spans per 100 words. Rationale: emphasis markup is a formatting habit "
    "that survives superficial markdown stripping only in some checkpoints.",
)
def _bold_span(s: _TextStats) -> float:
    return _p100(s, s.bold_count)


@_feature(
    "code_span_per_100w",
    "Inline code spans per 100 words. Rationale: code formatting signals a "
    "technical register and separates CS-oriented answers from philosophy ones.",
)
def _code_span(s: _TextStats) -> float:
    return _p100(s, s.code_count)


@_feature(
    "table_line_share",
    "Fraction of lines that look like Markdown table rows. Rationale: tabular "
    "output is a rare, highly model-specific formatting decision.",
)
def _table_line_share(s: _TextStats) -> float:
    return (s.table_count / s.n_lines) if s.n_lines else 0.0


# --------------------------- lexical richness ------------------------------ #

@_feature(
    "ttr",
    "Type-token ratio. Rationale: raw lexical diversity; high values indicate "
    "less repetition, a documented difference between Opus 4.8 and Sonnet 5.",
)
def _ttr(s: _TextStats) -> float:
    return s.ttr


@_feature(
    "root_ttr",
    "Types divided by sqrt(tokens) (root TTR). Rationale: length-corrected "
    "diversity, because plain TTR mechanically falls as passages get longer and "
    "passage length is itself model-correlated.",
)
def _root_ttr(s: _TextStats) -> float:
    return s.root_ttr


@_feature(
    "herdan_c",
    "Herdan's C = log(types) / log(tokens). Rationale: another length-robust "
    "diversity index that is close to constant for a given author and therefore "
    "a stable persona fingerprint.",
)
def _herdan_c(s: _TextStats) -> float:
    return s.herdan_c


@_feature(
    "yule_k_log",
    "log1p of Yule's K characteristic. Rationale: K measures vocabulary "
    "repetition concentration and is one of the classic authorship-attribution "
    "statistics requested for the Opus 4.8 vs Sonnet 5 pair.",
)
def _yule_k_log(s: _TextStats) -> float:
    return s.yule_k_log


@_feature(
    "simpson_d",
    "Simpson's index of word repetition. Rationale: probability that two random "
    "tokens are the same type; complements Yule's K with a bounded scale.",
)
def _simpson_d(s: _TextStats) -> float:
    return s.simpson_d


@_feature(
    "hapax_share",
    "Share of vocabulary types occurring exactly once. Rationale: hapax density "
    "is sensitive to one-off Latinate coinages, a marker of the more ornate "
    "Claude variants.",
)
def _hapax_share(s: _TextStats) -> float:
    return s.hapax_share


@_feature(
    "mean_word_length",
    "Mean letters per word (apostrophes stripped). Rationale: a core readability "
    "and formality proxy; GLM Flash compresses word length relative to GLM 5.3.",
)
def _mean_word_length(s: _TextStats) -> float:
    return s.mean_word_length


@_feature(
    "std_word_length",
    "Standard deviation of word length. Rationale: vocabulary mixing (short "
    "function words plus long technical terms) creates a wider distribution in "
    "expository Pro prose.",
)
def _std_word_length(s: _TextStats) -> float:
    return s.std_word_length


@_feature(
    "word_length_iqr",
    "Interquartile range of word length. Rationale: robust version of the "
    "previous feature, insensitive to a single 25-letter technical term.",
)
def _word_length_iqr(s: _TextStats) -> float:
    return s.word_length_iqr


@_feature(
    "chars_per_word",
    "Non-space characters per word, including punctuation. Rationale: the "
    "compression measure requested for the GLM pair; it blends word length with "
    "punctuation density, both of which flash variants reduce.",
)
def _chars_per_word(s: _TextStats) -> float:
    return s.chars_per_word


@_feature(
    "long_word_share",
    "Share of words with at least 9 letters. Rationale: long-word density is the "
    "simplest surface measure of a Latinate, academic register.",
)
def _long_word_share(s: _TextStats) -> float:
    return (s.long_word_count / s.n_words) if s.n_words else 0.0


@_feature(
    "short_word_share",
    "Share of words with at most 3 letters. Rationale: the opposite tail; "
    "conversational and narrative registers use more short words.",
)
def _short_word_share(s: _TextStats) -> float:
    return (s.short_word_count / s.n_words) if s.n_words else 0.0


@_feature(
    "polysyllable_share",
    "Share of words with three or more estimated syllables. Rationale: a "
    "classic readability dimension that tracks technical exposition depth.",
)
def _polysyllable_share(s: _TextStats) -> float:
    return (s.polysyllable_count / s.n_words) if s.n_words else 0.0


@_feature(
    "latinate_share",
    "Share of words carrying a Latinate/Greco-Latinate suffix. Rationale: the "
    "Latinate ratio is the requested register discriminator for Opus 4.8 vs "
    "Sonnet 5 and a formality proxy for the GPT siblings.",
)
def _latinate_share(s: _TextStats) -> float:
    return (s.latinate_count / s.n_words) if s.n_words else 0.0


@_feature(
    "abstract_noun_share",
    "Share of abstract-noun morphology (-tion, -ity, -ness, -ism, -ment). "
    "Rationale: nominalisation density distinguishes philosophical abstraction "
    "from concrete narrative prose.",
)
def _abstract_noun_share(s: _TextStats) -> float:
    return (s.abstract_noun_count / s.n_words) if s.n_words else 0.0


@_feature(
    "stopword_share",
    "Share of tokens that are function words. Rationale: high stopword rates "
    "indicate grammatical, connective-heavy prose; low rates indicate dense "
    "content-word packing.",
)
def _stopword_share(s: _TextStats) -> float:
    return (s.stopword_count / s.n_words) if s.n_words else 0.0


# ----------------------------- persona / tone ------------------------------ #

@_feature(
    "first_person_singular_per_100w",
    "First-person singular pronouns per 100 words. Rationale: authorial 'I' is "
    "far more frequent in narrative/reflective Claude variants than in "
    "impersonal technical siblings.",
)
def _fps(s: _TextStats) -> float:
    return _p100(s, s.fps_count)


@_feature(
    "first_person_plural_per_100w",
    "First-person plural pronouns per 100 words. Rationale: 'we' frames a shared "
    "inquiry and is a didactic register marker used unevenly across labs.",
)
def _fpp(s: _TextStats) -> float:
    return _p100(s, s.fpp_count)


@_feature(
    "second_person_per_100w",
    "Second-person pronouns per 100 words. Rationale: direct address is a "
    "conversational habit that separates chatty flashes from formal Pros.",
)
def _second_person(s: _TextStats) -> float:
    return _p100(s, s.second_count)


@_feature(
    "third_person_pronoun_per_100w",
    "Third-person pronouns per 100 words. Rationale: pronoun-heavy prose signals "
    "narrative continuity rather than abstract exposition.",
)
def _third_person(s: _TextStats) -> float:
    return _p100(s, s.third_count)


@_feature(
    "contraction_per_100w",
    "Contractions per 100 words. Rationale: contraction use is the strongest "
    "single informality cue and is suppressed in formal reasoning variants.",
)
def _contraction(s: _TextStats) -> float:
    return _p100(s, s.contraction_count)


@_feature(
    "hedge_per_100w",
    "Epistemic hedges per 100 words ('perhaps', 'might', 'seems', 'arguably'). "
    "Rationale: hedging density is a persona trait that survives topic changes.",
)
def _hedge(s: _TextStats) -> float:
    return _p100(s, len(_HEDGE_RE.findall(s.text)))


@_feature(
    "booster_per_100w",
    "Intensifiers and certainty boosters per 100 words ('clearly', 'crucial', "
    "'always'). Rationale: booster rate is the tonal opposite of hedging and "
    "separates assertive siblings from cautious ones.",
)
def _booster(s: _TextStats) -> float:
    return _p100(s, len(_BOOSTER_RE.findall(s.text)))


@_feature(
    "negation_per_100w",
    "Negation markers per 100 words. Rationale: contrastive 'not X but Y' "
    "constructions are a rhetorical signature that varies across checkpoints.",
)
def _negation(s: _TextStats) -> float:
    return _p100(s, len(_NEGATION_RE.findall(s.text)))


@_feature(
    "modal_per_100w",
    "Modal auxiliaries per 100 words. Rationale: modal density tracks speculative "
    "vs declarative stance, a documented GPT sibling difference.",
)
def _modal(s: _TextStats) -> float:
    return _p100(s, len(_MODAL_RE.findall(s.text)))


@_feature(
    "reasoning_marker_per_100w",
    "Explicit inference markers per 100 words ('therefore', 'consequently', "
    "'which means'). Rationale: the requested separator between GPT-5.6-sol "
    "reasoning discourse and GPT-5.6-terra grounded declaratives.",
)
def _reasoning_marker(s: _TextStats) -> float:
    return _p100(s, len(_REASONING_RE.findall(s.text)))


@_feature(
    "conclusion_marker_per_100w",
    "Closing/summary markers per 100 words ('in conclusion', 'overall', "
    "'ultimately'). Rationale: explicit wrap-up phrasing is a structural habit "
    "of assistant-tuned models and differs between siblings.",
)
def _conclusion_marker(s: _TextStats) -> float:
    return _p100(s, len(_CONCLUSION_RE.findall(s.text)))


@_feature(
    "grounded_marker_per_100w",
    "Grounded/exemplifying discourse markers per 100 words ('for example', "
    "'according to', 'in practice'). Rationale: ground-truth-anchored phrasing "
    "is the counterpart of abstract reasoning markers.",
)
def _grounded_marker(s: _TextStats) -> float:
    return _p100(s, len(_GROUNDED_RE.findall(s.text)))


@_feature(
    "example_marker_per_100w",
    "Explicit example introducers per 100 words. Rationale: the decision to "
    "illustrate rather than derive is a stable answering strategy.",
)
def _example_marker(s: _TextStats) -> float:
    return _p100(s, len(_EXAMPLE_RE.findall(s.text)))


@_feature(
    "evidence_marker_per_100w",
    "Empirical-evidence vocabulary per 100 words ('study', 'data', 'measured'). "
    "Rationale: evidence framing separates grounded factual variants from "
    "speculative or creative ones.",
)
def _evidence_marker(s: _TextStats) -> float:
    return _p100(s, len(_EVIDENCE_RE.findall(s.text)))


@_feature(
    "connective_per_100w",
    "Discourse connectives per 100 words. Rationale: explicit cohesion is a "
    "structural marker requested for the Gemini 3.6 vs 3.7 flash pair.",
)
def _connective(s: _TextStats) -> float:
    return _p100(s, len(_CONNECTIVE_RE.findall(s.text)))


@_feature(
    "contrastive_connective_per_100w",
    "Contrastive connectives per 100 words ('however', 'whereas', 'yet'). "
    "Rationale: concession/contrast is a rhetorical move that varies between "
    "Claude Opus 4.6 and Opus 5.",
)
def _contrastive_connective(s: _TextStats) -> float:
    return _p100(s, len(_CONTRASTIVE_RE.findall(s.text)))


@_feature(
    "technical_term_per_100w",
    "Technical vocabulary stems per 100 words. Rationale: a direct measure of "
    "technical exposition depth, used to separate Gemini 3.1 Pro from 3.7 Flash.",
)
def _technical_term(s: _TextStats) -> float:
    return _p100(s, len(_TECHNICAL_RE.findall(s.text)))


@_feature(
    "formality_index",
    "Composite formality score from Latinate share, word length, polysyllables, "
    "contractions and first-person rate. Rationale: vocabulary formality is the "
    "requested GPT-5.6-luna vs sol axis; a single composite is more stable than "
    "any one of its five noisy inputs.",
)
def _formality_index(s: _TextStats) -> float:
    return (
        0.45 * _latinate_share(s)
        + 0.25 * min(s.mean_word_length / 8.0, 1.5)
        + 0.20 * min(_polysyllable_share(s) * 4.0, 1.5)
        + 0.10 * min(_p100(s, len(_TECHNICAL_RE.findall(s.text))) / 10.0, 1.5)
        - 0.30 * min(_contraction(s) / 5.0, 1.5)
        - 0.15 * min(_fps(s) / 5.0, 1.5)
    )


@_feature(
    "uppercase_word_share",
    "Share of tokens starting with an uppercase letter. Rationale: capitalisation "
    "habits (including German-style nouns and title casing) survive upstream "
    "markdown stripping and are model-specific.",
)
def _uppercase_word_share(s: _TextStats) -> float:
    return (s.uppercase_words / s.n_words) if s.n_words else 0.0


@_feature(
    "allcaps_word_share",
    "Share of tokens that are fully uppercase. Rationale: acronym and emphasis "
    "habits differ by lab and by reasoning variant.",
)
def _allcaps_word_share(s: _TextStats) -> float:
    return (s.allcaps_words / s.n_words) if s.n_words else 0.0


@_feature(
    "titlecase_word_share",
    "Share of tokens in Title Case. Rationale: title casing appears in headings "
    "and named concepts and is a typographic signature of specific checkpoints.",
)
def _titlecase_word_share(s: _TextStats) -> float:
    return (s.titlecase_words / s.n_words) if s.n_words else 0.0


# ------------------------- narrative / sensory ----------------------------- #

@_feature(
    "dialogue_tag_per_100w",
    "Speech verbs per 100 words ('said', 'whispered', 'replied'). Rationale: "
    "dialogue tags are the clearest marker of Claude Fable's narrative mode vs "
    "Opus 4.8's expository mode.",
)
def _dialogue_tag(s: _TextStats) -> float:
    return _p100(s, len(_DIALOGUE_TAG_RE.findall(s.text)))


@_feature(
    "dialogue_sentence_share",
    "Fraction of sentences containing a double-quoted span. Rationale: quoted "
    "speech density distinguishes fiction-style completions from essays.",
)
def _dialogue_sentence_share(s: _TextStats) -> float:
    return (s.dialogue_sentences / s.n_sentences) if s.n_sentences else 0.0


@_feature(
    "narrative_marker_per_100w",
    "Temporal/narrative sequencers per 100 words ('then', 'suddenly', 'later'). "
    "Rationale: story-time sequencing is the pacing signal requested for the "
    "Fable vs Opus 4.8 pair.",
)
def _narrative_marker(s: _TextStats) -> float:
    return _p100(s, len(_NARRATIVE_RE.findall(s.text)))


@_feature(
    "action_verb_per_100w",
    "Concrete action verbs per 100 words ('walked', 'grabbed', 'turned'). "
    "Rationale: dynamic verb density separates scene-writing from reflection.",
)
def _action_verb(s: _TextStats) -> float:
    return _p100(s, len(_ACTION_RE.findall(s.text)))


@_feature(
    "sensory_descriptor_per_100w",
    "Sensory descriptors (colour, sound, texture, light, temperature) per 100 "
    "words. Rationale: sensory density is a hallmark of creative-prose variants "
    "and nearly absent from analytical completions.",
)
def _sensory(s: _TextStats) -> float:
    return _p100(s, len(_SENSORY_RE.findall(s.text)))


@_feature(
    "color_term_per_100w",
    "Colour terms per 100 words. Rationale: colour naming is a concrete, "
    "hard-to-fake signature of descriptive prose.",
)
def _color_term(s: _TextStats) -> float:
    return _p100(s, len(_COLOR_RE.findall(s.text)))


@_feature(
    "sound_term_per_100w",
    "Sound/auditory terms per 100 words. Rationale: auditory imagery clusters in "
    "narrative writing and is rare in technical exposition.",
)
def _sound_term(s: _TextStats) -> float:
    return _p100(s, len(_SOUND_RE.findall(s.text)))


@_feature(
    "texture_term_per_100w",
    "Tactile/material terms per 100 words. Rationale: texture vocabulary is a "
    "second sensory axis that varies independently of colour.",
)
def _texture_term(s: _TextStats) -> float:
    return _p100(s, len(_TEXTURE_RE.findall(s.text)))


@_feature(
    "light_term_per_100w",
    "Light/darkness terms per 100 words. Rationale: lighting imagery is a "
    "recurring stylistic tic in creative completions.",
)
def _light_term(s: _TextStats) -> float:
    return _p100(s, len(_LIGHT_RE.findall(s.text)))


@_feature(
    "temperature_term_per_100w",
    "Temperature terms per 100 words. Rationale: thermal imagery is a narrow but "
    "high-precision sensory signal.",
)
def _temperature_term(s: _TextStats) -> float:
    return _p100(s, len(_TEMPERATURE_RE.findall(s.text)))


@_feature(
    "narrative_pacing_index",
    "Short-sentence share amplified by narrative-marker rate. Rationale: pacing "
    "is not one statistic but the combination of clipped sentences and explicit "
    "time sequencing; this composite captures that interaction in one column.",
)
def _narrative_pacing(s: _TextStats) -> float:
    return s.sent_short_share * (
        1.0 + _p100(s, len(_NARRATIVE_RE.findall(s.text))) / 5.0
    )


# --------------------------------------------------------------------------- #
# Public schema
# --------------------------------------------------------------------------- #

FEATURE_NAMES: Tuple[str, ...] = tuple(spec.name for spec in _SPECS)
FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}
FEATURE_DOCS: Dict[str, str] = {spec.name: spec.doc for spec in _SPECS}
N_FEATURES: int = len(FEATURE_NAMES)
_SPEC_FNS: Tuple[_FeatureFn, ...] = tuple(spec.fn for spec in _SPECS)

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
    a = str(pair[0]).strip().lower()
    b = str(pair[1]).strip().lower()
    if not a or not b:
        raise ValueError("pair entries must be non-empty model names: %r" % (pair,))
    return (a, b) if a <= b else (b, a)


_PAIR_FEATURE_NAMES_RAW: Tuple[Tuple[Tuple[str, str], Tuple[str, ...]], ...] = (
    (
        ("claude-opus-4-6", "claude-opus-5"),
        (
            "em_dash_spaced_per_100w",
            "em_dash_spaced_share",
            "em_dash_style_entropy",
            "em_dash_per_sentence",
            "semicolon_per_100w",
            "paren_open_per_100w",
            "paren_pair_per_sentence",
            "colon_per_100w",
            "contrastive_connective_per_100w",
            "comma_per_100w",
        ),
    ),
    (
        ("claude-opus-4-8", "claude-sonnet-5"),
        (
            "ttr",
            "root_ttr",
            "yule_k_log",
            "simpson_d",
            "hapax_share",
            "std_sentence_words",
            "cv_sentence_words",
            "latinate_share",
            "polysyllable_share",
            "mean_word_length",
            "long_word_share",
            "abstract_noun_share",
        ),
    ),
    (
        ("claude-fable-5", "claude-opus-4-8"),
        (
            "narrative_pacing_index",
            "dialogue_tag_per_100w",
            "dialogue_sentence_share",
            "narrative_marker_per_100w",
            "action_verb_per_100w",
            "sensory_descriptor_per_100w",
            "color_term_per_100w",
            "sound_term_per_100w",
            "texture_term_per_100w",
            "light_term_per_100w",
            "first_person_singular_per_100w",
            "short_sentence_share",
        ),
    ),
    (
        ("gemini-3.6-flash", "gemini-3.7-flash"),
        (
            "mean_paragraph_words",
            "sentences_per_paragraph",
            "std_paragraph_words",
            "single_sentence_paragraph_share",
            "list_item_per_100w",
            "list_line_share",
            "connective_per_100w",
            "mean_line_words",
            "short_line_share",
            "log_word_count",
        ),
    ),
    (
        ("gemini-3.7-flash", "gemini-3.8-flash"),
        (
            "curly_dq_share",
            "dq_style_entropy",
            "curly_sq_share",
            "curly_quote_total_per_100w",
            "curly_apostrophe_share",
            "question_mark_per_sentence",
            "question_sentence_share",
        ),
    ),
    (
        ("gemini-3.1-pro-preview", "gemini-3.7-flash"),
        (
            "markdown_header_per_100w",
            "markdown_header_present",
            "technical_term_per_100w",
            "mean_sentence_words",
            "std_sentence_words",
            "polysyllable_share",
            "latinate_share",
            "bold_span_per_100w",
            "list_item_per_100w",
            "long_sentence_share",
            "mean_paragraph_words",
        ),
    ),
    (
        ("gpt-5.6-luna", "gpt-5.6-sol"),
        (
            "em_dash_total_per_100w",
            "em_dash_spaced_share",
            "em_dash_style_entropy",
            "formality_index",
            "latinate_share",
            "contraction_per_100w",
            "mean_word_length",
            "long_word_share",
            "hedge_per_100w",
            "booster_per_100w",
        ),
    ),
    (
        ("gpt-5.6-sol", "gpt-5.6-terra"),
        (
            "reasoning_marker_per_100w",
            "conclusion_marker_per_100w",
            "grounded_marker_per_100w",
            "example_marker_per_100w",
            "evidence_marker_per_100w",
            "connective_per_100w",
            "technical_term_per_100w",
            "number_token_per_100w",
            "mean_sentence_words",
            "std_sentence_words",
        ),
    ),
    (
        ("gpt-5.5", "gpt-6-astra"),
        (
            "curly_dq_share",
            "curly_sq_share",
            "curly_apostrophe_share",
            "dq_style_entropy",
            "em_dash_total_per_100w",
            "em_dash_spaced_share",
            "semicolon_per_100w",
            "colon_per_100w",
            "titlecase_word_share",
        ),
    ),
    (
        ("glm-5.3", "glm-5.3-flash"),
        (
            "mean_sentence_words",
            "median_sentence_words",
            "short_sentence_share",
            "long_sentence_share",
            "std_sentence_words",
            "sentence_p90_p10_range",
            "chars_per_word",
            "mean_word_length",
            "digit_char_share",
            "number_token_per_100w",
            "punctuation_per_sentence",
        ),
    ),
)

PAIR_FEATURE_NAMES: Dict[Tuple[str, str], Tuple[str, ...]] = {
    _normalize_pair(pair): names for pair, names in _PAIR_FEATURE_NAMES_RAW
}

# Fail fast at import time if a routed name is missing or duplicated.
for _pair_key, _pair_names in PAIR_FEATURE_NAMES.items():
    _missing = [n for n in _pair_names if n not in FEATURE_INDEX]
    if _missing:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "pair %r references unknown features: %s" % (_pair_key, _missing)
        )
    if len(set(_pair_names)) != len(_pair_names):  # pragma: no cover
        raise RuntimeError("pair %r contains duplicate features" % (_pair_key,))
del _pair_key, _pair_names, _missing


# --------------------------------------------------------------------------- #
# Normalisation and public extraction API
# --------------------------------------------------------------------------- #

def _coerce_texts(texts) -> List[str]:
    """Accept a list/tuple/ndarray/Series/generator or a single string."""
    if texts is None:
        return []
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _robust_zscore(matrix: np.ndarray, clip: float) -> np.ndarray:
    """median / (IQR / 1.349) scaling with std fallback and hard clipping."""
    if matrix.size == 0:
        return matrix.astype(np.float64, copy=True)
    x = np.asarray(matrix, dtype=np.float64)
    med = np.median(x, axis=0)
    q25 = np.percentile(x, 25, axis=0)
    q75 = np.percentile(x, 75, axis=0)
    scale = (q75 - q25) / 1.349
    std = x.std(axis=0)
    degenerate = ~np.isfinite(scale) | (scale < 1e-9)
    scale = np.where(degenerate, std, scale)
    degenerate = ~np.isfinite(scale) | (scale < 1e-9)
    scale = np.where(degenerate, 1.0, scale)
    z = (x - med) / scale
    z = np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(z, -clip, clip)


def _quantile_scale(matrix: np.ndarray) -> np.ndarray:
    """Per-column average-rank scaling to [-1, 1] (ties preserved)."""
    x = np.asarray(matrix, dtype=np.float64)
    n_rows = x.shape[0]
    if n_rows <= 1:
        return np.zeros_like(x)
    out = np.empty_like(x)
    denominator = float(n_rows - 1)
    for col in range(x.shape[1]):
        _, inverse, counts = np.unique(
            x[:, col], return_inverse=True, return_counts=True
        )
        cumulative = np.cumsum(counts).astype(np.float64)
        average_rank = cumulative - (counts.astype(np.float64) - 1.0) / 2.0
        out[:, col] = 2.0 * (average_rank[inverse] - 1.0) / denominator - 1.0
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def extract_raw_sibling_features(texts) -> np.ndarray:
    """
    Extract unnormalised sibling features as an ``(N, N_FEATURES)`` float64 array.

    Raw values are rates, ratios, entropies and counts in their natural units.
    They are useful for inspection, unit tests and for fitting a scaler on the
    training split only.  For modelling, prefer :func:`extract_sibling_features`,
    which applies batch-robust normalisation.
    """
    items = _coerce_texts(texts)
    matrix = np.zeros((len(items), N_FEATURES), dtype=np.float64)
    for row_index, text in enumerate(items):
        stats = _TextStats(text)
        row = matrix[row_index]
        for col_index, fn in enumerate(_SPEC_FNS):
            row[col_index] = fn(stats)
    return matrix


def extract_sibling_features(
    texts,
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
    Normalisation is *batch-level*: the matrix for a passage depends on the
    other passages in the call.  This is intentional for leaderboard inference
    (the hidden test set is scored as one batch) and for CV, where the scaler
    must be fit inside each fold to avoid leakage.
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
    """Return the routed feature names for a sibling pair (order-sensitive)."""
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

    The order encodes prior importance (most discriminative first) derived from
    grouped-CV ablations, so a specialised head can consume ``X[:, idx[:k]]``
    for any prefix ``k``.  Passing the pair in either order is equivalent.

    Raises
    ------
    KeyError
        If the pair is not one of the ten registered sibling pairs.
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
    3. Permutation equivariance of the batch-robust transform.
    4. Schema integrity: unique names, non-empty rationales, valid pair routes,
       order-insensitive pair lookup and rejection of unknown pairs.
    5. Stylometric sanity: curly quotes, em dashes, questions and word length
       move the expected features in the expected direction.

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
    count_features = (
        "em_dash_total_per_100w",
        "semicolon_per_100w",
        "question_mark_per_sentence",
        "list_item_per_100w",
        "digit_char_share",
        "number_token_per_100w",
    )
    assert (raw[:, [FEATURE_INDEX[n] for n in count_features]] >= 0.0).all()

    # 2. Degenerate batches --------------------------------------------------
    empty = extract_sibling_features([])
    assert empty.shape == (0, N_FEATURES), empty.shape
    assert empty.dtype == np.float64
    assert extract_raw_sibling_features([]).shape == (0, N_FEATURES)

    single = extract_sibling_features(["A single lonely passage."])
    assert single.shape == (1, N_FEATURES)
    assert np.isfinite(single).all()
    assert np.allclose(single, 0.0), "N=1 has no batch variance and must be 0"

    constant = extract_sibling_features(["identical text about nothing"] * 4)
    assert np.isfinite(constant).all()
    assert np.allclose(constant, 0.0), "constant batch must normalise to 0"

    # 3. Permutation equivariance -------------------------------------------
    perm = [3, 0, 7, 2, 8, 1, 6, 4, 5]
    assert sorted(perm) == list(range(len(texts)))
    permuted = extract_sibling_features([texts[i] for i in perm])
    inverse = np.argsort(perm)
    assert np.allclose(permuted[inverse], matrix, atol=1e-9), (
        "robust scaling must be permutation-equivariant"
    )
    assert np.allclose(
        extract_sibling_features(texts), matrix, atol=0.0
    ), "extraction must be deterministic"

    # 4. Quantile mode -------------------------------------------------------
    quantile = extract_sibling_features(texts, method="quantile")
    assert quantile.shape == matrix.shape
    assert quantile.dtype == np.float64
    assert np.isfinite(quantile).all()
    assert quantile.min() >= -1.0 - 1e-12 and quantile.max() <= 1.0 + 1e-12
    assert np.allclose(extract_sibling_features(texts, method="none"), raw)

    # 5. Schema integrity ----------------------------------------------------
    assert len(set(FEATURE_NAMES)) == N_FEATURES
    assert FEATURE_NAMES == tuple(spec.name for spec in _SPECS)
    assert all(FEATURE_DOCS[name].strip() for name in FEATURE_NAMES)
    assert all(len(FEATURE_DOCS[name]) > 40 for name in FEATURE_NAMES), (
        "every feature needs a substantive stylometric rationale"
    )
    assert len(PAIR_FEATURE_NAMES) == len(SIBLING_PAIRS) == 10

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
    curly_idx = FEATURE_INDEX["curly_dq_share"]
    assert quote_raw[0, curly_idx] > 0.99, quote_raw[0, curly_idx]
    assert quote_raw[1, curly_idx] == 0.0, quote_raw[1, curly_idx]

    dash_raw = extract_raw_sibling_features(
        ["The plan \u2014 flawed as it was \u2014 worked.", "The plan worked."]
    )
    dash_idx = FEATURE_INDEX["em_dash_total_per_100w"]
    assert dash_raw[0, dash_idx] > 0.0
    assert dash_raw[1, dash_idx] == 0.0
    spaced_idx = FEATURE_INDEX["em_dash_spaced_share"]
    assert dash_raw[0, spaced_idx] == 1.0, dash_raw[0, spaced_idx]

    question_raw = extract_raw_sibling_features(["Why? How? When? Where?"])
    assert question_raw[0, FEATURE_INDEX["question_sentence_share"]] == 1.0

    length_raw = extract_raw_sibling_features(
        ["Extraordinary incomprehensibility antidisestablishmentarianism.",
         "Cat dog sun map."]
    )
    chars_idx = FEATURE_INDEX["chars_per_word"]
    assert length_raw[0, chars_idx] > length_raw[1, chars_idx]

    digit_raw = extract_raw_sibling_features(["In 2024, 37 of 100 trials failed."])
    assert digit_raw[0, FEATURE_INDEX["digit_char_share"]] > 0.0
    assert digit_raw[0, FEATURE_INDEX["number_token_per_100w"]] >= 3.0

    print(
        "sibling_features invariants OK: %d features, %d sibling pairs, "
        "%d sample documents" % (N_FEATURES, len(PAIR_FEATURE_NAMES), len(texts))
    )
    return True


if __name__ == "__main__":  # pragma: no cover
    test_sibling_features_invariants()
