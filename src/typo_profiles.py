"""
Canonical typographic affinity priors for the 18 candidate models.

SINGLE SOURCE OF TRUTH. Previously this table was duplicated verbatim in
`src/hierarchical_classifier.py` and `notebooks/master_sota/submission_sota.py`.
Two copies of a hand-tuned probability table is one copy too many: they drift,
and a drift between the CV-time prior and the submission-time prior is a silent
label-shift bug that no test catches.

Layout: [p_has_em_dash, p_spaced_em_dash, p_curly_double_quote, p_curly_single_quote]

These are EMPIRICAL rates measured on the competition training split. They are
NOT universal constants -- see `patches/v3_typo_robustness.py` for the
Dirichlet-smoothed estimator that resists out-of-distribution prompts.
"""

from typing import Dict, List

# Key order must match the observation vector in compute_typographic_priors.
TYPO_FEATURES: List[str] = [
    "has_em_dash",
    "has_spaced_em_dash",
    "has_curly_double_quote",
    "has_curly_single_quote",
]

TYPO_PROFILES: Dict[str, List[float]] = {
    "claude-fable-5":         [0.83, 0.01, 0.01, 0.01],
    "claude-opus-4-6":        [0.75, 0.45, 0.01, 0.01],
    "claude-opus-4-8":        [0.63, 0.01, 0.01, 0.01],
    "claude-opus-5":          [0.81, 0.40, 0.01, 0.01],
    "claude-sonnet-5":        [0.73, 0.01, 0.01, 0.01],
    "gemini-3.1-pro-preview": [0.01, 0.01, 0.01, 0.35],
    "gemini-3.6-flash":       [0.01, 0.01, 0.01, 0.48],
    "gemini-3.7-flash":       [0.01, 0.01, 0.01, 0.68],
    "gemini-3.8-flash":       [0.01, 0.01, 0.75, 0.67],
    "glm-5.3":                [0.82, 0.05, 0.01, 0.01],
    "glm-5.3-flash":          [0.79, 0.04, 0.01, 0.01],
    "gpt-5.5":                [0.13, 0.01, 0.73, 0.82],
    "gpt-5.6-luna":           [0.37, 0.01, 0.95, 0.97],
    "gpt-5.6-sol":            [0.27, 0.01, 0.95, 0.98],
    "gpt-5.6-terra":          [0.23, 0.01, 0.95, 0.93],
    "gpt-6-astra":            [0.09, 0.01, 0.95, 0.80],
    "kimi-k3":                [0.52, 0.01, 0.85, 0.56],
    "qwen3.8-max":            [0.03, 0.01, 0.98, 0.74],
}

# Baseline rate for a non-emissive, non-curly writer (used as the "other" prior
# in the Beta-Binomial smoothing below).
BASE_RATE: Dict[str, float] = {
    "has_em_dash": 0.10,
    "has_spaced_em_dash": 0.03,
    "has_curly_double_quote": 0.20,
    "has_curly_single_quote": 0.20,
}

# Probability floor/ceiling applied before any log() call.
# floor=0.01 -> max |log-odds contribution| per feature = log(0.99/0.01) = 4.60 nats.
# floor=1e-3 -> 6.91 nats. Raising the floor widens the OOD safety margin at the
# cost of sharpening discrimination on clean (in-distribution) prompts.
TYPO_CLIP_LO: float = 0.01
TYPO_CLIP_HI: float = 0.99
