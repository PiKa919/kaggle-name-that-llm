"""VULNERABILITY 1 FIX: Smooth length-gated sigmoidal transition.
Replaces step-function np.where(is_short, 0.45, 0.15) with generalized sigmoid.
"""
import numpy as np

def smooth_typo_weight(word_counts, threshold=80.0, w_max=0.45, w_min=0.15, k=0.15):
    """
    Generalized logistic (Richards) smooth step.
    At threshold: weight = (w_max+w_min)/2.
    Derivative max at threshold = k*(w_max-w_min)/4.
    """
    return w_min + (w_max - w_min) / (1.0 + np.exp(-k * (threshold - word_counts)))

# Drop-in replacement for submission_sota.py line 356-357
# OLD: is_short = (word_counts_test < 80); w_typo = np.where(is_short, 0.45, 0.15)
# NEW: w_typo = smooth_typo_weight(word_counts_test, 80.0, 0.45, 0.15, k=0.12)[:, None]
