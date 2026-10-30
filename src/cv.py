"""
Deterministic Cross-Validation splitter for 'Name That LLM'.
Always groups by prompt_id to prevent prompt-topic leakage.
"""

from typing import Generator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import N_SPLITS, RANDOM_SEED


def get_grouped_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_SEED) -> StratifiedGroupKFold:
    """Return the official StratifiedGroupKFold splitter."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def split_folds(df: pd.DataFrame) -> np.ndarray:
    """
    Assign a 'fold' column (0 to n_splits-1) to the dataframe deterministically.
    """
    cv = get_grouped_cv()
    folds = np.zeros(len(df), dtype=int)
    for fold, (_, val_idx) in enumerate(cv.split(df, df["model"], groups=df["prompt_id"])):
        folds[val_idx] = fold
    return folds
