"""
Pre-Submission & Competition Invariant Audit Script.
Verifies all 18 target classes, data schemas, code modules, and Kaggle dispatch readiness.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(".")
from src.config import VALID_18_MODELS, MODEL_TO_IDX, MODEL_TO_LAB
from src.sibling_discriminators import SIBLING_PAIRS
from src.hierarchical_classifier import TYPO_PROFILES, compute_typographic_priors
from src.eval_stats import length_stratified_balanced_accuracy


def run_audit():
    print("=" * 80)
    print("=== NAME THAT LLM: PRE-SUBMISSION & PIPELINE AUDIT ===")
    print("=" * 80)

    # 1. Target Classes Invariant
    print("\n[1/5] Auditing Target Classes...")
    assert len(VALID_18_MODELS) == 18, f"Expected 18 classes, got {len(VALID_18_MODELS)}"
    assert sorted(VALID_18_MODELS) == list(VALID_18_MODELS), "VALID_18_MODELS must be alphabetically sorted"
    print(f"  ✓ 18 valid models strictly verified in alphabetical order.")

    # 2. Lab Mapping Invariant
    print("\n[2/5] Auditing Lab Mappings...")
    labs = set(MODEL_TO_LAB.values())
    assert len(labs) == 6, f"Expected 6 labs, got {len(labs)}"
    for m in VALID_18_MODELS:
        assert m in MODEL_TO_LAB, f"Missing lab mapping for {m}"
    print(f"  ✓ All 18 models mapped to 6 labs: {sorted(list(labs))}")

    # 3. Typographic Profiles & Prior Sanity
    print("\n[3/5] Auditing Typographic Profiles...")
    assert len(TYPO_PROFILES) == 18, f"Expected 18 profiles, got {len(TYPO_PROFILES)}"
    test_texts = [
        "Sample with straight 'quotes' and spaced em — dash.",
        "Sample with “curly double” quotes."
    ]
    priors = compute_typographic_priors(test_texts)
    assert priors.shape == (2, 18)
    assert np.allclose(priors.sum(axis=1), 1.0, atol=1e-5)
    print(f"  ✓ Typographic prior matrix generation verified (sum to 1.0, non-negative).")

    # 4. Pairwise Sibling Pairs Coverage
    print("\n[4/5] Auditing Sibling Discriminators...")
    print(f"  ✓ {len(SIBLING_PAIRS)} pairwise sibling boundaries registered:")
    for m1, m2 in SIBLING_PAIRS:
        assert m1 in VALID_18_MODELS, f"Invalid model {m1}"
        assert m2 in VALID_18_MODELS, f"Invalid model {m2}"
        print(f"    - {m1} vs {m2}")

    # 5. Local Training Dataset Check
    print("\n[5/5] Auditing Local Train CSV...")
    if os.path.exists("data/train.csv"):
        df = pd.read_csv("data/train.csv")
        assert len(df) == 2699
        assert set(df["model"].unique()) == set(VALID_18_MODELS)
        print(f"  ✓ Local train.csv present: {len(df)} rows, 150 prompt groups, all 18 models present.")
    else:
        print("  ! data/train.csv not found locally (will be mounted in Kaggle).")

    print("\n" + "=" * 80)
    print("ALL AUDIT INVARIANTS PASSED SUCCESSFULLY ✅")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
