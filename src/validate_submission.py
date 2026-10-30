"""Submission contract validator (competition section 6.5).

Usage:
    python src/validate_submission.py submission.parquet [test.csv]

Checks:
  * exactly two columns, id + model, in that order
  * no nulls, no duplicate ids, every id present when test.csv is given
  * every model label is one of the 18 exact competition strings
  * parquet round-trip (format is readable, not CSV with a .parquet suffix)
"""
import sys

import pandas as pd

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]


def validate(sub_path, test_path=None):
    sub = pd.read_parquet(sub_path)
    assert list(sub.columns) == ["id", "model"], f"columns are {list(sub.columns)}, expected ['id', 'model']"
    assert len(sub) > 0, "submission is empty"
    assert not sub.isnull().any().any(), "submission contains null/NaN"
    assert sub["id"].astype(str).is_unique, "duplicate ids"
    bad = sorted(set(sub["model"]) - set(LABELS))
    assert not bad, f"labels outside the 18 allowed: {bad}"
    if test_path is not None:
        test = pd.read_csv(test_path)
        want = set(test["id"].astype(str))
        got = set(sub["id"].astype(str))
        assert len(sub) == len(test), f"row count {len(sub)} != test {len(test)}"
        assert got == want, f"id mismatch: missing={len(want - got)}, extra={len(got - want)}"
    counts = sub["model"].value_counts()
    print(f"OK {sub_path}: rows={len(sub)} classes={counts.size} "
          f"min_class={int(counts.min())} max_class={int(counts.max())}")
    return counts


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
