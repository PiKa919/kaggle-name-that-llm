"""
Diagnostic Script: Passage Length Buckets & Sibling Confusion Geometry
Analyzes train.csv to quantify the short-passage distribution and sibling collision landscape.
"""

import numpy as np
import pandas as pd
from collections import Counter
from src.config import VALID_18_MODELS, MODEL_TO_LAB

df = pd.read_csv("data/train.csv")
df["text"] = df["text"].fillna("")
df["word_count"] = df["text"].apply(lambda t: len(t.split()))

print("=== PASSAGE LENGTH DISTRIBUTION (train.csv) ===")
print(f"Total passages: {len(df)}")
print(f"Word count: min={df['word_count'].min()}, max={df['word_count'].max()}, mean={df['word_count'].mean():.1f}, median={df['word_count'].median():.1f}")

bins = [0, 60, 90, 120, 160, 250, 1000]
labels = ["<60", "60-90", "90-120", "120-160", "160-250", ">250"]
df["bucket"] = pd.cut(df["word_count"], bins=bins, labels=labels)
print("\nPassages per bucket:")
print(df["bucket"].value_counts().sort_index())

print("\nMean word count per model:")
means = df.groupby("model")["word_count"].agg(["mean", "std", "count"])
print(means.sort_values(by="mean", ascending=False))
