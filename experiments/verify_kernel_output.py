"""
Kernel Output Verification & Telemetry Audit Script
Verifies that Kaggle Dual T4 execution completed cleanly and that submission.parquet
satisfies all competition schema invariants.
"""

import json
import os
import subprocess
import sys
import pandas as pd
import numpy as np

KERNEL_ID = "pikaaa/name-that-llm-sota-master-ensemble"
OUTPUT_DIR = "notebooks/master_sota/output_v5"

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]

def main():
    print(f"Checking status for kernel: {KERNEL_ID}...")
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID], capture_output=True, text=True)
    print(res.stdout.strip())

    if "COMPLETE" not in res.stdout:
        print("Kernel is not yet COMPLETE.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading output to {OUTPUT_DIR}...")
    dl_res = subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", OUTPUT_DIR], capture_output=True, text=True)
    print(dl_res.stdout.strip())

    parquet_path = os.path.join(OUTPUT_DIR, "submission.parquet")
    if not os.path.exists(parquet_path):
        print(f"ERROR: {parquet_path} not found!")
        sys.exit(2)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded submission.parquet with shape: {df.shape}")
    assert list(df.columns) == ["id", "model"], f"Columns mismatch: {list(df.columns)}"
    assert not df.isnull().any().any(), "Found null/NaN values!"
    assert df["model"].isin(LABELS).all(), "Invalid model strings!"
    print(f"Verification PASSED for {parquet_path}!")
    print("Class distribution:\n", df["model"].value_counts())

if __name__ == "__main__":
    main()
