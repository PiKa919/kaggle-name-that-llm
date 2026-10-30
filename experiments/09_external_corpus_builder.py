"""
Experiment 09: External Multi-Model Corpus Builder
Curates, standardizes, validates, and filters external completions across all 18 candidate LLMs.
Enforces competition invariants:
1. Strictly 1 single paragraph (no double newlines)
2. 40 to 250 words per completion
3. Exactly valid candidate model identifier in VALID_18_MODELS
4. Zero loss of casing, dashes, curly/straight quotes, and punctuation

Saves clean balanced pool to /kaggle/working/data/curated_pool_v2.parquet.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import VALID_18_MODELS, TRAIN_CSV
from src.data_harvest import validate_external_passage, clean_passage, filter_and_standardize_corpus


def build_curated_corpus():
    os.makedirs("/kaggle/working/data", exist_ok=True)
    
    candidates = []
    
    # 1. Primary Source: nishantdahal/ntl-two-stage-lora pool_v1.parquet
    primary_paths = [
        "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/pool_v1.parquet",
        "/kaggle/input/ntl-two-stage-lora/pool_v1.parquet",
    ]
    primary_file = None
    for p in primary_paths:
        if os.path.exists(p):
            primary_file = p
            break
            
    if not primary_file:
        print("Primary path not found directly. Downloading via kagglehub...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("nishantdahal/ntl-two-stage-lora")
            for f in glob.glob(f"{path}/**/*.parquet", recursive=True):
                if "pool" in f:
                    primary_file = f
                    break
        except Exception as e:
            print(f"kagglehub download error: {e}")
            
    if primary_file and os.path.exists(primary_file):
        print(f"Loading primary pool from {primary_file}...")
        df_p = pd.read_parquet(primary_file)
        print(f"Raw primary pool shape: {df_p.shape}, columns: {list(df_p.columns)}")
        
        text_col = "text" if "text" in df_p.columns else [c for c in df_p.columns if "text" in c.lower() or "passage" in c.lower()][0]
        model_col = "model" if "model" in df_p.columns else [c for c in df_p.columns if "model" in c.lower() or "label" in c.lower()][0]
        prompt_col = "prompt_id" if "prompt_id" in df_p.columns else None
        
        filtered_p = filter_and_standardize_corpus(df_p, text_col=text_col, model_col=model_col, prompt_col=prompt_col)
        print(f"Valid passages retained from primary pool: {len(filtered_p)}")
        candidates.append(filtered_p)
        
    # 2. Additional Source: Check for any other mounted parquet or csv in /kaggle/input
    extra_files = glob.glob("/kaggle/input/**/*.parquet", recursive=True) + glob.glob("/kaggle/input/**/*.csv", recursive=True)
    for ef in extra_files:
        if "competitions" in ef or (primary_file and ef == primary_file):
            continue
        print(f"Checking potential external source: {ef}")
        try:
            df_extra = pd.read_parquet(ef) if ef.endswith(".parquet") else pd.read_csv(ef)
            t_cols = [c for c in df_extra.columns if "text" in c.lower() or "passage" in c.lower()]
            m_cols = [c for c in df_extra.columns if "model" in c.lower() or "label" in c.lower()]
            if t_cols and m_cols:
                filtered_extra = filter_and_standardize_corpus(df_extra, text_col=t_cols[0], model_col=m_cols[0])
                if len(filtered_extra) > 100:
                    print(f"Retained {len(filtered_extra)} valid passages from {ef}")
                    candidates.append(filtered_extra)
        except Exception as e:
            pass
            
    # Combine all candidates
    if not candidates:
        raise RuntimeError("No valid external datasets could be found or loaded!")
        
    combined_df = pd.concat(candidates, ignore_index=True)
    print(f"\nTotal collected external candidates before deduplication: {len(combined_df)}")
    
    # Deduplicate on text
    combined_df = combined_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Total unique external passages: {len(combined_df)}")
    
    # Check model representation breakdown
    print("\n--- Model Representation Breakdown ---")
    model_counts = combined_df["model"].value_counts()
    for m in VALID_18_MODELS:
        cnt = model_counts.get(m, 0)
        print(f"  {m:25s}: {cnt:5d} passages")
        
    # Assign consistent unique prompt_ids if missing or non-uniform
    if "prompt_id" not in combined_df.columns or combined_df["prompt_id"].isnull().any():
        combined_df["prompt_id"] = [f"ext_p_{i // 18:05d}" for i in range(len(combined_df))]
        
    out_path = "/kaggle/working/data/curated_pool_v2.parquet"
    combined_df.to_parquet(out_path, index=False)
    print(f"\nSuccessfully wrote curated pool to {out_path} ({len(combined_df)} rows)")
    
    # Print length distribution summary
    lens = combined_df["text"].apply(lambda t: len(t.split()))
    print(f"Word length stats: Min={lens.min()}, Max={lens.max()}, Mean={lens.mean():.1f}, Median={lens.median():.1f}")
    
    return out_path


if __name__ == "__main__":
    build_curated_corpus()
