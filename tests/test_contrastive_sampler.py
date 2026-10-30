import pytest
import pandas as pd
import numpy as np
from src.config import VALID_18_MODELS

def test_contrastive_batch_composition():
    from src.contrastive_sampler import PromptClusterBatchSampler
    
    # Create synthetic dataset with 5 prompts x 18 models
    records = []
    for p_idx in range(5):
        pid = f"prompt_{p_idx}"
        for m in VALID_18_MODELS:
            records.append({
                "prompt_id": pid,
                "model": m,
                "text": f"Completion for {pid} by {m}"
            })
    df = pd.DataFrame(records)
    
    sampler = PromptClusterBatchSampler(df, batch_prompts=1, shuffle=True)
    assert len(sampler) > 0
    
    for batch_indices in sampler:
        batch_df = df.iloc[batch_indices]
        # Must have exactly 18 distinct models in the batch
        assert len(batch_df["model"].unique()) == 18
        # Must answer the same prompt (when batch_prompts=1)
        assert len(batch_df["prompt_id"].unique()) == 1
