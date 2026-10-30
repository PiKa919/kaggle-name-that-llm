"""
PromptClusterBatchSampler: PyTorch-compatible batch sampler that groups
completions by prompt_id so every batch contains all 18 candidate models
answering the exact same prompt (hard negative contrastive batching).
"""

import numpy as np
import pandas as pd
from typing import Iterator, List


class PromptClusterBatchSampler:
    """
    Yields batches of sample indices such that each batch contains all candidate
    models answering the same prompt(s). This allows contrastive and cross-entropy
    losses to compare sibling models on identical semantic conditions.
    """
    def __init__(self, df: pd.DataFrame, batch_prompts: int = 1, shuffle: bool = True, seed: int = 42):
        self.df = df
        self.batch_prompts = batch_prompts
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        
        # Group indices by prompt_id
        self.prompt_to_indices = {}
        for idx, pid in enumerate(df["prompt_id"].values):
            if pid not in self.prompt_to_indices:
                self.prompt_to_indices[pid] = []
            self.prompt_to_indices[pid].append(idx)
            
        self.unique_prompts = list(self.prompt_to_indices.keys())
        
    def __len__(self) -> int:
        return (len(self.unique_prompts) + self.batch_prompts - 1) // self.batch_prompts
        
    def __iter__(self) -> Iterator[List[int]]:
        prompts = list(self.unique_prompts)
        if self.shuffle:
            self.rng.shuffle(prompts)
            
        for i in range(0, len(prompts), self.batch_prompts):
            batch_pids = prompts[i:i + self.batch_prompts]
            batch_indices = []
            for pid in batch_pids:
                batch_indices.extend(self.prompt_to_indices[pid])
            yield batch_indices
