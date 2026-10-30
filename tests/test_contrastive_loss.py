import numpy as np
import pytest
import torch
import torch.nn as nn
from src.contrastive_loss import ContraXStyleLoss, random_truncation_augmentation


def test_random_truncation_augmentation():
    text = " ".join([f"word_{i}" for i in range(120)])
    # With p=1.0, it must truncate to between 40 and 75 words
    truncated = random_truncation_augmentation(text, min_words=40, max_words=75, p=1.0, seed=42)
    words = truncated.split()
    assert 40 <= len(words) <= 75
    # With p=0.0, it must leave text unchanged
    unchanged = random_truncation_augmentation(text, p=0.0)
    assert len(unchanged.split()) == 120


def test_contrax_style_loss_convergence():
    loss_fn = ContraXStyleLoss(temperature=0.1)
    
    # 4 samples:
    # Sample 0: Model 0, Prompt A
    # Sample 1: Model 0, Prompt B (Positive to Sample 0)
    # Sample 2: Model 1, Prompt A (Hard Negative to Sample 0: same prompt, different model)
    # Sample 3: Model 1, Prompt B
    y_models = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    p_prompts = torch.tensor([101, 102, 101, 102], dtype=torch.long)
    
    # Well-aligned embeddings: Model 0 embeddings close, Model 1 embeddings close
    Z_good = torch.tensor([
        [1.0, 0.0],
        [0.95, 0.05],
        [0.0, 1.0],
        [0.05, 0.95],
    ], dtype=torch.float32)
    Z_good = nn.functional.normalize(Z_good, p=2, dim=1)
    loss_good = loss_fn(Z_good, y_models, p_prompts)
    
    # Misaligned embeddings: Model 0 confounded with prompt A
    Z_bad = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.95, 0.05],
        [0.05, 0.95],
    ], dtype=torch.float32)
    Z_bad = nn.functional.normalize(Z_bad, p=2, dim=1)
    loss_bad = loss_fn(Z_bad, y_models, p_prompts)
    
    assert loss_good < loss_bad, f"Expected loss_good ({loss_good.item()}) < loss_bad ({loss_bad.item()})"
    assert loss_good.item() >= 0.0
