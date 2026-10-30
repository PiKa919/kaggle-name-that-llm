"""
Contra-X Contrastive Style vs. Topic Disentanglement Loss (arxiv:2209.11887)
and Sequence Truncation Augmentation Engine.

Enforces that representations of the same model answering different prompts
are pulled close together, while representations of different models answering
the exact same prompt are pushed apart.
"""

import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def random_truncation_augmentation(
    text: str,
    min_words: int = 40,
    max_words: int = 75,
    p: float = 0.30,
    seed: Optional[int] = None
) -> str:
    """
    Randomly crops passages to 40-75 words during training to force representation
    layers to learn short-passage stylometric fingerprints.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
        
    words = text.split()
    if len(words) <= min_words or rng.random() > p:
        return text
        
    target_len = rng.randint(min_words, min(max_words, len(words)))
    max_start = len(words) - target_len
    start_idx = rng.randint(0, max_start)
    return " ".join(words[start_idx : start_idx + target_len])


class ContraXStyleLoss(nn.Module):
    """
    Supervised Contrastive Loss with Prompt-Disentanglement:
    - Positive pairs: Same model, different prompts (m_i == m_j, p_i != p_j).
    - Hard Negative pairs: Different models, same prompt (m_i != m_j, p_i == p_j).
    """
    def __init__(self, temperature: float = 0.07, hard_negative_weight: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prompt_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) normalized feature vectors.
            labels: (B,) integer model class indices (0..17).
            prompt_ids: (B,) integer or tensor prompt identifiers.
        """
        device = embeddings.device
        B = embeddings.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Normalize embeddings to unit sphere
        z = F.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity matrix (B, B)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_exp = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())

        # Pairwise equality masks
        same_model = (labels.unsqueeze(1) == labels.unsqueeze(0))
        same_prompt = (prompt_ids.unsqueeze(1) == prompt_ids.unsqueeze(0))

        # Identity mask (diagonal)
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)

        # Positive pairs: same model, different prompts
        pos_mask = same_model & (~same_prompt) & (~diag_mask)

        # Hard negative pairs: different models, same prompt
        hard_neg_mask = (~same_model) & same_prompt & (~diag_mask)
        # Standard negatives: different models, different prompts
        std_neg_mask = (~same_model) & (~same_prompt) & (~diag_mask)

        # Weight hard negatives
        denom_weights = torch.ones_like(sim_exp)
        denom_weights[hard_neg_mask] = self.hard_negative_weight
        denom_weights[diag_mask] = 0.0

        # Compute loss for anchors with at least one positive
        pos_counts = pos_mask.sum(dim=1)
        valid_anchors = (pos_counts > 0)

        if not valid_anchors.any():
            # Fallback if no cross-prompt pairs in batch: use standard cross-entropy surrogate
            return torch.tensor(0.0, device=device, requires_grad=True)

        denom = (sim_exp * denom_weights).sum(dim=1) + 1e-12
        pos_sum = (sim_exp * pos_mask.float()).sum(dim=1) + 1e-12

        losses = -torch.log(pos_sum / denom)
        return losses[valid_anchors].mean()
