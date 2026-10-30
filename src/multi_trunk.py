"""
Multi-Trunk Dual-T4 Representation Engine (Qwen3-8B + OLMo2-7B + DeBERTa-v3).
Provides safe, memory-sequenced intermediate hidden state pooling across
complementary foundation models within Kaggle Dual T4 (2x16 GB) VRAM constraints.
"""

from dataclasses import dataclass, field
import gc
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrunkConfig:
    name: str
    model_candidates: List[str]
    hidden_dim: int
    target_layers: List[int]
    max_len: int = 384
    pooling: str = "mean"
    device_pref: str = "cuda:0"


# Presets for the 3 complementary trunks
TRUNK_PRESETS: Dict[str, TrunkConfig] = {
    "qwen3_8b": TrunkConfig(
        name="qwen3_8b",
        model_candidates=[
            "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
            "/kaggle/input/qwen-3/transformers/8b-base/1",
        ],
        hidden_dim=4096,
        target_layers=[2, 4, 6, 8, 12, 16, 20, 24],
        max_len=447,
        device_pref="cuda:0"
    ),
    "olmo2_7b": TrunkConfig(
        name="olmo2_7b",
        model_candidates=[
            "/kaggle/input/models/allenai/olmo-2-1124-7b",
            "allenai/OLMo-2-1124-7B",
        ],
        hidden_dim=4096,
        target_layers=[3, 6, 9, 12, 15, 18, 21, 24],
        max_len=384,
        device_pref="cuda:1"
    ),
    "deberta_v3": TrunkConfig(
        name="deberta_v3",
        model_candidates=[
            "/kaggle/input/models/microsoft/deberta-v3-large",
            "microsoft/deberta-v3-large",
        ],
        hidden_dim=1024,
        target_layers=[6, 12, 18, 24],
        max_len=384,
        device_pref="cpu"
    ),
}


def pool_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    target_layers: Sequence[int],
    pooling: str = "mean"
) -> torch.Tensor:
    """
    Applies masked pooling across specified intermediate layers.
    Args:
        hidden_states: tuple/list of hidden states from transformer model (B, S, H).
        attention_mask: (B, S) binary mask (1 for token, 0 for padding).
        target_layers: list of integer layer indices.
        pooling: "mean" or "concat_mean_max".
    Returns:
        (B, num_layers * H) or (B, num_layers * 2 * H) pooled tensor.
    """
    mask_exp = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
    mask_sum = torch.clamp(mask_exp.sum(dim=1), min=1e-8)  # (B, 1)

    layer_pools = []
    for lyr in target_layers:
        s = hidden_states[lyr].float()
        mean_p = (s * mask_exp).sum(dim=1) / mask_sum  # (B, H)

        if pooling == "concat_mean_max":
            max_p = s.masked_fill(mask_exp == 0, -1e4).max(dim=1).values
            layer_pools.extend([mean_p, max_p])
        else:
            layer_pools.append(mean_p)

    return torch.cat(layer_pools, dim=-1)


def cleanup_vram():
    """Forces garbage collection and GPU cache evacuation between trunks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
