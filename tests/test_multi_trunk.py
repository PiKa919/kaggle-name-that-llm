import numpy as np
import pytest
import torch
from src.multi_trunk import TrunkConfig, pool_hidden_states, TRUNK_PRESETS


def test_pool_hidden_states_mean():
    B, S, H = 2, 8, 32
    # Create 4 dummy hidden states layers
    hs = [torch.randn(B, S, H) for _ in range(4)]
    
    # Mask where sample 0 has length 6, sample 1 has length 4
    mask = torch.zeros(B, S)
    mask[0, :6] = 1.0
    mask[1, :4] = 1.0
    
    target_layers = [1, 3]
    pooled = pool_hidden_states(hs, mask, target_layers, pooling="mean")
    
    assert pooled.shape == (B, len(target_layers) * H)
    assert not torch.isnan(pooled).any()
    assert not torch.isinf(pooled).any()


def test_trunk_presets():
    assert "qwen3_8b" in TRUNK_PRESETS
    assert "olmo2_7b" in TRUNK_PRESETS
    assert "deberta_v3" in TRUNK_PRESETS
    
    qwen = TRUNK_PRESETS["qwen3_8b"]
    assert len(qwen.target_layers) == 8
    assert qwen.hidden_dim == 4096
