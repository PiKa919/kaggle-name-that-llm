"""
Multi-Trunk Representation Fusion Engine for "Name That LLM" (0.70+ Target).
Sequences Qwen3-8B + Qwen3-14B + OLMo2-7B with strict VRAM hygiene on Kaggle Dual Nvidia Tesla T4.

Architecture & Principles:
1. Multi-Backbone Lineage Diversity:
   - Trunk 1: Qwen3-8B (Alibaba pretraining, LoRA v1 prefix 36, pooled layers [2, 4, 6, 8, 12, 16, 20, 24])
   - Trunk 2: Qwen3-14B (Alibaba pretraining, LoRA v1 prefix 40 partitioned across Dual T4 via device_map='auto')
   - Trunk 3: OLMo2-7B (AllenAI pretraining, independent lineage, native-24 prefix with RMSNorm from lightningv08)
2. Attention-Sink Avoidance & Multi-Layer Pooling:
   - First token (index 0) attention sink is excluded: mask[:, 0] = 0.
   - Mean pooling over unmasked token hidden states followed by layer-specific LayerNorm.
   - Automatic FP16 overflow retry in bfloat16.
3. Absolute VRAM Hygiene on Dual T4 (2x16 GB = 32 GB):
   - Backbones are loaded and evaluated sequentially.
   - Explicit module deletion, gc.collect(), and torch.cuda.empty_cache() between trunks.
   - Peak memory never exceeds 28 GiB across both GPUs (13 GiB + 13 GiB partition for 14B).
4. Label Alignment:
   - OLMo2 checkpoint uses alphabetical class order; Qwen3 checkpoints use competition order.
   - Automatic projection to canonical label ordering.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Canonical competition label ordering
CANONICAL_LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"
]
K_CLASSES = len(CANONICAL_LABELS)

# OLMo2 assets alphabetical class ordering
OLMO2_CLASSES = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]

# Mapping indices: OLMo2 -> Canonical
REORDER_OLMO_TO_CANONICAL = np.asarray(
    [OLMO2_CLASSES.index(lbl) for lbl in CANONICAL_LABELS], dtype=int
)


def align_to_canonical(
    arr: np.ndarray,
    source_classes: Sequence[str],
    target_classes: Sequence[str] = CANONICAL_LABELS
) -> np.ndarray:
    """
    Permutes columns of arr (N, C) from source_classes to target_classes.
    """
    assert arr.ndim == 2, f"Expected 2D array, got shape {arr.shape}"
    assert len(source_classes) == arr.shape[1], (
        f"source_classes count ({len(source_classes)}) != arr columns ({arr.shape[1]})"
    )
    if list(source_classes) == list(target_classes):
        return arr.copy()
    idx = [list(source_classes).index(lbl) for lbl in target_classes]
    return arr[:, idx]


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    log_sum_exp = np.log(np.sum(exp_x, axis=axis, keepdims=True)) + x_max
    return x - log_sum_exp


# =====================================================================
# VRAM HYGIENE & RESOURCE MANAGEMENT
# =====================================================================

def cleanup_vram() -> Dict[int, Dict[str, float]]:
    """
    Forces immediate garbage collection and drains CUDA cache across all devices.
    Returns memory stats in MiB.
    """
    gc.collect()
    stats: Dict[int, Dict[str, float]] = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            stats[i] = {
                "allocated_mib": torch.cuda.memory_allocated(i) / (1024 ** 2),
                "reserved_mib": torch.cuda.memory_reserved(i) / (1024 ** 2),
                "max_reserved_mib": torch.cuda.max_memory_reserved(i) / (1024 ** 2),
            }
    return stats


def assert_vram_hygiene(max_reserved_gib: float = 1.0) -> None:
    """
    Asserts that CUDA memory is effectively zeroed between trunks.
    Prevents silent memory leaks leading to OOM on subsequent backbones.
    """
    if not torch.cuda.is_available():
        return
    cleanup_vram()
    for i in range(torch.cuda.device_count()):
        reserved_gib = torch.cuda.memory_reserved(i) / (1024 ** 3)
        assert reserved_gib <= max_reserved_gib, (
            f"VRAM leak detected on GPU {i}: {reserved_gib:.2f} GiB reserved "
            f"(threshold: {max_reserved_gib:.2f} GiB)"
        )


@contextmanager
def trunk_vram_scope(trunk_name: str, verbose: bool = True):
    """
    Context manager guaranteeing model teardown and VRAM reclamation.
    """
    if verbose:
        print(f"\n--- [TRUNK LIFECYCLE: {trunk_name}] Initializing ---", flush=True)
    t0 = time.time()
    cleanup_vram()
    try:
        yield
    finally:
        cleanup_vram()
        elapsed = time.time() - t0
        if verbose:
            mem_info = ""
            if torch.cuda.is_available():
                usages = [f"GPU{i}:{torch.cuda.memory_reserved(i)/(1024**2):.1f}MB"
                          for i in range(torch.cuda.device_count())]
                mem_info = f" | VRAM: {', '.join(usages)}"
            print(f"--- [TRUNK LIFECYCLE: {trunk_name}] Cleaned up in {elapsed:.1f}s{mem_info} ---", flush=True)


# =====================================================================
# CONFIGURATION & DATA STRUCTURES
# =====================================================================

@dataclass
class TrunkConfig:
    name: str
    model_type: str                   # 'qwen3', 'olmo2', etc.
    hidden_size: int
    keep_layers: int
    pool_layers: List[int]
    batch_size: int = 8
    max_len: int = 447
    w_plain: float = 0.20
    w_twin: float = 0.08
    device_strategy: str = "cuda0"    # 'cuda0', 'dual_t4_balanced', 'cpu'
    adapter_patterns: List[str] = field(default_factory=list)
    base_spec: Dict[str, Any] = field(default_factory=dict)
    extra_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrunkExecutionResult:
    name: str
    plain_logp: np.ndarray             # (N, K_CLASSES) in CANONICAL_LABELS order
    plain_prob: np.ndarray             # (N, K_CLASSES)
    twin_logp: Optional[np.ndarray] = None    # (N_twin, K_CLASSES)
    twin_prob: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None     # (N, D_pooled)
    train_features: Optional[np.ndarray] = None
    elapsed_seconds: float = 0.0
    peak_vram_mib: float = 0.0
    status: str = "OK"


@dataclass
class MultiBackboneEnsembleResult:
    fused_logits: np.ndarray           # (N, K_CLASSES)
    fused_probs: np.ndarray            # (N, K_CLASSES)
    trunk_results: Dict[str, TrunkExecutionResult]
    fused_features: Optional[np.ndarray] = None
    total_elapsed_seconds: float = 0.0
    peak_vram_gib: float = 0.0


# Standard presets for the 3 trunks
TRUNK_PRESETS: Dict[str, TrunkConfig] = {
    "qwen3_8b": TrunkConfig(
        name="qwen3_8b",
        model_type="qwen3",
        hidden_size=4096,
        keep_layers=36,
        pool_layers=[2, 4, 6, 8, 12, 16, 20, 24],
        batch_size=16,
        max_len=447,
        w_plain=0.226,
        w_twin=0.079,
        device_strategy="cuda0",
        adapter_patterns=["*lora8b_v1*"],
        base_spec={"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36},
    ),
    "qwen3_14b": TrunkConfig(
        name="qwen3_14b",
        model_type="qwen3",
        hidden_size=5120,
        keep_layers=40,
        pool_layers=[2, 4, 6, 8, 12, 16, 20, 24],
        batch_size=8,
        max_len=447,
        w_plain=0.429,
        w_twin=0.150,
        device_strategy="dual_t4_balanced",
        adapter_patterns=["*lora14b_v1*"],
        base_spec={"model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40},
    ),
    "olmo2_7b": TrunkConfig(
        name="olmo2_7b",
        model_type="olmo2",
        hidden_size=4096,
        keep_layers=24,
        pool_layers=[4, 8, 12, 16, 20, 24],
        batch_size=8,
        max_len=446,
        w_plain=0.030,
        w_twin=0.030,
        device_strategy="cuda0",
        adapter_patterns=[
            "*ntl-olmo2-native24-final-inference-assets*",
            "*olmo2*native24*",
        ],
        base_spec={"model_type": "olmo2", "hidden_size": 4096, "num_hidden_layers": 32},
    ),
}


# =====================================================================
# ATTENTION-SINK POOLING & FEATURE EXTRACTION
# =====================================================================

def pool_intermediate_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    target_layers: Sequence[int],
    layer_norms: Optional[nn.ModuleList] = None,
    avoid_attention_sink: bool = True,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Applies masked mean pooling across specified intermediate layers.
    Excludes the first token (attention sink token) from the mean:
        mask[:, 0] = 0.0
    Then applies layer-specific LayerNorm if provided.

    Args:
        hidden_states: Sequence of layer representations from transformer (B, S, H).
        attention_mask: (B, S) binary mask (1 for valid token, 0 for padding).
        target_layers: Indices of intermediate layers to pool.
        layer_norms: Optional ModuleList of LayerNorm modules matching len(target_layers).
        avoid_attention_sink: If True, sets attention_mask[:, 0] = 0.0.
        device: Target device for output tensor.

    Returns:
        Tensor of shape (B, len(target_layers) * H).
    """
    dev = device if device is not None else hidden_states[0].device
    m = attention_mask.unsqueeze(-1).float().to(dev)  # (B, S, 1)

    if avoid_attention_sink and m.shape[1] > 1:
        m = m.clone()
        m[:, 0] = 0.0

    den = m.sum(dim=1).clamp(min=1.0)  # (B, 1)

    pieces = []
    for i, layer_idx in enumerate(target_layers):
        hs = hidden_states[layer_idx].to(dev).float()  # (B, S, H)
        pooled_mean = (hs * m).sum(dim=1) / den        # (B, H)

        if layer_norms is not None and i < len(layer_norms):
            norm_mod = layer_norms[i].to(dev).float()
            normed = norm_mod(pooled_mean)
            pieces.append(normed)
        else:
            pieces.append(pooled_mean)

    return torch.cat(pieces, dim=-1)


def find_file_or_dir(pattern: str, search_roots: Optional[List[str]] = None) -> Optional[str]:
    """Robust glob search across input locations."""
    if search_roots is None:
        search_roots = ["/kaggle/input", "."]
    for root in search_roots:
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if hits:
            return hits[0]
    return None


def find_base_checkpoint(spec: Dict[str, Any], search_roots: Optional[List[str]] = None) -> Optional[str]:
    """Finds base model directory matching config.json keys."""
    if search_roots is None:
        search_roots = ["/kaggle/input", "."]
    for root in search_roots:
        for cfg_path in sorted(glob.glob(f"{root}/**/config.json", recursive=True)):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if all(cfg.get(k) == v for k, v in spec.items()):
                    return os.path.dirname(cfg_path)
            except Exception:
                continue
    return None


# =====================================================================
# IN-PLACE LORA WEIGHT MERGE (ZERO-PEFT OVERHEAD)
# =====================================================================

@torch.no_grad()
def merge_lora_into_model(model: nn.Module, adapter_dir: str) -> None:
    """
    Merges LoRA adapter weights directly into base model weights:
        W = W + (B @ A) * (lora_alpha / r)
    Preserves multi-GPU layer placement when model is distributed across devices.
    """
    from safetensors.torch import load_file

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        acfg = json.load(f)
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])

    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_file(weights_path)
    modules = dict(model.named_modules())

    for k, A in sd.items():
        if ".lora_A." in k:
            name = k.split(".lora_A.")[0]
            # Strip prefix to match model layer named_modules
            layer_key = name[name.index("layers."):] if "layers." in name else name
            if layer_key not in modules:
                # Try finding without prefix
                matches = [m for m_name, m in modules.items() if m_name.endswith(layer_key)]
                if matches:
                    lin = matches[0]
                else:
                    continue
            else:
                lin = modules[layer_key]

            Bm = sd[k.replace(".lora_A.", ".lora_B.")]
            target_device = lin.weight.device
            target_dtype = lin.weight.dtype

            delta = (Bm.float().to(target_device) @ A.float().to(target_device)) * scale
            lin.weight.add_(delta.to(target_dtype))


# =====================================================================
# TRUNK 1 & 2: QWEN3 PIPELINE EXECUTOR
# =====================================================================

def execute_qwen_trunk(
    config: TrunkConfig,
    test_texts: List[str],
    train_texts: Optional[List[str]] = None,
    groups: Optional[np.ndarray] = None,
    twin_rows: Optional[np.ndarray] = None,
    verbose: bool = True
) -> TrunkExecutionResult:
    """
    Executes a Qwen3 decoder trunk (8B or 14B) with:
    1. Truncated prefix loading (36 or 40 layers)
    2. LoRA weight merging
    3. Multi-layer intermediate hidden state pooling with attention-sink avoidance
    4. Plain head evaluation
    5. Centered twin evaluation on topic clusters
    6. Strict VRAM cleanup upon return
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    N = len(test_texts)
    # Locate base model
    base_dir = find_base_checkpoint(config.base_spec)
    if not base_dir:
        raise FileNotFoundError(f"Base model not found for {config.name}: {config.base_spec}")

    # Locate adapter
    adapter_meta = None
    for pattern in config.adapter_patterns:
        adapter_meta = find_file_or_dir(f"{pattern}/meta.json")
        if adapter_meta:
            break
        adapter_meta = find_file_or_dir(f"{pattern}/**/meta.json")
        if adapter_meta:
            break

    if not adapter_meta:
        raise FileNotFoundError(f"LoRA adapter metadata not found for {config.name}")

    ckpt_dir = os.path.dirname(adapter_meta)
    with open(adapter_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Checkpoint configuration
    cfg = meta.get("cfg", {})
    keep_layers = cfg.get("keep", config.keep_layers)
    target_layers = cfg.get("layers", config.pool_layers)
    max_len = cfg.get("max_len", config.max_len)
    batch_size = config.batch_size

    if verbose:
        print(f"[{config.name}] Loading base from {base_dir} (keep={keep_layers})...", flush=True)

    # Model configuration & memory mapping
    mc = AutoConfig.from_pretrained(base_dir)
    mc.num_hidden_layers = keep_layers
    mc.use_cache = False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[:keep_layers]

    # Select precision
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16

    # Dual T4 allocation logic
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if config.device_strategy == "dual_t4_balanced" and device_count > 1:
        # Enforce strict 13 GiB limit per T4 to never breach 28 GiB total
        mem = {i: "13GiB" for i in range(device_count)}
        model = AutoModel.from_pretrained(
            base_dir, config=mc, dtype=dtype, device_map="auto", max_memory=mem
        ).eval()
    elif torch.cuda.is_available():
        model = AutoModel.from_pretrained(
            base_dir, config=mc, dtype=dtype
        ).to("cuda:0").eval()
    else:
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=torch.float32).eval()

    model.norm = nn.Identity()

    # Merge LoRA
    adapter_path = os.path.join(ckpt_dir, "adapter")
    if os.path.isdir(adapter_path):
        if verbose:
            print(f"[{config.name}] Merging LoRA from {adapter_path}...", flush=True)
        merge_lora_into_model(model, adapter_path)

    # Load heads and LayerNorms
    heads_path = os.path.join(ckpt_dir, "heads.safetensors")
    heads_sd = load_file(heads_path)

    H = model.config.hidden_size
    head_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in target_layers]).to(head_device).float()
    norms.load_state_dict({k[len("norms."):]: v for k, v in heads_sd.items() if k.startswith("norms.")})
    norms.eval()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build sequence list: optionally combine test + train for single forward pass
    passages_to_run = test_texts.copy()
    has_train = (train_texts is not None and len(train_texts) > 0)
    if has_train:
        passages_to_run += train_texts

    # Tokenize
    raw_ids = tok(
        passages_to_run,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len
    )["input_ids"]
    all_token_ids = [ids + [tok.eos_token_id] for ids in raw_ids]

    # Order by length for efficient batch padding
    order = np.argsort([len(x) for x in all_token_ids], kind="stable")
    total_dim = H * len(target_layers)
    all_Z = np.zeros((len(all_token_ids), total_dim), dtype=np.float32)

    primary_device = next(model.parameters()).device
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok.pad({"input_ids": [all_token_ids[i] for i in idx]}, return_tensors="pt").to(primary_device)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states

            # Check FP16 overflow
            if not all(torch.isfinite(hs[l]).all() for l in target_layers):
                model.to(torch.bfloat16)
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                model.to(dtype)

            pooled = pool_intermediate_hidden_states(
                hidden_states=hs,
                attention_mask=enc["attention_mask"],
                target_layers=target_layers,
                layer_norms=norms,
                avoid_attention_sink=True,
                device=head_device
            )
            all_Z[idx] = pooled.cpu().numpy()

    # Partition test and train representations
    Z_test = all_Z[:N]
    Z_train = all_Z[N:] if has_train else None

    # Compute plain probabilities
    plain_w = heads_sd["plain.weight"].float().numpy()
    plain_b = heads_sd["plain.bias"].float().numpy()
    plain_logits = Z_test @ plain_w.T + plain_b
    plain_prob = softmax_stable(plain_logits, axis=1)
    plain_logp = log_softmax_stable(plain_logits, axis=1)

    # Compute centered twin probabilities if clusters available
    twin_prob = None
    twin_logp = None
    if twin_rows is not None and twin_rows.any() and groups is not None and "centered.weight" in heads_sd:
        centered_w = heads_sd["centered.weight"].float().numpy()
        centered_b = heads_sd["centered.bias"].float().numpy()

        Zc = Z_test.copy()
        for g in np.unique(groups[twin_rows]):
            g_mask = (groups == g)
            Zc[g_mask] -= Zc[g_mask].mean(axis=0)

        twin_logits = Zc[twin_rows] @ centered_w.T + centered_b
        twin_prob = softmax_stable(twin_logits, axis=1)
        twin_logp = log_softmax_stable(twin_logits, axis=1)

    # Teardown model and tensors
    del model, tok, norms, heads_sd, all_Z
    cleanup_vram()

    return TrunkExecutionResult(
        name=config.name,
        plain_logp=plain_logp,
        plain_prob=plain_prob,
        twin_logp=twin_logp,
        twin_prob=twin_prob,
        features=Z_test,
        train_features=Z_train,
        status="OK"
    )


# =====================================================================
# TRUNK 3: OLMO2-7B NATIVE-24 PIPELINE EXECUTOR
# =====================================================================

def execute_olmo2_trunk(
    config: TrunkConfig,
    test_texts: List[str],
    groups: Optional[np.ndarray] = None,
    twin_rows: Optional[np.ndarray] = None,
    verbose: bool = True
) -> TrunkExecutionResult:
    """
    Executes AllenAI OLMo2-7B native-24 prefix trunk from lightningv08 assets:
    1. Loads Olmo2Model prefix (retained_config.num_hidden_layers = 24).
    2. Attaches LoRA adapter (r=8, alpha=16) and LayerNorms for layers [4, 8, 12, 16, 20, 24].
    3. Excludes attention-sink token (mask[:, 0] = 0).
    4. Projects predictions into CANONICAL_LABELS order.
    5. Releases all GPU memory completely upon return.
    """
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Olmo2Config, Olmo2Model

    # Suppress torchao conflict in PEFT if present
    try:
        import peft.import_utils as peft_imports
        peft_imports.is_torchao_available = lambda *a, **kw: False
    except Exception:
        pass

    N = len(test_texts)
    base_dir = find_base_checkpoint(config.base_spec)
    if not base_dir:
        raise FileNotFoundError(f"OLMo2 base model not found: {config.base_spec}")

    # Locate assets directory
    asset_root = None
    for pattern in config.adapter_patterns:
        manifest = find_file_or_dir(f"{pattern}/asset_manifest.json")
        if manifest:
            asset_root = os.path.dirname(manifest)
            break
        manifest = find_file_or_dir(f"{pattern}/**/asset_manifest.json")
        if manifest:
            asset_root = os.path.dirname(manifest)
            break

    if not asset_root:
        raise FileNotFoundError("OLMo2 native24 inference assets directory not found")

    if verbose:
        print(f"[{config.name}] Loading native-24 prefix from {base_dir}...", flush=True)

    # 1. Truncate Olmo2Config to 24 layers
    original_cfg = Olmo2Config.from_pretrained(base_dir, local_files_only=True)
    retained_cfg = Olmo2Config.from_dict(original_cfg.to_dict())
    retained_cfg.num_hidden_layers = config.keep_layers  # 24

    # 2. Instantiate prefix model on single GPU (cuda:0)
    primary_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_map = {"": 0} if torch.cuda.is_available() else None

    base_model = Olmo2Model.from_pretrained(
        base_dir,
        config=retained_cfg,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        attn_implementation="eager" if torch.cuda.is_available() else "sdpa",
        local_files_only=True
    ).eval()
    base_model.requires_grad_(False)

    # 3. Attach LoRA adapter
    from peft import PeftModel
    adapter_dir = os.path.join(asset_root, "adapter")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
        local_files_only=True
    ).eval()

    # 4. Load LayerNorms and Readouts
    heads_path = os.path.join(asset_root, "heads.safetensors")
    heads_sd = load_file(heads_path)

    target_layers = config.pool_layers  # [4, 8, 12, 16, 20, 24]
    norms = nn.ModuleList([nn.LayerNorm(config.hidden_size) for _ in target_layers]).to(primary_device).float()
    norms.load_state_dict({k.removeprefix("norms."): v for k, v in heads_sd.items() if k.startswith("norms.")})
    norms.eval()
    norms.requires_grad_(False)

    readouts_path = os.path.join(asset_root, "readouts.npz")
    with np.load(readouts_path, allow_pickle=False) as ro:
        olmo_classes = ro["classes"].tolist()
        plain_coef = ro["plain_coef"].copy()          # (18, 24576)
        plain_intercept = ro["plain_intercept"].copy()# (18,)
        center_coef = ro["center_coef"].copy()        # (18, 24576)
        center_intercept = ro["center_intercept"].copy()

    # 5. Tokenizer & Token framing: [100257] + content + [100257]
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True, local_files_only=True)
    tok.padding_side = "right"
    content = tok(test_texts, add_special_tokens=False, truncation=True, max_length=config.max_len)["input_ids"]
    tokens = [[100257] + row + [100257] for row in content]

    # 6. Inference loop
    order = np.argsort([len(row) for row in tokens], kind="stable")
    feature_dim = config.hidden_size * len(target_layers)
    features = np.zeros((N, feature_dim), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, len(order), config.batch_size):
            idx = order[s:s + config.batch_size]
            batch = tok.pad({"input_ids": [tokens[i] for i in idx]}, return_tensors="pt").to(primary_device)
            hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states

            # Pool layers with attention-sink token avoidance
            pooled = pool_intermediate_hidden_states(
                hidden_states=hidden,
                attention_mask=batch["attention_mask"],
                target_layers=target_layers,
                layer_norms=norms,
                avoid_attention_sink=True,
                device=primary_device
            )
            features[idx] = pooled.cpu().numpy()

    # 7. Compute Readouts in OLMo2 class ordering
    raw_plain_logits = features @ plain_coef.T + plain_intercept
    raw_plain_logp = log_softmax_stable(raw_plain_logits, axis=1)
    raw_plain_prob = softmax_stable(raw_plain_logits, axis=1)

    # 8. Align to CANONICAL_LABELS
    plain_logp = align_to_canonical(raw_plain_logp, olmo_classes, CANONICAL_LABELS)
    plain_prob = align_to_canonical(raw_plain_prob, olmo_classes, CANONICAL_LABELS)

    # Centered twin on topic clusters
    twin_logp = None
    twin_prob = None
    if twin_rows is not None and twin_rows.any() and groups is not None:
        features_c = features.copy()
        for g in np.unique(groups[twin_rows]):
            g_mask = (groups == g)
            features_c[g_mask] -= features_c[g_mask].mean(axis=0)

        raw_twin_logits = features_c[twin_rows] @ center_coef.T + center_intercept
        raw_twin_logp = log_softmax_stable(raw_twin_logits, axis=1)
        raw_twin_prob = softmax_stable(raw_twin_logits, axis=1)

        twin_logp = align_to_canonical(raw_twin_logp, olmo_classes, CANONICAL_LABELS)
        twin_prob = align_to_canonical(raw_twin_prob, olmo_classes, CANONICAL_LABELS)

    # Teardown
    del model, base_model, tok, norms, heads_sd
    cleanup_vram()

    return TrunkExecutionResult(
        name=config.name,
        plain_logp=plain_logp,
        plain_prob=plain_prob,
        twin_logp=twin_logp,
        twin_prob=twin_prob,
        features=features,
        status="OK"
    )


# =====================================================================
# SYNTHETIC / SMOKE TRUNK EXECUTOR (OFFLINE MAC / CPU FALLBACK)
# =====================================================================

def execute_mock_trunk(
    config: TrunkConfig,
    test_texts: List[str],
    train_texts: Optional[List[str]] = None,
    groups: Optional[np.ndarray] = None,
    twin_rows: Optional[np.ndarray] = None
) -> TrunkExecutionResult:
    """
    Deterministic CPU mock generator for local tests and smoke validation.
    Generates representations and calibrated log-probabilities without GPU or weights.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    N = len(test_texts)
    total_dim = config.hidden_size * len(config.pool_layers)

    # Deterministic pseudo-features from text n-grams
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=min(128, total_dim))
    X_text = vec.fit_transform(test_texts).toarray().astype(np.float32)

    # Tile to target dimension
    reps = (total_dim + X_text.shape[1] - 1) // X_text.shape[1]
    features = np.tile(X_text, (1, reps))[:, :total_dim]

    # Predictable pseudo-logits
    rng = np.random.RandomState(abs(hash(config.name)) % (2 ** 32))
    W_mock = rng.randn(total_dim, K_CLASSES).astype(np.float32) * 0.05
    logits = features @ W_mock

    plain_prob = softmax_stable(logits, axis=1)
    plain_logp = log_softmax_stable(logits, axis=1)

    twin_prob = None
    twin_logp = None
    if twin_rows is not None and twin_rows.any():
        twin_logits = logits[twin_rows] * 0.85
        twin_prob = softmax_stable(twin_logits, axis=1)
        twin_logp = log_softmax_stable(twin_logits, axis=1)

    train_features = None
    if train_texts is not None and len(train_texts) > 0:
        X_train_text = vec.transform(train_texts).toarray().astype(np.float32)
        train_features = np.tile(X_train_text, (1, reps))[:, :total_dim]

    return TrunkExecutionResult(
        name=config.name,
        plain_logp=plain_logp,
        plain_prob=plain_prob,
        twin_logp=twin_logp,
        twin_prob=twin_prob,
        features=features,
        train_features=train_features,
        status="MOCK_OK"
    )


# =====================================================================
# MULTI-BACKBONE REPRESENTATION FUSION ENGINE
# =====================================================================

class MultiBackboneEngine:
    """
    High-level orchestrator for sequential multi-backbone inference and representation fusion.
    Sequences Qwen3-8B -> Qwen3-14B -> OLMo2-7B with zero VRAM spillover.
    """

    def __init__(
        self,
        configs: Optional[List[TrunkConfig]] = None,
        smoke_mode: bool = False,
        verbose: bool = True
    ):
        self.configs = configs or [
            TRUNK_PRESETS["qwen3_8b"],
            TRUNK_PRESETS["qwen3_14b"],
            TRUNK_PRESETS["olmo2_7b"],
        ]
        self.smoke_mode = smoke_mode
        self.verbose = verbose

    def execute_all_trunks(
        self,
        test_texts: List[str],
        train_texts: Optional[List[str]] = None,
        groups: Optional[np.ndarray] = None,
        twin_rows: Optional[np.ndarray] = None,
        return_fused_features: bool = False
    ) -> MultiBackboneEnsembleResult:
        """
        Executes all configured trunks sequentially with rigorous VRAM hygiene.

        Args:
            test_texts: List of test passage strings.
            train_texts: Optional list of competition training texts (passed to Trunk 1).
            groups: Inferred topic cluster IDs (1-based) from content distance.
            twin_rows: Boolean mask for passages belonging to eligible clusters.
            return_fused_features: If True, concatenates pooled representations.

        Returns:
            MultiBackboneEnsembleResult with blended logits, probabilities, and trunk results.
        """
        N = len(test_texts)
        fused_logits = np.zeros((N, K_CLASSES), dtype=np.float64)
        trunk_results: Dict[str, TrunkExecutionResult] = {}
        all_features: List[np.ndarray] = []
        t0 = time.time()

        for cfg in self.configs:
            with trunk_vram_scope(cfg.name, verbose=self.verbose):
                res: Optional[TrunkExecutionResult] = None

                # Check if real execution is possible or fallback to mock
                is_mock = self.smoke_mode or not torch.cuda.is_available()

                if not is_mock:
                    try:
                        if cfg.model_type == "olmo2":
                            res = execute_olmo2_trunk(
                                config=cfg,
                                test_texts=test_texts,
                                groups=groups,
                                twin_rows=twin_rows,
                                verbose=self.verbose
                            )
                        else:
                            # Qwen3-8B or Qwen3-14B
                            # Only Trunk 1 (8B) receives train texts to save compute
                            pass_train = train_texts if (cfg.name == "qwen3_8b") else None
                            res = execute_qwen_trunk(
                                config=cfg,
                                test_texts=test_texts,
                                train_texts=pass_train,
                                groups=groups,
                                twin_rows=twin_rows,
                                verbose=self.verbose
                            )
                    except (FileNotFoundError, ImportError, RuntimeError) as e:
                        if self.verbose:
                            print(f"[{cfg.name}] Real execution unavailable ({e}); falling back to mock.", flush=True)
                        is_mock = True

                if is_mock:
                    pass_train = train_texts if (cfg.name == "qwen3_8b") else None
                    res = execute_mock_trunk(
                        config=cfg,
                        test_texts=test_texts,
                        train_texts=pass_train,
                        groups=groups,
                        twin_rows=twin_rows
                    )

                assert res is not None, f"Execution failed for {cfg.name}"
                trunk_results[cfg.name] = res

                # 1. Accumulate plain log-probabilities
                # log p = log(clip(P_plain, 1e-9, 1.0))
                p_plain = np.clip(res.plain_prob, 1e-9, 1.0)
                fused_logits += cfg.w_plain * np.log(p_plain)

                # 2. Accumulate centered twin log-probabilities on topic clusters
                if (twin_rows is not None and twin_rows.any() and
                    res.twin_prob is not None and cfg.w_twin > 0):
                    p_twin = np.clip(res.twin_prob, 1e-9, 1.0)
                    fused_logits[twin_rows] += cfg.w_twin * np.log(p_twin)

                if return_fused_features and res.features is not None:
                    all_features.append(res.features)

        # Final probabilities
        fused_probs = softmax_stable(fused_logits, axis=1)
        fused_reps = np.concatenate(all_features, axis=1) if (return_fused_features and all_features) else None

        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) / (1024 ** 3)

        return MultiBackboneEnsembleResult(
            fused_logits=fused_logits,
            fused_probs=fused_probs,
            trunk_results=trunk_results,
            fused_features=fused_reps,
            total_elapsed_seconds=time.time() - t0,
            peak_vram_gib=peak_vram
        )
