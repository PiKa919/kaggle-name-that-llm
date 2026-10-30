"""
Experiment 10: Multi-Depth Contrastive LoRA Fine-Tuning on 34k External Corpus (Dual T4).
Implements Contra-X style vs. topic disentanglement with prompt-cluster hard negatives,
random sequence truncation augmentation (40-75 words), and multi-depth intermediate readout.

Target Compute: Kaggle Dual Nvidia Tesla T4 (2x16 GB = 32 GB VRAM).
Usage on Kaggle:
  !python3 /kaggle/working/experiments/10_contrastive_lora_train.py
"""

import gc
import json
import os
import sys
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

# Patch PEFT for Kaggle container compatibility
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
try:
    import peft.tuners.lora.torchao as lora_torchao
    lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass
from peft import LoraConfig, get_peft_model

sys.path.append("/kaggle/working")
from src.config import VALID_18_MODELS, MODEL_TO_IDX
from src.contrastive_loss import ContraXStyleLoss, random_truncation_augmentation
from src.contrastive_sampler import PromptClusterBatchSampler

# Paths & Settings
SEED = 42
MAX_LEN = 384
BATCH_PROMPTS = 1  # 18 models per prompt = 18 passages per batch
LR = 2e-4
WEIGHT_DECAY = 0.01
CONTRASTIVE_WEIGHT = 0.3
EPOCHS_STAGE1 = 1  # External 34k corpus
EPOCHS_STAGE2 = 2  # In-competition train.csv calibration
TARGET_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]

BASE_CANDIDATES = [
    "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
    "/kaggle/input/qwen-3/transformers/8b-base/1",
]
POOL_CANDIDATES = [
    "/kaggle/working/data/curated_pool_v2.parquet",
    "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/pool_v1.parquet",
]
TRAIN_CANDIDATES = [
    "/kaggle/input/competitions/name-that-llm/train.csv",
    "/kaggle/input/name-that-llm/train.csv",
    "data/train.csv",
]


def locate_file(cands: List[str], desc: str) -> str:
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{desc}: none found in {cands}")


class MultiLayerReadoutHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, num_layers: int = 8, num_classes: int = 18):
        super().__init__()
        self.num_layers = num_layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        self.classifier = nn.Linear(hidden_dim * num_layers, num_classes)

    def forward(self, layer_tensors: list) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = [self.norms[i](layer_tensors[i]) for i in range(self.num_layers)]
        fused = torch.cat(normed, dim=-1)
        logits = self.classifier(fused)
        proj = self.projector(fused)
        return logits, proj


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_training: bool = True):
        self.texts = df["text"].fillna("").tolist()
        self.labels = [MODEL_TO_IDX[m] for m in df["model"].values]
        # Map prompt_id to integer
        unique_pids = {p: i for i, p in enumerate(df["prompt_id"].unique())}
        self.prompt_ints = [unique_pids[p] for p in df["prompt_id"].values]
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        t = self.texts[idx]
        if self.is_training:
            t = random_truncation_augmentation(t, min_words=40, max_words=75, p=0.30)
        return {
            "text": t,
            "label": self.labels[idx],
            "prompt_int": self.prompt_ints[idx]
        }


def collate_fn(batch, tok):
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    prompts = torch.tensor([item["prompt_int"] for item in batch], dtype=torch.long)
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "prompts": prompts
    }


def main():
    print("=== EXPERIMENT 10: CONTRASTIVE TWO-STAGE LoRA FINE-TUNING ===")
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    base_dir = locate_file(BASE_CANDIDATES, "Base Model")
    train_path = locate_file(TRAIN_CANDIDATES, "Train CSV")
    print(f"Base model: {base_dir}")
    print(f"Train path: {train_path}")

    # Load Tokenizer & Config
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    cfg = AutoConfig.from_pretrained(base_dir, trust_remote_code=True)
    max_keep_layer = max(TARGET_LAYERS) + 1
    cfg.num_hidden_layers = max_keep_layer

    print(f"Loading Qwen3-8B up to layer {max_keep_layer} on Dual T4 (fp16)...")
    base_model = AutoModel.from_pretrained(
        base_dir,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    head = MultiLayerReadoutHead(
        hidden_dim=cfg.hidden_size,
        num_layers=len(TARGET_LAYERS),
        num_classes=len(VALID_18_MODELS)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head.to(device)

    # Optimizer & Losses
    params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    ce_loss_fn = nn.CrossEntropyLoss()
    contrax_loss_fn = ContraXStyleLoss(temperature=0.07, hard_negative_weight=2.0)

    # Stage 1: External Pool (if available)
    try:
        pool_path = locate_file(POOL_CANDIDATES, "Curated Pool")
        print(f"\n--- STAGE 1: Pre-training on External Pool ({pool_path}) ---")
        pool_df = pd.read_parquet(pool_path)
        print(f"Loaded {len(pool_df)} external passages.")
        ds_stage1 = TextDataset(pool_df, is_training=True)
        sampler1 = PromptClusterBatchSampler(pool_df, batch_prompts=BATCH_PROMPTS, shuffle=True)
        loader1 = DataLoader(ds_stage1, batch_sampler=sampler1, collate_fn=lambda b: collate_fn(b, tok))

        model.train()
        head.train()
        for batch in tqdm(loader1, desc="Stage 1 Epoch 1"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            p_ints = batch["prompts"].to(device)

            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = out.hidden_states
            mask_exp = mask.unsqueeze(-1).float()
            mask_sum = torch.clamp(mask_exp.sum(dim=1), min=1e-8)

            pools = []
            for lyr in TARGET_LAYERS:
                pooled = (hs[lyr] * mask_exp).sum(dim=1) / mask_sum
                pools.append(pooled.float())

            logits, proj = head(pools)
            l_ce = ce_loss_fn(logits, labels)
            l_contra = contrax_loss_fn(proj, labels, p_ints)
            loss = l_ce + CONTRASTIVE_WEIGHT * l_contra

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    except Exception as e:
        print(f"Skipping Stage 1 (External Pool not found or error: {e})")

    # Stage 2: Calibration on Official train.csv
    print(f"\n--- STAGE 2: Calibration on Official train.csv ---")
    train_df = pd.read_csv(train_path)
    ds_stage2 = TextDataset(train_df, is_training=True)
    sampler2 = PromptClusterBatchSampler(train_df, batch_prompts=BATCH_PROMPTS, shuffle=True)
    loader2 = DataLoader(ds_stage2, batch_sampler=sampler2, collate_fn=lambda b: collate_fn(b, tok))

    model.train()
    head.train()
    for ep in range(EPOCHS_STAGE2):
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(loader2, desc=f"Stage 2 Epoch {ep+1}/{EPOCHS_STAGE2}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            p_ints = batch["prompts"].to(device)

            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hs = out.hidden_states
            mask_exp = mask.unsqueeze(-1).float()
            mask_sum = torch.clamp(mask_exp.sum(dim=1), min=1e-8)

            pools = []
            for lyr in TARGET_LAYERS:
                pooled = (hs[lyr] * mask_exp).sum(dim=1) / mask_sum
                pools.append(pooled.float())

            logits, proj = head(pools)
            l_ce = ce_loss_fn(logits, labels)
            l_contra = contrax_loss_fn(proj, labels, p_ints)
            loss = l_ce + CONTRASTIVE_WEIGHT * l_contra

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        acc = correct / max(total, 1)
        print(f"Epoch {ep+1}: Loss = {total_loss / len(loader2):.4f}, Batch Accuracy = {acc:.4f}")

    # Save artifacts
    save_dir = "/kaggle/working/weights/contra_lora_qwen3"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(os.path.join(save_dir, "adapter"))
    torch.save(head.state_dict(), os.path.join(save_dir, "head.pt"))
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump({
            "target_layers": TARGET_LAYERS,
            "max_len": MAX_LEN,
            "labels": VALID_18_MODELS
        }, f, indent=2)

    print(f"\nSUCCESS: Saved contrastive LoRA adapter and readout head to {save_dir}")
    print(f"Completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
