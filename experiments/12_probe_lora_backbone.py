"""
Experiment 12: Load Qwen3-8B-Base + Pretrained LoRA8B Adapter + Multi-Layer Readout Head
Evaluates the 8-layer readout representation on train.csv.
Dual T4 inference with torch.float16 and device_map="auto".
"""

import gc
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL
from src.cv import split_folds
from src.cavity import loo_cavity_center
from src.metrics import evaluate_predictions, format_evaluation_summary


class MultiLayerReadoutHead(nn.Module):
    """
    Multi-layer readout head matching lora8b_v1:
    Normalizes 8 intermediate layer representations with LayerNorm/RMSNorm,
    concatenates into 32,768 dims, and applies plain and centered linear projections.
    """
    def __init__(self, hidden_dim: int = 4096, num_layers: int = 8, num_classes: int = 18):
        super().__init__()
        self.num_layers = num_layers
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.plain = nn.Linear(hidden_dim * num_layers, num_classes)
        self.centered = nn.Linear(hidden_dim * num_layers, num_classes)
        
    def load_from_safetensors(self, safetensors_path: str):
        with safe_open(safetensors_path, framework="pt") as f:
            for i in range(self.num_layers):
                self.norms[i].weight.data.copy_(f.get_tensor(f"norms.{i}.weight"))
                self.norms[i].bias.data.copy_(f.get_tensor(f"norms.{i}.bias"))
            self.plain.weight.data.copy_(f.get_tensor("plain.weight"))
            self.plain.bias.data.copy_(f.get_tensor("plain.bias"))
            self.centered.weight.data.copy_(f.get_tensor("centered.weight"))
            self.centered.bias.data.copy_(f.get_tensor("centered.bias"))
            
    def forward(self, layer_tensors: list, centered: bool = False):
        # layer_tensors: list of [B, hidden_dim] mean-pooled states
        normed = [self.norms[i](layer_tensors[i]) for i in range(self.num_layers)]
        fused = torch.cat(normed, dim=-1)  # [B, hidden_dim * 8]
        if centered:
            return self.centered(fused)
        return self.plain(fused)


def run_experiment_12():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA Available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}")
    
    # 1. Base Model & Adapter paths
    base_candidates = [
        "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1",
        "/kaggle/input/qwen-3/transformers/8b-base/1",
    ]
    base_model_path = None
    for c in base_candidates:
        if os.path.exists(os.path.join(c, "config.json")):
            base_model_path = c
            break
            
    adapter_path = "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1/adapter"
    heads_path = "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1/heads.safetensors"
    meta_path = "/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1/meta.json"
    
    if not base_model_path:
        raise FileNotFoundError("Qwen3-8B-Base model path not found in /kaggle/input/models!")
    if not os.path.exists(heads_path):
        raise FileNotFoundError(f"Missing {heads_path}!")
        
    with open(meta_path) as f:
        meta = json.load(f)
    print("Loaded meta.json:", meta["cfg"])
    
    target_layers = meta["cfg"]["layers"]  # [2, 4, 6, 8, 12, 16, 20, 24]
    max_len = meta["cfg"].get("max_len", 447)
    labels = meta["labels"]
    
    print(f"Target layers: {target_layers}, max_len: {max_len}")
    print(f"Meta labels ({len(labels)}): {labels}")
    
    # 2. Load Tokenizer & Model
    print(f"Loading tokenizer from {base_model_path}...")
    tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    print(f"Loading base model config (keeping {meta['cfg']['keep']} layers)...")
    cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    cfg.num_hidden_layers = meta["cfg"]["keep"]
    
    print("Loading base model in float16 with device_map='auto'...")
    model = AutoModel.from_pretrained(
        base_model_path,
        config=cfg,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # 3. Load Head
    print(f"Loading readout head from {heads_path}...")
    head = MultiLayerReadoutHead(hidden_dim=cfg.hidden_size, num_layers=len(target_layers), num_classes=len(labels))
    head.load_from_safetensors(heads_path)
    head.to(device)
    head.eval()
    
    # 4. Load Data
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    texts = df["text"].tolist()
    y_raw = df["model"].values
    prompts = df["prompt_id"].values
    
    print(f"\nRunning inference on {len(texts)} train passages...")
    batch_size = 16
    all_logits_plain = []
    
    # Map target layers to internal 0-indexed hidden state indices
    # output_hidden_states returns (embedding_output, layer_1, layer_2, ..., layer_keep)
    # So layer L corresponds to index L in hidden_states
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="LoRA8B Extraction"):
            b_texts = texts[i:i + batch_size]
            enc = tok(
                b_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)
            
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hs = out.hidden_states
            
            # Extract mean pooled representation for each target layer
            # mask: [B, S, 1]
            mask_exp = attention_mask.unsqueeze(-1).float()
            mask_sum = torch.clamp(mask_exp.sum(dim=1), min=1e-8)
            
            layer_pools = []
            for lyr in target_layers:
                # hs[lyr]: [B, S, H]
                state = hs[lyr]
                pooled = (state * mask_exp).sum(dim=1) / mask_sum  # [B, H]
                layer_pools.append(pooled.float())
                
            logits_p = head(layer_pools, centered=False)
            all_logits_plain.append(logits_p.cpu().numpy())
            
    logits = np.vstack(all_logits_plain)
    print(f"\nExtracted logits shape: {logits.shape}")
    
    # Evaluate Accuracy against meta labels
    preds_idx = np.argmax(logits, axis=1)
    preds = np.array([labels[i] for i in preds_idx])
    
    bacc = balanced_accuracy_score(y_raw, preds)
    print(f"\n==================================================")
    print(f"PRETRAINED LoRA8B PLAIN HEAD BALANCED ACCURACY: {bacc:.4f}")
    print(f"==================================================")
    
    rep = evaluate_predictions(y_raw, preds)
    print(format_evaluation_summary(rep))
    
    # Save extracted logits and features to /kaggle/working/
    np.save("/kaggle/working/lora8b_train_logits.npy", logits)
    print("Saved /kaggle/working/lora8b_train_logits.npy successfully!")


if __name__ == "__main__":
    run_experiment_12()
