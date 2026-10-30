"""
Experiment 05: Qwen2.5-1.5B Foundation Model with LoRA Fine-Tuning
Features:
- Qwen2.5-1.5B backbone (152k subword vocabulary for token boundary discrimination)
- LoRA applied to all linear projections (q, k, v, o, gate, up, down) with r=16, alpha=32
- Sequence truncation augmentation (25% samples to 35-65 words)
- Attentive mean-pooling across tokens
- Class-weighted CrossEntropyLoss with Label Smoothing (eps=0.05)
- Mixed precision training (fp16) on Kaggle Dual T4
"""

import os
import sys
import time
import math
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import balanced_accuracy_score

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL, RANDOM_SEED
from src.cv import split_folds
from src.metrics import evaluate_predictions, format_evaluation_summary

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_LEN = 256
BATCH_SIZE = 8
ACCUM_STEPS = 4
EPOCHS = 3
LR = 1.5e-4
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QwenTruncationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = MAX_LEN, is_train: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Runtime truncation augmentation for 25% of training samples
        if self.is_train and random.random() < 0.25:
            words = text.split()
            if len(words) > 65:
                crop_len = random.randint(35, 65)
                start_idx = random.randint(0, len(words) - crop_len)
                text = " ".join(words[start_idx : start_idx + crop_len])

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class QwenStylometryLoRA(nn.Module):
    """
    Qwen2.5 backbone with LoRA adapter and attentive pooling classification head.
    """
    def __init__(self, model_name: str = MODEL_NAME, num_classes: int = 18):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Apply LoRA to all attention and MLP projection weights
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.encoder = get_peft_model(base_model.model, peft_config)
        self.encoder.print_trainable_parameters()
        
        hidden_dim = base_model.config.hidden_size
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3]])
        # Place classifier on the same device as the last layer
        last_device = next(reversed(list(self.encoder.parameters()))).device
        self.classifier = nn.Linear(hidden_dim, num_classes).to(torch.float32).to(last_device)

    def forward(self, input_ids, attention_mask):
        first_device = next(self.encoder.parameters()).device
        input_ids = input_ids.to(first_device)
        attention_mask = attention_mask.to(first_device)
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state.to(torch.float32)
        
        # Mean pooling across valid tokens
        input_mask_expanded = attention_mask.to(last_hidden_state.device).unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).to(self.classifier.weight.device)
        
        logits = torch.mean(torch.stack([self.classifier(dropout(mean_pooled)) for dropout in self.dropouts], dim=0), dim=0)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, criterion, epoch: int = 1, accum_steps=ACCUM_STEPS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    device = model.classifier.weight.device
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{EPOCHS} [Train]", mininterval=1.0)
    for step, batch in pbar:
        labels = batch["label"].to(device)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, labels)
        loss = loss / accum_steps

        loss.backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        total_loss += loss.item() * accum_steps
        avg_loss = total_loss / (step + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="[Validating]", mininterval=1.0)
    for batch in pbar:
        logits = model(batch["input_ids"], batch["attention_mask"])
        probs = torch.softmax(logits, dim=-1)
            
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.append(probs.cpu().numpy())
        
    return np.array(all_preds), np.vstack(all_probs)


def run_qwen_experiment():
    seed_everything(RANDOM_SEED)
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    
    folds = split_folds(df)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"\n--- Training Qwen2.5-1.5B LoRA on Kaggle Dual T4 (Fold 0 Validation) ---")
    val_fold = 0
    trn_idx = np.where(folds != val_fold)[0]
    val_idx = np.where(folds == val_fold)[0]
    
    model = QwenStylometryLoRA(MODEL_NAME, num_classes=18)
    device = model.classifier.weight.device
    
    class_counts = np.bincount(y_int[trn_idx], minlength=18)
    class_weights = torch.tensor(1.0 / (class_counts / class_counts.sum()), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    train_dataset = QwenTruncationDataset([df["text"].iloc[i] for i in trn_idx], y_int[trn_idx], tokenizer, is_train=True)
    val_dataset = QwenTruncationDataset([df["text"].iloc[i] for i in val_idx], y_int[val_idx], tokenizer, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
    
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * 0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_bacc = 0.0
    best_probs = None
    best_preds = None
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, epoch=epoch)
        preds, probs = evaluate(model, val_loader)
        
        pred_labels = [IDX_TO_MODEL[p] for p in preds]
        bacc = balanced_accuracy_score(y_raw[val_idx], pred_labels)
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {loss:.4f} | Val Fold 0 BAcc: {bacc:.4f} | Time: {elapsed:.1f}s")
        
        if bacc > best_bacc:
            best_bacc = bacc
            best_probs = probs
            best_preds = pred_labels
            os.makedirs("/kaggle/working/weights", exist_ok=True)
            torch.save(model.state_dict(), "/kaggle/working/weights/qwen_best_fold0.pt")
            
    print(f"\nBest Qwen Fold 0 Balanced Accuracy: {best_bacc:.4f}")
    val_report = evaluate_predictions(y_raw[val_idx], np.array(best_preds))
    print("\n" + "="*50)
    print("QWEN2.5-1.5B FOLD 0 EVALUATION REPORT:")
    print(format_evaluation_summary(val_report))
    
    np.save("/kaggle/working/experiments/qwen_fold0_probs.npy", best_probs)
    print("\n[OK] Qwen Experiment 05 complete.")


if __name__ == "__main__":
    run_qwen_experiment()
