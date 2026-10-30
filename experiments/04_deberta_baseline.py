"""
Experiment 04: DeBERTa-v3-base Neural Classifier on Kaggle Dual T4
Features:
- Mean-pooling representation over all sequence tokens (not just [CLS])
- Sequence truncation augmentation (25% samples randomly cropped to 35-65 words)
- Class-weighted CrossEntropyLoss with Label Smoothing (eps=0.05)
- Cosine Annealing learning rate schedule with 10% linear warmup
- Mixed precision training (fp16) on CUDA
- StratifiedGroupKFold validation tracking official Balanced Accuracy
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
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from sklearn.metrics import balanced_accuracy_score

sys.path.append("/kaggle/working")
from src.config import TRAIN_CSV, TEST_CSV, VALID_18_MODELS, MODEL_TO_IDX, IDX_TO_MODEL, RANDOM_SEED
from src.cv import split_folds
from src.metrics import evaluate_predictions, format_evaluation_summary

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 384
BATCH_SIZE = 16
ACCUM_STEPS = 2
EPOCHS = 4
LR = 2.5e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TruncationDataset(Dataset):
    """
    Dataset with runtime truncation augmentation:
    Randomly crops 25% of training samples to 35-65 words to resolve short-passage hardness.
    """
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
        
        # Runtime truncation augmentation for training
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


class DebertaStylometryClassifier(nn.Module):
    """
    DeBERTa backbone with attentive mean-pooling across active tokens
    and multi-sample dropout classification head.
    """
    def __init__(self, model_name: str = MODEL_NAME, num_classes: int = 18):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_dim = self.config.hidden_size
        
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.1, 0.2, 0.3]])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Sequence output shape: (batch_size, seq_len, hidden_dim)
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling with attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Multi-sample dropout for regularization
        logits = torch.mean(torch.stack([self.classifier(dropout(mean_pooled)) for dropout in self.dropouts], dim=0), dim=0)
        return logits


def train_epoch(model, dataloader, optimizer, scheduler, criterion, accum_steps=ACCUM_STEPS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        # DeBERTa-v3 disentangled attention requires float32 to avoid fp16 overflow
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss = loss / accum_steps

        loss.backward()
        
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        total_loss += loss.item() * accum_steps
        
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_probs = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
            
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.append(probs.cpu().numpy())
        
    return np.array(all_preds), np.vstack(all_probs)


def run_deberta_experiment():
    seed_everything(RANDOM_SEED)
    print(f"Loading data from {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    df["text"] = df["text"].fillna("")
    y_raw = df["model"].values
    y_int = np.array([MODEL_TO_IDX[m] for m in y_raw])
    
    folds = split_folds(df)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"\n--- Training DeBERTa-v3 on Kaggle Dual T4 (Fold 0 Validation) ---")
    val_fold = 0
    trn_idx = np.where(folds != val_fold)[0]
    val_idx = np.where(folds == val_fold)[0]
    
    # Class weights for balanced loss
    class_counts = np.bincount(y_int[trn_idx], minlength=18)
    class_weights = torch.tensor(1.0 / (class_counts / class_counts.sum()), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    train_dataset = TruncationDataset([df["text"].iloc[i] for i in trn_idx], y_int[trn_idx], tokenizer, is_train=True)
    val_dataset = TruncationDataset([df["text"].iloc[i] for i in val_idx], y_int[val_idx], tokenizer, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    model = DebertaStylometryClassifier(MODEL_NAME, num_classes=18).to(DEVICE)
    
    total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * 0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_bacc = 0.0
    best_probs = None
    best_preds = None
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
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
            torch.save(model.state_dict(), "/kaggle/working/weights/deberta_best_fold0.pt")
            
    print(f"\nBest DeBERTa Fold 0 Balanced Accuracy: {best_bacc:.4f}")
    
    # Fold 0 Detailed Report
    val_report = evaluate_predictions(y_raw[val_idx], np.array(best_preds))
    print("\n" + "="*50)
    print("DEBERTA-V3 FOLD 0 EVALUATION REPORT:")
    print(format_evaluation_summary(val_report))
    
    # Save OOF predictions for fold 0
    np.save("/kaggle/working/experiments/deberta_fold0_probs.npy", best_probs)
    np.save("/kaggle/working/experiments/deberta_val_idx.npy", val_idx)
    
    # Evaluate blend with TF-IDF on Fold 0
    oof_tfidf_probs = np.load("/kaggle/working/experiments/oof_exp01_probs.npy")
    oof_lgb_probs = np.load("/kaggle/working/experiments/oof_exp02_probs.npy")
    
    # Test blend on fold 0 validation
    val_tfidf_p = oof_tfidf_probs[val_idx]
    val_lgb_p = oof_lgb_probs[val_idx]
    
    print("\n--- Evaluating Hybrid Neural + Stylometric Ensembles on Fold 0 ---")
    for w_deb in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        w_rest = 1.0 - w_deb
        w_tf = w_rest * 0.85
        w_lgb = w_rest * 0.15
        hybrid_p = w_deb * best_probs + w_tf * val_tfidf_p + w_lgb * val_lgb_p
        hybrid_preds = [IDX_TO_MODEL[p] for p in np.argmax(hybrid_p, axis=1)]
        h_bacc = balanced_accuracy_score(y_raw[val_idx], hybrid_preds)
        print(f"DeBERTa {w_deb:.2f} + TFIDF {w_tf:.2f} + LGBM {w_lgb:.2f} -> Fold 0 BAcc: {h_bacc:.4f}")
        
    print("\n[OK] DeBERTa Experiment 04 complete.")


if __name__ == "__main__":
    run_deberta_experiment()
