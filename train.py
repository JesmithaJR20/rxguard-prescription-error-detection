"""
train.py  –  Fine-tune DistilBERT for prescription error classification
Labels : safe | overdose | drug_interaction

Usage:
    python train.py                   # default: data/prescriptions.csv
    python train.py --data my.csv     # custom dataset path
    python train.py --epochs 5        # more epochs
"""

import os
import argparse
import json
import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 8           # small batch → CPU-friendly
EPOCHS       = 4
LEARNING_RATE = 2e-5
SAVE_DIR     = "model"

LABEL2ID = {"safe": 0, "overdose": 1, "drug_interaction": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
class PrescriptionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds  = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            total_loss += outputs.loss.item()
            preds  = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main(args):
    device = torch.device("cpu")   # CPU-only
    print(f"\n{'='*50}")
    print("  Medical Prescription Error Classifier")
    print(f"  Device : {device}")
    print(f"{'='*50}\n")

    # 1. Load data
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"])
    df["label_id"] = df["label"].map(LABEL2ID)
    print(f"📊  Loaded {len(df)} samples from {args.data}")
    print(df["label"].value_counts().to_string())

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(), df["label_id"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label_id"]
    )
    print(f"\n  Train : {len(X_train)}  |  Val : {len(X_val)}\n")

    # 3. Tokenizer + Datasets + Loaders
    print("⬇️   Loading DistilBERT tokenizer …")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PrescriptionDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = PrescriptionDataset(X_val,   y_val,   tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model
    print("🤖  Loading DistilBERT for sequence classification …")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop
    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss,   val_acc, val_preds, val_labels = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
            f"⏱ {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"   ✅  New best model saved → {SAVE_DIR}/")

    # 6. Final evaluation
    print(f"\n{'='*50}")
    print("📈  Final Evaluation on Validation Set")
    print('='*50)
    label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    print(classification_report(val_labels, val_preds, target_names=label_names))

    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=label_names, columns=label_names).to_string())

    # 7. Save metadata
    meta = {
        "model":       MODEL_NAME,
        "num_labels":  len(LABEL2ID),
        "label2id":    LABEL2ID,
        "id2label":    ID2LABEL,
        "max_len":     MAX_LEN,
        "best_val_acc": round(best_val_acc, 4),
        "history":     history,
    }
    with open(os.path.join(SAVE_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n🎉  Training complete!  Best val accuracy: {best_val_acc:.4f}")
    print(f"   Model saved to → {SAVE_DIR}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/prescriptions.csv")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(args)
