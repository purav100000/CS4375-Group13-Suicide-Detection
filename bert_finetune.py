"""
bert_finetune.py
----------------
Fine-tunes a pre-trained BERT (or DistilBERT) model for binary
suicidal ideation classification.

Requirements:
    pip install transformers torch scikit-learn tqdm

Models supported (via --model_name):
    distilbert-base-uncased   (faster, lighter — recommended to start)
    bert-base-uncased         (standard BERT)
    mental/mental-bert-base-uncased  (domain-specific, if available)

Usage:
    python bert_finetune.py \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --test_data  data/test.json  \
        --model_name distilbert-base-uncased \
        --epochs     4 \
        --batch_size 16 \
        --lr         2e-5 \
        --max_len    256
"""

import os
import json
import time
import random
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)

try:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               get_linear_schedule_with_warmup)
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False
    print("[error] transformers not installed.")
    print("  Run: pip install transformers")
    exit(1)

from tqdm import tqdm


# ── Dataset ───────────────────────────────────────────────────────────────────

class SuicideDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids"     : encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        records = json.load(f)
    texts  = [r["text"] for r in records]
    labels = [int(r["stars"]) - 1 for r in records]
    return texts, labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            outputs   = model(input_ids=input_ids,
                              attention_mask=attn_mask)
            preds     = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc       = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary",
                                zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="binary",
                             zero_division=0)
    return acc, f1, precision, recall, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",  required=True)
    parser.add_argument("--val_data",    required=True)
    parser.add_argument("--test_data",   default=None)
    parser.add_argument("--model_name",  default="distilbert-base-uncased",
                        help="HuggingFace model name or local path")
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max_len",     type=int,   default=256)
    parser.add_argument("--warmup_ratio",type=float, default=0.1)
    parser.add_argument("--save_model",  default="models/bert_best")
    parser.add_argument("--outdir",      default="results")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    os.makedirs(args.outdir,                exist_ok=True)
    os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else   # Apple Silicon
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Device : {device}")

    # ── Tokenizer & model ──
    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2)
    model.to(device)

    # ── Data ──
    print("Loading data ...")
    train_texts, train_labels = load_json(args.train_data)
    val_texts,   val_labels   = load_json(args.val_data)
    print(f"  Train : {len(train_texts)}   Val : {len(val_texts)}")

    train_ds = SuicideDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds   = SuicideDataset(val_texts,   val_labels,   tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Optimiser with linear warmup ──
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps    = warmup_steps,
        num_training_steps  = total_steps,
    )

    print(f"\nFine-tuning {args.model_name}")
    print(f"  Epochs={args.epochs}  BatchSize={args.batch_size}  "
          f"LR={args.lr}  MaxLen={args.max_len}")
    print(f"  Total steps={total_steps}  Warmup={warmup_steps}")
    print("=" * 55)

    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        start = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            optimizer.zero_grad()
            outputs   = model(input_ids=input_ids,
                              attention_mask=attn_mask,
                              labels=labels)
            loss      = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds       = torch.argmax(outputs.logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total
        avg_loss  = total_loss / len(train_loader)

        val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(
            model, val_loader, device)

        elapsed = time.time() - start
        print(f"\nEpoch {epoch+1:>2} | Loss {avg_loss:.4f} | "
              f"Train Acc {train_acc:.4f} | "
              f"Val Acc {val_acc:.4f}  F1 {val_f1:.4f}  "
              f"P {val_prec:.4f}  R {val_rec:.4f} | {elapsed:.0f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
            print(f"  ✓ Best model saved → {args.save_model}  (F1={best_val_f1:.4f})")

    # ── Test ──
    if args.test_data:
        print("\n" + "=" * 55)
        print("Loading best checkpoint for test evaluation ...")
        best_model = AutoModelForSequenceClassification.from_pretrained(
            args.save_model).to(device)

        test_texts, test_labels = load_json(args.test_data)
        test_ds     = SuicideDataset(test_texts, test_labels, tokenizer, args.max_len)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        t_acc, t_f1, t_prec, t_rec, preds, labels = evaluate(
            best_model, test_loader, device)

        print(f"\nTest  Acc {t_acc:.4f}  F1 {t_f1:.4f}  "
              f"Precision {t_prec:.4f}  Recall {t_rec:.4f}")
        print("\nDetailed Report:")
        print(classification_report(labels, preds,
                                    target_names=["non-suicidal", "suicidal"]))

        out_path = os.path.join(args.outdir, "bert_test.out")
        with open(out_path, "w") as f:
            for p in preds:
                f.write(str(p) + "\n")
        print(f"Predictions saved → {out_path}")

    print(f"\nBest validation F1 : {best_val_f1:.4f}")
