"""
ffnn.py
-------
Feed-Forward Neural Network with Bag-of-Words features for suicidal
ideation detection (binary classification).

The original skeleton targeted 5-class sentiment; this version is
adapted for 2-class output (suicidal vs non-suicidal) while keeping
the same interface so the training loop is unchanged.

Usage:
    python ffnn.py -hd 128 -e 10 \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --test_data  data/test.json  \
        --do_train
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import time
import json
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

unk = '<UNK>'


# ── Model ────────────────────────────────────────────────────────────────────

class FFNN(nn.Module):
    """
    Two-layer feed-forward network:
        input (BoW) → Linear → ReLU → Linear → LogSoftmax
    """
    def __init__(self, input_dim: int, h: int, output_dim: int = 2):
        super(FFNN, self).__init__()
        self.h          = h
        self.output_dim = output_dim

        self.W1         = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.dropout    = nn.Dropout(p=0.3)          # regularisation
        self.W2         = nn.Linear(h, output_dim)

        self.softmax    = nn.LogSoftmax(dim=-1)
        self.loss       = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Hidden layer
        hidden          = self.activation(self.W1(input_vector))
        hidden          = self.dropout(hidden)
        # Output layer
        output          = self.W2(hidden)
        # Probability distribution
        predicted_vector = self.softmax(output)
        return predicted_vector


# ── Vocabulary helpers ────────────────────────────────────────────────────────

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index, index2word = {}, {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


def convert_to_vector_representation(data, word2index):
    """Convert each document into a BoW count vector."""
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(train_path, val_path):
    """
    Expects JSON list of {"text": "...", "stars": 1|2}.
    stars=1 → label 0 (non-suicidal), stars=2 → label 1 (suicidal).
    """
    with open(train_path) as f:
        training   = json.load(f)
    with open(val_path) as f:
        validation = json.load(f)

    def parse(records):
        out = []
        for elt in records:
            label = int(elt["stars"]) - 1          # 0 or 1
            out.append((elt["text"].split(), label))
        return out

    return parse(training), parse(validation)


def load_test(test_path, word2index):
    with open(test_path) as f:
        test = json.load(f)
    data = [(elt["text"].split(), int(elt["stars"]) - 1) for elt in test]
    return convert_to_vector_representation(data, word2index)


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate(model, data, minibatch_size=16):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    N = len(data)
    with torch.no_grad():
        for mb in range(N // minibatch_size):
            for i in range(minibatch_size):
                vec, gold = data[mb * minibatch_size + i]
                pred_vec  = model(vec)
                pred      = torch.argmax(pred_vec).item()
                correct  += int(pred == gold)
                total    += 1
                all_preds.append(pred)
                all_labels.append(gold)

    accuracy  = correct / total
    f1        = f1_score(all_labels, all_preds, average="binary")
    precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    return accuracy, f1, precision, recall, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True)
    parser.add_argument("-e",  "--epochs",     type=int, required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data",   required=True)
    parser.add_argument("--test_data",  default=None)
    parser.add_argument("--do_train",   action="store_true")
    parser.add_argument("--save_model", default=None, help="Path to save best model")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    # ── Load & vectorise ──
    print("=" * 50)
    print("Loading data ...")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print(f"  Train samples : {len(train_data)}")
    print(f"  Val   samples : {len(valid_data)}")
    print(f"  Vocabulary    : {len(vocab)}")

    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # ── Model ──
    model     = FFNN(input_dim=len(vocab), h=args.hidden_dim, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_f1   = 0.0
    minibatch_size = 16

    print("=" * 50)
    print(f"Training for {args.epochs} epochs  |  hidden_dim={args.hidden_dim}")
    print("=" * 50)

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_data)
        correct, total = 0, 0
        loss_sum       = 0.0
        N              = len(train_data)
        start          = time.time()

        for mb in tqdm(range(N // minibatch_size), desc=f"Epoch {epoch+1} train"):
            optimizer.zero_grad()
            loss = None
            for i in range(minibatch_size):
                vec, gold = train_data[mb * minibatch_size + i]
                pred_vec  = model(vec)
                pred      = torch.argmax(pred_vec).item()
                correct  += int(pred == gold)
                total    += 1
                ex_loss   = model.compute_Loss(pred_vec.view(1, -1),
                                               torch.tensor([gold]))
                loss = ex_loss if loss is None else loss + ex_loss

            loss = loss / minibatch_size
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_acc = correct / total

        # Validation
        val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(
            model, valid_data, minibatch_size)

        elapsed = time.time() - start
        print(f"\nEpoch {epoch+1:>2} | "
              f"Loss {loss_sum/(N//minibatch_size):.4f} | "
              f"Train Acc {train_acc:.4f} | "
              f"Val Acc {val_acc:.4f}  F1 {val_f1:.4f}  "
              f"P {val_prec:.4f}  R {val_rec:.4f} | "
              f"{elapsed:.1f}s")

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if args.save_model:
                torch.save(model.state_dict(), args.save_model)
                print(f"  ✓ New best model saved → {args.save_model}")

    # ── Test set evaluation ──
    if args.test_data:
        print("\n" + "=" * 50)
        print("Evaluating on test set ...")
        test_data = load_test(args.test_data, word2index)
        t_acc, t_f1, t_prec, t_rec, preds, labels = evaluate(
            model, test_data, minibatch_size)
        print(f"Test  Acc {t_acc:.4f}  F1 {t_f1:.4f}  "
              f"Precision {t_prec:.4f}  Recall {t_rec:.4f}")
        print("\nDetailed report:")
        print(classification_report(labels, preds,
                                    target_names=["non-suicidal", "suicidal"]))

        os.makedirs("results", exist_ok=True)
        with open("results/ffnn_test.out", "w") as f:
            for p in preds:
                f.write(str(p) + "\n")
        print("Predictions written → results/ffnn_test.out")
