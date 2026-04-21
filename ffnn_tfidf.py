"""
ffnn_tfidf.py
-------------
FFNN that supports both BoW and TF-IDF feature vectors.
TF-IDF is computed over the training corpus with sklearn, then fed
into the same two-layer FFNN architecture from ffnn.py.

Usage (TF-IDF):
    python ffnn_tfidf.py -hd 128 -e 10 \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --test_data  data/test.json  \
        --feature tfidf

Usage (BoW — identical to original ffnn.py):
    python ffnn_tfidf.py -hd 128 -e 10 \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --feature bow
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
import json
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              classification_report)


# ── Model (unchanged from ffnn.py) ───────────────────────────────────────────

class FFNN(nn.Module):
    def __init__(self, input_dim: int, h: int, output_dim: int = 2):
        super(FFNN, self).__init__()
        self.W1         = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.dropout    = nn.Dropout(p=0.3)
        self.W2         = nn.Linear(h, output_dim)
        self.softmax    = nn.LogSoftmax(dim=-1)
        self.loss       = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden           = self.dropout(self.activation(self.W1(input_vector)))
        predicted_vector = self.softmax(self.W2(hidden))
        return predicted_vector


# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        records = json.load(f)
    texts  = [r["text"] for r in records]
    labels = [int(r["stars"]) - 1 for r in records]
    return texts, labels


def vectorize(train_texts, val_texts, feature="tfidf", max_features=30000):
    """
    Fit vectorizer on train, transform train + val.
    Returns scipy sparse → numpy dense → torch FloatTensor pairs.
    """
    if feature == "tfidf":
        vec = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = (1, 2),
            sublinear_tf = True,
            min_df       = 2,
        )
    else:
        vec = CountVectorizer(
            max_features = max_features,
            ngram_range  = (1, 2),
            min_df       = 2,
        )

    X_train = vec.fit_transform(train_texts).toarray().astype(np.float32)
    X_val   = vec.transform(val_texts).toarray().astype(np.float32)
    return (torch.from_numpy(X_train),
            torch.from_numpy(X_val),
            vec)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X, labels, minibatch_size=16):
    model.eval()
    correct, total = 0, 0
    all_preds = []

    with torch.no_grad():
        N = len(labels)
        for mb in range(N // minibatch_size):
            for i in range(minibatch_size):
                idx      = mb * minibatch_size + i
                vec      = X[idx]
                gold     = labels[idx]
                pred_vec = model(vec)
                pred     = torch.argmax(pred_vec).item()
                correct += int(pred == gold)
                total   += 1
                all_preds.append(pred)

    acc       = correct / total
    f1        = f1_score(labels[:total], all_preds, average="binary")
    precision = precision_score(labels[:total], all_preds, average="binary",
                                zero_division=0)
    recall    = recall_score(labels[:total], all_preds, average="binary",
                             zero_division=0)
    return acc, f1, precision, recall, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim",   type=int, required=True)
    parser.add_argument("-e",  "--epochs",        type=int, required=True)
    parser.add_argument("--train_data",  required=True)
    parser.add_argument("--val_data",    required=True)
    parser.add_argument("--test_data",   default=None)
    parser.add_argument("--feature",     default="tfidf",
                        choices=["tfidf", "bow"])
    parser.add_argument("--max_features", type=int, default=30000)
    parser.add_argument("--save_model",  default=None)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    # ── Load & vectorise ──
    print(f"Loading data and building {args.feature.upper()} features ...")
    train_texts, train_labels = load_json(args.train_data)
    val_texts,   val_labels   = load_json(args.val_data)

    X_train, X_val, vectorizer = vectorize(
        train_texts, val_texts, args.feature, args.max_features)
    input_dim = X_train.shape[1]
    print(f"  Feature dim : {input_dim}  |  "
          f"Train : {len(train_labels)}  Val : {len(val_labels)}")

    # Pair tensors with labels and shuffle
    train_data = list(zip(X_train, train_labels))
    val_data   = list(zip(X_val,   val_labels))

    # ── Model ──
    model     = FFNN(input_dim=input_dim, h=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_f1    = 0.0
    minibatch_size = 16

    print(f"Training FFNN+{args.feature.upper()}  hidden={args.hidden_dim}  "
          f"epochs={args.epochs}")
    print("=" * 55)

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_data)
        correct, total, loss_sum = 0, 0, 0.0
        N     = len(train_data)
        start = time.time()

        for mb in tqdm(range(N // minibatch_size), desc=f"Epoch {epoch+1}"):
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

        val_acc, val_f1, val_prec, val_rec, _ = evaluate(
            model, X_val, val_labels, minibatch_size)

        print(f"Epoch {epoch+1:>2} | Loss {loss_sum/(N//minibatch_size):.4f} | "
              f"Train {train_acc:.4f} | Val Acc {val_acc:.4f}  F1 {val_f1:.4f}  "
              f"P {val_prec:.4f}  R {val_rec:.4f} | {time.time()-start:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if args.save_model:
                torch.save(model.state_dict(), args.save_model)
                print(f"  ✓ Best model saved  (F1={best_val_f1:.4f})")

    # ── Test ──
    if args.test_data:
        test_texts, test_labels = load_json(args.test_data)
        X_test = torch.from_numpy(
            vectorizer.transform(test_texts).toarray().astype(np.float32))
        t_acc, t_f1, t_prec, t_rec, preds = evaluate(
            model, X_test, test_labels, minibatch_size)
        print(f"\nTest  Acc {t_acc:.4f}  F1 {t_f1:.4f}  "
              f"P {t_prec:.4f}  R {t_rec:.4f}")
        print(classification_report(test_labels[:len(preds)], preds,
                                    target_names=["non-suicidal", "suicidal"]))

        os.makedirs("results", exist_ok=True)
        tag = f"ffnn_{args.feature}"
        with open(f"results/{tag}_test.out", "w") as f:
            for p in preds:
                f.write(str(p) + "\n")
        print(f"Predictions written → results/{tag}_test.out")
