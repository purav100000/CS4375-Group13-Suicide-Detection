"""
compare_models.py
-----------------
Runs ALL models sequentially on the same train/val/test splits and
produces a unified results table + comparison bar chart.

Models evaluated:
  1. Logistic Regression + TF-IDF  (fastest baseline)
  2. FFNN + BoW
  3. FFNN + TF-IDF
  4. RNN  (requires word_embedding.pkl)
  5. LSTM (requires word_embedding.pkl)
  6. CNN  (requires word_embedding.pkl)

Usage:
    # With all models (embedding-based models need word_embedding.pkl):
    python compare_models.py \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --test_data  data/test.json  \
        --embedding  word_embedding.pkl

    # Baseline-only mode (no embeddings needed):
    python compare_models.py \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --test_data  data/test.json  \
        --baseline_only

Outputs:
    results/comparison_table.csv
    results/comparison_chart.png
    results/comparison_results.json
"""

import os
import json
import time
import random
import string
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score)

# ── Shared data utilities ─────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        records = json.load(f)
    texts  = [r["text"] for r in records]
    labels = [int(r["stars"]) - 1 for r in records]
    return texts, labels


def words_to_tensor(input_words, word_embedding):
    cleaned = " ".join(input_words)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation)).split()
    vectors = [word_embedding[w.lower()] if w.lower() in word_embedding
               else word_embedding["unk"] for w in cleaned]
    if not vectors:
        vectors = [word_embedding["unk"]]
    return torch.tensor(np.array(vectors), dtype=torch.float32)


def metrics(y_true, y_pred):
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred),          4),
        "f1"       : round(f1_score(y_true, y_pred, average="binary"), 4),
        "precision": round(precision_score(y_true, y_pred, average="binary",
                                           zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, average="binary",
                                        zero_division=0), 4),
    }


# ── Model definitions (inline so this file is self-contained) ─────────────────

class FFNN(nn.Module):
    def __init__(self, input_dim, h, output_dim=2):
        super().__init__()
        self.net     = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, output_dim))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss    = nn.NLLLoss()

    def forward(self, x):
        return self.softmax(self.net(x))

    def compute_Loss(self, pred, gold):
        return self.loss(pred.unsqueeze(0) if pred.dim() == 1 else pred, gold)


class RNNModel(nn.Module):
    def __init__(self, input_dim, h, output_dim=2):
        super().__init__()
        self.rnn     = nn.RNN(input_dim, h, 1, nonlinearity="tanh",
                              batch_first=False)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(h, output_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss    = nn.NLLLoss()

    def forward(self, x):            # x: (seq, 1, dim)
        _, h = self.rnn(x)
        return self.softmax(self.fc(self.dropout(h[-1])))

    def compute_Loss(self, pred, gold):
        return self.loss(pred, gold)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, h, output_dim=2):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, h, 1, batch_first=False)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(h, output_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss    = nn.NLLLoss()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.softmax(self.fc(self.dropout(h[-1])))

    def compute_Loss(self, pred, gold):
        return self.loss(pred.unsqueeze(0) if pred.dim() == 1 else pred, gold)


class TextCNN(nn.Module):
    def __init__(self, embed_dim, num_filters=128,
                 filter_sizes=(2, 3, 4, 5), output_dim=2):
        super().__init__()
        self.convs   = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss    = nn.NLLLoss()

    def forward(self, x):            # x: (1, seq, embed_dim)
        x = x.permute(0, 2, 1)
        pooled = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.softmax(self.fc(self.dropout(torch.cat(pooled, dim=1))))

    def compute_Loss(self, pred, gold):
        return self.loss(pred, gold)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_ffnn(X_train, y_train, X_val, y_val, hidden=128, epochs=10):
    model     = FFNN(X_train.shape[1], hidden)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mb        = 16
    N         = len(y_train)

    for _ in range(epochs):
        model.train()
        idx = list(range(N)); random.shuffle(idx)
        for start in range(0, N - mb + 1, mb):
            optimizer.zero_grad()
            loss = None
            for i in idx[start:start + mb]:
                pred    = model(X_train[i])
                ex_loss = model.compute_Loss(pred.view(1, -1),
                                             torch.tensor([y_train[i]]))
                loss = ex_loss if loss is None else loss + ex_loss
            (loss / mb).backward(); optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(y_val)):
            preds.append(torch.argmax(model(X_val[i])).item())
    return preds


def train_seq_model(model, train_data, val_data, word_embedding,
                    epochs=10, min_len=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mb        = 16
    N         = len(train_data)

    for _ in range(epochs):
        model.train()
        random.shuffle(train_data)
        for start in range(0, N - mb + 1, mb):
            optimizer.zero_grad()
            loss = None
            for words, gold in train_data[start:start + mb]:
                vec = words_to_tensor(words, word_embedding)
                is_cnn = isinstance(model, TextCNN)
                if is_cnn:
                    if vec.shape[0] < min_len:
                        pad = torch.zeros(min_len - vec.shape[0], vec.shape[1])
                        vec = torch.cat([vec, pad], dim=0)
                    t = vec.unsqueeze(0)           # (1, seq, dim)
                else:
                    t = vec.unsqueeze(1)           # (seq, 1, dim)
                pred    = model(t)
                ex_loss = model.compute_Loss(pred.view(1, -1),
                                             torch.tensor([gold]))
                loss = ex_loss if loss is None else loss + ex_loss
            if loss is not None:
                (loss / mb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for words, _ in val_data:
            vec = words_to_tensor(words, word_embedding)
            is_cnn = isinstance(model, TextCNN)
            if is_cnn:
                if vec.shape[0] < min_len:
                    pad = torch.zeros(min_len - vec.shape[0], vec.shape[1])
                    vec = torch.cat([vec, pad], dim=0)
                t = vec.unsqueeze(0)
            else:
                t = vec.unsqueeze(1)
            preds.append(torch.argmax(model(t)).item())
    return preds


# ── Results table & chart ─────────────────────────────────────────────────────

def print_table(results):
    header = f"\n{'Model':<30} {'Accuracy':>9} {'F1':>9} {'Precision':>10} {'Recall':>8} {'Time(s)':>9}"
    print("=" * len(header))
    print(header)
    print("─" * len(header))
    for r in results:
        print(f"  {r['model']:<28} {r['accuracy']:>9.4f} {r['f1']:>9.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['time']:>9.1f}")
    print("=" * len(header))


def save_chart(results, outdir):
    models  = [r["model"] for r in results]
    metrics_list = ["accuracy", "f1", "precision", "recall"]
    colors  = ["#14747E", "#2CA58D", "#F4A261", "#E63946"]
    x       = np.arange(len(models))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 2.2), 5))
    for i, (metric, color) in enumerate(zip(metrics_list, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
                    rotation=90)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Accuracy / F1 / Precision / Recall",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines[["top","right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(outdir, "comparison_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",    required=True)
    parser.add_argument("--val_data",      required=True)
    parser.add_argument("--test_data",     default=None)
    parser.add_argument("--embedding",     default="word_embedding.pkl")
    parser.add_argument("--hidden_dim",    type=int, default=128)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--baseline_only", action="store_true",
                        help="Skip embedding-based models")
    parser.add_argument("--outdir",        default="results")
    args = parser.parse_args()

    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    os.makedirs(args.outdir, exist_ok=True)

    # Choose split to report on
    eval_split = "test" if args.test_data else "val"
    print(f"\nEvaluating on: {eval_split} set")
    print("=" * 55)

    train_texts, train_labels = load_json(args.train_data)
    val_texts,   val_labels   = load_json(args.val_data)

    if args.test_data:
        eval_texts, eval_labels = load_json(args.test_data)
    else:
        eval_texts, eval_labels = val_texts, val_labels

    results = []

    # ── 1. Logistic Regression + TF-IDF ──────────────────────────────────────
    print("\n[1/6] Logistic Regression + TF-IDF ...")
    t0  = time.time()
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    X_tr = vec.fit_transform(train_texts)
    X_ev = vec.transform(eval_texts)
    clf  = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                               random_state=42)
    clf.fit(X_tr, train_labels)
    preds = clf.predict(X_ev).tolist()
    elapsed = time.time() - t0
    m = metrics(eval_labels[:len(preds)], preds)
    results.append({"model": "LR + TF-IDF", "time": elapsed, **m})
    with open(os.path.join(args.outdir, "lr_tfidf_preds.out"), "w") as f:
        f.writelines(str(p)+"\n" for p in preds)
    print(f"  Done in {elapsed:.1f}s  |  F1={m['f1']:.4f}")

    # ── 2. FFNN + BoW ─────────────────────────────────────────────────────────
    print("\n[2/6] FFNN + BoW ...")
    t0   = time.time()
    vec2 = CountVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    Xb_tr = torch.from_numpy(vec2.fit_transform(train_texts)
                              .toarray().astype(np.float32))
    Xb_ev = torch.from_numpy(vec2.transform(eval_texts)
                              .toarray().astype(np.float32))
    preds = train_ffnn(Xb_tr, train_labels, Xb_ev, eval_labels,
                       hidden=args.hidden_dim, epochs=args.epochs)
    elapsed = time.time() - t0
    m = metrics(eval_labels[:len(preds)], preds)
    results.append({"model": "FFNN + BoW", "time": elapsed, **m})
    with open(os.path.join(args.outdir, "ffnn_bow_preds.out"), "w") as f:
        f.writelines(str(p)+"\n" for p in preds)
    print(f"  Done in {elapsed:.1f}s  |  F1={m['f1']:.4f}")

    # ── 3. FFNN + TF-IDF ─────────────────────────────────────────────────────
    print("\n[3/6] FFNN + TF-IDF ...")
    t0   = time.time()
    vec3 = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                           sublinear_tf=True, min_df=2)
    Xt_tr = torch.from_numpy(vec3.fit_transform(train_texts)
                              .toarray().astype(np.float32))
    Xt_ev = torch.from_numpy(vec3.transform(eval_texts)
                              .toarray().astype(np.float32))
    preds = train_ffnn(Xt_tr, train_labels, Xt_ev, eval_labels,
                       hidden=args.hidden_dim, epochs=args.epochs)
    elapsed = time.time() - t0
    m = metrics(eval_labels[:len(preds)], preds)
    results.append({"model": "FFNN + TF-IDF", "time": elapsed, **m})
    with open(os.path.join(args.outdir, "ffnn_tfidf_preds.out"), "w") as f:
        f.writelines(str(p)+"\n" for p in preds)
    print(f"  Done in {elapsed:.1f}s  |  F1={m['f1']:.4f}")

    if not args.baseline_only and os.path.exists(args.embedding):
        print(f"\nLoading embeddings from {args.embedding} ...")
        word_embedding = pickle.load(open(args.embedding, "rb"))
        embed_dim      = next(iter(word_embedding.values())).shape[0]

        train_seq = [(t.split(), l) for t, l in zip(train_texts, train_labels)]
        eval_seq  = [(t.split(), l) for t, l in zip(eval_texts,  eval_labels)]

        # ── 4. RNN ───────────────────────────────────────────────────────────
        print("\n[4/6] RNN ...")
        t0    = time.time()
        model = RNNModel(embed_dim, args.hidden_dim)
        preds = train_seq_model(model, list(train_seq), list(eval_seq),
                                word_embedding, epochs=args.epochs)
        elapsed = time.time() - t0
        m = metrics(eval_labels[:len(preds)], preds)
        results.append({"model": "RNN", "time": elapsed, **m})
        with open(os.path.join(args.outdir, "rnn_preds.out"), "w") as f:
            f.writelines(str(p)+"\n" for p in preds)
        print(f"  Done in {elapsed:.1f}s  |  F1={m['f1']:.4f}")

        # ── 5. LSTM ──────────────────────────────────────────────────────────
        print("\n[5/6] LSTM ...")
        t0    = time.time()
        model = LSTMModel(embed_dim, args.hidden_dim)
        preds = train_seq_model(model, list(train_seq), list(eval_seq),
                                word_embedding, epochs=args.epochs)
        elapsed = time.time() - t0
        m = metrics(eval_labels[:len(preds)], preds)
        results.append({"model": "LSTM", "time": elapsed, **m})
        with open(os.path.join(args.outdir, "lstm_preds.out"), "w") as f:
            f.writelines(str(p)+"\n" for p in preds)
        print(f"  Done in {elapsed:.1f}s  |  F1={m['f1']:.4f}")

        # ── 6. CNN ───────────────────────────────────────────────────────────
        print("\n[6/6] TextCNN ...")
        t0    = time.time()
        model = TextCNN(embed_dim, num_filters=128)
        preds = train_seq_model(model, list(train_seq), list(eval_seq),
                                word_embedding, epochs=args.epochs, min_len=5)
        elapsed = time.time() - t0
        m = metrics(eval_labels[:len(preds)], preds)
        results.append({"model": "TextCNN", "time": elapsed, **m})
        with open(os.path.join(args.outdir, "cnn_preds.out"), "w") as f:
            f.writelines(str(p)+"\n" for p in preds)
        print(f"  Done in {elapsed:.1f}s  |  F1={m['f1']:.4f}")

    elif not args.baseline_only:
        print(f"\n[warn] {args.embedding} not found — skipping RNN/LSTM/CNN.")
        print("  Run preprocess.py with --glove to generate it.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print_table(results)
    save_chart(results, args.outdir)

    # CSV
    import csv
    csv_path = os.path.join(args.outdir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","accuracy","f1",
                                           "precision","recall","time"])
        w.writeheader(); w.writerows(results)
    print(f"Table saved  → {csv_path}")

    # JSON
    json_path = os.path.join(args.outdir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump({"split": eval_split, "models": results}, f, indent=2)
    print(f"JSON saved   → {json_path}")
