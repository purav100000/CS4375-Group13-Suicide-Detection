"""
hyperparameter_tuning.py
------------------------
Grid search over hidden_dim, learning_rate, and dropout for FFNN and LSTM.
Logs all results to results/hparam_results.csv and prints the best config.

Usage:
    python hyperparameter_tuning.py \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --model      ffnn            \
        --epochs     10

    python hyperparameter_tuning.py \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --model      lstm            \
        --embedding  word_embedding.pkl \
        --epochs     10
"""

import os
import csv
import json
import time
import random
import string
import pickle
import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

# ── Data ──────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        records = json.load(f)
    return [r["text"] for r in records], [int(r["stars"]) - 1 for r in records]


def words_to_tensor(words, word_embedding):
    cleaned = " ".join(words).translate(
        str.maketrans("", "", string.punctuation)).split()
    vecs = [word_embedding.get(w.lower(), word_embedding["unk"]) for w in cleaned]
    if not vecs:
        vecs = [word_embedding["unk"]]
    return torch.tensor(np.array(vecs), dtype=torch.float32)


# ── Models ────────────────────────────────────────────────────────────────────

class FFNN(nn.Module):
    def __init__(self, input_dim, h, dropout=0.3):
        super().__init__()
        self.net     = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, 2))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss    = nn.NLLLoss()

    def forward(self, x):
        return self.softmax(self.net(x))

    def compute_Loss(self, p, g):
        return self.loss(p, g)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, h, dropout=0.3):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, h, 1, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(h, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss    = nn.NLLLoss()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.softmax(self.fc(self.dropout(h[-1])))

    def compute_Loss(self, p, g):
        return self.loss(p, g)


# ── Training ──────────────────────────────────────────────────────────────────

def train_eval_ffnn(X_tr, y_tr, X_val, y_val, h, lr, dropout, epochs, mb=16):
    model = FFNN(X_tr.shape[1], h, dropout)
    opt   = optim.Adam(model.parameters(), lr=lr)
    N     = len(y_tr)
    data  = list(zip(range(N), y_tr))

    for _ in range(epochs):
        model.train()
        random.shuffle(data)
        for start in range(0, N - mb + 1, mb):
            opt.zero_grad()
            loss = None
            for idx, gold in data[start:start + mb]:
                pred    = model(X_tr[idx])
                ex_loss = model.compute_Loss(pred.view(1, -1), torch.tensor([gold]))
                loss    = ex_loss if loss is None else loss + ex_loss
            (loss / mb).backward(); opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(y_val)):
            preds.append(torch.argmax(model(X_val[i])).item())
    return f1_score(y_val[:len(preds)], preds, average="binary")


def train_eval_lstm(train_data, val_data, word_emb, h, lr, dropout, epochs, mb=16):
    embed_dim = next(iter(word_emb.values())).shape[0]
    model     = LSTMModel(embed_dim, h, dropout)
    opt       = optim.Adam(model.parameters(), lr=lr)
    N         = len(train_data)

    for _ in range(epochs):
        model.train()
        random.shuffle(train_data)
        for start in range(0, N - mb + 1, mb):
            opt.zero_grad()
            loss = None
            for words, gold in train_data[start:start + mb]:
                t       = words_to_tensor(words, word_emb).unsqueeze(1)
                pred    = model(t)
                ex_loss = model.compute_Loss(pred.view(1, -1), torch.tensor([gold]))
                loss    = ex_loss if loss is None else loss + ex_loss
            if loss:
                (loss / mb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for words, _ in val_data:
            t = words_to_tensor(words, word_emb).unsqueeze(1)
            preds.append(torch.argmax(model(t)).item())
    labels = [l for _, l in val_data]
    return f1_score(labels[:len(preds)], preds, average="binary")


# ── Grid search ───────────────────────────────────────────────────────────────

GRID = {
    "hidden_dim"    : [64, 128, 256],
    "learning_rate" : [1e-2, 1e-3, 5e-4],
    "dropout"       : [0.2, 0.3, 0.5],
}


def run_grid_search(model_type, train_texts, train_labels,
                    val_texts, val_labels, epochs, word_emb=None):
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    total  = len(combos)
    print(f"\nGrid search: {total} combinations × {epochs} epochs each")
    print(f"Parameters : {GRID}\n")

    results = []

    # Build TF-IDF once for FFNN
    if model_type == "ffnn":
        vec    = TfidfVectorizer(max_features=20000, ngram_range=(1,2),
                                 sublinear_tf=True, min_df=2)
        X_tr   = torch.from_numpy(
            vec.fit_transform(train_texts).toarray().astype(np.float32))
        X_val  = torch.from_numpy(
            vec.transform(val_texts).toarray().astype(np.float32))
    else:
        train_seq = [(t.split(), l) for t, l in zip(train_texts, train_labels)]
        val_seq   = [(t.split(), l) for t, l in zip(val_texts,   val_labels)]

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        print(f"  [{i+1:>2}/{total}] {cfg} ", end="", flush=True)
        t0  = time.time()

        try:
            if model_type == "ffnn":
                f1 = train_eval_ffnn(
                    X_tr, train_labels, X_val, val_labels,
                    h=cfg["hidden_dim"], lr=cfg["learning_rate"],
                    dropout=cfg["dropout"], epochs=epochs)
            else:
                f1 = train_eval_lstm(
                    list(train_seq), list(val_seq), word_emb,
                    h=cfg["hidden_dim"], lr=cfg["learning_rate"],
                    dropout=cfg["dropout"], epochs=epochs)
        except Exception as e:
            print(f"ERROR: {e}")
            f1 = 0.0

        elapsed = time.time() - t0
        print(f"→ F1={f1:.4f}  ({elapsed:.1f}s)")
        results.append({**cfg, "f1": round(f1, 4), "time_s": round(elapsed, 1)})

    results.sort(key=lambda r: r["f1"], reverse=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data",   required=True)
    parser.add_argument("--model",      default="ffnn",
                        choices=["ffnn", "lstm"])
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--embedding",  default="word_embedding.pkl")
    parser.add_argument("--outdir",     default="results")
    args = parser.parse_args()

    random.seed(42); torch.manual_seed(42); np.random.seed(42)
    os.makedirs(args.outdir, exist_ok=True)

    train_texts, train_labels = load_json(args.train_data)
    val_texts,   val_labels   = load_json(args.val_data)

    word_emb = None
    if args.model == "lstm":
        print(f"Loading embeddings from {args.embedding} ...")
        word_emb = pickle.load(open(args.embedding, "rb"))

    results = run_grid_search(
        args.model, train_texts, train_labels,
        val_texts, val_labels, args.epochs, word_emb)

    # ── Print top 5 ──
    print(f"\n{'='*55}")
    print(f"Top 5 configurations ({args.model.upper()})")
    print(f"{'='*55}")
    header = f"  {'hidden':>6} {'lr':>8} {'dropout':>8} {'F1':>8} {'time':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in results[:5]:
        print(f"  {r['hidden_dim']:>6} {r['learning_rate']:>8} "
              f"{r['dropout']:>8} {r['f1']:>8.4f} {r['time_s']:>7.1f}s")

    best = results[0]
    print(f"\n★ Best config: hidden={best['hidden_dim']}  "
          f"lr={best['learning_rate']}  dropout={best['dropout']}  "
          f"F1={best['f1']:.4f}")

    # ── Save CSV ──
    out_path = os.path.join(args.outdir, f"hparam_{args.model}_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(GRID.keys()) + ["f1", "time_s"])
        w.writeheader(); w.writerows(results)
    print(f"\nFull results saved → {out_path}")
