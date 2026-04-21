"""
logistic_regression.py
----------------------
Logistic Regression baseline using TF-IDF features.
Also supports plain Bag-of-Words (CountVectorizer) for direct
comparison with the FFNN BoW model.

Includes:
  - TF-IDF + Logistic Regression
  - BoW   + Logistic Regression
  - Manual oversampling (minority duplication) as SMOTE substitute
  - Full metrics: accuracy, F1, precision, recall + classification report
  - Saves results to results/lr_results.json

Usage:
    python logistic_regression.py \
        --train_data data/train.json \
        --val_data   data/val.json   \
        --test_data  data/test.json  \
        --feature    tfidf           \
        --oversample
"""

import json
import random
import argparse
import os
import numpy as np

from sklearn.linear_model   import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report,
                              confusion_matrix)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        records = json.load(f)
    texts  = [r["text"] for r in records]
    labels = [int(r["stars"]) - 1 for r in records]   # 0 or 1
    return texts, labels


# ── Manual oversampling (minority class duplication) ─────────────────────────

def oversample(texts, labels, seed=42):
    """
    Duplicates minority-class samples until both classes are balanced.
    Simple but effective substitute for SMOTE when imblearn is unavailable.
    """
    rng    = random.Random(seed)
    pairs  = list(zip(texts, labels))
    pos    = [p for p in pairs if p[1] == 1]
    neg    = [p for p in pairs if p[1] == 0]

    if len(pos) == len(neg):
        return texts, labels

    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    diff = len(majority) - len(minority)
    minority += rng.choices(minority, k=diff)

    combined = majority + minority
    rng.shuffle(combined)
    t, l = zip(*combined)
    print(f"  Oversampling: {len(minority) - diff} → {len(minority)} minority samples")
    return list(t), list(l)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X, y_true, split_name="Test"):
    y_pred    = model.predict(X)
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="binary")
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="binary", zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"{split_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["non-suicidal", "suicidal"]))
    return {"accuracy": acc, "f1": f1, "precision": precision,
            "recall": recall, "confusion_matrix": cm.tolist(),
            "predictions": y_pred.tolist()}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data",   required=True)
    parser.add_argument("--test_data",  default=None)
    parser.add_argument("--feature",    default="tfidf",
                        choices=["tfidf", "bow"],
                        help="Feature representation (default: tfidf)")
    parser.add_argument("--oversample", action="store_true",
                        help="Balance training set via minority oversampling")
    parser.add_argument("--max_features", type=int, default=30000,
                        help="Max vocabulary size for vectorizer (default 30000)")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularisation strength (default 1.0)")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    # ── Load data ──
    print("Loading data ...")
    train_texts, train_labels = load_json(args.train_data)
    val_texts,   val_labels   = load_json(args.val_data)

    print(f"  Train : {len(train_texts)}  "
          f"(pos={sum(train_labels)}, neg={train_labels.count(0)})")
    print(f"  Val   : {len(val_texts)}")

    # ── Oversample ──
    if args.oversample:
        print("Oversampling training set ...")
        train_texts, train_labels = oversample(train_texts, train_labels)

    # ── Vectorise ──
    print(f"\nBuilding {'TF-IDF' if args.feature == 'tfidf' else 'BoW'} features "
          f"(max_features={args.max_features}) ...")

    if args.feature == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features  = args.max_features,
            ngram_range   = (1, 2),      # unigrams + bigrams
            sublinear_tf  = True,        # log(1 + tf)
            min_df        = 2,
            strip_accents = "unicode",
            analyzer      = "word",
        )
    else:
        vectorizer = CountVectorizer(
            max_features = args.max_features,
            ngram_range  = (1, 2),
            min_df       = 2,
        )

    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    print(f"  Feature matrix : {X_train.shape}")

    # ── Train ──
    print(f"\nTraining Logistic Regression (C={args.C}) ...")
    clf = LogisticRegression(
        C           = args.C,
        max_iter    = 1000,
        solver      = "lbfgs",
        random_state= 42,
        class_weight= "balanced",     # handles remaining imbalance
    )
    clf.fit(X_train, train_labels)
    print("  Done.")

    # ── Evaluate ──
    val_results = evaluate(clf, X_val, val_labels, split_name="Validation")

    test_results = None
    if args.test_data:
        test_texts, test_labels = load_json(args.test_data)
        X_test       = vectorizer.transform(test_texts)
        test_results = evaluate(clf, X_test, test_labels, split_name="Test")

        # Save predictions
        os.makedirs("results", exist_ok=True)
        with open("results/lr_test.out", "w") as f:
            for p in test_results["predictions"]:
                f.write(str(p) + "\n")
        print("Predictions written → results/lr_test.out")

    # ── Save summary ──
    os.makedirs("results", exist_ok=True)
    summary = {
        "model"      : "LogisticRegression",
        "feature"    : args.feature,
        "C"          : args.C,
        "oversample" : args.oversample,
        "val"        : {k: v for k, v in val_results.items()
                        if k != "predictions"},
        "test"       : {k: v for k, v in test_results.items()
                        if k != "predictions"} if test_results else None,
    }
    out_path = "results/lr_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {out_path}")

    # ── Top features ──
    print("\nTop 20 features most predictive of SUICIDAL class:")
    feature_names = vectorizer.get_feature_names_out()
    coef          = clf.coef_[0]
    top_pos_idx   = np.argsort(coef)[-20:][::-1]
    top_neg_idx   = np.argsort(coef)[:20]
    print("  Suicidal     :", [feature_names[i] for i in top_pos_idx])
    print("  Non-suicidal :", [feature_names[i] for i in top_neg_idx])
