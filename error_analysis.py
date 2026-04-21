"""
error_analysis.py
-----------------
Post-hoc error analysis on model predictions vs ground truth.

Analyses:
  1. Confusion matrix with annotation
  2. False-positive examples (predicted suicidal, actually not)
  3. False-negative examples (predicted non-suicidal, actually suicidal)
  4. Text-length breakdown of errors
  5. Most common words in FP and FN examples
  6. Exports a CSV of all misclassified examples for manual review

Usage (after running a model and saving predictions):
    python error_analysis.py \
        --test_data  data/test.json        \
        --preds      results/lr_test.out   \
        --model_name "Logistic Regression" \
        --outdir     error_analysis/
"""

import os
import re
import json
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)


# ── Optional NLTK ─────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of",
        "with","is","was","are","were","be","been","i","my","me","it",
        "this","that","s","t","just","so","if","as","do","have","had",
        "has","not","no","can","will","would","could","should","from",
        "what","when","who","how","all","about","up","out","get","like",
        "know","think","want","feel","need","even","also","one","more",
    }

COLORS = {"tp": "#2CA58D", "tn": "#14747E",
          "fp": "#F4A261", "fn": "#E63946"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test(path):
    with open(path) as f:
        records = json.load(f)
    texts  = [r["text"] for r in records]
    labels = [int(r["stars"]) - 1 for r in records]
    return texts, labels


def load_preds(path):
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def clean_tokens(text):
    text   = re.sub(r"http\S+", " ", str(text).lower())
    text   = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)


# ── Analysis functions ────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name, outdir):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm,
               display_labels=["Non-suicidal", "Suicidal"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False,
              cmap="Blues",
              values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=13, fontweight="bold", pad=10)

    # Annotate with rates
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    ax.set_xlabel(
        f"Predicted Label\n\n"
        f"Accuracy={( tn+tp)/total:.3f}  "
        f"FPR={fp/(fp+tn):.3f}  "
        f"FNR={fn/(fn+tp):.3f}",
        fontsize=9)

    plt.tight_layout()
    path = os.path.join(outdir, "cm_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return tn, fp, fn, tp


def plot_error_length(fp_texts, fn_texts, all_texts, all_labels, outdir):
    """Compare word-count distribution of FP/FN vs correctly classified."""
    correct_texts = [t for t, l, p in zip(all_texts, all_labels,
                     [None]*len(all_texts))
                     if True]   # placeholder — handled inline below

    fp_lens = [len(t.split()) for t in fp_texts]
    fn_lens = [len(t.split()) for t in fn_texts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, lens, label, color in [
        (axes[0], fp_lens, "False Positives\n(predicted suicidal, not)",  COLORS["fp"]),
        (axes[1], fn_lens, "False Negatives\n(predicted safe, actually suicidal)", COLORS["fn"]),
    ]:
        if lens:
            ax.hist([min(l, 500) for l in lens], bins=40,
                    color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.axvline(np.mean(lens), color="black", linestyle="--",
                       linewidth=1.5, label=f"Mean={np.mean(lens):.0f}")
            ax.legend(fontsize=9)
        style_ax(ax, label, xlabel="Word Count", ylabel="Count")

    fig.suptitle("Word Count Distribution of Misclassified Examples",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "error_length_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_error_words(fp_texts, fn_texts, outdir, n=25):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, texts, label, color in [
        (axes[0], fp_texts, "Top Words in False Positives",  COLORS["fp"]),
        (axes[1], fn_texts, "Top Words in False Negatives",  COLORS["fn"]),
    ]:
        if not texts:
            ax.text(0.5, 0.5, "No examples", transform=ax.transAxes,
                    ha="center", fontsize=12)
            continue
        tokens  = []
        for t in texts:
            tokens.extend(clean_tokens(t))
        counter = collections.Counter(tokens)
        words, freqs = zip(*counter.most_common(n))
        y = np.arange(len(words))
        ax.barh(y, freqs, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_yticks(y); ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        style_ax(ax, label, xlabel="Frequency")

    fig.suptitle("Most Frequent Words in Misclassified Examples",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "error_word_frequency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def export_misclassified(texts, y_true, y_pred, outdir):
    rows = []
    for i, (text, gold, pred) in enumerate(zip(texts, y_true, y_pred)):
        if gold != pred:
            error_type = "FP" if pred == 1 else "FN"
            rows.append({
                "index"      : i,
                "error_type" : error_type,
                "true_label" : "suicidal" if gold == 1 else "non-suicidal",
                "pred_label" : "suicidal" if pred == 1 else "non-suicidal",
                "word_count" : len(text.split()),
                "text"       : text,
            })

    df = pd.DataFrame(rows)
    path = os.path.join(outdir, "misclassified_examples.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(df)} misclassified examples)")
    return df


def print_examples(texts, y_true, y_pred, error_type="FP", n=5):
    """Print n examples of a specific error type."""
    label_map = {0: "non-suicidal", 1: "suicidal"}
    print(f"\n{'─'*60}")
    print(f"  {n} {error_type} Examples")
    print(f"{'─'*60}")
    count = 0
    for text, gold, pred in zip(texts, y_true, y_pred):
        if error_type == "FP" and gold == 0 and pred == 1:
            pass
        elif error_type == "FN" and gold == 1 and pred == 0:
            pass
        else:
            continue
        snippet = text[:300] + ("..." if len(text) > 300 else "")
        print(f"\n  True: {label_map[gold]}  |  Pred: {label_map[pred]}"
              f"  |  Words: {len(text.split())}")
        print(f"  \"{snippet}\"")
        count += 1
        if count >= n:
            break


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",  required=True)
    parser.add_argument("--preds",      required=True,
                        help="Path to predictions file (one int per line)")
    parser.add_argument("--model_name", default="Model")
    parser.add_argument("--outdir",     default="error_analysis")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load ──
    texts,  y_true = load_test(args.test_data)
    y_pred         = load_preds(args.preds)

    # Align lengths (in case model skipped last incomplete minibatch)
    n = min(len(texts), len(y_true), len(y_pred))
    texts, y_true, y_pred = texts[:n], y_true[:n], y_pred[:n]

    print(f"\nAnalysing {n} predictions for: {args.model_name}")
    print("=" * 55)

    # ── Metrics ──
    tn, fp, fn, tp = plot_confusion_matrix(y_true, y_pred,
                                            args.model_name, args.outdir)
    print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["non-suicidal", "suicidal"]))

    # Collect error subsets
    fp_texts = [t for t, g, p in zip(texts, y_true, y_pred) if g==0 and p==1]
    fn_texts = [t for t, g, p in zip(texts, y_true, y_pred) if g==1 and p==0]

    print(f"\n  False Positives : {len(fp_texts):>4} "
          f"({len(fp_texts)/n*100:.1f}%)")
    print(f"  False Negatives : {len(fn_texts):>4} "
          f"({len(fn_texts)/n*100:.1f}%)")

    # ── Plots ──
    print("\nGenerating error analysis plots ...")
    plot_error_length(fp_texts, fn_texts, texts, y_true, args.outdir)
    plot_error_words(fp_texts, fn_texts, args.outdir)

    # ── Export CSV ──
    df_errors = export_misclassified(texts, y_true, y_pred, args.outdir)

    # ── Print qualitative examples ──
    print_examples(texts, y_true, y_pred, error_type="FP", n=5)
    print_examples(texts, y_true, y_pred, error_type="FN", n=5)

    # ── Length analysis ──
    print(f"\n{'─'*55}")
    print("  Error breakdown by text length")
    print(f"{'─'*55}")
    bins = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 9999)]
    print(f"  {'Length Range':<20} {'FP':>6} {'FN':>6} {'Total Errors':>14}")
    for lo, hi in bins:
        fp_n = sum(1 for t in fp_texts if lo <= len(t.split()) < hi)
        fn_n = sum(1 for t in fn_texts if lo <= len(t.split()) < hi)
        print(f"  {f'{lo}–{hi} words':<20} {fp_n:>6} {fn_n:>6} {fp_n+fn_n:>14}")

    print(f"\nAll outputs saved to ./{args.outdir}/")
