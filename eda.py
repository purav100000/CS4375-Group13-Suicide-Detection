"""
eda.py
------
Exploratory Data Analysis for the Reddit suicidal ideation dataset.

Produces (saved to plots/):
  1. Class distribution bar chart
  2. Text length distributions (by class)
  3. Top-40 word frequency bar charts (suicidal vs non-suicidal)
  4. Word frequency heatmap (top words × class)
  5. Summary statistics printed to console

Usage:
    python eda.py --data suicidal_ideation_reddit_annotated.csv
    python eda.py --data suicidal_ideation_reddit_annotated.csv --outdir my_plots
"""

import re
import os
import argparse
import collections

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        "its","this","that","he","she","they","we","you","your","his",
        "her","their","our","s","t","just","so","if","as","do","have",
        "had","has","not","no","can","will","would","could","should",
        "from","what","when","who","how","all","about","up","out","go",
        "get","got","like","know","think","want","feel","need","really",
        "even","also","one","more","into","there","than","because","after",
        "going","m","re","ve","ll","d","don","didn","doesn","isn","wasn",
        "weren","hasn","haven","hadn","won","wouldn","couldn","shouldn",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_tokens(text: str):
    text   = str(text).lower()
    text   = re.sub(r"http\S+|www\S+", " ", text)
    text   = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return tokens


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)


COLORS = {"suicidal": "#2CA58D", "non_suicidal": "#0D2137",
          "bar1": "#14747E",     "bar2": "#F4A261"}


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_class_distribution(df, outdir):
    counts = df["label"].value_counts().sort_index()
    labels = ["Non-suicidal (0)", "Suicidal (1)"]
    colors = [COLORS["non_suicidal"], COLORS["suicidal"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5,
                       edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", va="bottom", fontsize=11,
                     fontweight="bold")
    style_ax(axes[0], "Class Distribution", ylabel="Count")
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight("bold")
    axes[1].set_title("Class Proportion", fontsize=13, fontweight="bold")

    fig.suptitle("Reddit Suicidal Ideation Dataset — Class Balance",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(outdir, "01_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_text_length(df, outdir):
    df = df.copy()
    df["word_count"] = df["usertext"].apply(lambda t: len(str(t).split()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for cls, label, color in [(1, "Suicidal",     COLORS["suicidal"]),
                               (0, "Non-suicidal", COLORS["non_suicidal"])]:
        subset = df[df["label"] == cls]["word_count"]
        axes[0].hist(subset.clip(upper=500), bins=60, alpha=0.65,
                     label=label, color=color, edgecolor="white", linewidth=0.3)

    style_ax(axes[0], "Word Count Distribution (capped at 500)",
             xlabel="Word Count", ylabel="Frequency")
    axes[0].legend()

    # Box plot
    data = [df[df["label"] == 1]["word_count"].values,
            df[df["label"] == 0]["word_count"].values]
    bp = axes[1].boxplot(data, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2),
                          labels=["Suicidal", "Non-suicidal"])
    for patch, color in zip(bp["boxes"], [COLORS["suicidal"],
                                            COLORS["non_suicidal"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    style_ax(axes[1], "Word Count Box Plot", ylabel="Word Count")

    # Print stats
    print("\n  Text length statistics (word count):")
    for cls, name in [(1, "Suicidal"), (0, "Non-suicidal")]:
        s = df[df["label"] == cls]["word_count"]
        print(f"    {name:14s}: mean={s.mean():.1f}  median={s.median():.1f}"
              f"  max={s.max()}  std={s.std():.1f}")

    plt.tight_layout()
    path = os.path.join(outdir, "02_text_length_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_top_words(df, outdir, n=40):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, cls, label, color in [
        (axes[0], 1, "Suicidal",     COLORS["suicidal"]),
        (axes[1], 0, "Non-suicidal", COLORS["non_suicidal"]),
    ]:
        tokens = []
        for text in df[df["label"] == cls]["usertext"]:
            tokens.extend(clean_tokens(text))

        counter = collections.Counter(tokens)
        words, freqs = zip(*counter.most_common(n))

        # Horizontal bar
        y_pos = np.arange(len(words))
        bars  = ax.barh(y_pos, freqs, color=color, alpha=0.85,
                        edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        for bar, freq in zip(bars, freqs):
            ax.text(bar.get_width() + max(freqs) * 0.01, bar.get_y() +
                    bar.get_height() / 2, str(freq), va="center", fontsize=7)
        style_ax(ax, f"Top {n} Words — {label}", xlabel="Frequency")
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Most Frequent Words by Class (stop-words removed)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "03_top_words.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_word_heatmap(df, outdir, n=30):
    """
    Heatmap showing normalised frequency of the top words for each class.
    Highlights words that are distinctive to one class.
    """
    counters = {}
    totals   = {}
    for cls in [0, 1]:
        tokens = []
        for text in df[df["label"] == cls]["usertext"]:
            tokens.extend(clean_tokens(text))
        counters[cls] = collections.Counter(tokens)
        totals[cls]   = sum(counters[cls].values())

    # Union of top-n words from each class
    top_words = set()
    for cls in [0, 1]:
        top_words.update(w for w, _ in counters[cls].most_common(n))
    top_words = sorted(top_words)

    matrix = np.array([
        [counters[cls].get(w, 0) / totals[cls] * 1000 for w in top_words]
        for cls in [1, 0]
    ])

    fig, ax = plt.subplots(figsize=(max(12, len(top_words) * 0.45), 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Suicidal", "Non-suicidal"], fontsize=11)
    ax.set_xticks(range(len(top_words)))
    ax.set_xticklabels(top_words, rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, label="Frequency per 1 000 tokens")
    ax.set_title("Word Frequency Heatmap (per 1 000 tokens)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "04_word_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_unique_words(df, outdir, n=25):
    """Words with the highest relative frequency ratio between the two classes."""
    counters = {}
    totals   = {}
    for cls in [0, 1]:
        tokens = []
        for text in df[df["label"] == cls]["usertext"]:
            tokens.extend(clean_tokens(text))
        counters[cls] = collections.Counter(tokens)
        totals[cls]   = sum(counters[cls].values())

    # Ratio: freq_suicidal / (freq_non_suicidal + epsilon)
    all_words = set(counters[0]) | set(counters[1])
    ratios    = {}
    for w in all_words:
        f1 = counters[1].get(w, 0) / totals[1]
        f0 = counters[0].get(w, 0) / totals[0]
        if counters[1].get(w, 0) >= 20:    # min frequency filter
            ratios[w] = f1 / (f0 + 1e-9)

    top_suicidal     = sorted(ratios, key=ratios.get, reverse=True)[:n]
    top_non_suicidal = sorted(ratios, key=ratios.get)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, words, label, color in [
        (axes[0], top_suicidal,     "Most Distinctive — Suicidal",     COLORS["suicidal"]),
        (axes[1], top_non_suicidal, "Most Distinctive — Non-suicidal", COLORS["non_suicidal"]),
    ]:
        vals  = [ratios[w] for w in words]
        y_pos = np.arange(len(words))
        ax.barh(y_pos, vals, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        style_ax(ax, label, xlabel="Frequency ratio (suicidal / non-suicidal)")

    fig.suptitle("Words Most Distinctive to Each Class",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "05_distinctive_words.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to raw CSV")
    parser.add_argument("--outdir", default="plots", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data ...")
    df = pd.read_csv(args.data).dropna(subset=["usertext", "label"])
    df["label"] = df["label"].astype(int)
    print(f"  {len(df)} records  |  "
          f"suicidal={df['label'].sum()}  "
          f"non-suicidal={(df['label']==0).sum()}")

    print("\nGenerating plots ...")
    plot_class_distribution(df, args.outdir)
    plot_text_length(df, args.outdir)
    plot_top_words(df, args.outdir)
    plot_word_heatmap(df, args.outdir)
    plot_unique_words(df, args.outdir)

    print(f"\nAll plots saved to ./{args.outdir}/")

    # ── Console summary ──
    df["word_count"] = df["usertext"].apply(lambda t: len(str(t).split()))
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"  Total records     : {len(df):,}")
    print(f"  Suicidal (1)      : {df['label'].sum():,} "
          f"({df['label'].mean()*100:.1f}%)")
    print(f"  Non-suicidal (0)  : {(df['label']==0).sum():,} "
          f"({(df['label']==0).mean()*100:.1f}%)")
    print(f"  Avg word count    : {df['word_count'].mean():.1f}")
    print(f"  Median word count : {df['word_count'].median():.1f}")
    print(f"  Max word count    : {df['word_count'].max()}")
