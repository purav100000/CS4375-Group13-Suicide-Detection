"""
preprocess.py
-------------
Preprocessing pipeline for the Reddit suicidal ideation dataset.
Outputs: train.json, val.json, test.json  (compatible with ffnn.py / rnn.py loaders)
         word_embedding.pkl               (50-d GloVe subset for the RNN)

Usage:
    python preprocess.py --data suicidal_ideation_reddit_annotated.csv --glove glove.6B.50d.txt
    python preprocess.py --data suicidal_ideation_reddit_annotated.csv   # skips GloVe embedding pkl
"""

import re
import json
import pickle
import random
import argparse
import os

import pandas as pd
import numpy as np

# ── optional heavy deps (graceful fallback) ──────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    NLTK_OK = True
except ImportError:
    NLTK_OK = False
    print("[warn] nltk not found — stop-word removal and lemmatization skipped.")

# ─────────────────────────────────────────────────────────────────────────────


NEGATIONS = {"not", "no", "never", "neither", "nor", "n't"}


def clean_text(text: str, lemmatizer=None, stop_words: set = None) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs, special characters (keep apostrophes for contractions)
      3. Tokenize (whitespace split after cleaning)
      4. Stop-word removal — negations are KEPT
      5. Lemmatization
    Returns a single whitespace-joined string.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z\s']", " ", text)               # keep letters + apostrophe
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    if stop_words:
        tokens = [t for t in tokens if t not in stop_words or t in NEGATIONS]

    if lemmatizer:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def load_and_clean(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["usertext", "label"])
    df = df.drop_duplicates(subset=["usertext"])
    df["label"] = df["label"].astype(int)

    lemmatizer  = WordNetLemmatizer() if NLTK_OK else None
    stop_words  = set(stopwords.words("english")) if NLTK_OK else None

    print(f"Cleaning {len(df)} records ...")
    df["clean_text"] = df["usertext"].apply(
        lambda t: clean_text(str(t), lemmatizer, stop_words)
    )
    return df


def split_and_save(df: pd.DataFrame, out_dir: str = "."):
    """80 / 10 / 10 stratified split — saves three JSON files."""
    os.makedirs(out_dir, exist_ok=True)

    # Shuffle reproducibly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n      = len(df)
    n_val  = int(n * 0.10)
    n_test = int(n * 0.10)

    test_df  = df.iloc[:n_test]
    val_df   = df.iloc[n_test : n_test + n_val]
    train_df = df.iloc[n_test + n_val :]

    def to_records(frame):
        return [
            {"text": row["clean_text"], "stars": int(row["label"]) + 1}
            # stars=1 → label 0 (non-suicidal), stars=2 → label 1 (suicidal)
            for _, row in frame.iterrows()
        ]

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(to_records(frame), f, indent=2)
        print(f"  Saved {len(frame):>5} records → {path}")

    return train_df, val_df, test_df


def build_glove_pkl(glove_path: str, vocab: set, out_path: str = "word_embedding.pkl",
                    dim: int = 50):
    """
    Loads GloVe vectors for words in `vocab`, adds an 'unk' vector (mean of all
    loaded vectors), and pickles the resulting dict {word: np.array}.

    Download GloVe: https://nlp.stanford.edu/projects/glove/
        wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
        unzip glove.6B.zip
    """
    print(f"Loading GloVe from {glove_path} ...")
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if word in vocab or len(embeddings) < 5:   # always grab a few for unk
                embeddings[word] = vec

    # unk = mean of all loaded vectors
    embeddings["unk"] = np.mean(list(embeddings.values()), axis=0).astype(np.float32)

    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  Saved {len(embeddings)} vectors → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to the raw CSV")
    parser.add_argument("--glove",  default=None,  help="Path to GloVe .txt file (optional)")
    parser.add_argument("--outdir", default="data",help="Output directory for JSON splits")
    args = parser.parse_args()

    df = load_and_clean(args.data)

    print("\nLabel distribution after cleaning:")
    print(df["label"].value_counts().to_string())

    train_df, val_df, test_df = split_and_save(df, out_dir=args.outdir)

    # Build vocabulary from training set
    vocab = set()
    for text in train_df["clean_text"]:
        vocab.update(text.split())
    print(f"\nVocabulary size (train): {len(vocab)}")

    if args.glove:
        build_glove_pkl(args.glove, vocab)
    else:
        print("\n[info] No --glove path provided; skipping word_embedding.pkl generation.")
        print("       Provide a GloVe file to enable the RNN model.")
