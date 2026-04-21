"""
smote_balance.py
----------------
Class imbalance handling for the Reddit suicide detection dataset.

Strategies implemented (no imblearn required):
  1. Random oversampling  — duplicate minority samples randomly
  2. Random undersampling — drop majority samples randomly
  3. Manual SMOTE         — interpolate between minority sample pairs
  4. Class-weighted loss  — pass pos_weight to BCELoss / weight to NLLLoss
  5. Augmentation         — synonym swap + random word deletion on minority

Outputs balanced train.json variants:
    data/train_oversampled.json
    data/train_undersampled.json
    data/train_smote.json
    data/train_augmented.json

Also prints before/after class distribution for each strategy.

Usage:
    python smote_balance.py \
        --train_data data/train.json \
        --strategy   all
"""

import os
import json
import copy
import random
import argparse
import numpy as np
from collections import Counter


# ── Load / save ───────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    labels = [r["stars"] - 1 for r in records]
    c = Counter(labels)
    print(f"  Saved {len(records):>6} records → {path}  "
          f"(class 0: {c[0]}, class 1: {c[1]})")


def split_by_class(records):
    pos = [r for r in records if r["stars"] == 2]   # suicidal
    neg = [r for r in records if r["stars"] == 1]   # non-suicidal
    return pos, neg


# ── Strategy 1: Random oversampling ──────────────────────────────────────────

def random_oversample(records, seed=42):
    rng = random.Random(seed)
    pos, neg = split_by_class(records)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    diff     = len(majority) - len(minority)
    minority = minority + rng.choices(minority, k=diff)
    result   = majority + minority
    rng.shuffle(result)
    return result


# ── Strategy 2: Random undersampling ─────────────────────────────────────────

def random_undersample(records, seed=42):
    rng = random.Random(seed)
    pos, neg = split_by_class(records)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    majority_sampled = rng.sample(majority, len(minority))
    result = minority + majority_sampled
    rng.shuffle(result)
    return result


# ── Strategy 3: Manual SMOTE (text interpolation) ────────────────────────────

def text_smote(records, seed=42, k=5):
    """
    Text-space SMOTE: for each minority sample, find k nearest neighbours
    by Jaccard similarity on token sets, then create a synthetic sample by
    taking the union of a random pair's overlapping tokens + unique tokens
    from one side (approximates feature-space interpolation in text).
    """
    rng = random.Random(seed)
    pos, neg = split_by_class(records)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    diff = len(majority) - len(minority)

    def jaccard(a, b):
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / (len(sa | sb) + 1e-9)

    def synthetic(r1, r2):
        """Blend two texts: shared words + random subset of unique words."""
        t1, t2   = r1["text"].split(), r2["text"].split()
        s1, s2   = set(t1), set(t2)
        shared   = list(s1 & s2)
        unique1  = list(s1 - s2)
        unique2  = list(s2 - s1)
        # Take all shared words + ~50% of unique from each side
        blend    = shared
        blend   += rng.sample(unique1, max(0, len(unique1) // 2))
        blend   += rng.sample(unique2, max(0, len(unique2) // 2))
        rng.shuffle(blend)
        new_rec  = copy.deepcopy(r1)
        new_rec["text"] = " ".join(blend) if blend else r1["text"]
        return new_rec

    print(f"  Running text-SMOTE: generating {diff} synthetic samples ...")
    synthetic_samples = []
    minority_texts    = [r["text"] for r in minority]

    for _ in range(diff):
        anchor  = rng.choice(minority)
        # Find k nearest neighbours by jaccard
        scores  = [(jaccard(anchor["text"], t), i)
                   for i, t in enumerate(minority_texts)]
        scores.sort(reverse=True)
        # Pick randomly from top-k (excluding self)
        top_k   = [minority[i] for _, i in scores[1:k+1]]
        if not top_k:
            synthetic_samples.append(copy.deepcopy(anchor))
        else:
            neighbour = rng.choice(top_k)
            synthetic_samples.append(synthetic(anchor, neighbour))

    result = majority + minority + synthetic_samples
    rng.shuffle(result)
    return result


# ── Strategy 4: Text augmentation ────────────────────────────────────────────

# Simple synonym map for common emotional/distress words
SYNONYM_MAP = {
    "sad":       ["unhappy", "depressed", "miserable", "sorrowful"],
    "happy":     ["glad", "joyful", "pleased", "content"],
    "tired":     ["exhausted", "weary", "drained", "fatigued"],
    "hurt":      ["pain", "ache", "wounded", "harmed"],
    "alone":     ["lonely", "isolated", "solitary", "abandoned"],
    "help":      ["support", "assist", "aid", "rescue"],
    "die":       ["perish", "pass away", "end", "cease"],
    "life":      ["existence", "living", "world", "journey"],
    "feel":      ["sense", "experience", "notice", "perceive"],
    "think":     ["believe", "consider", "feel", "suppose"],
    "want":      ["desire", "wish", "need", "hope"],
    "bad":       ["terrible", "awful", "horrible", "dreadful"],
    "good":      ["fine", "okay", "decent", "alright"],
    "people":    ["others", "everyone", "folks", "individuals"],
    "family":    ["relatives", "loved ones", "kin", "household"],
    "friend":    ["companion", "buddy", "pal", "acquaintance"],
    "love":      ["care", "cherish", "adore", "value"],
    "hate":      ["despise", "loathe", "dislike", "resent"],
    "afraid":    ["scared", "fearful", "terrified", "anxious"],
    "angry":     ["furious", "upset", "mad", "irritated"],
}


def augment_text(text, rng, swap_prob=0.15, delete_prob=0.1):
    """
    Two augmentations:
      - Synonym swap: replace a word with a synonym (swap_prob per token)
      - Random deletion: drop a token (delete_prob per token)
    """
    tokens = text.split()
    result = []
    for token in tokens:
        t_lower = token.lower()
        if rng.random() < delete_prob and len(tokens) > 5:
            continue     # drop token
        elif t_lower in SYNONYM_MAP and rng.random() < swap_prob:
            result.append(rng.choice(SYNONYM_MAP[t_lower]))
        else:
            result.append(token)
    return " ".join(result) if result else text


def augmentation_oversample(records, seed=42, augment_factor=1):
    """
    For each minority sample, create `augment_factor` augmented copies.
    Continues until minority == majority size.
    """
    rng = random.Random(seed)
    pos, neg = split_by_class(records)
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    diff = len(majority) - len(minority)

    augmented = []
    pool      = minority * (augment_factor + 1)
    rng.shuffle(pool)

    for i in range(diff):
        src     = pool[i % len(pool)]
        new_rec = copy.deepcopy(src)
        new_rec["text"] = augment_text(src["text"], rng)
        augmented.append(new_rec)

    result = majority + minority + augmented
    rng.shuffle(result)
    return result


# ── Class weight helper (for use in model training) ───────────────────────────

def compute_class_weights(records):
    """
    Returns pos_weight for BCEWithLogitsLoss or weight tensor for NLLLoss.
    pos_weight = num_negative / num_positive
    """
    labels    = [r["stars"] - 1 for r in records]
    n_pos     = sum(labels)
    n_neg     = len(labels) - n_pos
    pos_weight = n_neg / (n_pos + 1e-9)
    print(f"\n  Class weight info:")
    print(f"    Positive (suicidal)     : {n_pos}")
    print(f"    Negative (non-suicidal) : {n_neg}")
    print(f"    pos_weight (neg/pos)    : {pos_weight:.3f}")
    print(f"\n  Usage in PyTorch:")
    print(f"    # For NLLLoss:")
    print(f"    weight = torch.tensor([1.0, {pos_weight:.3f}])")
    print(f"    loss_fn = nn.NLLLoss(weight=weight)")
    print(f"\n    # For BCEWithLogitsLoss:")
    print(f"    pos_weight = torch.tensor([{pos_weight:.3f}])")
    print(f"    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)")
    return pos_weight


# ── Summary stats ─────────────────────────────────────────────────────────────

def print_distribution(records, label=""):
    labels = [r["stars"] - 1 for r in records]
    c      = Counter(labels)
    total  = len(labels)
    print(f"  {label:<20}: total={total:>6}  "
          f"suicidal={c[1]:>5} ({c[1]/total*100:.1f}%)  "
          f"non-suicidal={c[0]:>5} ({c[0]/total*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--strategy",   default="all",
                        choices=["all", "oversample", "undersample",
                                 "smote", "augment", "weights"])
    parser.add_argument("--outdir",     default="data")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42)

    print("Loading training data ...")
    records = load_json(args.train_data)
    print_distribution(records, "Original")

    run_all = args.strategy == "all"

    if run_all or args.strategy == "oversample":
        print("\n[1] Random Oversampling")
        result = random_oversample(records)
        print_distribution(result, "After")
        save_json(result, os.path.join(args.outdir, "train_oversampled.json"))

    if run_all or args.strategy == "undersample":
        print("\n[2] Random Undersampling")
        result = random_undersample(records)
        print_distribution(result, "After")
        save_json(result, os.path.join(args.outdir, "train_undersampled.json"))

    if run_all or args.strategy == "smote":
        print("\n[3] Text SMOTE")
        result = text_smote(records)
        print_distribution(result, "After")
        save_json(result, os.path.join(args.outdir, "train_smote.json"))

    if run_all or args.strategy == "augment":
        print("\n[4] Augmentation Oversampling")
        result = augmentation_oversample(records)
        print_distribution(result, "After")
        save_json(result, os.path.join(args.outdir, "train_augmented.json"))

    if run_all or args.strategy == "weights":
        print("\n[5] Class Weights (no new file — use weights in model)")
        compute_class_weights(records)

    print("\nDone. Use any balanced train file as --train_data in your models.")
