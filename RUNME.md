# How to Run — VS Code Setup Guide
## ML for Suicide Prevention | Group 13

---

## Prerequisites — Install Once

Open the **VS Code terminal** (`Ctrl+`` ` or Terminal → New Terminal`) and run:

```bash
conda activate base
pip install torch torchvision tqdm scikit-learn matplotlib pandas numpy nltk transformers
```

Then download NLTK data (run once):
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

Download GloVe embeddings (required for RNN, LSTM, CNN, BERT):
```bash
# Download from Stanford (~822 MB)
curl -O https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip
# You only need glove.6B.50d.txt — the others can be deleted
```

---

## Step 1 — Open the Folder in VS Code

1. Open VS Code
2. **File → Open Folder** → select your project folder containing all `.py` files
3. Open the integrated terminal: `Ctrl+`` `
4. Make sure you see your project files in the Explorer panel on the left

---

## Step 2 — Preprocess the Data

Cleans the CSV, runs the full NLP pipeline, and creates the JSON splits and GloVe embedding pickle.

```bash
python preprocess.py \
  --data suicidal_ideation_reddit_annotated.csv \
  --glove glove.6B.50d.txt \
  --outdir data
```

**Output:**
```
data/
  train.json       (10,092 records)
  val.json         (1,262 records)
  test.json        (1,262 records)
word_embedding.pkl (GloVe 50d vocab subset)
```

---

## Step 3 — Run EDA (Exploratory Data Analysis)

Generates 5 plots into a `plots/` folder.

```bash
python eda.py \
  --data suicidal_ideation_reddit_annotated.csv \
  --outdir plots
```

**Output files:**
- `plots/01_class_distribution.png`
- `plots/02_text_length_distribution.png`
- `plots/03_top_words.png`
- `plots/04_word_heatmap.png`
- `plots/05_distinctive_words.png`

---

## Step 4 — Handle Class Imbalance (Optional)

Creates balanced training set variants.

```bash
python smote_balance.py \
  --train_data data/train.json \
  --strategy all \
  --outdir data
```

**Output:**
```
data/train_oversampled.json
data/train_undersampled.json
data/train_smote.json
data/train_augmented.json
```

To use a balanced set in any model, pass e.g. `--train_data data/train_smote.json`.

---

## Step 5 — Run Individual Models

### 5a. Logistic Regression (fastest, ~30 sec)
```bash
python logistic_regression.py \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json \
  --feature tfidf
```

### 5b. FFNN + BoW
```bash
python ffnn.py \
  -hd 128 -e 10 \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json
```

### 5c. FFNN + TF-IDF
```bash
python ffnn_tfidf.py \
  -hd 128 -e 10 \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json \
  --feature tfidf
```

### 5d. RNN
```bash
python rnn.py \
  -hd 128 -e 10 \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json \
  --model_type rnn
```

### 5e. LSTM
```bash
python rnn.py \
  -hd 128 -e 10 \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json \
  --model_type lstm
```

### 5f. TextCNN
```bash
python cnn.py \
  -e 10 \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json
```

### 5g. BERT (slowest — needs GPU or ~2 hrs on CPU)
```bash
pip install transformers   # if not already installed
python bert_finetune.py \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json \
  --model_name distilbert-base-uncased \
  --epochs 4
```

---

## Step 6 — Hyperparameter Tuning

Runs a 27-configuration grid search and saves a ranked CSV.

```bash
# Tune FFNN
python hyperparameter_tuning.py \
  --train_data data/train.json \
  --val_data   data/val.json \
  --model ffnn \
  --epochs 10

# Tune LSTM
python hyperparameter_tuning.py \
  --train_data data/train.json \
  --val_data   data/val.json \
  --model lstm \
  --epochs 10
```

**Output:** `results/hparam_ffnn_results.csv` and `results/hparam_lstm_results.csv`

---

## Step 7 — Run Full Comparison (All Models at Once)

Runs all models sequentially and produces a side-by-side table and bar chart.

```bash
# With all models (requires word_embedding.pkl):
python compare_models.py \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json \
  --embedding  word_embedding.pkl \
  --epochs 10

# Baseline-only (no GloVe needed):
python compare_models.py \
  --train_data data/train.json \
  --val_data   data/val.json \
  --test_data  data/test.json \
  --baseline_only
```

**Output:**
```
results/comparison_table.csv
results/comparison_chart.png
results/comparison_results.json
results/lr_tfidf_preds.out
results/ffnn_bow_preds.out
results/ffnn_tfidf_preds.out
results/rnn_preds.out
results/lstm_preds.out
results/cnn_preds.out
```

---

## Step 8 — Error Analysis

Run on any model's `.out` predictions file. Example for LR:

```bash
python error_analysis.py \
  --test_data  data/test.json \
  --preds      results/lr_tfidf_preds.out \
  --model_name "LR + TF-IDF" \
  --outdir     error_analysis/lr

# Run for BERT predictions too:
python error_analysis.py \
  --test_data  data/test.json \
  --preds      results/bert_test.out \
  --model_name "BERT" \
  --outdir     error_analysis/bert
```

**Output per model:**
```
error_analysis/<model>/
  cm_confusion_matrix.png
  error_length_distribution.png
  error_word_frequency.png
  misclassified_examples.csv
```

---

## Full Project File Map

```
project/
│
├── suicidal_ideation_reddit_annotated.csv   ← raw dataset
├── glove.6B.50d.txt                         ← GloVe embeddings (download)
│
├── preprocess.py          ← Step 2: clean data, create splits
├── eda.py                 ← Step 3: exploratory data analysis
├── smote_balance.py       ← Step 4: class imbalance handling
│
├── logistic_regression.py ← Step 5a: LR + TF-IDF baseline
├── ffnn.py                ← Step 5b: FFNN + BoW
├── ffnn_tfidf.py          ← Step 5c: FFNN + TF-IDF
├── rnn.py                 ← Step 5d/e: RNN or LSTM
├── cnn.py                 ← Step 5f: TextCNN
├── bert_finetune.py       ← Step 5g: fine-tuned BERT
│
├── hyperparameter_tuning.py  ← Step 6: grid search
├── compare_models.py         ← Step 7: run all models, print table
├── error_analysis.py         ← Step 8: FP/FN analysis
│
├── data/                  ← created by preprocess.py
│   ├── train.json
│   ├── val.json
│   └── test.json
│
├── plots/                 ← created by eda.py
├── results/               ← created by model runs
├── error_analysis/        ← created by error_analysis.py
│
└── word_embedding.pkl     ← created by preprocess.py (GloVe subset)
```

---

## Recommended Quick Run Order (No GloVe)

If you don't have GloVe yet, run baselines first:

```bash
python preprocess.py --data suicidal_ideation_reddit_annotated.csv
python eda.py --data suicidal_ideation_reddit_annotated.csv
python compare_models.py --train_data data/train.json --val_data data/val.json --test_data data/test.json --baseline_only
python error_analysis.py --test_data data/test.json --preds results/lr_tfidf_preds.out --model_name "LR + TF-IDF"
```

Once GloVe is downloaded, re-run compare_models.py without `--baseline_only`.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'X'` | Run `conda activate base` then `pip install X` |
| `python3` vs `python` confusion | Always use `python` inside conda, not `python3` |
| `No such file: word_embedding.pkl` | Run `preprocess.py` with `--glove glove.6B.50d.txt` first |
| `No such file: data/train.json` | Run `preprocess.py` first |
| BERT runs out of memory | Reduce `--batch_size 8` or `--max_len 128` |
| RNN loss is NaN | Already handled by gradient clipping (max_norm=5.0) |
