# 🎬 Movie Review Sentiment Analyzer (BERT Fine-Tuned)

This project fine-tunes a BERT model and builds a Logistic Regression baseline for sentiment analysis on IMDb movie reviews. It also features a **LIVE Streamlit Web App** for demo!

---

## 📁 Project Structure

```
data/
  processed/
    imdb_reviews.csv      # Preprocessed dataset (POS/NEG)
models/
  bert/                   # Fine-tuned BERT model and tokenizer
  bert_sentiment/         # (Optional) Older checkpoints
  logreg_model.pkl        # Logistic Regression model
  vectorizer.pkl          # TF-IDF vectorizer
outputs/
  bert/                   # Evaluation metrics and plots
scripts/
  preprocess.py           # Preprocessing raw IMDb data
  train.py                # Train Logistic Regression model
  infer.py                # Inference using Logistic Regression
  bert_train.py           # Fine-tune BERT model
  infer_bert.py           # Inference using fine-tuned BERT
  streamlit_app.py        # Streamlit web application
```

---

## 🚀 How to Run

### 1. Preprocessing (if raw IMDb dataset available)
```bash
python scripts/preprocess.py
```

### 2. Train Logistic Regression
```bash
python scripts/train.py
```

### 3. Fine-tune BERT
```bash
python scripts/bert_train.py
```

### 4. Inference using Logistic Regression
```bash
python scripts/infer.py
```

### 5. Inference using Fine-Tuned BERT
```bash
python scripts/infer_bert.py
```

### 6. Launch Streamlit App (Live Demo)
```bash
streamlit run scripts/streamlit_app.py
```

---

## 📊 Outputs

- `metrics.txt` – Classification reports
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `threshold_analysis.png`
- `false_positives.csv`, `false_negatives.csv`
- `top_false_positives.csv`, `top_false_negatives.csv`

---

## 📈 Highlights

- **Preprocessing**: Text cleaning + Negation handling
- **Baseline**: TF-IDF + Logistic Regression
- **Fine-Tuning**: BERT with EarlyStopping
- **Threshold Optimization**: F2-score tuning
- **Confidence Calibration**: Smoothed probability outputs
- **Explainability**: Influential tokens highlighted (Gradients)
- **Streamlit App**: Live review prediction & evaluation metrics

---

## 🛠️ Tech Stack

- Python 3.10
- PyTorch
- Huggingface Transformers
- Scikit-Learn
- Streamlit
- Seaborn + Matplotlib

---

## ⚠️ Note on Large Files

- The directories `data/`, `models/`, and `outputs/` are excluded from version control via `.gitignore` to avoid pushing large datasets, model checkpoints, and generated outputs to the repository.
- If you need the trained models or datasets, please contact the maintainer or refer to the instructions in the scripts for downloading or generating them.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.