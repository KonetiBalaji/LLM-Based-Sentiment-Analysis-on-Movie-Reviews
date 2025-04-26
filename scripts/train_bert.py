#!/usr/bin/env python3
"""
Fine-tunes BERT on the IMDb dataset with:
- Comprehensive metric tracking
- Optimal threshold tuning
- Training/validation curves
- Classic-style model saving
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    fbeta_score, 
    precision_recall_curve,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Configuration ----------------------------------------------------------------
CONFIG = {
    "model_name": "bert-base-uncased",
    "data_path": "data/processed/imdb_reviews.csv",
    "model_dir": "models/bert/",
    "output_dir": "outputs/bert/",
    "max_length": 256,
    "batch_size": 4,
    "epochs": 1,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "threshold_steps": 50,
    "optimize_metric": "f2",
    "early_stopping_patience": 2,
    "seed": 42
}

# Setup ------------------------------------------------------------------------
os.makedirs(CONFIG["model_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Set all random seeds
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])

# Load and validate data -------------------------------------------------------
try:
    df = pd.read_csv(CONFIG["data_path"])
    df = df.sample(500, random_state=CONFIG["seed"])
    assert {"text", "sentiment"}.issubset(df.columns)
    texts = df["text"].astype(str).tolist()
    labels = df["sentiment"].map({"pos": 1, "neg": 0}).tolist()
except Exception as e:
    raise ValueError(f"Data loading failed: {str(e)}")

# Dataset class ----------------------------------------------------------------
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Tokenization -----------------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])
encodings = tokenizer(
    texts, 
    truncation=True, 
    padding=True, 
    max_length=CONFIG["max_length"]
)

# Train/val split --------------------------------------------------------------
train_idx, val_idx = train_test_split(
    range(len(labels)), 
    test_size=0.2, 
    random_state=CONFIG["seed"], 
    stratify=labels
)

train_dataset = IMDbDataset(
    {k: [v[i] for i in train_idx] for k, v in encodings.items()},
    [labels[i] for i in train_idx]
)

val_dataset = IMDbDataset(
    {k: [v[i] for i in val_idx] for k, v in encodings.items()},
    [labels[i] for i in val_idx]
)

# Metrics computation ----------------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": fbeta_score(labels, preds, beta=1),
        "f2": fbeta_score(labels, preds, beta=2)
    }

# Training setup ---------------------------------------------------------------
model = BertForSequenceClassification.from_pretrained(
    CONFIG["model_name"], 
    num_labels=2
)

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    num_train_epochs=CONFIG["epochs"],
    weight_decay=CONFIG["weight_decay"],
    load_best_model_at_end=True,
    metric_for_best_model=CONFIG["optimize_metric"],
    logging_dir=f"{CONFIG['output_dir']}/logs",
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=CONFIG["seed"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG["early_stopping_patience"]
        )
    ]
)

# Training ---------------------------------------------------------------------
print("\nStarting training...")
trainer.train()

# Save artifacts ---------------------------------------------------------------
model.save_pretrained(CONFIG["model_dir"])
tokenizer.save_pretrained(CONFIG["model_dir"])

# Evaluation -------------------------------------------------------------------
print("\nEvaluating...")
predictions = trainer.predict(val_dataset)
y_true = predictions.label_ids
y_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, CONFIG["threshold_steps"])
scores = [fbeta_score(y_true, (y_proba >= t).astype(int), beta=2) for t in thresholds]
optimal_idx = np.argmax(scores)
optimal_threshold = thresholds[optimal_idx]

# Generate reports
y_pred = (y_proba >= optimal_threshold).astype(int)
report = classification_report(y_true, y_pred, target_names=["neg", "pos"])
conf_mat = confusion_matrix(y_true, y_pred)

# Save metrics and plots -------------------------------------------------------
# Config
CONFIG.update({
    "optimal_threshold": float(optimal_threshold),
    "val_f2_score": float(scores[optimal_idx])
})

with open(os.path.join(CONFIG["output_dir"], "config.yaml"), "w") as f:
    yaml.dump(CONFIG, f)

# Text reports
with open(os.path.join(CONFIG["output_dir"], "metrics.txt"), "w") as f:
    f.write(report)
    f.write(f"\nOptimal Threshold: {optimal_threshold:.4f}")

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_mat, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    xticklabels=["neg", "pos"],
    yticklabels=["neg", "pos"]
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"))
plt.close()

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_true, y_proba)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(CONFIG["output_dir"], "precision_recall_curve.png"))
plt.close()

# Training history
history = pd.DataFrame(trainer.state.log_history)
history.to_csv(os.path.join(CONFIG["output_dir"], "training_history.csv"), index=False)

# Loss curve
plt.figure()
history[["loss", "eval_loss"]].ffill().plot()
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training/Validation Loss")
plt.savefig(os.path.join(CONFIG["output_dir"], "loss_curve.png"))
plt.close()

# Metric curves
plt.figure()
history[["eval_accuracy", "eval_f1", "eval_f2"]].ffill().plot()
plt.xlabel("Training Steps")
plt.ylabel("Score")
plt.title("Validation Metrics")
plt.savefig(os.path.join(CONFIG["output_dir"], "metric_curves.png"))
plt.close()

print(f"\nTraining complete! Model saved to {CONFIG['model_dir']}")
print(f"Best validation F2-score: {scores[optimal_idx]:.4f} at threshold {optimal_threshold:.4f}")