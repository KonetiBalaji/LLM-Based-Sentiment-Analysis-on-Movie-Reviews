# scripts/streamlit_app.py

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, BertForSequenceClassification
from typing import List, Dict

# Configuration
CONFIG = {
    "model_dir": "models/bert/",
    "config_path": "outputs/bert/config.yaml",
    "output_dir": "outputs/bert/",
    "manual_threshold": None,
    "confidence_squash": 1.3,
    "min_review_length": 3,
    "explain_top_k": 5,
    "low_confidence_range": (0.4, 0.6)
}

LABEL_MAP = {0: "Negative", 1: "Positive"}

# Sentiment Analyzer
class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_dir"])
        self.model = BertForSequenceClassification.from_pretrained(CONFIG["model_dir"]).to(self.device)
        with open(CONFIG["config_path"], 'r') as f:
            self.config = yaml.safe_load(f)
        self.threshold = CONFIG["manual_threshold"] if CONFIG["manual_threshold"] is not None else self.config.get("optimal_threshold", 0.5)

    def calibrate_confidence(self, prob: float) -> float:
        return 0.5 + np.sign(prob - 0.5) * abs(prob - 0.5) ** CONFIG["confidence_squash"]

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config["max_length"]).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().squeeze()
        pos_prob = self.calibrate_confidence(probs[1].item())
        prediction = 1 if pos_prob >= self.threshold else 0
        confidence = pos_prob if prediction == 1 else 1 - pos_prob
        return {
            "prediction": LABEL_MAP[prediction],
            "confidence": confidence,
            "pos_probability": pos_prob,
            "threshold": self.threshold,
            "low_confidence": CONFIG["low_confidence_range"][0] <= pos_prob <= CONFIG["low_confidence_range"][1]
        }

# Load artifacts
analyzer = SentimentAnalyzer()

# Streamlit App
st.set_page_config(page_title="Movie Review Sentiment Analyzer (BERT Fine-Tuned)", layout="centered")
st.title("\U0001F3AC Movie Review Sentiment Analyzer (BERT Fine-Tuned)")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("🔮 Predict Sentiment", "📊 Model Evaluation Metrics"))

# Predict Sentiment Page
if page == "🔮 Predict Sentiment":
    st.subheader("🔍 Enter a Movie Review")
    user_input = st.text_area("Type your review here:", height=150)

    if st.button("Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a valid review!")
        else:
            result = analyzer.predict(user_input)
            st.markdown(f"**Prediction:** {result['prediction']}")
            st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%")
            if result['low_confidence']:
                st.warning("Low confidence prediction. Review may be ambiguous.")

# Model Evaluation Metrics Page
elif page == "📊 Model Evaluation Metrics":
    st.subheader("📈 Evaluation Metrics")

    try:
        # Load config
        with open(CONFIG["config_path"], 'r') as f:
            bert_config = yaml.safe_load(f)

        st.markdown(f"**ROC AUC Score:** {bert_config.get('val_f2_score', 'N/A'):.4f}")
        st.markdown(f"**Optimal Threshold:** {bert_config.get('optimal_threshold', 'N/A'):.4f}")
        st.markdown(f"**Validation F2-Score:** {bert_config.get('val_f2_score', 'N/A'):.4f}")

        # Load and show plots
        plot_files = {
            "Confusion Matrix": os.path.join(CONFIG["output_dir"], "confusion_matrix.png"),
            "Precision-Recall Curve": os.path.join(CONFIG["output_dir"], "precision_recall_curve.png"),
            "Training/Validation Loss Curve": os.path.join(CONFIG["output_dir"], "loss_curve.png"),
            "Validation Metric Curves": os.path.join(CONFIG["output_dir"], "metric_curves.png")
        }

        for title, path in plot_files.items():
            if os.path.exists(path):
                st.markdown(f"### {title}")
                st.image(path)
            else:
                st.warning(f"{title} plot not found.")

    except Exception as e:
        st.error(f"Failed to load evaluation metrics: {str(e)}")
