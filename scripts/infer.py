#!/usr/bin/env python3
"""
Final Optimized Sentiment Analysis Inference Script
- Synchronized negation handling with training
- Enhanced "not for X" pattern handling
- Short review filtering
- Confidence calibration
- Prediction explanations for edge cases
- Batch processing support
"""

import os
import sys
import joblib
import yaml
import re
import numpy as np
from typing import Tuple, List, Dict

# Configuration - Matches training script
MANUAL_THRESHOLD = 0.5          # Override trained threshold
CONFIDENCE_SQUASH = 1.3         # Compresses extreme confidences
LOW_CONFIDENCE_RANGE = (0.4, 0.6)  # Range for uncertainty warnings
MIN_REVIEW_LENGTH = 3           # Matches training Config.MIN_REVIEW_LENGTH

# Paths - Matches training script
PATHS = {
    'vectorizer': "models/vectorizer.pkl",
    'model': "models/logreg_model.pkl",
    'config': "outputs/config.yaml"
}

# Sentiment Labels
LABELS = {0: "Negative", 1: "Positive"}

def load_artifacts() -> Tuple:
    """Load model artifacts with comprehensive error handling"""
    try:
        vectorizer = joblib.load(PATHS['vectorizer'])
        model = joblib.load(PATHS['model'])
        with open(PATHS['config'], 'r') as f:
            config = yaml.safe_load(f)
        threshold = MANUAL_THRESHOLD if MANUAL_THRESHOLD is not None else config.get('optimal_threshold', 0.5)
        return vectorizer, model, threshold
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {str(e)}")
        print("Please run training script first (scripts/train.py)")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        sys.exit(1)

def preprocess_text(text: str) -> str:
    """Enhanced preprocessing identical to training with new negation pattern"""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    
    # Negation handling matching training script
    text = re.sub(
        r'\b(not|no|never)\s+(\w+ly)?\s*(\w+)',
        lambda m: f"NOT_{m.group(3)}",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'\bnot\s+(very|really|extremely)\s+(\w+)',
        lambda m: f"NOT_{m.group(1)}_{m.group(2)}",
        text,
        flags=re.IGNORECASE
    )
    # New pattern: Handle "not for X" cases
    text = re.sub(
        r'\b(not|no|never)\s+(for|to)\s+(\w+)',
        lambda m: f"NOT_{m.group(3)}",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\bnot for (\w+)', lambda m: f"NOT_{m.group(1)}", text, flags=re.IGNORECASE)

    
    return text.lower().strip()

def calibrate_confidence(p: float) -> float:
    """Squash extreme probabilities toward 0.5 for better calibration"""
    return 0.5 + np.sign(p - 0.5) * abs(p - 0.5) ** CONFIDENCE_SQUASH

def predict_sentiment(text: str, vectorizer, model, threshold: float) -> Tuple[int, float, List[Tuple[str, float, float]]]:
    """Enhanced prediction with short review handling and edge case explanations"""
    word_count = len(text.split())
    if word_count < MIN_REVIEW_LENGTH:
        return 0, 0.5, []  # Neutral for short reviews
    
    processed_text = preprocess_text(text)
    vector = vectorizer.transform([processed_text])
    raw_proba = model.predict_proba(vector)[0][1]
    proba = calibrate_confidence(raw_proba)
    
    # Generate explanations for edge cases
    explanation = []
    if word_count < 8 or (proba > 0.6 and "NOT_" in processed_text):
        terms = vectorizer.get_feature_names_out()
        coef = model.coef_[0]
        explanation = [
            (terms[i], coef[i], val)
            for i, val in zip(vector.indices, vector.data)
        ][:5]
    
    return (1 if proba >= threshold else 0), proba, explanation

def predict_batch(texts: List[str], vectorizer, model, threshold: float) -> List[Tuple[int, float]]:
    """Optimized batch prediction without explanations"""
    results = []
    for text in texts:
        word_count = len(text.split())
        if word_count < MIN_REVIEW_LENGTH:
            results.append((0, 0.5))
            continue
        processed_text = preprocess_text(text)
        vector = vectorizer.transform([processed_text])
        raw_proba = model.predict_proba(vector)[0][1]
        proba = calibrate_confidence(raw_proba)
        results.append((1 if proba >= threshold else 0, proba))
    return results

def print_result(text: str, label: int, proba: float, explanation: List[Tuple[str, float, float]]):
    """Enhanced output formatting with diagnostics"""
    sentiment = LABELS[label]
    confidence = proba if label == 1 else 1 - proba
    
    print(f"\n📝 Review: {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"📣 Prediction: {sentiment}")
    print(f"🔍 Confidence: {confidence:.1%}")
    
    # Enhanced warnings
    word_count = len(text.split())
    if word_count < MIN_REVIEW_LENGTH:
        print(f"⚠️ Warning: Very short review ({word_count} words)")
    elif LOW_CONFIDENCE_RANGE[0] <= proba <= LOW_CONFIDENCE_RANGE[1]:
        print("⚠️ Warning: Low confidence prediction")
    elif any(term.startswith("NOT_") for term, _, _ in explanation) and label == 1:
        print("⚠️ Strong Warning: Positive prediction with negation terms")
    
    if explanation:
        print("\n🔍 Key Factors:")
        for term, weight, value in sorted(explanation, key=lambda x: abs(x[1] * x[2]), reverse=True):
            print(f"{term:20s} {weight*value:+.2f} ({'↑Positive' if weight*value > 0 else '↓Negative'})")

def main():
    print("\n🔮 Sentiment Analysis Inference Engine")
    print("-------------------------------------")
    
    # Load model artifacts
    vectorizer, model, threshold = load_artifacts()
    print(f"⚖️ Decision threshold: {threshold:.2f}")
    print(f"🛠️ Confidence calibration: {CONFIDENCE_SQUASH}x")
    
    # Batch mode for file processing
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"\n📂 Processing file: {sys.argv[1]}")
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predict_batch(texts, vectorizer, model, threshold)
        for text, (label, proba) in zip(texts, results):
            print(f"{text[:80]}... | {LABELS[label]} ({proba:.2f})")
        sys.exit(0)
    
    # Interactive mode
    input_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("\n📝 Enter review: ").strip()
    if not input_text:
        print("No input provided.")
        sys.exit(0)
    
    # Get prediction with explanation
    label, proba, explanation = predict_sentiment(input_text, vectorizer, model, threshold)
    print_result(input_text, label, proba, explanation)

if __name__ == "__main__":
    main()