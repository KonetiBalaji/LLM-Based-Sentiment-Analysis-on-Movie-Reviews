import os
import yaml
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                           fbeta_score, precision_recall_curve, roc_curve,
                           roc_auc_score, precision_score, recall_score)

# Configuration
class Config:
    DEFAULT_THRESHOLD = 0.5
    OPTIMIZE_FOR = 'f2'  # Options: 'f2', 'f1', 'precision', 'recall'
    THRESHOLD_STEPS = 50
    TOP_ERROR_SAMPLES = 20
    MIN_REVIEW_LENGTH = 3  # Minimum words to consider

# Paths
PATHS = {
    'data': "data/processed/imdb_reviews.csv",
    'vectorizer': "models/vectorizer.pkl",
    'model': "models/logreg_model.pkl",
    'metrics': "outputs/metrics.txt",
    'conf_matrix': "outputs/confusion_matrix.png",
    'pr_curve': "outputs/precision_recall_curve.png",
    'roc_curve': "outputs/roc_curve.png",
    'threshold_plot': "outputs/threshold_analysis.png",
    'config': "outputs/config.yaml",
    'false_positives': "outputs/false_positives.csv",
    'false_negatives': "outputs/false_negatives.csv",
    'top_fp': "outputs/top_false_positives.csv",
    'top_fn': "outputs/top_false_negatives.csv"
}

def preprocess_text(text: str) -> str:
    """Enhanced preprocessing with advanced negation handling"""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    
    # Advanced negation handling
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
    text = re.sub(
        r'\b(not|no|never)\s+(for|to)\s+(\w+)',
        lambda m: f"NOT_{m.group(3)}",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r'\bnot for (\w+)', lambda m: f"NOT_{m.group(1)}", text, flags=re.IGNORECASE)

    return text.lower().strip()

def find_optimal_threshold(y_true, y_proba, optimize_for='f2'):
    """Find optimal threshold based on specified metric with progress bar"""
    thresholds = np.linspace(0.1, 0.9, Config.THRESHOLD_STEPS)
    scores = []
    
    for t in tqdm(thresholds, desc="Threshold Tuning"):
        y_pred = (y_proba >= t).astype(int)
        if optimize_for == 'f2':
            scores.append(fbeta_score(y_true, y_pred, beta=2))
        elif optimize_for == 'f1':
            scores.append(fbeta_score(y_true, y_pred, beta=1))
        elif optimize_for == 'precision':
            scores.append(precision_score(y_true, y_pred))
        elif optimize_for == 'recall':
            scores.append(recall_score(y_true, y_pred))
    
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx], scores[optimal_idx]

def plot_threshold_analysis(y_true, y_proba, optimal_threshold):
    """Visualize metric performance across thresholds"""
    thresholds = np.linspace(0.1, 0.9, Config.THRESHOLD_STEPS)
    f1_scores = [fbeta_score(y_true, y_proba >= t, beta=1) for t in tqdm(thresholds, desc="Calculating F1")]
    f2_scores = [fbeta_score(y_true, y_proba >= t, beta=2) for t in tqdm(thresholds, desc="Calculating F2")]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, f2_scores, label='F2 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold Optimization (Optimizing for {Config.OPTIMIZE_FOR.upper()})')
    plt.legend()
    plt.grid()
    plt.savefig(PATHS['threshold_plot'])
    plt.close()

def plot_roc_curve(y_true, y_proba):
    """Generate and save ROC curve with AUC score"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(PATHS['roc_curve'])
    plt.close()
    return roc_auc

def save_config(optimal_threshold, roc_auc):
    """Save configuration and key metrics for reproducibility"""
    config = {
        'model': 'LogisticRegression',
        'optimize_for': Config.OPTIMIZE_FOR,
        'default_threshold': Config.DEFAULT_THRESHOLD,
        'optimal_threshold': float(optimal_threshold),
        'roc_auc': float(roc_auc),
        'features': 'TF-IDF (20k)',
        'ngram_range': [1, 3],
        'preprocessing': {
            'negation_handling': True,
            'min_length': Config.MIN_REVIEW_LENGTH
        }
    }
    with open(PATHS['config'], 'w') as f:
        yaml.dump(config, f)

def main():
    # Setup directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv(PATHS['data'])
    df['text'] = df['text'].apply(preprocess_text)
    df = df[df['text'].str.split().str.len() >= Config.MIN_REVIEW_LENGTH]  # Filter short reviews
    
    X = df["text"]
    y = df["sentiment"].map({"pos": 1, "neg": 0})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Vectorize text
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),
        stop_words='english',
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    print("Training model...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=500,
        solver='liblinear'
    )
    model.fit(X_train_vec, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    
    # ROC Analysis
    roc_auc = plot_roc_curve(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Find optimal threshold
    optimal_threshold, optimal_score = find_optimal_threshold(
        y_test, y_pred_proba, optimize_for=Config.OPTIMIZE_FOR)
    print(f"Optimal {Config.OPTIMIZE_FOR.upper()} threshold: {optimal_threshold:.3f}")
    print(f"{Config.OPTIMIZE_FOR.upper()} at optimal threshold: {optimal_score:.4f}")

    # Save configuration
    save_config(optimal_threshold, roc_auc)

    # Evaluate at both thresholds
    results = {}
    for threshold, label in [(Config.DEFAULT_THRESHOLD, "Default"),
                           (optimal_threshold, "Optimal")]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        results[label] = {
            'threshold': threshold,
            'report': classification_report(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2),
            'conf_mat': confusion_matrix(y_test, y_pred)
        }

    # Save metrics
    with open(PATHS['metrics'], "w") as f:
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        for label, data in results.items():
            f.write(f"=== {label} Threshold ({data['threshold']:.2f}) ===\n")
            f.write(data['report'])
            f.write(f"F2 Score: {data['f2']:.4f}\n\n")

    # Visualizations
    plot_threshold_analysis(y_test, y_pred_proba, optimal_threshold)
    
    # Confusion matrix at optimal threshold
    plt.figure(figsize=(6, 4))
    sns.heatmap(results['Optimal']['conf_mat'], annot=True, fmt="d",
               cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Threshold={optimal_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(PATHS['conf_matrix'])
    plt.close()

    # Error analysis
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    df_eval = pd.DataFrame({
        "text": X_test.values,
        "actual": y_test.values,
        "predicted": y_pred_optimal,
        "probability": y_pred_proba
    })
    
    # Save error samples
    for error_type in ['false_positives', 'false_negatives']:
        condition = ((df_eval.actual == 0) & (df_eval.predicted == 1)) if error_type == 'false_positives' else \
                   ((df_eval.actual == 1) & (df_eval.predicted == 0))
        errors = df_eval[condition]
        errors.to_csv(PATHS[error_type], index=False)
        
        # Save top errors
        output_key = 'top_fp' if error_type == 'false_positives' else 'top_fn'
        top_errors = errors.nlargest(Config.TOP_ERROR_SAMPLES, "probability") if error_type == 'false_positives' else \
                     errors.nsmallest(Config.TOP_ERROR_SAMPLES, "probability")
        top_errors.to_csv(PATHS[output_key], index=False)

    # Save model artifacts
    joblib.dump(vectorizer, PATHS['vectorizer'])
    joblib.dump(model, PATHS['model'])
    
    print("\nTraining complete. All outputs saved.")

if __name__ == "__main__":
    main()