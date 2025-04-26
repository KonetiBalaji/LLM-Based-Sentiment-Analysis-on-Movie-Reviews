#!/usr/bin/env python3
"""
Production-Grade BERT Inference Script
"""

import os
import sys
import yaml
import torch
import numpy as np
import json
from typing import List, Dict
from transformers import BertTokenizerFast, BertForSequenceClassification

# Configuration ----------------------------------------------------------------
CONFIG = {
    "model_dir": "models/bert/",
    "config_path": "outputs/bert/config.yaml",
    "manual_threshold": None,
    "confidence_squash": 1.3,
    "min_review_length": 3,
    "explain_top_k": 5,
    "low_confidence_range": (0.4, 0.6),
    "max_display_length": 200
}

LABEL_MAP = {0: "Negative", 1: "Positive"}

# Sentiment Analyzer -----------------------------------------------------------
class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        """Load model and tokenizer"""
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_dir"])
            self.model = BertForSequenceClassification.from_pretrained(CONFIG["model_dir"]).to(self.device)

            with open(CONFIG["config_path"], 'r') as f:
                self.config = yaml.safe_load(f)

            self.threshold = (
                CONFIG["manual_threshold"]
                if CONFIG["manual_threshold"] is not None
                else self.config.get("optimal_threshold", 0.5)
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """Clean the input text"""
        return " ".join(str(text).strip().split())

    def calibrate_confidence(self, prob: float) -> float:
        """Squash extreme probabilities toward 0.5"""
        return 0.5 + np.sign(prob - 0.5) * abs(prob - 0.5) ** CONFIG["confidence_squash"]

    def validate_input(self, text: str) -> bool:
        """Check minimum word count"""
        words = text.split()
        return len(words) >= CONFIG["min_review_length"]

    def get_token_importance(self, text: str) -> List[Dict]:
        """Get top influential tokens"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        self.model.eval()
        input_embeds = self.model.bert.embeddings(input_ids)
        input_embeds.retain_grad()

        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        max_prob, _ = torch.max(probs, dim=1)
        max_prob.backward()

        grads = input_embeds.grad.abs()
        scores = grads.sum(dim=-1).squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

        important_tokens = []
        for token, score in zip(tokens, scores):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            if token.startswith("##"):
                if important_tokens:
                    important_tokens[-1]["token"] += token[2:]
                    important_tokens[-1]["score"] += score.item()
                continue
            important_tokens.append({"token": token, "score": score.item()})

        if important_tokens:
            max_score = max(t["score"] for t in important_tokens)
            for t in important_tokens:
                t["score"] /= max_score
            important_tokens.sort(key=lambda x: x["score"], reverse=True)

        return important_tokens[:CONFIG["explain_top_k"]]

    def predict(self, text: str, explain: bool = False) -> Dict:
        """Predict sentiment"""
        text = self.preprocess_text(text)
        result = {
            "text": text[:CONFIG["max_display_length"]] + "..." if len(text) > CONFIG["max_display_length"] else text,
            "threshold": self.threshold
        }

        if not self.validate_input(text):
            result.update({
                "prediction": "Invalid",
                "confidence": 0.5,
                "probability": 0.5,
                "warning": f"Input too short (min {CONFIG['min_review_length']} words required)"
            })
            return result

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"]
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().squeeze()

        pos_prob = self.calibrate_confidence(probs[1].item())
        prediction = 1 if pos_prob >= self.threshold else 0
        confidence = pos_prob if prediction == 1 else 1 - pos_prob

        result.update({
            "prediction": LABEL_MAP[prediction],
            "confidence": round(confidence, 4),
            "probability": round(pos_prob, 4),
            "is_low_confidence": (
                CONFIG["low_confidence_range"][0] <= pos_prob <= CONFIG["low_confidence_range"][1]
            )
        })

        if explain:
            result["tokens"] = self.get_token_importance(text)

        return result

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict batch of reviews"""
        return [self.predict(text, explain=False) for text in texts]

# Pretty Print -----------------------------------------------------------------
def print_result(result: Dict) -> None:
    """Print output nicely"""
    print(f"\n📝 Review: {result['text']}")
    print(f"📣 Prediction: {result['prediction']}")
    print(f"🔍 Confidence: {result['confidence']:.1%}")
    print(f"⚖️ Decision Threshold: {result['threshold']:.2f}")

    if result.get("warning"):
        print(f"⚠️ Warning: {result['warning']}")
    elif result["is_low_confidence"]:
        print("⚠️ Warning: Low confidence prediction")

    if "tokens" in result:
        print("\n🔍 Most Influential Tokens:")
        for token in result["tokens"]:
            print(f"• {token['token']} (score: {token['score']:.2f})")

# Main -------------------------------------------------------------------------
def main():
    print("\n🔮 BERT Sentiment Analysis Engine")
    print("---------------------------------")

    try:
        analyzer = SentimentAnalyzer()
        print(f"✅ Model loaded (threshold={analyzer.threshold:.2f})")

        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            print(f"\n📂 Processing file: {sys.argv[1]}")
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

            results = analyzer.predict_batch(texts)
            output_file = f"predictions_{os.path.basename(sys.argv[1])}"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📥 Saved predictions to {output_file}")
            return

        input_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("\n📝 Enter review (or 'quit'): ").strip()
        if input_text.lower() in ('quit', 'exit'):
            return

        result = analyzer.predict(input_text, explain=True)
        print_result(result)

    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
