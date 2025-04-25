# scripts/preprocess.py - Step 2: Convert IMDb TXT Reviews to CSV with Progress Bar

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw/")
PROCESSED_PATH = Path("data/processed/imdb_reviews.csv")
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_reviews_from_folder(folder_path, sentiment):
    data = []
    files = list(folder_path.glob("*.txt"))
    for file in tqdm(sorted(files), desc=f"Loading {sentiment} reviews"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            data.append({"text": text, "sentiment": sentiment})
    return data

print("Loading IMDb dataset from", RAW_DIR)

pos_reviews = load_reviews_from_folder(RAW_DIR / "pos", "pos")
neg_reviews = load_reviews_from_folder(RAW_DIR / "neg", "neg")

print(f"Loaded {len(pos_reviews)} positive and {len(neg_reviews)} negative reviews")

# Combine and save
df = pd.DataFrame(pos_reviews + neg_reviews)
df.to_csv(PROCESSED_PATH, index=False)
print(f"Saved processed dataset to {PROCESSED_PATH} ({len(df)} total reviews)")