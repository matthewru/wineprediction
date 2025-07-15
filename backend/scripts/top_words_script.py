import pandas as pd
from collections import Counter
import re

INPUT_CSV = "backend/data/wine_clean.csv"   # Path to your wine dataset
TEXT_COLUMN = "description"
OUTPUT_FILE = "top_words.txt"
TOP_N = 1000  # Change as needed

def clean_text(text):
    # Lowercase, remove punctuation, numbers, etc.
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)       # remove digits
    return text.split()

def main():
    print("🔄 Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column '{TEXT_COLUMN}' not found in CSV")

    counter = Counter()
    for desc in df[TEXT_COLUMN].dropna():
        words = clean_text(desc)
        counter.update(words)

    most_common = counter.most_common(TOP_N)
    print(f"✅ Found {len(most_common)} words. Saving to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w') as f:
        for word, _ in most_common:
            f.write(f"{word}\n")

if __name__ == "__main__":
    main()
