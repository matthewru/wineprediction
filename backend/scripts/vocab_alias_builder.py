import pandas as pd
import re
import json
from collections import Counter

# Load descriptions
df = pd.read_csv("backend/data/wine_clean.csv")
descriptions = df["description"].dropna().astype(str).tolist()

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s\-]", "", text)
    return text.split()

# Build full word frequency counter
word_counter = Counter()
for desc in descriptions:
    word_counter.update(tokenize(desc))

# Load existing vocab if available
def load_vocab(path):
    try:
        with open(path) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

taste_vocab = load_vocab("backend/data/taste_vocab.json")
mouthfeel_vocab = load_vocab("backend/data/mouthfeel_vocab.json")

# Combine current vocab
existing_vocab = taste_vocab.union(mouthfeel_vocab)

# Find candidate new tokens not in existing vocab
min_freq = 25  # Tune this
new_terms = {word: freq for word, freq in word_counter.items()
             if freq >= min_freq and word not in existing_vocab}

# Sort by frequency
sorted_candidates = sorted(new_terms.items(), key=lambda x: x[1], reverse=True)

# Print top 50 for manual review
print("\n🆕 Top flavor/mouthfeel candidates not yet in vocab:")
for word, freq in sorted_candidates[:50]:
    print(f"{word}: {freq}")

# Optionally write to file for curation
with open("backend/data/flavor_candidates.json", "w") as f:
    json.dump(sorted_candidates, f, indent=2)
    
    import json

# Manually defined from your frequency analysis
flavor_aliases = {
    "fruit": ["fruit", "fruits", "cherry", "berry", "blackberry", "plum", "apple", "ripe"],
    "spice": ["spice", "pepper", "clove", "cinnamon"],
    "oak": ["oak", "vanilla", "toast", "smoky"],
    "earth": ["earthy", "mushroom", "truffle", "soil"],
    "citrus": ["lemon", "lime", "orange", "grapefruit", "citrus"]
}

mouthfeel_aliases = {
    "tannic": ["tannins", "grippy", "astringent", "tight"],
    "acidic": ["acidity", "zippy", "crisp", "sharp"],
    "soft": ["soft", "buttery", "silky", "smooth", "velvety"],
    "rich": ["rich", "creamy", "dense", "full-bodied", "heavy"]
}

# Save them
with open("backend/data/flavor_aliases.json", "w") as f:
    json.dump(flavor_aliases, f, indent=2)

with open("backend/data/mouthfeel_aliases.json", "w") as f:
    json.dump(mouthfeel_aliases, f, indent=2)

print("✅ Saved flavor and mouthfeel vocab alias files.")

