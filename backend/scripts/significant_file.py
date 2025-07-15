import json

def save_significant_terms(vocab_path, out_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    sig_terms = [word for word, v in vocab.items() if v]
    with open(out_path, 'w') as out:
        json.dump(sig_terms, out, indent=2)

# Usage
save_significant_terms('word_significance.json', 'backend/data/significant_terms.json')
