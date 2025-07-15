import json
import os

WORDS_FILE = "top_words.txt"             # One word per line
OUTPUT_FILE = "word_significance.json"   # Stores labeling progress

def load_words(path):
    with open(path, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    return words

def load_progress(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_progress(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Progress saved to {path}")

def main():
    words = load_words(WORDS_FILE)
    labeled = load_progress(OUTPUT_FILE)
    total = len(words)
    
    for i, word in enumerate(words):
        if word in labeled:
            continue
        
        print(f"\nWord: {word}")
        print(f"Progress: {len(labeled)}/{total} words labeled.")
        decision = input("Mark as (s)ignificant / (i)nsignificant / (q)uit: ").strip().lower()

        if decision == 'q':
            break
        elif decision == 's':
            labeled[word] = True
        elif decision == 'i':
            labeled[word] = False
        else:
            print("❌ Invalid input. Use 's', 'i', or 'q'.")
            continue

    save_progress(OUTPUT_FILE, labeled)

if __name__ == "__main__":
    main()
