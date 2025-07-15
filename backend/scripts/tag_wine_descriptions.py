import csv
import json
import re
from collections import defaultdict

# Load the alias data
with open('backend/data/term_aliases.json', 'r') as f:
    alias_data = json.load(f)

# Build alias to (term, category) lookup
alias_to_term = {}
for category, terms in alias_data.items():
    for term, aliases in terms.items():
        for alias in aliases:
            alias_to_term[alias.lower()] = (term, category)

# Normalize text to words
def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

# Tag a description
def tag_description(description):
    tokens = tokenize(description)
    tags = defaultdict(set)
    
    for token in tokens:
        if token in alias_to_term:
            term, category = alias_to_term[token]
            tags[category].add(term)
    
    # Convert sets to sorted lists
    return {
        'flavor_tags': sorted(tags['flavor']) if 'flavor' in tags else [],
        'mouthfeel_tags': sorted(tags['mouthfeel']) if 'mouthfeel' in tags else []
    }

# Input and output files
input_csv = 'backend/data/wine_clean.csv'               # Replace with your actual input file
output_csv = 'backend/data/wine_clean_tagged.csv'       # Output with new columns

# Tag each row and write output
with open(input_csv, 'r', encoding='utf-8') as infile, \
     open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['flavor_tags', 'mouthfeel_tags']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        desc = row.get('description', '')
        tags = tag_description(desc)
        row['flavor_tags'] = ', '.join(tags['flavor_tags'])
        row['mouthfeel_tags'] = ', '.join(tags['mouthfeel_tags'])
        writer.writerow(row)

print("✅ Tagging complete. Output saved to:", output_csv)
