import pandas as pd
import json

# Read both datasets
print("Reading datasets...")
mouthfeel_df = pd.read_csv('backend/data/wine_mouthfeel_with_predictions.csv')
flavor_df = pd.read_csv('backend/data/wine_clean_flavors_only.csv')

# Get counts
mouthfeel_counts = mouthfeel_df['variety'].value_counts()
flavor_counts = flavor_df['variety'].value_counts()

# Define synonyms - combine counts for same grape with different names
synonyms = {
    'Syrah': ['Syrah', 'Shiraz'],  # Combine Syrah and Shiraz
    'Pinot Gris': ['Pinot Gris', 'Pinot Grigio']  # Combine Pinot Gris and Pinot Grigio
}

# Define varieties to exclude (blends, not single grapes)
exclude_varieties = {
    'Red Blend', 'White Blend', 'Bordeaux-style Red Blend', 'Rhône-style Red Blend',
    'Portuguese Red', 'Portuguese White', 'Sparkling Blend', 'Rosé',
    'Champagne Blend', 'Alsace white blend', 'Italian Red', 'Italian White',
    'Austrian white blend', 'GSM', 'Meritage', 'Super Tuscan', 'Port',
    'Fortified', 'Dessert', 'Sherry', 'Madeira', 'Tempranillo Blend',
    'Rhône-style White Blend', 'Bordeaux-style White Blend', 'Cabernet Blend',
    'Corvina, Rondinella, Molinara'  # This is a blend, not a single grape
}

# Get all unique varieties
all_varieties = set(mouthfeel_counts.index) | set(flavor_counts.index)

# Create combined counts with synonym handling
grape_data = {}

for variety in all_varieties:
    # Skip if it's in exclude list
    if variety in exclude_varieties or any(excl in variety for excl in ['Blend', 'blend', 'style']):
        continue
    
    mouthfeel_count = int(mouthfeel_counts.get(variety, 0))
    flavor_count = int(flavor_counts.get(variety, 0))
    
    # Check if this variety is a synonym
    canonical_name = variety
    for canonical, synonym_list in synonyms.items():
        if variety in synonym_list:
            canonical_name = canonical
            break
    
    # Add to grape data (combining synonyms)
    if canonical_name not in grape_data:
        grape_data[canonical_name] = {
            'mouthfeel_count': 0,
            'flavor_count': 0,
            'varieties_included': []
        }
    
    grape_data[canonical_name]['mouthfeel_count'] += mouthfeel_count
    grape_data[canonical_name]['flavor_count'] += flavor_count
    grape_data[canonical_name]['varieties_included'].append(variety)

# Calculate total counts and sort
grape_list = []
for grape, data in grape_data.items():
    total_count = data['mouthfeel_count'] + data['flavor_count']
    grape_list.append({
        'variety': grape,
        'mouthfeel_count': data['mouthfeel_count'],
        'flavor_count': data['flavor_count'],
        'total_count': total_count,
        'varieties_included': data['varieties_included']
    })

# Sort by total count
grape_list.sort(key=lambda x: x['total_count'], reverse=True)

# Get top 27 single grapes
top_27_grapes = grape_list[:27]

# Calculate totals
total_wines = len(mouthfeel_df) + len(flavor_df)
cumulative_count = sum(grape['total_count'] for grape in top_27_grapes)
coverage_percentage = (cumulative_count / total_wines) * 100

print(f"\nCLEANED TOP 27 SINGLE GRAPE VARIETIES:")
print("=" * 70)
print("Rank | Grape Variety              | Total   | Coverage | Synonyms")
print("-" * 70)

for i, grape in enumerate(top_27_grapes, 1):
    percentage = (grape['total_count'] / total_wines) * 100
    synonyms_str = " + ".join(grape['varieties_included']) if len(grape['varieties_included']) > 1 else ""
    print(f"{i:2d}   | {grape['variety']:<25} | {grape['total_count']:6,} | {percentage:4.1f}% | {synonyms_str}")

print("-" * 70)
print(f"Total coverage: {cumulative_count:,} wines ({coverage_percentage:.1f}%)")

# Create clean JSON output
output_data = {
    "top_27_single_grapes": {
        "total_wines_in_datasets": total_wines,
        "grape_varieties": []
    },
    "simple_grape_list": [],
    "synonym_info": {
        "notes": "Shiraz and Syrah combined, Pinot Gris and Pinot Grigio combined",
        "synonyms_merged": synonyms
    }
}

for i, grape in enumerate(top_27_grapes, 1):
    grape_data = {
        "rank": i,
        "variety": grape['variety'],
        "mouthfeel_count": grape['mouthfeel_count'],
        "flavor_count": grape['flavor_count'],
        "total_count": grape['total_count'],
        "percentage": round((grape['total_count'] / total_wines) * 100, 1),
        "includes_synonyms": grape['varieties_included'] if len(grape['varieties_included']) > 1 else []
    }
    output_data["top_27_single_grapes"]["grape_varieties"].append(grape_data)
    output_data["simple_grape_list"].append(grape['variety'])

output_data["top_27_single_grapes"]["coverage"] = {
    "total_wines_covered": cumulative_count,
    "coverage_percentage": round(coverage_percentage, 1)
}

# Save clean JSON
with open('top_27_clean_grape_varieties.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved cleaned list to 'top_27_clean_grape_varieties.json'")
print(f"\nChanges made:")
print("✅ Combined Syrah + Shiraz")
print("✅ Combined Pinot Gris + Pinot Grigio") 
print("✅ Removed 'Corvina, Rondinella, Molinara' blend")
print("✅ Ensured all 27 are single grape varieties") 