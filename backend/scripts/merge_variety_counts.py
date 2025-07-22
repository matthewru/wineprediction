import pandas as pd
import json

# Read mouthfeel dataset
print("Reading mouthfeel dataset...")
mouthfeel_df = pd.read_csv('backend/data/wine_mouthfeel_with_predictions.csv')
mouthfeel_counts = mouthfeel_df['variety'].value_counts()

# Read flavor dataset  
print("Reading flavor dataset...")
flavor_df = pd.read_csv('backend/data/wine_clean_flavors_only.csv')
flavor_counts = flavor_df['variety'].value_counts()

# Get all unique varieties from both datasets
all_varieties = set(mouthfeel_counts.index) | set(flavor_counts.index)

# Create combined counts
combined_data = []
for variety in all_varieties:
    mouthfeel_count = int(mouthfeel_counts.get(variety, 0))
    flavor_count = int(flavor_counts.get(variety, 0))
    total_count = mouthfeel_count + flavor_count
    
    combined_data.append({
        'variety': variety,
        'mouthfeel_count': mouthfeel_count,
        'flavor_count': flavor_count,
        'total_count': total_count
    })

# Sort by total count descending
combined_data.sort(key=lambda x: x['total_count'], reverse=True)

# Get top 25
top_25 = combined_data[:25]

# Calculate total wines across both datasets
total_mouthfeel_wines = len(mouthfeel_df)
total_flavor_wines = len(flavor_df)
total_combined_wines = total_mouthfeel_wines + total_flavor_wines

# Calculate coverage
top_25_total_count = sum(item['total_count'] for item in top_25)
coverage_percentage = (top_25_total_count / total_combined_wines) * 100

print(f"\nTop 25 Combined Wine Varieties:")
print(f"Total wines across both datasets: {total_combined_wines:,}")
print(f"Top 25 coverage: {top_25_total_count:,} wines ({coverage_percentage:.1f}%)")
print("\nRank | Variety | Mouthfeel | Flavor | Total")
print("-" * 60)

for i, item in enumerate(top_25, 1):
    print(f"{i:2d}   | {item['variety']:<25} | {item['mouthfeel_count']:8,} | {item['flavor_count']:6,} | {item['total_count']:5,}")

# Create JSON output
output_data = {
    "combined_datasets": {
        "total_mouthfeel_wines": total_mouthfeel_wines,
        "total_flavor_wines": total_flavor_wines,
        "total_combined_wines": total_combined_wines,
        "top_25_varieties": []
    },
    "simple_variety_list": []
}

for i, item in enumerate(top_25, 1):
    variety_data = {
        "rank": i,
        "variety": item['variety'],
        "mouthfeel_count": item['mouthfeel_count'],
        "flavor_count": item['flavor_count'],
        "total_count": item['total_count'],
        "percentage": round((item['total_count'] / total_combined_wines) * 100, 1)
    }
    output_data["combined_datasets"]["top_25_varieties"].append(variety_data)
    output_data["simple_variety_list"].append(item['variety'])

output_data["combined_datasets"]["top_25_coverage"] = {
    "total_wines_covered": top_25_total_count,
    "coverage_percentage": round(coverage_percentage, 1)
}

# Save to JSON file
with open('top_25_combined_wine_varieties.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved results to top_25_combined_wine_varieties.json") 