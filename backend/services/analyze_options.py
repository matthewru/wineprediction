import pandas as pd
import json

df = pd.read_csv('backend/data/wine_clean.csv')

unique_countries = df['country'].unique()
unique_regions = df['region_hierarchy'].unique()
unique_varieties = df['variety'].unique()

wines_per_country = df['country'].value_counts()
wines_per_region = df['region_hierarchy'].value_counts()
wines_per_variety = df['variety'].value_counts()

print(wines_per_country)
print(wines_per_region)
print(wines_per_variety)

included_countries = ['US', 'Italy', 'France', 'Spain', 'Portugal', 'Chile', 'Argentina', 'Austria', 'Germany', 'Australia']

# Filter data for included countries
included_df = df[df['country'].isin(included_countries)]

# Function to extract primary region (middle value) from hierarchy
def extract_primary_region(hierarchy):
    if pd.isna(hierarchy):
        return None
    parts = hierarchy.split(' > ')
    if len(parts) >= 2:
        return parts[1]  # Middle value (primary region)
    return None

# Function to extract secondary region (third value) from hierarchy
def extract_secondary_region(hierarchy):
    if pd.isna(hierarchy):
        return None
    parts = hierarchy.split(' > ')
    if len(parts) >= 3:
        return parts[2]  # Third value (secondary region)
    return None

# Extract primary and secondary regions
included_df['primary_region'] = included_df['region_hierarchy'].apply(extract_primary_region)
included_df['secondary_region'] = included_df['region_hierarchy'].apply(extract_secondary_region)

# Get wine counts per country
country_wine_counts = included_df['country'].value_counts()
print(f"\n=== WINE COUNTS PER COUNTRY ===")
for country in included_countries:
    count = country_wine_counts.get(country, 0)
    print(f"{country}: {count} wines")

# Get wine counts per primary region
primary_region_counts = included_df['primary_region'].value_counts()

# Filter regions based on country wine counts
valid_primary_regions = set()

for country in included_countries:
    country_wine_count = country_wine_counts.get(country, 0)
    
    # Get regions for this country
    country_data = included_df[included_df['country'] == country]
    country_region_counts = country_data['primary_region'].value_counts()
    
    # Set threshold based on country wine count
    if country_wine_count > 5000:
        threshold = 150  # Use 150-200 wines threshold for large countries
        print(f"\n{country} has {country_wine_count} wines → using {threshold}+ wines threshold")
    else:
        threshold = 50   # Use 50-100 wines threshold for smaller countries
        print(f"\n{country} has {country_wine_count} wines → using {threshold}+ wines threshold")
    
    # Get valid regions for this country
    valid_country_regions = country_region_counts[country_region_counts >= threshold].index
    
    # Special handling for US - add additional regions regardless of count
    if country == 'US':
        additional_us_regions = ['Texas', 'Missouri', 'Michigan', 'Colorado']
        # Check which additional regions exist in the data
        existing_additional_regions = [region for region in additional_us_regions if region in country_region_counts.index]
        valid_country_regions = list(valid_country_regions) + existing_additional_regions
        print(f"  Added additional US regions: {existing_additional_regions}")
    
    valid_primary_regions.update(valid_country_regions)
    
    print(f"  Valid regions: {sorted(valid_country_regions)}")

valid_primary_regions = list(valid_primary_regions)

print(f"\n=== ALL VALID PRIMARY REGIONS ===")
print(f"Total valid primary regions: {len(valid_primary_regions)}")
print(f"Regions: {sorted(valid_primary_regions)}")

# Create JSON structure
regions_json = {}

# Get secondary regions for each primary region and build JSON
print(f"\n=== TOP 6 SECONDARY REGIONS PER PRIMARY REGION ===")
for primary_region in sorted(valid_primary_regions):
    # Filter data for this primary region
    primary_data = included_df[included_df['primary_region'] == primary_region]
    secondary_region_counts = primary_data['secondary_region'].value_counts()
    
    # Get top 6 secondary regions by wine count
    top_secondary_regions = secondary_region_counts.head(6)
    
    print(f"\n{primary_region}:")
    print(f"  Number of secondary regions: {len(secondary_region_counts)}")
    print(f"  Top 6 secondary regions:")
    for region, count in top_secondary_regions.items():
        print(f"    {region}: {count} wines")
    
    # Find which country this primary region belongs to
    country_data = included_df[included_df['primary_region'] == primary_region]
    country = country_data['country'].iloc[0]
    
    # Add to JSON structure
    if country not in regions_json:
        regions_json[country] = {}
    
    regions_json[country][primary_region] = list(top_secondary_regions.index)

# Get primary regions for each included country (filtered)
print("\n=== PRIMARY REGIONS PER INCLUDED COUNTRY (filtered) ===")
for country in included_countries:
    country_data = included_df[included_df['country'] == country]
    country_primary_regions = country_data['primary_region'].dropna().unique()
    # Filter to only include regions in our valid set
    valid_country_regions = [region for region in country_primary_regions if region in valid_primary_regions]
    print(f"\n{country}:")
    print(f"  Number of valid primary regions: {len(valid_country_regions)}")
    print(f"  Valid primary regions: {sorted(valid_country_regions)}")

# Summary
print("\n=== SUMMARY ===")
total_regions = 0
for country in included_countries:
    country_data = included_df[included_df['country'] == country]
    country_primary_regions = country_data['primary_region'].dropna().unique()
    valid_country_regions = [region for region in country_primary_regions if region in valid_primary_regions]
    total_regions += len(valid_country_regions)
    print(f"{country}: {len(valid_country_regions)} regions")

print(f"\nTotal valid primary regions across all included countries: {total_regions}")

# Show wine counts for each valid region
print(f"\n=== WINE COUNTS PER VALID PRIMARY REGION ===")
for region in sorted(valid_primary_regions):
    count = primary_region_counts.get(region, 0)
    print(f"{region}: {count} wines")

# Save JSON file
print(f"\n=== SAVING JSON FILE ===")
with open('backend/data/regions.json', 'w') as f:
    json.dump(regions_json, f, indent=2)

print("JSON file saved to: backend/data/regions.json")
print("\nJSON structure:")
print(json.dumps(regions_json, indent=2))




