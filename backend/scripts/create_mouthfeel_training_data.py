#!/usr/bin/env python3
"""
Create mouthfeel training data from wine_clean_tagged.csv

This script will:
1. Load wine_clean_tagged.csv with mouthfeel tags
2. Filter for only the top 200 mouthfeel tags by frequency 
3. Remove wines with missing values in: country, province, region_hierarchy, age, mouthfeel_tags
4. Create a dataset similar to wine_flavors_with_predictions.csv but for mouthfeel
5. Add price and rating predictions using existing models
6. Save as wine_mouthfeel_with_predictions.csv
"""

import pandas as pd
import numpy as np
import sys
import os
from collections import Counter

# Add the services directory to path to import prediction functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

# Import the prediction functions
from predict_price_lite import predict_price_lite
from predict_rating_lite import predict_rating_lite

def load_and_filter_data():
    """Load wine_clean_tagged.csv and filter for mouthfeel data"""
    print("📂 Loading wine_clean_tagged.csv...")
    df = pd.read_csv('backend/data/wine_clean_tagged.csv')
    print(f"Loaded {len(df)} wines with {len(df.columns)} columns")
    
    # Check required columns exist
    required_columns = ['country', 'province', 'region_hierarchy', 'age', 'mouthfeel_tags', 'variety']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Columns: {list(df.columns)}")
    
    # Display sample data
    print("\n📊 Sample mouthfeel data:")
    for i, row in df.head(3).iterrows():
        mouthfeel = row.get('mouthfeel_tags', '')
        if mouthfeel:
            print(f"  {row['variety']}: {mouthfeel}")
    
    return df

def filter_top_mouthfeel_tags(df, top_n=200):
    """Filter for only the top N mouthfeel tags by frequency"""
    print(f"\n🏷️  Filtering for top {top_n} mouthfeel tags...")
    
    # Get all mouthfeel tags and their counts
    all_mouthfeel_tags = []
    for tags in df['mouthfeel_tags'].fillna(''):
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            all_mouthfeel_tags.extend(tag_list)
    
    mouthfeel_counts = Counter(all_mouthfeel_tags)
    print(f"Found {len(mouthfeel_counts)} unique mouthfeel tags")
    print(f"Total mouthfeel tag instances: {len(all_mouthfeel_tags)}")
    
    # Get top N tags
    top_tags = mouthfeel_counts.most_common(top_n)
    top_tag_names = {tag for tag, count in top_tags}
    
    print(f"\nTop 20 mouthfeel tags:")
    for i, (tag, count) in enumerate(top_tags[:20]):
        print(f"  {i+1:2d}. {tag:<15} ({count:4d} wines)")
    
    # Filter wines to only include top tags
    def filter_tags(tag_string):
        """Keep only top N tags from a comma-separated string"""
        if pd.isna(tag_string) or not tag_string.strip():
            return ''
        
        tags = [tag.strip() for tag in tag_string.split(',') if tag.strip()]
        filtered_tags = [tag for tag in tags if tag in top_tag_names]
        return ', '.join(sorted(filtered_tags)) if filtered_tags else ''
    
    # Apply filtering
    df['mouthfeel_tags_filtered'] = df['mouthfeel_tags'].apply(filter_tags)
    
    # Remove wines with no mouthfeel tags after filtering
    initial_count = len(df)
    df_with_mouthfeel = df[df['mouthfeel_tags_filtered'].str.strip() != ''].copy()
    removed_count = initial_count - len(df_with_mouthfeel)
    
    print(f"\nAfter filtering:")
    print(f"  Removed {removed_count} wines with no top-{top_n} mouthfeel tags ({removed_count/initial_count*100:.1f}%)")
    print(f"  Remaining: {len(df_with_mouthfeel)} wines")
    
    # Update the mouthfeel_tags column with filtered tags
    df_with_mouthfeel['mouthfeel_tags'] = df_with_mouthfeel['mouthfeel_tags_filtered']
    df_with_mouthfeel.drop(columns=['mouthfeel_tags_filtered'], inplace=True)
    
    return df_with_mouthfeel, top_tag_names

def remove_missing_values(df):
    """Remove wines with missing values in required categories"""
    print("\n🧹 Removing wines with missing values...")
    
    required_fields = ['country', 'province', 'region_hierarchy', 'age', 'mouthfeel_tags']
    
    initial_count = len(df)
    print(f"Initial dataset: {initial_count} wines")
    
    # Check each field
    for field in required_fields:
        field_missing = df[field].isna() | (df[field].astype(str).str.strip() == '')
        missing_count = field_missing.sum()
        print(f"  {field}: {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
    
    # Remove wines with any missing required values
    df_clean = df.dropna(subset=required_fields).copy()
    
    # Also remove wines where string fields are empty strings
    for field in ['country', 'province', 'region_hierarchy', 'mouthfeel_tags']:
        df_clean = df_clean[df_clean[field].astype(str).str.strip() != ''].copy()
    
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    
    print(f"\nAfter removing missing values:")
    print(f"  Removed: {removed_count} wines ({removed_count/initial_count*100:.1f}%)")
    print(f"  Remaining: {final_count} wines")
    
    return df_clean

def prepare_base_features(df):
    """Prepare the base features needed for the training dataset"""
    print("\n🔧 Preparing base features...")
    
    # Select the core features that match wine_flavors_with_predictions.csv structure
    base_features = ['variety', 'country', 'province', 'age', 'region_hierarchy', 'mouthfeel_tags']
    
    # Ensure all base features exist
    missing_features = [col for col in base_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing base features: {missing_features}")
    
    df_base = df[base_features].copy()
    
    print(f"Base features prepared: {list(df_base.columns)}")
    print(f"Dataset shape: {df_base.shape}")
    
    # Show sample
    print(f"\nSample wines:")
    for i, row in df_base.head(3).iterrows():
        print(f"  {row['variety']:<15} ({row['country']}): {row['mouthfeel_tags']}")
    
    return df_base

def add_price_predictions(df):
    """Add price_min and price_max columns using predict_price_lite"""
    print("\n💰 Adding price predictions...")
    
    def get_price_bounds(row):
        try:
            input_dict = {
                "variety": row["variety"],
                "country": row["country"], 
                "province": row["province"],
                "age": row["age"],
                "region_hierarchy": row["region_hierarchy"]
            }
            price_output = predict_price_lite(input_dict)
            return (float(price_output['weighted_lower']), float(price_output['weighted_upper']))
        except Exception as e:
            print(f"Error predicting price for row {row.name}: {e}")
            return (None, None)
    
    # Apply price prediction to all rows with progress tracking
    total_rows = len(df)
    price_results = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 1000 == 0 or (i + 1) == total_rows:
            print(f"  Processed {i + 1}/{total_rows} rows ({(i+1)/total_rows*100:.1f}%)")
        
        price_bounds = get_price_bounds(row)
        price_results.append(price_bounds)
    
    # Add the price columns
    price_df = pd.DataFrame(price_results, columns=['price_min', 'price_max'], index=df.index)
    df = pd.concat([df, price_df], axis=1)
    
    # Remove rows where price prediction failed
    initial_count = len(df)
    df = df.dropna(subset=['price_min', 'price_max'])
    removed_count = initial_count - len(df)
    print(f"  Removed {removed_count} rows with failed price predictions")
    print(f"  Remaining: {len(df)} wines")
    
    return df

def add_rating_predictions(df):
    """Add rating column using predict_rating_lite"""
    print("\n⭐ Adding rating predictions...")
    
    def get_rating_prediction(row):
        try:
            input_dict = {
                "variety": row["variety"],
                "country": row["country"],
                "province": row["province"], 
                "age": row["age"],
                "region_hierarchy": row["region_hierarchy"],
                "price_min": row["price_min"],
                "price_max": row["price_max"]
            }
            rating_output = predict_rating_lite(input_dict)
            return float(rating_output['predicted_rating'])
        except Exception as e:
            print(f"Error predicting rating for row {row.name}: {e}")
            return None
    
    # Apply rating prediction to all rows with progress tracking
    total_rows = len(df)
    rating_results = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 1000 == 0 or (i + 1) == total_rows:
            print(f"  Processed {i + 1}/{total_rows} rows ({(i+1)/total_rows*100:.1f}%)")
        
        rating = get_rating_prediction(row)
        rating_results.append(rating)
    
    # Add the rating column
    df['rating'] = rating_results
    
    # Remove rows where rating prediction failed
    initial_count = len(df)
    df = df.dropna(subset=['rating'])
    removed_count = initial_count - len(df)
    print(f"  Removed {removed_count} rows with failed rating predictions")
    print(f"  Remaining: {len(df)} wines")
    
    return df

def save_final_dataset(df, top_tag_names):
    """Save the final mouthfeel training dataset"""
    output_file = 'backend/data/wine_mouthfeel_with_predictions.csv'
    print(f"\n💾 Saving final dataset to {output_file}...")
    
    # Reorder columns to match wine_flavors_with_predictions.csv structure
    column_order = ['variety', 'country', 'province', 'age', 'region_hierarchy', 'mouthfeel_tags', 'price_min', 'price_max', 'rating']
    df_final = df[column_order].copy()
    
    df_final.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df_final)} wines with mouthfeel predictions")
    
    # Show statistics
    print(f"\nFinal dataset statistics:")
    print(f"  - Wines: {len(df_final)}")
    print(f"  - Features: {list(df_final.columns)}")
    print(f"  - Mouthfeel tags used: {len(top_tag_names)}")
    
    # Show sample
    print(f"\nSample wines:")
    for i, row in df_final.head(5).iterrows():
        print(f"  {row['variety']:<15}: {row['mouthfeel_tags']}")
    
    return df_final

def main():
    """Main pipeline to create mouthfeel training data"""
    print("🍷 Creating Mouthfeel Training Dataset")
    print("=" * 60)
    
    try:
        # Step 1: Load and examine data
        df = load_and_filter_data()
        
        # Step 2: Filter for top 200 mouthfeel tags
        df_filtered, top_tag_names = filter_top_mouthfeel_tags(df, top_n=200)
        
        # Step 3: Remove wines with missing values in required categories
        df_clean = remove_missing_values(df_filtered)
        
        # Step 4: Prepare base features
        df_base = prepare_base_features(df_clean)
        
        # Step 5: Add price predictions
        df_with_prices = add_price_predictions(df_base)
        
        # Step 6: Add rating predictions
        df_with_ratings = add_rating_predictions(df_with_prices)
        
        # Step 7: Save final dataset
        df_final = save_final_dataset(df_with_ratings, top_tag_names)
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Mouthfeel Training Dataset Created!")
        print(f"   - Source: wine_clean_tagged.csv")
        print(f"   - Output: wine_mouthfeel_with_predictions.csv")
        print(f"   - Wines: {len(df_final)}")
        print(f"   - Mouthfeel tags: {len(top_tag_names)} (top 200 by frequency)")
        print(f"   - Features: price_min, price_max, rating added via prediction")
        print("   - Ready for mouthfeel model training!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 