#!/usr/bin/env python3
"""
Clean up existing flavor tags from wine_clean_tagged.csv

This script will:
1. Load existing flavor tags from the original tagging
2. Remove non-flavor words (aroma, blend, etc.)
3. Consolidate plural/singular forms
4. Remove vague color words without context
5. Keep interesting descriptive terms 
6. Create a clean dataset with only good flavor tags
"""

import pandas as pd
from collections import Counter
import re

def clean_flavor_tags():
    """Clean up the existing flavor tags and create a better dataset"""
    
    print("Loading existing tagged data...")
    df = pd.read_csv('backend/data/wine_clean_tagged.csv')
    print(f"Loaded {len(df)} wines")
    
    # Remove wines with no flavor tags
    initial_count = len(df)
    df = df[df['flavor_tags'].fillna('').str.strip() != '']
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} wines with no flavor tags ({removed_count/initial_count*100:.1f}%)")
    print(f"Working with {len(df)} wines that have flavor tags")
    
    # Get all existing tags and their counts
    all_flavor_tags = []
    for tags in df['flavor_tags'].fillna(''):
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
            all_flavor_tags.extend(tag_list)
    
    tag_counts = Counter(all_flavor_tags)
    print(f"Found {len(tag_counts)} unique existing flavor tags")
    
    # Define cleaning rules based on your feedback
    
    # 1. Non-flavor words to REMOVE completely
    non_flavors = {
        'blend', 'delicious', 'lead', 
        'tannic', 'drinkable', 'pleasant', 'notes', 'palate', 'finish', 'nose', 
        'wine', 'drink', 'drinking', 'bottle', 'years', 'year', 'time', 'good',
        'fine', 'nice', 'excellent', 'ready', 'now'
    }
    
    # 2. Vague color words to REMOVE (need fruit context)
    vague_colors = {'red', 'black', 'white', 'green', 'pale'}
    
    # 3. Plural to singular mappings AND special replacements
    replacements = {
        'fruits': 'fruit',
        'spices': 'spice', 
        'herbs': 'herb',
        'plums': 'plum',
        'apples': 'apple',
        'pears': 'pear',
        'currants': 'currant',
        'flowers': 'flower',
        'berries': 'berry',
        'cherries': 'cherry',
        'grapes': 'grape',
        'notes': 'note',
        'tannins': 'tannin',
        'aroma': 'aromatic',
        'aromas': 'aromatic',
        'easy': 'palatable',
        'spice': 'spiced'
    }
    
    # 4. Keep interesting vague descriptors (as you suggested)
    keep_interesting = {
        'rustic', 'elegant', 'complex', 'vibrant', 'intense', 'rich', 'bold', 
        'crisp', 'fresh', 'juicy', 'ripe', 'concentrated', 'smooth', 'soft',
        'dry', 'sweet', 'tart', 'earthy', 'mineral', 'floral', 'fruity', 
        'spicy', 'herbal', 'woody', 'smoky', 'savory'
    }
    
    def clean_tag_list(tag_string):
        """Clean a comma-separated string of tags"""
        if pd.isna(tag_string) or not tag_string.strip():
            return []
        
        tags = [tag.strip() for tag in tag_string.split(',')]
        cleaned_tags = []
        
        for tag in tags:
            if not tag:
                continue
            
            # Apply replacements first (plurals to singular, aroma to aromatic)
            if tag in replacements:
                tag = replacements[tag]
                
            # Remove non-flavors (but not aroma since we convert it to aromatic)
            if tag in non_flavors:
                continue
                
            # Remove vague colors
            if tag in vague_colors:
                continue
            
            # Keep everything else (including interesting descriptors and converted tags)
            cleaned_tags.append(tag)
        
        # Remove duplicates and sort
        return sorted(list(set(cleaned_tags)))
    
    # Apply cleaning to all wines
    print("Cleaning flavor tags...")
    df['flavor_tags_cleaned'] = df['flavor_tags'].apply(lambda x: clean_tag_list(x))
    df['flavor_tags_str'] = df['flavor_tags_cleaned'].apply(lambda tags: ', '.join(tags))
    
    # Remove wines with no flavor tags after cleaning
    initial_count = len(df)
    df_with_flavors = df[df['flavor_tags_cleaned'].apply(len) > 0].copy()
    dropped_count = initial_count - len(df_with_flavors)
    print(f"Dropped {dropped_count} wines with no tags after cleaning ({dropped_count/initial_count*100:.1f}%)")
    
    # Analyze cleaned tags
    all_cleaned_tags = []
    for tags in df_with_flavors['flavor_tags_cleaned']:
        all_cleaned_tags.extend(tags)
    
    cleaned_counts = Counter(all_cleaned_tags)
    print(f"\nCleaned flavor tag statistics:")
    print(f"  - Unique flavor tags after cleaning: {len(cleaned_counts)}")
    print(f"  - Total tag instances: {len(all_cleaned_tags)}")
    print(f"  - Average tags per wine: {len(all_cleaned_tags)/len(df_with_flavors):.1f}")
    
    # Show top tags after cleaning
    print(f"\nTop 30 flavor tags after cleaning:")
    for i, (tag, count) in enumerate(cleaned_counts.most_common(30)):
        print(f"  {i+1:2d}. {tag:<15} ({count:4d} wines)")
    
    # Show what was removed
    removed_tags = set(tag_counts.keys()) - set(cleaned_counts.keys())
    print(f"\nRemoved {len(removed_tags)} problematic tags:")
    removed_sorted = sorted([(tag, tag_counts[tag]) for tag in removed_tags], 
                           key=lambda x: x[1], reverse=True)
    for tag, count in removed_sorted[:20]:
        print(f"  - {tag:<15} ({count:4d} wines)")
    
    # Create final dataset with required features + cleaned tags
    required_features = ['variety', 'country', 'province', 'age', 'region_hierarchy']
    final_columns = required_features + ['flavor_tags_str']
    
    # Only keep wines with all required features
    df_final = df_with_flavors.dropna(subset=required_features)[final_columns].copy()
    df_final.rename(columns={'flavor_tags_str': 'flavor_tags'}, inplace=True)
    
    print(f"\nFinal dataset:")
    print(f"  - Wines: {len(df_final)}")
    print(f"  - Features: {list(df_final.columns)}")
    
    # Save the cleaned dataset
    output_file = 'backend/data/wine_clean_flavors_only.csv'
    df_final.to_csv(output_file, index=False)
    print(f"  - Saved to: {output_file}")
    
    # Show sample
    print(f"\nSample wines:")
    for i, row in df_final.head(5).iterrows():
        print(f"  {row['variety']:<15}: {row['flavor_tags']}")
    
    return df_final, cleaned_counts

if __name__ == "__main__":
    print("🍷 Cleaning existing flavor tags")
    print("=" * 50)
    
    df_clean, flavor_counts = clean_flavor_tags()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully cleaned flavor tags")
    print(f"   - {len(df_clean)} wines with clean flavor tags")
    print(f"   - {len(flavor_counts)} unique flavor terms")
    print(f"   - Removed non-flavors, plurals, and vague colors")
    print(f"   - Kept interesting descriptive terms")
    print(f"   - Ready for model training!") 