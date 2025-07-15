#!/usr/bin/env python3
"""
Wine Tag Analysis Script

This script analyzes the tag distribution in the wine_clean_tagged dataset
to identify:
1. How many tags each wine has for flavor and mouthfeel
2. Which wines are untagged (if any)
3. Distribution statistics for tag counts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def count_tags(tag_string):
    """Count the number of tags in a comma-separated string"""
    if pd.isna(tag_string) or tag_string == '' or tag_string.strip() == '':
        return 0
    # Split by comma and strip whitespace
    tags = [tag.strip() for tag in tag_string.split(',')]
    # Filter out empty strings
    tags = [tag for tag in tags if tag]
    return len(tags)

def analyze_wine_tags():
    """Main analysis function"""
    print("Loading wine dataset...")
    
    # Load the dataset
    df = pd.read_csv('backend/data/wine_clean_tagged.csv')
    
    print(f"Dataset loaded: {len(df)} wines\n")
    
    # Count tags for each wine
    df['flavor_tag_count'] = df['flavor_tags'].apply(count_tags)
    df['mouthfeel_tag_count'] = df['mouthfeel_tags'].apply(count_tags)
    
    # Basic statistics
    print("=== TAG COUNT STATISTICS ===")
    print("\nFlavor Tags:")
    print(f"  Mean tags per wine: {df['flavor_tag_count'].mean():.2f}")
    print(f"  Median tags per wine: {df['flavor_tag_count'].median():.1f}")
    print(f"  Min tags: {df['flavor_tag_count'].min()}")
    print(f"  Max tags: {df['flavor_tag_count'].max()}")
    print(f"  Standard deviation: {df['flavor_tag_count'].std():.2f}")
    
    print("\nMouthfeel Tags:")
    print(f"  Mean tags per wine: {df['mouthfeel_tag_count'].mean():.2f}")
    print(f"  Median tags per wine: {df['mouthfeel_tag_count'].median():.1f}")
    print(f"  Min tags: {df['mouthfeel_tag_count'].min()}")
    print(f"  Max tags: {df['mouthfeel_tag_count'].max()}")
    print(f"  Standard deviation: {df['mouthfeel_tag_count'].std():.2f}")
    
    # Check for untagged wines
    print("\n=== UNTAGGED WINES ANALYSIS ===")
    
    # Wines with no flavor tags
    no_flavor_tags = df[df['flavor_tag_count'] == 0]
    print(f"\nWines with NO flavor tags: {len(no_flavor_tags)} ({len(no_flavor_tags)/len(df)*100:.1f}%)")
    
    # Wines with no mouthfeel tags
    no_mouthfeel_tags = df[df['mouthfeel_tag_count'] == 0]
    print(f"Wines with NO mouthfeel tags: {len(no_mouthfeel_tags)} ({len(no_mouthfeel_tags)/len(df)*100:.1f}%)")
    
    # Wines with no tags at all
    no_tags_at_all = df[(df['flavor_tag_count'] == 0) & (df['mouthfeel_tag_count'] == 0)]
    print(f"Wines with NO tags at all: {len(no_tags_at_all)} ({len(no_tags_at_all)/len(df)*100:.1f}%)")
    
    # Distribution of tag counts
    print("\n=== TAG COUNT DISTRIBUTIONS ===")
    
    print("\nFlavor Tag Count Distribution:")
    flavor_dist = Counter(df['flavor_tag_count'])
    for count in sorted(flavor_dist.keys()):
        percentage = flavor_dist[count] / len(df) * 100
        print(f"  {count:2d} tags: {flavor_dist[count]:5d} wines ({percentage:5.1f}%)")
    
    print("\nMouthfeel Tag Count Distribution:")
    mouthfeel_dist = Counter(df['mouthfeel_tag_count'])
    for count in sorted(mouthfeel_dist.keys()):
        percentage = mouthfeel_dist[count] / len(df) * 100
        print(f"  {count:2d} tags: {mouthfeel_dist[count]:5d} wines ({percentage:5.1f}%)")
    
    # Sample untagged wines (if any)
    if len(no_flavor_tags) > 0:
        print("\n=== SAMPLE WINES WITH NO FLAVOR TAGS ===")
        sample_size = min(5, len(no_flavor_tags))
        sample_wines = no_flavor_tags[['title', 'variety', 'description', 'flavor_tags']].head(sample_size)
        for idx, wine in sample_wines.iterrows():
            print(f"\nWine: {wine['title']}")
            print(f"Variety: {wine['variety']}")
            print(f"Description: {wine['description'][:100]}...")
            print(f"Flavor tags: '{wine['flavor_tags']}'")
    
    if len(no_mouthfeel_tags) > 0:
        print("\n=== SAMPLE WINES WITH NO MOUTHFEEL TAGS ===")
        sample_size = min(5, len(no_mouthfeel_tags))
        sample_wines = no_mouthfeel_tags[['title', 'variety', 'description', 'mouthfeel_tags']].head(sample_size)
        for idx, wine in sample_wines.iterrows():
            print(f"\nWine: {wine['title']}")
            print(f"Variety: {wine['variety']}")
            print(f"Description: {wine['description'][:100]}...")
            print(f"Mouthfeel tags: '{wine['mouthfeel_tags']}'")
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Wine Tag Analysis', fontsize=16, fontweight='bold')
    
    # Flavor tag count histogram
    axes[0, 0].hist(df['flavor_tag_count'], bins=range(df['flavor_tag_count'].max()+2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Flavor Tag Counts')
    axes[0, 0].set_xlabel('Number of Flavor Tags')
    axes[0, 0].set_ylabel('Number of Wines')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mouthfeel tag count histogram
    axes[0, 1].hist(df['mouthfeel_tag_count'], bins=range(df['mouthfeel_tag_count'].max()+2), 
                    alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Distribution of Mouthfeel Tag Counts')
    axes[0, 1].set_xlabel('Number of Mouthfeel Tags')
    axes[0, 1].set_ylabel('Number of Wines')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot comparison
    tag_data = [df['flavor_tag_count'], df['mouthfeel_tag_count']]
    axes[1, 0].boxplot(tag_data, labels=['Flavor Tags', 'Mouthfeel Tags'])
    axes[1, 0].set_title('Tag Count Comparison')
    axes[1, 0].set_ylabel('Number of Tags')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: flavor vs mouthfeel tags
    axes[1, 1].scatter(df['flavor_tag_count'], df['mouthfeel_tag_count'], 
                       alpha=0.6, s=20, color='purple')
    axes[1, 1].set_title('Flavor Tags vs Mouthfeel Tags')
    axes[1, 1].set_xlabel('Number of Flavor Tags')
    axes[1, 1].set_ylabel('Number of Mouthfeel Tags')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wine_tag_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'wine_tag_analysis.png'")
    
    # Analysis and recommendations
    print("\n=== ANALYSIS AND RECOMMENDATIONS ===")
    
    untagged_percentage = (len(no_flavor_tags) + len(no_mouthfeel_tags)) / (2 * len(df)) * 100
    
    if untagged_percentage > 10:
        print("⚠️  HIGH CONCERN: A significant portion of wines are untagged")
    elif untagged_percentage > 5:
        print("⚠️  MODERATE CONCERN: Some wines are untagged")
    else:
        print("✅ LOW CONCERN: Most wines are properly tagged")
    
    print(f"\nOverall untagged rate: {untagged_percentage:.1f}%")
    
    print("\nIs it bad if some wines are untagged?")
    print("- YES, if you're training ML models - untagged data reduces training quality")
    print("- YES, if you're doing flavor/mouthfeel analysis - missing data skews results")
    print("- MAYBE, if tags were supposed to be comprehensive - indicates incomplete processing")
    print("- NO, if some wines genuinely lack describable flavor/mouthfeel characteristics")
    
    print("\nPossible reasons for untagged wines:")
    print("1. Vocabulary sorting issues (as you suspected)")
    print("2. Text processing failures during tag extraction")
    print("3. Very brief or vague wine descriptions")
    print("4. Wines with unusual flavor profiles not captured in your vocabulary")
    print("5. Processing errors or edge cases in the tagging algorithm")
    
    return df

if __name__ == "__main__":
    df = analyze_wine_tags()
    
    # Save the enhanced dataset with tag counts
    print(f"\nSaving enhanced dataset with tag counts to 'wine_with_tag_counts.csv'...")
    df.to_csv('wine_with_tag_counts.csv', index=False)
    print("Done!") 