import pandas as pd
import numpy as np
import joblib
import os

# Get the directory this script is in and construct absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Global variables for lazy loading
_ordinal_encoder = None
_target_encoder = None
_variety_target_encoders = None
_variety_encoding_cache = None

def _load_basic_encoders():
    """Lazy load the basic encoders (ordinal and target)"""
    global _ordinal_encoder, _target_encoder
    if _ordinal_encoder is None:
        _ordinal_encoder = joblib.load(os.path.join(MODELS_DIR, "ordinal_encoder.pkl"))
    if _target_encoder is None:
        _target_encoder = joblib.load(os.path.join(MODELS_DIR, "target_encoder.pkl"))
    return _ordinal_encoder, _target_encoder

def _load_variety_encoders():
    """Lazy load the variety target encoders and cache"""
    global _variety_target_encoders, _variety_encoding_cache
    
    if _variety_target_encoders is None:
        variety_encoders_path = os.path.join(MODELS_DIR, "variety_target_encoders.pkl")
        if os.path.exists(variety_encoders_path):
            print("Loading variety target encoders...")
            _variety_target_encoders = joblib.load(variety_encoders_path)
            print(f"Loaded {len(_variety_target_encoders)} variety target encoders")
        else:
            raise FileNotFoundError("Variety target encoders not found. Train the flavor model first.")
    
    if _variety_encoding_cache is None:
        cache_file = os.path.join(MODELS_DIR, "variety_encoding_cache.pkl")
        if os.path.exists(cache_file):
            print("Loading variety encoding cache...")
            _variety_encoding_cache = joblib.load(cache_file)
            print(f"Loaded variety encoding cache with {len(_variety_encoding_cache)} varieties")
        else:
            print("Variety encoding cache not found. Will create it during first use.")
            _variety_encoding_cache = {}
    
    return _variety_target_encoders, _variety_encoding_cache

def encode_inputs(grape, country, province, age, region_hierarchy, price_min, price_max, rating):
    # Lazy load basic encoders only when needed
    ordinal_encoder, target_encoder = _load_basic_encoders()
    
    # Construct DataFrame
    df = pd.DataFrame([{
        "variety": grape,
        "country": country,
        "province": province,
        "age": age,
        "region_hierarchy": region_hierarchy,
        "price_min": price_min,
        "price_max": price_max,
        "rating": rating
    }])

    # Apply ordinal encoding
    df[["variety", "country", "province"]] = ordinal_encoder.transform(
        df[["variety", "country", "province"]]
    )

    # Apply target encoding
    df["region_hierarchy"] = target_encoder.transform(df["region_hierarchy"])

    # Optional: Scale numeric features if needed (add scaler here if used)

    return df.values.astype(np.float32)

def _compute_variety_encoding(grape):
    """Compute target encoding for a specific variety across all flavor tags."""
    variety_target_encoders, _ = _load_variety_encoders()
    
    variety_df = pd.DataFrame([{"variety": grape}])
    variety_encoded = np.zeros(len(variety_target_encoders))
    
    for i, (flavor_tag, encoder) in enumerate(variety_target_encoders.items()):
        try:
            encoded_value = encoder.transform(variety_df)
            variety_encoded[i] = encoded_value['variety'].values[0]
        except Exception as e:
            # Handle unknown varieties gracefully
            variety_encoded[i] = 0.0  # Default value for unknown varieties
    
    return variety_encoded

def encode_inputs_with_variety_target_encoding(grape, country, province, age, region_hierarchy, price_min, price_max, rating):
    """
    Enhanced encoding function that uses target encoding for variety based on flavor tags.
    This should be used for flavor prediction models.
    
    Now optimized with caching for much faster inference.
    """
    # Lazy load variety encoders and cache only when needed
    variety_target_encoders, variety_encoding_cache = _load_variety_encoders()
    
    # First get the basic encoded features (without variety)
    basic_features = encode_inputs("placeholder", country, province, age, region_hierarchy, price_min, price_max, rating)
    other_features = basic_features[0][1:]  # Skip the variety column
    
    # Check if we have this variety's encoding cached
    if grape in variety_encoding_cache:
        variety_encoded = variety_encoding_cache[grape]
    else:
        # Compute and cache the encoding for this variety
        print(f"Computing target encoding for new variety: {grape}")
        variety_encoded = _compute_variety_encoding(grape)
        variety_encoding_cache[grape] = variety_encoded
        
        # Save updated cache with absolute path
        cache_file = os.path.join(MODELS_DIR, "variety_encoding_cache.pkl")
        joblib.dump(variety_encoding_cache, cache_file)
        print(f"Cached encoding for {grape}. Cache now has {len(variety_encoding_cache)} varieties.")
    
    # Combine target-encoded variety with other features
    combined_features = np.concatenate([variety_encoded, other_features])
    
    return combined_features.astype(np.float32).reshape(1, -1)

def create_variety_encoding_cache():
    """
    Pre-compute encodings for all known varieties to speed up inference.
    This should be run after training the flavor model.
    """
    variety_target_encoders, variety_encoding_cache = _load_variety_encoders()
    
    # Load the training data to get all known varieties
    try:
        data_file = os.path.join(BASE_DIR, "..", "data", "wine_with_predictions.csv")
        df = pd.read_csv(data_file)
        unique_varieties = df['variety'].unique()
        print(f"Found {len(unique_varieties)} unique varieties in training data")
        
        cache = {}
        for i, variety in enumerate(unique_varieties):
            if i % 100 == 0:
                print(f"Pre-computing encodings: {i}/{len(unique_varieties)} varieties...")
            
            try:
                cache[variety] = _compute_variety_encoding(variety)
            except Exception as e:
                print(f"Warning: Could not encode variety '{variety}': {e}")
                continue
        
        # Save the cache with absolute path
        cache_file = os.path.join(MODELS_DIR, "variety_encoding_cache.pkl")
        joblib.dump(cache, cache_file)
        print(f"✅ Pre-computed and cached encodings for {len(cache)} varieties")
        print(f"Cache saved to {cache_file}")
        
        # Update global cache
        global _variety_encoding_cache
        _variety_encoding_cache = cache
        
    except Exception as e:
        print(f"Error creating variety encoding cache: {e}")

if __name__ == "__main__":
    # Create the variety encoding cache
    print("Creating variety encoding cache for faster inference...")
    create_variety_encoding_cache()
