#!/usr/bin/env python3
"""
Mouthfeel prediction service for wine characteristics.

This module provides functions to predict mouthfeel tags for wines using
the trained PyTorch multilabel classifier.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import pickle
from typing import List, Dict
import joblib

# PyTorch imports
import torch
import torch.nn as nn

# Add the services directory to path
sys.path.append(os.path.dirname(__file__))

# Global variables for model caching
_model = None
_scaler = None
_encoders = None

# Model paths
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

class MouthfeelClassifier(nn.Module):
    """PyTorch neural network for multilabel mouthfeel classification"""
    
    def __init__(self, input_size, num_mouthfeel_tags, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        super(MouthfeelClassifier, self).__init__()
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_mouthfeel_tags))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))

def _load_model_and_encoders():
    """Load the model, scaler, and encoders with detailed progress"""
    global _model, _scaler, _encoders
    
    if _model is not None and _scaler is not None and _encoders is not None:
        return _model, _scaler, _encoders
    
    print("🍷 Loading Mouthfeel Prediction Model...")
    start_time = time.time()
    
    # Step 1: Load multilabel binarizer first to get number of classes
    print("  📊 Loading multilabel binarizer...")
    mlb_path = os.path.join(MODELS_DIR, "mouthfeel_multilabel_binarizer.pkl")
    mlb = joblib.load(mlb_path)
    print(f"     ✅ {len(mlb.classes_)} mouthfeel tags mapped")
    
    # Step 2: Initialize and load PyTorch model
    print("  🧠 Loading PyTorch neural network...")
    model_start = time.time()
    
    # Initialize model with correct architecture
    input_size = 95  # 88 variety encodings + 3 ordinal + 4 numeric
    num_mouthfeel_tags = len(mlb.classes_)
    
    _model = MouthfeelClassifier(
        input_size=input_size,
        num_mouthfeel_tags=num_mouthfeel_tags,
        hidden_sizes=[512, 256, 128],
        dropout_rate=0.3
    )
    
    # Load model weights
    model_path = os.path.join(MODELS_DIR, "mouthfeel_model.pth")
    
    # Load with threading safety - avoid potential deadlock in Flask
    try:
        # Disable threading for torch.load to prevent Flask conflicts
        torch.set_num_threads(1)
        _model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        _model.eval()
        model_time = time.time() - model_start
        print(f"     ✅ Neural network loaded in {model_time:.2f}s")
    except Exception as e:
        print(f"     ❌ Failed to load PyTorch model: {e}")
        # Try alternative loading method
        try:
            print("     🔄 Trying alternative PyTorch loading method...")
            state_dict = torch.load(model_path, map_location='cpu')
            _model.load_state_dict(state_dict)
            _model.eval()
            model_time = time.time() - model_start
            print(f"     ✅ Neural network loaded with fallback method in {model_time:.2f}s")
        except Exception as e2:
            print(f"     ❌ Both PyTorch loading methods failed: {e2}")
            raise
    
    # Step 3: Load feature scaler
    print("  📏 Loading feature scaler...")
    scaler_start = time.time()
    scaler_path = os.path.join(MODELS_DIR, "mouthfeel_scaler.pkl")
    _scaler = joblib.load(scaler_path)
    scaler_time = time.time() - scaler_start
    print(f"     ✅ Feature scaler loaded in {scaler_time:.2f}s")
    
    # Step 4: Load encoders
    print("  🔧 Loading feature encoders...")
    encoders_start = time.time()
    
    # Load base encoders
    print("     🏗️  Loading base encoders...")
    base_encoders_path = os.path.join(MODELS_DIR, "mouthfeel_encoders.pkl")
    base_encoders = joblib.load(base_encoders_path)
    print(f"        ✅ Ordinal encoder loaded")
    
    # Load variety target encoders
    print("     🎯 Loading variety target encoders...")
    variety_encoders_path = os.path.join(MODELS_DIR, "mouthfeel_variety_target_encoders.pkl")
    variety_start = time.time()
    
    try:
        # Use standard pickle instead of joblib to avoid threading issues
        with open(variety_encoders_path, 'rb') as f:
            variety_target_encoders = pickle.load(f)
        variety_time = time.time() - variety_start
        print(f"        ✅ {len(variety_target_encoders)} variety encoders loaded in {variety_time:.2f}s")
    except Exception as e:
        print(f"        ❌ Failed to load variety encoders with pickle: {e}")
        print(f"        🔄 Falling back to joblib (may hang)...")
        try:
            variety_target_encoders = joblib.load(variety_encoders_path)
            variety_time = time.time() - variety_start
            print(f"        ✅ {len(variety_target_encoders)} variety encoders loaded with joblib in {variety_time:.2f}s")
        except Exception as e2:
            print(f"        ❌ Failed with joblib too: {e2}")
            raise
    
    _encoders = {
        'variety_target_encoders': variety_target_encoders,
        'mlb': mlb,
        'ordinal_encoder': base_encoders['ordinal_encoder']
    }
    
    encoders_time = time.time() - encoders_start
    total_time = time.time() - start_time
    
    print(f"  ✅ All encoders loaded in {encoders_time:.2f}s")
    print(f"🎉 Mouthfeel model fully loaded in {total_time:.2f}s")
    print(f"   Ready to predict from {len(variety_target_encoders)} varieties")
    print(f"   Can predict {len(mlb.classes_)} different mouthfeel tags")
    
    return _model, _scaler, _encoders

def encode_input_features(variety, country, province, age, region_hierarchy, price_min, price_max, rating):
    """Encode input features using the same pipeline as training"""
    _, _, encoders = _load_model_and_encoders()
    
    # Create DataFrame
    df = pd.DataFrame([{
        "variety": variety,
        "country": country,
        "province": province,
        "age": age,
        "region_hierarchy": region_hierarchy,
        "price_min": price_min,
        "price_max": price_max,
        "rating": rating
    }])
    
    # Apply encodings in the same order as training
    
    # 1. Variety target encoding - create vector of all mouthfeel tag encodings
    variety_target_encoders = encoders['variety_target_encoders']
    mlb = encoders['mlb']
    
    variety_encoded_matrix = np.zeros((1, len(mlb.classes_)))
    for i, mouthfeel_tag in enumerate(mlb.classes_):
        encoder = variety_target_encoders[mouthfeel_tag]
        try:
            encoded_col = encoder.transform(df[['variety']])
            variety_encoded_matrix[0, i] = encoded_col['variety'].values[0]
        except Exception:
            # Handle unknown varieties gracefully
            variety_encoded_matrix[0, i] = 0.0
    
    # 2. Ordinal encoding for country, province, region_hierarchy
    ordinal_encoder = encoders['ordinal_encoder']
    ordinal_features = ['country', 'province', 'region_hierarchy']
    try:
        df_ordinal_encoded = ordinal_encoder.transform(df[ordinal_features])
    except Exception:
        # Handle unknown categories gracefully
        df_ordinal_encoded = pd.DataFrame(np.zeros((1, len(ordinal_features))), 
                                         columns=ordinal_features)
    
    # 3. Combine all encoded features
    # Structure: [variety_encoding(n_mouthfeels), country, province, region_hierarchy, age, price_min, price_max, rating]
    X_encoded = np.hstack([
        variety_encoded_matrix,  # Target-encoded variety (n_mouthfeel_tags columns)
        df_ordinal_encoded.values,  # Ordinal-encoded country, province, region_hierarchy
        df[['age', 'price_min', 'price_max', 'rating']].values  # Numeric features
    ])
    
    return X_encoded.astype(np.float32)

def predict_mouthfeel_tags(variety: str, country: str, province: str, age: float, 
                          region_hierarchy: str, price_min: float, price_max: float, 
                          rating: float, confidence_threshold: float = 0.5, 
                          top_k: int = 10) -> List[Dict]:
    """
    Predict mouthfeel tags for a wine using the trained multilabel classifier.
    
    Args:
        variety: Wine grape variety (e.g., "Chardonnay")
        country: Country of origin (e.g., "France")
        province: Province/state (e.g., "Burgundy")
        age: Wine age in years
        region_hierarchy: Region hierarchy string (e.g., "France > Burgundy")
        price_min: Minimum price from price prediction
        price_max: Maximum price from price prediction
        rating: Wine rating from rating prediction
        confidence_threshold: Minimum confidence to include a mouthfeel tag (0.0-1.0)
        top_k: Maximum number of mouthfeel tags to return
    
    Returns:
        List of dictionaries with 'mouthfeel' and 'confidence' keys, sorted by confidence
    """
    
    # Load model and encoders
    model, scaler, encoders = _load_model_and_encoders()
    
    # Encode input features
    X_encoded = encode_input_features(
        variety, country, province, age, region_hierarchy, 
        price_min, price_max, rating
    )
    
    # Scale features
    X_scaled = scaler.transform(X_encoded)
    
    # Convert to PyTorch tensor
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Make prediction
    with torch.no_grad():
        model.eval()
        predictions = model(X_tensor)
        probabilities = predictions.cpu().numpy()[0]  # Get first (and only) sample
    
    # Get mouthfeel tag names
    mlb = encoders['mlb']
    mouthfeel_tags = mlb.classes_
    
    # Create results list
    results = []
    for i, (mouthfeel_tag, confidence) in enumerate(zip(mouthfeel_tags, probabilities)):
        if confidence >= confidence_threshold:
            results.append({
                'mouthfeel': mouthfeel_tag,
                'confidence': float(confidence)
            })
    
    # Sort by confidence (highest first) and limit to top_k
    results.sort(key=lambda x: x['confidence'], reverse=True)
    results = results[:top_k]
    
    return results

def predict_mouthfeel_tags_from_dict(input_dict: Dict, confidence_threshold: float = 0.5, 
                                    top_k: int = 10) -> List[Dict]:
    """
    Convenience function to predict mouthfeel tags from a dictionary input.
    
    Args:
        input_dict: Dictionary with wine features
        confidence_threshold: Minimum confidence to include a mouthfeel tag
        top_k: Maximum number of mouthfeel tags to return
    
    Returns:
        List of dictionaries with 'mouthfeel' and 'confidence' keys
    """
    
    return predict_mouthfeel_tags(
        variety=input_dict.get('variety', 'Unknown'),
        country=input_dict.get('country', 'Unknown'),
        province=input_dict.get('province', 'Unknown'),
        age=input_dict.get('age', 0),
        region_hierarchy=input_dict.get('region_hierarchy', 'Unknown'),
        price_min=input_dict.get('price_min', 0.0),
        price_max=input_dict.get('price_max', 0.0),
        rating=input_dict.get('rating', 80.0),
        confidence_threshold=confidence_threshold,
        top_k=top_k
    )

if __name__ == "__main__":
    # Test the mouthfeel prediction
    print("🍷 Testing Mouthfeel Prediction")
    print("=" * 50)
    
    # Example wine
    test_wine = {
        "variety": "Cabernet Sauvignon",
        "country": "France",
        "province": "Bordeaux",
        "age": 5,
        "region_hierarchy": "France > Bordeaux",
        "price_min": 45.0,
        "price_max": 65.0,
        "rating": 92.0
    }
    
    print("Test wine:")
    for key, value in test_wine.items():
        print(f"  {key}: {value}")
    
    print("\nPredicted mouthfeel:")
    mouthfeel = predict_mouthfeel_tags_from_dict(test_wine, confidence_threshold=0.3, top_k=15)
    
    for i, mouthfeel_info in enumerate(mouthfeel):
        print(f"  {i+1:2d}. {mouthfeel_info['mouthfeel']:<15} ({mouthfeel_info['confidence']:.3f})")
    
    print(f"\n✅ Found {len(mouthfeel)} mouthfeel tags above confidence threshold") 