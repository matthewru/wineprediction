#!/usr/bin/env python3
"""
Predict wine flavors using the trained PyTorch multilabel classifier.

This module loads the trained FlavorNet model and all encoders to predict
flavor tags for wine based on variety, country, province, age, region_hierarchy,
price_min, price_max, and rating.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import pickle
import time
from typing import Dict, List, Tuple

# Get the directory this script is in and construct absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Global variables for lazy loading
_model = None
_scaler = None
_encoders = None
_model_info = None

def check_models_exist():
    """Check if all required model files exist without loading them"""
    required_files = [
        "flavor_model_info.pkl",
        "flavor_model.pth", 
        "flavor_scaler.pkl",
        "variety_target_encoders.pkl",
        "flavor_multilabel_binarizer.pkl",
        "flavor_encoders.pkl"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files

class FlavorNet(nn.Module):
    """PyTorch Neural Network for multilabel flavor classification"""
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(FlavorNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate * 0.7)
        
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(dropout_rate * 0.7)
        
        # Output layer
        self.fc_out = nn.Linear(64, output_dim)
    
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Layer 4
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # Output layer with sigmoid for multilabel
        x = self.fc_out(x)
        x = torch.sigmoid(x)
        
        return x

def _load_model_and_encoders():
    """Load the model, scaler, and encoders with detailed progress"""
    global _model, _scaler, _encoders, _model_info
    
    if _model is not None and _scaler is not None and _encoders is not None:
        return _model, _scaler, _encoders  # Already loaded
    
    print("🔄 Loading flavor prediction model components...")
    start_time = time.time()
    
    # Step 1: Load model info
    print("  📋 Loading model configuration...")
    model_info_path = os.path.join(MODELS_DIR, "flavor_model_info.pkl")
    if not os.path.exists(model_info_path):
        raise FileNotFoundError("Flavor model not found. Train the model first with train_flavor_predictor.py")
    
    _model_info = joblib.load(model_info_path)
    print(f"     ✅ Model config: {_model_info['input_dim']} inputs → {_model_info['output_dim']} outputs")
    
    # Step 2: Create and load PyTorch model
    print("  🧠 Creating neural network architecture...")
    model_start = time.time()
    _model = FlavorNet(_model_info['input_dim'], _model_info['output_dim'])
    
    print("  📦 Loading trained model weights...")
    model_path = os.path.join(MODELS_DIR, "flavor_model.pth")
    
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
    scaler_path = os.path.join(MODELS_DIR, "flavor_scaler.pkl")
    _scaler = joblib.load(scaler_path)
    scaler_time = time.time() - scaler_start
    print(f"     ✅ Feature scaler loaded in {scaler_time:.2f}s")
    
    # Step 4: Load encoders (this is the problematic part)
    print("  🔧 Loading feature encoders...")
    encoders_start = time.time()
    
    # Load small files first
    print("     📊 Loading multilabel binarizer...")
    mlb_path = os.path.join(MODELS_DIR, "flavor_multilabel_binarizer.pkl")
    mlb = joblib.load(mlb_path)
    print(f"        ✅ {len(mlb.classes_)} flavor tags mapped")
    
    print("     🏗️  Loading base encoders...")
    flavor_encoders_path = os.path.join(MODELS_DIR, "flavor_encoders.pkl")
    flavor_encoders = joblib.load(flavor_encoders_path)
    print(f"        ✅ Ordinal and target encoders loaded")
    
    # Load the big file with safer approach (use pickle instead of joblib)
    print("     🎯 Loading variety target encoders...")
    variety_encoders_path = os.path.join(MODELS_DIR, "variety_target_encoders.pkl")
    print(f"        📦 Loading 18MB encoder file with Python pickle (safer for threading)...")
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
        'ordinal_encoder': flavor_encoders['ordinal_encoder'],
        'target_encoder': flavor_encoders['target_encoder']
    }
    
    encoders_time = time.time() - encoders_start
    total_time = time.time() - start_time
    
    print(f"  ✅ All encoders loaded in {encoders_time:.2f}s")
    print(f"🎉 Flavor model fully loaded in {total_time:.2f}s")
    print(f"   Ready to predict from {len(variety_target_encoders)} varieties")
    print(f"   Can predict {len(mlb.classes_)} different flavor tags")
    
    return _model, _scaler, _encoders

def load_model_eagerly():
    """Public function to eagerly load the model during app startup"""
    try:
        print("🍷 Initializing flavor prediction model...")
        _load_model_and_encoders()
        return True
    except Exception as e:
        print(f"❌ Failed to load flavor model: {e}")
        return False

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
    
    # 1. Variety target encoding - create vector of all flavor tag encodings
    variety_target_encoders = encoders['variety_target_encoders']
    mlb = encoders['mlb']
    
    variety_encoded_matrix = np.zeros((1, len(mlb.classes_)))
    for i, flavor_tag in enumerate(mlb.classes_):
        encoder = variety_target_encoders[flavor_tag]
        try:
            encoded_col = encoder.transform(df[['variety']])
            variety_encoded_matrix[0, i] = encoded_col['variety'].values[0]
        except Exception:
            # Handle unknown varieties gracefully
            variety_encoded_matrix[0, i] = 0.0
    
    # 2. Ordinal encoding for country, province
    ordinal_encoder = encoders['ordinal_encoder']
    ordinal_features = ['country', 'province']
    try:
        df_ordinal_encoded = ordinal_encoder.transform(df[ordinal_features])
    except Exception:
        # Handle unknown categories gracefully
        df_ordinal_encoded = pd.DataFrame(np.zeros((1, len(ordinal_features))), 
                                         columns=ordinal_features)
    
    # 3. Target encoding for region_hierarchy
    target_encoder = encoders['target_encoder']
    try:
        df_region_encoded = target_encoder.transform(df[['region_hierarchy']])
    except Exception:
        # Handle unknown regions gracefully
        df_region_encoded = pd.DataFrame(np.zeros((1, 1)), columns=['region_hierarchy'])
    
    # 4. Combine all encoded features
    # Structure: [variety_encoding(n_flavors), country, province, age, region_hierarchy, price_min, price_max, rating]
    X_encoded = np.hstack([
        variety_encoded_matrix,  # Target-encoded variety (n_flavor_tags columns)
        df_ordinal_encoded.values,  # Ordinal-encoded country, province
        df[['age']].values,  # Age (numeric)
        df_region_encoded.values,  # Target-encoded region_hierarchy
        df[['price_min', 'price_max', 'rating']].values  # Price and rating (numeric)
    ])
    
    return X_encoded.astype(np.float32)

def predict_flavor_tags(variety: str, country: str, province: str, age: float, 
                       region_hierarchy: str, price_min: float, price_max: float, 
                       rating: float, confidence_threshold: float = 0.5, 
                       top_k: int = 10) -> List[Dict]:
    """
    Predict flavor tags for a wine using the trained multilabel classifier.
    
    Args:
        variety: Wine grape variety (e.g., "Chardonnay")
        country: Country of origin (e.g., "France")
        province: Province/state (e.g., "Burgundy")
        age: Wine age in years
        region_hierarchy: Region hierarchy string (e.g., "France > Burgundy")
        price_min: Minimum price from price prediction
        price_max: Maximum price from price prediction
        rating: Wine rating from rating prediction
        confidence_threshold: Minimum confidence to include a flavor tag (0.0-1.0)
        top_k: Maximum number of flavor tags to return
    
    Returns:
        List of dictionaries with 'flavor' and 'confidence' keys, sorted by confidence
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
    
    # Get flavor tag names
    mlb = encoders['mlb']
    flavor_tags = mlb.classes_
    
    # Create results list
    results = []
    for i, (flavor_tag, confidence) in enumerate(zip(flavor_tags, probabilities)):
        if confidence >= confidence_threshold:
            results.append({
                'flavor': flavor_tag,
                'confidence': float(confidence)
            })
    
    # Sort by confidence (highest first) and limit to top_k
    results.sort(key=lambda x: x['confidence'], reverse=True)
    results = results[:top_k]
    
    return results

def predict_flavor_tags_from_dict(input_dict: Dict, confidence_threshold: float = 0.5, 
                                 top_k: int = 10) -> List[Dict]:
    """
    Convenience function to predict flavor tags from a dictionary input.
    
    Args:
        input_dict: Dictionary with wine features
        confidence_threshold: Minimum confidence to include a flavor tag
        top_k: Maximum number of flavor tags to return
    
    Returns:
        List of dictionaries with 'flavor' and 'confidence' keys
    """
    
    return predict_flavor_tags(
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
    # Test the flavor prediction
    print("🍷 Testing Flavor Prediction")
    print("=" * 50)
    
    # Example wine
    test_wine = {
        "variety": "Chardonnay",
        "country": "France",
        "province": "Burgundy",
        "age": 3,
        "region_hierarchy": "France > Burgundy",
        "price_min": 25.0,
        "price_max": 35.0,
        "rating": 88.5
    }
    
    print("Test wine:")
    for key, value in test_wine.items():
        print(f"  {key}: {value}")
    
    print("\nPredicted flavors:")
    flavors = predict_flavor_tags_from_dict(test_wine, confidence_threshold=0.3, top_k=15)
    
    for i, flavor_info in enumerate(flavors):
        print(f"  {i+1:2d}. {flavor_info['flavor']:<15} ({flavor_info['confidence']:.3f})")
    
    print(f"\n✅ Found {len(flavors)} flavor tags above confidence threshold") 