#!/usr/bin/env python3
"""
Train a multilabel classifier to predict wine mouthfeel.

This script will:
1. Load wine_mouthfeel_with_predictions.csv as training data
2. Apply ordinal encoding to most features, target encoding for variety
3. Train a PyTorch neural network multilabel classifier for mouthfeel tags
4. Save the trained model and encoders
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, hamming_loss, multilabel_confusion_matrix
from category_encoders import OrdinalEncoder, TargetEncoder
import joblib
import pickle
from pathlib import Path

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Cache directory
CACHE_DIR = 'backend/cache'

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_paths():
    """Get paths for cached files"""
    return {
        'X_encoded': os.path.join(CACHE_DIR, 'mouthfeel_X_encoded.pkl'),
        'y_binary': os.path.join(CACHE_DIR, 'mouthfeel_y_binary.pkl'),
        'encoders': os.path.join(CACHE_DIR, 'mouthfeel_encoders_cache.pkl'),
        'mlb': os.path.join(CACHE_DIR, 'mouthfeel_mlb_cache.pkl'),
        'data_hash': os.path.join(CACHE_DIR, 'mouthfeel_data_hash.txt')
    }

def get_data_hash(df):
    """Get a hash of the dataframe to detect changes"""
    # Simple hash based on shape and a few sample values
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    sample_str = str(df.iloc[0].values) if len(df) > 0 else ""
    columns_str = str(sorted(df.columns.tolist()))
    return hash(shape_str + sample_str + columns_str)

def save_encoded_data(X_encoded, y_binary, encoders, mlb, data_hash):
    """Save encoded data and encoders to cache"""
    ensure_cache_dir()
    cache_paths = get_cache_paths()
    
    print("💾 Caching encoded data for future runs...")
    
    # Save encoded features
    with open(cache_paths['X_encoded'], 'wb') as f:
        pickle.dump(X_encoded, f)
    print(f"  ✅ Cached X_encoded to {cache_paths['X_encoded']}")
    
    # Save binary targets
    with open(cache_paths['y_binary'], 'wb') as f:
        pickle.dump(y_binary, f)
    print(f"  ✅ Cached y_binary to {cache_paths['y_binary']}")
    
    # Save encoders
    with open(cache_paths['encoders'], 'wb') as f:
        pickle.dump(encoders, f)
    print(f"  ✅ Cached encoders to {cache_paths['encoders']}")
    
    # Save multilabel binarizer
    with open(cache_paths['mlb'], 'wb') as f:
        pickle.dump(mlb, f)
    print(f"  ✅ Cached mlb to {cache_paths['mlb']}")
    
    # Save data hash
    with open(cache_paths['data_hash'], 'w') as f:
        f.write(str(data_hash))
    print(f"  ✅ Cached data hash to {cache_paths['data_hash']}")

def load_encoded_data():
    """Load encoded data and encoders from cache"""
    cache_paths = get_cache_paths()
    
    print("📂 Loading encoded data from cache...")
    
    # Load encoded features
    with open(cache_paths['X_encoded'], 'rb') as f:
        X_encoded = pickle.load(f)
    print(f"  ✅ Loaded X_encoded: {X_encoded.shape}")
    
    # Load binary targets
    with open(cache_paths['y_binary'], 'rb') as f:
        y_binary = pickle.load(f)
    print(f"  ✅ Loaded y_binary: {y_binary.shape}")
    
    # Load encoders
    with open(cache_paths['encoders'], 'rb') as f:
        encoders = pickle.load(f)
    print(f"  ✅ Loaded encoders")
    
    # Load multilabel binarizer
    with open(cache_paths['mlb'], 'rb') as f:
        mlb = pickle.load(f)
    print(f"  ✅ Loaded mlb: {len(mlb.classes_)} mouthfeel tags")
    
    return X_encoded, y_binary, encoders, mlb

def is_cache_valid(df):
    """Check if cached data is valid for current dataframe"""
    cache_paths = get_cache_paths()
    
    # Check if all cache files exist
    for path in cache_paths.values():
        if not os.path.exists(path):
            return False
    
    # Check if data hash matches
    try:
        with open(cache_paths['data_hash'], 'r') as f:
            cached_hash = int(f.read().strip())
        current_hash = get_data_hash(df)
        return cached_hash == current_hash
    except:
        return False

def clear_cache():
    """Clear all cached files"""
    cache_paths = get_cache_paths()
    cleared_count = 0
    
    for name, path in cache_paths.items():
        if os.path.exists(path):
            os.remove(path)
            cleared_count += 1
    
    if cleared_count > 0:
        print(f"🗑️  Cleared {cleared_count} cached files")

def load_and_prepare_data():
    """Load the wine_mouthfeel_with_predictions.csv and prepare initial features"""
    print("📂 Loading wine_mouthfeel_with_predictions.csv...")
    df = pd.read_csv('backend/data/wine_mouthfeel_with_predictions.csv')
    print(f"Loaded {len(df)} wines with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Display sample data
    print("\n📊 Sample data:")
    for i, row in df.head(3).iterrows():
        print(f"  {row['variety']}: {row['mouthfeel_tags']}")
    
    return df

def prepare_mouthfeel_targets(df):
    """Prepare the multilabel targets from mouthfeel_tags"""
    print("\n🏷️  Preparing mouthfeel targets...")
    
    # Parse mouthfeel tags into lists
    mouthfeel_lists = []
    for tags_str in df['mouthfeel_tags']:
        if pd.isna(tags_str) or tags_str.strip() == '':
            mouthfeel_lists.append([])
        else:
            # Split by comma and clean
            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            mouthfeel_lists.append(tags)
    
    # Use MultiLabelBinarizer to create binary matrix
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(mouthfeel_lists)
    
    print(f"  Found {len(mlb.classes_)} unique mouthfeel tags")
    print(f"  Average tags per wine: {y_binary.sum(axis=1).mean():.1f}")
    print(f"  Binary matrix shape: {y_binary.shape}")
    
    # Show most common mouthfeel tags
    tag_counts = y_binary.sum(axis=0)
    common_tags = [(mlb.classes_[i], tag_counts[i]) for i in range(len(mlb.classes_))]
    common_tags.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 20 most common mouthfeel tags:")
    for i, (tag, count) in enumerate(common_tags[:20]):
        print(f"    {i+1:2d}. {tag:<15} ({count:4d} wines)")
    
    return y_binary, mlb

def encode_features(df):
    """Encode the features for training"""
    print("\n🔧 Encoding features...")
    
    # Define feature sets
    features = ['variety', 'country', 'province', 'age', 'region_hierarchy', 'price_min', 'price_max', 'rating']
    X = df[features].copy()
    
    print(f"  Original features: {list(X.columns)}")
    print(f"  Feature matrix shape: {X.shape}")
    
    # Split features by encoding type
    ordinal_features = ['country', 'province', 'region_hierarchy']  # Ordinal encoding
    target_features = ['variety']  # Target encoding with mouthfeel tags
    numeric_features = ['age', 'price_min', 'price_max', 'rating']  # Fixed: removed region_hierarchy
    
    return X, ordinal_features, target_features, numeric_features

def train_variety_target_encoders(X, y_binary, mlb):
    """Train target encoders for variety based on each mouthfeel tag"""
    print("\n🎯 Training variety target encoders for mouthfeel tags...")
    
    variety_target_encoders = {}
    
    for i, mouthfeel_tag in enumerate(mlb.classes_):
        if (i + 1) % 10 == 0 or (i + 1) == len(mlb.classes_):
            print(f"  Training encoder {i + 1}/{len(mlb.classes_)}: {mouthfeel_tag}")
        
        # Target is whether this wine has this specific mouthfeel tag
        target = y_binary[:, i]
        
        # Train target encoder for this mouthfeel tag
        encoder = TargetEncoder(cols=['variety'], smoothing=5)
        encoder.fit(X[['variety']], target)
        
        variety_target_encoders[mouthfeel_tag] = encoder
    
    print(f"  ✅ Trained {len(variety_target_encoders)} variety target encoders")
    return variety_target_encoders

def encode_training_data(X, y_binary, mlb, ordinal_features, target_features, numeric_features):
    """Apply all encodings to create final training matrix"""
    print("\n🔄 Applying feature encodings...")
    
    # 1. Train variety target encoders
    variety_target_encoders = train_variety_target_encoders(X, y_binary, mlb)
    
    # 2. Ordinal encoding for country, province, region_hierarchy
    ordinal_encoder = OrdinalEncoder(cols=ordinal_features)
    X_ordinal = ordinal_encoder.fit_transform(X[ordinal_features])
    
    # 3. Variety target encoding - create vector of all mouthfeel tag encodings
    variety_encoded_matrix = np.zeros((len(X), len(mlb.classes_)))
    for i, mouthfeel_tag in enumerate(mlb.classes_):
        encoder = variety_target_encoders[mouthfeel_tag]
        encoded_col = encoder.transform(X[['variety']])
        variety_encoded_matrix[:, i] = encoded_col['variety'].values
    
    # 4. Combine all encoded features
    # Structure: [variety_encoding(n_mouthfeels), country, province, region_hierarchy, age, price_min, price_max, rating]
    X_encoded = np.hstack([
        variety_encoded_matrix,  # Target-encoded variety (n_mouthfeel_tags columns)
        X_ordinal.values,  # Ordinal-encoded country, province, region_hierarchy
        X[['age', 'price_min', 'price_max', 'rating']].values  # Numeric features
    ])
    
    print(f"  Final encoded matrix shape: {X_encoded.shape}")
    print(f"  Features breakdown:")
    print(f"    - Variety target encoding: {len(mlb.classes_)} columns")
    print(f"    - Ordinal features (country, province, region_hierarchy): {len(ordinal_features)} columns")
    print(f"    - Numeric features (age, price_min, price_max, rating): {len(numeric_features)} columns")
    print(f"    - Total: {X_encoded.shape[1]} columns")
    
    # Save encoders for later use
    encoders = {
        'variety_target_encoders': variety_target_encoders,
        'mlb': mlb,
        'ordinal_encoder': ordinal_encoder
    }
    
    return X_encoded, encoders

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

def train_pytorch_model(X_encoded, y_binary):
    """Train the PyTorch neural network multilabel classifier"""
    print("\n🧠 Training PyTorch Multilabel Mouthfeel Classifier...")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_encoded)
    y_tensor = torch.FloatTensor(y_binary)
    
    # Split into train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_tensor, y_tensor, test_size=0.3, random_state=42, stratify=None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
    )
    
    print(f"  Data splits:")
    print(f"    Training: {X_train.shape[0]} samples")
    print(f"    Validation: {X_val.shape[0]} samples")
    print(f"    Test: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Create data loaders
    batch_size = 128
    train_dataset = TensorDataset(X_train_tensor, y_train)
    val_dataset = TensorDataset(X_val_tensor, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_encoded.shape[1]
    num_mouthfeel_tags = y_binary.shape[1]
    
    model = MouthfeelClassifier(
        input_size=input_size,
        num_mouthfeel_tags=num_mouthfeel_tags,
        hidden_sizes=[512, 256, 128],
        dropout_rate=0.3
    )
    
    print(f"  Model architecture:")
    print(f"    Input size: {input_size}")
    print(f"    Hidden layers: [512, 256, 128]")
    print(f"    Output size: {num_mouthfeel_tags} mouthfeel tags")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"\n  Starting training for {num_epochs} epochs...")
    print(f"  Early stopping patience: {patience}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.numel()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.numel()
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'backend/models/mouthfeel_model_best.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"    Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load('backend/models/mouthfeel_model_best.pth'))
    
    # Final test evaluation
    print(f"\n  Final evaluation on test set:")
    model.eval()
    
    with torch.no_grad():
        output = model(X_test_tensor)
        test_loss = criterion(output, y_test).item()
        
        predicted = (output > 0.5).float()
        test_correct = (predicted == y_test).sum().item()
        test_total = y_test.numel()
        
        # Calculate additional metrics
        y_pred_np = predicted.cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
        # Hamming loss
        hamming = hamming_loss(y_test_np, y_pred_np)
        
        # Per-tag accuracy
        tag_accuracies = []
        for i in range(min(10, y_binary.shape[1])):
            tag_acc = (y_test_np[:, i] == y_pred_np[:, i]).mean()
            tag_accuracies.append(tag_acc)
        
        avg_tag_acc = np.mean(tag_accuracies)
    
    test_acc = test_correct / test_total
    
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  Average Tag Accuracy (top 10): {avg_tag_acc:.4f}")
    
    # Prepare training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }
    
    return model, scaler, (X_test_scaled, y_test, output.cpu().numpy()), history

def save_pytorch_model_and_encoders(model, scaler, encoders):
    """Save the trained model, scaler, and encoders"""
    print("\n💾 Saving trained model and encoders...")
    
    # Create models directory if it doesn't exist
    models_dir = 'backend/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save PyTorch model
    model_path = os.path.join(models_dir, "mouthfeel_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  ✅ PyTorch model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, "mouthfeel_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ Feature scaler saved to {scaler_path}")
    
    # Save multilabel binarizer
    mlb_path = os.path.join(models_dir, "mouthfeel_multilabel_binarizer.pkl")
    joblib.dump(encoders['mlb'], mlb_path)
    print(f"  ✅ Multilabel binarizer saved to {mlb_path}")
    
    # Save ordinal encoder (no separate target encoder needed now)
    ordinal_encoder_path = os.path.join(models_dir, "mouthfeel_encoders.pkl")
    joblib.dump({'ordinal_encoder': encoders['ordinal_encoder']}, ordinal_encoder_path)
    print(f"  ✅ Ordinal encoder saved to {ordinal_encoder_path}")
    
    # Save variety target encoders (large file)
    variety_encoders_path = os.path.join(models_dir, "mouthfeel_variety_target_encoders.pkl")
    joblib.dump(encoders['variety_target_encoders'], variety_encoders_path)
    print(f"  ✅ Variety target encoders saved to {variety_encoders_path}")
    
    print(f"  🎉 All mouthfeel model components saved to {models_dir}/")

def prepare_and_encode_data(df, use_cache=True):
    """Prepare and encode data, using cache if available and valid"""
    
    # Check if we can use cache
    if use_cache and is_cache_valid(df):
        print("🚀 Using cached encoded data...")
        X_encoded, y_binary, encoders, mlb = load_encoded_data()
        return X_encoded, y_binary, encoders, mlb
    
    print("🔄 Cache not found or invalid, encoding from scratch...")
    
    # Prepare multilabel targets
    y_binary, mlb = prepare_mouthfeel_targets(df)
    
    # Encode features
    X, ordinal_features, target_features, numeric_features = encode_features(df)
    
    # Apply encodings
    X_encoded, encoders = encode_training_data(X, y_binary, mlb, ordinal_features, target_features, numeric_features)
    
    # Cache the results
    if use_cache:
        data_hash = get_data_hash(df)
        save_encoded_data(X_encoded, y_binary, encoders, mlb, data_hash)
    
    return X_encoded, y_binary, encoders, mlb

def main():
    """Main training pipeline"""
    print("🍷 Training Multilabel Wine Mouthfeel Classifier (PyTorch)")
    print("=" * 70)
    
    # Add command line options for cache control
    use_cache = True
    clear_cache_first = False
    
    # Simple command line argument parsing
    if len(sys.argv) > 1:
        if '--no-cache' in sys.argv:
            use_cache = False
            print("🚫 Cache disabled via --no-cache flag")
        if '--clear-cache' in sys.argv:
            clear_cache_first = True
            print("🗑️  Will clear cache before training")
    
    try:
        # Clear cache if requested
        if clear_cache_first:
            clear_cache()
        
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Prepare and encode data (with caching)
        X_encoded, y_binary, encoders, mlb = prepare_and_encode_data(df, use_cache=use_cache)
        
        # Train PyTorch neural network
        model, scaler, eval_data, history = train_pytorch_model(X_encoded, y_binary)
        
        # Save everything
        save_pytorch_model_and_encoders(model, scaler, encoders)
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS: PyTorch Multilabel Mouthfeel Classifier Trained!")
        print(f"   - Training data: {len(df)} wines")
        print(f"   - Features: {X_encoded.shape[1]} total")
        print(f"   - Mouthfeel tags: {len(mlb.classes_)} unique")
        print(f"   - Architecture: Deep PyTorch network with batch norm & dropout")
        print(f"   - Models saved to backend/models/")
        if use_cache:
            print(f"   - Encoded data cached to backend/cache/ for faster future runs")
            print(f"   - Use --no-cache to disable caching or --clear-cache to refresh")
        print("=" * 70)
        
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