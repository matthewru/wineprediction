#!/usr/bin/env python3
"""
Train a multilabel classifier to predict wine flavors.

This script will:
1. Load wine_clean_flavors_only.csv as training data
2. Add price_min and price_max columns using predict_price_lite.py
3. Add rating column using predict_rating_lite.py  
4. Apply ordinal encoding to most features, target encoding for variety
5. Train a PyTorch neural network multilabel classifier for flavor tags
6. Save the enhanced dataset and trained model
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

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Add the services directory to path to import prediction functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

# Import the prediction functions
from predict_price_lite import predict_price_lite
from predict_rating_lite import predict_rating_lite

def load_and_prepare_data():
    """Load the wine_clean_flavors_only.csv and prepare initial features"""
    print("📂 Loading wine_clean_flavors_only.csv...")
    df = pd.read_csv('backend/data/wine_clean_flavors_only.csv')
    print(f"Loaded {len(df)} wines with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Display sample data
    print("\n📊 Sample data:")
    for i, row in df.head(3).iterrows():
        print(f"  {row['variety']}: {row['flavor_tags']}")
    
    return df

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

def save_enhanced_dataset(df):
    """Save the dataset with predictions for caching"""
    output_file = 'backend/data/wine_with_predictions.csv'
    print(f"\n💾 Saving enhanced dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} wines with 8 features")

def prepare_flavor_targets(df):
    """Prepare the multilabel targets from flavor_tags"""
    print("\n🏷️  Preparing flavor targets...")
    
    # Parse flavor tags into lists
    flavor_lists = []
    for tags_str in df['flavor_tags']:
        if pd.isna(tags_str) or tags_str.strip() == '':
            flavor_lists.append([])
        else:
            # Split by comma and clean
            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            flavor_lists.append(tags)
    
    # Use MultiLabelBinarizer to create binary matrix
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform(flavor_lists)
    
    print(f"  Found {len(mlb.classes_)} unique flavor tags")
    print(f"  Average tags per wine: {y_binary.sum(axis=1).mean():.1f}")
    print(f"  Binary matrix shape: {y_binary.shape}")
    
    # Show most common flavor tags
    tag_counts = y_binary.sum(axis=0)
    common_tags = [(mlb.classes_[i], tag_counts[i]) for i in range(len(mlb.classes_))]
    common_tags.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 20 most common flavor tags:")
    for i, (tag, count) in enumerate(common_tags[:20]):
        print(f"    {i+1:2d}. {tag:<15} ({count:4d} wines)")
    
    return y_binary, mlb

def encode_features(df):
    """Encode the 8 features for training"""
    print("\n🔧 Encoding features...")
    
    # Define feature sets
    features = ['variety', 'country', 'province', 'age', 'region_hierarchy', 'price_min', 'price_max', 'rating']
    X = df[features].copy()
    
    print(f"  Original features: {list(X.columns)}")
    print(f"  Feature matrix shape: {X.shape}")
    
    # Split features by encoding type
    ordinal_features = ['country', 'province']  # Ordinal encoding
    target_features = ['variety']  # Target encoding with flavor tags
    numeric_features = ['age', 'region_hierarchy', 'price_min', 'price_max', 'rating']
    
    return X, ordinal_features, target_features, numeric_features

def train_variety_target_encoders(X, y_binary, mlb):
    """Train target encoders for variety based on each flavor tag"""
    print("\n🎯 Training variety target encoders...")
    
    variety_target_encoders = {}
    
    for i, flavor_tag in enumerate(mlb.classes_):
        # Create target encoder for this specific flavor tag
        encoder = TargetEncoder(cols=['variety'], smoothing=10)
        
        # Fit encoder on variety -> flavor_tag relationship
        variety_df = X[['variety']].copy()
        target_values = y_binary[:, i]  # Binary target for this flavor tag
        
        encoder.fit(variety_df, target_values)
        variety_target_encoders[flavor_tag] = encoder
        
        if (i + 1) % 50 == 0:
            print(f"  Trained {i + 1}/{len(mlb.classes_)} variety encoders...")
    
    print(f"✅ Trained {len(variety_target_encoders)} variety target encoders")
    return variety_target_encoders

def encode_training_data(X, y_binary, mlb, ordinal_features, target_features, numeric_features):
    """Apply all encodings to create final training matrix"""
    print("\n🔄 Applying feature encodings...")
    
    # 1. Train variety target encoders
    variety_target_encoders = train_variety_target_encoders(X, y_binary, mlb)
    
    # 2. Ordinal encoding for country, province
    ordinal_encoder = OrdinalEncoder(cols=ordinal_features)
    X_ordinal = ordinal_encoder.fit_transform(X[ordinal_features])
    
    # 3. Target encoding for region_hierarchy (using overall flavor density as target)
    # Use average number of flavors per wine as the target
    overall_flavor_density = y_binary.sum(axis=1)  # Number of flavors per wine
    target_encoder = TargetEncoder(cols=['region_hierarchy'], smoothing=13)
    X_region_encoded = target_encoder.fit_transform(X[['region_hierarchy']], overall_flavor_density)
    
    # 4. Variety target encoding - create vector of all flavor tag encodings
    variety_encoded_matrix = np.zeros((len(X), len(mlb.classes_)))
    for i, flavor_tag in enumerate(mlb.classes_):
        encoder = variety_target_encoders[flavor_tag]
        encoded_col = encoder.transform(X[['variety']])
        variety_encoded_matrix[:, i] = encoded_col['variety'].values
    
    # 5. Combine all encoded features
    # Structure: [variety_encoding(n_flavors), country, province, age, region_hierarchy, price_min, price_max, rating]
    X_encoded = np.hstack([
        variety_encoded_matrix,  # Target-encoded variety (n_flavor_tags columns)
        X_ordinal.values,  # Ordinal-encoded country, province
        X[['age']].values,  # Age (numeric)
        X_region_encoded.values,  # Target-encoded region_hierarchy
        X[['price_min', 'price_max', 'rating']].values  # Price and rating (numeric)
    ])
    
    print(f"  Final encoded matrix shape: {X_encoded.shape}")
    print(f"  Features breakdown:")
    print(f"    - Variety target encoding: {len(mlb.classes_)} features")
    print(f"    - Ordinal encoding: {len(ordinal_features)} features")
    print(f"    - Numeric features: {len(numeric_features)} features")
    
    # Save encoders for later use
    encoders = {
        'variety_target_encoders': variety_target_encoders,
        'ordinal_encoder': ordinal_encoder,
        'target_encoder': target_encoder,
        'mlb': mlb
    }
    
    return X_encoded, encoders

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
        self.dropout3 = nn.Dropout(dropout_rate * 0.7)  # Reduce dropout in deeper layers
        
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(dropout_rate * 0.7)
        
        # Output layer
        self.fc_out = nn.Linear(64, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
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

def train_pytorch_model(X_encoded, y_binary):
    """Train the PyTorch neural network multilabel classifier"""
    print("\n🎓 Training PyTorch neural network...")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_binary, test_size=0.2, random_state=42
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples") 
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = FlavorNet(X_train.shape[1], y_binary.shape[1]).to(device)
    print(f"\n🧠 Created FlavorNet:")
    print(f"  Input dimensions: {X_train.shape[1]}")
    print(f"  Output dimensions: {y_binary.shape[1]} (flavor tags)")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for multilabel
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("  🚀 Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy (using 0.5 threshold)
            predicted = (output > 0.5).float()
            train_correct += (predicted == target).sum().item()
            train_total += target.numel()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                predicted = (output > 0.5).float()
                val_correct += (predicted == target).sum().item()
                val_total += target.numel()
        
        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'backend/models/best_flavor_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"    Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('backend/models/best_flavor_model.pth'))
    
    # Final evaluation on test set
    print("\n📊 Final evaluation on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        output = model(X_test_tensor)
        test_loss = criterion(output, y_test_tensor).item()
        
        predicted = (output > 0.5).float()
        test_correct = (predicted == y_test_tensor).sum().item()
        test_total = y_test_tensor.numel()
        
        # Calculate additional metrics
        y_pred_np = predicted.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        
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
    """Save the trained PyTorch model and all encoders"""
    print("\n💾 Saving PyTorch model and encoders...")
    
    # Save the PyTorch model
    torch.save(model.state_dict(), 'backend/models/flavor_model.pth')
    print("  ✅ Saved flavor_model.pth")
    
    # Save model architecture info for loading
    model_info = {
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'model_class': 'FlavorNet'
    }
    joblib.dump(model_info, 'backend/models/flavor_model_info.pkl')
    print("  ✅ Saved flavor_model_info.pkl")
    
    # Save the scaler
    joblib.dump(scaler, 'backend/models/flavor_scaler.pkl')
    print("  ✅ Saved flavor_scaler.pkl")
    
    # Save individual encoders
    joblib.dump(encoders['variety_target_encoders'], 'backend/models/variety_target_encoders.pkl')
    joblib.dump(encoders['mlb'], 'backend/models/flavor_multilabel_binarizer.pkl')
    print("  ✅ Saved variety target encoders and multilabel binarizer")
    
    # Save combined encoder file for the flavor model
    flavor_encoders = {
        'ordinal_encoder': encoders['ordinal_encoder'],
        'target_encoder': encoders['target_encoder'],  # For region_hierarchy
        'mlb': encoders['mlb']
    }
    joblib.dump(flavor_encoders, 'backend/models/flavor_encoders.pkl')
    print("  ✅ Saved flavor_encoders.pkl")

def main():
    """Main training pipeline"""
    print("🍷 Training Multilabel Wine Flavor Classifier (PyTorch)")
    print("=" * 70)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Add predictions from existing models
        df = add_price_predictions(df)
        df = add_rating_predictions(df) 
        
        # Save enhanced dataset for caching
        save_enhanced_dataset(df)
        
        # Prepare multilabel targets
        y_binary, mlb = prepare_flavor_targets(df)
        
        # Encode features
        X, ordinal_features, target_features, numeric_features = encode_features(df)
        
        # Apply encodings
        X_encoded, encoders = encode_training_data(X, y_binary, mlb, ordinal_features, target_features, numeric_features)
        
        # Train PyTorch neural network
        model, scaler, eval_data, history = train_pytorch_model(X_encoded, y_binary)
        
        # Save everything
        save_pytorch_model_and_encoders(model, scaler, encoders)
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS: PyTorch Multilabel Flavor Classifier Trained!")
        print(f"   - Training data: {len(df)} wines")
        print(f"   - Features: {X_encoded.shape[1]} total")
        print(f"   - Flavor tags: {len(mlb.classes_)} unique")
        print(f"   - Architecture: Deep PyTorch network with batch norm & dropout")
        print(f"   - Models saved to backend/models/")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
