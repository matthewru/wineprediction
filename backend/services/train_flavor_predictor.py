import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, hamming_loss, jaccard_score
from predict_price_lite import predict_price_lite
from predict_rating_lite import predict_rating_lite
from encode_flavor_inputs import encode_inputs  # your custom encoder
from torch.nn import BCEWithLogitsLoss
import itertools
import time
import json

import os

def make_json_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

param_grid = {
    'hidden_sizes': [[64], [128], [128, 64], [256, 128], [256, 128, 64]],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'batch_size': [64, 128, 256],
    'activation': ['relu', 'leaky_relu', 'gelu'],
}


# === Step 1: Load dataset (with caching for predictions) ===
if os.path.exists("data/wine_with_predictions.csv"):
    print("Loading cached wine data with predictions...")
    df = pd.read_csv("data/wine_with_predictions.csv")
    print(f"Loaded {len(df)} wine samples from cache")
else:
    print("Loading raw wine data and generating predictions...")
    df = pd.read_csv("data/wine_clean_tagged.csv")
    print(f"Loaded {len(df)} wine samples")
    
    # Clean data - drop rows with missing values in required columns
    print("Cleaning data - removing rows with missing required features...")
    required_columns = ['variety', 'country', 'province', 'age', 'region_hierarchy', 'flavor_tags']
    initial_count = len(df)

    # Check for missing values
    missing_counts = df[required_columns].isnull().sum()
    print("Missing values per column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  - {col}: {count} missing ({count/len(df)*100:.1f}%)")

    # Drop rows with any missing required features
    df = df.dropna(subset=required_columns)
    dropped_count = initial_count - len(df)
    print(f"Dropped {dropped_count} rows with missing data ({dropped_count/initial_count*100:.1f}%)")
    print(f"Remaining samples: {len(df)}")

    df.drop(columns=['points', 'title', 'price', 'description'], inplace=True)
    print("Dropped unnecessary columns")
    
    # === Predict price and rating (only when not cached) ===
    print("Predicting price and rating...")
    df['pred_price_result'] = df.apply(predict_price_lite, axis=1)
    df['pred_rating_result'] = df.apply(predict_rating_lite, axis=1)

        # Extract values
    df['price_min'] = df['pred_price_result'].apply(lambda x: x['weighted_lower'])
    df['price_max'] = df['pred_price_result'].apply(lambda x: x['weighted_upper'])
    df['rating'] = df['pred_rating_result'].apply(lambda x: x['predicted_rating'])

    # Drop raw results
    df.drop(columns=['pred_price_result', 'pred_rating_result'], inplace=True)

    # Save cached version
    df.to_csv("data/wine_with_predictions.csv", index=False)
    print("Cached predictions saved to data/wine_with_predictions.csv")

# === Step 2: Final data cleaning (for cached data too) ===
if 'points' in df.columns or 'title' in df.columns or 'price' in df.columns or 'description' in df.columns:
    df.drop(columns=[col for col in ['points', 'title', 'price', 'description'] if col in df.columns], inplace=True)
    print("Dropped unnecessary columns from cached data")

# Check if we still need to clean cached data
required_columns = ['variety', 'country', 'province', 'age', 'region_hierarchy', 'flavor_tags']
if df[required_columns].isnull().any().any():
    print("Cleaning cached data - removing rows with missing required features...")
    initial_count = len(df)
    df = df.dropna(subset=required_columns)
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} rows with missing data from cached file")

# === Step 3: Preprocess tags and encode inputs (with caching) ===
# Check if we have cached encoded features
encoded_cache_file = "data/encoded_features.npz"
flavor_mlb_file = "models/flavor_mlb.pkl"

if os.path.exists(encoded_cache_file) and os.path.exists(flavor_mlb_file):
    print("Loading cached encoded features and flavor labels...")
    
    # Load cached encodings
    cache_data = np.load(encoded_cache_file)
    X = cache_data['X']
    y = cache_data['y']
    X_mean = cache_data['X_mean']
    X_std = cache_data['X_std']
    
    # Load cached flavor label binarizer
    flavor_mlb = joblib.load(flavor_mlb_file)
    
    print(f"Loaded cached encodings. Feature shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Loaded {y.shape[1]} unique flavor labels from cache")
else:
    print("Processing flavor tags and encoding inputs...")
    
    # Process flavor tags
    print("Preprocessing flavor tags...")
    df['flavor_tags'] = df['flavor_tags'].apply(lambda x: x.split(',') if pd.notna(x) else [])

    flavor_mlb = MultiLabelBinarizer()
    flavor_labels = flavor_mlb.fit_transform(df['flavor_tags'])
    print(f"Created {flavor_labels.shape[1]} unique flavor labels")
    
    # Encode inputs
    print("Encoding inputs...")
    encoded_features = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f"  - Encoded {i}/{len(df)} samples...")
        
    features = encode_inputs(
        grape=row['variety'],
        country=row['country'],
        province=row['province'],
        age=row['age'],
        region_hierarchy=row['region_hierarchy'],
        price_min=row['price_min'],
        price_max=row['price_max'],
        rating=row['rating']
    )
    encoded_features.append(features[0])

    X = np.array(encoded_features, dtype=np.float32)
    y = flavor_labels.astype(np.float32)

    # Normalize input features
    print("Normalizing input features...")
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # Avoid division by zero
    X_std = np.where(X_std == 0, 1, X_std)
    X = (X - X_mean) / X_std

    # Cache everything
    print("Caching encoded features and flavor labels...")
    joblib.dump(flavor_mlb, flavor_mlb_file)  # Save flavor label binarizer
    np.savez(encoded_cache_file, 
             X=X, y=y, X_mean=X_mean, X_std=X_std)
    print(f"Cached encodings saved to {encoded_cache_file}")
    print(f"Cached flavor MultiLabelBinarizer to {flavor_mlb_file}")

# Save normalization parameters for inference (always save these)
np.save('models/X_mean.npy', X_mean)
np.save('models/X_std.npy', X_std)
print("Saved normalization parameters")

print(f"Input encoding completed. Feature shape: {X.shape}, Labels shape: {y.shape}")
print(f"Input range: [{X.min():.3f}, {X.max():.3f}]")
print(f"Labels range: [{y.min():.3f}, {y.max():.3f}]")

# === Step 4: Hyperparameter Search ===
class WineFlavorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FlavorTagger(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[64], dropout=0.2, activation='relu'):
        super().__init__()
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn)
            layers.append(nn.BatchNorm1d(hidden_size))
            if i < len(hidden_sizes) - 1:  # Don't add dropout before output
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (no activation - BCEWithLogitsLoss handles it)
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# === Evaluation Function ===
def evaluate_model(model, data_loader, criterion, threshold=0.5):
    """Evaluate model performance on given dataset"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            preds = model(batch_X)
            preds_clamped = torch.clamp(preds, min=1e-7, max=1-1e-7)
            loss = criterion(preds_clamped, batch_y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(preds)
            binary_preds = (probs > threshold).float()

            # Convert to binary predictions using threshold
            all_preds.append(binary_preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    
    # Precision, Recall, F1 (macro and micro averaged)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro', zero_division=0
    )
    
    # Hamming loss (fraction of incorrect labels)
    hamming = hamming_loss(all_targets, all_preds)
    
    # Jaccard score (intersection over union)
    jaccard = jaccard_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Exact match ratio (all labels correct for a sample)
    exact_match = np.mean(np.all(all_preds == all_targets, axis=1))
    
    avg_preds_per_sample = all_preds.sum(axis=1).mean()
    
    return {
        'loss': avg_loss,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'exact_match_ratio': exact_match,
        'avg_preds_per_sample': avg_preds_per_sample
    }

def create_data_loaders(X, y, batch_size, train_split=0.8):
    """Create train/val data loaders with specified batch size"""
    dataset = WineFlavorDataset(X, y)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use same random seed for reproducible splits
    torch.manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, train_size, val_size

def train_model_with_params(X, y, params, max_epochs=20, patience=5, verbose=False):
    """Train a model with given hyperparameters and return best validation performance"""
    
    # Create data loaders
    train_loader, val_loader, train_size, val_size = create_data_loaders(
        X, y, params['batch_size']
    )
    
    # Create model
    model = FlavorTagger(
        input_dim=X.shape[1], 
        output_dim=y.shape[1],
        hidden_sizes=params['hidden_sizes'],
        dropout=params['dropout'],
        activation=params['activation']
    )
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Compute class weights
    label_counts = y.sum(axis=0)
    pos_weight = (len(y) - label_counts) / (label_counts + 1e-7)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training loop with early stopping
    best_val_f1 = 0
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, threshold=0.8)
        
        if verbose:
            print(f"  Epoch {epoch+1}: Val F1_micro: {val_metrics['f1_micro']:.4f}, "
                  f"Val F1_macro: {val_metrics['f1_macro']:.4f}")
        
        # Early stopping check
        if val_metrics['f1_micro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_micro']
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and get final metrics
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test different thresholds to find best
    best_threshold = 0.5
    best_threshold_f1 = 0
    
    for thresh in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]:
        val_metrics = evaluate_model(model, val_loader, criterion, threshold=thresh)
        if val_metrics['f1_micro'] > best_threshold_f1:
            best_threshold_f1 = val_metrics['f1_micro']
            best_threshold = thresh
    
    final_metrics = evaluate_model(model, val_loader, criterion, threshold=best_threshold)
    
    return {
        'f1_micro': final_metrics['f1_micro'],
        'f1_macro': final_metrics['f1_macro'],
        'precision_micro': final_metrics['precision_micro'],
        'recall_micro': final_metrics['recall_micro'],
        'hamming_loss': final_metrics['hamming_loss'],
        'exact_match_ratio': final_metrics['exact_match_ratio'],
        'best_threshold': best_threshold,
        'epochs_trained': epoch + 1,
        'avg_preds_per_sample': final_metrics['avg_preds_per_sample']
    }

def hyperparameter_search(X, y, param_grid, max_trials=20):
    """Perform random search over hyperparameter grid"""
    print("Starting hyperparameter search...")
    print(f"Will test up to {max_trials} parameter combinations")
    
    # Generate all possible combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    
    # Randomly sample combinations if too many
    if len(all_combinations) > max_trials:
        import random
        random.seed(42)
        combinations_to_test = random.sample(all_combinations, max_trials)
    else:
        combinations_to_test = all_combinations
    
    print(f"Testing {len(combinations_to_test)} combinations...")
    
    best_score = 0
    best_params = None
    best_metrics = None
    all_results = []
    
    for i, param_combo in enumerate(combinations_to_test):
        params = dict(zip(param_names, param_combo))
        
        print(f"\nTrial {i+1}/{len(combinations_to_test)}")
        print(f"  Params: {params}")
        
        start_time = time.time()
        try:
            metrics = train_model_with_params(X, y, params, max_epochs=15, patience=3)
            elapsed = time.time() - start_time
            
            print(f"  Results: F1_micro={metrics['f1_micro']:.4f}, "
                  f"F1_macro={metrics['f1_macro']:.4f}, "
                  f"Avg_preds={metrics['avg_preds_per_sample']:.2f}, "
                  f"Time={elapsed:.1f}s")
            
            # Store results
            result = params.copy()
            result.update(metrics)
            result['trial'] = i + 1
            result['training_time'] = elapsed
            all_results.append(result)
            
            # Check if this is the best
            if metrics['f1_micro'] > best_score:
                best_score = metrics['f1_micro']
                best_params = params
                best_metrics = metrics
                print(f"  ★ New best F1_micro: {best_score:.4f}")
            
        except Exception as e:
            print(f"  ❌ Trial failed: {e}")
            continue
    
    return best_params, best_metrics, all_results

print(f"Dataset ready for hyperparameter tuning. Feature shape: {X.shape}, Labels shape: {y.shape}")

# Perform hyperparameter search
best_params, best_metrics, all_results = hyperparameter_search(X, y, param_grid, max_trials=15)

print("\n" + "="*60)
print("HYPERPARAMETER SEARCH RESULTS")
print("="*60)
print(f"Best parameters found:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

print(f"\nBest performance:")
for key, value in best_metrics.items():
    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

# Save hyperparameter search results
with open('models/hyperparameter_search.json', 'w') as f:
    json.dump(make_json_serializable({
        'best_params': best_params,
        'best_metrics': best_metrics,
        'all_results': all_results
    }), f, indent=2)
print("\nHyperparameter search results saved to models/hyperparameter_search.json")

# Now train final model with best parameters
print("\n" + "="*60)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("="*60)

# Create final model with best parameters
model = FlavorTagger(
    input_dim=X.shape[1], 
    output_dim=y.shape[1],
    hidden_sizes=best_params['hidden_sizes'],
    dropout=best_params['dropout'],
    activation=best_params['activation']
)

optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

# Create data loaders with best batch size
train_loader, val_loader, train_size, val_size = create_data_loaders(
    X, y, best_params['batch_size']
)

print(f"Model initialized with best parameters:")
print(f"  Architecture: {best_params['hidden_sizes']}")
print(f"  Dropout: {best_params['dropout']}")
print(f"  Learning rate: {best_params['learning_rate']}")
print(f"  Batch size: {best_params['batch_size']}")
print(f"  Activation: {best_params['activation']}")
print(f"  Train/Val split: {train_size}/{val_size} samples")

# === Step 6: Training Loop ===
print("Starting training...")
best_val_f1 = 0
training_history = []
epochs_without_improvement = 0
early_stopping_patience = 5
max_epochs = 50

# Track best threshold
best_threshold = 0.5
best_threshold_f1 = 0
threshold_history = []

# Compute label imbalance
label_counts = y.sum(axis=0)
pos_weight = (len(y) - label_counts) / (label_counts + 1e-7)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        
        # Debug: Check prediction ranges
        if batch_count == 0 and epoch == 0:
            print(f"First batch - Predictions range: [{preds.min():.6f}, {preds.max():.6f}]")
            print(f"First batch - Targets range: [{batch_y.min():.6f}, {batch_y.max():.6f}]")
        
        # Clamp predictions to valid range for BCE loss
        preds = torch.clamp(preds, min=1e-7, max=1-1e-7)
        
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    # Calculate training metrics
    train_loss = total_loss / batch_count
    
    # Evaluate on validation set
    val_metrics = evaluate_model(model, val_loader, criterion, threshold=best_threshold if best_threshold_f1 > 0 else 0.5)
    
    print("\n🔎 Threshold Sweep on Validation Set:")
    epoch_best_threshold = 0.5
    epoch_best_f1 = 0
    
    for thresh in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]:
        val_metrics_thresh = evaluate_model(model, val_loader, criterion, threshold=thresh)
        f1_micro = val_metrics_thresh['f1_micro']
        print(f"  Threshold {thresh:.2f} → F1_micro: {f1_micro:.4f}, "
            f"Recall_micro: {val_metrics_thresh['recall_micro']:.4f}, "
            f"Precision_micro: {val_metrics_thresh['precision_micro']:.4f}, "
            f"Hamming: {val_metrics_thresh['hamming_loss']:.4f}")
        
        # Track best threshold for this epoch
        if f1_micro > epoch_best_f1:
            epoch_best_f1 = f1_micro
            epoch_best_threshold = thresh
    
    print(f"  → Best threshold this epoch: {epoch_best_threshold:.2f} (F1: {epoch_best_f1:.4f})")
    
    # Update global best threshold
    if epoch_best_f1 > best_threshold_f1:
        best_threshold_f1 = epoch_best_f1
        best_threshold = epoch_best_threshold
        print(f"  ★ New overall best threshold: {best_threshold:.2f} (F1: {best_threshold_f1:.4f})")
    
    # Store threshold history
    threshold_history.append({
        'epoch': epoch + 1,
        'best_threshold': epoch_best_threshold,
        'best_f1_micro': epoch_best_f1
    })
    
    # Store history
    epoch_history = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'val_f1_macro': val_metrics['f1_macro'],
        'val_f1_micro': val_metrics['f1_micro'],
        'val_exact_match': val_metrics['exact_match_ratio']
    }
    training_history.append(epoch_history)
    
    # Print epoch results
    print(f"Epoch {epoch+1}/{max_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}, F1 (micro): {val_metrics['f1_micro']:.4f}")
    print(f"  Val Exact Match: {val_metrics['exact_match_ratio']:.4f}")
    
    # Early stopping logic
    if val_metrics['f1_macro'] > best_val_f1:
        best_val_f1 = val_metrics['f1_macro']
        epochs_without_improvement = 0
        print(f"  * New best F1 score: {best_val_f1:.4f}")
        
        # Save best model
        torch.save(model.state_dict(), "models/best_flavor_model.pth")
        print(f"  * Saved best model to models/best_flavor_model.pth")
    else:
        epochs_without_improvement += 1
        print(f"  - No improvement ({epochs_without_improvement}/{early_stopping_patience})")
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best validation F1 (macro): {best_val_f1:.4f}")
            break

print(f"\nTraining completed after {epoch+1} epochs!")
print(f"Best validation F1 (macro): {best_val_f1:.4f}")
print(f"Best threshold for inference: {best_threshold:.2f} (F1_micro: {best_threshold_f1:.4f})")

# Load best model for final evaluation
print("Loading best model for final evaluation...")
model.load_state_dict(torch.load("models/best_flavor_model.pth"))
model.eval()

# === Step 7: Final Evaluation ===
print("\n" + "="*50)
print("FINAL MODEL EVALUATION")
print("="*50)

# Evaluate on both train and validation sets
print("\nTraining Set Performance:")
train_final_metrics = evaluate_model(model, train_loader, criterion, threshold=best_threshold)
for metric, value in train_final_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nValidation Set Performance:")
val_final_metrics = evaluate_model(model, val_loader, criterion, threshold=best_threshold)
for metric, value in val_final_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Analyze label distribution and performance
print(f"\nDataset Statistics:")
print(f"  Total samples: {len(dataset)}")
print(f"  Training samples: {len(train_ds)}")
print(f"  Validation samples: {len(val_ds)}")
print(f"  Number of unique flavor labels: {y.shape[1]}")
print(f"  Average labels per sample: {y.mean(axis=0).sum():.2f}")

# Sample predictions analysis
print(f"\nSample Predictions Analysis:")
model.eval()
with torch.no_grad():
    sample_batch_X, sample_batch_y = next(iter(val_loader))
    sample_preds = model(sample_batch_X[:5])  # First 5 samples
    sample_binary = (sample_preds > 0.9).float()
    
    for i in range(5):
        true_labels = sample_batch_y[i].sum().item()
        pred_labels = sample_binary[i].sum().item()
        print(f"  Sample {i+1}: True labels: {true_labels:.0f}, Predicted labels: {pred_labels:.0f}")

# === Step 8: Save model ===
print("Saving model...")
torch.save(model.state_dict(), "models/flavor_model.pth")
print("Model saved to models/flavor_model.pth")

# Save training history
import json
with open('models/training_history.json', 'w') as f:
    json.dump(make_json_serializable(training_history), f, indent=2)
print("Training history saved to models/training_history.json")

# Save threshold optimization results
threshold_results = {
    'best_threshold': best_threshold,
    'best_threshold_f1_micro': best_threshold_f1,
    'threshold_history': threshold_history
}
with open('models/threshold_optimization.json', 'w') as f:
    json.dump(make_json_serializable(threshold_results), f, indent=2)
print(f"Threshold optimization results saved to models/threshold_optimization.json")

# Save best threshold as a simple text file for easy reading
with open('models/best_threshold.txt', 'w') as f:
    f.write(f"{best_threshold:.2f}")
print(f"Best threshold ({best_threshold:.2f}) saved to models/best_threshold.txt")

print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Best validation F1 score achieved: {best_val_f1:.4f}")
print(f"Final validation exact match ratio: {val_final_metrics['exact_match_ratio']:.4f}")
print("Model and evaluation metrics saved to models/ directory")
