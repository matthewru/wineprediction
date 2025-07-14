import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from category_encoders import OrdinalEncoder, TargetEncoder
from xgboost import XGBClassifier
import joblib

# Load and preprocess
df = pd.read_csv('backend/data/wine_clean.csv')
df = df[df['price'] <= 300]  # Optional: filter out extreme outliers

# Create bucketed target
bucket_size = 15
df['price_bucket'] = (df['price'] // bucket_size * bucket_size).astype(int)

# Encode target into class indices
le = LabelEncoder()
df['price_bucket_encoded'] = le.fit_transform(df['price_bucket'])

# Features user might realistically give
features = ['variety', 'country', 'province', 'age', 'region_hierarchy']
df = df.dropna(subset=features + ['price_bucket_encoded'])

X = df[features]
y = df['price_bucket_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Ordinal encode selected categorical fields
categorical = ['variety', 'country', 'province']
ordinal_encoder = OrdinalEncoder(cols=categorical)
X_train_ord = ordinal_encoder.fit_transform(X_train[categorical])
X_test_ord = ordinal_encoder.transform(X_test[categorical])

# Step 2: Concatenate ordinal-encoded with raw numeric + target-encoded field
X_train_combined = pd.concat([X_train_ord.reset_index(drop=True),
                              X_train[['age', 'region_hierarchy']].reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([X_test_ord.reset_index(drop=True),
                             X_test[['age', 'region_hierarchy']].reset_index(drop=True)], axis=1)

# Step 3: Target encode ONLY 'region_hierarchy'
target_encoder = TargetEncoder(cols=["region_hierarchy"], smoothing=13)

# Apply target encoding only to the column, then replace in the combined dataframe
region_train_encoded = target_encoder.fit_transform(
    X_train_combined[["region_hierarchy"]].reset_index(drop=True),
    y_train.reset_index(drop=True)
)
region_test_encoded = target_encoder.transform(
    X_test_combined[["region_hierarchy"]].reset_index(drop=True)
)

# Now replace the original 'region_hierarchy' column with the encoded one
X_train_combined["region_hierarchy"] = region_train_encoded
X_test_combined["region_hierarchy"] = region_test_encoded

# These are the final encoded features
X_train_encoded = X_train_combined
X_test_encoded = X_test_combined

# Train classifier
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_encoded, y_train)

# Predict
y_pred = model.predict(X_test_encoded)

# Convert back to bucketed prices
y_pred_bucket = le.inverse_transform(y_pred)
y_test_bucket = le.inverse_transform(y_test)

# Calculate weighted lower/upper bounds
probs = model.predict_proba(X_test_encoded)
price_buckets = le.inverse_transform(np.arange(len(probs[0])))

success_count = 0
y_test_prices = le.inverse_transform(y_test)
for i, y_true_price in enumerate(y_test_prices):
    lower = np.sum(probs[i] * price_buckets)
    upper = np.sum(probs[i] * (price_buckets + bucket_size))
    if lower <= y_true_price <= upper:
        success_count += 1

weighted_range_acc = success_count / len(y_test)
print(f"Weighted Range Accuracy: {weighted_range_acc:.2f}")

# Evaluation
acc = accuracy_score(y_test, y_pred)
avg_bucket_error = np.abs(y_pred_bucket - y_test_bucket).mean()
print(f"✅ Accuracy: {acc:.2f}")
print(f"✅ Avg Bucket Error: ${avg_bucket_error:.2f}")
# ±1 bucket accuracy
within_1 = np.abs(y_pred_bucket - y_test_bucket) <= bucket_size
acc_within_1 = np.mean(within_1)
print(f"✅ ±1 Bucket Accuracy: {acc_within_1:.2f}")

# Save artifacts
joblib.dump(model, "backend/models/price_model.pkl")
joblib.dump(ordinal_encoder, "backend/models/ordinal_encoder.pkl")
joblib.dump(target_encoder, "backend/models/target_encoder.pkl")
joblib.dump(le, "backend/models/label_encoder.pkl")
