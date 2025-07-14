import joblib
import numpy as np
import pandas as pd
import os

# Load model and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "../models/price_model.pkl"))
ordinal_encoder = joblib.load(os.path.join(BASE_DIR, "../models/ordinal_encoder.pkl"))
target_encoder = joblib.load(os.path.join(BASE_DIR, "../models/target_encoder.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "../models/label_encoder.pkl"))

# Constants
bucket_size = 15
price_buckets = le.inverse_transform(np.arange(len(le.classes_)))

# Exactly match training pipeline
ORDINAL_COLUMNS = ['variety', 'country', 'province']
TARGET_COLUMNS = ['region_hierarchy']
NUMERIC_COLUMNS = ['age']
ALL_COLUMNS = ORDINAL_COLUMNS + TARGET_COLUMNS + NUMERIC_COLUMNS

def predict_price_lite(user_input: dict) -> dict:
    # Create a one-row DataFrame from input
    df = pd.DataFrame([user_input])

    # Fill in any missing columns
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown" if col in ORDINAL_COLUMNS + TARGET_COLUMNS else 0

    # Reorder columns to match training order
    df = df[ALL_COLUMNS]

    try:
        # Split and encode
        df_ord = df[ORDINAL_COLUMNS]
        df_tar = df[TARGET_COLUMNS]
        df_num = df[NUMERIC_COLUMNS]
        
        # Debug check
        if df_ord.shape[1] != len(ORDINAL_COLUMNS):
            raise ValueError(f"[DEBUG] df_ord shape mismatch: got {df_ord.shape[1]} columns, expected {len(ORDINAL_COLUMNS)}")
        if df_tar.shape[1] != len(TARGET_COLUMNS):
            raise ValueError(f"[DEBUG] df_tar shape mismatch: got {df_tar.shape[1]} columns, expected {len(TARGET_COLUMNS)}")
        if df_num.shape[1] != len(NUMERIC_COLUMNS):
            raise ValueError(f"[DEBUG] df_num shape mismatch: got {df_num.shape[1]} columns, expected {len(NUMERIC_COLUMNS)}")

        # Transform
        df_ord_enc = ordinal_encoder.transform(df_ord)
        df_tar_enc = target_encoder.transform(df_tar)
        df_encoded = np.hstack((df_ord_enc, df_tar_enc, df_num.values))

    except Exception as e:
        raise ValueError(f"Encoder transformation failed: {e}")

    # Predict probability distribution
    probs = model.predict_proba(df_encoded)[0]
    if probs.size == 0:
        raise ValueError("Model returned empty probability vector.")

    # Most likely price bucket
    top_class = np.argmax(probs)
    top_bucket = price_buckets[top_class]
    top_range = f"${top_bucket}–${top_bucket + bucket_size}"

    # Weighted prediction using top 2 buckets
    top2_indices = np.argsort(probs)[-2:]
    top2_probs = probs[top2_indices]
    top2_buckets = price_buckets[top2_indices]
    weighted_lower = np.sum(top2_probs * top2_buckets)
    weighted_upper = np.sum(top2_probs * (top2_buckets + bucket_size))
    weighted_range = f"${weighted_lower:.2f}–${weighted_upper:.2f}"

    # Confidence within ±1 bucket
    top3 = np.argsort(probs)[-3:]
    confidence_1off = float(np.sum(probs[top3]))

    return {
        "top_bucket": top_range,
        "weighted_range": weighted_range,
        "confidence_within_±1": confidence_1off,
        "raw_probs": probs.tolist(),
        "weighted_upper": weighted_upper,
        "weighted_lower": weighted_lower,
    }
