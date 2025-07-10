import joblib
import numpy as np
import pandas as pd

model = joblib.load("backend/models/price_model.pkl")
ordinal_encoder = joblib.load("backend/models/ordinal_encoder.pkl")
target_encoder = joblib.load("backend/models/target_encoder.pkl")
le = joblib.load("backend/models/label_encoder.pkl")

bucket_size = 15
price_buckets = le.inverse_transform(np.arange(len(le.classes_)))

def predict_price_lite(user_input: dict) -> dict:
    df = pd.DataFrame([user_input])
    
    df = ordinal_encoder.transform(df)
    df = target_encoder.transform(df)
    
    probs = model.predict_proba(df)[0]
    
    top_class = np.argmax(probs)
    top_bucket = price_buckets[top_class]
    top_range = f"${top_bucket}–${top_bucket + bucket_size}"
    
    lower_class = np.argmax(probs[:top_class])
    lower_bucket = price_buckets[lower_class]
    lower_range = f"${lower_bucket}–${lower_bucket + bucket_size}"
    
    # Get top 2 buckets
    top2_indices = np.argsort(probs)[-2:]
    top2_probs = probs[top2_indices]
    top2_buckets = price_buckets[top2_indices]
    
    # Calculate weighted range using only top 2 buckets
    weighted_lower = np.sum(top2_probs * top2_buckets)
    weighted_upper = np.sum(top2_probs * (top2_buckets + bucket_size))
    weighted_range = f"${weighted_lower:.2f}–${weighted_upper:.2f}"
    
    top2 = sorted(np.argsort(probs)[-3:])
    top2_buckets = price_buckets[top2]
    confidence_1off = float(np.sum([probs[i] for i in top2]))
    
    return {
        "top_bucket": top_range,
        "weighted_range": weighted_range,
        "confidence_within_±1": confidence_1off,
        "raw_probs": probs.tolist()  # optional for debugging
    }
