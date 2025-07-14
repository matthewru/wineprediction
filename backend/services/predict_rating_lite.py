import os
import joblib
import numpy as np
import pandas as pd

# Load model and encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "../models/rating_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "../models/rating_encoder.pkl"))

# Define expected input columns
CATEGORICAL_COLUMNS = ['variety', 'country', 'province', 'region_hierarchy']
NUMERIC_COLUMNS = ['age', 'price_min', 'price_max']
ALL_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

def predict_rating_lite(user_input: dict) -> dict:
    # Convert input to DataFrame
    df = pd.DataFrame([user_input])

    # Ensure all required fields exist
    for col in ALL_COLUMNS:
        if col not in df:
            df[col] = "Unknown" if col in CATEGORICAL_COLUMNS else 0

    try:
        df_cat = encoder.transform(df[CATEGORICAL_COLUMNS])
        df_num = df[NUMERIC_COLUMNS].values
        X = np.hstack((df_cat, df_num))
        rating = model.predict(X)[0]
        return {
            "predicted_rating": float(rating)
        }
    except Exception as e:
        raise ValueError(f"Rating prediction failed: {e}")
