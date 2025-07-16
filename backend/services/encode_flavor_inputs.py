import pandas as pd
import numpy as np
import joblib

# Load encoders
ordinal_encoder = joblib.load("models/ordinal_encoder.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

def encode_inputs(grape, country, province, age, region_hierarchy, price_min, price_max, rating):
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
