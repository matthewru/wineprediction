import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
from predict_price_lite import predict_price_lite  # adjust path if needed
import sys
import joblib
# Load data
df = pd.read_csv('backend/data/wine_clean.csv')

# Drop rows missing required inputs for price predictor
required_price_fields = ['variety', 'country', 'province', 'age', 'region_hierarchy']
df = df.dropna(subset=required_price_fields)

# Use price_predictor_lite to generate price_min and price_max
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
        weighted_lower = float(price_output['weighted_lower'])
        weighted_upper = float(price_output['weighted_upper'])
        return (weighted_lower, weighted_upper)
    except Exception as e:
        print(f"Error predicting price for row {row.name}: {e}", file=sys.stderr)
        return None, None

# Apply to all rows
def progress_apply(df, func):
    total = len(df)
    results = []
    for i, (idx, row) in enumerate(df.iterrows()):
        result = func(row)
        results.append(result)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} rows", file=sys.stderr)
    return pd.Series(results, index=df.index)

price_bounds = progress_apply(df, get_price_bounds)
df[["price_min", "price_max"]] = pd.DataFrame(price_bounds.tolist(), index=df.index)
df = df.dropna(subset=["price_min", "price_max"])

# Prepare training data
df = df[['variety', 'country', 'province', 'age', 'region_hierarchy', 'points', 'price_min', 'price_max']]
X = df[['variety', 'country', 'province', 'age', 'region_hierarchy', 'price_min', 'price_max']]
y = df['points']

# Encode categorical features
categorical_cols = ['variety', 'country', 'province', 'region_hierarchy']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(X[categorical_cols])

# Combine with numeric features
X_num = X[['age', 'price_min', 'price_max']].values
X_all = np.hstack((X_cat, X_num))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_dist = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'gamma': uniform(0, 5),
    'min_child_weight': randint(1, 10)
}

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    verbose=1,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
print(f"✅ Best hyperparameters: {search.best_params_}")

# Evaluate
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"📊 Tuned RMSE: {rmse:.2f}")
print(f"📊 Tuned MAE: {mae:.2f}")


# Save the trained rating model
joblib.dump(best_model, "backend/models/rating_model.pkl")

# Save the encoder if you need it during prediction
joblib.dump(encoder, "backend/models/rating_encoder.pkl")

