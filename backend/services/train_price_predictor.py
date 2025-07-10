import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from category_encoders import TargetEncoder, OrdinalEncoder
from xgboost import XGBRegressor
import joblib

def mape(y_true, y_pred):
    y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_scorer = make_scorer(mape, greater_is_better=False)

df = pd.read_csv('backend/data/wine_clean.csv')


# Transform the price to a log scale because more price values are concentrated at the lower end of the scale
df['log_price'] = np.log1p(df['price'])
features = ['variety', 'country', 'province', 'region_hierarchy', 'age', 'points', 'winery']
target = 'log_price'
df = df.dropna(subset=features + [target])

x = df[features]
y = df[target]


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Encoding the values to numeric values for training and testing

# Encode categorical features
categorical = ['variety', 'country', 'province']
ordinal_encoder = OrdinalEncoder(cols=categorical)
x_train_encoded = ordinal_encoder.fit_transform(x_train)
x_test_encoded = ordinal_encoder.transform(x_test)

target_encoder = TargetEncoder(cols=["region_hierarchy", "winery"], smoothing=13)

x_train_encoded = target_encoder.fit_transform(x_train_encoded, y_train)
x_test_encoded = target_encoder.transform(x_test_encoded)

# Start training --- Use the XGB Regressor model for continuous values

model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(x_train_encoded, y_train)

# Grid search to find the best parameters
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


grid_search = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring=mape_scorer,
    verbose=1,
    n_jobs=-1  # Use all cores
)

grid_search.fit(x_train_encoded, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best MAPE (log scale):", -grid_search.best_score_)  # Flip sign back

best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test_encoded)

# Convert predictions back to original scale for evaluation
y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mae = mean_absolute_error(y_test_original, y_pred_original)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
mask = y_test_original > 10
filtered_mape = np.mean(np.abs((y_pred_original[mask] - y_test_original[mask]) / y_test_original[mask])) * 100
print(f"Filtered MAPE: {filtered_mape:.2f}%")
mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
print(f"MAPE: {mape:.2f}%")






