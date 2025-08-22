import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("final_dashboard_energy_dataset.xlsx")

# --- Features and targets ---
X = df[[
    "province", "dwelling_type_mix", "adjusted_avg_income_scaled",
    "floor_area_sqft_scaled", "occupants", "climate_zone", "region_type"
]]

y_use = df["energy_consumption_total_kwh_scaled"]
y_cost = df["energy_cost_scaled"]

# --- Preprocessing: one-hot encode categorical features ---
categorical_features = ["province", "dwelling_type_mix", "climate_zone", "region_type"]
numeric_features = ["adjusted_avg_income_scaled", "floor_area_sqft_scaled", "occupants"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
])

# --- Pipelines for both models ---
use_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

cost_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- Train models ---
use_pipeline.fit(X, y_use)
cost_pipeline.fit(X, y_cost)

# --- Save models ---
joblib.dump(use_pipeline, "model/energy_use_model.pkl")
joblib.dump(cost_pipeline, "model/energy_cost_model.pkl")

print("✅ Models trained and saved successfully.")
