import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Simulated realistic training data
data = {
    "floor_area": [800, 1200, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    "adjusted_income": [25000, 35000, 45000, 60000, 75000, 90000, 110000, 130000, 150000, 180000],
    "occupants": [1, 2, 3, 4, 4, 5, 5, 6, 6, 7],
    "region_type": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "climate_zone": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
    "dwelling_type": [0, 1, 1, 2, 0, 2, 1, 0, 2, 1],
    "energy_consumption_total": [9000, 12000, 15000, 18000, 21000, 25000, 28000, 32000, 35000, 40000],
    "energy_cost": [900, 1200, 1400, 1800, 2100, 2500, 2800, 3100, 3400, 3800],
}

df = pd.DataFrame(data)

X = df.drop(columns=["energy_consumption_total", "energy_cost"])
y_use = df["energy_consumption_total"]
y_cost = df["energy_cost"]

# Train the models using your local environment
use_model = RandomForestRegressor(n_estimators=200, random_state=42)
cost_model = RandomForestRegressor(n_estimators=200, random_state=42)

use_model.fit(X, y_use)
cost_model.fit(X, y_cost)

# Save them directly into your project's model folder
joblib.dump(use_model, "model/energy_use_model.pkl")
joblib.dump(cost_model, "model/energy_cost_model.pkl")

print("✅ Models trained and saved successfully.")
