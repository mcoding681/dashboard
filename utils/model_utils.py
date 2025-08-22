import joblib
import numpy as np

# Load models once
consumption_model = joblib.load("model/energy_consumption_model.joblib")
cost_model = joblib.load("model/energy_cost_model.joblib")

def predict_energy(province, dwelling, income, floor_area, occupants, region):
    # NOTE: Replace this with your real preprocessing logic
    # Dummy input: only floor_area & occupants
    input_vector = np.array([[floor_area, occupants]])

    predicted_consumption = consumption_model.predict(input_vector)[0]
    predicted_cost = cost_model.predict(input_vector)[0]

    return {
        "consumption": round(predicted_consumption, 2),
        "cost": round(predicted_cost, 2)
    }
