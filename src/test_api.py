import requests

# The URL of our running Flask API
url = "http://127.0.0.1:5000/predict"

# Example data for a new ZIP code
test_features = {
    "total_population": 50000,
    "median_age": 38.5,
    "avg_pct_broadband_25_3": 0.95,
    "avg_pct_broadband_100_20": 0.88,
}

# --- Test 1: Use the XGBoost model (the new default) ---
payload_xgb = {"model": "xgboost", "features": test_features}
response_xgb = requests.post(url, json=payload_xgb)
print("--- XGBoost Prediction ---")
print(response_xgb.json())

# --- Test 2: Use the Random Forest model ---
payload_rf = {"model": "random_forest", "features": test_features}
response_rf = requests.post(url, json=payload_rf)
print("\n--- Random Forest Prediction ---")
print(response_rf.json())

# --- Test 3: Use the SVR model ---
payload_svr = {"model": "svr", "features": test_features}
response_svr = requests.post(url, json=payload_svr)
print("\n--- SVR Prediction ---")
print(response_svr.json())

# --- Test 4: Use the Linear Regression model ---
payload_lr = {"model": "linear", "features": test_features}
response_lr = requests.post(url, json=payload_lr)
print("\n--- Linear Regression Prediction ---")
print(response_lr.json())
