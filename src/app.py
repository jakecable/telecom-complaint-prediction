from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# --- Load all trained models at startup ---
try:
    lr_model = joblib.load("linear_model.joblib")
    rf_model = joblib.load("complaint_model.joblib")
    svr_model = joblib.load("svr_model.joblib")
    xgb_model = joblib.load("xgboost_model.joblib")
    print("All 4 models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Please run model_training.py first.")
    exit()

# Define the feature names that the models expect
EXPECTED_FEATURES = [
    "total_population",
    "median_age",
    "avg_pct_broadband_25_3",
    "avg_pct_broadband_100_20",
]


# --- Define the API endpoint for prediction ---
@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No input data provided"}), 400

    # Default to xgboost if no model is specified
    model_choice = json_data.get("model", "xgboost").lower()
    input_data = json_data.get("features")

    if not input_data:
        return jsonify({"error": 'Missing "features" key in request'}), 400

    try:
        input_features = [input_data.get(feature) for feature in EXPECTED_FEATURES]
        df = pd.DataFrame([input_features], columns=EXPECTED_FEATURES)
    except Exception as e:
        return jsonify({"error": f"Error processing features: {e}"}), 400

    # --- Model Prediction Logic ---
    prediction = None
    model_used = None

    if model_choice == "linear":
        prediction = lr_model.predict(df)[0]
        model_used = "Linear Regression"
    elif model_choice == "random_forest":
        prediction = rf_model.predict(df)[0]
        model_used = "Random Forest"
    elif model_choice == "svr":
        prediction = svr_model.predict(df)[0]
        model_used = "SVR"
    elif model_choice == "xgboost":
        prediction = xgb_model.predict(df)[0]
        # XGBoost prediction might be a numpy type, so we cast it to float
        prediction = float(prediction)
        model_used = "XGBoost"
    else:
        return jsonify(
            {
                "error": 'Invalid model choice. Use "linear", "random_forest", "svr", or "xgboost".'
            }
        ), 400

    return jsonify({"model_used": model_used, "predicted_complaint_volume": prediction})


# --- Run the Flask App ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
