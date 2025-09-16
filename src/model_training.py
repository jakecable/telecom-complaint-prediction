import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


data = pd.read_csv("unified_model_dataset_corrected.csv")

# Prepare the data for modeling
features = data.drop(
    columns=["zip_code", "stab", "complaint_volume", "total_housing_units"]
)
target = data["complaint_volume"]
features = features.fillna(0)

print("\nFeatures being used for training:")
print(features.columns.tolist())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
print(
    f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples."
)

# Train and Evaluate Linear Regression
print("\nTraining Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_r2 = r2_score(y_test, lr_model.predict(X_test))


# Train and Evaluate Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_r2 = r2_score(y_test, rf_model.predict(X_test))


# Train and Evaluate Support Vector Regressor (SVR)
print("\nTraining SVR model...")
svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_r2 = r2_score(y_test, svr_model.predict(X_test))

# Train and Evaluate XGBoost Regressor
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=100, random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_r2 = r2_score(y_test, xgb_model.predict(X_test))


# R^2 Comparison
print("\n--- Model Comparison (R-squared) ---")
print(f"Linear Regression: {lr_r2:.4f}")
print(f"SVR:               {svr_r2:.4f}")
print(f"Random Forest:     {rf_r2:.4f}")
print(f"XGBoost:           {xgb_r2:.4f}")


# Save All Four Trained Models
joblib.dump(lr_model, "linear_model.joblib")

joblib.dump(
    rf_model, "complaint_model.joblib"
)  # Keeping original name for Random Forest

joblib.dump(svr_model, "svr_model.joblib")

joblib.dump(xgb_model, "xgboost_model.joblib")
