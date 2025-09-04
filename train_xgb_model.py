import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Load your dataset
df = pd.read_csv("data.csv")  # Change this to bdl_torpedo_data.csv or nasa_cleaned.csv if needed

# Split features and target
X = df.drop(columns=["RUL"])
y = df["RUL"]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train XGBoost model
xgb = XGBRegressor(objective="reg:squarederror")
xgb.fit(X_scaled, y)

# Save the model properly (this creates a .pkl file)
joblib.dump(xgb, "xgb_model.pkl")
