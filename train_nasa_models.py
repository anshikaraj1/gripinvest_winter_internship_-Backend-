import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load NASA dataset
df = pd.read_csv("nasa_cleaned.csv")

# Separate features and target
X = df.drop(columns=["RUL"])
y = df["RUL"]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)
joblib.dump(rf_model, "nasa_rf_model.pkl")

# Train XGBoost
xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model.fit(X_scaled, y)
joblib.dump(xgb_model, "nasa_xgb_model.pkl")

print("✅ NASA models trained and saved successfully.")
