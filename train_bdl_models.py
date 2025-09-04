import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("bdl_torpedo_data.csv")

# Drop RUL column
X = df.drop(columns=["RUL"])
y = df["RUL"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🖨️ Print shape to verify
print("🔍 Training feature count:", X_scaled.shape[1])

# Train RF
rf_model = RandomForestRegressor()
rf_model.fit(X_scaled, y)
joblib.dump(rf_model, "bdl_rf_model.pkl")

# Train XGB
xgb_model = XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_scaled, y)
joblib.dump(xgb_model, "bdl_xgb_model.pkl")

print("✅ Trained and saved both BDL models.")


