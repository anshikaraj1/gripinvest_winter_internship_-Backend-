import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the BDL dataset
df = pd.read_csv("bdl_torpedo_data.csv")  # Make sure this CSV is in the same folder

# Drop target column from features
X = df.drop(columns=['RUL'])
y = df['RUL']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Save the trained model
joblib.dump(model, 'bdl_rul_model.pkl')

# Plot
plt.plot(y_test[:50].values, label="Actual")
plt.plot(y_pred[:50], label="Predicted")
plt.legend()
plt.title("BDL Torpedo Predicted vs Actual RUL")
plt.show()
