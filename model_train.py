import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("data.csv")  # Put your CSV file in same folder

# Drop columns
features = df.drop(columns=['RUL'])
target = df['RUL']

# Scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(features)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the trained model to file
joblib.dump(model, 'best_model.pkl')

joblib.dump(scaler, "scaler.pkl")

# Load model from file
loaded_model = joblib.load('best_model.pkl') 

scaler = joblib.load("scaler.pkl") 
# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

plt.plot(y_test[:50].values, label="Actual")
plt.plot(y_pred[:50], label="Predicted")
plt.legend()
plt.title("Predicted vs Actual RUL")
plt.show()

