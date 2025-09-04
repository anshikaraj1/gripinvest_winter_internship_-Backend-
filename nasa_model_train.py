import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib

# Step 1: Load NASA cleaned data
df = pd.read_csv("nasa_cleaned.csv")

# Step 2: Split features and target
X = df.drop(columns=['RUL'])
y = df['RUL']

# Step 3: Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Step 5: Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Step 7: Save the model
joblib.dump(model, "nasa_model.pkl")
