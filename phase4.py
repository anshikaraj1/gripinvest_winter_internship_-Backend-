import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Step 1: Load your cleaned data
df = pd.read_csv("data.csv")

# Step 2: Visualize the distribution of RUL
sns.histplot(df['RUL'], kde=True)
plt.title("Distribution of Remaining Useful Life (RUL)")
plt.xlabel("RUL")
plt.ylabel("Frequency")
plt.show()

# Step 3: Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Boxplot for all features
df.boxplot(figsize=(15, 6), rot=45)
plt.title("Boxplot of All Features")
plt.tight_layout()
plt.show()

# Step 5: Feature importance using Random Forest
X = df.drop(columns=['RUL'])
y = df['RUL']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

importances = model.feature_importances_
feature_names = X.columns

feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title("Feature Importance using Random Forest")
plt.tight_layout()
plt.show()

# Step 6: Keep only top 6 features + target
top_features = feature_df['Feature'].head(6).tolist()
X_selected = df[top_features]
X_selected['RUL'] = y
X_selected.to_csv("final_data.csv", index=False)
