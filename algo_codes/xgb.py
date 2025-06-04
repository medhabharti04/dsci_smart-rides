import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "/home/medha/dsci_proj/cleaned.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_csv(file_path)

# Convert time columns
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Feature selection
features = ['Ride_Distance', 'V_TAT', 'Booking_Value']
target = 'Fare_amount'

# Handle missing values (use assignment instead of inplace)
for col in features:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

df[target] = df[target].fillna(df[target].median())

# Prepare data
X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=features)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model with regularization
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=50,  
    learning_rate=0.05,  
    max_depth=4,  
    subsample=0.8,  
    colsample_bytree=0.8  
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Scatter plot for actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Ideal Fit")
plt.xlabel("Actual Fare Amount")
plt.ylabel("Predicted Fare Amount")
plt.legend()
plt.title("XGBoost Model Predictions vs Actual Values")

# Save and Show Plot
plt.savefig("/home/medha/dsci_proj/final_plot.png")  # Save plot
plt.show()  # Display plot

print("Plot saved as final_plot.png")
