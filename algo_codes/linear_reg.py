import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = "cleaned.csv"  # Ensure this file is in your working directory
df = pd.read_csv(file_path)

# Convert Date and Time columns to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Drop rows with missing values in Date or Time
df = df.dropna(subset=['Date', 'Time'])

# Calculate trip duration in minutes
df['Trip_Duration'] = (df['Time'] - df['Date']).dt.total_seconds() / 60

# Selecting relevant columns
df = df[['Ride_Distance', 'Trip_Duration', 'Booking_Value']].dropna()

# Rename for consistency
df.rename(columns={'Booking_Value': 'Fare'}, inplace=True)

# Define features (X) and target variable (y)
X = df[['Ride_Distance', 'Trip_Duration']]
y = df['Fare']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model evaluation
print("Model Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Plot actual vs predicted fares
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")
plt.show()
