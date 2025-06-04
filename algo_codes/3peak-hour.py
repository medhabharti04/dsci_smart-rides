import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Load the dataset
df = pd.read_csv('bookings.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Handle missing or invalid data
df = df.dropna(subset=['Date'])  # Remove rows with missing 'Date'

# Set the 'Date' as the index
df.set_index('Date', inplace=True)

# Resample data to get hourly booking counts (or choose another time frame like daily, weekly)
df_hourly = df.resample('H').size()

# Split the data into training and test sets (e.g., last 20% as test data)
train_size = int(len(df_hourly) * 0.8)
train_data, test_data = df_hourly[:train_size], df_hourly[train_size:]

# Fit ARIMA model on training data
model = ARIMA(train_data, order=(5,1,0))  # Adjust the order if needed
model_fit = model.fit()

# Make predictions on test data
predictions = model_fit.forecast(steps=len(test_data))

# Plot the actual vs predicted ride demand
plt.figure(figsize=(10,6))
plt.plot(train_data.index, train_data, label='Actual (Train)', color='blue')
plt.plot(test_data.index, test_data, label='Actual (Test)', color='green')
plt.plot(test_data.index, predictions, label='Predicted (ARIMA)', color='red', linestyle='dashed')

plt.title('Ride Demand Forecasting with ARIMA')
plt.xlabel('Time')
plt.ylabel('Ride Demand (Bookings)')
plt.legend(loc='best')
plt.show()
