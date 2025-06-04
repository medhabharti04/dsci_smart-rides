import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("E:\\DSCI-PROJECT\\bookings.csv")

# Step 2: Clean the data
# Ensure 'Ride_Distance' and 'Fare_amount' are numeric, drop rows with missing values
df_cleaned = df[['Ride_Distance', 'Fare_amount']].dropna()

# Step 3: Prepare the data for Linear Regression
X = df_cleaned[['Ride_Distance']]  # Independent variable (feature)
y = df_cleaned['Fare_amount']      # Dependent variable (target)

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)              # R² score (coefficient of determination)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Step 8: Visualize the regression line with the data
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.title('Linear Regression: Fare vs Ride Distance')
plt.xlabel('Ride Distance')
plt.ylabel('Fare Amount')
plt.legend()
plt.show()
