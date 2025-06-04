import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

# Step 1: Load Data
df = pd.read_csv("E:\\DSCI-PROJECT\\bookings.csv")

# Step 2: Clean Data
# Convert 'Date' and 'Time' columns to datetime if they are not already
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.time

# Handle missing 'Date' and 'Time' values (drop rows or impute if necessary)
df_cleaned = df.dropna(subset=['Date', 'Time'])
df_cleaned['Date'] = df_cleaned['Date'].fillna(df_cleaned['Date'].mode()[0])

# Step 3: Scale the data (select numeric features)
# Here, we assume you want to use features like 'Ride_Distance', 'Fare_amount', etc.
numeric_features = ['Ride_Distance', 'Fare_amount', 'Driver_Ratings', 'Customer_Rating']
df_scaled = df_cleaned[numeric_features]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_scaled)

# Step 4: Apply K-Means clustering (directly choose the value of K)
optimal_k = 3  # Set K directly (you can change this number)

# Apply KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 5: Visualize the Clusters
# Use PCA to reduce the dimensionality of the data to 2D for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Plot clusters using the predicted cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_cleaned['Cluster'], cmap='viridis', s=50)
plt.title(f'K-Means Clustering (K={optimal_k}) - PCA Reduced')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Optional: Visualize clusters using Seaborn for better aesthetics
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df_cleaned['Cluster'], palette='viridis')
plt.title(f'Cluster Visualization (K={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
