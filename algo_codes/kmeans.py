import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/home/medha/dsci_proj/cleaned.csv")

# Create 'Route' column
df['Route'] = df['Pickup_Location'] + " → " + df['Drop_Location']

# Get the most frequent routes
route_counts = df['Route'].value_counts().reset_index()
route_counts.columns = ['Route', 'Count']

# Normalize route frequency
route_counts['Normalized_Count'] = (route_counts['Count'] - route_counts['Count'].min()) / \
                                   (route_counts['Count'].max() - route_counts['Count'].min())

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
route_counts['Cluster'] = kmeans.fit_predict(route_counts[['Normalized_Count']])

# Print cluster results
print(route_counts[['Route', 'Cluster']].head(10))

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(route_counts.index, route_counts['Normalized_Count'], c=route_counts['Cluster'], cmap='viridis')
plt.xlabel("Route Index")
plt.ylabel("Normalized Frequency")
plt.title("K-Means Clustering of Routes")
plt.colorbar(label="Cluster")
plt.show()
