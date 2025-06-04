import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ✅ Load dataset before using df
df = pd.read_csv("/home/medha/dsci_proj/cleaned.csv")

# ✅ Ensure required columns exist
if 'Pickup_Location' in df.columns and 'Drop_Location' in df.columns:
    df['Route'] = df['Pickup_Location'] + " → " + df['Drop_Location']
else:
    print("Error: 'Pickup_Location' or 'Drop_Location' column not found in dataset.")
    exit()

# ✅ Check if latitude & longitude exist
if {'Pickup_Lat', 'Pickup_Lon', 'Drop_Lat', 'Drop_Lon'}.issubset(df.columns):
    coords = df[['Pickup_Lat', 'Pickup_Lon', 'Drop_Lat', 'Drop_Lon']].dropna()

    # Standardize the data
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # Apply DBSCAN
    clustering = DBSCAN(eps=0.3, min_samples=5).fit(coords_scaled)
    df['Cluster'] = clustering.labels_

    print(df[['Pickup_Location', 'Drop_Location', 'Cluster']].sample(10))
else:
    print("Latitude & Longitude not found. Trying frequency-based clustering...")

