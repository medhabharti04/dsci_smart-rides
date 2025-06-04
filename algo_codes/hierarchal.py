from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load dataset
df = pd.read_csv("/home/medha/dsci_proj/cleaned.csv")

# Create a Route column
df['Route'] = df['Pickup_Location'] + " → " + df['Drop_Location']

# Reduce dataset to most popular 500 routes
top_routes = df['Route'].value_counts().head(500).index  
df = df[df['Route'].isin(top_routes)]

# Convert routes to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Route'])

# Perform hierarchical clustering
cluster_model = AgglomerativeClustering(n_clusters=5, linkage='ward')
df['Cluster'] = cluster_model.fit_predict(X.toarray())

# Show some results
print(df[['Route', 'Cluster']].sample(10))
