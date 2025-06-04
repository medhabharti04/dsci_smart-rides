# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import networkx as nx

# Sample data (replace this with your actual dataset)
data = {
    'Booking_ID': [1, 2, 3, 4, 5],
    'Pickup_Location': ['Location A', 'Location B', 'Location A', 'Location C', 'Location A'],
    'Drop_Location': ['Location B', 'Location C', 'Location A', 'Location A', 'Location C']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# One-hot encode the pickup location only (you can also try with drop location)
encoder = OneHotEncoder(sparse_output=False)
pickup_data = df['Pickup_Location'].values.reshape(-1, 1)
encoded_data = pd.DataFrame(encoder.fit_transform(pickup_data), columns=encoder.categories_[0])

# Create a DataFrame with encoded locations
df_encoded = pd.concat([df, encoded_data], axis=1)

# Convert the encoded data to boolean (True/False) values
df_encoded[encoder.categories_[0]] = df_encoded[encoder.categories_[0]].astype(bool)

# Show the encoded DataFrame
print("Encoded Dataframe:")
print(df_encoded)

# Apply Apriori algorithm to find frequent itemsets (lower min_support=0.05 for more frequent itemsets)
frequent_itemsets = apriori(df_encoded[encoder.categories_[0]], min_support=0.05, use_colnames=True)

# Show the frequent itemsets
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules with a lower confidence threshold (min_threshold=0.3)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# Show the association rules
print("\nAssociation Rules:")
print(rules)

# If you'd like, visualize the results of association rules (optional)
if not rules.empty:
    # Create a graph for visualizing the association rules
    G = nx.DiGraph()

    for index, row in rules.iterrows():
        G.add_edge(str(row['antecedents']), str(row['consequents']), weight=row['confidence'])

    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title('Association Rules Visualization')
    plt.show()
else:
    print("No association rules were found.")
