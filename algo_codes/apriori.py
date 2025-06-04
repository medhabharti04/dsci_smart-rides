from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Load dataset
df = pd.read_csv("/home/medha/dsci_proj/cleaned.csv")

# Create a Route column
df['Route'] = df['Pickup_Location'] + " → " + df['Drop_Location']

# Reduce dataset to the 500 most popular routes
top_routes = df['Route'].value_counts().head(500).index  
df = df[df['Route'].isin(top_routes)]

# ✅ Group routes by Booking_ID
basket = df.groupby('Booking_ID')['Route'].apply(set).reset_index()

# ✅ Convert to one-hot encoding
basket_encoded = basket['Route'].explode().str.get_dummies().groupby(basket['Booking_ID']).max()

# ✅ Ensure bool type (fixes warning)
basket_encoded = basket_encoded.astype(bool)

# ✅ Print summary
print("Basket shape:", basket_encoded.shape)
print("Top 10 most frequent routes:\n", basket_encoded.sum().sort_values(ascending=False).head(10))

# ✅ Run Apriori with adjusted min_support
min_support = max(0.002, 5 / len(basket_encoded))  # Ensures at least 5 occurrences
frequent_routes = apriori(basket_encoded, min_support=min_support, use_colnames=True)

# ✅ Print frequent itemsets
if frequent_routes.empty:
    print("No frequent routes found. Try lowering min_support.")
else:
    rules = association_rules(frequent_routes, metric="confidence", min_threshold=0.1)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
          .sort_values(by='confidence', ascending=False)
          .head(10))

