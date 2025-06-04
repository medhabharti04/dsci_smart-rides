from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

# Load dataset
df = pd.read_csv("/home/medha/dsci_proj/cleaned.csv")

# Create a Route column
df['Route'] = df['Pickup_Location'] + " → " + df['Drop_Location']

# Reduce dataset to the 500 most popular routes
top_routes = df['Route'].value_counts().head(500).index  
df = df[df['Route'].isin(top_routes)]

# ✅ Group by Booking_ID and create route baskets
basket = df.groupby('Booking_ID')['Route'].apply(lambda x: list(x)).reset_index()

# ✅ Convert list format to one-hot encoded DataFrame
basket_encoded = basket['Route'].explode().str.get_dummies().groupby(basket['Booking_ID']).max()

# ✅ Print summary
print("Basket shape:", basket_encoded.shape)
print("Top 10 most frequent routes:\n", basket_encoded.sum().sort_values(ascending=False).head(10))

# ✅ Run FP-Growth with lower support threshold
min_support = max(0.0034, 5 / len(basket_encoded))
frequent_routes = fpgrowth(basket_encoded, min_support=min_support, use_colnames=True)

# ✅ Print frequent itemsets
if frequent_routes.empty:
    print("No frequent routes found. Try further lowering min_support.")
else:
    rules = association_rules(frequent_routes, metric="confidence", min_threshold=0.1)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
          .sort_values(by='confidence', ascending=False)
          .head(10))

