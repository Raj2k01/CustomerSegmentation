import joblib
import pandas as pd
import matplotlib as plt 
from sklearn.metrics import silhouette_score

df = pd.read_csv(
    "data/processed/customers_processed.csv"
)

model = joblib.load(
    "models/kmeans.pkl"
)

labels = model.predict(df)

score = silhouette_score(df, labels)

print("="*40)
print("MODEL EVALUATION")
print("="*40)
print(f"Silhouette Score : {score:.3f}")


#visualize the cluster
raw_df = pd.read_csv("data/raw/Mall_Customers.csv")

raw_df["Cluster"] = labels

#scatter plot
plt.figure(figsize=(8,6))

plt.scatter(
    raw_df["Annual Income (k$)"],
    raw_df["Spending Score (1-100)"],
    c=raw_df["Cluster"]
)

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.title("Customer Segments")

plt.show()

