from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#loading the dataset
DATA_PATH = Path("data/processed/customers_processed.csv")

df = pd.read_csv(DATA_PATH)

print(df.head())

#finding the K(elbow method)

inertia = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(df)

    inertia.append(model.inertia_)


# plotting the elbow curve

plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()

#trainng the model

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans.fit(df)

#every customer get a cluster label

df["Cluster"] = kmeans.labels_

print(df.head())

#evaluation: using silhouette score

score = silhouette_score(df.drop(columns=["Cluster"]), df["Cluster"])

print("Silhouette Score:", round(score,3))

#saving the model

MODEL_PATH = Path("models")

MODEL_PATH.mkdir(exist_ok=True)

joblib.dump(
    kmeans,
    MODEL_PATH / "kmeans.pkl"
)

print("Model Saved Successfully")

#saving clustered data into processed data
df.to_csv(
    "data/processed/customer_clusters.csv",
    index=False
)