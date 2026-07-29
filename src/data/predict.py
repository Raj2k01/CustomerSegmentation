import joblib
import pandas as pd

model = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/gender_encoder.pkl")

customer = {
    "Gender": "Male",
    "Age": 30,
    "Annual Income (k$)": 80,
    "Spending Score (1-100)": 85
}

df = pd.DataFrame([customer])

df["Gender"] = encoder.transform(df["Gender"])

scaled_data = scaler.transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

cluster = model.predict(scaled_df)[0]

segment = {
    0: "Budget Customer",
    1: "Regular Customer",
    2: "High Value Customer",
    3: "Premium Customer",
    4: "Luxury Customer"
}

print("Cluster :", cluster)
print("Segment :", segment[cluster])