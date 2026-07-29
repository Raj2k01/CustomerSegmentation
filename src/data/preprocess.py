from pathlib import Path
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#load dataset
RAW_DATA = Path("data/raw/Mall_Customers.csv")
df = pd.read_csv(RAW_DATA)

#removing custo ID
df = df.drop(columns=["CustomerID"])

#Encode gender(male -1, female -0)
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

#feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#convert back to DF
processed_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

#saving the processed data
processed_path = Path("data/processed")

processed_path.mkdir(parents=True, exist_ok=True)

processed_df.to_csv(
    processed_path / "customers_processed.csv",
    index=False
)

#saving scaler
models_path = Path("models")

models_path.mkdir(exist_ok=True)

joblib.dump(
    scaler,
    models_path / "scaler.pkl"
)

#saving Encoder
joblib.dump(
    encoder,
    models_path / "gender_encoder.pkl"
)

print("Preprocessing Complete")

print(processed_df.head())