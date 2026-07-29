from pathlib import Path
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

RAW_DATA = Path("data/raw/Mall_Customers.csv")

df = pd.read_csv(RAW_DATA)
df = df.drop(columns=["CustomerID"])

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

processed_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

processed_path = Path("data/processed")

processed_path.mkdir(parents=True, exist_ok=True)

processed_df.to_csv(
    processed_path / "customers_processed.csv",
    index=False
)

models_path = Path("models")

models_path.mkdir(exist_ok=True)

joblib.dump(
    scaler,
    models_path / "scaler.pkl"
)

joblib.dump(
    encoder,
    models_path / "gender_encoder.pkl"
)

print("Preprocessing Complete")

print(processed_df.head())