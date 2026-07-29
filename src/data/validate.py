from pathlib import Path
import pandas as pd


def validate_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load dataset and perform basic validation.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    print("=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)

    print(f"Rows           : {df.shape[0]}")
    print(f"Columns        : {df.shape[1]}")
    print(f"Missing Values : {df.isnull().sum().sum()}")
    print(f"Duplicate Rows : {df.duplicated().sum()}")

    print("\nData Types")
    print(df.dtypes)

    return df


if __name__ == "__main__":
    dataset = Path("data/raw/Mall_Customers.csv")
    validate_dataset(dataset)