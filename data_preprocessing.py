import pandas as pd


DATA_PATH = "data/final_dataset.csv"


def load_and_prepare_data():
    """Load and validate the AeroSense final AQI dataset."""

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "PM2_5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "Temperature",
        "Humidity",
        "Wind_Speed",
        "AQI"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows where AQI is missing
    df = df.dropna(subset=["AQI"])

    print("AeroSense Dataset")
    print("-----------------")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


if __name__ == "__main__":
    data = load_and_prepare_data()
    print("\nDataset loaded successfully.")
    print(data.head())
