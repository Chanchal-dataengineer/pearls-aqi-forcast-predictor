import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "processed" / "aqi_processed_data.csv"


def main():
    print("\n" + "=" * 60)
    print("🔎 AQI Data Quality Monitoring")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # Missing values
    missing = df.isnull().sum()

    print("\nMissing values:")
    print(missing)

    # Duplicate timestamps
    duplicate_timestamps = df["datetime"].duplicated().sum()

    print("\nDuplicate timestamps:", duplicate_timestamps)

    # AQI validation
    invalid_aqi = (~df["AQI"].between(1, 5)).sum()

    print("Invalid AQI values:", invalid_aqi)

    # Datetime validation
    is_sorted = df["datetime"].is_monotonic_increasing

    print("Datetime sorted:", is_sorted)

    # Latest record
    latest_timestamp = df["datetime"].max()

    print("Latest timestamp:", latest_timestamp)

    # Validation
    if missing.sum() > 0:
        raise ValueError(
            "Monitoring failed: missing values detected."
        )

    if duplicate_timestamps > 0:
        raise ValueError(
            "Monitoring failed: duplicate timestamps detected."
        )

    if invalid_aqi > 0:
        raise ValueError(
            "Monitoring failed: invalid AQI values detected."
        )

    if not is_sorted:
        raise ValueError(
            "Monitoring failed: datetime is not sorted."
        )

    print("\n" + "=" * 60)
    print("✅ Data quality monitoring passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()