import pandas as pd
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "aqi_raw_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "aqi_processed_data.csv"


# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "datetime",
    "AQI",
    "CO",
    "NO",
    "NO2",
    "O3",
    "SO2",
    "PM2_5",
    "PM10",
    "NH3"
]


# =========================================================
# LOAD RAW DATA
# =========================================================

print("Loading raw AQI data...")

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_FILE}"
    )

df = pd.read_csv(RAW_FILE)

print("Raw data shape:", df.shape)


# =========================================================
# COLUMN VALIDATION
# =========================================================

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Keep only required columns
df = df[REQUIRED_COLUMNS].copy()


# =========================================================
# DATETIME CONVERSION
# =========================================================

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

# Remove rows with invalid datetime
df = df.dropna(subset=["datetime"])


# =========================================================
# NUMERIC CONVERSION
# =========================================================

numeric_columns = [
    "AQI",
    "CO",
    "NO",
    "NO2",
    "O3",
    "SO2",
    "PM2_5",
    "PM10",
    "NH3"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

print("Missing values before cleaning:")
print(df.isnull().sum())

# Sort chronologically first
df = df.sort_values("datetime").reset_index(drop=True)

# Time-series friendly interpolation
df[numeric_columns] = (
    df[numeric_columns]
    .interpolate(method="linear")
    .ffill()
    .bfill()
)


# =========================================================
# REMOVE DUPLICATES
# =========================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["datetime"],
    keep="last"
).reset_index(drop=True)

removed_duplicates = before_duplicates - len(df)

print("Duplicate rows removed:", removed_duplicates)


# =========================================================
# VALIDATE NUMERIC DATA
# =========================================================

if df[numeric_columns].isnull().sum().sum() > 0:
    raise ValueError(
        "Missing numeric values remain after preprocessing."
    )


# =========================================================
# VALIDATE AQI RANGE
# =========================================================

# Project AQI is normalized between 1 and 5.
invalid_aqi = ~df["AQI"].between(1, 5)

if invalid_aqi.any():
    print(
        "Warning:",
        invalid_aqi.sum(),
        "rows contain AQI outside the expected 1–5 range."
    )

    df.loc[invalid_aqi, "AQI"] = (
        df.loc[invalid_aqi, "AQI"]
        .clip(1, 5)
    )


# =========================================================
# FINAL SORT
# =========================================================

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# =========================================================
# SAVE PROCESSED DATA
# =========================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    PROCESSED_FILE,
    index=False
)


# =========================================================
# FINAL VALIDATION
# =========================================================

print("\nPreprocessing completed successfully.")
print("Processed data shape:", df.shape)
print("Output file:", PROCESSED_FILE)

print("\nLatest record:")
print(df.tail(1).to_string(index=False))

print("\nMissing values after preprocessing:")
print(df.isnull().sum())