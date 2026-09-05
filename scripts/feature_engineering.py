import pandas as pd
from pathlib import Path


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "aqi_processed_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "AQI_feature_engineered.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading processed AQI data...")

df = pd.read_csv(INPUT_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df = df.sort_values("datetime").reset_index(drop=True)


# =========================================================
# TIME-BASED FEATURES
# =========================================================

df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday


# =========================================================
# LAG FEATURE
# =========================================================

df["AQI_lag_1"] = df["AQI"].shift(1)


# =========================================================
# AQI CHANGE
# =========================================================

df["AQI_change"] = (
    df["AQI"] - df["AQI_lag_1"]
)

# First observation has no previous AQI
df["AQI_change"] = df["AQI_change"].fillna(0)


# =========================================================
# ROLLING AQI
# =========================================================

df["AQI_rolling_avg"] = (
    df["AQI"]
    .shift(1)
    .rolling(window=3, min_periods=1)
    .mean()
)

# Initial rows do not have enough history
df["AQI_rolling_avg"] = (
    df["AQI_rolling_avg"]
    .bfill()
)


# =========================================================
# SAVE FEATURES
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# VALIDATION
# =========================================================

print()
print("==============================================")
print("FEATURE ENGINEERING SUCCESSFUL")
print("==============================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nGenerated features:")
print([
    "hour",
    "day",
    "month",
    "weekday",
    "AQI_lag_1",
    "AQI_change",
    "AQI_rolling_avg"
])

print("\nOutput:")
print(OUTPUT_FILE)

print("\nLatest feature record:")
print(df.tail(1).to_string(index=False))

print("==============================================")