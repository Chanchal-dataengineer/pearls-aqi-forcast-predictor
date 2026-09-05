import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.357
LON = 70.755

BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "aqi_raw_data.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# VALIDATE API KEY
# =========================================================

if not API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY not found. "
        "Please add it to your .env file."
    )


# =========================================================
# FETCH CURRENT AIR POLLUTION DATA
# =========================================================

params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY
}

print("Fetching latest air-quality data from OpenWeather...")

response = requests.get(
    BASE_URL,
    params=params,
    timeout=30
)

response.raise_for_status()

data = response.json()


# =========================================================
# VALIDATE API RESPONSE
# =========================================================

if "list" not in data or not data["list"]:
    raise ValueError("Invalid or empty API response.")

record = data["list"][0]

if "main" not in record or "components" not in record:
    raise ValueError("Required AQI/components data missing.")


# =========================================================
# EXTRACT DATA
# =========================================================

timestamp = record["dt"]

datetime_value = datetime.fromtimestamp(
    timestamp,
    tz=timezone.utc
).replace(tzinfo=None)


aqi = record["main"]["aqi"]

components = record["components"]


new_row = {
    "datetime": datetime_value,
    "AQI": aqi,
    "CO": components.get("co"),
    "NO": components.get("no"),
    "NO2": components.get("no2"),
    "O3": components.get("o3"),
    "SO2": components.get("so2"),
    "PM2_5": components.get("pm2_5"),
    "PM10": components.get("pm10"),
    "NH3": components.get("nh3")
}


new_df = pd.DataFrame([new_row])


# =========================================================
# LOAD EXISTING DATA
# =========================================================

if RAW_FILE.exists():

    existing_df = pd.read_csv(RAW_FILE)

    if not existing_df.empty:

        existing_df["datetime"] = pd.to_datetime(
            existing_df["datetime"],
            errors="coerce"
        )

        combined_df = pd.concat(
            [existing_df, new_df],
            ignore_index=True
        )

    else:
        combined_df = new_df

else:

    combined_df = new_df


# =========================================================
# CLEAN + DEDUPLICATE
# =========================================================

combined_df["datetime"] = pd.to_datetime(
    combined_df["datetime"],
    errors="coerce"
)

combined_df = combined_df.dropna(
    subset=["datetime"]
)

combined_df = combined_df.drop_duplicates(
    subset=["datetime"],
    keep="last"
)

combined_df = combined_df.sort_values(
    "datetime"
).reset_index(drop=True)


# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = [
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

missing_columns = [
    column
    for column in required_columns
    if column not in combined_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =========================================================
# SAVE DATA
# =========================================================

combined_df = combined_df[required_columns]

combined_df.to_csv(
    RAW_FILE,
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n==============================================")
print("AQI DATA COLLECTION SUCCESSFUL")
print("==============================================")

print(f"New timestamp : {datetime_value}")
print(f"AQI           : {aqi}")
print(f"Total rows    : {len(combined_df)}")
print(f"Output file   : {RAW_FILE}")

print("\nLatest record:")
print(combined_df.tail(1).to_string(index=False))

print("\n==============================================")