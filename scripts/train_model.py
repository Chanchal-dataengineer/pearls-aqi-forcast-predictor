import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "AQI_feature_engineered.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "aqi_model.pkl"
)

FEATURE_COLUMNS_FILE = (
    BASE_DIR
    / "models"
    / "feature_columns.pkl"
)

METADATA_FILE = (
    BASE_DIR
    / "models"
    / "model_metadata.json"
)


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "CO",
    "NO",
    "NO2",
    "O3",
    "SO2",
    "PM2_5",
    "PM10",
    "NH3",
    "hour",
    "day",
    "month",
    "weekday",
    "AQI",
    "AQI_lag_1",
    "AQI_change",
    "AQI_rolling_avg",
]


# =========================================================
# LOAD DATA
# =========================================================

print("Loading feature-engineered dataset...")

df = pd.read_csv(FEATURE_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df = df.sort_values("datetime").reset_index(drop=True)


# =========================================================
# CREATE NEXT-STEP TARGET
# =========================================================

df["AQI_future"] = df["AQI"].shift(-1)


# Remove rows where target/features are unavailable
model_df = df[
    FEATURE_COLUMNS + ["AQI_future"]
].dropna().copy()


X = model_df[FEATURE_COLUMNS]
y = model_df["AQI_future"]


# =========================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# =========================================================

split_index = int(len(model_df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# =========================================================
# MODELS
# =========================================================

models = {

    "Ridge": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(model, X_eval, y_eval):

    predictions = model.predict(X_eval)

    mae = mean_absolute_error(
        y_eval,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_eval,
            predictions
        )
    )

    r2 = r2_score(
        y_eval,
        predictions
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# =========================================================
# TRAIN + EVALUATE
# =========================================================

results = {}

print()
print("Training candidate models...")

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    results[name] = metrics

    print()
    print(name)
    print("MAE :", round(metrics["MAE"], 4))
    print("RMSE:", round(metrics["RMSE"], 4))
    print("R²  :", round(metrics["R2"], 4))


# =========================================================
# PERSISTENCE BASELINE
# =========================================================

baseline_predictions = X_test["AQI"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

results["Persistence_Baseline"] = {
    "MAE": float(baseline_mae),
    "RMSE": float(baseline_rmse),
    "R2": float(baseline_r2)
}


print()
print("Persistence baseline")
print("MAE :", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R²  :", round(baseline_r2, 4))


# =========================================================
# SELECT MODEL
# =========================================================

candidate_names = [
    name
    for name in models
    if name in results
]

best_model_name = min(
    candidate_names,
    key=lambda name: results[name]["RMSE"]
)

best_model = models[
    best_model_name
]


# =========================================================
# REFIT SELECTED MODEL ON ALL AVAILABLE TRAINING DATA
# =========================================================

best_model.fit(
    X,
    y
)


# =========================================================
# SAVE MODEL
# =========================================================

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_FILE
)

joblib.dump(
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_FILE
)


# =========================================================
# METADATA
# =========================================================

metadata = {
    "model_name": best_model_name,
    "model_type": type(best_model).__name__,
    "trained_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "training_rows": int(len(X)),
    "feature_count": len(FEATURE_COLUMNS),
    "target": "AQI_future",
    "evaluation": results,
    "selection_metric": "RMSE",
    "feature_columns": FEATURE_COLUMNS
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# =========================================================
# FINAL OUTPUT
# =========================================================

print()
print("==============================================")
print("MODEL TRAINING SUCCESSFUL")
print("==============================================")

print("Selected model:", best_model_name)
print("Training rows :", len(X))
print("Features      :", len(FEATURE_COLUMNS))

print()
print("Saved model:")
print(MODEL_FILE)

print()
print("Saved metadata:")
print(METADATA_FILE)

print("==============================================")