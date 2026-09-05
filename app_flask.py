from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/aqi_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():

    return jsonify(
        {
            "status": "running",
            "service": "Pearls AQI Predictor",
            "model": type(model).__name__,
            "feature_count": len(feature_columns),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    required_inputs = [
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

    missing = [feature for feature in required_inputs if feature not in data]

    if missing:

        return (
            jsonify(
                {"error": "Missing required features", "missing_features": missing}
            ),
            400,
        )

    sample = pd.DataFrame([{feature: data[feature] for feature in required_inputs}])

    sample = sample.reindex(columns=feature_columns, fill_value=0)

    sample = sample.astype(float)

    prediction = float(model.predict(sample)[0])

    prediction = float(np.clip(prediction, 1, 5))

    return jsonify(
        {"predicted_aqi": round(prediction, 3), "model": type(model).__name__}
    )


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False)
