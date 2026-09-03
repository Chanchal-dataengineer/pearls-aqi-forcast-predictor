from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and feature columns
model = joblib.load("models/aqi_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


@app.route("/")
def home():
    return jsonify({
        "message": "Pearls AQI Predictor API is Running!",
        "status": "success"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    sample = pd.DataFrame({
        "CO": [data["CO"]],
        "NO": [data["NO"]],
        "NO2": [data["NO2"]],
        "O3": [data["O3"]],
        "SO2": [data["SO2"]],
        "PM2_5": [data["PM2_5"]],
        "PM10": [data["PM10"]],
        "NH3": [data["NH3"]],
        "hour": [data["hour"]],
        "day": [data["day"]],
        "weekday": [data["weekday"]]
    })

    # Encode weekday exactly as during training
    sample = pd.get_dummies(
        sample,
        columns=["weekday"],
        drop_first=True
    )

    # Match training feature columns
    sample = sample.reindex(
        columns=feature_columns,
        fill_value=0
    )

    sample = sample.astype(float)

    prediction = model.predict(sample)

    return jsonify({
        "Predicted AQI": float(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)