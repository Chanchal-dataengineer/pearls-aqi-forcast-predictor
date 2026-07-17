from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

@app.route("/")
def home():
    return "Pearls AQI Predictor API is Running!"

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
        "NH3": [data["NH3"]]
    })

    prediction = model.predict(sample)

    return jsonify({
        "Predicted AQI": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)