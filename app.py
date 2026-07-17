import streamlit as st
import joblib
import pandas as pd
model=joblib.load("models/random_forest_model.pkl")

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Pearls AQI Predictor")

st.write("Predict Air Quality Index using Machine Learning")

st.sidebar.header("Weather Inputs")

co = st.sidebar.number_input("CO", value=1.0)

no = st.sidebar.number_input("NO", value=1.0)

no2 = st.sidebar.number_input("NO2", value=1.0)

o3 = st.sidebar.number_input("O3", value=30.0)

so2 = st.sidebar.number_input("SO2", value=5.0)

pm25 = st.sidebar.number_input("PM2_5", value=10.0)

pm10 = st.sidebar.number_input("PM10", value=20.0)

nh3 = st.sidebar.number_input("NH3", value=1.0)


if st.button("Predict AQI"):

    sample = pd.DataFrame({
        "CO": [co],
        "NO": [no],
        "NO2":[no2],
        "O3": [o3],
        "SO2": [so2],
        "PM2_5": [pm25],
        "PM10": [pm10],
        "NH3": [nh3]
    })

    prediction = model.predict(sample)

    st.subheader("Prediction Result")
    st.success(f"Predicted AQI: {prediction[0]:.2f}")

    if prediction[0] <= 50:
        st.success("🟢 Air Quality: Good")

    elif prediction[0] <= 100:
        st.warning("🟡 Air Quality: Moderate")

    else:
        st.error("🔴 Air Quality: Poor")