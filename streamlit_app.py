import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌍", layout="wide")


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("models/aqi_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


# =========================================================
# HEADER
# =========================================================

st.title("🌍 Pearls AQI Predictor")

st.markdown("""
    ### AI-Powered Air Quality Prediction System

    Enter the pollutant and time information below to predict
    the Air Quality Index using the trained Machine Learning model.
    """)

st.divider()


# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("🌫️ Pollutant Inputs")

co = st.sidebar.number_input("CO", min_value=0.0, value=71.0, step=0.1)

no = st.sidebar.number_input("NO", min_value=0.0, value=0.05, step=0.01)

no2 = st.sidebar.number_input("NO2", min_value=0.0, value=1.0, step=0.01)

o3 = st.sidebar.number_input("O3", min_value=0.0, value=40.0, step=0.1)

so2 = st.sidebar.number_input("SO2", min_value=0.0, value=1.0, step=0.01)

pm25 = st.sidebar.number_input("PM2.5", min_value=0.0, value=11.0, step=0.1)

pm10 = st.sidebar.number_input("PM10", min_value=0.0, value=50.0, step=0.1)

nh3 = st.sidebar.number_input("NH3", min_value=0.0, value=0.23, step=0.01)


# =========================================================
# TIME INPUTS
# =========================================================

st.sidebar.header("🕒 Time Information")

selected_datetime = st.sidebar.datetime_input(
    "Prediction Date & Time", value=datetime.now()
)

hour = selected_datetime.hour
day = selected_datetime.day
weekday = selected_datetime.strftime("%A")

st.sidebar.info(f"Selected Day: {weekday}\n\n" f"Hour: {hour}")


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("ℹ️ Model Information"):

    st.write("**Model Type:**", type(model).__name__)

    st.write("**Expected Features:**")

    st.write(feature_columns)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict AQI", use_container_width=True):

    # -----------------------------------------------------
    # Create input dataframe
    # -----------------------------------------------------

    sample = pd.DataFrame(
        {
            "CO": [co],
            "NO": [no],
            "NO2": [no2],
            "O3": [o3],
            "SO2": [so2],
            "PM2_5": [pm25],
            "PM10": [pm10],
            "NH3": [nh3],
            "hour": [hour],
            "day": [day],
            "weekday": [weekday],
        }
    )

    # -----------------------------------------------------
    # Encode weekday exactly like training
    # -----------------------------------------------------

    sample = pd.get_dummies(sample, columns=["weekday"], drop_first=True)

    # -----------------------------------------------------
    # Match training feature columns
    # -----------------------------------------------------

    sample = sample.reindex(columns=feature_columns, fill_value=0)

    # Make sure all values are numeric
    sample = sample.astype(float)

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(sample)

    predicted_aqi = float(prediction[0])

    # =====================================================
    # RESULT
    # =====================================================

    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted AQI", f"{predicted_aqi:.2f}")

    with col2:
        st.metric("Prediction Date", selected_datetime.strftime("%d %b %Y"))

    with col3:
        st.metric("Prediction Time", selected_datetime.strftime("%H:%M"))

    # =====================================================
    # AQI INTERPRETATION
    # =====================================================

    st.subheader("🌱 Air Quality Interpretation")

    if predicted_aqi <= 1:
        st.success("🟢 AQI Level: Good")

    elif predicted_aqi <= 2:
        st.info("🔵 AQI Level: Fair")

    elif predicted_aqi <= 3:
        st.warning("🟡 AQI Level: Moderate")

    else:
        st.error("🔴 AQI Level: Poor")

    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    with st.expander("🔍 View Input Data"):

        st.dataframe(sample, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("Pearls AQI Predictor | End-to-End Machine Learning Project")
