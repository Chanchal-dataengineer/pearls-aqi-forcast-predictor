import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from aqi_utils import get_aqi_category

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

    Predict the next 3 days of Air Quality Index using
    Machine Learning and pollutant information.
    """)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🌫️ Current Air Quality Inputs")

current_aqi = st.sidebar.number_input(
    "Current AQI", min_value=1.0, max_value=5.0, value=1.0, step=0.1
)

co = st.sidebar.number_input("CO", min_value=0.0, value=71.0, step=0.1)

no = st.sidebar.number_input("NO", min_value=0.0, value=0.05, step=0.01)

no2 = st.sidebar.number_input("NO2", min_value=0.0, value=1.0, step=0.01)

o3 = st.sidebar.number_input("O3", min_value=0.0, value=40.0, step=0.1)

so2 = st.sidebar.number_input("SO2", min_value=0.0, value=1.0, step=0.01)

pm25 = st.sidebar.number_input("PM2.5", min_value=0.0, value=11.0, step=0.1)

pm10 = st.sidebar.number_input("PM10", min_value=0.0, value=50.0, step=0.1)

nh3 = st.sidebar.number_input("NH3", min_value=0.0, value=0.23, step=0.01)


# =========================================================
# DATE/TIME
# =========================================================

st.sidebar.header("🕒 Forecast Information")

start_datetime = st.sidebar.datetime_input("Forecast Start", value=datetime.now())


# =========================================================
# CURRENT AQI STATUS
# =========================================================

current_status = get_aqi_category(current_aqi)

st.subheader("📍 Current Air Quality")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current AQI", f"{current_aqi:.2f}")

with col2:
    st.metric("Status", f"{current_status['level']} {current_status['category']}")

with col3:
    st.info(current_status["message"])


# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("ℹ️ Model Information"):

    st.write("**Model:**", type(model).__name__)

    st.write("**Forecast Horizon:**", "3 Days")

    st.write("**Feature Count:**", len(feature_columns))

    st.write("**Features:**")

    st.write(feature_columns)


# =========================================================
# FORECAST FUNCTION
# =========================================================


def create_feature_row(
    current_aqi_value, lag_aqi, previous_change, rolling_average, forecast_datetime
):

    row = {
        "CO": co,
        "NO": no,
        "NO2": no2,
        "O3": o3,
        "SO2": so2,
        "PM2_5": pm25,
        "PM10": pm10,
        "NH3": nh3,
        "hour": forecast_datetime.hour,
        "day": forecast_datetime.day,
        "month": forecast_datetime.month,
        "weekday": forecast_datetime.weekday(),
        "AQI": current_aqi_value,
        "AQI_lag_1": lag_aqi,
        "AQI_change": previous_change,
        "AQI_rolling_avg": rolling_average,
    }

    return pd.DataFrame([row])


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict Next 3 Days", use_container_width=True):

    predictions = []

    previous_aqi = float(current_aqi)

    lag_aqi = float(current_aqi)

    previous_change = 0.0

    rolling_values = [float(current_aqi)]

    for day_number in range(1, 4):

        forecast_datetime = start_datetime + timedelta(days=day_number)

        rolling_average = float(np.mean(rolling_values[-3:]))

        sample = create_feature_row(
            current_aqi_value=previous_aqi,
            lag_aqi=lag_aqi,
            previous_change=previous_change,
            rolling_average=rolling_average,
            forecast_datetime=forecast_datetime,
        )

        sample = sample[feature_columns]

        sample = sample.astype(float)

        predicted_aqi = float(model.predict(sample)[0])

        predicted_aqi = float(np.clip(predicted_aqi, 1, 5))

        status = get_aqi_category(predicted_aqi)

        predictions.append(
            {
                "Date": forecast_datetime.strftime("%Y-%m-%d"),
                "Time": forecast_datetime.strftime("%H:%M"),
                "Predicted AQI": round(predicted_aqi, 3),
                "Category": status["category"],
            }
        )

        lag_aqi = previous_aqi

        previous_change = predicted_aqi - previous_aqi

        previous_aqi = predicted_aqi

        rolling_values.append(predicted_aqi)

    # =====================================================
    # RESULTS
    # =====================================================

    forecast_df = pd.DataFrame(predictions)

    st.subheader("📊 3-Day AQI Forecast")

    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    # =====================================================
    # FORECAST CARDS
    # =====================================================

    st.subheader("🌱 Forecast Interpretation")

    columns = st.columns(3)

    for index, prediction in enumerate(predictions):

        with columns[index]:

            aqi_value = prediction["Predicted AQI"]

            status = get_aqi_category(aqi_value)

            st.metric(f"Day {index + 1}", f"{aqi_value:.2f}")

            if status["category"] == "Good":
                st.success(f"{status['level']} Good")

            elif status["category"] == "Moderate":
                st.warning(f"{status['level']} Moderate")

            elif status["category"] == "Unhealthy":
                st.warning(f"{status['level']} Unhealthy")

            else:
                st.error(f"{status['level']} {status['category']}")

            st.caption(status["message"])

    # =====================================================
    # FORECAST TREND
    # =====================================================

    st.subheader("📈 Forecast Trend")

    chart_df = forecast_df[["Date", "Predicted AQI"]].copy()

    chart_df = chart_df.set_index("Date")

    st.line_chart(chart_df)

    # =====================================================
    # ALERT
    # =====================================================

    maximum_aqi = forecast_df["Predicted AQI"].max()

    maximum_status = get_aqi_category(maximum_aqi)

    if maximum_status["category"] in ["Unhealthy", "Poor", "Hazardous"]:

        st.error(
            f"🚨 Air Quality Alert: "
            f"Highest predicted AQI is "
            f"{maximum_aqi:.2f} "
            f"({maximum_status['category']})."
        )

    else:

        st.success(
            f"✅ No high-severity AQI alert. "
            f"Highest predicted AQI is "
            f"{maximum_aqi:.2f}."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("Pearls AQI Predictor | End-to-End Machine Learning System")
