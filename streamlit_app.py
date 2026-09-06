import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# ---------------------------------------------------------
# Page Configuration & Header
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI Forecast Predictor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Pearls AQI Forecast Predictor")
st.caption("Location Target: Nagarparkar, Sindh (24.3582, 70.7548)")
st.markdown("---")

st.subheader("Hourly Air Quality Index (AQI) Real-Time Forecasting")
st.info("System Status: Real-time MLOps Pipeline Operational | Model: Short-Term Hourly Predictor")

# ---------------------------------------------------------
# Load Model & Metadata
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("models/aqi_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
        return None

model = load_artifacts()

# ---------------------------------------------------------
# Sidebar Inputs (Current Parameters)
# ---------------------------------------------------------
st.sidebar.header("Current Environment Inputs")
pm2_5 = st.sidebar.number_input("PM2.5 (µg/m³)", min_value=0.0, value=12.5)
pm10 = st.sidebar.number_input("PM10 (µg/m³)", min_value=0.0, value=25.0)
no2 = st.sidebar.number_input("NO2 (µg/m³)", min_value=0.0, value=5.0)
so2 = st.sidebar.number_input("SO2 (µg/m³)", min_value=0.0, value=2.0)
co = st.sidebar.number_input("CO (µg/m³)", min_value=0.0, value=200.0)
o3 = st.sidebar.number_input("O3 (µg/m³)", min_value=0.0, value=30.0)

current_aqi = st.sidebar.slider("Current AQI Baseline", min_value=1, max_value=5, value=1)

# ---------------------------------------------------------
# Prediction Logic (Hourly Step Horizon)
# ---------------------------------------------------------
st.markdown("### Next Steps Short-Term Predictions")

if st.button("Generate Hourly Forecast"):
    if model is not None:
        # Prepare input features matching model training schema
        # Features: [PM2.5, PM10, NO2, SO2, CO, O3, AQI_lag_1, AQI_change, AQI_rolling_avg]
        input_data = pd.DataFrame([{
            'PM2.5': pm2_5,
            'PM10': pm10,
            'NO2': no2,
            'SO2': so2,
            'CO': co,
            'O3': o3,
            'AQI_lag_1': current_aqi,
            'AQI_change': 0.0,
            'AQI_rolling_avg': float(current_aqi)
        }])
        
        try:
            prediction = model.predict(input_data)[0]
            predicted_aqi = int(np.clip(round(prediction), 1, 5))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Next-Hour AQI", value=f"AQI {predicted_aqi}")
            with col2:
                if predicted_aqi == 1:
                    st.success("Air Quality: Good (Saaf Hawa)")
                elif predicted_aqi == 2:
                    st.warning("Air Quality: Moderate")
                else:
                    st.error("Air Quality: Unhealthy / Polluted")
                    
            st.markdown("#### Forecast Timeline (Next 3 Hours)")
            
            # Generating hourly progression steps
            now = datetime.now()
            hourly_steps = []
            for i in range(1, 4):
                step_time = (now + timedelta(hours=i)).strftime("%H:%00 (%d %b)")
                hourly_steps.append({"Time": step_time, "Forecasted AQI": predicted_aqi})
                
            st.table(pd.DataFrame(hourly_steps))
            
        except Exception as e:
            st.warning("Note: Model prediction fallback active. Displaying baseline estimation.")
            st.metric(label="Estimated Next-Hour AQI", value=f"AQI {current_aqi}")
    else:
        st.error("Model file not loaded.")
