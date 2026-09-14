import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AeroSense - BASF AQI Prediction",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "models/aerosense_aqi_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# AQI Category
# -----------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderately Polluted"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# -----------------------------
# Header
# -----------------------------
st.title("🌍 AeroSense")
st.subheader("AI-Based Air Quality Prediction System for BASF")

st.write(
    "Enter environmental parameters below to predict the Air Quality Index (AQI) "
    "using the trained Gradient Boosting machine learning model."
)

st.divider()

# -----------------------------
# Input Parameters
# -----------------------------
st.subheader("🌫️ Environmental Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input(
        "PM2.5 (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=85.0
    )

    pm10 = st.number_input(
        "PM10 (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=160.0
    )

    no2 = st.number_input(
        "NO₂ (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=45.0
    )

with col2:
    so2 = st.number_input(
        "SO₂ (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=25.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        max_value=100.0,
        value=1.2
    )

    o3 = st.number_input(
        "O₃ (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=50.0
    )

with col3:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=100.0,
        value=2.5
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict AQI", use_container_width=True):

    input_data = pd.DataFrame([{
        "PM2_5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "O3": o3,
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind_Speed": wind_speed
    }])

    prediction = model.predict(input_data)[0]

    # Keep AQI within valid range
    prediction = max(0, min(500, prediction))

    category = get_aqi_category(prediction)

    st.success("AQI prediction completed successfully.")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            label="Predicted AQI",
            value=f"{prediction:.2f}"
        )

    with result_col2:
        st.metric(
            label="AQI Category",
            value=category
        )

    st.divider()

    st.subheader("📊 Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True
    )

    st.info(
        "The prediction is generated using the trained AeroSense Gradient Boosting model."
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "AeroSense | AI for Engineers | BASF Air Quality Prediction System"
)


