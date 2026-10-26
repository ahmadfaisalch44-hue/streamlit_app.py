import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Weather Forecast App", layout="wide")
st.title("🌤️ Temperature Forecasting App")

@st.cache_data
def load_data():
    rng = np.random.default_rng(42)
    n_days = 3 * 365 + 1
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values
    season = np.sin(2 * np.pi * (doy - 110) / 365.25)

    noise = np.zeros(n_days)
    for i in range(1, n_days):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 1.6)
    temp = 22 + 11 * season + noise

    humidity = np.clip(60 - 1.2 * (temp - 22) + rng.normal(0, 8, n_days), 15, 100)
    pressure = 1013 - 0.35 * (temp - 22) + rng.normal(0, 3, n_days)
    wind = np.clip(12 - 0.15 * (pressure - 1013) + rng.normal(0, 3, n_days), 0, None)
    rain = np.clip((humidity - 65) * 0.25 + rng.normal(0, 2, n_days), 0, None)

    return pd.DataFrame({"date": dates, "temp": temp, "humidity": humidity,
                         "pressure": pressure, "wind": wind, "rain": rain}).set_index("date")

df = load_data()

# Prepare features
data = df.copy()
for k in (1, 2, 3):
    data[f"temp_lag{k}"] = data["temp"].shift(k)
data["temp_roll7"] = data["temp"].rolling(7).mean()
data["pressure_change"] = data["pressure"].diff()

doy = data.index.dayofyear
data["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
data["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
data["target_temp"] = data["temp"].shift(-1)
data = data.dropna()

FEATURES = ["temp", "temp_lag1", "temp_lag2", "temp_lag3", "temp_roll7",
            "humidity", "pressure", "pressure_change", "wind", "rain",
            "doy_sin", "doy_cos"]

X, y = data[FEATURES], data["target_temp"]
model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]).fit(X, y)

# Sidebar Inputs for Interactive Prediction
st.sidebar.header("Input Weather Conditions")
input_date = st.sidebar.date_input("Date")
temp = st.sidebar.number_input("Today's Temp (°C)", value=30.0)
temp_lag1 = st.sidebar.number_input("Yesterday's Temp (°C)", value=29.0)
temp_lag2 = st.sidebar.number_input("Temp 2 Days Ago (°C)", value=28.0)
temp_lag3 = st.sidebar.number_input("Temp 3 Days Ago (°C)", value=28.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 55)
pressure = st.sidebar.number_input("Pressure (hPa)", value=1008.0)
pressure_prev = st.sidebar.number_input("Yesterday's Pressure (hPa)", value=1010.0)
wind = st.sidebar.number_input("Wind Speed (km/h)", value=10.0)
rain = st.sidebar.number_input("Rainfall (mm)", value=0.0)

# Forecast
d = pd.Timestamp(input_date).dayofyear
row = pd.DataFrame([{
    "temp": temp, "temp_lag1": temp_lag1, "temp_lag2": temp_lag2, "temp_lag3": temp_lag3,
    "temp_roll7": np.mean([temp, temp_lag1, temp_lag2, temp_lag3]),
    "humidity": humidity, "pressure": pressure, "pressure_change": pressure - pressure_prev,
    "wind": wind, "rain": rain,
    "doy_sin": np.sin(2 * np.pi * d / 365.25), "doy_cos": np.cos(2 * np.pi * d / 365.25),
}])[FEATURES]

pred = model.predict(row)[0]

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Tomorrow Temperature", value=f"{pred:.1f} °C")

with col2:
    st.line_chart(df[["temp", "humidity"]].tail(30))
