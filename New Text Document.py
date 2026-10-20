import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = True
np.random.seed(42)

USE_REAL_DATA = False
CSV_PATH = "weather.csv"   # columns: date,temp,humidity,pressure,wind,rain

def make_synthetic_weather(n_days=3 * 365 + 1, seed=42):
    """Daily weather with seasonality, noise and realistic cross-variable relationships."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values
    season = np.sin(2 * np.pi * (doy - 110) / 365.25)          # peaks in summer

    # temperature: seasonal cycle + persistent (autocorrelated) weather noise
    noise = np.zeros(n_days)
    for i in range(1, n_days):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 1.6)
    temp = 22 + 11 * season + noise

    humidity = np.clip(60 - 1.2 * (temp - 22) + rng.normal(0, 8, n_days), 15, 100)
    pressure = 1013 - 0.35 * (temp - 22) + rng.normal(0, 3, n_days)
    wind = np.clip(12 - 0.15 * (pressure - 1013) + rng.normal(0, 3, n_days), 0, None)
    rain = np.clip((humidity - 65) * 0.25 + rng.normal(0, 2, n_days), 0, None)

    return pd.DataFrame({"date": dates, "temp": temp, "humidity": humidity,
                         "pressure": pressure, "wind": wind, "rain": rain})

if USE_REAL_DATA:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
else:
    df = make_synthetic_weather()

df = df.sort_values("date").set_index("date")
df.head()

print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum(), sep="")
df.describe().round(2)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
df["temp"].plot(ax=axes[0], color="tab:red", lw=0.8, title="Daily average temperature (°C)")
df["humidity"].plot(ax=axes[1], color="tab:blue", lw=0.8, title="Daily humidity (%)")
plt.tight_layout(); plt.show()

# Correlation heat-map (matplotlib only)
corr = df.corr()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.grid(False); plt.colorbar(im); plt.title("Correlation matrix"); plt.show()

data = df.copy()

# lag & rolling features (use only past information)
for k in (1, 2, 3):
    data[f"temp_lag{k}"] = data["temp"].shift(k)
data["temp_roll7"] = data["temp"].rolling(7).mean()
data["pressure_change"] = data["pressure"].diff()      # falling pressure often = weather change

# seasonality
doy = data.index.dayofyear
data["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
data["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

# target: tomorrow's temperature
data["target_temp"] = data["temp"].shift(-1)

# keep the last row (no target yet) so we can forecast "tomorrow" later
latest_row = data.iloc[[-1]].drop(columns="target_temp")
data = data.dropna()

FEATURES = ["temp", "temp_lag1", "temp_lag2", "temp_lag3", "temp_roll7",
            "humidity", "pressure", "pressure_change", "wind", "rain",
            "doy_sin", "doy_cos"]
X, y = data[FEATURES], data["target_temp"]
print(X.shape, y.shape)
X.head()

split = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f"Train: {X_train.index.min().date()} → {X_train.index.max().date()}  ({len(X_train)} days)")
print(f"Test : {X_test.index.min().date()} → {X_test.index.max().date()}  ({len(X_test)} days)")

model = Pipeline([
    ("scaler", StandardScaler()),      # puts features on one scale → comparable coefficients
    ("reg", LinearRegression()),
])
model.fit(X_train, y_train)
print("Model trained ✔")

def report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "MAE (°C)": mae, "RMSE (°C)": rmse, "R²": r2}

pred_lr   = model.predict(X_test)
pred_base = X_test["temp"].values            # persistence

results = pd.DataFrame([
    report("Persistence baseline", y_test, pred_base),
    report("Linear Regression",    y_test, pred_lr),
]).set_index("Model").round(3)
results

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# (a) Actual vs predicted over time
axes[0].plot(y_test.index, y_test, label="Actual", color="black", lw=1)
axes[0].plot(y_test.index, pred_lr, label="Linear Regression", color="tab:red", lw=1, alpha=0.8)
axes[0].set_title("Next-day temperature: actual vs predicted (test set)")
axes[0].set_ylabel("°C"); axes[0].legend()

# (b) Scatter
axes[1].scatter(y_test, pred_lr, s=10, alpha=0.5)
lims = [y_test.min(), y_test.max()]
axes[1].plot(lims, lims, "r--", label="Perfect prediction")
axes[1].set_xlabel("Actual °C"); axes[1].set_ylabel("Predicted °C")
axes[1].set_title("Predicted vs actual"); axes[1].legend()
plt.tight_layout(); plt.show()

# Residual analysis: errors should be centred on 0 with no obvious pattern
residuals = y_test - pred_lr
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].hist(residuals, bins=30, color="tab:purple", alpha=0.8)
axes[0].axvline(0, color="k", ls="--"); axes[0].set_title("Residual distribution")
axes[1].plot(residuals.index, residuals, lw=0.7)
axes[1].axhline(0, color="k", ls="--"); axes[1].set_title("Residuals over time")
plt.tight_layout(); plt.show()
print(f"Mean residual: {residuals.mean():.3f} °C | Std: {residuals.std():.3f} °C")

coefs = pd.Series(model.named_steps["reg"].coef_, index=FEATURES).sort_values()
coefs.plot(kind="barh", color=np.where(coefs > 0, "tab:green", "tab:red"), figsize=(7, 5))
plt.title("Linear Regression coefficients (standardised)")
plt.xlabel("Effect on tomorrow's temperature (°C per 1 std-dev)"); plt.show()
print("Intercept:", round(model.named_steps["reg"].intercept_, 3))

tscv = TimeSeriesSplit(n_splits=5)
mae_scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
print("MAE per fold:", mae_scores.round(3))
print(f"Mean MAE: {mae_scores.mean():.3f} °C  (± {mae_scores.std():.3f})")

# Re-train on ALL available data, then predict from the most recent day
final_model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]).fit(X, y)

tomorrow = final_model.predict(latest_row[FEATURES])[0]
today = latest_row.index[0]
print(f"Latest observation : {today.date()}  ({latest_row['temp'].iloc[0]:.1f} °C)")
print(f"Forecast for {(today + pd.Timedelta(days=1)).date()}: {tomorrow:.1f} °C  (typical error ≈ ±{mae_scores.mean():.1f} °C)")

# Predict from your own custom input
def forecast_from_inputs(temp, temp_lag1, temp_lag2, temp_lag3, humidity, pressure,
                         pressure_prev, wind, rain, date):
    d = pd.Timestamp(date).dayofyear
    row = pd.DataFrame([{
        "temp": temp, "temp_lag1": temp_lag1, "temp_lag2": temp_lag2, "temp_lag3": temp_lag3,
        "temp_roll7": np.mean([temp, temp_lag1, temp_lag2, temp_lag3]),   # approximation
        "humidity": humidity, "pressure": pressure, "pressure_change": pressure - pressure_prev,
        "wind": wind, "rain": rain,
        "doy_sin": np.sin(2 * np.pi * d / 365.25), "doy_cos": np.cos(2 * np.pi * d / 365.25),
    }])[FEATURES]
    return final_model.predict(row)[0]

print("Example forecast:",
      round(forecast_from_inputs(30, 29, 28, 28, 55, 1008, 1010, 10, 0, "2026-07-15"), 1), "°C")