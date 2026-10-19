"""Topic 13: Time-series analysis (pandas, statsmodels)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller

R = Report(__file__)
df = load("monthly_sales.csv")
s = pd.Series(df.sales.to_numpy(), index=pd.to_datetime(df.date), name="sales").asfreq("MS")

# ---- 3. Worked example: moving average ------------------------------------------------------
R.h("WORKED EXAMPLE: 3-point moving average of [10, 12, 14, 13, 15]")
w = pd.Series([10, 12, 14, 13, 15]).rolling(3).mean()
R.p("MA(3) =", w.round(3).tolist(), " -> (10+12+14)/3 = 12, (12+14+13)/3 = 13, (14+13+15)/3 = 14")

# ---- 4/5. Dataset ------------------------------------------------------------------------------------
R.h("DATASET: 120 months of sales (Jan 2015 - Dec 2024)")
R.p(s.describe().round(2).to_string())
dec = seasonal_decompose(s, model="additive", period=12)
t = np.arange(len(s))
slope, intercept = np.polyfit(t, s.to_numpy(), 1)
R.p(f"\nLinear trend: sales = {intercept:.1f} + {slope:.3f} * month   (about {slope * 12:.1f} units/year)")
seas = dec.seasonal.iloc[:12]
R.p("\nSeasonal component by month:")
R.p(pd.Series(seas.to_numpy(), index=seas.index.strftime("%b")).round(2).to_string())
amp = seas.max() - seas.min()
peak, trough = seas.idxmax().strftime("%B"), seas.idxmin().strftime("%B")
R.p(f"\nPeak month = {peak}, trough month = {trough}, seasonal range = {amp:.1f}")

R.h("STATIONARITY (Augmented Dickey-Fuller)")
adf_raw = adfuller(s)
adf_diff = adfuller(s.diff().dropna())
R.p(f"Original series  : ADF = {adf_raw[0]:.3f}, p = {adf_raw[1]:.4f}  -> {'stationary' if adf_raw[1] < .05 else 'NON-stationary'}")
R.p(f"First difference : ADF = {adf_diff[0]:.3f}, p = {adf_diff[1]:.4f}  -> {'stationary' if adf_diff[1] < .05 else 'NON-stationary'}")

R.h("AUTOCORRELATION (lags 1, 6, 12)")
ac = acf(s, nlags=12)
R.p(f"ACF lag 1 = {ac[1]:.3f}, lag 6 = {ac[6]:.3f}, lag 12 = {ac[12]:.3f}")

R.h("FORECAST: Holt-Winters (additive trend + additive seasonality), 12 months ahead")
train, test = s[:-12], s[-12:]
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
pred = hw.forecast(12)
mae = (test - pred).abs().mean()
rmse = np.sqrt(((test - pred) ** 2).mean())
mape = ((test - pred).abs() / test).mean() * 100
R.p(f"Hold-out (2024): MAE = {mae:.2f}, RMSE = {rmse:.2f}, MAPE = {mape:.2f}%")
final = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12).fit()
future = final.forecast(12)
R.p("\nForecast for 2025:\n" + future.round(1).to_string())

# ---- Plots ----------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(s, label="Sales"); ax.plot(dec.trend, "r", lw=2, label="Trend (12-mo centred MA)")
ax.set_title("Monthly sales with trend"); ax.legend(); ax.set_ylabel("Sales")
R.save_fig(fig, "py_time_series_trend.png")

fig = dec.plot(); fig.set_size_inches(9, 8)
R.save_fig(fig, "py_decomposition.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
plot_acf(s.diff().dropna(), lags=36, ax=ax[0]); plot_pacf(s.diff().dropna(), lags=36, ax=ax[1], method="ywm")
ax[0].set_title("ACF (differenced series)"); ax[1].set_title("PACF (differenced series)")
R.save_fig(fig, "py_acf_pacf.png")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(s[-48:], label="Observed"); ax.plot(pred, "g--", label="Hold-out forecast")
ax.plot(future, "r", label="2025 forecast"); ax.legend(); ax.set_title("Holt-Winters forecast")
R.save_fig(fig, "py_forecast.png")

R.h("INTERPRETATION")
R.p(f"- Sales trend upward by about {slope * 12:.0f} units a year.")
R.p(f"- A clear 12-month seasonal cycle: highest around {peak}, lowest around {trough} (range {amp:.0f}).")
R.p("- The raw series is non-stationary (trend); differencing removes it, which ARIMA-type models require.")
R.p(f"- Holt-Winters predicts the held-out year with MAPE of {mape:.1f}%, and a 2025 forecast continues trend + seasonality.")
R.finish()
