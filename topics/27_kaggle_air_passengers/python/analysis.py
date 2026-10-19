"""Kaggle Air Passengers: time series (T13) and log transform (T15)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from common.helpers import Report, load

R = Report(__file__)
df = load("kaggle/air_passengers.csv"); s = pd.Series(df.Passengers.values, index=pd.date_range(df.Month.iloc[0], periods=len(df), freq="MS"))
R.h("DATASET: Kaggle Air Passengers (monthly, 1949-1960, 144 obs)"); R.p(s.describe().round(1))
yearly = s.groupby(s.index.year).mean(); R.p("Yearly mean:\n" + yearly.round(1).to_string())

R.h("DECOMPOSITION")
add = seasonal_decompose(s, model="additive"); mul = seasonal_decompose(s, model="multiplicative")
R.p(f"Seasonal swing grows with the level -> multiplicative is appropriate. Residual SD additive = {add.resid.std():.2f}; multiplicative (ratio) = {mul.resid.std():.3f}")
R.p("Seasonal indices (multiplicative):\n" + mul.seasonal.iloc[:12].round(3).set_axis(range(1, 13)).to_string())

R.h("STATIONARITY (ADF)")
for name, x in [("raw", s), ("log", np.log(s)), ("log, seasonal diff(12) + diff(1)", np.log(s).diff(12).diff().dropna())]:
    st, p = adfuller(x)[:2]; R.p(f"{name:36s} ADF = {st:6.3f}, p = {p:.4f} -> {'stationary' if p < .05 else 'NON-stationary'}")
R.p("ACF lags 1, 12:", acf(s, nlags=12)[[1, 12]].round(3))

R.h("FORECAST: Holt-Winters multiplicative, hold out last 24 months")
tr, te = s[:-24], s[-24:]
hw = ExponentialSmoothing(tr, trend="add", seasonal="mul", seasonal_periods=12).fit(); fc = hw.forecast(24)
mae = (te - fc).abs().mean(); rmse = np.sqrt(((te - fc) ** 2).mean()); mape = ((te - fc).abs() / te).mean() * 100
R.p(f"MAE = {mae:.1f}, RMSE = {rmse:.1f}, MAPE = {mape:.1f}%")
full = ExponentialSmoothing(s, trend="add", seasonal="mul", seasonal_periods=12).fit(); nxt = full.forecast(12)
R.p("Next 12 months:\n" + nxt.round(0).astype(int).to_string())

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].plot(s); ax[0, 0].set_title("Monthly passengers (thousands)")
ax[0, 1].plot(np.log(s)); ax[0, 1].set_title("Log scale: constant seasonal swing")
mul.seasonal.iloc[:12].plot(ax=ax[1, 0], marker="o"); ax[1, 0].set_title("Seasonal index (multiplicative)")
ax[1, 1].plot(tr, label="train"); ax[1, 1].plot(te, label="actual"); ax[1, 1].plot(fc, "r--", label="forecast"); ax[1, 1].plot(nxt, "g--", label="next 12"); ax[1, 1].legend(); ax[1, 1].set_title("Holt-Winters forecast")
R.save_fig(fig, "py_airpassengers_overview.png")

peak = mul.seasonal.iloc[:12].idxmax().month_name(); low = mul.seasonal.iloc[:12].idxmin().month_name()
R.h("INTERPRETATION")
R.p(f"- Traffic grew about {s.iloc[-12:].mean()/s.iloc[:12].mean():.1f}x from 1949 to 1960 with a strong yearly cycle peaking in {peak} and lowest in {low}.")
R.p("- The seasonal swing widens as the level rises, so a multiplicative model (or log transform) is right; the raw series is non-stationary.")
R.p(f"- After log + seasonal and ordinary differencing the series is stationary; Holt-Winters forecasts the last 2 years with MAPE {mape:.1f}%.")
R.finish()
