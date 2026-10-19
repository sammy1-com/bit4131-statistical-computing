# Topic 13: Time-series analysis (stats, forecast, tseries)
source("common/helpers.R"); start_topic("13_time_series")
need("forecast", "tseries")

h("WORKED EXAMPLE: 3-point moving average of c(10,12,14,13,15)")
print(stats::filter(c(10, 12, 14, 13, 15), rep(1 / 3, 3), sides = 1))

df <- load_data("monthly_sales.csv")
s <- ts(df$sales, start = c(2015, 1), frequency = 12)

h("DATASET: 120 months of sales (Jan 2015 - Dec 2024)")
print(summary(s))
dec <- decompose(s, type = "additive")
tt <- seq_along(s); fit <- lm(as.numeric(s) ~ tt)
cat(sprintf("\nLinear trend: sales = %.1f + %.3f*month (about %.1f units/year)\n", coef(fit)[1], coef(fit)[2], coef(fit)[2] * 12))
seas <- dec$figure; names(seas) <- month.abb
cat("\nSeasonal component by month:\n"); print(round(seas, 2))
cat(sprintf("Peak = %s | trough = %s | range = %.1f\n", names(which.max(seas)), names(which.min(seas)), diff(range(seas))))

h("STATIONARITY (Augmented Dickey-Fuller)")
print(tseries::adf.test(s)); print(tseries::adf.test(diff(s)))

h("AUTOCORRELATION")
a <- acf(s, lag.max = 12, plot = FALSE)
cat(sprintf("ACF lag 1 = %.3f, lag 6 = %.3f, lag 12 = %.3f\n", a$acf[2], a$acf[7], a$acf[13]))

h("FORECAST: ETS (Holt-Winters type), hold-out on the last 12 months")
train <- window(s, end = c(2023, 12)); test <- window(s, start = c(2024, 1))
m <- forecast::ets(train); fc <- forecast::forecast(m, h = 12)
print(forecast::accuracy(fc, test))
final <- forecast::forecast(forecast::ets(s), h = 12)
cat("\n2025 forecast:\n"); print(round(final$mean, 1))

save_plot("time_series_trend", { plot(s, ylab = "Sales", main = "Monthly sales with trend"); lines(dec$trend, col = "red", lwd = 2) })
save_plot("decomposition", plot(dec), height = 800)
save_plot("acf_pacf", { par(mfrow = c(1, 2)); acf(diff(s), lag.max = 36, main = "ACF (differenced)"); pacf(diff(s), lag.max = 36, main = "PACF (differenced)") }, width = 1200)
save_plot("forecast", plot(final, main = "ETS forecast for 2025", ylab = "Sales"))

h("INTERPRETATION")
cat(sprintf("- Sales trend upward by about %.0f units a year.\n", coef(fit)[2] * 12))
cat(sprintf("- Clear 12-month seasonality: highest in %s, lowest in %s.\n", names(which.max(seas)), names(which.min(seas))))
cat("- The raw series is non-stationary (trend); differencing removes it, as ARIMA-type models require.\n")
end_topic()
