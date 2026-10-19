# Kaggle Air Passengers in R: time series (T13), log transform (T15)
source("common/helpers.R"); start_topic("27_kaggle_air_passengers"); need("forecast", "tseries")
df <- load_data("kaggle/air_passengers.csv"); s <- ts(df$Passengers, start = c(1949, 1), frequency = 12)
h("DATASET: Kaggle Air Passengers (monthly, 1949-1960)"); print(summary(s)); print(round(tapply(s, floor(time(s)), mean), 1))
h("DECOMPOSITION (multiplicative)"); dm <- decompose(s, type = "multiplicative"); print(round(dm$figure, 3))
h("STATIONARITY (Augmented Dickey-Fuller, constant, 12 lags; 5% critical value about -2.88)")
adf <- function(x, k = 12) {  # regress dx_t on x_{t-1} and k lagged differences; the t-statistic on x_{t-1} is the ADF statistic
  x <- as.numeric(x); dx <- diff(x); n <- length(dx)
  X <- cbind(x[k:(n)], sapply(1:k, function(i) dx[(k - i + 1):(n - i)]))
  fit <- lm(dx[(k + 1):n] ~ X); summary(fit)$coefficients[2, "t value"]
}
for (nm in c("raw", "log", "log + diff(12) + diff(1)")) {
  x <- switch(nm, raw = s, log = log(s), diff(diff(log(s), 12))); st <- adf(x)
  cat(sprintf("%-28s ADF = %6.3f -> %s\n", nm, st, ifelse(st < -2.88, "stationary", "NON-stationary")))
}
h("FORECAST: Holt-Winters multiplicative, hold out last 24 months")
tr <- window(s, end = c(1958, 12)); te <- window(s, start = c(1959, 1))
hw <- HoltWinters(tr, seasonal = "multiplicative"); fc <- predict(hw, 24)
cat(sprintf("MAE = %.1f, RMSE = %.1f, MAPE = %.1f%%\n", mean(abs(te - fc)), sqrt(mean((te - fc)^2)), mean(abs(te - fc) / te) * 100))
nxt <- predict(HoltWinters(s, seasonal = "multiplicative"), 12); cat("Next 12 months:", round(nxt), "\n")
save_plot("airpassengers_overview", {
  par(mfrow = c(2, 2)); plot(s, main = "Monthly passengers (thousands)"); plot(log(s), main = "Log scale")
  plot(dm$figure, type = "b", main = "Seasonal index", xlab = "Month", ylab = ""); plot(s, xlim = c(1957, 1962), main = "Holt-Winters forecast"); lines(fc, col = "red", lty = 2); lines(nxt, col = "green4", lty = 2)
}, width = 1200, height = 900)
h("INTERPRETATION")
cat(sprintf("- Traffic grew about %.1fx over 1949-1960 with a strong yearly cycle peaking in month %d (lowest in month %d).\n", mean(tail(s, 12)) / mean(head(s, 12)), which.max(dm$figure), which.min(dm$figure)))
cat("- The seasonal swing widens with the level, so a multiplicative model / log transform is right; the raw series is non-stationary.\n")
end_topic()
