# Kaggle Pima Diabetes in R: hidden missing values (T22), hypothesis tests (T6), PCA (T10)
source("common/helpers.R"); start_topic("25_kaggle_pima_diabetes")
df <- load_data("kaggle/pima_diabetes.csv")
h("DATASET: Kaggle Pima Indians Diabetes (768 women)"); print(summary(df))

h("HIDDEN MISSING VALUES (Topic 22)")
cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
z <- colSums(df[, cols] == 0); print(data.frame(zeros = z, pct = round(z / nrow(df) * 100, 1)))
d <- df; d[cols] <- lapply(d[cols], function(v) replace(v, v == 0, NA))
imp <- d; for (c in cols) imp[[c]][is.na(imp[[c]])] <- ave(imp[[c]], imp$Outcome, FUN = function(v) median(v, na.rm = TRUE))[is.na(imp[[c]])]
cat(sprintf("Insulin mean: zeros kept = %.1f; true observed = %.1f; median-imputed = %.1f\n", mean(df$Insulin), mean(d$Insulin, na.rm = TRUE), mean(imp$Insulin)))

h("HYPOTHESIS TESTS (Topic 6): Outcome 1 vs 0")
for (v in c("Glucose", "BMI", "Age", "BloodPressure")) {
  t <- t.test(imp[[v]] ~ imp$Outcome); a <- imp[[v]][imp$Outcome == 1]; b <- imp[[v]][imp$Outcome == 0]
  cat(sprintf("%-14s diabetic = %6.1f, non = %6.1f, Welch t = %6.2f, p = %.2e, Cohen d = %.2f\n", v, mean(a), mean(b), -t$statistic, t$p.value, (mean(a) - mean(b)) / sqrt((var(a) + var(b)) / 2)))
}

h("PCA (Topic 10)")
pc <- prcomp(imp[, names(imp) != "Outcome"], scale. = TRUE)
ev <- pc$sdev^2; cat("Eigenvalues:", round(ev, 3), "\nExplained variance:", round(ev / sum(ev), 3), "\nCumulative:", round(cumsum(ev / sum(ev)), 3), "\n")
print(round(pc$rotation[, 1:2], 2))
save_plot("pima_overview", {
  par(mfrow = c(1, 3)); barplot(ev / sum(ev), names.arg = 1:8, main = "Scree plot", xlab = "Component")
  plot(pc$x[, 1:2], col = ifelse(imp$Outcome == 1, "red", "blue"), pch = 16, cex = .5, main = "PC1 vs PC2 (red = diabetic)")
  boxplot(Glucose ~ Outcome, data = imp, main = "Glucose by outcome")
}, width = 1500, height = 500)
h("INTERPRETATION")
cat(sprintf("- %.0f%% of Insulin and %.0f%% of SkinThickness values are coded 0 (really missing); this drags the Insulin mean from %.0f to %.0f.\n", z["Insulin"] / nrow(df) * 100, z["SkinThickness"] / nrow(df) * 100, mean(d$Insulin, na.rm = TRUE), mean(df$Insulin)))
cat("- Glucose separates the groups most strongly (largest Cohen d).\n")
cat(sprintf("- The first two components explain %.0f%% of variance; classification needs several variables.\n", sum(ev[1:2]) / sum(ev) * 100))
end_topic()
