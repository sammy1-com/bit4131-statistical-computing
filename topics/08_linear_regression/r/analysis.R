# Topic 8: Linear regression and least squares (stats, car, broom)
source("common/helpers.R"); start_topic("08_linear_regression")
need("car", "broom")

h("WORKED EXAMPLE: least squares by hand, x = 1:5, y = c(2,4,5,4,5)")
x <- 1:5; y <- c(2, 4, 5, 4, 5)
b1 <- sum((x - mean(x)) * (y - mean(y))) / sum((x - mean(x))^2); b0 <- mean(y) - b1 * mean(x)
cat(sprintf("b1 = Sxy/Sxx = 6/10 = %.2f; b0 = %.2f -> y-hat = %.1f + %.1fx\n", b1, b0, b0, b1))
X <- cbind(1, x); cat("Normal equations:", round(solve(t(X) %*% X, t(X) %*% y), 3), "\n")

df <- load_data("students.csv")

h("SIMPLE REGRESSION: exam_score ~ study_hours")
simple <- lm(exam_score ~ study_hours, data = df); print(summary(simple))

h("MULTIPLE REGRESSION")
model <- lm(exam_score ~ study_hours + attendance_pct + sleep_hours + prior_gpa, data = df)
print(summary(model))
cat("\nbroom::tidy:\n"); print(broom::tidy(model, conf.int = TRUE))
cat("\nbroom::glance:\n"); print(broom::glance(model))
cat("\nVariance inflation factors (multicollinearity):\n"); print(car::vif(model))

h("DIAGNOSTICS")
res <- residuals(model)
print(shapiro.test(res))
cat("Durbin-Watson:\n"); print(car::durbinWatsonTest(model))
cat("Non-constant variance test:\n"); print(car::ncvTest(model))

h("HOLD-OUT CHECK (75/25 split)")
idx <- sample(nrow(df), 0.75 * nrow(df)); tr <- df[idx, ]; te <- df[-idx, ]
m2 <- lm(exam_score ~ study_hours + attendance_pct + sleep_hours + prior_gpa, data = tr)
pred <- predict(m2, te)
cat(sprintf("Test R2 = %.3f | Test RMSE = %.2f\n", 1 - sum((te$exam_score - pred)^2) / sum((te$exam_score - mean(te$exam_score))^2),
            sqrt(mean((te$exam_score - pred)^2))))

save_plot("fitted_regression", {
  plot(df$study_hours, df$exam_score, pch = 16, col = rgb(0, 0, 1, .4), xlab = "Study hours", ylab = "Exam score", main = "Fitted regression line")
  abline(simple, col = "red", lwd = 2)
})
save_plot("residual_diagnostics", { par(mfrow = c(2, 2)); plot(model) }, width = 1100, height = 900)

h("INTERPRETATION")
sm <- summary(model)
sig <- names(which(sm$coefficients[-1, 4] < 0.05))
cat(sprintf("- Model explains %.0f%% of variation (adj R2 = %.2f).\n", sm$r.squared * 100, sm$adj.r.squared))
cat(sprintf("- Each extra study hour adds about %.2f marks, other variables held fixed.\n", coef(model)[["study_hours"]]))
cat("- Significant predictors (p < 0.05):", paste(sig, collapse = ", "), "\n")
end_topic()
