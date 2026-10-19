# Kaggle Red Wine Quality in R: correlation (T7), regression (T8), ANOVA (T16), transformation (T15)
source("common/helpers.R"); start_topic("26_kaggle_wine_quality"); need("car")
df <- load_data("kaggle/wine_quality_red.csv"); names(df) <- gsub("\\.", "_", make.names(names(df)))
if (names(df)[1] != "fixed_acidity") names(df)[1] <- "fixed_acidity"
h("DATASET: Kaggle Red Wine Quality (1,599 wines)"); print(summary(df[, c("alcohol", "volatile_acidity", "chlorides", "quality")]))

h("CORRELATION (Topic 7)")
cr <- sort(cor(df)[, "quality"][names(df) != "quality"]); print(round(cr, 3))
ct <- cor.test(df$alcohol, df$quality); cat(sprintf("alcohol vs quality: r = %.3f, p = %.2e\n", ct$estimate, ct$p.value))

h("MULTIPLE REGRESSION (Topic 8)")
m <- lm(quality ~ alcohol + volatile_acidity + sulphates + total_sulfur_dioxide + chlorides + pH, data = df); print(summary(m))
cat("VIF:\n"); print(round(car::vif(m), 2)); cat("Shapiro p (residuals):", signif(shapiro.test(residuals(m))$p.value, 3), "\n")

h("ANOVA (Topic 16): alcohol by quality grade")
df$grade <- cut(df$quality, c(0, 5, 6, 10), labels = c("low(<=5)", "medium(6)", "high(>=7)"))
fit <- aov(alcohol ~ grade, data = df); print(summary(fit)); ss <- summary(fit)[[1]][["Sum Sq"]]; cat(sprintf("eta^2 = %.3f\n", ss[1] / sum(ss)))
print(TukeyHSD(fit)); print(car::leveneTest(alcohol ~ grade, data = df))

h("TRANSFORMATION (Topic 15)")
sk <- function(v) { m <- mean(v); mean((v - m)^3) / mean((v - m)^2)^1.5 }
cat(sprintf("chlorides skewness raw = %.2f; log = %.2f; sqrt = %.2f\n", sk(df$chlorides), sk(log(df$chlorides)), sk(sqrt(df$chlorides))))
save_plot("wine_overview", {
  par(mfrow = c(1, 3)); plot(df$alcohol, df$quality, pch = 16, cex = .4, col = rgb(0, 0, 1, .2), main = "Quality vs alcohol", xlab = "alcohol", ylab = "quality"); abline(lm(quality ~ alcohol, df), col = "red")
  boxplot(alcohol ~ grade, data = df, main = "Alcohol by grade"); plot(fitted(m), residuals(m), pch = 16, cex = .4, main = "Residuals vs fitted"); abline(h = 0, col = "red")
}, width = 1500, height = 500)
h("INTERPRETATION")
cat(sprintf("- Alcohol has the strongest positive link with quality (r = %.2f); volatile acidity the strongest negative (%.2f).\n", ct$estimate, cr["volatile_acidity"]))
cat(sprintf("- The model explains %.0f%% of variation in quality (adj R2 = %.2f).\n", summary(m)$r.squared * 100, summary(m)$adj.r.squared))
cat(sprintf("- Each extra 1%% alcohol is associated with +%.2f quality points, other variables fixed.\n", coef(m)["alcohol"]))
cat("- Alcohol differs significantly across quality grades; a log transform cuts the chlorides skew.\n")
end_topic()
