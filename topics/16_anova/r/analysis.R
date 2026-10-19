# Topic 16: ANOVA and experimental design (stats, car, agricolae)
source("common/helpers.R"); start_topic("16_anova")
need("car")

h("WORKED EXAMPLE: G1 = 5,6,7  G2 = 8,9,10  G3 = 11,12,13")
g <- data.frame(y = c(5:7, 8:10, 11:13), grp = factor(rep(c("G1", "G2", "G3"), each = 3)))
print(summary(aov(y ~ grp, g)))
cat("By hand: SSB = 54, SSW = 6, F = (54/2)/(6/6) = 27\n")

df <- load_data("students.csv"); df$method <- factor(df$method); df$gender <- factor(df$gender)

h("ONE-WAY ANOVA: exam_score ~ method")
fit <- aov(exam_score ~ method, data = df); print(summary(fit))
cat("\nType II table (car::Anova):\n"); print(car::Anova(fit, type = 2))
ss <- summary(fit)[[1]][["Sum Sq"]]; cat(sprintf("eta-squared = %.3f\n", ss[1] / sum(ss)))
cat("\nGroup means:\n"); print(aggregate(exam_score ~ method, df, function(v) c(n = length(v), mean = mean(v), sd = sd(v))))

h("ASSUMPTION CHECKS")
print(car::leveneTest(exam_score ~ method, data = df)); print(shapiro.test(residuals(fit)))

h("POST-HOC: Tukey HSD")
tk <- TukeyHSD(fit); print(tk)
if (opt("agricolae")) {
  cat("\nagricolae::HSD.test grouping letters:\n"); print(agricolae::HSD.test(fit, "method", group = TRUE)$groups)
}

h("TWO-WAY ANOVA: method * gender")
fit2 <- aov(exam_score ~ method * gender, data = df); print(summary(fit2))

save_plot("anova_groups", {
  par(mfrow = c(1, 2))
  boxplot(exam_score ~ method, data = df, col = "lightblue", main = "Score by method")
  m <- tapply(df$exam_score, df$method, mean); se <- tapply(df$exam_score, df$method, function(v) sd(v) / sqrt(length(v)))
  plot(1:3, m, pch = 16, xaxt = "n", xlab = "Method", ylab = "Mean score", main = "Group means with 95% CI", ylim = range(c(m - 2 * se, m + 2 * se)))
  axis(1, 1:3, names(m)); arrows(1:3, m - 1.96 * se, 1:3, m + 1.96 * se, angle = 90, code = 3, length = .05)
}, width = 1200)
save_plot("tukey_hsd", { par(las = 1); plot(tk) })
save_plot("interaction_plot", interaction.plot(df$method, df$gender, df$exam_score, xlab = "Method", ylab = "Mean score", trace.label = "Gender"))

h("INTERPRETATION")
p <- summary(fit)[[1]][["Pr(>F)"]][1]
cat(sprintf("- F-test p = %.2g: %s\n", p, ifelse(p < .05, "at least one method differs.", "no evidence of a difference.")))
cat(sprintf("- Method explains %.0f%% of score variance (eta-squared).\n", ss[1] / sum(ss) * 100))
cat("- Tukey HSD shows which pairs differ (see table above); interaction p-value is in the two-way table.\n")
end_topic()
