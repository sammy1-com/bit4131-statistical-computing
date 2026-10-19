# Topic 7: Correlation and covariance (stats, Hmisc, corrplot)
source("common/helpers.R"); start_topic("07_correlation_covariance")
need("Hmisc", "corrplot")

h("WORKED EXAMPLE: x = 1:5, y = c(2,4,5,4,5)")
x <- 1:5; y <- c(2, 4, 5, 4, 5)
cat("cov(x,y) =", cov(x, y), "(= 6/4)\n")
cat("r = cov/(sx*sy) =", round(cov(x, y) / (sd(x) * sd(y)), 4), "| cor():", round(cor(x, y), 4), "\n")

df <- load_data("students.csv")
cols <- c("study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score")
num <- df[, cols]

h("COVARIANCE MATRIX"); print(round(cov(num), 2))
h("PEARSON CORRELATION MATRIX"); print(round(cor(num), 3))
h("SPEARMAN CORRELATION MATRIX"); print(round(cor(num, method = "spearman"), 3))

h("SIGNIFICANCE (Hmisc::rcorr)")
rc <- rcorr(as.matrix(num))
cat("p-values:\n"); print(signif(rc$P, 3))
h("cor.test: study_hours vs exam_score"); print(cor.test(num$study_hours, num$exam_score))

save_plot("correlation_heatmap", corrplot(cor(num), method = "color", addCoef.col = "black", tl.col = "black",
                                          title = "Correlation matrix", mar = c(0, 0, 2, 0)))
save_plot("scatter_plots", {
  par(mfrow = c(1, 2))
  plot(df$study_hours, df$exam_score, pch = 16, col = rgb(0, 0, 1, .4), xlab = "Study hours", ylab = "Exam score",
       main = sprintf("r = %.2f", cor(df$study_hours, df$exam_score)))
  abline(lm(exam_score ~ study_hours, df), col = "red", lwd = 2)
  plot(df$sleep_hours, df$exam_score, pch = 16, col = rgb(0, .6, 0, .4), xlab = "Sleep hours", ylab = "Exam score",
       main = sprintf("r = %.2f", cor(df$sleep_hours, df$exam_score)))
}, width = 1200)

h("INTERPRETATION")
r_all <- cor(num)[1:4, "exam_score"]
best <- names(which.max(abs(r_all)))
cat(sprintf("- Strongest linear link with exam score: %s (r = %+.2f).\n", best, r_all[[best]]))
cat("- Correlation is unit-free in [-1, 1]; covariance depends on units. Correlation is not causation.\n")
end_topic()
