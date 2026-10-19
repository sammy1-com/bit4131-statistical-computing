# Topic 1: Measures of central tendency and dispersion (base R, dplyr, psych)
source("common/helpers.R"); start_topic("01_central_tendency_dispersion")
need("dplyr", "psych")

df <- load_data("students.csv"); x <- df$exam_score

h("WORKED EXAMPLE: data = c(4, 8, 6, 5, 3, 8, 9)")
w <- c(4, 8, 6, 5, 3, 8, 9)
cat("mean =", mean(w), "| median =", median(w), "| mode =", get_mode(w), "\n")
cat("variance =", round(var(w), 4), "| sd =", round(sd(w), 4), "(sample, n-1)\n")

h("DATASET: exam_score of 300 students")
q <- quantile(x, c(.25, .5, .75)); iqr <- IQR(x)
fences <- c(q[[1]] - 1.5 * iqr, q[[3]] + 1.5 * iqr)
out <- x[x < fences[1] | x > fences[2]]
cat(sprintf("Mean %.2f | Median %.2f | Mode(rounded) %.0f\n", mean(x), median(x), get_mode(round(x))))
cat(sprintf("Variance %.2f | SD %.2f | Range %.1f to %.1f\n", var(x), sd(x), min(x), max(x)))
cat(sprintf("Q1 %.2f | Q2 %.2f | Q3 %.2f | IQR %.2f\n", q[[1]], q[[2]], q[[3]], iqr))
cat(sprintf("Outlier fences [%.2f, %.2f] -> %d outlier(s)\n", fences[1], fences[2], length(out)))
cat(sprintf("Coefficient of variation %.1f%%\n", sd(x) / mean(x) * 100))
cat("\npsych::describe output:\n"); print(psych::describe(x))

h("Dispersion by teaching method")
print(df %>% group_by(method) %>%
        summarise(n = n(), mean = mean(exam_score), median = median(exam_score), sd = sd(exam_score)))

save_plot("boxplot_hist", {
  par(mfrow = c(1, 2))
  boxplot(x, horizontal = TRUE, col = "lightblue", main = "Boxplot of exam scores", xlab = "Exam score")
  hist(x, breaks = 20, col = "lightblue", border = "white", main = "Histogram", xlab = "Exam score")
  abline(v = mean(x), col = "red", lwd = 2); abline(v = median(x), col = "darkgreen", lty = 2, lwd = 2)
  legend("topright", c("Mean", "Median"), col = c("red", "darkgreen"), lty = c(1, 2), lwd = 2, bty = "n")
}, width = 1200)
save_plot("boxplot_by_method", boxplot(exam_score ~ method, data = df, col = c("#fb9a99", "#a6cee3", "#b2df8a"),
                                       main = "Exam score by teaching method", xlab = "Method", ylab = "Exam score"))

h("INTERPRETATION")
cat(sprintf("- Mean (%.1f) vs median (%.1f): %s\n", mean(x), median(x),
            ifelse(abs(mean(x) - median(x)) < 0.05 * sd(x), "approximately symmetric.", ifelse(mean(x) > median(x), "right-skewed.", "left-skewed."))))
cat(sprintf("- Typical deviation from the mean is %.1f marks; the middle 50%% spans %.1f marks.\n", sd(x), iqr))
cat(sprintf("- %d value(s) lie beyond the 1.5 x IQR fences.\n", length(out)))
end_topic()
