# Topic 19: Non-parametric statistical methods (stats, coin, rstatix)
source("common/helpers.R"); start_topic("19_nonparametric")
need("coin", "rstatix")

h("WORKED EXAMPLE: Mann-Whitney U by hand, a = c(3,5,8), b = c(4,9,12,15)")
a <- c(3, 5, 8); b <- c(4, 9, 12, 15); rk <- rank(c(a, b))
cat("Combined ranks:", rk, "| R_a =", sum(rk[1:3]), "| U_a = R_a - 6 =", sum(rk[1:3]) - 6, "\n")
cat("wilcox.test statistic W =", suppressWarnings(wilcox.test(a, b)$statistic), "\n")

tk <- load_data("support_tickets.csv"); tk$priority <- factor(tk$priority, levels = c("Low", "Medium", "High"))
tut <- load_data("tutoring_scores.csv"); df <- load_data("students.csv")

h("DATA ARE SKEWED -> parametric assumptions fail")
cat(sprintf("resolution_hours skewness = %.2f\n", skewness(tk$resolution_hours))); print(shapiro.test(tk$resolution_hours))

h("TEST 1: Mann-Whitney U (wilcox.test) - High vs Low priority")
sub <- droplevels(subset(tk, priority %in% c("Low", "High")))
print(suppressWarnings(wilcox.test(resolution_hours ~ priority, data = sub)))
cat("Welch t-test for comparison:\n"); print(t.test(resolution_hours ~ priority, data = sub))
print(tryCatch(rstatix::wilcox_effsize(sub, resolution_hours ~ priority), error = function(e) NULL))
cat("Permutation (coin) version:\n")
print(tryCatch(coin::wilcox_test(resolution_hours ~ priority, data = sub, distribution = coin::approximate(nresample = 10000)), error = function(e) NULL))

h("TEST 2: Kruskal-Wallis, all three priorities")
print(kruskal.test(resolution_hours ~ priority, data = tk))
print(suppressWarnings(pairwise.wilcox.test(tk$resolution_hours, tk$priority, p.adjust.method = "bonferroni")))

h("TEST 3: Wilcoxon signed-rank - tutoring pre vs post")
print(suppressWarnings(wilcox.test(tut$post_score, tut$pre_score, paired = TRUE)))
print(t.test(tut$post_score, tut$pre_score, paired = TRUE))

h("TEST 4: Spearman vs Pearson correlation (study hours vs score)")
print(suppressWarnings(cor.test(df$study_hours, df$exam_score, method = "spearman")))
cat("Pearson r =", round(cor(df$study_hours, df$exam_score), 3), "\n")

h("TEST 5: Kolmogorov-Smirnov two-sample, method A vs C")
print(suppressWarnings(ks.test(df$exam_score[df$method == "A"], df$exam_score[df$method == "C"])))

save_plot("nonparametric_groups", {
  par(mfrow = c(1, 2))
  boxplot(resolution_hours ~ priority, data = tk, col = "lightblue", outline = FALSE, main = "Resolution time by priority", ylab = "Hours")
  hist(tk$resolution_hours, breaks = 40, col = "lightblue", border = "white", main = "Right-skewed data", xlab = "Hours")
}, width = 1200)
save_plot("paired_scores", {
  plot(tut$pre_score, tut$post_score, pch = 16, xlab = "Pre", ylab = "Post", main = "Paired scores (above line = improved)"); abline(0, 1, col = "red", lty = 2)
})

h("INTERPRETATION")
kw <- kruskal.test(resolution_hours ~ priority, data = tk)
med <- tapply(tk$resolution_hours, tk$priority, median)
cat("- Resolution times are heavily skewed, so rank-based tests are safer than t-tests.\n")
cat(sprintf("- Kruskal-Wallis p = %.2g: priorities differ; median hours Low %.1f, Medium %.1f, High %.1f.\n", kw$p.value, med[["Low"]], med[["Medium"]], med[["High"]]))
cat("- Wilcoxon and paired t agree on the tutoring data: scores improved.\n")
end_topic()
