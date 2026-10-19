# Topic 5: Confidence intervals (stats, DescTools, boot)
source("common/helpers.R"); start_topic("05_confidence_intervals")
need("boot")

h("WORKED EXAMPLE: n=25, mean=50, s=10, 95% CI")
n <- 25; xbar <- 50; s <- 10; tcrit <- qt(.975, n - 1); me <- tcrit * s / sqrt(n)
cat(sprintf("SE = %.2f; t(0.975, 24) = %.4f; margin = %.3f; CI = (%.2f, %.2f)\n", s / sqrt(n), tcrit, me, xbar - me, xbar + me))

df <- load_data("students.csv"); x <- df$exam_score

h("CI FOR MEAN EXAM SCORE (t-interval)")
for (lv in c(.90, .95, .99)) {
  ci <- t.test(x, conf.level = lv)$conf.int
  cat(sprintf("%.0f%% CI: (%.2f, %.2f) width = %.2f\n", lv * 100, ci[1], ci[2], diff(ci)))
}
cat(sprintf("Point estimate = %.2f\n", mean(x)))

h("CI FOR PASS PROPORTION (Wilson)")
if (opt("DescTools")) {
  print(DescTools::BinomCI(sum(df$passed), nrow(df), conf.level = .95, method = "wilson"))
} else {  # prop.test without continuity correction IS the Wilson interval
  print(prop.test(sum(df$passed), nrow(df), correct = FALSE)$conf.int)
}

h("CI FOR DIFFERENCE IN MEANS (method C - A, Welch)")
a <- df$exam_score[df$method == "A"]; cc <- df$exam_score[df$method == "C"]
tt <- t.test(cc, a)
cat(sprintf("Difference = %.2f; 95%% CI = (%.2f, %.2f)\n", unname(tt$estimate[1] - tt$estimate[2]), tt$conf.int[1], tt$conf.int[2]))

h("BOOTSTRAP CI FOR THE MEAN (boot package)")
b <- boot(x, function(d, i) mean(d[i]), R = 2000)
print(boot.ci(b, type = c("perc", "bca")))

h("COVERAGE SIMULATION: 100 intervals from samples of n = 30")
mu_true <- mean(x)
ints <- t(replicate(100, { smp <- sample(x, 30); t.test(smp)$conf.int[1:2] }))
hit <- ints[, 1] <= mu_true & ints[, 2] >= mu_true
cat(sprintf("%d/100 intervals contain the true mean %.2f\n", sum(hit), mu_true))
save_plot("ci_coverage", {
  plot(NA, xlim = range(ints), ylim = c(1, 100), xlab = "Exam score", ylab = "Sample #", main = sprintf("100 95%% CIs (%d cover)", sum(hit)))
  segments(ints[, 1], 1:100, ints[, 2], 1:100, col = ifelse(hit, "steelblue", "red"), lwd = 1.5)
  abline(v = mu_true, lty = 2)
}, height = 800)

h("INTERPRETATION")
ci95 <- t.test(x)$conf.int
cat(sprintf("- 95%% confident the true mean exam score is between %.1f and %.1f.\n", ci95[1], ci95[2]))
cat("- 'Confidence' describes the method: about 95 of 100 such intervals capture the true mean.\n")
cat(sprintf("- C - A interval (%.1f, %.1f) %s 0.\n", tt$conf.int[1], tt$conf.int[2], ifelse(tt$conf.int[1] > 0, "excludes", "includes")))
end_topic()
