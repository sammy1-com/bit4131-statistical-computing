# Topic 20: Bootstrap and resampling methods (boot, rsample, infer)
source("common/helpers.R"); start_topic("20_bootstrap")
need("boot")

h("WORKED EXAMPLE: resampling c(3, 5, 7, 9, 11) with replacement")
tiny <- c(3, 5, 7, 9, 11)
for (i in 1:3) { s <- sample(tiny, 5, replace = TRUE); cat("resample", i, ":", s, "-> mean", mean(s), "\n") }

tk <- load_data("support_tickets.csv")$resolution_hours; df <- load_data("students.csv"); x <- df$exam_score
B <- 5000

h("BOOTSTRAP: median resolution time (boot package)")
bm <- boot(tk, function(d, i) median(d[i]), R = B)
cat(sprintf("Sample median = %.3f | bootstrap SE = %.4f | bias = %+.4f\n", median(tk), sd(bm$t), mean(bm$t) - median(tk)))
print(boot.ci(bm, type = c("perc", "bca")))

h("BOOTSTRAP: mean exam score vs t-interval")
bmean <- boot(x, function(d, i) mean(d[i]), R = B)
cat(sprintf("Bootstrap SE = %.4f vs analytic %.4f\n", sd(bmean$t), sd(x) / sqrt(length(x))))
cat("Bootstrap percentile CI:", round(quantile(bmean$t, c(.025, .975)), 2), "| t-interval:", round(t.test(x)$conf.int, 2), "\n")

h("BOOTSTRAP: correlation (study_hours, exam_score)")
bcor <- boot(df, function(d, i) cor(d$study_hours[i], d$exam_score[i]), R = B)
cat(sprintf("r = %.3f | SE = %.4f | 95%% CI = (%.3f, %.3f)\n", cor(df$study_hours, df$exam_score), sd(bcor$t), quantile(bcor$t, .025), quantile(bcor$t, .975)))

h("BOOTSTRAP: difference in means, C - A")
a <- df$exam_score[df$method == "A"]; cc <- df$exam_score[df$method == "C"]
bd <- replicate(B, mean(sample(cc, replace = TRUE)) - mean(sample(a, replace = TRUE)))
cat(sprintf("Observed %.2f | 95%% CI = (%.2f, %.2f)\n", mean(cc) - mean(a), quantile(bd, .025), quantile(bd, .975)))

if (opt("rsample")) {
  h("rsample: tidy bootstrap of the median")
  bs <- rsample::bootstraps(data.frame(h = tk), times = 2000)
  med <- sapply(bs$splits, function(s) median(rsample::analysis(s)$h))
  cat(sprintf("rsample bootstrap SE = %.4f; percentile CI = (%.3f, %.3f)\n", sd(med), quantile(med, .025), quantile(med, .975)))
}
if (opt("infer")) {
  h("infer: bootstrap CI for the mean exam score")
  res <- tryCatch({
    library(infer)
    obs <- data.frame(exam_score = x)
    bsd <- obs %>% specify(response = exam_score) %>% generate(reps = 2000, type = "bootstrap") %>% calculate(stat = "mean")
    get_confidence_interval(bsd, level = .95, type = "percentile")
  }, error = function(e) NULL)
  if (!is.null(res)) print(res)
}

save_plot("bootstrap_distributions", {
  par(mfrow = c(1, 3))
  hist(bm$t, breaks = 40, col = "lightblue", border = "white", main = "Bootstrap: median resolution time", xlab = "Median (h)")
  abline(v = quantile(bm$t, c(.025, .975)), col = "red", lty = 2)
  hist(bmean$t, breaks = 40, col = "lightgreen", border = "white", main = "Bootstrap: mean exam score", xlab = "Mean")
  hist(bcor$t, breaks = 40, col = "orange", border = "white", main = "Bootstrap: correlation", xlab = "r")
}, width = 1400, height = 500)

h("INTERPRETATION")
cat(sprintf("- The median has no simple SE formula; bootstrap gives SE = %.3f h and a CI of (%.2f, %.2f).\n", sd(bm$t), quantile(bm$t, .025), quantile(bm$t, .975)))
cat("- For the mean, bootstrap and t-intervals nearly coincide, validating the bootstrap.\n")
cat("- Correlations and mean differences get honest uncertainty estimates without normality assumptions.\n")
end_topic()
