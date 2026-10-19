# Topic 6: Hypothesis testing (stats, car, rstatix)
source("common/helpers.R"); start_topic("06_hypothesis_testing")
need("car", "rstatix")

decide <- function(p, alpha = 0.05) sprintf("p = %.4g -> %s at alpha = %.2f", p, ifelse(p < alpha, "REJECT H0", "FAIL TO REJECT H0"), alpha)

h("WORKED EXAMPLE: one-sample t-test, H0: mu = 60 (n=36, xbar=63, s=9)")
t0 <- (63 - 60) / (9 / sqrt(36)); p0 <- 2 * pt(abs(t0), 35, lower.tail = FALSE)
cat(sprintf("t = %.3f, df = 35, %s\n", t0, decide(p0)))

df <- load_data("students.csv"); tut <- load_data("tutoring_scores.csv")
df$method <- factor(df$method)

h("TEST 1: one-sample t-test, H0: mean = 60")
r1 <- t.test(df$exam_score, mu = 60); print(r1)

h("TEST 2: Welch two-sample t-test, A vs C")
ac <- subset(df, method %in% c("A", "C")); ac$method <- droplevels(ac$method)
r2 <- t.test(exam_score ~ method, data = ac); print(r2)
d <- tryCatch(rstatix::cohens_d(ac, exam_score ~ method), error = function(e) NULL)
if (!is.null(d)) print(d)

h("TEST 3: paired t-test, tutoring pre vs post")
r3 <- t.test(tut$post_score, tut$pre_score, paired = TRUE); print(r3)

h("TEST 4: one-proportion test, H0: pass rate = 0.70")
r4 <- prop.test(sum(df$passed), nrow(df), p = 0.70, correct = FALSE); print(r4)
cat(sprintf("z = %.3f; %s\n", sign(r4$estimate - 0.7) * sqrt(r4$statistic), decide(r4$p.value)))

h("TEST 5: Levene test for equal variances (A vs C)")
print(car::leveneTest(exam_score ~ method, data = ac))

save_plot("rejection_region", {
  dfree <- r2$parameter; xs <- seq(-5, 8, length.out = 500)
  plot(xs, dt(xs, dfree), type = "l", lwd = 2, xlab = "t", ylab = "density", main = "Two-sample t-test: A vs C")
  crit <- qt(.975, dfree)
  polygon(c(xs[xs > crit], rev(xs[xs > crit])), c(dt(xs[xs > crit], dfree), rep(0, sum(xs > crit))), col = rgb(1, 0, 0, .3), border = NA)
  polygon(c(xs[xs < -crit], rev(xs[xs < -crit])), c(dt(xs[xs < -crit], dfree), rep(0, sum(xs < -crit))), col = rgb(1, 0, 0, .3), border = NA)
  abline(v = abs(r2$statistic), col = "blue", lwd = 2)
})
save_plot("method_boxplot", boxplot(exam_score ~ method, data = df, col = "lightblue", main = "Scores by method"))

h("INTERPRETATION")
cat("- A p-value is the probability of data this extreme if H0 were true; p < 0.05 rejects H0.\n")
cat(sprintf("- Mean score vs 60: %s\n", decide(r1$p.value)))
cat(sprintf("- Method A vs C: %s\n", decide(r2$p.value)))
cat(sprintf("- Tutoring (paired): %s\n", decide(r3$p.value)))
cat("- Always report an effect size next to the p-value: significance is not importance.\n")
end_topic()
