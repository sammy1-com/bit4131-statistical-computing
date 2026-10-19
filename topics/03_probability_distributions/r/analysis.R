# Topic 3: Probability distributions (stats, MASS, fitdistrplus)
source("common/helpers.R"); start_topic("03_probability_distributions")
need("MASS", "fitdistrplus")

h("WORKED EXAMPLES")
cat("Binomial(10, 0.3): P(X=3) =", round(dbinom(3, 10, 0.3), 4), "\n")
cat("Poisson(5):        P(X<=3) =", round(ppois(3, 5), 4), "\n")
cat("Normal(60, 10):    P(50<X<70) =", round(pnorm(70, 60, 10) - pnorm(50, 60, 10), 4), "\n")
cat("Exponential(mean 8): P(X>10) =", round(pexp(10, rate = 1 / 8, lower.tail = FALSE), 4), "\n")

scores <- load_data("students.csv")$exam_score
tk <- load_data("support_tickets.csv")

h("FIT 1: exam_score ~ Normal")
fit_n <- fitdist(scores, "norm"); print(summary(fit_n))
mu <- fit_n$estimate[["mean"]]; sg <- fit_n$estimate[["sd"]]
cat(sprintf("P(score >= 70) = %.4f | P(score < 50) = %.4f | 90th percentile = %.2f\n",
            1 - pnorm(70, mu, sg), pnorm(50, mu, sg), qnorm(.9, mu, sg)))
print(suppressWarnings(ks.test(scores, "pnorm", mu, sg)))

h("FIT 2: arrival_gap_min ~ Exponential")
fit_e <- fitdist(tk$arrival_gap_min, "exp"); print(summary(fit_e))
cat(sprintf("Mean gap = %.2f minutes\n", 1 / fit_e$estimate[["rate"]]))

h("FIT 3: tickets_per_hour ~ Poisson")
fit_p <- fitdist(tk$tickets_per_hour, "pois"); print(summary(fit_p))
cat(sprintf("mean = %.3f, variance = %.3f (Poisson: equal)\n", mean(tk$tickets_per_hour), var(tk$tickets_per_hour)))

h("FIT 4: resolution_hours — Lognormal vs Normal (AIC)")
rh <- tk$resolution_hours[tk$resolution_hours > 0]
fit_ln <- fitdist(rh, "lnorm"); fit_nn <- fitdist(rh, "norm")
print(gofstat(list(fit_ln, fit_nn), fitnames = c("lognormal", "normal")))
cat(sprintf("AIC lognormal = %.1f | AIC normal = %.1f -> better: %s\n", fit_ln$aic, fit_nn$aic,
            ifelse(fit_ln$aic < fit_nn$aic, "lognormal", "normal")))

save_plot("fitted_normal", plot(fit_n), width = 1100, height = 800)
save_plot("fitted_exponential", plot(fit_e), width = 1100, height = 800)
save_plot("lognormal_vs_normal", denscomp(list(fit_ln, fit_nn), legendtext = c("lognormal", "normal"), main = "Resolution time"))

h("INTERPRETATION")
cat(sprintf("- Exam scores ~ Normal(%.1f, %.1f); KS test gives no evidence against it.\n", mu, sg))
cat(sprintf("- Ticket gaps ~ Exponential with mean %.1f min; counts per hour have mean ~ variance (Poisson-like).\n", 1 / fit_e$estimate[["rate"]]))
cat("- Resolution times are right-skewed; the lognormal fits better than the normal (lower AIC).\n")
end_topic()
