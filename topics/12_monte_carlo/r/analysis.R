# Topic 12: Random numbers and Monte Carlo methods (stats, MonteCarlo, boot)
source("common/helpers.R"); start_topic("12_monte_carlo")

h("WORKED EXAMPLE: estimate pi")
N <- 100000
x <- runif(N); y <- runif(N); inside <- x^2 + y^2 <= 1
pi_hat <- 4 * mean(inside)
cat(sprintf("pi-hat = %.5f (true %.5f); MC standard error = %.5f\n", pi_hat, pi, 4 * sqrt(mean(inside) * (1 - mean(inside)) / N)))

h("INVERSE-TRANSFORM SAMPLING: Exponential(rate 0.125)")
u <- runif(50000); s <- -log(1 - u) / 0.125
cat(sprintf("sample mean = %.3f (theory 8); sample variance = %.2f (theory 64)\n", mean(s), var(s)))

h("MONTE CARLO INTEGRATION: integral of exp(-x^2) over [0,1]")
v <- exp(-runif(200000)^2)
cat(sprintf("MC = %.5f +/- %.5f | integrate() = %.5f\n", mean(v), 1.96 * sd(v) / sqrt(length(v)), integrate(function(t) exp(-t^2), 0, 1)$value))

df <- load_data("students.csv")
h("DATASET: simulate a class of 40 students (resampled from the data)")
cm <- replicate(20000, mean(sample(df$exam_score, 40, replace = TRUE)))
cat(sprintf("P(class mean > 65) = %.4f | mean of class means = %.2f, SD = %.2f\n", mean(cm > 65), mean(cm), sd(cm)))
pass_rate <- mean(df$passed)
fails <- rbinom(20000, 40, 1 - pass_rate)
cat(sprintf("Simulated P(more than 12 of 40 fail) = %.4f | exact binomial = %.4f\n", mean(fails > 12), 1 - pbinom(12, 40, 1 - pass_rate)))

if (opt("MonteCarlo")) {
  h("MonteCarlo package: grid of sample sizes")
  pi_sim <- function(n) { a <- runif(n); b <- runif(n); list("pi_hat" = 4 * mean(a^2 + b^2 <= 1)) }
  mc <- tryCatch(MonteCarlo::MonteCarlo(pi_sim, nrep = 500, param_list = list(n = c(100, 1000, 10000)), print_progress = FALSE), error = function(e) NULL)
  if (!is.null(mc)) print(summary(mc))
}

save_plot("monte_carlo_convergence", {
  par(mfrow = c(1, 2))
  run <- 4 * cumsum(inside) / seq_len(N)
  plot(50:N, run[50:N], type = "l", log = "x", xlab = "Number of points", ylab = "Estimate of pi", main = "Convergence")
  abline(h = pi, col = "red", lty = 2)
  idx <- sample(N, 3000)
  plot(x[idx], y[idx], pch = 16, cex = .4, col = ifelse(inside[idx], "steelblue", "red"), asp = 1, xlab = "x", ylab = "y", main = "Random points")
  t <- seq(0, pi / 2, length.out = 100); lines(cos(t), sin(t))
}, width = 1200)
save_plot("mc_error_rate", {
  ns <- round(10^seq(2, 5, length.out = 20))
  err <- sapply(ns, function(n) mean(replicate(30, abs(4 * mean(runif(n)^2 + runif(n)^2 <= 1) - pi))))
  plot(ns, err, log = "xy", type = "b", pch = 16, xlab = "N", ylab = "Mean absolute error", main = "Error shrinks like 1/sqrt(N)")
  lines(ns, 1.6 / sqrt(ns), col = "red", lty = 2)
})

h("INTERPRETATION")
cat(sprintf("- With %d points pi is estimated as %.4f (error %.4f).\n", N, pi_hat, abs(pi_hat - pi)))
cat("- Inverse-transform sampling turns uniforms into any distribution with an invertible CDF.\n")
cat("- Error falls like 1/sqrt(N): 100x more simulations buy only 10x more accuracy.\n")
end_topic()
