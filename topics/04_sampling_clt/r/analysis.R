# Topic 4: Sampling distributions and the Central Limit Theorem (stats, sampling, ggplot2)
source("common/helpers.R"); start_topic("04_sampling_clt")
need("ggplot2")
if (opt("sampling")) library(sampling)  # optional: srswor() for sampling without replacement

pop <- load_data("support_tickets.csv")$resolution_hours
mu <- mean(pop); sigma <- sqrt(mean((pop - mean(pop))^2))   # population SD (divide by N)

h("WORKED EXAMPLE: standard error")
cat(sprintf("mu = %.3f, sigma = %.3f, skewness = %.2f\n", mu, sigma, skewness(pop)))
cat(sprintf("n = 30: SE = sigma/sqrt(n) = %.4f\n", sigma / sqrt(30)))

h("SAMPLING DISTRIBUTION OF THE MEAN (5,000 samples each)")
sizes <- c(2, 5, 30, 100, 500)
res <- lapply(sizes, function(n) replicate(5000, mean(sample(pop, n, replace = TRUE))))
cat(sprintf("%5s %14s %12s %10s %7s\n", "n", "mean of means", "SD of means", "theory SE", "skew"))
for (i in seq_along(sizes)) {
  m <- res[[i]]
  cat(sprintf("%5d %14.4f %12.4f %10.4f %7.2f\n", sizes[i], mean(m), sd(m), sigma / sqrt(sizes[i]), skewness(m)))
}

d <- do.call(rbind, lapply(seq_along(sizes), function(i) data.frame(n = factor(sizes[i], levels = sizes), mean = res[[i]])))
p <- ggplot(d, aes(mean)) + geom_histogram(bins = 40, fill = "steelblue", colour = "white") +
  facet_wrap(~n, scales = "free", labeller = label_both) +
  labs(title = "CLT: sample means of a skewed population", x = "Sample mean", y = "Count")
save_plot("clt_visualization", print(p), width = 1200, height = 800)
save_plot("population", hist(pop, breaks = 40, col = "orange", border = "white", main = "Population (right-skewed)", xlab = "Resolution hours"))
se_df <- data.frame(n = sizes, simulated = sapply(res, sd), theory = sigma / sqrt(sizes))
save_plot("standard_error", {
  plot(se_df$n, se_df$simulated, log = "x", type = "b", pch = 16, xlab = "n", ylab = "Standard error", main = "SE vs n")
  lines(se_df$n, se_df$theory, col = "red", lty = 2)
  legend("topright", c("Simulated", "sigma/sqrt(n)"), col = c("black", "red"), lty = c(1, 2), bty = "n")
})

h("INTERPRETATION")
cat(sprintf("- Population skewness %.2f; sample-mean skewness falls from %.2f (n=2) to %.2f (n=100).\n",
            skewness(pop), skewness(res[[1]]), skewness(res[[4]])))
cat("- Mean of sample means equals mu; their SD matches sigma/sqrt(n); quadrupling n halves the SE.\n")
end_topic()
