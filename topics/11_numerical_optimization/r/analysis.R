# Topic 11: Numerical optimization (stats::optim, optimx, nloptr)
source("common/helpers.R"); start_topic("11_numerical_optimization")
need("optimx")

h("WORKED EXAMPLE: minimise f(x) = (x-3)^2 + 2")
f <- function(x) (x - 3)^2 + 2
o <- optimize(f, c(-10, 10))
cat(sprintf("Analytic x* = 3, f = 2 | optimize(): x* = %.6f, f = %.6f\n", o$minimum, o$objective))

h("OBJECTIVE: Rosenbrock (true minimum at (1,1))")
fr  <- function(p) 100 * (p[2] - p[1]^2)^2 + (1 - p[1])^2
grr <- function(p) c(-400 * p[1] * (p[2] - p[1]^2) - 2 * (1 - p[1]), 200 * (p[2] - p[1]^2))
x0 <- c(-1.5, 2)
for (m in c("Nelder-Mead", "BFGS", "CG")) {
  r <- optim(x0, fr, if (m == "Nelder-Mead") NULL else grr, method = m, control = list(maxit = 5000))
  cat(sprintf("%-12s x = (%.5f, %.5f), f = %.2e, fn evals = %d, convergence code = %d (0 = success)\n",
              m, r$par[1], r$par[2], r$value, r$counts[1], r$convergence))
}
cat("\noptimx comparison:\n")
print(optimx::optimx(x0, fr, grr, method = c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B")))

h("DATASET: MLE of Lognormal for ticket resolution times")
tk <- load_data("support_tickets.csv")$resolution_hours
nll <- function(th) -sum(dlnorm(tk, meanlog = th[1], sdlog = exp(th[2]), log = TRUE))
mle <- optim(c(1, 0), nll, method = "BFGS")
cat(sprintf("Numerical MLE: mu = %.4f, sigma = %.4f | closed form: mu = %.4f, sigma = %.4f\n",
            mle$par[1], exp(mle$par[2]), mean(log(tk)), sqrt(mean((log(tk) - mean(log(tk)))^2))))

h("CONSTRAINED: maximise 4*sqrt(h1) + 3*sqrt(h2) subject to h1 + h2 = 20")
obj <- function(h1) -(4 * sqrt(h1) + 3 * sqrt(20 - h1))     # substitute h2 = 20 - h1
oc <- optimize(obj, c(0.01, 19.99))
cat(sprintf("Optimal split: h1 = %.3f, h2 = %.3f, benefit = %.4f (Lagrange: 12.8 / 7.2)\n", oc$minimum, 20 - oc$minimum, -oc$objective))
if (opt("nloptr")) {
  nl <- tryCatch(nloptr::nloptr(c(10, 10), function(h) -(4 * sqrt(h[1]) + 3 * sqrt(h[2])),
                                eval_g_eq = function(h) h[1] + h[2] - 20, lb = c(0.01, 0.01), ub = c(20, 20),
                                opts = list(algorithm = "NLOPT_LN_COBYLA", xtol_rel = 1e-8, maxeval = 5000)), error = function(e) NULL)
  if (!is.null(nl)) cat(sprintf("nloptr (COBYLA) cross-check: h1 = %.3f, h2 = %.3f\n", nl$solution[1], nl$solution[2]))
}

save_plot("optimization_surface", {
  gx <- seq(-2, 2, length.out = 200); gy <- seq(-1, 3, length.out = 200)
  z <- outer(gx, gy, Vectorize(function(a, b) fr(c(a, b))))
  contour(gx, gy, log10(z + 1), nlevels = 25, xlab = "x", ylab = "y", main = "Rosenbrock (log10 scale) and optima")
  points(1, 1, pch = 8, cex = 2, col = "red"); points(x0[1], x0[2], pch = 17, col = "blue")
  legend("topright", c("true minimum", "start"), pch = c(8, 17), col = c("red", "blue"), bty = "n")
})
save_plot("likelihood_slice", {
  mus <- seq(mle$par[1] - .4, mle$par[1] + .4, length.out = 100)
  plot(mus, sapply(mus, function(m) nll(c(m, mle$par[2]))), type = "l", xlab = "mu (sigma fixed)", ylab = "Negative log-likelihood", main = "Likelihood slice")
  abline(v = mle$par[1], col = "red", lty = 2)
})

h("INTERPRETATION")
cat("- All methods reach (1,1); gradient-based methods need fewer function evaluations than Nelder-Mead.\n")
cat("- Numerical MLE agrees with the closed form: optimisation is how MLE is computed when no formula exists.\n")
cat("- The constrained optimum puts more hours into the higher-benefit subject (Lagrange: 12.8 / 7.2).\n")
end_topic()
