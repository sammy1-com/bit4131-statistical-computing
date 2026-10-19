# Topic 2: Probability and conditional probability (base R, stats)
# Note: the 'prob' package has been archived on CRAN; expand.grid() from base R builds the same sample spaces.
source("common/helpers.R"); start_topic("02_probability_conditional")

h("WORKED EXAMPLE: two fair dice")
S <- expand.grid(d1 = 1:6, d2 = 1:6)
S$sum7 <- (S$d1 + S$d2) == 7
S$has6 <- S$d1 == 6 | S$d2 == 6
cat("P(sum = 7)               =", round(mean(S$sum7), 4), "\n")
cat("P(at least one 6)        =", round(mean(S$has6), 4), "\n")
cat("P(sum=7 AND a 6)         =", round(mean(S$sum7 & S$has6), 4), "\n")
cat("P(sum=7 | at least one 6)=", round(mean(S$sum7 & S$has6) / mean(S$has6), 4), "(= 2/11)\n")

df <- load_data("students.csv")
h("DATASET: probabilities from the students table")
p_pass <- mean(df$passed); p_B <- mean(df$method == "B")
cat(sprintf("P(pass) = %.4f | P(method B) = %.4f | P(Statistics) = %.4f\n", p_pass, p_B, mean(df$program == "Statistics")))
tab <- table(df$method, df$passed, dnn = c("method", "passed"))
cat("\nContingency table:\n"); print(addmargins(tab))
cond <- tapply(df$passed, df$method, mean)
cat("\nP(pass | method):\n"); print(round(cond, 4))
cat("\nrow-wise proportions:\n"); print(round(prop.table(tab, 1), 4))
p_joint <- mean(df$method == "B" & df$passed == 1)
cat(sprintf("\nP(B and pass) = %.4f vs P(B)P(pass) = %.4f -> not independent\n", p_joint, p_B * p_pass))
bayes <- cond[["C"]] * mean(df$method == "C") / p_pass
cat(sprintf("Bayes: P(method C | pass) = %.4f\n", bayes))

h("SIMULATION: P(sum=7 | at least one 6), 200,000 rolls")
N <- 200000
d1 <- sample(1:6, N, replace = TRUE); d2 <- sample(1:6, N, replace = TRUE)
has6 <- d1 == 6 | d2 == 6
est <- sum((d1 + d2 == 7) & has6) / sum(has6)
cat(sprintf("Simulated = %.4f | Exact = %.4f\n", est, 2 / 11))
running <- cumsum((d1 + d2 == 7) & has6) / pmax(cumsum(has6), 1)

save_plot("simulation_convergence", {
  plot(100:N, running[100:N], type = "l", log = "x", xlab = "Rolls (log scale)", ylab = "Estimated probability",
       main = "Simulation converges to exact 2/11")
  abline(h = 2 / 11, col = "red", lty = 2)
})
save_plot("conditional_probability", {
  barplot(cond, col = "lightblue", ylim = c(0, 1), main = "P(pass | method)", ylab = "Probability")
  abline(h = p_pass, col = "red", lty = 2)
})

h("INTERPRETATION")
cat(sprintf("- Overall P(pass) = %.2f, but it ranges from %.2f (A) to %.2f (C): conditioning changes the probability.\n",
            p_pass, cond[["A"]], cond[["C"]]))
cat("- Simulation agrees with the exact result 2/11.\n")
end_topic()
