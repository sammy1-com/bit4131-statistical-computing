# Topic 18: Logistic regression (stats::glm, MASS, broom)
source("common/helpers.R"); start_topic("18_logistic_regression")
need("MASS", "broom")

h("WORKED EXAMPLE: logit arithmetic (b0 = -4, b1 = 0.5)")
for (x in c(4, 8, 12)) {
  eta <- -4 + 0.5 * x
  cat(sprintf("x = %2d: log-odds = %+.1f, odds = %.3f, P = %.3f\n", x, eta, exp(eta), plogis(eta)))
}
cat("Odds ratio per unit = exp(0.5) =", round(exp(0.5), 3), "\n")

df <- load_data("students.csv")
fml <- passed ~ study_hours + attendance_pct + sleep_hours + prior_gpa

h("LOGISTIC REGRESSION")
model <- glm(fml, data = df, family = binomial); print(summary(model))
h("ODDS RATIOS with 95% CI (broom)")
or <- broom::tidy(model, exponentiate = TRUE, conf.int = TRUE); print(or)
cat(sprintf("\nMcFadden pseudo R2 = %.3f\n", 1 - model$deviance / model$null.deviance))
cat("\nStepwise AIC selection (MASS::stepAIC):\n"); print(MASS::stepAIC(model, trace = 0)$formula)

h("CLASSIFICATION PERFORMANCE (70/30 train-test)")
idx <- sample(nrow(df), 0.7 * nrow(df)); tr <- df[idx, ]; te <- df[-idx, ]
m2 <- glm(fml, data = tr, family = binomial)
prob <- predict(m2, te, type = "response"); pred <- as.integer(prob >= 0.5)
cm <- table(Actual = te$passed, Predicted = factor(pred, levels = 0:1)); print(cm)
acc <- sum(diag(cm)) / sum(cm); prec <- cm[2, 2] / sum(cm[, 2]); rec <- cm[2, 2] / sum(cm[2, ])
auc <- function(score, y) { rk <- rank(score); (sum(rk[y == 1]) - sum(y == 1) * (sum(y == 1) + 1) / 2) / (sum(y == 1) * sum(y == 0)) }
a <- auc(prob, te$passed)
cat(sprintf("Accuracy = %.3f | Precision = %.3f | Recall = %.3f | F1 = %.3f | AUC = %.3f\n", acc, prec, rec, 2 * prec * rec / (prec + rec), a))
cat(sprintf("Baseline (always 'pass') accuracy = %.3f\n", max(mean(te$passed), 1 - mean(te$passed))))

new <- data.frame(study_hours = c(6, 12, 18), attendance_pct = c(70, 85, 95), sleep_hours = c(6, 7, 8), prior_gpa = c(2.3, 3.0, 3.6))
cat("\nPredicted pass probability for three hypothetical students:\n"); print(round(predict(model, new, type = "response"), 3))

save_plot("roc_curve", {
  ths <- sort(unique(c(0, prob, 1)))
  tpr <- sapply(ths, function(t) mean(prob[te$passed == 1] >= t)); fpr <- sapply(ths, function(t) mean(prob[te$passed == 0] >= t))
  plot(fpr, tpr, type = "l", lwd = 2, xlab = "False positive rate", ylab = "True positive rate", main = sprintf("ROC curve (AUC = %.2f)", a))
  abline(0, 1, lty = 2)
})
save_plot("probability_curve", {
  xs <- seq(min(df$study_hours), max(df$study_hours), length.out = 100)
  g <- data.frame(study_hours = xs, attendance_pct = mean(df$attendance_pct), sleep_hours = mean(df$sleep_hours), prior_gpa = mean(df$prior_gpa))
  plot(df$study_hours, df$passed, pch = 16, col = rgb(0, 0, 1, .25), xlab = "Study hours", ylab = "P(pass)", main = "Fitted probability curve")
  lines(xs, predict(model, g, type = "response"), col = "red", lwd = 2)
})

h("INTERPRETATION")
sig <- or[or$term != "(Intercept)" & or$p.value < .05, ]
for (i in seq_len(nrow(sig))) cat(sprintf("- %s: odds ratio %.2f -> each unit multiplies the odds of passing by %.2f (p = %.2g).\n", sig$term[i], sig$estimate[i], sig$estimate[i], sig$p.value[i]))
cat(sprintf("- Test accuracy %.0f%% vs baseline %.0f%%: accuracy alone is misleading with imbalanced classes; AUC = %.2f and recall = %.2f are more informative.\n",
            acc * 100, max(mean(te$passed), 1 - mean(te$passed)) * 100, a, rec))
end_topic()
