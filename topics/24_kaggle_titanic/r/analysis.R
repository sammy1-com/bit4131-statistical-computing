# Kaggle Titanic in R: missing data (T22), chi-square (T17), non-parametric (T19), logistic regression (T18)
source("common/helpers.R"); start_topic("24_kaggle_titanic")
df <- load_data("kaggle/titanic.csv")
h("DATASET: Kaggle Titanic (891 passengers)")
print(summary(df[, c("Survived", "Pclass", "Age", "Fare")]))

h("MISSING DATA (Topic 22)")
print(round(colMeans(is.na(df)) * 100, 1)[colMeans(is.na(df)) > 0])
df$Age_missing <- is.na(df$Age)
cat("Survival when Age missing:", round(mean(df$Survived[df$Age_missing]), 3), "| observed:", round(mean(df$Survived[!df$Age_missing]), 3), "\n")
med <- ave(df$Age, df$Pclass, df$Sex, FUN = function(x) median(x, na.rm = TRUE))
df$Age_imp <- ifelse(is.na(df$Age), med, df$Age)
cat(sprintf("Age SD before = %.2f, after group-median imputation = %.2f\n", sd(df$Age, na.rm = TRUE), sd(df$Age_imp)))

h("CHI-SQUARE (Topic 17)")
tab <- table(df$Pclass, df$Survived, dnn = c("Pclass", "Survived")); print(tab)
ct <- chisq.test(tab); print(ct)
cat(sprintf("Cramer's V = %.3f\n", sqrt(unname(ct$statistic) / (sum(tab) * (min(dim(tab)) - 1)))))
print(chisq.test(table(df$Sex, df$Survived)))

h("NON-PARAMETRIC (Topic 19): Fare by survival")
cat("Median fare: survived =", median(df$Fare[df$Survived == 1]), " died =", median(df$Fare[df$Survived == 0]), "\n")
print(wilcox.test(Fare ~ Survived, data = df))

h("LOGISTIC REGRESSION (Topic 18)")
m <- glm(Survived ~ factor(Pclass) + Sex + Age_imp + SibSp + Fare, data = df, family = binomial)
print(summary(m))
orr <- exp(cbind(OR = coef(m), suppressMessages(confint.default(m)))); cat("\nOdds ratios:\n"); print(round(orr, 3))
set.seed(4131); idx <- sample(nrow(df), .7 * nrow(df))
mt <- glm(Survived ~ factor(Pclass) + Sex + Age_imp + SibSp + Fare, data = df[idx, ], family = binomial)
pr <- predict(mt, df[-idx, ], type = "response"); pred <- as.integer(pr > .5); y <- df$Survived[-idx]
cm <- table(Actual = y, Predicted = pred); print(cm)
auc <- { r <- rank(pr); (sum(r[y == 1]) - sum(y == 1) * (sum(y == 1) + 1) / 2) / (sum(y == 1) * sum(y == 0)) }
cat(sprintf("Accuracy = %.3f, AUC = %.3f\n", mean(pred == y), auc))

save_plot("titanic_overview", {
  par(mfrow = c(1, 3))
  barplot(tapply(df$Survived, df$Pclass, mean), main = "Survival rate by class", col = "#1f78b4", ylab = "Rate")
  barplot(tapply(df$Survived, df$Sex, mean), main = "Survival rate by sex", col = "#33a02c")
  o <- order(pr, decreasing = TRUE); plot(c(0, cumsum(y[o] == 0) / sum(y == 0)), c(0, cumsum(y[o] == 1) / sum(y == 1)), type = "l", xlab = "FPR", ylab = "TPR", main = sprintf("ROC (AUC %.2f)", auc)); abline(0, 1, lty = 2)
}, width = 1500, height = 500)

h("INTERPRETATION")
cat(sprintf("- Age is missing for %.0f%% of passengers; group-median imputation keeps every row.\n", mean(is.na(df$Age)) * 100))
cat(sprintf("- Class (chi2 = %.1f, p = %.2g) and sex are strongly associated with survival.\n", ct$statistic, ct$p.value))
cat(sprintf("- Being male multiplies the odds of survival by %.2f; 3rd class by %.2f (vs 1st).\n", orr["Sexmale", "OR"], orr["factor(Pclass)3", "OR"]))
cat(sprintf("- Held-out accuracy %.0f%%, AUC %.2f.\n", mean(pred == y) * 100, auc))
end_topic()
