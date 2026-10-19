# Topic 17: Chi-square and categorical data analysis (stats, vcd, MASS)
source("common/helpers.R"); start_topic("17_chi_square")
need("vcd", "MASS")

h("WORKED EXAMPLE: 2x2 table [[30,20],[20,30]]")
obs <- matrix(c(30, 20, 20, 30), 2, byrow = TRUE)
ex <- outer(rowSums(obs), colSums(obs)) / sum(obs)
cat("Expected counts:\n"); print(ex)
chi <- sum((obs - ex)^2 / ex); cat(sprintf("chi2 = %.2f, df = 1, p = %.4f\n", chi, pchisq(chi, 1, lower.tail = FALSE)))

sv <- load_data("survey.csv")
h("TEST OF INDEPENDENCE: programme vs preferred language")
tab <- table(sv$program, sv$preferred_language, dnn = c("program", "language"))
cat("Observed:\n"); print(tab)
ct <- chisq.test(tab); print(ct)
cat("\nExpected frequencies:\n"); print(round(ct$expected, 1))
cat("\nStandardised (Pearson) residuals:\n"); print(round(ct$residuals, 2))
cat("\nAssociation strength (vcd::assocstats):\n"); print(vcd::assocstats(tab))

h("GOODNESS OF FIT: are the 4 languages equally popular?")
cnt <- table(sv$preferred_language); print(cnt); print(chisq.test(cnt))

df <- load_data("students.csv")
h("2x2: gender vs passed")
t22 <- table(df$gender, df$passed, dnn = c("gender", "passed")); print(t22)
print(chisq.test(t22)); print(fisher.test(t22))

h("LOG-LINEAR MODEL (MASS::loglm)")
print(MASS::loglm(~ program + language, data = tab))

save_plot("chi_square_bars", barplot(t(tab), beside = TRUE, legend.text = TRUE, col = c("#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"),
                                     main = "Language preference by programme", ylab = "Respondents"))
save_plot("mosaic_residuals", vcd::mosaic(tab, shade = TRUE, legend = TRUE, main = "Mosaic plot (shaded by residuals)"))

h("INTERPRETATION")
cat(sprintf("- chi2 = %.1f, df = %d, p = %.2g: %s\n", ct$statistic, ct$parameter, ct$p.value,
            ifelse(ct$p.value < .05, "programme and language preference are associated.", "no evidence of association.")))
cramer <- sqrt(unname(ct$statistic) / (sum(tab) * (min(dim(tab)) - 1)))
cat(sprintf("- Cramer's V = %.2f (0 none, 1 perfect).\n", cramer))
cat(sprintf("- Smallest expected count = %.1f (rule of thumb: >= 5).\n", min(ct$expected)))
end_topic()
