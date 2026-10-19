# Topic 22: Missing-data analysis and imputation (mice, Amelia, VIM)
source("common/helpers.R"); start_topic("22_missing_data")
need("mice", "VIM", "Amelia")

full <- load_data("students.csv"); miss <- load_data("students_missing.csv")
cols <- c("study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score")
M <- miss[, cols]; mask <- is.na(M)

h("WORKED EXAMPLE: mean imputation on c(4, 6, NA, 10)")
v <- c(4, 6, NA, 10); cat("observed mean =", mean(v, na.rm = TRUE), "-> imputed:", ifelse(is.na(v), mean(v, na.rm = TRUE), v), "\n")
cat("Mean imputation keeps the mean but shrinks the variance.\n")

h("MISSING-DATA SUMMARY")
print(data.frame(n_missing = colSums(mask), pct_missing = round(colMeans(mask) * 100, 1)))
cat(sprintf("Rows with >= 1 missing: %d of %d | complete cases: %d\n", sum(rowSums(mask) > 0), nrow(M), sum(complete.cases(M))))
cat("\nMissing-data patterns (mice::md.pattern):\n")
pat <- mice::md.pattern(M, plot = FALSE)
print(pat)

h("MECHANISM: is missingness related to OBSERVED prior_gpa? (Fisher exact)")
low <- miss$prior_gpa < 2.7; ok <- !is.na(low)
for (var in c("attendance_pct", "sleep_hours")) {
  m <- is.na(miss[[var]]); t <- table(low[ok], m[ok])
  cat(sprintf("%-15s missing %.0f%% when GPA<2.7 vs %.0f%% otherwise; Fisher p = %.3g\n", var, mean(m[ok & low]) * 100, mean(m[ok & !low]) * 100, fisher.test(t)$p.value))
}

h("IMPUTATION METHODS vs TRUTH")
rmse <- function(imp, var) sqrt(mean((imp[[var]][mask[, var]] - full[[var]][mask[, var]])^2))
mean_imp <- as.data.frame(lapply(M, function(c) { c[is.na(c)] <- mean(c, na.rm = TRUE); c }))
imp <- mice::mice(M, m = 5, method = "pmm", seed = 4131, printFlag = FALSE)
mice1 <- mice::complete(imp, 1)
am <- tryCatch(Amelia::amelia(M, m = 5, p2s = 0), error = function(e) NULL)
sets <- list(Mean = mean_imp, `mice (pmm)` = mice1)
if (!is.null(am)) sets[["Amelia"]] <- am$imputations[[1]]
res <- do.call(rbind, lapply(names(sets), function(n) data.frame(method = n,
  RMSE_study_hours = rmse(sets[[n]], "study_hours"), RMSE_attendance = rmse(sets[[n]], "attendance_pct"),
  RMSE_prior_gpa = rmse(sets[[n]], "prior_gpa"), SD_attendance = sd(sets[[n]]$attendance_pct))))
print(res, digits = 4)
cat(sprintf("\nTrue SD of attendance = %.3f | after deletion = %.3f\n", sd(full$attendance_pct), sd(M$attendance_pct, na.rm = TRUE)))

h("BEFORE vs AFTER: regression slope of exam_score on study_hours")
sl <- function(d) coef(lm(exam_score ~ study_hours, d))[["study_hours"]]
cat(sprintf("Truth %.4f | complete cases %.4f\n", sl(full), sl(M)))
for (n in names(sets)) cat(sprintf("%-12s %.4f\n", n, sl(sets[[n]])))
pooled <- summary(mice::pool(with(imp, lm(exam_score ~ study_hours))))
cat("\nmice pooled estimate (Rubin's rules, 5 imputations):\n"); print(pooled)

save_plot("missing_aggr", VIM::aggr(M, numbers = TRUE, sortVars = TRUE, cex.axis = .7, gap = 2), width = 1100)
save_plot("imputation_comparison", {
  par(mfrow = c(1, 2))
  hist(full$attendance_pct, breaks = 25, col = rgb(0, 0, 1, .4), freq = FALSE, main = "Mean imputation", xlab = "Attendance")
  hist(mean_imp$attendance_pct, breaks = 25, col = rgb(1, 0, 0, .4), freq = FALSE, add = TRUE)
  hist(full$attendance_pct, breaks = 25, col = rgb(0, 0, 1, .4), freq = FALSE, main = "mice (pmm)", xlab = "Attendance")
  hist(mice1$attendance_pct, breaks = 25, col = rgb(1, 0, 0, .4), freq = FALSE, add = TRUE)
}, width = 1200)

h("INTERPRETATION")
cat(sprintf("- Listwise deletion would discard %.0f%% of students: wasteful and potentially biased.\n", (1 - sum(complete.cases(M)) / nrow(M)) * 100))
cat(sprintf("- Mean imputation shrinks the SD of attendance to %.2f (true %.2f), understating uncertainty.\n", res$SD_attendance[res$method == "Mean"], sd(full$attendance_pct)))
cat("- Model-based imputation helps most for variables that are related to the others; compare the RMSE table above per variable.\n")
cat("- Multiple imputation (mice, Amelia) carries imputation uncertainty into the final estimates via Rubin's rules.\n")
end_topic()
