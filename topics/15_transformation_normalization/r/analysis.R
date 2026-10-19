# Topic 15: Data transformation and normalization (dplyr, caret, recipes)
source("common/helpers.R"); start_topic("15_transformation_normalization")
need("dplyr", "caret", "recipes", "MASS")

h("WORKED EXAMPLE: data = c(10, 20, 30, 40, 100)")
d <- c(10, 20, 30, 40, 100)
cat("mean =", mean(d), "| sd =", round(sd(d), 4), "\n")
cat("z-score :", round((d - mean(d)) / sd(d), 3), "\n")
cat("min-max :", round((d - min(d)) / (max(d) - min(d)), 3), "\n")
cat("log10   :", round(log10(d), 3), "\n")

tk <- load_data("support_tickets.csv"); r <- tk$resolution_hours
df <- load_data("students.csv")

h("DATASET 1: skewed variable - ticket resolution_hours")
bc <- caret::BoxCoxTrans(r)
tr <- data.frame(original = r, log = log(r), sqrt = sqrt(r), boxcox = predict(bc, r))
print(round(sapply(tr, skewness), 4))
cat(sprintf("Box-Cox lambda = %.3f (0 means log)\n", bc$lambda))

h("DATASET 2: scaling numeric predictors (students)")
cols <- c("study_hours", "attendance_pct", "exam_score"); X <- df[, cols]
std <- as.data.frame(predict(caret::preProcess(X, method = c("center", "scale")), X))
mm  <- as.data.frame(predict(caret::preProcess(X, method = "range"), X))
cat("Original:\n");     print(round(sapply(X,   function(v) c(mean = mean(v), sd = sd(v), min = min(v), max = max(v))), 2))
cat("\nStandardised:\n"); print(round(sapply(std, function(v) c(mean = mean(v), sd = sd(v), min = min(v), max = max(v))), 3))
cat("\nMin-max:\n");      print(round(sapply(mm,  function(v) c(mean = mean(v), sd = sd(v), min = min(v), max = max(v))), 3))

h("recipes: reproducible preprocessing pipeline")
rec <- recipes::recipe(exam_score ~ study_hours + attendance_pct, data = df) %>% recipes::step_normalize(recipes::all_predictors())
baked <- recipes::bake(recipes::prep(rec, training = df), new_data = NULL)
print(round(sapply(baked, function(v) c(mean = mean(v), sd = sd(v))), 3))

save_plot("skew_transformations", {
  par(mfrow = c(2, 2))
  for (nm in names(tr)) hist(tr[[nm]], breaks = 30, col = "lightblue", border = "white", main = sprintf("%s (skew %.2f)", nm, skewness(tr[[nm]])), xlab = "")
}, width = 1100, height = 800)
save_plot("scaling_comparison", {
  par(mfrow = c(1, 3))
  boxplot(X, main = "Original"); boxplot(std, main = "Standardised"); boxplot(mm, main = "Min-max")
}, width = 1200)

h("INTERPRETATION")
sk <- sapply(tr, skewness); best <- names(which.min(abs(sk[-1])))
cat(sprintf("- Resolution time skewness %.2f; the %s transform reduces it to %.2f.\n", sk[["original"]], best, sk[[best]]))
cat("- z-scores give mean 0 / SD 1; min-max maps to [0,1] but is sensitive to outliers.\n")
cat("- Scale before distance-based methods (k-means, PCA); fit scalers on training data only to avoid leakage.\n")
end_topic()
