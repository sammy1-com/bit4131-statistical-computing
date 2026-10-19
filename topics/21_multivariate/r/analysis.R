# Topic 21: Multivariate statistical analysis (MASS, FactoMineR, vegan, cluster)
source("common/helpers.R"); start_topic("21_multivariate")
need("MASS", "cluster")

df <- load_data("students.csv"); df$method <- factor(df$method)
cols <- c("study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score")
X <- df[, cols]

h("WORKED EXAMPLE: Mahalanobis distance in 2D")
S <- matrix(c(4, 2, 2, 3), 2); pt <- c(2, 2)
d2 <- as.numeric(t(pt) %*% solve(S) %*% pt)
cat(sprintf("d^2 = %.4f, d = %.4f | Euclidean distance = %.4f\n", d2, sqrt(d2), sqrt(sum(pt^2))))
cat("mahalanobis():", mahalanobis(pt, c(0, 0), S), "\n")

h("MULTIVARIATE SUMMARY")
cat("Mean vector:\n"); print(round(colMeans(X), 2))
cat("\nCovariance matrix:\n"); print(round(cov(X), 2))
cat(sprintf("\nGeneralised variance |S| = %.4g | total variance = %.2f\n", det(cov(X)), sum(diag(cov(X)))))

h("MULTIVARIATE OUTLIERS (Mahalanobis, chi-square cut-off)")
md2 <- mahalanobis(X, colMeans(X), cov(X)); cut <- qchisq(.999, ncol(X))
out <- df$student_id[md2 > cut]
cat(sprintf("Cut-off = %.2f; %d outlier(s): student_id %s\n", cut, length(out), paste(out, collapse = ", ")))

h("MANOVA: do the 5 measures jointly differ by method?")
mv <- manova(as.matrix(X) ~ method, data = df)
print(summary(mv, test = "Wilks"))

h("K-MEANS CLUSTERING (standardised)")
Z <- scale(X)
sil <- numeric(0)
for (k in 2:6) {
  km <- kmeans(Z, k, nstart = 25)
  sil[as.character(k)] <- mean(cluster::silhouette(km$cluster, dist(Z))[, 3])
  cat(sprintf("k = %d: tot.withinss = %.1f, silhouette = %.3f\n", k, km$tot.withinss, sil[as.character(k)]))
}
kbest <- as.integer(names(which.max(sil)))
km <- kmeans(Z, kbest, nstart = 25); df$cluster <- km$cluster
cat("\nBest k by silhouette =", kbest, "\nCluster profiles (means):\n"); print(round(aggregate(X, list(cluster = df$cluster), mean), 2))
cat("Cluster vs method table:\n"); print(table(df$cluster, df$method))

h("HIERARCHICAL CLUSTERING (Ward)")
hc <- hclust(dist(Z), method = "ward.D2"); cl <- cutree(hc, kbest); cat("Ward cluster sizes:", table(cl), "\n")

if (opt("vegan")) {
  h("vegan: permutational MANOVA (adonis2)")
  print(tryCatch(vegan::adonis2(dist(Z) ~ method, data = df, permutations = 499), error = function(e) NULL))
}

pc <- prcomp(Z)
save_plot("multivariate_clusters", {
  par(mfrow = c(1, 2))
  plot(pc$x[, 1], pc$x[, 2], col = df$cluster, pch = 16, xlab = "PC1", ylab = "PC2", main = "K-means clusters in PCA space")
  plot(hc, labels = FALSE, hang = -1, main = "Ward dendrogram"); rect.hclust(hc, k = kbest, border = "red")
}, width = 1300)
save_plot("mahalanobis", {
  hist(md2, breaks = 30, freq = FALSE, col = "lightblue", border = "white", main = "Mahalanobis distances follow chi-square", xlab = "d^2")
  curve(dchisq(x, ncol(X)), add = TRUE, col = "red", lwd = 2); abline(v = cut, lty = 2)
})

h("INTERPRETATION")
wp <- summary(mv, test = "Wilks")$stats[1, "Pr(>F)"]
cat(sprintf("- MANOVA Wilks p = %.2g: %s\n", wp, ifelse(wp < .05, "methods differ on the combined profile.", "no joint difference.")))
cat(sprintf("- K-means picks k = %d (silhouette %.2f); a silhouette below ~0.25 means weak cluster structure.\n", kbest, max(sil)))
cat(sprintf("- %d multivariate outlier(s) flagged by Mahalanobis distance.\n", length(out)))
end_topic()
