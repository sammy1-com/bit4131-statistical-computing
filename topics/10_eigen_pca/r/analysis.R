# Topic 10: Eigenvalues, eigenvectors and PCA (stats, FactoMineR, factoextra)
source("common/helpers.R"); start_topic("10_eigen_pca")
need("FactoMineR", "factoextra", "ggplot2")

h("WORKED EXAMPLE: A = [[2,1],[1,2]]")
A <- matrix(c(2, 1, 1, 2), 2); e <- eigen(A)
cat("Eigenvalues:", e$values, "(solutions of (2-L)^2 - 1 = 0: L = 3, 1)\n")
cat("Eigenvectors:\n"); print(round(e$vectors, 4))
cat("Check A v = L v:", A %*% e$vectors[, 1], "=", e$values[1] * e$vectors[, 1], "\n")

df <- load_data("students.csv")
cols <- c("study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score")
Z <- scale(df[, cols])

h("PCA BY EIGEN-DECOMPOSITION of the correlation matrix")
ev <- eigen(cor(Z))
cat("Eigenvalues:", round(ev$values, 4), "| sum =", sum(ev$values), "\n")
cat("Explained variance ratio:", round(ev$values / sum(ev$values), 4), "\n")
cat("Cumulative:", round(cumsum(ev$values / sum(ev$values)), 4), "\n")
rownames(ev$vectors) <- cols; colnames(ev$vectors) <- paste0("PC", 1:5)
cat("\nLoadings (eigenvectors):\n"); print(round(ev$vectors, 3))

h("prcomp() cross-check")
pc <- prcomp(df[, cols], scale. = TRUE); print(summary(pc))

h("FactoMineR::PCA")
res <- FactoMineR::PCA(df[, cols], graph = FALSE)
print(round(res$eig, 3))

save_plot("scree_plot", print(factoextra::fviz_eig(res, addlabels = TRUE, main = "Scree plot")))
save_plot("pca_biplot", print(factoextra::fviz_pca_biplot(res, geom.ind = "point", col.ind = factor(df$method), addEllipses = FALSE,
                                                          legend.title = "Method", title = "PCA biplot")), width = 1000, height = 800)

h("INTERPRETATION")
cat(sprintf("- %d component(s) have eigenvalue > 1 (Kaiser rule); first two explain %.0f%% of variance.\n",
            sum(ev$values > 1), sum(ev$values[1:2]) / sum(ev$values) * 100))
top <- names(sort(abs(ev$vectors[, 1]), decreasing = TRUE))[1:2]
cat(sprintf("- PC1 is dominated by %s and %s.\n", top[1], top[2]))
cat("- Eigenvalues sum to the number of standardised variables; each is the variance of its component.\n")
end_topic()
