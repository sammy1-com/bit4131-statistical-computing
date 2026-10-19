# Topic 9: Matrix algebra in statistics (base R, Matrix)
source("common/helpers.R"); start_topic("09_matrix_algebra")
need("Matrix")

A <- matrix(c(2, 1, 1, 3), 2, byrow = TRUE); B <- matrix(c(1, 2, 0, 1), 2, byrow = TRUE)

h("WORKED EXAMPLE: A = [[2,1],[1,3]], B = [[1,2],[0,1]]")
cat("A + B =\n"); print(A + B)
cat("A %*% B =\n"); print(A %*% B)
cat("B %*% A =\n"); print(B %*% A); cat("(not commutative)\n")
cat("t(A) =\n"); print(t(A))
cat("det(A) =", det(A), "\n")
cat("solve(A) =\n"); print(solve(A))
cat("A %*% solve(A) =\n"); print(round(A %*% solve(A), 10))

h("SOLVING LINEAR EQUATIONS: 2x + y = 5, x + 3y = 10")
sol <- solve(A, c(5, 10)); cat("Solution (x, y) =", sol, "| check:", A %*% sol, "\n")

df <- load_data("students.csv")
X <- cbind(1, as.matrix(df[, c("study_hours", "attendance_pct", "prior_gpa")])); y <- df$exam_score

h("DATASET: regression through the normal equations beta = (X'X)^-1 X'y")
XtX <- crossprod(X); Xty <- crossprod(X, y)
beta <- solve(XtX, Xty)
cat(sprintf("det(X'X) = %.4g | condition number = %.3g | rank(X) = %d\n", det(XtX), kappa(XtX, exact = TRUE), Matrix::rankMatrix(X)[1]))
print(round(beta, 4))
cat("lm() gives the same coefficients:", all.equal(as.vector(beta), unname(coef(lm(exam_score ~ study_hours + attendance_pct + prior_gpa, df)))), "\n")

H <- X %*% solve(XtX) %*% t(X)
cat(sprintf("Hat matrix: trace = %.4f (= #parameters 4); idempotent: %s; symmetric: %s\n",
            sum(diag(H)), isTRUE(all.equal(H %*% H, H)), isTRUE(all.equal(H, t(H)))))
resid <- y - X %*% beta; sigma2 <- sum(resid^2) / (nrow(X) - ncol(X))
cat("Standard errors:", round(sqrt(diag(sigma2 * solve(XtX))), 4), "\n")
qrX <- qr(X); cat("QR solution matches:", isTRUE(all.equal(as.vector(qr.coef(qrX, y)), as.vector(beta))), "\n")

save_plot("matrix_regression", {
  plot(y, X %*% beta, pch = 16, col = rgb(0, 0, 1, .4), xlab = "Actual", ylab = "Fitted (X beta)", main = "Fitted values from matrix solution")
  abline(0, 1, col = "red", lty = 2)
})

h("INTERPRETATION")
cat("- Matrix algebra gives the whole regression in one equation: beta = (X'X)^-1 X'y.\n")
cat("- det(A) = 5 != 0, so the 2x2 system has a unique solution.\n")
cat("- trace(H) equals the number of parameters; H projects y onto the fitted values.\n")
end_topic()
