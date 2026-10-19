"""Topic 9: Matrix algebra in statistics (numpy, scipy.linalg)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from common.helpers import Report, load

R = Report(__file__)
np.set_printoptions(precision=4, suppress=True)

A = np.array([[2., 1.], [1., 3.]])
B = np.array([[1., 2.], [0., 1.]])

R.h("WORKED EXAMPLE: A=[[2,1],[1,3]], B=[[1,2],[0,1]]")
R.p("A + B =\n", A + B)
R.p("A @ B =\n", A @ B)
R.p("B @ A =\n", B @ A, "\n(matrix multiplication is NOT commutative)")
R.p("A transpose =\n", A.T)
R.p("det(A) = 2*3 - 1*1 =", round(np.linalg.det(A), 4))
R.p("A inverse = (1/5)[[3,-1],[-1,2]] =\n", np.linalg.inv(A))
R.p("A @ inv(A) =\n", A @ np.linalg.inv(A))

R.h("SOLVING LINEAR EQUATIONS: 2x + y = 5, x + 3y = 10")
sol = np.linalg.solve(A, np.array([5., 10.]))
R.p("Solution (x, y) =", sol, "-> check A @ sol =", A @ sol)
lu, piv = sla.lu_factor(A)
R.p("Same solution by LU factorisation (scipy.linalg):", sla.lu_solve((lu, piv), np.array([5., 10.])))

# ---- Dataset: OLS through matrix algebra ----------------------------------------------------
df = load("students.csv")
cols = ["study_hours", "attendance_pct", "prior_gpa"]
X = np.column_stack([np.ones(len(df)), df[cols].to_numpy()])
y = df.exam_score.to_numpy()

R.h("DATASET: regression through the NORMAL EQUATIONS  beta = (X'X)^-1 X'y")
XtX, Xty = X.T @ X, X.T @ y
beta = np.linalg.solve(XtX, Xty)
R.p("X'X shape:", XtX.shape, "  det(X'X) =", f"{np.linalg.det(XtX):.4g}", "  condition number =", f"{np.linalg.cond(XtX):.3g}")
for name, b in zip(["intercept"] + cols, beta):
    R.p(f"  beta[{name}] = {b:.4f}")

H = X @ np.linalg.inv(XtX) @ X.T
R.p(f"\nHat matrix H = X(X'X)^-1 X': trace = {np.trace(H):.4f} (= number of parameters 4)")
R.p("H is idempotent (H@H == H):", np.allclose(H @ H, H), "  symmetric:", np.allclose(H, H.T))
resid = y - X @ beta
sigma2 = resid @ resid / (len(y) - X.shape[1])
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtX)))
R.p("Standard errors = sqrt(diag(sigma^2 (X'X)^-1)):", se.round(4))

Q, Rr = np.linalg.qr(X)
R.p("QR route (numerically stabler) gives same beta:", np.allclose(np.linalg.solve(Rr, Q.T @ y), beta))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(y, X @ beta, alpha=0.5, s=16); lims = [y.min(), y.max()]
ax[0].plot(lims, lims, "r--"); ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Fitted (X beta)")
ax[0].set_title("Fitted values from matrix solution")
im = ax[1].imshow(np.corrcoef(X[:, 1:].T), cmap="RdBu_r", vmin=-1, vmax=1); ax[1].grid(False)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(cols, rotation=30); ax[1].set_yticks(range(3)); ax[1].set_yticklabels(cols)
ax[1].set_title("Predictor correlations (conditioning of X'X)")
fig.colorbar(im, ax=ax[1])
R.save_fig(fig, "py_matrix_regression.png")

R.h("INTERPRETATION")
R.p("- Matrix algebra turns a regression with many variables into one equation: beta = (X'X)^-1 X'y.")
R.p("- A is invertible because det(A) = 5 != 0, so the linear system has a unique solution.")
R.p(f"- Trace of the hat matrix equals the parameter count ({np.trace(H):.0f}); H projects y onto the fitted values.")
R.p("- A large condition number would signal near-collinear predictors; QR gives a stabler solution.")
R.finish()
