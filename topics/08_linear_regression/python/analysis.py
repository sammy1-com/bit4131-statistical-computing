"""Topic 8: Linear regression and least squares (statsmodels, scikit-learn, numpy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from common.helpers import Report, load
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

R = Report(__file__)

# ---- 3. Worked example: least squares by hand -------------------------------------------
R.h("WORKED EXAMPLE: x=[1,2,3,4,5], y=[2,4,5,4,5]")
x = np.array([1, 2, 3, 4, 5.]); y = np.array([2, 4, 5, 4, 5.])
b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b0 = y.mean() - b1 * x.mean()
R.p(f"b1 = Sxy/Sxx = 6/10 = {b1:.2f};  b0 = ybar - b1*xbar = 4 - 0.6*3 = {b0:.2f}")
R.p(f"y-hat = {b0:.1f} + {b1:.1f}x")
X = np.column_stack([np.ones(5), x])
R.p("Normal equations (X'X)^-1 X'y =", np.linalg.solve(X.T @ X, X.T @ y).round(3))

# ---- 4/5. Simple regression -------------------------------------------------------------
df = load("students.csv")
R.h("SIMPLE REGRESSION: exam_score ~ study_hours")
simple = sm.OLS(df.exam_score, sm.add_constant(df[["study_hours"]])).fit()
R.p(simple.summary().as_text())

# ---- Multiple regression ------------------------------------------------------------------
R.h("MULTIPLE REGRESSION: exam_score ~ study_hours + attendance + sleep + prior_gpa")
cols = ["study_hours", "attendance_pct", "sleep_hours", "prior_gpa"]
Xf = sm.add_constant(df[cols])
model = sm.OLS(df.exam_score, Xf).fit()
R.p(model.summary().as_text())
R.p("\nRegression equation:")
R.p("exam_score = " + f"{model.params['const']:.2f} " + " ".join(f"{v:+.2f}*{k}" for k, v in model.params.items() if k != "const"))
R.p(f"R-squared = {model.rsquared:.3f}; adj R-squared = {model.rsquared_adj:.3f}; RMSE = {np.sqrt(model.mse_resid):.2f}")

# ---- Diagnostics ---------------------------------------------------------------------------
resid, fitted = model.resid, model.fittedvalues
sw = stats.shapiro(resid)
R.h("DIAGNOSTICS")
R.p(f"Shapiro-Wilk on residuals: W={sw.statistic:.3f}, p={sw.pvalue:.3f}")
R.p(f"Durbin-Watson = {sm.stats.durbin_watson(resid):.2f}  (about 2 means no autocorrelation)")
from statsmodels.stats.diagnostic import het_breuschpagan
bp = het_breuschpagan(resid, Xf)
R.p(f"Breusch-Pagan test for heteroscedasticity: p = {bp[1]:.3f}")

# ---- scikit-learn cross-check + holdout -----------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(df[cols], df.exam_score, test_size=0.25, random_state=1)
sk = LinearRegression().fit(Xtr, ytr)
R.h("SCIKIT-LEARN HOLDOUT CHECK (75/25 split)")
R.p("Coefficients:", dict(zip(cols, sk.coef_.round(3))), " intercept:", round(sk.intercept_, 3))
R.p(f"Test R2 = {r2_score(yte, sk.predict(Xte)):.3f}; Test RMSE = {np.sqrt(mean_squared_error(yte, sk.predict(Xte))):.2f}")

# ---- Plots -----------------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.scatter(df.study_hours, df.exam_score, alpha=0.5, s=18)
xs = np.linspace(df.study_hours.min(), df.study_hours.max(), 50)
ax.plot(xs, simple.params["const"] + simple.params["study_hours"] * xs, "r", lw=2,
        label=f"y = {simple.params['const']:.1f} + {simple.params['study_hours']:.2f}x")
ax.set_xlabel("Study hours"); ax.set_ylabel("Exam score"); ax.set_title("Fitted regression line"); ax.legend()
R.save_fig(fig, "py_fitted_regression.png")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].scatter(fitted, resid, alpha=0.5, s=16); ax[0].axhline(0, color="r")
ax[0].set_xlabel("Fitted"); ax[0].set_ylabel("Residual"); ax[0].set_title("Residuals vs fitted")
sm.qqplot(resid, line="s", ax=ax[1]); ax[1].set_title("Normal Q-Q")
ax[2].hist(resid, bins=20, color="#9ecae1", edgecolor="white"); ax[2].set_title("Residual histogram")
R.save_fig(fig, "py_residual_diagnostics.png")

sig = [k for k in cols if model.pvalues[k] < 0.05]
R.h("INTERPRETATION")
R.p(f"- Model explains {model.rsquared:.0%} of variation in exam scores (adj R2 = {model.rsquared_adj:.2f}).")
R.p(f"- Each extra study hour adds about {model.params['study_hours']:.2f} marks, holding other variables fixed.")
R.p(f"- Significant predictors (p < 0.05): {', '.join(sig)}.")
R.p("- Residuals are " + ("approximately normal and show no strong pattern: assumptions look reasonable." if sw.pvalue > 0.05
    else "not perfectly normal: interpret p-values with some caution."))
R.finish()
