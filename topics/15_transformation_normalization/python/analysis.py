"""Topic 15: Data transformation and normalization (pandas, numpy, sklearn.preprocessing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

R = Report(__file__)

R.h("WORKED EXAMPLE: data = [10, 20, 30, 40, 100]")
d = np.array([10, 20, 30, 40, 100.])
mu, sd = d.mean(), d.std(ddof=1)
R.p(f"mean = {mu}, sd = {sd:.4f}")
R.p("z-score    =", ((d - mu) / sd).round(3), " (z = (x - mean)/sd)")
R.p("min-max    =", ((d - d.min()) / (d.max() - d.min())).round(3), " ((x - min)/(max - min))")
R.p("log10      =", np.log10(d).round(3))

tk = load("support_tickets.csv")
r = tk["resolution_hours"]
df = load("students.csv")

R.h("DATASET 1: skewed variable - ticket resolution_hours")
tr = pd.DataFrame({
    "original": r,
    "log": np.log(r),
    "sqrt": np.sqrt(r),
    "boxcox": stats.boxcox(r)[0],
    "yeo_johnson": PowerTransformer("yeo-johnson").fit_transform(r.to_frame()).ravel(),
})
summary = pd.DataFrame({"skewness": tr.apply(stats.skew), "shapiro_p": tr.apply(lambda c: stats.shapiro(c).pvalue)})
R.p(summary.round(4).to_string())
best = summary.skewness.abs().drop("original").idxmin()
R.p(f"\nBox-Cox lambda = {stats.boxcox(r)[1]:.3f} (0 means log); least skewed transform: {best}")

R.h("DATASET 2: scaling numeric predictors (students)")
cols = ["study_hours", "attendance_pct", "exam_score"]
X = df[cols]
std = pd.DataFrame(StandardScaler().fit_transform(X), columns=cols)
mm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=cols)
rb = pd.DataFrame(RobustScaler().fit_transform(X), columns=cols)
R.p("Original:\n" + X.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
R.p("\nStandardised (z-score):\n" + std.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
R.p("\nMin-max [0,1]:\n" + mm.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

# ---- Plots ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for a, c in zip(ax.ravel(), ["original", "log", "sqrt", "boxcox", "yeo_johnson"]):
    a.hist(tr[c], bins=30, color="#9ecae1", edgecolor="white")
    a.set_title(f"{c} (skew {stats.skew(tr[c]):.2f})")
stats.probplot(tr["log"], plot=ax[1, 2]); ax[1, 2].set_title("Q-Q plot of log transform")
fig.suptitle("Transforming a right-skewed variable")
R.save_fig(fig, "py_skew_transformations.png")

fig, ax = plt.subplots(1, 4, figsize=(15, 4))
for a, (name, frame) in zip(ax, [("Original", X), ("Standardised", std), ("Min-max", mm), ("Robust", rb)]):
    a.boxplot([frame[c] for c in cols], tick_labels=["hours", "attend.", "score"]); a.set_title(name)
fig.suptitle("Same data, different scales")
R.save_fig(fig, "py_scaling_comparison.png")

R.h("INTERPRETATION")
R.p(f"- Resolution time has skewness {summary.loc['original', 'skewness']:.2f}; the {best} transform cuts it to {summary.loc[best, 'skewness']:.2f}.")
R.p("- z-scores give mean 0 / SD 1 (compare variables on one scale); min-max maps to [0,1] but is sensitive to outliers.")
R.p("- Scaling before distance-based methods (k-means, PCA, KNN) stops large-unit variables dominating.")
R.p("- Always fit scalers on training data only, then apply to test data, to avoid leakage.")
R.finish()
