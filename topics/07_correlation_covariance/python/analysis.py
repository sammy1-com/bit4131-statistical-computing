"""Topic 7: Correlation and covariance (pandas, numpy, scipy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats

R = Report(__file__)

# ---- 3. Worked example ---------------------------------------------------------------
R.h("WORKED EXAMPLE: x=[1,2,3,4,5], y=[2,4,5,4,5]")
x = np.array([1, 2, 3, 4, 5]); y = np.array([2, 4, 5, 4, 5])
cov = ((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1)
r = cov / (x.std(ddof=1) * y.std(ddof=1))
R.p(f"x-bar = {x.mean()}, y-bar = {y.mean()}")
R.p(f"cov(x,y) = sum((x-xbar)(y-ybar))/(n-1) = 6/4 = {cov}")
R.p(f"r = cov/(sx*sy) = {r:.4f}   (numpy check: {np.corrcoef(x, y)[0, 1]:.4f})")

# ---- 4/5. Dataset ---------------------------------------------------------------------
df = load("students.csv")
num = df[["study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score"]]

R.h("COVARIANCE MATRIX")
R.p(num.cov().round(2).to_string())
R.h("PEARSON CORRELATION MATRIX")
corr = num.corr()
R.p(corr.round(3).to_string())
R.h("SPEARMAN (rank) CORRELATION MATRIX")
R.p(num.corr(method="spearman").round(3).to_string())

R.h("SIGNIFICANCE of correlation with exam_score")
rows = []
for c in num.columns[:-1]:
    r_, p_ = stats.pearsonr(num[c], num["exam_score"])
    rows.append((c, r_, p_))
    R.p(f"{c:<15} r = {r_:+.3f}   p = {p_:.3g}")

# ---- Plots ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(corr.columns, rotation=40, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(corr.columns); ax.grid(False)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im); ax.set_title("Correlation matrix")
R.save_fig(fig, "py_correlation_heatmap.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(df.study_hours, df.exam_score, alpha=0.5, s=18)
b, a = np.polyfit(df.study_hours, df.exam_score, 1)
xs = np.linspace(df.study_hours.min(), df.study_hours.max(), 50)
ax[0].plot(xs, a + b * xs, "r"); ax[0].set_xlabel("Study hours"); ax[0].set_ylabel("Exam score")
ax[0].set_title(f"r = {corr.loc['study_hours', 'exam_score']:.2f}")
ax[1].scatter(df.sleep_hours, df.exam_score, alpha=0.5, s=18, color="#31a354")
ax[1].set_xlabel("Sleep hours"); ax[1].set_title(f"r = {corr.loc['sleep_hours', 'exam_score']:.2f}")
R.save_fig(fig, "py_scatter_plots.png")

best = max(rows, key=lambda t: abs(t[1]))
R.h("INTERPRETATION")
R.p(f"- Strongest linear link with exam score: {best[0]} (r = {best[1]:+.2f}, p = {best[2]:.2g}).")
R.p("- Positive r: both rise together. |r| near 1 is strong; near 0 means no linear relationship.")
R.p("- Covariance depends on units (hard to compare); correlation is the unit-free version in [-1, 1].")
R.p("- Correlation measures association, not causation: study time may proxy for motivation.")
R.finish()
