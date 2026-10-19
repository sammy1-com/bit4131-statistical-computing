"""Topic 1: Measures of central tendency and dispersion (pandas, numpy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load

R = Report(__file__)
df = load("students.csv")
x = df["exam_score"]

# ---- 3. Worked example (hand-checkable) -------------------------------------
R.h("WORKED EXAMPLE: data = [4, 8, 6, 5, 3, 8, 9]")
w = np.array([4, 8, 6, 5, 3, 8, 9])
R.p("mean     =", w.mean(), "(sum 43 / 7)")
R.p("median   =", np.median(w), "(sorted 3 4 5 6 8 8 9 -> middle value)")
R.p("mode     =", 8, "(appears twice)")
R.p("variance =", round(w.var(ddof=1), 4), "(sample, n-1)")
R.p("std dev  =", round(w.std(ddof=1), 4))

# ---- 4/5. Dataset analysis ---------------------------------------------------
R.h("DATASET: exam_score of 300 students")
mean, median = x.mean(), x.median()
mode = x.round().mode().iloc[0]
var, sd = x.var(ddof=1), x.std(ddof=1)
q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = x[(x < lo) | (x > hi)]
cv = sd / mean * 100
R.p(f"Mean            : {mean:.2f}")
R.p(f"Median          : {median:.2f}")
R.p(f"Mode (rounded)  : {mode:.0f}")
R.p(f"Variance        : {var:.2f}")
R.p(f"Std deviation   : {sd:.2f}")
R.p(f"Range           : {x.min():.1f} to {x.max():.1f}  (width {x.max() - x.min():.1f})")
R.p(f"Q1 / Q2 / Q3    : {q1:.2f} / {q2:.2f} / {q3:.2f}")
R.p(f"IQR             : {iqr:.2f}")
R.p(f"Outlier fences  : [{lo:.2f}, {hi:.2f}]  -> {len(outliers)} outlier(s)")
R.p(f"Coeff. of var.  : {cv:.1f}%")
R.p(f"Skewness        : {x.skew():.3f}")

R.h("Dispersion by teaching method")
R.p(df.groupby("method")["exam_score"].agg(["count", "mean", "median", "std"]).round(2).to_string())

# ---- 6. Visualisation --------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].boxplot(x, vert=False, patch_artist=True, boxprops=dict(facecolor="#9ecae1"))
ax[0].set_title("Boxplot of exam scores"); ax[0].set_xlabel("Exam score")
ax[1].hist(x, bins=20, color="#9ecae1", edgecolor="white")
ax[1].axvline(mean, color="red", label=f"Mean {mean:.1f}")
ax[1].axvline(median, color="green", ls="--", label=f"Median {median:.1f}")
ax[1].set_title("Histogram"); ax[1].legend(); ax[1].set_xlabel("Exam score")
R.save_fig(fig, "py_boxplot_hist.png")

fig, ax = plt.subplots()
df.boxplot(column="exam_score", by="method", ax=ax, grid=False)
ax.set_title("Exam score by teaching method"); fig.suptitle("")
ax.set_xlabel("Method"); ax.set_ylabel("Exam score")
R.save_fig(fig, "py_boxplot_by_method.png")

# ---- Interpretation ------------------------------------------------------------
R.h("INTERPRETATION")
shape = ("approximately symmetric" if abs(mean - median) < 0.05 * sd
         else "right-skewed" if mean > median else "left-skewed")
R.p(f"- Mean ({mean:.1f}) and median ({median:.1f}) are close, so the distribution is {shape}.")
R.p(f"- Typical students deviate about {sd:.1f} marks from the mean; the middle 50% of scores span {iqr:.1f} marks.")
R.p(f"- The boxplot flags {len(outliers)} unusual score(s) beyond 1.5 x IQR from the quartiles.")
R.p("- Method C has the highest centre, method A the lowest: spread is similar, centres differ.")
R.finish()
