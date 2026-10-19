"""Topic 5: Confidence intervals (scipy.stats, statsmodels)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

R = Report(__file__)
rng = np.random.default_rng(5)

# ---- 3. Worked example --------------------------------------------------------------
R.h("WORKED EXAMPLE: n=25, mean=50, s=10, 95% CI")
n, xbar, s = 25, 50, 10
tcrit = stats.t.ppf(0.975, n - 1)
me = tcrit * s / np.sqrt(n)
R.p(f"SE = s/sqrt(n) = {s / np.sqrt(n):.2f};  t(0.975, 24) = {tcrit:.4f}")
R.p(f"Margin = {me:.3f};  CI = ({xbar - me:.2f}, {xbar + me:.2f})")

# ---- 4/5. Dataset ---------------------------------------------------------------------
df = load("students.csv")
x = df["exam_score"]
n, xbar, s = len(x), x.mean(), x.std(ddof=1)
R.h("CI FOR MEAN EXAM SCORE (t-interval)")
for conf in (0.90, 0.95, 0.99):
    lo, hi = stats.t.interval(conf, n - 1, loc=xbar, scale=s / np.sqrt(n))
    R.p(f"{conf:.0%} CI: ({lo:.2f}, {hi:.2f})  width = {hi - lo:.2f}")
R.p(f"Point estimate = {xbar:.2f}")

R.h("CI FOR PASS PROPORTION (Wilson)")
k = df["passed"].sum()
lo, hi = proportion_confint(k, len(df), alpha=0.05, method="wilson")
R.p(f"p-hat = {k / len(df):.4f}; 95% Wilson CI = ({lo:.4f}, {hi:.4f})")

R.h("CI FOR DIFFERENCE IN MEANS (method C - method A, Welch)")
a, c = df[df.method == "A"].exam_score, df[df.method == "C"].exam_score
diff = c.mean() - a.mean()
se = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
dof = se**4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (c.var(ddof=1) / len(c)) ** 2 / (len(c) - 1))
t = stats.t.ppf(0.975, dof)
R.p(f"Difference = {diff:.2f}; 95% CI = ({diff - t * se:.2f}, {diff + t * se:.2f})")

# ---- Coverage simulation: what '95%' really means ---------------------------------------
R.h("COVERAGE SIMULATION: 100 intervals from samples of n=30")
mu_true = x.mean()
hits, ints = 0, []
for _ in range(100):
    smp = rng.choice(x, 30)
    lo, hi = stats.t.interval(0.95, 29, loc=smp.mean(), scale=stats.sem(smp))
    ok = lo <= mu_true <= hi
    hits += ok
    ints.append((lo, hi, ok))
R.p(f"{hits}/100 intervals contain the true mean {mu_true:.2f}")
fig, ax = plt.subplots(figsize=(7, 7))
for i, (lo, hi, ok) in enumerate(ints):
    ax.plot([lo, hi], [i, i], color="#2c7fb8" if ok else "#d7301f", lw=1.5)
ax.axvline(mu_true, color="k", ls="--"); ax.set_xlabel("Exam score"); ax.set_ylabel("Sample #")
ax.set_title(f"100 95% CIs ({hits} cover the true mean; red = miss)")
R.save_fig(fig, "py_ci_coverage.png")

fig, ax = plt.subplots()
widths = [stats.t.interval(c, n - 1, xbar, s / np.sqrt(n)) for c in (0.8, 0.9, 0.95, 0.99)]
for i, (c, (lo, hi)) in enumerate(zip((0.8, 0.9, 0.95, 0.99), widths)):
    ax.plot([lo, hi], [i, i], "o-", lw=3); ax.text(hi + 0.05, i, f"{c:.0%}", va="center")
ax.axvline(xbar, color="k", ls="--"); ax.set_yticks([]); ax.set_xlabel("Mean exam score")
ax.set_title("Higher confidence -> wider interval")
R.save_fig(fig, "py_ci_widths.png")

R.h("INTERPRETATION")
R.p(f"- We are 95% confident the true mean exam score lies between {xbar - 1.968 * s / np.sqrt(n):.1f} and {xbar + 1.968 * s / np.sqrt(n):.1f}.")
R.p("- 'Confidence' describes the method: about 95 of every 100 such intervals capture the true mean.")
R.p("- The interval for method C - A excludes 0, so C genuinely outperforms A." if diff - t * se > 0
    else "- The interval for method C - A includes 0: no clear difference.")
R.p("- Raising confidence from 90% to 99% widens the interval: more certainty costs precision.")
R.finish()
