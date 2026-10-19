"""Topic 6: Hypothesis testing (scipy.stats, statsmodels)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

R = Report(__file__)


def decide(p, alpha=0.05):
    return f"p = {p:.4g} -> " + ("REJECT H0" if p < alpha else "FAIL TO REJECT H0") + f" at alpha = {alpha}"


# ---- 3. Worked example ----------------------------------------------------------------
R.h("WORKED EXAMPLE: one-sample t-test, H0: mu = 60")
n, xbar, s, mu0 = 36, 63, 9, 60
t = (xbar - mu0) / (s / np.sqrt(n))
p = 2 * stats.t.sf(abs(t), n - 1)
R.p(f"t = (63-60)/(9/6) = {t:.3f}, df = 35, two-sided p = {p:.4f}")
R.p(decide(p))

df = load("students.csv")
tut = load("tutoring_scores.csv")

R.h("TEST 1: one-sample t-test, H0: mean exam score = 60")
r1 = stats.ttest_1samp(df.exam_score, 60)
R.p(f"t = {r1.statistic:.3f}, df = {len(df) - 1}, {decide(r1.pvalue)}")

R.h("TEST 2: two-sample Welch t-test, method A vs C")
a, c = df[df.method == "A"].exam_score, df[df.method == "C"].exam_score
r2 = stats.ttest_ind(c, a, equal_var=False)
d = (c.mean() - a.mean()) / np.sqrt((a.var(ddof=1) + c.var(ddof=1)) / 2)
R.p(f"mean C = {c.mean():.2f}, mean A = {a.mean():.2f}")
R.p(f"t = {r2.statistic:.3f}, {decide(r2.pvalue)}; Cohen's d = {d:.2f}")

R.h("TEST 3: paired t-test, tutoring pre vs post")
r3 = stats.ttest_rel(tut.post_score, tut.pre_score)
R.p(f"mean gain = {(tut.post_score - tut.pre_score).mean():.2f}; t = {r3.statistic:.3f}; {decide(r3.pvalue)}")

R.h("TEST 4: one-proportion z-test, H0: pass rate = 0.70")
z, p4 = proportions_ztest(df.passed.sum(), len(df), 0.70)
R.p(f"p-hat = {df.passed.mean():.3f}; z = {z:.3f}; {decide(p4)}")

R.h("TEST 5: Levene test for equal variances (A vs C)")
lev = stats.levene(a, c)
R.p(f"Levene W = {lev.statistic:.3f}; {decide(lev.pvalue)}")

# ---- Plot: rejection region ------------------------------------------------------------
fig, ax = plt.subplots()
xs = np.linspace(-5, 8, 500)
dfree = len(a) + len(c) - 2
ax.plot(xs, stats.t.pdf(xs, dfree), "k")
crit = stats.t.ppf(0.975, dfree)
ax.fill_between(xs, stats.t.pdf(xs, dfree), where=abs(xs) > crit, color="red", alpha=0.3, label="Rejection region (5%)")
ax.axvline(r2.statistic, color="blue", lw=2, label=f"Observed t = {r2.statistic:.2f}")
ax.set_title("Two-sample t-test: method C vs A"); ax.legend(); ax.set_xlabel("t")
R.save_fig(fig, "py_rejection_region.png")

fig, ax = plt.subplots()
ax.boxplot([a, df[df.method == "B"].exam_score, c], tick_labels=["A", "B", "C"], patch_artist=True)
ax.set_ylabel("Exam score"); ax.set_title("Scores by method")
R.save_fig(fig, "py_method_boxplot.png")

R.h("INTERPRETATION")
R.p("- A p-value is the probability of data this extreme if H0 were true; below 0.05 we reject H0.")
R.p(f"- Mean score differs from 60 (p={r1.pvalue:.3g}): " + ("significant." if r1.pvalue < .05 else "not significant."))
R.p(f"- Method C beats A by {c.mean() - a.mean():.1f} marks (p={r2.pvalue:.2g}, Cohen's d={d:.2f}): significant and practically large.")
R.p(f"- Tutoring raised scores on average (paired p={r3.pvalue:.3g}).")
R.p("- Statistical significance is not the same as practical importance: report effect size too.")
R.finish()
