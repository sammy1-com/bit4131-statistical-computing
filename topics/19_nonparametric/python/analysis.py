"""Topic 19: Non-parametric statistical methods (scipy.stats)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats

R = Report(__file__)

R.h("WORKED EXAMPLE: Mann-Whitney U by hand")
a, b = np.array([3, 5, 8]), np.array([4, 9, 12, 15])
ranks = stats.rankdata(np.concatenate([a, b]))
R.p("Combined ranks:", ranks, " (a gets", ranks[:3], ", b gets", ranks[3:], ")")
Ra = ranks[:3].sum()
U = Ra - 3 * 4 / 2
R.p(f"R_a = {Ra};  U_a = R_a - n_a(n_a+1)/2 = {Ra} - 6 = {U};  scipy: {stats.mannwhitneyu(a, b).statistic}")

tk = load("support_tickets.csv")
tut = load("tutoring_scores.csv")
df = load("students.csv")

R.h("DATA ARE SKEWED -> parametric assumptions fail")
r = tk.resolution_hours
R.p(f"resolution_hours skewness = {stats.skew(r):.2f}; Shapiro-Wilk p = {stats.shapiro(r).pvalue:.3g}")

R.h("TEST 1: Mann-Whitney U — High vs Low priority resolution time")
hi, lo = tk[tk.priority == "High"].resolution_hours, tk[tk.priority == "Low"].resolution_hours
mw = stats.mannwhitneyu(hi, lo, alternative="two-sided")
tt = stats.ttest_ind(hi, lo, equal_var=False)
R.p(f"medians: High = {hi.median():.2f}h, Low = {lo.median():.2f}h")
R.p(f"Mann-Whitney U = {mw.statistic:.0f}, p = {mw.pvalue:.3g}")
R.p(f"(parametric Welch t-test for comparison: t = {tt.statistic:.2f}, p = {tt.pvalue:.3g})")
rbc = 1 - 2 * mw.statistic / (len(hi) * len(lo))
R.p(f"Rank-biserial correlation (effect size) = {rbc:.3f}")

R.h("TEST 2: Kruskal-Wallis — all three priorities")
groups = [tk[tk.priority == p].resolution_hours for p in ("Low", "Medium", "High")]
kw = stats.kruskal(*groups)
R.p(f"H = {kw.statistic:.3f}, df = 2, p = {kw.pvalue:.3g}")
R.p("Post-hoc pairwise Mann-Whitney (Bonferroni, alpha = 0.05/3):")
for (n1, g1), (n2, g2) in [(("Low", groups[0]), ("Medium", groups[1])), (("Low", groups[0]), ("High", groups[2])), (("Medium", groups[1]), ("High", groups[2]))]:
    pv = stats.mannwhitneyu(g1, g2).pvalue
    R.p(f"  {n1} vs {n2}: p = {pv:.3g}  {'significant' if pv < 0.05 / 3 else 'not significant'}")

R.h("TEST 3: Wilcoxon signed-rank — tutoring pre vs post")
wx = stats.wilcoxon(tut.post_score, tut.pre_score)
pt = stats.ttest_rel(tut.post_score, tut.pre_score)
R.p(f"Wilcoxon W = {wx.statistic:.1f}, p = {wx.pvalue:.3g}   |   paired t: t = {pt.statistic:.2f}, p = {pt.pvalue:.3g}")

R.h("TEST 4: Spearman rank correlation vs Pearson (study hours vs score)")
sp = stats.spearmanr(df.study_hours, df.exam_score)
pe = stats.pearsonr(df.study_hours, df.exam_score)
R.p(f"Spearman rho = {sp.statistic:.3f} (p = {sp.pvalue:.3g});  Pearson r = {pe.statistic:.3f}")

R.h("TEST 5: Kolmogorov-Smirnov two-sample — method A vs C score distributions")
ks = stats.ks_2samp(df[df.method == "A"].exam_score, df[df.method == "C"].exam_score)
R.p(f"D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].boxplot(groups, tick_labels=["Low", "Medium", "High"], patch_artist=True, showfliers=False)
ax[0].set(title="Resolution time by priority (skewed)", ylabel="Hours")
ax[1].hist(r, bins=40, color="#9ecae1", edgecolor="white"); ax[1].set(title="Why non-parametric: right-skewed data", xlabel="Hours")
R.save_fig(fig, "py_nonparametric_groups.png")

fig, ax = plt.subplots()
ax.scatter(tut.pre_score, tut.post_score, s=25); lim = [tut[["pre_score", "post_score"]].min().min(), tut[["pre_score", "post_score"]].max().max()]
ax.plot(lim, lim, "r--", label="No change line"); ax.set(xlabel="Pre score", ylabel="Post score", title="Paired scores (above line = improved)"); ax.legend()
R.save_fig(fig, "py_paired_scores.png")

R.h("INTERPRETATION")
R.p(f"- Resolution times are heavily skewed, so rank-based tests are safer than t-tests.")
R.p(f"- Kruskal-Wallis (p = {kw.pvalue:.2g}) shows priority levels differ; High-priority tickets resolve in a median of {hi.median():.1f}h vs {lo.median():.1f}h for Low.")
R.p(f"- Non-parametric and parametric tests agree on the paired tutoring data (Wilcoxon p = {wx.pvalue:.2g}): tutoring helped.")
R.p(f"- Spearman ({sp.statistic:.2f}) and Pearson ({pe.statistic:.2f}) are similar because the relationship is roughly linear.")
R.finish()
