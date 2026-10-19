"""Topic 16: ANOVA and experimental design (statsmodels, scipy.stats)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from common.helpers import Report, load
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

R = Report(__file__)

# ---- 3. Worked example: by-hand one-way ANOVA ------------------------------------------------
R.h("WORKED EXAMPLE: groups G1=[5,6,7], G2=[8,9,10], G3=[11,12,13]")
g = [np.array([5, 6, 7.]), np.array([8, 9, 10.]), np.array([11, 12, 13.])]
grand = np.concatenate(g).mean()
ssb = sum(len(x) * (x.mean() - grand) ** 2 for x in g)
ssw = sum(((x - x.mean()) ** 2).sum() for x in g)
dfb, dfw = 2, 6
F = (ssb / dfb) / (ssw / dfw)
R.p(f"Grand mean = {grand};  SSB = {ssb};  SSW = {ssw}")
R.p(f"MSB = {ssb / dfb};  MSW = {ssw / dfw};  F = {F:.2f};  p = {stats.f.sf(F, dfb, dfw):.5f}")
R.p("scipy check:", stats.f_oneway(*g))

df = load("students.csv")

R.h("ONE-WAY ANOVA: exam_score ~ method")
model = smf.ols("exam_score ~ C(method)", df).fit()
table = sm.stats.anova_lm(model, typ=2)
R.p(table.round(4).to_string())
F1, p1 = table.loc["C(method)", "F"], table.loc["C(method)", "PR(>F)"]
eta2 = table.loc["C(method)", "sum_sq"] / table["sum_sq"].sum()
R.p(f"\nF = {F1:.3f}, p = {p1:.3g}, eta-squared = {eta2:.3f}")
R.p("\nGroup means:\n" + df.groupby("method").exam_score.agg(["count", "mean", "std"]).round(2).to_string())

R.h("ASSUMPTION CHECKS")
R.p(f"Levene (equal variance): p = {stats.levene(*[df[df.method == m].exam_score for m in 'ABC']).pvalue:.4f}")
R.p(f"Shapiro (residual normality): p = {stats.shapiro(model.resid).pvalue:.4f}")

R.h("POST-HOC: Tukey HSD")
tk = pairwise_tukeyhsd(df.exam_score, df.method, alpha=0.05)
R.p(str(tk.summary()))

R.h("TWO-WAY ANOVA: exam_score ~ method * gender")
m2 = smf.ols("exam_score ~ C(method) * C(gender)", df).fit()
t2 = sm.stats.anova_lm(m2, typ=2)
R.p(t2.round(4).to_string())

# ---- Plots -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].boxplot([df[df.method == m].exam_score for m in "ABC"], tick_labels=list("ABC"), patch_artist=True)
ax[0].set(title="Score by teaching method", xlabel="Method", ylabel="Score")
means = df.groupby("method").exam_score.mean()
ci = df.groupby("method").exam_score.apply(lambda x: 1.96 * x.std() / np.sqrt(len(x)))
ax[1].errorbar(means.index, means.values, yerr=ci.values, fmt="o-", capsize=6, lw=2)
ax[1].set(title="Group means with 95% CI", xlabel="Method", ylabel="Mean score")
R.save_fig(fig, "py_anova_groups.png")

fig = plt.figure(figsize=(7, 4.5)); tk.plot_simultaneous(figsize=(7, 4.5)); plt.title("Tukey HSD simultaneous CIs")
R.save_fig(plt.gcf(), "py_tukey_hsd.png")

fig, ax = plt.subplots()
for gdr, gg in df.groupby("gender"):
    ax.plot(["A", "B", "C"], gg.groupby("method").exam_score.mean(), "o-", label=f"Gender {gdr}")
ax.set(title="Interaction plot: method x gender", xlabel="Method", ylabel="Mean score"); ax.legend()
R.save_fig(fig, "py_interaction_plot.png")

sig_pairs = [f"{r[0]}-{r[1]}" for r in tk.summary().data[1:] if str(r[-1]) == "True"]
R.h("INTERPRETATION")
R.p(f"- The F-test (F = {F1:.1f}, p = {p1:.2g}) " + ("rejects H0: at least one method has a different mean." if p1 < .05 else "does not reject H0."))
R.p(f"- Method explains {eta2:.0%} of score variance (eta-squared): a medium effect.")
R.p(f"- Tukey HSD identifies which pairs differ: {', '.join(sig_pairs) if sig_pairs else 'none'}.")
R.p(f"- Interaction method x gender p = {t2.loc['C(method):C(gender)', 'PR(>F)']:.3f}: " +
    ("no evidence the method effect depends on gender." if t2.loc['C(method):C(gender)', 'PR(>F)'] > .05 else "the effect depends on gender."))
R.finish()
