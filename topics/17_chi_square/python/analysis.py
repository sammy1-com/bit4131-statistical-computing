"""Topic 17: Chi-square and categorical data analysis (scipy.stats, statsmodels)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load
from scipy import stats

R = Report(__file__)

# ---- 3. Worked example -------------------------------------------------------------------------
R.h("WORKED EXAMPLE: 2x2 table [[30, 20], [20, 30]]")
obs = np.array([[30, 20], [20, 30]])
exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
chi = ((obs - exp) ** 2 / exp).sum()
R.p("Expected = row total x col total / grand total =\n", exp)
R.p(f"chi2 = sum((O-E)^2/E) = 4*(25/25) = {chi:.2f}; df = 1; p = {stats.chi2.sf(chi, 1):.4f}")

# ---- 4/5. Test of independence ---------------------------------------------------------------------
sv = load("survey.csv")
R.h("TEST OF INDEPENDENCE: programme vs preferred programming language")
tab = pd.crosstab(sv.program, sv.preferred_language)
R.p("Observed counts:\n" + tab.to_string())
chi2, p, dof, expected = stats.chi2_contingency(tab)
R.p("\nExpected frequencies:\n" + pd.DataFrame(expected, index=tab.index, columns=tab.columns).round(1).to_string())
R.p(f"\nchi2 = {chi2:.3f}, df = {dof}, p = {p:.3g}")
R.p(f"Smallest expected count = {expected.min():.1f} (rule of thumb: all >= 5 -> {'OK' if expected.min() >= 5 else 'violated'})")
cramer = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
R.p(f"Cramer's V = {cramer:.3f}")
resid = (tab.values - expected) / np.sqrt(expected)
R.p("\nStandardised (Pearson) residuals:\n" + pd.DataFrame(resid, index=tab.index, columns=tab.columns).round(2).to_string())

R.h("GOODNESS OF FIT: are the 4 languages equally popular?")
counts = sv.preferred_language.value_counts()
gof = stats.chisquare(counts.values)
R.p(counts.to_string())
R.p(f"chi2 = {gof.statistic:.2f}, df = 3, p = {gof.pvalue:.3g}")

df = load("students.csv")
R.h("2x2: gender vs passed (Fisher exact and chi-square)")
t22 = pd.crosstab(df.gender, df.passed)
R.p(t22.to_string())
c22 = stats.chi2_contingency(t22)
orr, pf = stats.fisher_exact(t22)
R.p(f"chi2 (Yates) = {c22[0]:.3f}, p = {c22[1]:.3f};  Fisher exact p = {pf:.3f}, odds ratio = {orr:.2f}")

# ---- Plots -------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
tab.plot(kind="bar", ax=ax[0], rot=10); ax[0].set(title="Observed counts", ylabel="Respondents")
prop = tab.div(tab.sum(1), axis=0) * 100
prop.plot(kind="bar", stacked=True, ax=ax[1], rot=10); ax[1].set(title="Percent within programme", ylabel="%")
R.save_fig(fig, "py_chi_square_bars.png")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
im = ax.imshow(resid, cmap="RdBu_r", vmin=-4, vmax=4); ax.grid(False)
ax.set_xticks(range(4)); ax.set_xticklabels(tab.columns); ax.set_yticks(range(3)); ax.set_yticklabels(tab.index)
for i in range(3):
    for j in range(4):
        ax.text(j, i, f"{resid[i, j]:.1f}", ha="center", va="center")
fig.colorbar(im); ax.set_title("Pearson residuals (|r|>2 = notable)")
R.save_fig(fig, "py_residual_heatmap.png")

big = [(tab.index[i], tab.columns[j], resid[i, j]) for i in range(3) for j in range(4) if abs(resid[i, j]) > 2]
R.h("INTERPRETATION")
R.p(f"- chi2 = {chi2:.1f} with df = {dof}, p = {p:.2g}: " + ("programme and language preference are associated." if p < .05 else "no evidence of association."))
R.p(f"- Cramer's V = {cramer:.2f} indicates a {'strong' if cramer > .3 else 'moderate' if cramer > .15 else 'weak'} association.")
R.p("- Biggest contributors: " + "; ".join(f"{a}/{b} ({r:+.1f})" for a, b, r in sorted(big, key=lambda t: -abs(t[2]))[:3]))
R.p(f"- Gender and passing are independent here (Fisher p = {pf:.2f}).")
R.finish()
