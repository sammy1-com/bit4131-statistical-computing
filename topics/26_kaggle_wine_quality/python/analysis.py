"""Kaggle Red Wine Quality: correlation (T7), regression (T8), ANOVA (T16), normalisation (T15)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from common.helpers import Report, load

R = Report(__file__)
df = load("kaggle/wine_quality_red.csv"); df.columns = [c.replace(" ", "_") for c in df.columns]
R.h("DATASET: Kaggle Red Wine Quality (1,599 wines)")
R.p(df.describe().round(2).T[["mean", "std", "min", "max"]])

R.h("CORRELATION with quality (Topic 7)")
cor = df.corr()["quality"].drop("quality").sort_values(); R.p(cor.round(3).to_string())
r, p = stats.pearsonr(df.alcohol, df.quality); R.p(f"alcohol vs quality: r = {r:.3f}, p = {p:.2e}")

R.h("MULTIPLE REGRESSION (Topic 8)")
m = smf.ols("quality ~ alcohol + volatile_acidity + sulphates + total_sulfur_dioxide + chlorides + pH", data=df).fit()
R.p(m.summary().tables[0]); R.p(m.summary().tables[1])
X = sm.add_constant(df[["alcohol", "volatile_acidity", "sulphates", "total_sulfur_dioxide", "chlorides", "pH"]])
R.p("VIF:", {c: round(variance_inflation_factor(X.values, i), 2) for i, c in enumerate(X.columns) if c != "const"})
R.p(f"Residual normality (Shapiro): p = {stats.shapiro(m.resid)[1]:.3g}")

R.h("ANOVA (Topic 16): alcohol content by quality grade")
df["grade"] = pd.cut(df.quality, [0, 5, 6, 10], labels=["low(<=5)", "medium(6)", "high(>=7)"])
fit = smf.ols("alcohol ~ C(grade)", data=df).fit(); a = anova_lm(fit); R.p(a)
R.p(f"eta^2 = {a.sum_sq.iloc[0]/a.sum_sq.sum():.3f}"); R.p(pairwise_tukeyhsd(df.alcohol, df.grade))
R.p("Levene p =", round(stats.levene(*[g.alcohol for _, g in df.groupby("grade", observed=True)])[1], 4))

R.h("TRANSFORMATION (Topic 15): chlorides is right-skewed")
R.p(f"skewness raw = {df.chlorides.skew():.2f}; log = {np.log(df.chlorides).skew():.2f}; sqrt = {np.sqrt(df.chlorides).skew():.2f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].scatter(df.alcohol, df.quality, s=6, alpha=.25); b = np.polyfit(df.alcohol, df.quality, 1); xs = np.linspace(df.alcohol.min(), df.alcohol.max(), 50)
ax[0].plot(xs, np.polyval(b, xs), "r"); ax[0].set_title("Quality vs alcohol")
df.boxplot("alcohol", by="grade", ax=ax[1]); ax[1].set_title("Alcohol by grade"); plt.suptitle("")
ax[2].scatter(m.fittedvalues, m.resid, s=6, alpha=.3); ax[2].axhline(0, c="r"); ax[2].set_title("Residuals vs fitted")
R.save_fig(fig, "py_wine_overview.png")

R.h("INTERPRETATION")
R.p(f"- Alcohol has the strongest positive link with quality (r = {r:.2f}); volatile acidity the strongest negative ({cor['volatile_acidity']:.2f}).")
R.p(f"- The model explains {m.rsquared:.0%} of the variation in quality (adj R2 = {m.rsquared_adj:.2f}): useful but far from complete.")
R.p(f"- Each extra 1% alcohol is associated with +{m.params['alcohol']:.2f} quality points, holding the other variables fixed.")
R.p(f"- Mean alcohol differs across quality grades (F = {a.F.iloc[0]:.1f}, p = {a['PR(>F)'].iloc[0]:.2g}); log transform reduces chlorides skew a lot.")
R.p("- Quality is an integer score, so residuals are not perfectly normal; treat p-values as approximate.")
R.finish()
