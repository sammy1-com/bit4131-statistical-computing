"""Kaggle Titanic: missing data (T22), chi-square (T17), logistic regression (T18), non-parametric (T19)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from common.helpers import Report, load

R = Report(__file__)
df = load("kaggle/titanic.csv")
R.h("DATASET: Kaggle Titanic (891 passengers)")
R.p(df[["Survived", "Pclass", "Age", "Fare"]].describe().round(2))

R.h("MISSING DATA (Topic 22)")
miss = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
R.p(miss[miss > 0].to_string())
df["Age_missing"] = df["Age"].isna().astype(int)
R.p("Survival rate when Age missing:", round(df.loc[df.Age_missing == 1, "Survived"].mean(), 3),
    "| observed:", round(df.loc[df.Age_missing == 0, "Survived"].mean(), 3))
df["Age_imp"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))
R.p(f"Age SD before = {df.Age.std():.2f}, after group-median imputation = {df.Age_imp.std():.2f}")

R.h("CHI-SQUARE (Topic 17): passenger class vs survival")
tab = pd.crosstab(df.Pclass, df.Survived); R.p(tab)
chi2, p, dof, exp = stats.chi2_contingency(tab)
V = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
R.p(f"chi2 = {chi2:.2f}, df = {dof}, p = {p:.3g}, Cramer's V = {V:.3f}")
tab2 = pd.crosstab(df.Sex, df.Survived); c2, p2, _, _ = stats.chi2_contingency(tab2)
R.p(f"Sex vs survival: chi2 = {c2:.2f}, p = {p2:.3g}")

R.h("NON-PARAMETRIC (Topic 19): Fare by survival (Fare is heavily skewed)")
f1, f0 = df.Fare[df.Survived == 1], df.Fare[df.Survived == 0]
u, pu = stats.mannwhitneyu(f1, f0)
R.p(f"Fare skewness = {df.Fare.skew():.2f}; medians: survived = {f1.median():.2f}, died = {f0.median():.2f}")
R.p(f"Mann-Whitney U = {u:.0f}, p = {pu:.3g}")

R.h("LOGISTIC REGRESSION (Topic 18)")
m = smf.logit("Survived ~ C(Pclass) + C(Sex) + Age_imp + SibSp + Fare", data=df).fit(disp=0)
R.p(m.summary().tables[1])
ci = np.exp(m.conf_int()); orr = pd.DataFrame({"OR": np.exp(m.params), "lo": ci[0], "hi": ci[1], "p": m.pvalues}).round(3)
R.p("\nOdds ratios:\n" + orr.to_string())
X = pd.get_dummies(df[["Pclass", "Sex", "Age_imp", "SibSp", "Fare"]], columns=["Pclass", "Sex"], drop_first=True).astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, df.Survived, test_size=.3, random_state=4131, stratify=df.Survived)
clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr); pr = clf.predict_proba(Xte)[:, 1]; pred = (pr > .5).astype(int)
R.p("\nConfusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(yte, pred))
R.p(f"Accuracy = {accuracy_score(yte, pred):.3f}, AUC = {roc_auc_score(yte, pr):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
(df.groupby("Pclass").Survived.mean()).plot.bar(ax=ax[0], color="#1f78b4"); ax[0].set_title("Survival rate by class"); ax[0].set_ylabel("Rate")
(df.groupby("Sex").Survived.mean()).plot.bar(ax=ax[1], color="#33a02c"); ax[1].set_title("Survival rate by sex")
fpr, tpr, _ = roc_curve(yte, pr); ax[2].plot(fpr, tpr); ax[2].plot([0, 1], [0, 1], "k--"); ax[2].set_title(f"ROC (AUC {roc_auc_score(yte, pr):.2f})")
R.save_fig(fig, "py_titanic_overview.png")

R.h("INTERPRETATION")
R.p(f"- Age is missing for {miss['Age']:.0f}% of passengers; group-median imputation keeps all rows but shrinks the age SD.")
R.p(f"- Class and survival are strongly associated (chi2 = {chi2:.1f}, p = {p:.2g}); sex is stronger still (chi2 = {c2:.1f}).")
R.p(f"- Being male multiplies the odds of survival by {orr.loc['C(Sex)[T.male]', 'OR']:.2f}; 3rd class by {orr.loc['C(Pclass)[T.3]', 'OR']:.2f} (vs 1st).")
R.p(f"- The model classifies {accuracy_score(yte, pred):.0%} of held-out passengers correctly (AUC {roc_auc_score(yte, pr):.2f}).")
R.finish()
