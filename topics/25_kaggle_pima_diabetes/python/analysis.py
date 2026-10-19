"""Kaggle Pima Indians Diabetes: hidden missing values (T22), hypothesis tests (T6), PCA (T10), multivariate (T21)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from common.helpers import Report, load

R = Report(__file__)
df = load("kaggle/pima_diabetes.csv")
R.h("DATASET: Kaggle Pima Indians Diabetes (768 women)")
R.p(df.describe().round(2).T[["count", "mean", "std", "min", "max"]])

R.h("HIDDEN MISSING VALUES (Topic 22): zeros that are physiologically impossible")
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zeros = (df[cols] == 0).sum(); R.p(pd.DataFrame({"zeros": zeros, "pct": (zeros / len(df) * 100).round(1)}))
d = df.copy(); d[cols] = d[cols].replace(0, np.nan)
imp = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(d.drop(columns="Outcome")), columns=d.columns[:-1]); imp["Outcome"] = df.Outcome
R.p(f"Insulin mean: zeros kept = {df.Insulin.mean():.1f}; true observed = {d.Insulin.mean():.1f}; KNN-imputed = {imp.Insulin.mean():.1f}")

R.h("HYPOTHESIS TESTS (Topic 6): diabetic vs non-diabetic (imputed data)")
for c in ["Glucose", "BMI", "Age", "BloodPressure"]:
    a, b = imp.loc[imp.Outcome == 1, c], imp.loc[imp.Outcome == 0, c]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sd = np.sqrt((a.var() + b.var()) / 2)
    R.p(f"{c:14s} diabetic mean = {a.mean():6.1f}, non = {b.mean():6.1f}, Welch t = {t:6.2f}, p = {p:.2e}, Cohen d = {(a.mean()-b.mean())/sd:.2f}")

R.h("PCA (Topic 10) on standardised predictors")
X = StandardScaler().fit_transform(imp.drop(columns="Outcome")); pca = PCA().fit(X)
R.p("Eigenvalues:", pca.explained_variance_.round(3)); R.p("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
R.p("Cumulative:", pca.explained_variance_ratio_.cumsum().round(3))
load_ = pd.DataFrame(pca.components_[:2].T, index=imp.columns[:-1], columns=["PC1", "PC2"]).round(2); R.p(load_)
sc = pca.transform(X)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].bar(range(1, 9), pca.explained_variance_ratio_); ax[0].set_title("Scree plot"); ax[0].set_xlabel("Component")
ax[1].scatter(sc[:, 0], sc[:, 1], c=imp.Outcome, cmap="coolwarm", s=10, alpha=.6); ax[1].set_title("PC1 vs PC2 (red = diabetic)")
imp.boxplot("Glucose", by="Outcome", ax=ax[2]); ax[2].set_title("Glucose by outcome"); plt.suptitle("")
R.save_fig(fig, "py_pima_overview.png")

R.h("INTERPRETATION")
R.p(f"- {zeros['Insulin']/len(df):.0%} of Insulin and {zeros['SkinThickness']/len(df):.0%} of SkinThickness values are coded 0 (really missing); leaving them in drags the Insulin mean from {d.Insulin.mean():.0f} to {df.Insulin.mean():.0f}.")
R.p("- Glucose differs strongly between groups (see Welch p and Cohen d above): the strongest single marker of diabetes here.")
R.p(f"- The first two components explain {pca.explained_variance_ratio_[:2].sum():.0%} of the variance; no clean separation, so several variables are needed to classify.")
R.finish()
