"""Kaggle Heart Disease (Cleveland): CI (T5), hypothesis tests (T6), logistic regression (T18), bootstrap (T20)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from common.helpers import Report, load

R = Report(__file__)
df = load("kaggle/heart_disease.csv"); df.columns = [c.strip("﻿") for c in df.columns]
R.h("DATASET: Kaggle Heart Disease (Cleveland, 303 patients; target as labelled by Kaggle)")
R.p("NOTE: physiology suggests target=1 is the LOWER-risk group (higher max heart rate, less ST depression); the Kaggle description says target=1 = disease. Results below follow the label exactly as given; verify against the UCI source before clinical claims.")
R.p(df.target.value_counts().to_string()); R.p(df[["age", "trestbps", "chol", "thalach"]].describe().round(1))

R.h("CONFIDENCE INTERVALS (Topic 5)")
x = df.chol; se = x.std(ddof=1) / np.sqrt(len(x)); t = stats.t.ppf(.975, len(x) - 1)
R.p(f"Cholesterol mean = {x.mean():.1f}, 95% CI = ({x.mean()-t*se:.1f}, {x.mean()+t*se:.1f})")
p = df.target.mean(); lo, hi = stats.binomtest(int(df.target.sum()), len(df)).proportion_ci(.95, method="wilson")
R.p(f"Proportion with target=1 = {p:.3f}, 95% Wilson CI = ({lo:.3f}, {hi:.3f})")

R.h("HYPOTHESIS TESTS (Topic 6) and BOOTSTRAP (Topic 20): max heart rate (thalach) by target")
a, b = df.thalach[df.target == 1], df.thalach[df.target == 0]
tt, pp = stats.ttest_ind(a, b, equal_var=False); R.p(f"target=1 mean = {a.mean():.1f}, target=0 = {b.mean():.1f}, Welch t = {tt:.2f}, p = {pp:.2e}")
rng = np.random.default_rng(4131); diffs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)])
R.p(f"Bootstrap 95% CI for the difference = ({np.percentile(diffs, 2.5):.1f}, {np.percentile(diffs, 97.5):.1f}), SE = {diffs.std():.2f}")
c2, pc, _, _ = stats.chi2_contingency(pd.crosstab(df.sex, df.target)); R.p(f"Sex vs target: chi2 = {c2:.2f}, p = {pc:.3g}")

R.h("LOGISTIC REGRESSION (Topic 18)")
m = smf.logit("target ~ age + C(sex) + C(cp) + trestbps + chol + thalach + C(exang) + oldpeak", data=df).fit(disp=0)
R.p(m.summary().tables[1]); ci = np.exp(m.conf_int())
orr = pd.DataFrame({"OR": np.exp(m.params), "lo": ci[0], "hi": ci[1], "p": m.pvalues}).round(3); R.p("\nOdds ratios:\n" + orr.to_string())
X = df[["age", "sex", "trestbps", "chol", "thalach", "exang", "oldpeak"]]; X = pd.concat([X, pd.get_dummies(df.cp, prefix="cp", drop_first=True)], axis=1).astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, df.target, test_size=.3, random_state=4131, stratify=df.target)
sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
pr = clf.predict_proba(sc.transform(Xte))[:, 1]; pred = (pr > .5).astype(int)
R.p("\nConfusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(yte, pred))
R.p(f"Accuracy = {accuracy_score(yte, pred):.3f}, Precision = {precision_score(yte, pred):.3f}, Recall = {recall_score(yte, pred):.3f}, AUC = {roc_auc_score(yte, pr):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].hist([b, a], bins=15, label=["target=0", "target=1"], color=["#1f78b4", "#e31a1c"]); ax[0].legend(); ax[0].set_title("Max heart rate")
ax[1].hist(diffs, bins=40, color="#33a02c"); ax[1].set_title("Bootstrap: mean difference (thalach)")
fpr, tpr, _ = roc_curve(yte, pr); ax[2].plot(fpr, tpr); ax[2].plot([0, 1], [0, 1], "k--"); ax[2].set_title(f"ROC (AUC {roc_auc_score(yte, pr):.2f})")
R.save_fig(fig, "py_heart_overview.png")

R.h("INTERPRETATION")
R.p(f"- About {p:.0%} of patients carry label target=1 (95% CI {lo:.0%} to {hi:.0%}).")
R.p(f"- Patients with target=1 reach a higher peak heart rate by {a.mean()-b.mean():.1f} bpm (p = {pp:.1e}); the bootstrap CI agrees with the t-test.")
R.p(f"- Each extra unit of ST depression (oldpeak) multiplies the odds of target=1 by {orr.loc['oldpeak', 'OR']:.2f} (below 1 = lowers the odds, consistent with target=1 being the lower-risk label).")
R.p(f"- The model reaches AUC {roc_auc_score(yte, pr):.2f} on held-out patients; recall {recall_score(yte, pred):.2f} matters most in screening (missed cases).")
R.finish()
