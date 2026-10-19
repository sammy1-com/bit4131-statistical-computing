"""Topic 18: Logistic regression (statsmodels, scikit-learn)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common.helpers import Report, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

R = Report(__file__)

R.h("WORKED EXAMPLE: logit arithmetic")
b0, b1 = -4.0, 0.5
for x in (4, 8, 12):
    eta = b0 + b1 * x
    R.p(f"x = {x:>2}: log-odds = {eta:+.1f}, odds = {np.exp(eta):.3f}, P = 1/(1+e^-eta) = {1 / (1 + np.exp(-eta)):.3f}")
R.p(f"Odds ratio per unit of x = e^0.5 = {np.exp(0.5):.3f}")

df = load("students.csv")
cols = ["study_hours", "attendance_pct", "sleep_hours", "prior_gpa"]
X = sm.add_constant(df[cols])
y = df["passed"]

R.h("LOGISTIC REGRESSION (statsmodels): passed ~ predictors")
model = sm.Logit(y, X).fit(disp=0)
R.p(model.summary().as_text())
R.h("ODDS RATIOS with 95% CI")
orr = pd.DataFrame({"OR": np.exp(model.params), "lower": np.exp(model.conf_int()[0]), "upper": np.exp(model.conf_int()[1]), "p": model.pvalues})
R.p(orr.drop("const").round(4).to_string())
R.p(f"\nMcFadden pseudo R2 = {model.prsquared:.3f}")

R.h("CLASSIFICATION PERFORMANCE (70/30 train-test, scikit-learn)")
Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.3, random_state=3, stratify=y)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
prob = clf.predict_proba(Xte)[:, 1]
pred = (prob >= 0.5).astype(int)
cm = confusion_matrix(yte, pred)
R.p("Confusion matrix [[TN FP],[FN TP]]:\n" + str(cm))
acc, prec, rec, f1, auc = (accuracy_score(yte, pred), precision_score(yte, pred), recall_score(yte, pred),
                           f1_score(yte, pred), roc_auc_score(yte, prob))
R.p(f"Accuracy = {acc:.3f}; Precision = {prec:.3f}; Recall = {rec:.3f}; F1 = {f1:.3f}; AUC = {auc:.3f}")
R.p(f"Baseline (always predict 'pass') accuracy = {max(yte.mean(), 1 - yte.mean()):.3f}")

new = pd.DataFrame({"const": 1, "study_hours": [6, 12, 18], "attendance_pct": [70, 85, 95], "sleep_hours": [6, 7, 8], "prior_gpa": [2.3, 3.0, 3.6]})
R.p("\nPredicted pass probability for three hypothetical students:")
for (i, row), pp in zip(new.iterrows(), model.predict(new)):
    R.p(f"  hours={row.study_hours:.0f}, attend={row.attendance_pct:.0f}%, gpa={row.prior_gpa}: P(pass) = {pp:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"]).plot(ax=ax[0], colorbar=False, cmap="Blues"); ax[0].set_title("Confusion matrix"); ax[0].grid(False)
fpr, tpr, _ = roc_curve(yte, prob)
ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.2f}"); ax[1].plot([0, 1], [0, 1], "k--")
ax[1].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate"); ax[1].legend()
xs = np.linspace(df.study_hours.min(), df.study_hours.max(), 100)
grid = pd.DataFrame({"const": 1, "study_hours": xs, "attendance_pct": df.attendance_pct.mean(), "sleep_hours": df.sleep_hours.mean(), "prior_gpa": df.prior_gpa.mean()})
ax[2].scatter(df.study_hours, y, alpha=0.3, s=12); ax[2].plot(xs, model.predict(grid), "r", lw=2)
ax[2].set(title="Fitted probability curve", xlabel="Study hours", ylabel="P(pass)")
R.save_fig(fig, "py_logistic_performance.png")

fig, ax = plt.subplots()
o = orr.drop("const")
ax.errorbar(o.OR, range(len(o)), xerr=[o.OR - o.lower, o.upper - o.OR], fmt="o", capsize=5)
ax.axvline(1, color="r", ls="--"); ax.set_yticks(range(len(o))); ax.set_yticklabels(o.index); ax.set_xscale("log")
ax.set_title("Odds ratios (95% CI, log scale)")
R.save_fig(fig, "py_odds_ratios.png")

sig = o[o.p < 0.05]
R.h("INTERPRETATION")
for k, rw in sig.iterrows():
    R.p(f"- {k}: odds ratio {rw.OR:.2f} -> each unit increase multiplies the odds of passing by {rw.OR:.2f} (p = {rw.p:.2g}).")
base_acc = max(yte.mean(), 1 - yte.mean())
R.p(f"- Test accuracy is {acc:.0%} versus a {base_acc:.0%} baseline of always predicting 'pass': only a modest gain, "
    "because the classes are imbalanced, so accuracy alone is misleading.")
R.p(f"- AUC = {auc:.2f} and recall = {rec:.2f} are more informative: the model ranks likely passers well above likely failers.")
R.p("- Odds ratio > 1 raises the odds of the event; < 1 lowers it; 1 means no effect.")
R.finish()
