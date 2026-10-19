"""Topic 22: Missing-data analysis and imputation (pandas, sklearn.impute)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

R = Report(__file__)
full = load("students.csv")
miss = load("students_missing.csv")
cols = ["study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score"]

R.h("WORKED EXAMPLE: mean imputation on [4, 6, NaN, 10]")
v = np.array([4, 6, np.nan, 10])
R.p("observed mean = (4+6+10)/3 =", np.nanmean(v), "-> imputed series:", np.where(np.isnan(v), np.nanmean(v), v))
R.p("Note: mean imputation keeps the mean but SHRINKS the variance.")

R.h("MISSING-DATA SUMMARY")
summ = pd.DataFrame({"n_missing": miss[cols].isna().sum(), "pct_missing": (miss[cols].isna().mean() * 100).round(1)})
R.p(summ.to_string())
R.p(f"\nRows with at least one missing value: {miss[cols].isna().any(axis=1).sum()} of {len(miss)} ({miss[cols].isna().any(axis=1).mean():.0%})")
R.p(f"Complete cases (listwise deletion) leave n = {miss[cols].dropna().shape[0]}")

R.h("MECHANISM: is missingness related to OBSERVED prior_gpa? (MAR check, Fisher exact)")
obs_gpa = miss.prior_gpa.notna()
low = miss.prior_gpa < 2.7
mech = {}
for var in ["attendance_pct", "sleep_hours"]:
    m = miss[var].isna()
    t = pd.crosstab(low[obs_gpa], m[obs_gpa])
    _, pf = stats.fisher_exact(t)
    mech[var] = pf
    R.p(f"{var:<15}: missing {m[obs_gpa & low].mean():.0%} when GPA < 2.7 vs {m[obs_gpa & ~low].mean():.0%} otherwise; Fisher p = {pf:.3g}")
R.p("Small p -> missingness depends on an observed variable (MAR); large p -> consistent with MCAR.")

R.h("IMPUTATION METHODS vs TRUTH (we know the original values)")
mask = miss[cols].isna()
X = miss[cols]
imputed = {
    "Mean": SimpleImputer(strategy="mean"),
    "Median": SimpleImputer(strategy="median"),
    "KNN (k=5)": KNNImputer(n_neighbors=5),
    "Iterative (regression)": IterativeImputer(max_iter=15, random_state=0),
}
res, sets = [], {}
for name, imp in imputed.items():
    arr = pd.DataFrame(imp.fit_transform(X), columns=cols)
    sets[name] = arr
    row = {"method": name}
    for c in ["study_hours", "attendance_pct", "prior_gpa"]:
        row[f"RMSE_{c}"] = np.sqrt(((arr.loc[mask[c], c] - full.loc[mask[c], c]) ** 2).mean())
    row["SD_attendance"] = arr.attendance_pct.std()
    res.append(row)
res = pd.DataFrame(res).set_index("method")
R.p(res.round(3).to_string())
R.p(f"\nTrue SD of attendance (full data) = {full.attendance_pct.std():.3f}; after deletion = {X.attendance_pct.std():.3f}")

R.h("BEFORE vs AFTER IMPUTATION: regression slope of exam_score on study_hours")
def slope(d): return np.polyfit(d.study_hours, d.exam_score, 1)[0]
R.p(f"Truth (full data)       : {slope(full):.4f}")
R.p(f"Complete cases          : {slope(X.dropna()):.4f}")
for name, arr in sets.items():
    R.p(f"{name:<24}: {slope(arr):.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
(miss[cols].isna().mean() * 100).plot(kind="barh", ax=ax, color="#fdae6b"); ax.set(title="Percent missing by variable", xlabel="% missing")
R.save_fig(fig, "py_missing_summary.png")

fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(miss[cols].isna().to_numpy().T, aspect="auto", cmap="Greys", interpolation="nearest"); ax.grid(False)
ax.set_yticks(range(5)); ax.set_yticklabels(cols); ax.set(title="Missingness map (black = missing)", xlabel="Student row")
R.save_fig(fig, "py_missingness_map.png")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for a, name in zip(ax, ["Mean", "KNN (k=5)", "Iterative (regression)"]):
    a.hist(full.attendance_pct, bins=25, alpha=0.5, label="Original", density=True)
    a.hist(sets[name].attendance_pct, bins=25, alpha=0.5, label=f"{name} imputed", density=True); a.legend(fontsize=8); a.set_title(name)
fig.suptitle("Attendance distribution before / after imputation")
R.save_fig(fig, "py_imputation_comparison.png")

best_sh = res["RMSE_study_hours"].idxmin()
best_att = res["RMSE_attendance_pct"].idxmin()
slopes = {n: slope(a) for n, a in sets.items()}
closest = min(slopes, key=lambda n: abs(slopes[n] - slope(full)))
R.h("INTERPRETATION")
R.p(f"- Listwise deletion would discard {1 - miss[cols].dropna().shape[0] / len(miss):.0%} of students: wasteful and potentially biased.")
R.p(f"- Attendance missingness vs low GPA: Fisher p = {mech['attendance_pct']:.2g} -> " +
    ("depends on an observed variable (MAR)." if mech['attendance_pct'] < .05 else "no clear dependence detected.") +
    f" Sleep: p = {mech['sleep_hours']:.2g} -> " + ("MAR." if mech['sleep_hours'] < .05 else "consistent with MCAR."))
R.p(f"- study_hours is imputed best by {best_sh} (RMSE {res.loc[best_sh, 'RMSE_study_hours']:.2f} vs {res.loc['Mean', 'RMSE_study_hours']:.2f} for the mean): it is related to exam_score, which the model can exploit.")
R.p(f"- attendance_pct is best imputed by {best_att}: it is only weakly related to the other variables, so model-based methods have little to add.")
R.p(f"- Mean imputation shrinks the SD of attendance to {res.loc['Mean', 'SD_attendance']:.2f} (true {full.attendance_pct.std():.2f}), understating uncertainty.")
R.p(f"- Regression slope closest to the truth ({slope(full):.3f}): {closest} ({slopes[closest]:.3f}). All methods land within {max(abs(v - slope(full)) for v in slopes.values()):.2f} of it here.")
R.p("- Lesson: there is no universally best imputation; the benefit depends on how strongly the missing variable relates to the rest.")
R.finish()
