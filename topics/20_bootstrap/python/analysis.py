"""Topic 20: Bootstrap and resampling methods (numpy, scipy, scikit-learn)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats
from sklearn.utils import resample

R = Report(__file__)
rng = np.random.default_rng(20)

R.h("WORKED EXAMPLE: one bootstrap resample of [3, 5, 7, 9, 11]")
tiny = np.array([3, 5, 7, 9, 11])
for i in range(3):
    s = rng.choice(tiny, size=5, replace=True)
    R.p(f"resample {i + 1}: {s}  -> mean {s.mean():.1f}")
R.p("Sampling WITH replacement: values may repeat, others be left out.")

tk = load("support_tickets.csv")["resolution_hours"].to_numpy()
df = load("students.csv")
B = 10_000

R.h("BOOTSTRAP: median resolution time (no simple formula for its SE)")
boot_med = np.array([np.median(rng.choice(tk, len(tk))) for _ in range(B)])
pct = np.percentile(boot_med, [2.5, 97.5])
R.p(f"Sample median = {np.median(tk):.3f}; bootstrap SE = {boot_med.std(ddof=1):.4f}; bias = {boot_med.mean() - np.median(tk):+.4f}")
R.p(f"95% percentile CI = ({pct[0]:.3f}, {pct[1]:.3f})")
res = stats.bootstrap((tk,), np.median, n_resamples=B, confidence_level=0.95, method="BCa", random_state=1)
R.p(f"95% BCa CI (scipy)  = ({res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f})")

R.h("BOOTSTRAP: mean exam score — compared with the t-interval")
x = df.exam_score.to_numpy()
bm = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
t_lo, t_hi = stats.t.interval(0.95, len(x) - 1, x.mean(), stats.sem(x))
R.p(f"Bootstrap SE = {bm.std(ddof=1):.4f} vs analytic s/sqrt(n) = {stats.sem(x):.4f}")
R.p(f"Bootstrap 95% CI = ({np.percentile(bm, 2.5):.2f}, {np.percentile(bm, 97.5):.2f});  t 95% CI = ({t_lo:.2f}, {t_hi:.2f})")

R.h("BOOTSTRAP: correlation (study_hours, exam_score)")
n = len(df); bc = []
for _ in range(B):
    idx = rng.integers(0, n, n)
    bc.append(np.corrcoef(df.study_hours.to_numpy()[idx], x[idx])[0, 1])
bc = np.array(bc)
R.p(f"r = {np.corrcoef(df.study_hours, x)[0, 1]:.3f}; bootstrap SE = {bc.std(ddof=1):.4f}; 95% CI = ({np.percentile(bc, 2.5):.3f}, {np.percentile(bc, 97.5):.3f})")

R.h("BOOTSTRAP: difference in mean score, method C - A")
a, c = df[df.method == "A"].exam_score.to_numpy(), df[df.method == "C"].exam_score.to_numpy()
bd = np.array([resample(c).mean() - resample(a).mean() for _ in range(B)])
R.p(f"Observed diff = {c.mean() - a.mean():.2f}; 95% CI = ({np.percentile(bd, 2.5):.2f}, {np.percentile(bd, 97.5):.2f})")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].hist(boot_med, bins=40, color="#9ecae1", edgecolor="white")
for v in pct: ax[0].axvline(v, color="r", ls="--")
ax[0].axvline(np.median(tk), color="k"); ax[0].set(title="Bootstrap: median resolution time", xlabel="Median (h)")
ax[1].hist(bm, bins=40, color="#a1d99b", edgecolor="white"); ax[1].set(title="Bootstrap: mean exam score", xlabel="Mean")
ax[2].hist(bc, bins=40, color="#fdae6b", edgecolor="white"); ax[2].set(title="Bootstrap: correlation", xlabel="r")
R.save_fig(fig, "py_bootstrap_distributions.png")

fig, ax = plt.subplots()
stats.probplot(bm, plot=ax); ax.set_title("Q-Q plot of bootstrap means")
R.save_fig(fig, "py_bootstrap_qq.png")

R.h("INTERPRETATION")
R.p(f"- The median has no easy SE formula, yet the bootstrap gives SE = {boot_med.std(ddof=1):.3f}h and a CI of ({pct[0]:.2f}, {pct[1]:.2f}).")
R.p("- For the mean, bootstrap and classical t-intervals nearly coincide: the bootstrap is trustworthy.")
R.p("- Correlation and mean differences get honest uncertainty estimates without normality assumptions.")
R.p("- The bootstrap approximates the sampling distribution by resampling the data itself.")
R.finish()
