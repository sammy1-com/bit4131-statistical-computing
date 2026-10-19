"""Topic 3: Probability distributions (scipy.stats, numpy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats

R = Report(__file__)

# ---- 3. Worked examples ---------------------------------------------------------
R.h("WORKED EXAMPLES")
R.p("Binomial(n=10, p=0.3): P(X=3) =", round(stats.binom.pmf(3, 10, 0.3), 4),
    "= C(10,3)(0.3)^3(0.7)^7")
R.p("Poisson(lambda=5):      P(X<=3) =", round(stats.poisson.cdf(3, 5), 4))
R.p("Normal(60, 10):         P(50<X<70) =", round(stats.norm.cdf(70, 60, 10) - stats.norm.cdf(50, 60, 10), 4))
R.p("Exponential(mean 8):    P(X>10) =", round(stats.expon.sf(10, scale=8), 4))

# ---- 4/5. Fitting to data ---------------------------------------------------------
scores = load("students.csv")["exam_score"].to_numpy()
tk = load("support_tickets.csv")

R.h("FIT 1: exam_score ~ Normal")
mu, sigma = stats.norm.fit(scores)
ks = stats.kstest(scores, stats.norm(mu, sigma).cdf)
R.p(f"Fitted mu = {mu:.2f}, sigma = {sigma:.2f}")
R.p(f"P(score >= 70)  = {stats.norm.sf(70, mu, sigma):.4f}")
R.p(f"P(score < 50)   = {stats.norm.cdf(50, mu, sigma):.4f}")
R.p(f"90th percentile = {stats.norm.ppf(0.9, mu, sigma):.2f}")
R.p(f"KS test: D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}")

R.h("FIT 2: arrival_gap_min ~ Exponential")
loc, scale = stats.expon.fit(tk["arrival_gap_min"], floc=0)
ks2 = stats.kstest(tk["arrival_gap_min"], stats.expon(0, scale).cdf)
R.p(f"Fitted mean gap = {scale:.2f} min (rate = {1 / scale:.4f}/min)")
R.p(f"KS test: D = {ks2.statistic:.4f}, p = {ks2.pvalue:.4f}")

R.h("FIT 3: tickets_per_hour ~ Poisson")
lam = tk["tickets_per_hour"].mean()
R.p(f"Estimated lambda = {lam:.3f}; variance = {tk['tickets_per_hour'].var():.3f} (Poisson: mean = variance)")

R.h("FIT 4: resolution_hours ~ Lognormal vs Normal (AIC)")
s, loc_l, sc_l = stats.lognorm.fit(tk["resolution_hours"], floc=0)
ll_ln = stats.lognorm.logpdf(tk["resolution_hours"], s, loc_l, sc_l).sum()
m, sd = stats.norm.fit(tk["resolution_hours"])
ll_n = stats.norm.logpdf(tk["resolution_hours"], m, sd).sum()
aic_ln, aic_n = 2 * 2 - 2 * ll_ln, 2 * 2 - 2 * ll_n
R.p(f"Lognormal: sigma={s:.3f}, median={sc_l:.3f}h, AIC={aic_ln:.1f}")
R.p(f"Normal   : mean={m:.3f}, sd={sd:.3f}, AIC={aic_n:.1f}")
R.p("Better fit (lower AIC):", "Lognormal" if aic_ln < aic_n else "Normal")

# ---- Plots ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
xs = np.linspace(scores.min(), scores.max(), 300)
ax[0].hist(scores, bins=20, density=True, color="#9ecae1", edgecolor="white")
ax[0].plot(xs, stats.norm.pdf(xs, mu, sigma), "r", lw=2, label="Fitted Normal")
ax[0].set_title("Exam scores"); ax[0].legend()
g = tk["arrival_gap_min"]; xg = np.linspace(0, g.max(), 300)
ax[1].hist(g, bins=30, density=True, color="#a1d99b", edgecolor="white")
ax[1].plot(xg, stats.expon.pdf(xg, 0, scale), "r", lw=2, label="Fitted Exponential")
ax[1].set_title("Gap between tickets (min)"); ax[1].legend()
R.save_fig(fig, "py_fitted_curves.png")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
k = np.arange(0, 15)
ax[0].hist(tk["tickets_per_hour"], bins=np.arange(-0.5, 15.5), density=True, color="#fdae6b", edgecolor="white")
ax[0].plot(k, stats.poisson.pmf(k, lam), "ko-", label=f"Poisson({lam:.2f})")
ax[0].set_title("Tickets per hour"); ax[0].legend()
r = tk["resolution_hours"]; xr = np.linspace(0, r.quantile(0.99), 300)
ax[1].hist(r, bins=40, density=True, color="#bcbddc", edgecolor="white", range=(0, r.quantile(0.99)))
ax[1].plot(xr, stats.lognorm.pdf(xr, s, loc_l, sc_l), "r", lw=2, label="Lognormal")
ax[1].plot(xr, stats.norm.pdf(xr, m, sd), "k--", label="Normal")
ax[1].set_title("Resolution time (h)"); ax[1].legend()
R.save_fig(fig, "py_discrete_and_skewed_fits.png")

R.h("INTERPRETATION")
R.p(f"- Exam scores fit a Normal(mu={mu:.1f}, sigma={sigma:.1f}); KS p = {ks.pvalue:.2f} gives no evidence against it.")
R.p(f"- Gaps between tickets follow an Exponential with mean {scale:.1f} min (memoryless waiting times).")
R.p(f"- Hourly ticket counts have mean {lam:.2f} close to variance {tk['tickets_per_hour'].var():.2f}: consistent with Poisson.")
R.p("- Resolution times are right-skewed; the lognormal beats the normal by AIC.")
R.finish()
