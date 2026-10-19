"""Topic 4: Sampling distributions and the Central Limit Theorem."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats

R = Report(__file__)
rng = np.random.default_rng(7)

# Population = a clearly NON-normal variable: ticket resolution times (right-skewed)
pop = load("support_tickets.csv")["resolution_hours"].to_numpy()
mu, sigma = pop.mean(), pop.std(ddof=0)

R.h("WORKED EXAMPLE: standard error")
R.p(f"Population mean mu = {mu:.3f}, sigma = {sigma:.3f}, skewness = {stats.skew(pop):.2f}")
R.p(f"For n = 30: SE = sigma/sqrt(n) = {sigma:.3f}/{np.sqrt(30):.3f} = {sigma / np.sqrt(30):.4f}")

R.h("SAMPLING DISTRIBUTION OF THE MEAN (5,000 samples each)")
R.p(f"{'n':>4} {'mean of means':>14} {'SD of means':>12} {'theory SE':>10} {'skew':>7} {'Shapiro p':>10}")
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
rows = {}
for ax, n in zip(axes.ravel()[1:], [2, 5, 30, 100, 500]):
    means = rng.choice(pop, size=(5000, n)).mean(axis=1)
    se = sigma / np.sqrt(n)
    sk = stats.skew(means)
    sw = stats.shapiro(means[:2000]).pvalue
    rows[n] = (means.mean(), means.std(ddof=1), se, sk, sw)
    R.p(f"{n:>4} {means.mean():>14.4f} {means.std(ddof=1):>12.4f} {se:>10.4f} {sk:>7.2f} {sw:>10.4f}")
    ax.hist(means, bins=40, density=True, color="#9ecae1", edgecolor="white")
    xs = np.linspace(means.min(), means.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, se), "r", lw=2)
    ax.set_title(f"Sample means, n = {n}")
axes[0, 0].hist(pop, bins=40, density=True, color="#fdae6b", edgecolor="white")
axes[0, 0].set_title("Population (skewed)")
fig.suptitle("Central Limit Theorem: skewed population -> normal sample means (red = N(mu, sigma/sqrt n))")
R.save_fig(fig, "py_clt_visualization.png")

# SE shrinks like 1/sqrt(n)
ns = np.array([2, 5, 30, 100, 500])
fig, ax = plt.subplots()
ax.plot(ns, [rows[n][1] for n in ns], "o-", label="Simulated SD of means")
ax.plot(ns, sigma / np.sqrt(ns), "r--", label="Theory sigma/sqrt(n)")
ax.set_xscale("log"); ax.set_xlabel("Sample size n"); ax.set_ylabel("Standard error")
ax.set_title("Standard error vs sample size"); ax.legend()
R.save_fig(fig, "py_standard_error.png")

R.h("INTERPRETATION")
R.p(f"- The population is strongly right-skewed (skew {stats.skew(pop):.2f}) and clearly non-normal.")
R.p(f"- At n=2 the means are still skewed ({rows[2][3]:.2f}); by n=100 the skew is {rows[100][3]:.2f}, close to a bell shape.")
R.p("- The mean of the sample means equals the population mean (unbiased), and their SD matches sigma/sqrt(n).")
R.p("- Quadrupling n halves the standard error: precision grows with the square root of sample size.")
R.finish()
