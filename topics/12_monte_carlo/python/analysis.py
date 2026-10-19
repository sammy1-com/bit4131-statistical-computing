"""Topic 12: Random numbers and Monte Carlo methods (numpy.random, scipy.stats)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import stats

R = Report(__file__)
rng = np.random.default_rng(12)

# ---- 3. Worked example: estimating pi --------------------------------------------------------
R.h("WORKED EXAMPLE: estimate pi (area of quarter circle = pi/4)")
N = 100_000
x, y = rng.random(N), rng.random(N)
inside = x**2 + y**2 <= 1
pi_hat = 4 * inside.mean()
se = 4 * np.sqrt(inside.mean() * (1 - inside.mean()) / N)
R.p(f"pi-hat = 4 * {inside.sum()}/{N} = {pi_hat:.5f}  (true {np.pi:.5f}); MC standard error = {se:.5f}")

R.h("INVERSE-TRANSFORM SAMPLING: Exponential(rate 0.125) from Uniform(0,1)")
u = rng.random(50_000)
samples = -np.log(1 - u) / 0.125
R.p(f"sample mean = {samples.mean():.3f} (theory 8);  sample var = {samples.var():.2f} (theory 64)")

# ---- MC integration ------------------------------------------------------------------------------
R.h("MONTE CARLO INTEGRATION: integral of exp(-x^2) over [0,1]")
from scipy.integrate import quad
exact = quad(lambda t: np.exp(-t**2), 0, 1)[0]
v = np.exp(-rng.random(200_000) ** 2)
R.p(f"MC estimate = {v.mean():.5f} +/- {1.96 * v.std() / np.sqrt(len(v)):.5f};  exact = {exact:.5f}")

# ---- Dataset-based MC: pass probability of a random class ----------------------------------------------
df = load("students.csv")
R.h("DATASET: simulate a class of 40 students (resample from data)")
class_means = np.array([rng.choice(df.exam_score, 40).mean() for _ in range(20_000)])
p_class_above_65 = (class_means > 65).mean()
R.p(f"Simulated P(class mean > 65) = {p_class_above_65:.4f}")
R.p(f"Class mean average = {class_means.mean():.2f}, SD = {class_means.std():.2f}")
pass_rate = df.passed.mean()
fails = rng.binomial(40, 1 - pass_rate, 20_000)
R.p(f"Simulated P(more than 12 of 40 fail) = {(fails > 12).mean():.4f}   exact binomial = {stats.binom.sf(12, 40, 1 - pass_rate):.4f}")

# ---- Convergence plot -------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
run = 4 * np.cumsum(inside) / np.arange(1, N + 1)
ax[0].plot(np.arange(1, N + 1)[50:], run[50:], lw=1); ax[0].axhline(np.pi, color="r", ls="--", label="pi")
ax[0].set_xscale("log"); ax[0].set_xlabel("Number of points"); ax[0].set_ylabel("Estimate of pi")
ax[0].set_title("Monte Carlo convergence"); ax[0].legend()
idx = rng.choice(N, 3000, replace=False)
ax[1].scatter(x[idx], y[idx], c=np.where(inside[idx], "#2c7fb8", "#d7301f"), s=4)
t = np.linspace(0, np.pi / 2, 100); ax[1].plot(np.cos(t), np.sin(t), "k")
ax[1].set_aspect("equal"); ax[1].set_title("Random points: blue inside quarter circle")
R.save_fig(fig, "py_monte_carlo_convergence.png")

fig, ax = plt.subplots()
ns = np.logspace(2, 5, 25).astype(int)
errs = [np.mean([abs(4 * (rng.random(n) ** 2 + rng.random(n) ** 2 <= 1).mean() - np.pi) for _ in range(30)]) for n in ns]
ax.loglog(ns, errs, "o-", label="Mean absolute error"); ax.loglog(ns, 1.6 / np.sqrt(ns), "r--", label="~ 1/sqrt(N)")
ax.set_xlabel("N"); ax.set_ylabel("Error"); ax.set_title("Monte Carlo error shrinks like 1/sqrt(N)"); ax.legend()
R.save_fig(fig, "py_mc_error_rate.png")

R.h("INTERPRETATION")
R.p(f"- With {N:,} random points pi is estimated as {pi_hat:.4f}, within {abs(pi_hat - np.pi):.4f} of the truth.")
R.p("- Inverse-transform sampling turns uniform random numbers into any distribution with an invertible CDF.")
R.p("- Error falls like 1/sqrt(N): 100x more simulations buy only 10x more accuracy.")
R.p("- The simulated failure probability agrees closely with the exact binomial value, and the MC integral with quadrature.")
R.finish()
