"""Topic 11: Numerical optimization (scipy.optimize)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
from common.helpers import Report, load
from scipy import optimize, stats

R = Report(__file__)

# ---- 3. Worked example: minimise f(x) = (x-3)^2 + 2 ---------------------------------------
R.h("WORKED EXAMPLE: minimise f(x) = (x - 3)^2 + 2")
f = lambda x: (x - 3) ** 2 + 2
res = optimize.minimize_scalar(f)
R.p("Analytic: f'(x) = 2(x-3) = 0 -> x* = 3, f(x*) = 2")
R.p(f"Numeric : x* = {res.x:.6f}, f(x*) = {res.fun:.6f}, converged = {res.success}")

# ---- Rosenbrock: classic hard surface -------------------------------------------------------
R.h("OBJECTIVE: Rosenbrock f(x,y) = (1-x)^2 + 100(y-x^2)^2   (true minimum at (1,1))")
rosen = optimize.rosen            # (1-x)^2 + 100(y-x^2)^2 in scipy's n-dim form
x0 = np.array([-1.5, 2.0])
path, iters = {}, {}
for method in ["Nelder-Mead", "BFGS", "CG"]:
    trace = [x0.copy()]
    jac = None if method == "Nelder-Mead" else optimize.rosen_der   # analytic gradient
    r = optimize.minimize(rosen, x0, jac=jac, method=method,
                          callback=lambda xk, *a: trace.append(np.array(xk)))
    path[method], iters[method] = np.array(trace), r.nit
    R.p(f"{method:<12} -> x = ({r.x[0]:.5f}, {r.x[1]:.5f}), f = {r.fun:.2e}, iterations = {r.nit}, converged = {r.success}")

# ---- Dataset: Maximum likelihood by numerical optimisation ------------------------------------
tk = load("support_tickets.csv")["resolution_hours"].to_numpy()
R.h("DATASET: MLE of Lognormal(mu, sigma) for ticket resolution times")
nll = lambda th: -stats.lognorm.logpdf(tk, th[1], 0, np.exp(th[0])).sum() if th[1] > 0 else np.inf
mle = optimize.minimize(nll, x0=[1.0, 1.0], method="Nelder-Mead")
mu_hat, s_hat = np.log(tk).mean(), np.log(tk).std()
R.p(f"Numerical MLE  : mu = {mle.x[0]:.4f}, sigma = {mle.x[1]:.4f}, converged = {mle.success}")
R.p(f"Closed form    : mu = {mu_hat:.4f}, sigma = {s_hat:.4f}")
R.p(f"Negative log-likelihood at optimum = {mle.fun:.2f}")

# ---- Constrained optimisation --------------------------------------------------------------------
R.h("CONSTRAINED: maximise study benefit 4*sqrt(h1)+3*sqrt(h2) subject to h1+h2 = 20")
obj = lambda h: -(4 * np.sqrt(h[0]) + 3 * np.sqrt(h[1]))
cons = {"type": "eq", "fun": lambda h: h[0] + h[1] - 20}
sol = optimize.minimize(obj, [10, 10], bounds=[(0.01, 20)] * 2, constraints=cons, method="SLSQP")
R.p(f"Optimal split: h1 = {sol.x[0]:.3f}, h2 = {sol.x[1]:.3f}; benefit = {-sol.fun:.4f}")
R.p("Analytic (Lagrange) answer: h1 = 16/25*20 = 12.8, h2 = 9/25*20 = 7.2")

# ---- Plots ----------------------------------------------------------------------------------------
gx, gy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.contour(gx, gy, (1 - gx) ** 2 + 100 * (gy - gx**2) ** 2, levels=np.logspace(-1, 3.5, 20), cmap="viridis", alpha=0.7)
for m, c in zip(path, ["#d7301f", "#2c7fb8", "#31a354"]):
    ax.plot(path[m][:, 0], path[m][:, 1], "o-", ms=3, lw=1, color=c, label=f"{m} ({len(path[m]) - 1} it)")
ax.plot(1, 1, "k*", ms=16, label="True minimum"); ax.legend(); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Optimisation paths on the Rosenbrock surface")
R.save_fig(fig, "py_optimization_paths.png")

fig, ax = plt.subplots()
mus = np.linspace(mle.x[0] - 0.4, mle.x[0] + 0.4, 100)
ax.plot(mus, [nll([m, mle.x[1]]) for m in mus]); ax.axvline(mle.x[0], color="r", ls="--", label="MLE")
ax.set_xlabel("mu (sigma fixed)"); ax.set_ylabel("Negative log-likelihood"); ax.set_title("Likelihood surface slice"); ax.legend()
R.save_fig(fig, "py_likelihood_slice.png")

R.h("INTERPRETATION")
R.p(f"- All three methods reach (1,1). Iterations: Nelder-Mead {iters['Nelder-Mead']}, BFGS {iters['BFGS']}, CG {iters['CG']}: "
    "gradient-based methods need fewer steps than the derivative-free simplex.")
R.p("- Numerical MLE agrees with the closed-form estimates: optimisation is how MLE is computed when no formula exists.")
R.p("- The constrained solution allocates more hours to the higher-benefit subject, matching the Lagrange result.")
R.finish()
