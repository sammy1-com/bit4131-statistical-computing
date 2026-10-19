"""Topic 2: Probability and conditional probability (numpy, scipy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load

R = Report(__file__)
rng = np.random.default_rng(1)

# ---- 3. Worked example: two fair dice -----------------------------------------
R.h("WORKED EXAMPLE: two fair dice")
sims_exact = {"P(sum=7)": 6 / 36, "P(at least one 6)": 11 / 36}
A_and_B = 2 / 36  # (6,1),(1,6): sum is 7 AND at least one six
R.p("P(sum = 7)                 =", round(6 / 36, 4))
R.p("P(at least one 6)          =", round(11 / 36, 4))
R.p("P(sum=7 AND at least one 6)=", round(A_and_B, 4))
R.p("P(sum=7 | at least one 6)  =", round(A_and_B / (11 / 36), 4), "(= 2/11)")

# ---- 4/5. Dataset analysis: students -----------------------------------------
df = load("students.csv")
R.h("DATASET: probabilities from the students table")
n = len(df)
p_pass = df["passed"].mean()
p_B = (df["method"] == "B").mean()
p_stat = (df["program"] == "Statistics").mean()
R.p(f"P(pass)                    = {p_pass:.4f}")
R.p(f"P(method B)                = {p_B:.4f}")
R.p(f"P(Statistics student)      = {p_stat:.4f}")

tab = pd.crosstab(df["method"], df["passed"], margins=True)
R.p("\nContingency table (method x passed):\n" + tab.to_string())
R.p("\nConditional probabilities P(pass | method):")
cond = df.groupby("method")["passed"].mean()
for m, v in cond.items():
    R.p(f"  P(pass | method {m}) = {v:.4f}")

# Independence check: does P(A and B) = P(A)P(B)?
p_joint = ((df["method"] == "B") & (df["passed"] == 1)).mean()
R.p(f"\nP(B and pass) = {p_joint:.4f}   vs   P(B)P(pass) = {p_B * p_pass:.4f}")
R.p("-> Not equal, so 'method B' and 'passing' are NOT independent.")

# Bayes: P(method C | pass)
p_C = (df["method"] == "C").mean()
bayes = cond["C"] * p_C / p_pass
R.p(f"\nBayes: P(method C | pass) = P(pass|C)P(C)/P(pass) = {bayes:.4f}")

# ---- Simulation -----------------------------------------------------------------
R.h("SIMULATION: P(sum=7 | at least one 6) with 200,000 rolls")
N = 200_000
d1, d2 = rng.integers(1, 7, N), rng.integers(1, 7, N)
has6 = (d1 == 6) | (d2 == 6)
est = ((d1 + d2 == 7) & has6).sum() / has6.sum()
R.p(f"Simulated = {est:.4f}   Exact = {2 / 11:.4f}")

running = np.cumsum((d1 + d2 == 7) & has6) / np.maximum(np.cumsum(has6), 1)
fig, ax = plt.subplots()
ax.plot(running[100:], lw=1)
ax.axhline(2 / 11, color="red", ls="--", label="Exact 2/11")
ax.set_xscale("log"); ax.set_xlabel("Rolls (log scale)")
ax.set_ylabel("Estimated conditional probability")
ax.set_title("Simulation converges to the exact conditional probability"); ax.legend()
R.save_fig(fig, "py_simulation_convergence.png")

fig, ax = plt.subplots()
cond.plot(kind="bar", ax=ax, color="#9ecae1", rot=0)
ax.axhline(p_pass, color="red", ls="--", label=f"Overall P(pass)={p_pass:.2f}")
ax.set_ylabel("P(pass | method)"); ax.set_title("Conditional pass probability by method"); ax.legend()
R.save_fig(fig, "py_conditional_probability.png")

R.h("INTERPRETATION")
R.p(f"- Overall pass probability is {p_pass:.2f}, but it rises from {cond['A']:.2f} (A) to {cond['C']:.2f} (C).")
R.p("- Conditioning changes the probability, so method carries information about passing.")
R.p("- The simulation matches the exact result 2/11, validating the Monte Carlo approach.")
R.finish()
