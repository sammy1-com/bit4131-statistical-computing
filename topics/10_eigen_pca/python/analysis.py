"""Topic 10: Eigenvalues, eigenvectors and PCA (numpy.linalg, sklearn.decomposition)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

R = Report(__file__)
np.set_printoptions(precision=4, suppress=True)

R.h("WORKED EXAMPLE: A = [[2,1],[1,2]]")
A = np.array([[2., 1.], [1., 2.]])
vals, vecs = np.linalg.eigh(A)
R.p("Characteristic equation: (2-L)^2 - 1 = 0  ->  L = 1 and L = 3")
R.p("Eigenvalues (numpy):", vals)
R.p("Eigenvectors (columns):\n", vecs)
R.p("Check A v = L v for L=3:", A @ vecs[:, 1], "=", 3 * vecs[:, 1])

df = load("students.csv")
cols = ["study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score"]
Z = StandardScaler().fit_transform(df[cols])

R.h("PCA BY EIGEN-DECOMPOSITION of the correlation matrix")
C = np.corrcoef(Z.T)
ev, evec = np.linalg.eigh(C)
order = np.argsort(ev)[::-1]
ev, evec = ev[order], evec[:, order]
R.p("Eigenvalues:", ev, "  sum =", round(ev.sum(), 4), "(= number of variables)")
R.p("Explained variance ratio:", (ev / ev.sum()).round(4))
R.p("Cumulative:", np.cumsum(ev / ev.sum()).round(4))
R.p("\nLoadings (eigenvectors):")
R.p(pd.DataFrame(evec, index=cols, columns=[f"PC{i + 1}" for i in range(5)]).round(3).to_string())

R.h("SCIKIT-LEARN PCA (cross-check)")
pca = PCA().fit(Z)
R.p("Explained variance ratio:", pca.explained_variance_ratio_.round(4))
R.p("Matches eigen route:", np.allclose(pca.explained_variance_, ev, atol=1e-2 * ev.max()))
scores = pca.transform(Z)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, 6), ev, "o-"); ax.axhline(1, color="r", ls="--", label="Kaiser criterion (eigenvalue = 1)")
ax.set_xlabel("Component"); ax.set_ylabel("Eigenvalue"); ax.set_title("Scree plot"); ax.legend()
R.save_fig(fig, "py_scree_plot.png")

fig, ax = plt.subplots(figsize=(7.5, 6))
cmap = {"A": "#d7301f", "B": "#2c7fb8", "C": "#31a354"}
for m, g in df.groupby("method"):
    ax.scatter(scores[g.index, 0], scores[g.index, 1], s=18, alpha=0.6, label=f"Method {m}", color=cmap[m])
for i, c in enumerate(cols):
    ax.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3, color="k", head_width=0.07)
    ax.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2, c, fontsize=9)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("PCA biplot"); ax.legend()
R.save_fig(fig, "py_pca_biplot.png")

k = int((ev > 1).sum())
R.h("INTERPRETATION")
R.p(f"- {k} component(s) have eigenvalue > 1 (Kaiser rule); the first two explain {ev[:2].sum() / ev.sum():.0%} of variance.")
top = pd.Series(abs(evec[:, 0]), index=cols).sort_values(ascending=False)
R.p(f"- PC1 is dominated by {top.index[0]} and {top.index[1]}: a general 'academic effort/achievement' axis.")
R.p("- Eigenvalues sum to the number of standardised variables; each is the variance captured by its component.")
R.p("- Keeping the leading components compresses the data while preserving most of the information.")
R.finish()
