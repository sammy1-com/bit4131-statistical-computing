"""Topic 21: Multivariate statistical analysis (numpy, scikit-learn, statsmodels)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.helpers import Report, load
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA

R = Report(__file__)
np.set_printoptions(precision=3, suppress=True)
df = load("students.csv")
cols = ["study_hours", "attendance_pct", "sleep_hours", "prior_gpa", "exam_score"]
X = df[cols]

R.h("WORKED EXAMPLE: Mahalanobis distance in 2D")
S = np.array([[4., 2.], [2., 3.]]); mean = np.array([0., 0.]); pt = np.array([2., 2.])
d2 = pt @ np.linalg.inv(S) @ pt
R.p("S =\n", S, "\nd^2 = (x-mu)' S^-1 (x-mu) =", round(d2, 4), "-> d =", round(np.sqrt(d2), 4))
R.p("Euclidean distance would be", round(np.linalg.norm(pt), 4), "- Mahalanobis accounts for scale AND correlation.")

R.h("MULTIVARIATE SUMMARY")
R.p("Mean vector:\n" + X.mean().round(2).to_string())
R.p("\nCovariance matrix:\n" + X.cov().round(2).to_string())
R.p(f"\nGeneralised variance |S| = {np.linalg.det(X.cov()):.4g};  total variance tr(S) = {np.trace(X.cov()):.2f}")

R.h("MULTIVARIATE OUTLIERS (Mahalanobis distance, chi-square cut-off)")
Sinv = np.linalg.inv(X.cov())
diff = X.to_numpy() - X.mean().to_numpy()
md2 = np.einsum("ij,jk,ik->i", diff, Sinv, diff)
cut = stats.chi2.ppf(0.999, len(cols))
out = df.loc[md2 > cut, "student_id"].tolist()
R.p(f"Cut-off chi2(0.999, df={len(cols)}) = {cut:.2f};  {len(out)} multivariate outlier(s): student_id {out}")

R.h("MANOVA: do the 5 measures jointly differ by teaching method?")
mv = MANOVA.from_formula("study_hours + attendance_pct + sleep_hours + prior_gpa + exam_score ~ method", df)
tab = mv.mv_test().results["method"]["stat"]
R.p(tab.round(4).to_string())
wilks_p = tab.loc["Wilks' lambda", "Pr > F"]

R.h("K-MEANS CLUSTERING (standardised data)")
Z = StandardScaler().fit_transform(X)
sil = {}
for k in range(2, 7):
    km = KMeans(k, n_init=10, random_state=1).fit(Z)
    sil[k] = silhouette_score(Z, km.labels_)
    R.p(f"k = {k}: inertia = {km.inertia_:.1f}, silhouette = {sil[k]:.3f}")
kbest = max(sil, key=sil.get)
km = KMeans(kbest, n_init=10, random_state=1).fit(Z)
df["cluster"] = km.labels_
R.p(f"\nBest k by silhouette = {kbest}")
R.p("Cluster profiles (means):\n" + df.groupby("cluster")[cols].mean().round(2).to_string())
R.p("Cluster sizes:", df.cluster.value_counts().sort_index().tolist())
R.p(f"Adjusted Rand index vs teaching method = {adjusted_rand_score(df.method, df.cluster):.3f} (0 = unrelated)")

R.h("HIERARCHICAL CLUSTERING (Ward)")
Lk = linkage(Z, "ward")
hc = fcluster(Lk, kbest, "maxclust")
R.p(f"Ward cut into {kbest} clusters; sizes = {np.bincount(hc)[1:].tolist()};  ARI vs k-means = {adjusted_rand_score(km.labels_, hc):.3f}")

pc = PCA(2).fit_transform(Z)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for c in range(kbest):
    m = df.cluster == c; ax[0].scatter(pc[m, 0], pc[m, 1], s=16, alpha=0.7, label=f"Cluster {c}")
ax[0].set(title="K-means clusters in PCA space", xlabel="PC1", ylabel="PC2")
ax[0].legend()
ax[1].plot(list(sil), list(sil.values()), "o-"); ax[1].set(title="Silhouette by k", xlabel="k", ylabel="Silhouette")
dendrogram(Lk, truncate_mode="lastp", p=20, ax=ax[2]); ax[2].set_title("Ward dendrogram (last 20 merges)")
R.save_fig(fig, "py_multivariate_clusters.png")

fig, ax = plt.subplots()
ax.hist(md2, bins=30, density=True, color="#9ecae1", edgecolor="white")
xs = np.linspace(0, md2.max(), 200); ax.plot(xs, stats.chi2.pdf(xs, len(cols)), "r", lw=2, label="chi2(5)")
ax.axvline(cut, color="k", ls="--", label="99.9% cut-off"); ax.set_title("Mahalanobis distances follow chi-square"); ax.legend()
R.save_fig(fig, "py_mahalanobis.png")

R.h("INTERPRETATION")
R.p(f"- MANOVA Wilks' lambda p = {wilks_p:.2g}: " + ("methods differ on the combined profile of outcomes." if wilks_p < .05 else "no joint difference."))
prof = df.groupby("cluster")[cols].mean()
gap = ((prof.max() - prof.min()) / df[cols].std()).sort_values(ascending=False)
R.p(f"- K-means picks k = {kbest} (best silhouette {sil[kbest]:.2f}). Clusters differ most on {gap.index[0]} and {gap.index[1]}.")
R.p(f"- A silhouette below ~0.25 means weak structure: the data form a continuum, so the clusters are a convenient split, not natural groups.")
R.p(f"- Clusters do not line up with teaching method (ARI = {adjusted_rand_score(df.method, df.cluster):.2f}, near 0), even though MANOVA finds method differences in means.")
R.p(f"- Mahalanobis distance flags {len(out)} unusual multivariate observation(s) that single-variable checks could miss.")
R.finish()
